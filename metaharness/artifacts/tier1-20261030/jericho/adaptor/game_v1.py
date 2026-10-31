"""TurnGame adaptor for local Jericho Z-machine stories."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jericho
from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = True
SEEDS = {
    # The ordered manifest places the four training titles first and the four
    # held-out titles second.  Thus these groups exercise disjoint stories;
    # the larger values provide independent VM RNG episodes within each title.
    "train": [0, 1, 2, 3, 8, 9, 10, 11],
    "heldout": [4, 5, 6, 7, 12, 13, 14, 15],
    "dev": [16, 17, 18, 19],
    "validation": [20, 21, 22, 23, 24, 25, 26, 27],
}
EVAL_KWARGS: dict[str, Any] = {}

_ROOT = Path(__file__).resolve().parent
_DEFAULT_MANIFEST = _ROOT / "runtime" / "manifest.json"
_META_COMMANDS = {"restart", "restore", "save", "quit", "undo", "script", "transcript"}


def _read_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("games") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest {path} has no games")
    result: list[dict[str, Any]] = []
    version = str(raw.get("manifest_version", "unknown")) if isinstance(raw, dict) else "unknown"
    for ent in entries:
        if not isinstance(ent, dict):
            raise ValueError("manifest entries must be objects")
        required = ("id", "story", "sha256", "max_score")
        if any(k not in ent for k in required):
            raise ValueError(f"manifest entry missing one of {required}")
        story = Path(str(ent["story"]))
        if not story.is_absolute():
            story = path.parent / story
        story = story.resolve()
        if not story.is_file():
            raise FileNotFoundError(story)
        digest = hashlib.sha256(story.read_bytes()).hexdigest()
        if digest.lower() != str(ent["sha256"]).lower():
            raise ValueError(f"sha256 mismatch for {ent['id']}: {digest}")
        maximum = int(ent["max_score"])
        if maximum <= 0:
            raise ValueError(f"max_score must be positive for {ent['id']}")
        copy = dict(ent)
        copy["story"] = str(story)
        copy["sha256"] = digest
        copy["max_score"] = maximum
        result.append(copy)
    ids = [str(e["id"]) for e in result]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest game IDs must be unique")
    pools = {str(e.get("pool", "")) for e in result}
    if not {"train", "heldout"}.issubset(pools):
        raise ValueError("manifest must partition games into train and heldout pools")
    return result, version


class JerichoGame:
    name = "jericho-interactive-fiction"
    action_spec = ActionSpec(
        names=("command",),
        descriptions={"command": "One ordinary parser command, such as 'east' or 'put carrot in chest'."},
        noop="look",
        notes=("The action value is a single string command (UTF-8, at most 200 characters, no newline or NUL). "
               "Commands are interpreted by the story's parser; one command is submitted per turn."),
    )

    def __init__(self, manifest: str | os.PathLike[str] | None = None, max_steps: int | None = None,
                 observation_mode: str | None = None):
        manifest_path = Path(manifest or os.environ.get("JERICHO_GAME_MANIFEST", _DEFAULT_MANIFEST)).expanduser()
        self._manifest_path = manifest_path.resolve()
        self._games, self._manifest_version = _read_manifest(self._manifest_path)
        self._max_steps = int(max_steps or os.environ.get("JERICHO_MAX_STEPS", "1000"))
        if self._max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._observation_mode = observation_mode or os.environ.get("JERICHO_OBSERVATION_MODE", "text")
        if self._observation_mode not in {"text", "actions", "state_search"}:
            raise ValueError("observation_mode must be text, actions, or state_search")
        self._env: Any = None
        self._entry: dict[str, Any] | None = None
        self._seed = self._vm_seed = None
        self._text = ""
        self._last_action: str | None = None
        self._last_delta = 0
        self._initial_score = 0
        self._steps = 0
        self._terminated = self._truncated = False
        self._won = self._lost = False
        self._reason: str | None = None
        self._valid = True
        self._invalid_reason: str | None = None

    def _close_env(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            self._env = None

    def reset(self, seed: int) -> dict[str, Any]:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._close_env()
        self._seed = int(seed)
        self._entry = self._games[seed % len(self._games)]
        self._vm_seed = 1 + ((seed // len(self._games)) % 32767)
        self._env = jericho.FrotzEnv(self._entry["story"], seed=self._vm_seed)
        reset_result = self._env.reset()
        self._text = str(reset_result[0] if isinstance(reset_result, tuple) else reset_result)
        self._initial_score = int(self._env.get_score())
        self._last_action = None
        self._last_delta = 0
        self._steps = 0
        self._terminated = self._truncated = False
        self._won = bool(self._env.victory())
        self._lost = bool(self._env.game_over()) and not self._won
        self._reason = None
        self._valid = True
        self._invalid_reason = None
        try:
            native_max = int(self._env.get_max_score())
            if native_max != int(self._entry["max_score"]):
                self._valid = False
                self._invalid_reason = (f"manifest max_score {self._entry['max_score']} differs from "
                                        f"native max_score {native_max}")
        except Exception as exc:
            self._valid = False
            self._invalid_reason = f"could not read native maximum score: {type(exc).__name__}: {exc}"[:500]
        return self.observe()

    def _snapshot(self, error: str | None = None) -> dict[str, Any]:
        score = int(self._env.get_score()) if self._env is not None else self._initial_score
        maximum = int(self._entry["max_score"]) if self._entry else 0
        try:
            moves: int | None = int(self._env.get_moves()) if self._env is not None else None
        except Exception:
            moves = None
        out: dict[str, Any] = {
            "game_id": self._entry.get("id") if self._entry else None,
            "episode_seed": self._seed,
            "vm_seed": self._vm_seed,
            "story_sha256": self._entry.get("sha256") if self._entry else None,
            "text": self._text,
            "last_action": self._last_action,
            "score": score,
            "max_score": maximum,
            "score_delta": self._last_delta,
            "native_moves": moves,
            "step_count": self._steps,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "won": self._won,
            "lost": self._lost,
            "termination_reason": self._reason,
            "observation_mode": self._observation_mode,
        }
        if error:
            out["action_error"] = error
        return out

    def observe(self) -> dict[str, Any]:
        return self._snapshot()

    def step(self, action: Any) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        if self.done:
            return self._snapshot("episode already finished")
        self._steps += 1
        error: str | None = None
        before = int(self._env.get_score())
        command: str | None = None
        if not isinstance(action, str):
            error = "action must be a string"
        elif not action or len(action) > 200 or "\x00" in action or "\n" in action or "\r" in action:
            error = "action must be a nonempty single line of at most 200 characters"
        elif action.strip().lower().split(" ", 1)[0] in _META_COMMANDS:
            error = "lifecycle commands are controlled by the evaluator"
        else:
            command = action
        if command is not None:
            try:
                result = self._env.step(command)
                self._text = str(result[0])
                self._won = bool(self._env.victory())
                self._lost = bool(self._env.game_over()) and not self._won
                if bool(result[2]) or self._won or self._lost:
                    self._terminated = True
                    self._reason = "victory" if self._won else ("game_over" if self._lost else "terminated")
            except Exception as exc:  # preserve an observable infrastructure error
                self._valid = False
                self._invalid_reason = f"Jericho step failed: {type(exc).__name__}: {exc}"[:500]
                error = "interpreter error"
        after = int(self._env.get_score())
        self._last_action = action if isinstance(action, str) else None
        self._last_delta = after - before
        if self._steps >= self._max_steps and not self._terminated:
            self._truncated = True
            self._reason = "step_limit"
        return self._snapshot(error)

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def score(self) -> float:
        if self._env is None or self._entry is None:
            return 0.0
        return float(self._env.get_score()) / float(self._entry["max_score"])

    def score_breakdown(self) -> dict[str, float]:
        native = int(self._env.get_score()) if self._env is not None else self._initial_score
        maximum = int(self._entry["max_score"]) if self._entry else 0
        return {"normalized_native_score": self.score, "native_score": float(native),
                "maximum_score": float(maximum), "initial_score": float(self._initial_score),
                "steps": float(self._steps), "won": float(self._won), "lost": float(self._lost)}

    def info(self) -> dict[str, Any]:
        native = int(self._env.get_score()) if self._env is not None else self._initial_score
        return {"valid_trial": self._valid, "invalid_reason": self._invalid_reason,
                "game_id": self._entry.get("id") if self._entry else None,
                "story_sha256": self._entry.get("sha256") if self._entry else None,
                "episode_seed": self._seed, "vm_seed": self._vm_seed,
                "manifest": str(self._manifest_path), "manifest_version": self._manifest_version,
                "native_score": native, "max_score": self._entry.get("max_score") if self._entry else None,
                "initial_score": self._initial_score, "steps": self._steps,
                "won": self._won, "lost": self._lost, "termination_reason": self._reason}

    def describe(self) -> str:
        return ("You are playing an offline Jericho Z-machine interactive fiction story. "
                "The objective is to solve the story and maximize its native score. "
                "Each step accepts exactly one parser command as a string; use ordinary commands such as "
                "'east', 'look', 'take carrot', or 'put carrot in chest'. The observation includes the latest "
                "parser text, score, move count, and terminal flags. Failed parser commands are still submitted "
                "turns. The episode ends on victory/game over or at the evaluator step limit.")

    def close(self) -> None:
        self._close_env()


def make_game(**kwargs: Any) -> JerichoGame:
    return JerichoGame(**kwargs)
