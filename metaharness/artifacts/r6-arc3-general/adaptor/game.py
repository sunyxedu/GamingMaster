"""ARC-AGI-3 public SDK adaptor. No game source or private state is inspected."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from importlib.metadata import version
from numbers import Integral
from typing import Any

from metaharness.games.base import ActionSpec

TIER = 1
# Initial frames repeat, but full trajectories / engine globals are not proven deterministic.
DETERMINISTIC = False
NAME = "ARC-AGI-3"
ROOT = Path(__file__).resolve().parent
MANIFEST = (
    ("cd82-fb555c5d", 6), ("lf52-271a04aa", 10),
    ("ls20-9607627b", 7), ("sb26-7fbdac44", 8),
    ("sp80-589a99af", 6), ("tn36-ef4dde99", 7),
    ("tu93-0768757b", 9), ("vc33-5430563c", 7),
)
ACTION_NAMES = {0: "reset", 1: "up", 2: "down", 3: "left", 4: "right",
                5: "action5", 6: "click 32 32", 7: "undo"}
MEANINGS = {
    "noop": "Observe without advancing the engine; consumes one controller action.",
    "reset": "Explicit in-game retry (RESET/0); may restart the entire game.",
    "up": "ACTION1: conventionally up; discover its effect from play.",
    "down": "ACTION2: conventionally down; discover its effect from play.",
    "left": "ACTION3: conventionally left; discover its effect from play.",
    "right": "ACTION4: conventionally right; discover its effect from play.",
    "action5": "ACTION5: context-dependent interaction.",
    "click 32 32": "ACTION6: click pixel (32,32); grammar: click X Y, each integer 0..63.",
    "undo": "ACTION7: undo when available; exact effect depends on the game.",
}


class InterfaceError(RuntimeError):
    """A mutating SDK call failed; its outcome is uncertain and must not be retried."""


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


class ARCGame:
    name = NAME

    def __init__(self, *, max_actions: int = 10000, max_seconds: float | None = None,
                 max_invalid_actions: int = 20, screenshots: bool = True):
        self.max_actions = _integer(max_actions, "max_actions", 1)
        self.max_invalid_actions = _integer(max_invalid_actions, "max_invalid_actions", 1)
        if max_seconds is not None:
            import math
            if isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or max_seconds <= 0:
                raise ValueError("max_seconds must be finite and positive, or None")
            max_seconds = float(max_seconds)
        self.max_seconds = max_seconds
        self.screenshots = bool(screenshots)
        if os.environ.get("OPERATION_MODE", "").lower() != "offline":
            raise RuntimeError("Set OPERATION_MODE=offline; this adaptor never connects online.")
        cache = os.environ.get("ENVIRONMENTS_DIR")
        if not cache or not Path(cache).expanduser().is_dir():
            raise RuntimeError("ENVIRONMENTS_DIR must point to the existing eight-game cache.")
        if os.environ.get("ONLY_RESET_LEVELS"):
            raise RuntimeError("Unset ONLY_RESET_LEVELS to preserve the pinned default retry semantics.")
        for key, default in (("MPLCONFIGDIR", ROOT / ".cache/matplotlib"),
                             ("XDG_CACHE_HOME", ROOT / ".cache")):
            os.environ.setdefault(key, str(default))
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLBACKEND", "Agg")
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction
        self._GameAction = GameAction
        logger = logging.getLogger("metaharness.arc3")
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())  # stderr, never protocol stdout
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        self._arc = Arcade(operation_mode=OperationMode.OFFLINE,
                           environments_dir=str(Path(cache).expanduser().resolve()),
                           recordings_dir=os.environ.get("RECORDINGS_DIR", str(ROOT / "recordings")),
                           logger=logger)
        if self._arc.operation_mode != OperationMode.OFFLINE:
            raise RuntimeError("SDK resolved to a non-offline operation mode")
        available = {e.game_id for e in self._arc.get_environments()}
        missing = [gid for gid, _ in MANIFEST if gid not in available]
        if missing:
            raise RuntimeError(f"Missing pinned cached games: {', '.join(missing)}")
        self._versions = {p: version(p) for p in ("arc-agi", "arcengine", "numpy", "pydantic")}
        self._artifacts = ROOT / "observations"
        self._artifacts.mkdir(exist_ok=True)
        self._env = None
        self._card = None
        self._obs = None
        self._score = 0
        self._reason = None
        self._started = None
        self.action_spec = self._spec([])

    @staticmethod
    def _spec(ids: list[int]) -> ActionSpec:
        names = ["noop", "reset"] + [ACTION_NAMES[i] for i in ids if i != 0]
        return ActionSpec(names=names, descriptions={n: MEANINGS[n] for n in names}, noop="noop",
                          notes="Read the current action_spec/available_actions each turn. "
                          "click 32 32 represents all click X Y commands; bare click is invalid.")

    def reset(self, seed: int) -> dict[str, Any]:
        seed = _integer(seed, "seed")
        self.close()
        self._seed = seed
        self._game_id, self._win_levels = MANIFEST[seed % len(MANIFEST)]
        self._score = 0
        self._reward = 0
        self._reason = None
        self._obs = None
        self._observation_id = 0
        self._episode_actions = 0
        self._numbered_actions = 0
        self._retries = 0
        self._undos = 0
        self._noops = 0
        self._invalid_actions = 0
        self._sdk_calls = 0
        self._started = time.monotonic()
        self._card = self._arc.open_scorecard()
        try:
            self._env = self._arc.make(self._game_id, seed=0, scorecard_id=self._card,
                                       save_recording=False, include_frame_data=True, render_mode=None)
            if self._env is None:
                raise InterfaceError(f"SDK could not construct pinned game {self._game_id}")
            # make() already performs the initial RESET. Do not reset twice.
            self._accept(self._env.observation_space)
        except Exception:
            self.close()
            raise
        return self.observe()

    def _accept(self, response: Any) -> None:
        if response is None or not response.frame:
            raise InterfaceError("SDK returned no observation or an empty frame sequence")
        import numpy as np
        grids = []
        for frame in response.frame:
            a = np.asarray(frame)
            if a.shape != (64, 64) or not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 15:
                raise InterfaceError("Expected a 64x64 integer palette grid (0..15)")
            grids.append(a.tolist())
        current = int(response.levels_completed)
        total = int(response.win_levels)
        if total != self._win_levels or not 0 <= current <= total:
            raise InterfaceError("SDK progress does not match the pinned manifest")
        ids = sorted(set(int(i) for i in response.available_actions))
        if any(i not in ACTION_NAMES for i in ids):
            raise InterfaceError("SDK advertised an unknown action")
        state = response.state.value
        if state not in {"NOT_PLAYED", "NOT_FINISHED", "GAME_OVER", "WIN"}:
            raise InterfaceError(f"Unknown public game state {state!r}")
        # Content-addressed immutable artifacts preserve previous observations.
        animation_path = None
        if len(grids) > 1:
            payload = json.dumps(grids, separators=(",", ":")).encode()
            path = self._artifacts / (hashlib.sha256(payload).hexdigest() + ".json")
            if not path.exists():
                path.write_bytes(payload)
            animation_path = str(path)
        screenshot = None
        if self.screenshots:
            from PIL import Image
            from arc_agi.rendering import COLOR_MAP
            pixels = np.asarray(grids[-1], dtype=np.uint8)
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            path = self._artifacts / (digest + ".png")
            if not path.exists():
                palette = np.array([tuple(bytes.fromhex(COLOR_MAP[i][1:])[:3]) for i in range(16)], dtype=np.uint8)
                Image.fromarray(palette[pixels]).resize((512, 512), Image.Resampling.NEAREST).save(path)
            screenshot = str(path)
        action_input = response.action_input
        echoed = {"id": int(action_input.id.value), "data": dict(action_input.data or {})}
        previous = self._score
        self._score = max(previous, min(current, total))
        self._reward = self._score - previous
        self._obs = {
            "state": state, "levels_completed": current, "win_levels": total,
            "available_actions": ids, "full_reset": bool(response.full_reset),
            "frame": grids[-1], "screenshot": screenshot,
            "animation_frames_path": animation_path, "animation_frame_count": len(grids),
            "last_action": echoed,
        }
        self._guid = str(response.guid)
        self.action_spec = self._spec(ids if state not in {"WIN", "GAME_OVER"} else [])
        if state == "WIN":
            self.action_spec = ActionSpec(names=["noop"], descriptions={"noop": MEANINGS["noop"]}, noop="noop")

    def _parse(self, action: Any) -> tuple[int | None, dict[str, int] | None]:
        if isinstance(action, str):
            parts = action.strip().lower().split()
            if parts == ["noop"]:
                return None, None
            simple = {v: k for k, v in ACTION_NAMES.items() if k != 6}
            if len(parts) == 1 and parts[0] in simple:
                action = {"id": simple[parts[0]]}
            elif len(parts) == 3 and parts[0] == "click":
                try:
                    action = {"id": 6, "x": int(parts[1]), "y": int(parts[2])}
                except ValueError:
                    raise ValueError("Use click X Y with integer pixel coordinates") from None
            else:
                raise ValueError("Use a current action name, click X Y, or {id: N, x: X, y: Y}")
        if not isinstance(action, dict) or "id" not in action:
            raise ValueError("Action must be a documented string or an id dictionary")
        action_id = _integer(action["id"], "action id")
        if action_id not in ACTION_NAMES:
            raise ValueError("Action id must be in 0..7")
        if set(action) != ({"id", "x", "y"} if action_id == 6 else {"id"}):
            raise ValueError("Only click actions take x and y; both coordinates are required")
        data = None
        if action_id == 6:
            x, y = _integer(action["x"], "x"), _integer(action["y"], "y")
            if x > 63 or y > 63:
                raise ValueError("Click coordinates must be in 0..63")
            data = {"x": x, "y": y}
        if action_id and (self._obs["state"] == "GAME_OVER" or action_id not in self._obs["available_actions"]):
            raise ValueError("Numbered action unavailable; read available_actions, or reset after GAME_OVER")
        return action_id, data

    def step(self, action: Any) -> dict[str, Any]:
        self._require_episode()
        if self.done:
            self._reward = 0
            return self.observe()  # Never send post-terminal actions to the engine.
        self._episode_actions += 1
        self._observation_id += 1
        self._reward = 0
        try:
            action_id, data = self._parse(action)
        except ValueError:
            self._invalid_actions += 1
            if self._invalid_actions >= self.max_invalid_actions:
                self._reason = "invalid_action_limit"
            raise
        if action_id is None:
            self._noops += 1
        else:
            self._sdk_calls += 1
            if action_id == 0:
                self._retries += 1
            else:
                self._numbered_actions += 1
                if action_id == 7:
                    self._undos += 1
            try:
                response = self._env.step(self._GameAction.from_id(action_id), data=data)
                self._accept(response)
            except Exception as exc:
                self._reason = "interface_failure"
                raise InterfaceError(f"SDK action {action_id} failed in {self._game_id}; "
                                     "last good observation retained; no automatic retry") from exc
        return self.observe()

    def _require_episode(self) -> None:
        if self._env is None or self._obs is None:
            raise RuntimeError("Call reset(seed) to start a fresh episode first")

    def _budget(self) -> None:
        if self._obs is None or self._obs["state"] == "WIN" or self._reason:
            return
        if self._episode_actions >= self.max_actions:
            self._reason = "action_limit"
        elif self.max_seconds is not None and self.elapsed_seconds >= self.max_seconds:
            self._reason = "time_limit"

    @property
    def elapsed_seconds(self) -> float:
        return 0.0 if self._started is None else time.monotonic() - self._started

    @property
    def done(self) -> bool:
        self._budget()
        return bool(self._reason or (self._obs and self._obs["state"] == "WIN"))

    @property
    def score(self) -> float:
        return float(self._score)

    def observe(self) -> dict[str, Any]:
        self._require_episode()
        self._budget()
        observation = copy.deepcopy(self._obs)
        observation.update({
            "observation_id": self._observation_id,
            "legal_actions": list(self.action_spec.names),
            "episode_actions": self._episode_actions, "numbered_actions": self._numbered_actions,
            "retries": self._retries, "undos": self._undos, "noops": self._noops,
            "invalid_actions": self._invalid_actions, "score": self.score,
            "reward": float(self._reward), "completion_fraction": self.score / self._win_levels,
            "won": self._obs["state"] == "WIN", "done": self.done,
            "truncated": self._reason is not None, "truncation_reason": self._reason,
            "actions_remaining": max(0, self.max_actions - self._episode_actions),
        })
        return observation

    def info(self) -> dict[str, Any]:
        obs = self.observe()
        return {**{k: v for k, v in obs.items() if k not in {"frame", "screenshot", "animation_frames_path"}},
                "seed": self._seed, "game_id": self._game_id, "engine_seed": 0,
                "split": "development" if self._seed % 8 < 4 else "held_out",
                "guid": self._guid, "versions": dict(self._versions),
                "elapsed_seconds": self.elapsed_seconds, "sdk_action_calls": self._sdk_calls,
                "max_actions": self.max_actions, "max_seconds": self.max_seconds,
                "max_invalid_actions": self.max_invalid_actions, "operation_mode": "offline"}

    def describe(self) -> str:
        return (
            "ARC-AGI-3 is a turn-based visual rule-discovery game. Infer objects, controls, "
            "constraints and each level's objective from pixels and action effects; no hidden "
            "rules, object labels or goal masks are supplied. Complete all levels to WIN. "
            "Thinking does not advance the engine. Use one action string per turn: noop, reset, "
            "up, down, left, right, action5, undo, or click X Y (integer pixel coordinates 0..63, "
            "origin top-left). Directions correspond to ACTION1..4, action5 to ACTION5, "
            "click to ACTION6, undo to ACTION7; effects are game-dependent. Only numbered "
            "actions in available_actions are legal. action_spec lists currently available "
            "forms; click 32 32 is an example of the full click X Y grammar, not the only target. "
            "Dictionary equivalents are {id: N} or {id: 6, x: X, y: Y}. "
            "reset is RESET/0: generally retries the current level after an action, but can "
            "restart the entire game at zero actions or after a level transition; inspect "
            "full_reset and levels_completed. GAME_OVER permits explicit reset recovery. "
            "WIN ends the episode. noop returns the same engine state. Every submitted request "
            "including noop, retry, undo and invalid input consumes the controller budget; "
            "initialization does not. Invalid actions raise ValueError before engine mutation. "
            "Observations: frame is the final 64x64 palette grid, indexed frame[y][x], values "
            "0..15 (0 white, 5 black); screenshot is its enlarged PNG path. "
            "animation_frame_count counts temporal frames; animation_frames_path, when present, "
            "is a JSON file containing ALL transition grids in order, including the final grid. "
            "state, levels_completed, win_levels, available_actions, full_reset and last_action "
            "are public SDK data; last_action echoes the last SDK action (noop does not change it). "
            "legal_actions mirrors action_spec. observation_id counts submitted requests. "
            "episode_actions, numbered_actions, retries, undos, noops and invalid_actions are "
            "separate counters. score is the highest levels_completed this episode, bounded by "
            "win_levels; reward is its increase on the last request, never duplicate retry credit. "
            "completion_fraction=score/win_levels; won, done, truncated, truncation_reason and "
            "actions_remaining distinguish success from budgets/interface failure. Observe is "
            "read-only; repeated reads do not pay reward again. Elapsed wall time is in info(). "
            "A new evaluator reset creates a fresh game with no adaptor game-specific memory; "
            "retain learned mechanics between levels within that episode. The evaluator must "
            "clear its own game-specific memory between episodes and hold out entire game families. "
            f"Budgets: {self.max_actions} total requests, {self.max_invalid_actions} invalid requests, "
            f"wall time {self.max_seconds!r} seconds (None means unlimited)."
        )

    def close(self) -> None:
        """Release the live wrapper and close diagnostics without altering earned score."""
        self._env = None
        if self._card is not None:
            card, self._card = self._card, None
            self._arc.close_scorecard(card)


def make_game(**kwargs: Any) -> ARCGame:
    return ARCGame(**kwargs)
