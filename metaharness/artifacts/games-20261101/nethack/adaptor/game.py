"""NetHack Learning Environment adaptor for the Meta-Harness turn interface.

The engine is local and headless.  ``NetHackScore`` supplies a complete NetHack
turn for each call to :meth:`step`; this module only converts its numpy
observations to strict JSON and gives actions readable names.
"""

from __future__ import annotations

import math
from typing import Any

from metaharness.games.base import ActionSpec

try:  # NLE 1.3 and later
    from nle.env.tasks import NetHackScore, NetHackStaircase, TASK_ACTIONS
except ImportError:  # pragma: no cover - compatibility with older NLE releases
    from nle.env import NetHackScore  # type: ignore
    from nle.env.tasks import NetHackStaircase, TASK_ACTIONS  # type: ignore

from nle import nethack


TIER = 1
DETERMINISTIC = True

# The four groups use the same broad mix of random roles, maps and starting
# inventories.  They are deliberately disjoint so search cannot train on the
# held-out episodes.  A seed selects a reproducible engine RNG stream; it does
# not select a fixed map.
SEEDS = {
    "train": [1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108],
    "heldout": [2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208],
    "dev": [3304, 3305],
    "validation": [4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408],
}
EVAL_KWARGS: dict[str, Any] = {}


_ACTION_NAMES = (
    "more",
    "move n",
    "move e",
    "move s",
    "move w",
    "move ne",
    "move se",
    "move sw",
    "move nw",
    "run n",
    "run e",
    "run s",
    "run w",
    "run ne",
    "run se",
    "run sw",
    "run nw",
    "up",
    "down",
    "wait",
    "kick",
    "eat",
    "search",
)

_ACTION_DESCRIPTIONS = {
    "more": "Dismiss a --More-- prompt or advance a text screen.",
    "move n": "Move one square north (vi key k).",
    "move e": "Move one square east (vi key l).",
    "move s": "Move one square south (vi key j).",
    "move w": "Move one square west (vi key h).",
    "move ne": "Move one square northeast (vi key u).",
    "move se": "Move one square southeast (vi key n).",
    "move sw": "Move one square southwest (vi key b).",
    "move nw": "Move one square northwest (vi key y).",
    "run n": "Run north until an obstacle or event.",
    "run e": "Run east until an obstacle or event.",
    "run s": "Run south until an obstacle or event.",
    "run w": "Run west until an obstacle or event.",
    "run ne": "Run northeast until an obstacle or event.",
    "run se": "Run southeast until an obstacle or event.",
    "run sw": "Run southwest until an obstacle or event.",
    "run nw": "Run northwest until an obstacle or event.",
    "up": "Climb an upstairs when standing on one.",
    "down": "Descend a downstairs when standing on one.",
    "wait": "Wait one turn (the safe no-op when no decision is ready).",
    "kick": "Kick in a direction; a following direction may be requested.",
    "eat": "Eat an edible item; a following inventory letter may be requested.",
    "search": "Search adjacent squares for hidden doors and traps.",
}

# Friendly aliases are useful when a model spells out a direction.  The
# canonical names above remain the only names advertised to the harness.
_ALIASES = {
    "n": "move n",
    "e": "move e",
    "s": "move s",
    "w": "move w",
    "ne": "move ne",
    "se": "move se",
    "sw": "move sw",
    "nw": "move nw",
    "north": "move n",
    "east": "move e",
    "south": "move s",
    "west": "move w",
    "wait/noop": "wait",
    "noop": "wait",
}

_STATS = {
    "x": nethack.NLE_BL_X,
    "y": nethack.NLE_BL_Y,
    "strength": nethack.NLE_BL_STR25,
    "dexterity": nethack.NLE_BL_DEX,
    "constitution": nethack.NLE_BL_CON,
    "intelligence": nethack.NLE_BL_INT,
    "wisdom": nethack.NLE_BL_WIS,
    "charisma": nethack.NLE_BL_CHA,
    "hp": nethack.NLE_BL_HP,
    "hp_max": nethack.NLE_BL_HPMAX,
    "armor_class": nethack.NLE_BL_AC,
    "experience_level": nethack.NLE_BL_XP,
    "experience_points": nethack.NLE_BL_EXP,
    "gold": nethack.NLE_BL_GOLD,
    "depth": nethack.NLE_BL_DEPTH,
    "dungeon_level": nethack.NLE_BL_DLEVEL,
    "dungeon_number": nethack.NLE_BL_DNUM,
    "hunger": nethack.NLE_BL_HUNGER,
    "time": nethack.NLE_BL_TIME,
    "score": nethack.NLE_BL_SCORE,
    "alignment": nethack.NLE_BL_ALIGN,
    "capacity": nethack.NLE_BL_CAP,
    "conditions": nethack.NLE_BL_CONDITION,
}


def _seed_triplet(seed: int) -> tuple[int, int, int]:
    """Derive all NLE RNG streams from the evaluator's one integer seed."""

    modulus = 2**31 - 1
    core = int(seed) % modulus
    # Distinct streams avoid coupling level generation to ordinary actions.
    display = (core * 1103515245 + 12345) % modulus
    level = (core ^ 0x5DEECE66D) % modulus
    return core, display, level


def _decode_bytes(value: Any) -> str:
    try:
        raw = bytes(int(x) & 0xFF for x in value)
    except Exception:
        return ""
    raw = raw.split(b"\0", 1)[0]
    return raw.decode("utf-8", errors="replace")


def _text_rows(value: Any) -> list[str]:
    try:
        return [_decode_bytes(row) for row in value]
    except Exception:
        return []


def _array_list(value: Any) -> list[Any]:
    """Convert a numpy array (or an absent optional field) to plain lists."""
    try:
        return value.tolist()
    except AttributeError:
        return list(value) if value is not None else []
    except Exception:
        return []


class NetHackGame:
    name = "NetHack (NetHack Learning Environment)"
    action_spec = ActionSpec(
        names=_ACTION_NAMES,
        descriptions=_ACTION_DESCRIPTIONS,
        noop="wait",
        notes=(
            "Each action consumes one NetHack turn. Directions use vi-key conventions. "
            "Actions such as kick/eat can leave a prompt in message/tty_chars; answer it "
            "with another action (MORE clears --More-- screens). Integer action indices "
            "0..22 are accepted as an equivalent compact grammar."
        ),
    )

    def __init__(
        self,
        *,
        character: str = "@",
        max_episode_steps: int = 5000,
        penalty_step: float = -0.01,
        penalty_time: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._character = character
        self._max_episode_steps = int(max_episode_steps)
        self._penalty_step = float(penalty_step)
        self._penalty_time = float(penalty_time)
        self._extra_kwargs = dict(kwargs)
        self._env: Any | None = None
        self._obs: dict[str, Any] | None = None
        self._raw_info: dict[str, Any] = {}
        self._done = False
        self._truncated = False
        self._closed = False
        self._last_reward = 0.0
        self._last_error: str | None = None
        self._valid_trial = True
        self._max_depth = 0
        self._seed: int | None = None
        self._scenario = "score"
        self._task_kind = "score"
        self._make_env()

    def _make_env(self, task_kind: str | None = None) -> None:
        kind = task_kind or self._task_kind
        task_cls = NetHackStaircase if kind == "staircase" else NetHackScore
        env_kwargs = dict(
            character=self._character,
            max_episode_steps=self._max_episode_steps,
            penalty_step=self._penalty_step,
            penalty_time=self._penalty_time,
            # Let the observation expose prompts instead of silently choosing a
            # dangerous answer. NLE still handles its ordinary startup screens.
            allow_all_yn_questions=True,
            allow_all_modes=True,
            fix_moon_phase=True,
        )
        env_kwargs.update(self._extra_kwargs)
        self._env = task_cls(**env_kwargs)

    def reset(self, seed: int) -> dict[str, Any]:
        if self._closed:
            self._closed = False
        # Every fourth stream is a short staircase curriculum episode.  It uses
        # the same action/observation interface and terminal score, but ends as
        # soon as a downstairs is reached, giving search a useful progress
        # milestone without fixing any map.
        task_kind = "staircase" if int(seed) % 4 == 0 else "score"
        if self._env is None or task_kind != self._task_kind:
            old_env, self._env = self._env, None
            if old_env is not None:
                try:
                    old_env.close()
                except Exception:
                    pass
            self._task_kind = task_kind
            self._make_env(task_kind)
        assert self._env is not None
        self._seed = int(seed)
        core, display, level = _seed_triplet(self._seed)
        # NLE's Gym seed alone seeds its Python RNG. Explicitly seed all three
        # NetHack streams so a fresh adaptor reproduces the same first screen.
        self._env.seed(core=core, disp=display, reseed=False, lgen=level)
        result = self._env.reset(seed=self._seed)
        self._obs, self._raw_info = self._split_reset(result)
        self._done = False
        self._truncated = False
        self._last_reward = 0.0
        self._last_error = None
        self._valid_trial = True
        self._max_depth = self._depth(self._obs)
        self._scenario = task_kind
        return self.observe()

    @staticmethod
    def _split_reset(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
            return result[0], dict(result[1] or {})
        return result, {}

    @staticmethod
    def _split_step(result: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})
        obs, reward, done, info = result
        return obs, float(reward), bool(done), False, dict(info or {})

    @staticmethod
    def _depth(obs: dict[str, Any] | None) -> int:
        if not obs:
            return 0
        try:
            return max(0, int(obs["blstats"][nethack.NLE_BL_DEPTH]))
        except Exception:
            return 0

    def _resolve_action(self, action: Any) -> int | None:
        if isinstance(action, bool):
            return None
        if isinstance(action, int):
            return action if 0 <= action < len(TASK_ACTIONS) else None
        if isinstance(action, str):
            key = " ".join(action.strip().lower().split())
            key = _ALIASES.get(key, key)
            try:
                return _ACTION_NAMES.index(key)
            except ValueError:
                # A compact numeric string is convenient for scripted clients.
                try:
                    number = int(key, 0)
                except ValueError:
                    return None
                return number if 0 <= number < len(TASK_ACTIONS) else None
        return None

    def step(self, action: Any) -> dict[str, Any]:
        if self._obs is None:
            self._last_error = "step called before reset"
            return self.observe()
        index = self._resolve_action(action)
        if index is None:
            self._last_error = f"ignored malformed action: {action!r}"[:240]
            return self.observe()
        if self._done or self._env is None:
            self._last_error = "action ignored after episode ended"
            return self.observe()
        try:
            result = self._env.step(index)
            self._obs, self._last_reward, terminated, truncated, self._raw_info = self._split_step(result)
            self._done = terminated or truncated
            self._truncated = truncated
            self._max_depth = max(self._max_depth, self._depth(self._obs))
            self._last_error = None
        except Exception as exc:  # infrastructure failures are reported in info()
            self._valid_trial = False
            self._last_error = f"engine error: {type(exc).__name__}: {exc}"[:240]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        if self._obs is None:
            return {"ready": False, "error": self._last_error or "reset has not been called"}
        source = self._obs
        blstats = [int(x) for x in source.get("blstats", [])]
        chars = source.get("chars", [])
        tty_chars = source.get("tty_chars", [])
        stats = {
            name: int(blstats[index])
            for name, index in _STATS.items()
            if index < len(blstats)
        }
        result: dict[str, Any] = {
            "glyphs": _array_list(source.get("glyphs", [])),
            "chars": _array_list(source.get("chars", [])),
            "colors": _array_list(source.get("colors", [])),
            "specials": _array_list(source.get("specials", [])),
            "blstats": blstats,
            "stats": stats,
            "message": _decode_bytes(source.get("message", [])),
            "map": _text_rows(chars),
            "tty": _text_rows(tty_chars),
            "tty_colors": _array_list(source.get("tty_colors", [])),
            "tty_cursor": [int(x) for x in source.get("tty_cursor", [])],
            "inventory": _text_rows(source.get("inv_strs", [])),
            "inventory_letters": [int(x) for x in source.get("inv_letters", [])],
            "score": float(self.score),
            "max_depth": int(self._max_depth),
            "ascended": bool(self._raw_info.get("is_ascended", False)),
            "scenario_success": bool(self._raw_info.get("is_ascended", False))
            or int(self._raw_info.get("end_status", 0)) == 2,
            "done": bool(self._done),
            "truncated": bool(self._truncated),
            "turn_reward": float(self._last_reward),
            "seed": self._seed,
            "scenario": self._scenario,
        }
        if self._last_error:
            result["error"] = self._last_error
        return result

    @property
    def done(self) -> bool:
        return bool(self._done)

    @property
    def score(self) -> float:
        try:
            value = float(self._obs["blstats"][nethack.NLE_BL_SCORE]) if self._obs else 0.0
            return value if math.isfinite(value) else 0.0
        except Exception:
            return 0.0

    def score_breakdown(self) -> dict[str, float]:
        success = bool(self._raw_info.get("is_ascended", False)) or int(self._raw_info.get("end_status", 0)) == 2
        return {
            "terminal_score": float(self.score),
            "maximum_dungeon_depth": float(self._max_depth),
            "ascended": 1.0 if self._raw_info.get("is_ascended", False) else 0.0,
            "scenario_success": 1.0 if success else 0.0,
        }

    def info(self) -> dict[str, Any]:
        success = bool(self._raw_info.get("is_ascended", False)) or int(self._raw_info.get("end_status", 0)) == 2
        info: dict[str, Any] = {
            "valid_trial": bool(self._valid_trial),
            "seed": self._seed,
            "scenario": self._scenario,
            "score": float(self.score),
            "max_depth": int(self._max_depth),
            "is_ascended": bool(self._raw_info.get("is_ascended", False)),
            "scenario_success": success,
            "end_status": int(self._raw_info.get("end_status", 0)),
        }
        if not self._valid_trial:
            info["invalid_reason"] = self._last_error or "NLE engine failure"
        if self._last_error:
            info["last_error"] = self._last_error
        return info

    def describe(self) -> str:
        return (
            "NetHack is a turn-based dungeon exploration game. Your objective is to survive, "
            "gain experience and gold, descend through the dungeon, and ultimately retrieve "
            "the Amulet and ascend; the primary score is NetHack's terminal score, with depth "
            "and ascension reported separately. Every step consumes one command and monsters "
            "may act during it. Observation fields include a 21x79 glyph and character map, "
            "terminal text (tty), cursor, latest message, inventory text, raw blstats and a "
            "named stats dictionary (HP, hunger, gold, depth, score and turn). Use action names "
            "from action_spec; movement is `move <n|e|s|w|ne|se|sw|nw>`, running is `run <dir>`, "
            "and prompts are answered by another command. `wait` is the no-op."
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        env, self._env = self._env, None
        if env is not None:
            try:
                env.close()
            except Exception:
                self._valid_trial = False


def make_game(**kwargs: Any) -> NetHackGame:
    """Construct one independent local NetHack episode."""

    return NetHackGame(**kwargs)
