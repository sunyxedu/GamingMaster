"""TurnGame adaptor for the open source Baba Is AI benchmark."""
from __future__ import annotations

import json
import math
from typing import Any

import baba
from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = True

# The upstream project supplies a finite catalogue of generated puzzle templates.
# We pin a versioned, sorted selection here so a seed identifies a concrete
# template/configuration rather than merely reseeding one board.
_LEVELS = [
    "env/goto_win", "env/goto_win-distr_obj", "env/goto_win-distr_rule",
    "env/make_win", "env/make_win-distr_obj", "env/make_win-distr_rule",
    "env/make_win-distr_obj-irrelevant_rule", "env/make_win-distr_obj_rule",
    "env/two_room-goto_win", "env/two_room-goto_win-distr_obj",
    "env/two_room-break_stop-goto_win", "env/two_room-break_stop-goto_win-distr_obj",
    "env/two_room-make_win", "env/two_room-make_win-distr_obj",
    "env/two_room-break_stop-make_win", "env/two_room-break_stop-make_win-distr_obj",
    "env/two_room-make_you", "env/two_room-make_you-make_win",
    "env/two_room-make_wall_win", "env/you_win-fixed_you",
]
SEEDS = {
    "train": list(range(100, 108)),
    "heldout": list(range(200, 208)),
    "dev": [300, 301],
    "validation": list(range(400, 408)),
}
_GROUP_LEVELS = {
    **{s: [_LEVELS[i] for i in range(0, 8)] for s in SEEDS["train"]},
    **{s: [_LEVELS[i] for i in range(8, 16)] for s in SEEDS["heldout"]},
    300: [_LEVELS[4]], 301: [_LEVELS[12]],
    **{s: [_LEVELS[i] for i in range(8, 16)] for s in SEEDS["validation"]},
}

_ACTIONS = {"noop": 0, "up": 1, "right": 2, "down": 3, "left": 4}


def _jsonable(value: Any) -> Any:
    """Convert numpy/enums/sets and engine objects to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return str(value)


class BabaGame:
    name = "Baba Is AI"
    action_spec = ActionSpec(
        names=("noop", "up", "right", "down", "left"),
        noop="noop",
        descriptions={
            "noop": "Wait one game turn without moving.",
            "up": "Move the controlled object one cell upward (north).",
            "right": "Move the controlled object one cell right (east).",
            "down": "Move the controlled object one cell downward (south).",
            "left": "Move the controlled object one cell left (west).",
        },
        notes="Submit one primitive action per turn. Invalid action values are ignored and do not advance the engine.",
    )

    def __init__(self, max_steps: int = 200):
        self.max_steps = int(max_steps)
        self.env = None
        self.seed = None
        self.level_id = None
        self._done = False
        self._success = False
        self._termination = "not_started"
        self._last_error = None

    def reset(self, seed: int):
        self.close()
        self.seed = int(seed)
        if seed in _GROUP_LEVELS:
            levels = _GROUP_LEVELS[seed]
            level_id = levels[(seed - min(SEEDS.get("train", [seed]))) % len(levels)] if seed not in (300, 301) else levels[0]
        else:
            level_id = _LEVELS[abs(self.seed) % len(_LEVELS)]
        self.level_id = level_id
        self.env = baba.make(level_id, max_steps=self.max_steps)
        self.env.reset(seed=self.seed)
        self._done = False
        self._success = False
        self._termination = "running"
        self._last_error = None
        return self.observe()

    @staticmethod
    def _object(obj: Any) -> dict[str, Any]:
        out = {"type": getattr(obj, "type", type(obj).__name__), "color": getattr(obj, "color", None)}
        for attr in ("name", "object", "property", "obj_color", "dir", "has_moved"):
            if hasattr(obj, attr):
                out[attr] = _jsonable(getattr(obj, attr))
        return out

    def observe(self) -> dict[str, Any]:
        if self.env is None:
            return {"game": self.name, "status": "not_started"}
        cells = []
        for y in range(self.env.height):
            row = []
            for x in range(self.env.width):
                stack = self.env.grid.get(x, y, "all")
                row.append([self._object(o) for o in stack if o is not None])
            cells.append(row)
        rules_obj = self.env.get_ruleset()
        rules = _jsonable(getattr(rules_obj, "ruleset_dict", rules_obj))
        return {
            "game": self.name, "level_id": self.level_id, "seed": self.seed,
            "width": int(self.env.width), "height": int(self.env.height),
            "board": cells, "agent_pos": _jsonable(tuple(self.env.agent_pos)),
            "agent_dir": int(self.env.agent_dir), "rules": rules,
            "step_count": int(self.env.step_count), "max_steps": int(self.env.max_steps),
            "success": bool(self._success), "terminated": bool(self._done),
            "termination_reason": self._termination,
            "last_error": self._last_error,
        }

    def step(self, action: Any):
        if self.env is None:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            return self.observe()
        if isinstance(action, str):
            key = action.strip().lower()
            if key not in _ACTIONS:
                self._last_error = f"invalid action: {action!r}"
                return self.observe()
            native = _ACTIONS[key]
        elif isinstance(action, int) and not isinstance(action, bool) and action in range(5):
            native = int(action)
        else:
            self._last_error = f"invalid action: {action!r}"
            return self.observe()
        _, reward, done, _ = self.env.step(native)
        self._success = bool(getattr(self.env, "is_win", False))
        self._done = bool(done)
        if self._success:
            self._termination = "solved"
        elif bool(getattr(self.env, "is_lose", False)):
            self._termination = "loss"
        elif self._done:
            self._termination = "step_limit"
        self._last_error = None
        return self.observe()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def score(self) -> float:
        return 1.0 if self._success else 0.0

    def score_breakdown(self) -> dict[str, float]:
        attempted = 1.0 if self.env is not None else 0.0
        return {"puzzles_solved": self.score, "puzzles_attempted": attempted, "solved_fraction": self.score / attempted if attempted else 0.0}

    def info(self) -> dict[str, Any]:
        return {"valid_trial": self.env is not None, "seed": self.seed, "level_id": self.level_id,
                "step_count": int(self.env.step_count) if self.env is not None else 0,
                "termination_reason": self._termination}

    def describe(self) -> str:
        return ("Baba Is AI is a fully observable, symbolic, turn-based puzzle. Push physical and rule tiles "
                "to change rules; an active '<object> is you' object is controlled, and touching an active "
                "'<object> is win' object solves the scheduled puzzle. The observation contains every stacked "
                "board object, active rules, coordinates, direction, and terminal status. Coordinates are "
                "(x,y), origin at the top-left. Actions are the five names in action_spec; each advances one turn. "
                "A puzzle scores exactly one point only when the native engine reports win; loss or step limit scores zero.")

    def close(self):
        if self.env is not None:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
        self.env = None


def make_game(**kwargs) -> BabaGame:
    return BabaGame(**kwargs)
