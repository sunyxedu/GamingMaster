"""Offline Sokoban adaptor backed by a pinned Boxoban XSB subset.

The game is deliberately small and self contained: the evaluator sees only JSON
state and primitive movement actions, while this module owns parsing, reachability,
push accounting, and deterministic scenario selection.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = True

# The groups are disjoint seeds drawn from the same Boxoban train/valid corpus.
SEEDS = {
    "train": list(range(100, 110)),
    "heldout": list(range(200, 210)),
    "dev": [300, 301],
    "validation": list(range(400, 410)),
}

# A generous cap leaves room for planning while keeping accidental wandering
# finite.  Callers may override it through EVAL_KWARGS or SOKOBAN_MAX_ACTIONS.
EVAL_KWARGS = {"max_actions": 2000}

_DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_ACTION_SPEC = ActionSpec(
    names=["up", "down", "left", "right", "noop"],
    descriptions={
        "up": "Walk up; if a box is directly above, push it one square up when free.",
        "down": "Walk down; if a box is directly below, push it one square down when free.",
        "left": "Walk left; if a box is directly left, push it one square left when free.",
        "right": "Walk right; if a box is directly right, push it one square right when free.",
        "noop": "Take no movement action; the board and counters other than actions stay unchanged.",
    },
    noop="noop",
    notes="A blocked move is legal but leaves the player and boxes unchanged. Coordinates are [row, column], 0-based.",
)


@dataclass(frozen=True)
class _Level:
    level_id: str
    rows: tuple[str, ...]
    goals: frozenset[tuple[int, int]]
    boxes: frozenset[tuple[int, int]]
    player: tuple[int, int]

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return max((len(row) for row in self.rows), default=0)


def _bundled_dir() -> Path:
    return Path(__file__).resolve().parent / "runtime" / "boxoban"


def _stable_index(seed: int, count: int) -> int:
    digest = hashlib.sha256(str(int(seed)).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _parse_file(path: Path) -> list[_Level]:
    """Parse Boxoban's ``; number`` separated XSB file."""
    lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    levels: list[_Level] = []
    current: list[str] = []
    declared: str | None = None

    def flush() -> None:
        nonlocal current, declared
        if not current:
            return
        # Blank separators are not part of a board.  Interior spaces are.
        while current and current[-1] == "":
            current.pop()
        if not current:
            return
        width = max(map(len, current))
        # Missing cells in a ragged XSB row are outside the board and therefore
        # walls; explicit spaces that occur inside a row remain floor.
        rows = tuple(row.ljust(width, "#") for row in current)
        goals: set[tuple[int, int]] = set()
        boxes: set[tuple[int, int]] = set()
        player: tuple[int, int] | None = None
        valid = set("# .$@*+")
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch not in valid:
                    raise ValueError(f"unsupported XSB symbol {ch!r} in {path}")
                if ch in ".*+":
                    goals.add((r, c))
                if ch in "$*":
                    boxes.add((r, c))
                if ch in "@+":
                    if player is not None:
                        raise ValueError(f"multiple players in {path} level {declared}")
                    player = (r, c)
        if player is None or not goals or len(goals) != len(boxes):
            raise ValueError(
                f"invalid XSB level {declared!r} in {path}: player/goals/boxes mismatch"
            )
        level_no = declared if declared is not None else str(len(levels))
        levels.append(_Level(f"{path.stem}:{level_no}", rows, frozenset(goals), frozenset(boxes), player))
        current = []
        declared = None

    for line in lines:
        if line.startswith(";"):
            flush()
            declared = line[1:].strip() or str(len(levels))
        elif line == "" and current:
            # Boxoban has a blank line after every board.
            flush()
        elif line:
            current.append(line)
    flush()
    return levels


def _load_levels(levelset: str | os.PathLike[str] | None) -> tuple[list[_Level], str]:
    root = Path(levelset).expanduser() if levelset else _bundled_dir()
    if not root.exists():
        raise FileNotFoundError(f"Sokoban level corpus does not exist: {root}")
    files = [root] if root.is_file() else sorted(
        [*root.glob("*.txt"), *root.glob("*.xsb")]
    )
    files = [p for p in files if p.name.lower() != "license.txt"]
    if not files:
        raise ValueError(f"no .txt XSB files found in {root}")
    levels: list[_Level] = []
    hashes: list[str] = []
    for path in files:
        data = path.read_bytes()
        hashes.append(f"{path.name}:{hashlib.sha256(data).hexdigest()}")
        levels.extend(_parse_file(path))
    if not levels:
        raise ValueError(f"the level corpus {root} contains no levels")
    corpus_id = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    return levels, corpus_id


class SokobanGame:
    name = "Sokoban (Boxoban ASCII)"
    action_spec = _ACTION_SPEC

    def __init__(self, levelset: str | os.PathLike[str] | None = None, max_actions: int | None = None, **_: Any):
        env_levelset = os.environ.get("SOKOBAN_LEVELSET")
        self._levels, self._corpus_id = _load_levels(levelset or env_levelset)
        configured = max_actions
        if configured is None and os.environ.get("SOKOBAN_MAX_ACTIONS"):
            configured = int(os.environ["SOKOBAN_MAX_ACTIONS"])
        self.max_actions = max(1, int(configured if configured is not None else 2000))
        self._seed: int | None = None
        self._level: _Level | None = None
        self._player: tuple[int, int] = (0, 0)
        self._boxes: set[tuple[int, int]] = set()
        self._actions = 0
        self._pushes = 0
        self._walk_steps = 0
        self._done = False
        self._solved = False
        self._last_action: str | None = None
        self._last_event = "unstarted"
        self._invalid_action: str | None = None

    def reset(self, seed: int) -> dict[str, Any]:
        self._seed = int(seed)
        self._level = self._levels[_stable_index(self._seed, len(self._levels))]
        self._player = self._level.player
        self._boxes = set(self._level.boxes)
        self._actions = self._pushes = self._walk_steps = 0
        self._done = False
        self._solved = self._boxes <= self._level.goals and len(self._boxes) == len(self._level.goals)
        self._last_action = None
        self._last_event = "reset"
        self._invalid_action = None
        if self._solved:
            self._done = True
            self._last_event = "solved"
        return self.observe()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def score(self) -> float:
        # Solved levels dominate; boxes and efficiency break ties while retaining
        # a meaningful gradient for unfinished episodes.
        breakdown = self.score_breakdown()
        return float(
            breakdown["solved"] * 10000.0
            + breakdown["boxes_on_goals"] * 100.0
            - breakdown["pushes"] * 0.5
            - breakdown["total_actions"] * 0.01
        )

    def score_breakdown(self) -> dict[str, float]:
        return {
            "solved": float(1 if self._solved else 0),
            "boxes_on_goals": float(self._boxes_on_goals),
            "pushes": float(self._pushes),
            "walk_steps": float(self._walk_steps),
            "total_actions": float(self._actions),
            "action_cap_terminated": float(1 if self._actions >= self.max_actions and not self._solved else 0),
        }

    @property
    def _boxes_on_goals(self) -> int:
        return len(self._boxes & (self._level.goals if self._level else frozenset()))

    def _inside_floor(self, pos: tuple[int, int]) -> bool:
        assert self._level is not None
        r, c = pos
        return 0 <= r < self._level.height and 0 <= c < self._level.width and self._level.rows[r][c] != "#"

    def _reachable(self) -> set[tuple[int, int]]:
        seen = {self._player}
        todo = [self._player]
        while todo:
            r, c = todo.pop()
            for dr, dc in _DIRECTIONS.values():
                nxt = (r + dr, c + dc)
                if nxt not in seen and self._inside_floor(nxt) and nxt not in self._boxes:
                    seen.add(nxt)
                    todo.append(nxt)
        return seen

    def _valid_actions(self) -> list[str]:
        if self._done:
            return []
        valid: list[str] = ["noop"]
        for action, (dr, dc) in _DIRECTIONS.items():
            nxt = (self._player[0] + dr, self._player[1] + dc)
            if not self._inside_floor(nxt):
                continue
            if nxt not in self._boxes:
                valid.append(action)
                continue
            beyond = (nxt[0] + dr, nxt[1] + dc)
            if self._inside_floor(beyond) and beyond not in self._boxes:
                valid.append(action)
        return valid

    def _render(self) -> str:
        assert self._level is not None
        # Static XSB rows contain the initial player/box glyphs. Rebuild the
        # terrain first so moved pieces do not leave stale characters behind.
        rows = [
            ["#" if ch == "#" else "." if (r, c) in self._level.goals else " " for c, ch in enumerate(row)]
            for r, row in enumerate(self._level.rows)
        ]
        for r, c in self._boxes:
            rows[r][c] = "*" if (r, c) in self._level.goals else "$"
        pr, pc = self._player
        rows[pr][pc] = "+" if (pr, pc) in self._level.goals else "@"
        return "\n".join("".join(row) for row in rows)

    def observe(self) -> dict[str, Any]:
        if self._level is None:
            raise RuntimeError("reset(seed) must be called before observe()")
        reachable = sorted(self._reachable())
        return {
            "level_id": self._level.level_id,
            "seed": self._seed,
            "rows": self._level.height,
            "cols": self._level.width,
            "board": self._render(),
            "player": list(self._player),
            "boxes": [list(p) for p in sorted(self._boxes)],
            "goals": [list(p) for p in sorted(self._level.goals)],
            "reachable": [list(p) for p in reachable],
            "valid_actions": self._valid_actions(),
            "last_action": self._last_action,
            "last_event": self._last_event,
            "pushes": self._pushes,
            "walk_steps": self._walk_steps,
            "actions": self._actions,
            "boxes_on_goals": self._boxes_on_goals,
            "done": self._done,
            "action_cap": self.max_actions,
            "invalid_action": self._invalid_action,
        }

    def step(self, action: Any) -> dict[str, Any]:
        if self._level is None:
            raise RuntimeError("reset(seed) must be called before step()")
        self._invalid_action = None
        self._last_action = action if isinstance(action, str) else None
        if self._done:
            self._last_event = "already_done"
            return self.observe()
        self._actions += 1
        if not isinstance(action, str) or action not in self.action_spec.names:
            self._invalid_action = repr(action)[:120]
            self._last_event = "invalid_action"
        elif action == "noop":
            self._last_event = "noop"
        else:
            dr, dc = _DIRECTIONS[action]
            nxt = (self._player[0] + dr, self._player[1] + dc)
            if not self._inside_floor(nxt):
                self._last_event = "blocked"
            elif nxt in self._boxes:
                beyond = (nxt[0] + dr, nxt[1] + dc)
                if not self._inside_floor(beyond) or beyond in self._boxes:
                    self._last_event = "blocked"
                else:
                    self._boxes.remove(nxt)
                    self._boxes.add(beyond)
                    self._player = nxt
                    self._pushes += 1
                    self._last_event = "push"
            else:
                self._player = nxt
                self._walk_steps += 1
                self._last_event = "walk"
        if not self._solved and self._boxes_on_goals == len(self._level.goals):
            self._solved = True
            self._done = True
            self._last_event = "solved"
        if not self._done and self._actions >= self.max_actions:
            self._done = True
            self._last_event = "action_cap"
        return self.observe()

    def info(self) -> dict[str, Any]:
        return {
            "valid_trial": True,
            "seed": self._seed,
            "level_id": self._level.level_id if self._level else None,
            "level_index": _stable_index(self._seed, len(self._levels)) if self._seed is not None else None,
            "corpus_sha256": self._corpus_id,
            "corpus_levels": len(self._levels),
            "max_actions": self.max_actions,
        }

    def describe(self) -> str:
        return (
            "Sokoban is a deterministic turn-based box-pushing puzzle. Reach every goal ('.') with a box. "
            "The player ('@') walks on floor and cannot pull boxes; pushing a box requires the square behind it "
            "to be free, and a push cannot be undone unless another route allows it. Walls are '#', boxes '$', "
            "boxes on goals '*', player on a goal '+'. Each step accepts exactly one action string: up, down, "
            "left, right, or noop. A blocked or malformed action consumes one action but leaves the board still. "
            "The observation is JSON with the ASCII board, coordinates, reachable player cells, legal actions, "
            "pushes, walking steps, total actions, boxes_on_goals, done, and the action cap. The episode ends "
            "when all goals have boxes or the action cap is reached. Score prioritizes solved, then boxes on goals, "
            "then fewer pushes and actions; score_breakdown() exposes solved, boxes_on_goals, pushes, walk_steps, "
            "total_actions, and action-cap termination. Seeds select fixed Boxoban levels by SHA-256 modulo the "
            "pinned corpus, so reset(seed) is reproducible."
        )

    def close(self) -> None:
        # The adaptor owns no external process or connection.  Clearing state
        # makes accidental post-close use fail loudly while remaining idempotent.
        self._level = None
        self._boxes.clear()


def make_game(**kwargs: Any) -> SokobanGame:
    return SokobanGame(**kwargs)
