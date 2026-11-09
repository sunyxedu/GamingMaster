"""Headless, symbolic, real-time Guideline-style Tetris adaptor.

The adaptor deliberately keeps the game engine in this file.  It has no display,
network client, account, or background process: the evaluator's calls to ``tick``
are the wall-clock cadence.  A tick represents ``tick_s`` seconds and applies one
named input before advancing gravity and lock delay.
"""
from __future__ import annotations

import random
import threading
from typing import Any

try:
    from metaharness.games.base import ActionSpec
except ImportError:  # Useful when loading this file outside the metaharness checkout.
    from dataclasses import dataclass, field

    @dataclass
    class ActionSpec:  # type: ignore[no-redef]
        names: list[str]
        descriptions: dict[str, str] = field(default_factory=dict)
        noop: str | None = None
        notes: str = ""

        def describe(self) -> str:
            lines = [f"- {n}: {self.descriptions.get(n, '')}".rstrip() for n in self.names]
            if self.noop:
                lines.append(f"(real-time games: ticks apply '{self.noop}' until the first action is posted; "
                             "afterwards a tick without a fresh action repeats the last posted action)")
            if self.notes:
                lines.append(self.notes)
            return "\n".join(lines)


TIER = 2
DETERMINISTIC = True
# These are ordinary empty-board marathon seeds.  Optional controlled scenarios are
# selected with make_game(scenario=...), not by making ordinary seed reset implicit.
SEEDS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "heldout": [101, 102, 103, 104, 105, 106, 107],
}

WIDTH = 10
HEIGHT = 20
LOCK_DELAY_S = 0.5
MAX_LOCK_RESETS = 15
PIECES = ("I", "J", "L", "O", "S", "T", "Z")
PIECE_IDS = {p: i + 1 for i, p in enumerate(PIECES)}

# Tetris Worlds / Guideline gravity, seconds per cell.  The last entries are
# deliberately bounded so a 20 Hz interface still has a usable deterministic clock.
GRAVITY_S = (
    1.000, 0.793, 0.618, 0.473, 0.355, 0.262, 0.190, 0.135, 0.094,
    0.064, 0.043, 0.029, 0.019, 0.012, 0.008, 0.005, 0.003, 0.002,
)

# Each matrix is written top-to-bottom; it is converted to local coordinates with
# y=0 at the bottom.  Repeated clockwise rotation uses the SRS 3x3 pivot for
# JLSTZ/T/S/Z and the 4x4 pivot for I.  O has a fixed orientation.
_BASE_MATRICES: dict[str, tuple[str, ...]] = {
    "J": ("J..", "JJJ", "..."),
    "L": ("..L", "LLL", "..."),
    "S": (".SS", "SS.", "..."),
    "Z": ("ZZ.", ".ZZ", "..."),
    "T": (".T.", "TTT", "..."),
    "I": ("....", "IIII", "....", "...."),
    "O": ("....", ".OO.", ".OO.", "...."),
}


def _cw(matrix: tuple[str, ...]) -> tuple[str, ...]:
    n = len(matrix)
    return tuple("".join(matrix[n - 1 - r][c] for r in range(n)) for c in range(n))


def _matrix_cells(matrix: tuple[str, ...], piece: str) -> tuple[tuple[int, int], ...]:
    n = len(matrix)
    out: list[tuple[int, int]] = []
    for row, values in enumerate(matrix):
        for col, value in enumerate(values):
            if value == piece:
                out.append((col, n - 1 - row))
    return tuple(out)


def _all_orientations(piece: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    if piece == "O":
        cells = _matrix_cells(_BASE_MATRICES[piece], piece)
        return (cells, cells, cells, cells)
    matrix = _BASE_MATRICES[piece]
    result = []
    for _ in range(4):
        result.append(_matrix_cells(matrix, piece))
        matrix = _cw(matrix)
    return tuple(result)


SHAPES = {piece: _all_orientations(piece) for piece in PIECES}

# SRS kick offsets, with positive y meaning upward (our board uses y upward).
_JLSTZ_KICKS: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (1, 0): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (1, 2): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (2, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (2, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (0, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
}
_I_KICKS: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
    (1, 0): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    (2, 1): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
    (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
    (3, 0): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
    (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
}

_ACTIONS = ("left", "right", "soft_drop", "rotate_cw", "rotate_ccw", "hard_drop", "hold", "noop")
_ACTION_DESCRIPTIONS = {
    "left": "Move the active tetromino one column left if legal.",
    "right": "Move the active tetromino one column right if legal.",
    "soft_drop": "Move down one cell if legal; awards one point per cell moved.",
    "rotate_cw": "Rotate clockwise using the Guideline SRS wall-kick tests.",
    "rotate_ccw": "Rotate counter-clockwise using the Guideline SRS wall-kick tests.",
    "hard_drop": "Drop to the ghost position, award two points per cell, and lock immediately.",
    "hold": "Swap with hold (once per falling piece), or store the current piece if hold is empty.",
    "noop": "Take no input; gravity and lock delay still advance.",
}


class _Tetris:
    """The stateful game exposed through the RealtimeGame protocol."""

    name = "Tetris (Guideline-compatible marathon)"
    tick_s = 0.05
    action_spec = ActionSpec(
        names=list(_ACTIONS),
        descriptions=_ACTION_DESCRIPTIONS,
        noop="noop",
        notes=(
            "Actions are strings. A dict {'action': name} is also accepted. Each tick is 0.05 s; "
            "the current action is applied before gravity. Invalid actions are ignored and counted."
        ),
    )

    def __init__(
        self,
        *,
        scenario: str = "standard",
        max_pieces: int = 200,
        max_ticks: int = 20_000,
        tick_s: float = 0.05,
        **_: Any,
    ) -> None:
        self._lock = threading.RLock()
        self.scenario = str(scenario).lower()
        if self.scenario not in {"standard", "low", "medium", "high", "recovery"}:
            raise ValueError("scenario must be standard, low, medium, high, or recovery")
        self.max_pieces = max(1, int(max_pieces))
        self.max_ticks = max(1, int(max_ticks))
        self.tick_s = float(tick_s)
        if not (self.tick_s > 0.0 and self.tick_s < 10.0):
            raise ValueError("tick_s must be positive and below 10 seconds")
        self._closed = False
        self.reset(0)

    def reset(self, seed: int) -> dict[str, Any]:
        with self._lock:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be an int")
            self._closed = False
            self.seed = int(seed)
            self._rng = random.Random(self.seed)
            self._bag: list[str] = []
            self._board: list[list[int]] = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
            self._queue: list[str] = []
            self._hold: str | None = None
            self._can_hold = True
            self._active_piece: str | None = None
            self._orientation = 0
            self._x = 3
            self._y = 17
            self._gravity_accum = 0.0
            self._lock_elapsed: float | None = None
            self._lock_resets = 0
            self._last_rotation = False
            self._last_action = "noop"
            self._last_attempted_action = "noop"
            self._events: list[dict[str, Any]] = [{"type": "reset", "seed": self.seed}]
            self._score = 0
            self._line_score = 0
            self._drop_score = 0
            self._combo_score = 0
            self._b2b_score = 0
            self._lines = 0
            self._level = self._scenario_level()
            self._combo = -1
            self._back_to_back = False
            self._pieces = 0
            self._ticks = 0
            self._elapsed_s = 0.0
            self._illegal_actions = 0
            self._done = False
            self._terminal_reason: str | None = None
            self._top_out_height = 0
            self._last_clear: dict[str, Any] = {"lines": 0, "kind": "none", "score": 0}
            if self.scenario == "recovery":
                self._make_recovery_start()
            for _ in range(5):
                self._queue.append(self._draw_piece())
            self._spawn_piece(self._queue.pop(0))
            self._fill_queue()
            return self.observe()

    def _scenario_level(self) -> int:
        return {"standard": 1, "low": 1, "medium": 5, "high": 10, "recovery": 3}[self.scenario]

    def _make_recovery_start(self) -> None:
        # A deterministic controlled board for optional recovery evaluations: five
        # uneven rows, holes in the middle, and a clean right-side well.
        patterns = (
            "1111111100",
            "1101111000",
            "1110111100",
            "1011111100",
            "1111011100",
        )
        for y, row in enumerate(patterns):
            self._board[y] = [1 if c == "1" else 0 for c in row]

    def _draw_piece(self) -> str:
        if not self._bag:
            self._bag = list(PIECES)
            self._rng.shuffle(self._bag)
        return self._bag.pop()

    def _fill_queue(self) -> None:
        while len(self._queue) < 5:
            self._queue.append(self._draw_piece())

    def _spawn_piece(self, piece: str) -> bool:
        self._active_piece = piece
        self._orientation = 0
        self._x = 3
        self._y = 17
        self._can_hold = True
        self._gravity_accum = 0.0
        self._lock_elapsed = None
        self._lock_resets = 0
        self._last_rotation = False
        if not self._can_place(piece, 0, self._x, self._y):
            self._finish("top_out")
            return False
        return True

    def _finish(self, reason: str) -> None:
        self._done = True
        self._terminal_reason = reason
        self._events.append({"type": "terminal", "reason": reason})

    def _cells(self, piece: str | None = None, orientation: int | None = None,
               x: int | None = None, y: int | None = None) -> tuple[tuple[int, int], ...]:
        p = self._active_piece if piece is None else piece
        if p is None:
            return ()
        o = self._orientation if orientation is None else orientation % 4
        xx = self._x if x is None else x
        yy = self._y if y is None else y
        return tuple((xx + cx, yy + cy) for cx, cy in SHAPES[p][o])

    def _can_place(self, piece: str, orientation: int, x: int, y: int) -> bool:
        for cx, cy in self._cells(piece, orientation, x, y):
            if cx < 0 or cx >= WIDTH or cy < 0:
                return False
            if cy < HEIGHT and self._board[cy][cx]:
                return False
        return True

    def _is_grounded(self) -> bool:
        if self._active_piece is None:
            return False
        return not self._can_place(self._active_piece, self._orientation, self._x, self._y - 1)

    def _set_manual_success(self, *, rotation: bool = False) -> None:
        if rotation:
            self._last_rotation = True
        else:
            self._last_rotation = False
        if self._is_grounded():
            if self._lock_elapsed is None:
                self._lock_elapsed = 0.0
            elif self._lock_resets < MAX_LOCK_RESETS:
                self._lock_elapsed = 0.0
                self._lock_resets += 1
        else:
            self._lock_elapsed = None

    def _move(self, dx: int, dy: int) -> bool:
        if self._active_piece is None:
            return False
        if self._can_place(self._active_piece, self._orientation, self._x + dx, self._y + dy):
            self._x += dx
            self._y += dy
            self._set_manual_success()
            return True
        return False

    def _rotate(self, direction: int) -> bool:
        if self._active_piece is None:
            return False
        old = self._orientation
        new = (old + direction) % 4
        if self._active_piece == "O":
            self._orientation = new
            self._set_manual_success(rotation=True)
            return True
        kicks = _I_KICKS if self._active_piece == "I" else _JLSTZ_KICKS
        for dx, dy in kicks.get((old, new), ((0, 0),)):
            if self._can_place(self._active_piece, new, self._x + dx, self._y + dy):
                self._x += dx
                self._y += dy
                self._orientation = new
                self._set_manual_success(rotation=True)
                return True
        return False

    def _ghost_y(self) -> int:
        if self._active_piece is None:
            return self._y
        y = self._y
        while self._can_place(self._active_piece, self._orientation, self._x, y - 1):
            y -= 1
        return y

    def _hard_drop(self) -> None:
        if self._active_piece is None:
            return
        target = self._ghost_y()
        distance = max(0, self._y - target)
        self._y = target
        self._drop_score += distance * 2
        self._score += distance * 2
        self._events.append({"type": "hard_drop", "cells": distance, "points": distance * 2})
        self._lock_piece()

    def _hold_piece(self) -> None:
        if not self._can_hold or self._active_piece is None:
            self._illegal_actions += 1
            self._events.append({"type": "illegal_action", "action": "hold", "reason": "hold_already_used"})
            return
        current = self._active_piece
        self._can_hold = False
        self._last_rotation = False
        if self._hold is None:
            self._hold = current
            next_piece = self._queue.pop(0)
            self._fill_queue()
        else:
            self._hold, next_piece = current, self._hold
        self._spawn_piece(next_piece)
        # Spawning normally enables hold for a newly locked piece; a hold action
        # is the exception, so the replacement cannot be held again until it locks.
        self._can_hold = False
        self._events.append({"type": "hold", "piece": self._hold, "active": next_piece})

    def _t_spin_kind(self) -> str:
        if self._active_piece != "T" or not self._last_rotation:
            return "none"
        px, py = self._x + 1, self._y + 1
        occupied = 0
        for dx, dy in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
            xx, yy = px + dx, py + dy
            if xx < 0 or xx >= WIDTH or yy < 0 or yy >= HEIGHT or self._board[yy][xx]:
                occupied += 1
        if occupied >= 3:
            return "t_spin"
        if occupied == 2:
            return "t_spin_mini"
        return "none"

    def _lock_piece(self) -> None:
        if self._active_piece is None or self._done:
            return
        piece = self._active_piece
        kind = self._t_spin_kind()
        cells = self._cells()
        above_top = any(y >= HEIGHT for _, y in cells)
        for x, y in cells:
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                self._board[y][x] = PIECE_IDS[piece]
        self._pieces += 1
        self._top_out_height = max(self._top_out_height, self._stack_height())
        full_rows = [y for y in range(HEIGHT) if all(self._board[y][x] for x in range(WIDTH))]
        lines = len(full_rows)
        if full_rows:
            full = set(full_rows)
            remaining = [row for y, row in enumerate(self._board) if y not in full]
            self._board = remaining + [[0 for _ in range(WIDTH)] for _ in full_rows]
        level_for_score = self._level
        difficult = (lines == 4) or (kind == "t_spin" and lines > 0) or (kind == "t_spin_mini" and lines > 0)
        base = 0
        if kind == "t_spin":
            base = (400, 800, 1200, 1600)[lines] if lines <= 3 else 1600
        elif kind == "t_spin_mini":
            base = (100, 200)[lines] if lines <= 1 else (200 + 400 * (lines - 1))
        elif lines in (1, 2, 3, 4):
            base = (100, 300, 500, 800)[lines - 1]
        line_points = base * level_for_score
        b2b_points = 0
        if difficult and self._back_to_back:
            boosted = (line_points * 3) // 2
            b2b_points = boosted - line_points
            line_points = boosted
        combo_points = 0
        if lines > 0:
            self._combo = 0 if self._combo < 0 else self._combo + 1
            if self._combo > 0:
                combo_points = 50 * self._combo * level_for_score
            if difficult:
                self._back_to_back = True
            else:
                self._back_to_back = False
        else:
            self._combo = -1
        self._line_score += line_points
        self._combo_score += combo_points
        self._b2b_score += b2b_points
        self._score += line_points + combo_points
        # b2b_points is included in the score but kept separate for breakdown.
        self._score += b2b_points
        self._lines += lines
        self._level = max(self._scenario_level(), 1 + self._lines // 10)
        self._last_clear = {"lines": lines, "kind": kind, "score": line_points + combo_points + b2b_points}
        self._events.append({
            "type": "lock", "piece": piece, "cells": [list(c) for c in cells],
            "lines": lines, "clear_kind": kind, "points": line_points + combo_points + b2b_points,
        })
        if lines:
            self._events.append({"type": "line_clear", "lines": lines})
        if above_top:
            self._finish("top_out")
            return
        if self._pieces >= self.max_pieces:
            self._finish("piece_cap")
            return
        self._fill_queue()
        self._spawn_piece(self._queue.pop(0))
        self._fill_queue()

    def _advance_time(self, dt: float) -> None:
        remaining = float(dt)
        guard = 0
        while remaining > 1e-9 and not self._done and guard < 100:
            guard += 1
            if self._active_piece is None:
                return
            if self._is_grounded():
                if self._lock_elapsed is None:
                    self._lock_elapsed = 0.0
                until_lock = max(0.0, LOCK_DELAY_S - self._lock_elapsed)
                if remaining + 1e-12 >= until_lock:
                    remaining -= until_lock
                    self._lock_elapsed = LOCK_DELAY_S
                    self._lock_piece()
                    continue
                self._lock_elapsed += remaining
                return
            self._lock_elapsed = None
            interval = self._gravity_interval()
            until_gravity = max(0.0, interval - self._gravity_accum)
            if remaining + 1e-12 < until_gravity:
                self._gravity_accum += remaining
                return
            remaining -= until_gravity
            self._gravity_accum = 0.0
            if self._can_place(self._active_piece, self._orientation, self._x, self._y - 1):
                self._y -= 1
                self._events.append({"type": "gravity", "y": self._y})
                if self._is_grounded():
                    self._lock_elapsed = 0.0
            else:
                self._lock_elapsed = 0.0

    def _gravity_interval(self) -> float:
        index = max(0, min(len(GRAVITY_S) - 1, self._level - 1))
        return max(self.tick_s, GRAVITY_S[index])

    def _stack_height(self) -> int:
        for y in range(HEIGHT - 1, -1, -1):
            if any(self._board[y]):
                return y + 1
        return 0

    def _column_heights(self) -> list[int]:
        heights = []
        for x in range(WIDTH):
            height = 0
            for y in range(HEIGHT - 1, -1, -1):
                if self._board[y][x]:
                    height = y + 1
                    break
            heights.append(height)
        return heights

    def _well_metrics(self) -> tuple[int, int, int]:
        heights = self._column_heights()
        holes = 0
        for x in range(WIDTH):
            top = heights[x]
            holes += sum(1 for y in range(top) if self._board[y][x] == 0)
        aggregate = sum(heights)
        bumpiness = sum(abs(heights[i] - heights[i - 1]) for i in range(1, WIDTH))
        return holes, aggregate, bumpiness

    def _normalise_action(self, action: Any) -> str | None:
        if isinstance(action, dict):
            action = action.get("action")
        if isinstance(action, str) and action in _ACTIONS:
            return action
        return None

    def tick(self, action: Any) -> None:
        with self._lock:
            if self._closed or self._done:
                return
            self._ticks += 1
            self._elapsed_s = self._ticks * self.tick_s
            if self._ticks >= self.max_ticks:
                self._finish("time_cap")
                return
            parsed = self._normalise_action(action)
            self._last_attempted_action = str(action)[:80]
            if parsed is None:
                self._illegal_actions += 1
                self._events = [{"type": "illegal_action", "action": self._last_attempted_action}]
                parsed = self._last_action if self._last_action in _ACTIONS else "noop"
            else:
                self._events = []
                self._last_action = parsed
            if parsed == "left":
                if self._move(-1, 0):
                    self._events.append({"type": "move", "direction": "left"})
            elif parsed == "right":
                if self._move(1, 0):
                    self._events.append({"type": "move", "direction": "right"})
            elif parsed == "soft_drop":
                if self._move(0, -1):
                    self._drop_score += 1
                    self._score += 1
                    self._events.append({"type": "soft_drop", "cells": 1, "points": 1})
            elif parsed == "rotate_cw":
                if self._rotate(1):
                    self._events.append({"type": "rotate", "direction": "cw"})
            elif parsed == "rotate_ccw":
                if self._rotate(-1):
                    self._events.append({"type": "rotate", "direction": "ccw"})
            elif parsed == "hard_drop":
                self._hard_drop()
            elif parsed == "hold":
                self._hold_piece()
            self._advance_time(self.tick_s)
            if self._ticks >= self.max_ticks and not self._done:
                self._finish("time_cap")

    def observe(self) -> dict[str, Any]:
        with self._lock:
            active = None
            ghost: list[list[int]] = []
            if self._active_piece is not None:
                active = {
                    "type": self._active_piece,
                    "orientation": self._orientation,
                    "x": self._x,
                    "y": self._y,
                    "cells": [list(c) for c in self._cells()],
                }
                gy = self._ghost_y()
                ghost = [list(c) for c in self._cells(y=gy)]
            holes, aggregate, bumpiness = self._well_metrics()
            return {
                "ruleset": "guideline-marathon-v1",
                "engine": "metaharness-tetris-native-v1",
                "board": [list(row) for row in reversed(self._board)],
                "active": active,
                "ghost_cells": ghost,
                "hold": self._hold,
                "can_hold": bool(self._can_hold and not self._done),
                "queue": list(self._queue),
                "score": self._score,
                "score_breakdown": {
                    "line_clear": self._line_score,
                    "drop": self._drop_score,
                    "combo": self._combo_score,
                    "back_to_back": self._b2b_score,
                },
                "lines": self._lines,
                "level": self._level,
                "combo": self._combo,
                "back_to_back": self._back_to_back,
                "pieces": self._pieces,
                "survival_s": round(self._elapsed_s, 6),
                "gravity_s_per_cell": self._gravity_interval(),
                "lock_remaining_s": (
                    None if self._lock_elapsed is None else round(max(0.0, LOCK_DELAY_S - self._lock_elapsed), 6)
                ),
                "lock_resets": self._lock_resets,
                "holes": holes,
                "aggregate_height": aggregate,
                "bumpiness": bumpiness,
                "top_out_height": self._top_out_height,
                "illegal_actions": self._illegal_actions,
                "last_action": self._last_action,
                "last_attempted_action": self._last_attempted_action,
                "last_clear": dict(self._last_clear),
                "events": list(self._events),
                "seed": self.seed,
                "scenario": self.scenario,
                "starting_level": self._scenario_level(),
                "ticks": self._ticks,
                "game_over": bool(self._done),
                "terminal_reason": self._terminal_reason,
            }

    @property
    def done(self) -> bool:
        return bool(self._done)

    @property
    def score(self) -> float:
        return float(self._score)

    @property
    def ticks(self) -> int:
        return int(self._ticks)

    def info(self) -> dict[str, Any]:
        with self._lock:
            holes, aggregate, bumpiness = self._well_metrics()
            return {
                "valid_trial": True,
                "invalid_reason": None,
                "ruleset": "guideline-marathon-v1",
                "seed": self.seed,
                "scenario": self.scenario,
                "terminal_reason": self._terminal_reason,
                "score": float(self._score),
                "lines": self._lines,
                "level": self._level,
                "pieces": self._pieces,
                "survival_s": round(self._elapsed_s, 6),
                "top_out": self._terminal_reason == "top_out",
                "top_out_height": self._top_out_height,
                "holes": holes,
                "aggregate_height": aggregate,
                "bumpiness": bumpiness,
                "illegal_actions": self._illegal_actions,
            }

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            holes, aggregate, bumpiness = self._well_metrics()
            return {
                "score": float(self._score),
                "line_clear_points": float(self._line_score),
                "drop_points": float(self._drop_score),
                "combo_points": float(self._combo_score),
                "back_to_back_bonus": float(self._b2b_score),
                "lines": float(self._lines),
                "level": float(self._level),
                "pieces": float(self._pieces),
                "survival_s": float(self._elapsed_s),
                "holes": float(holes),
                "aggregate_height": float(aggregate),
                "bumpiness": float(bumpiness),
                "top_out_height": float(self._top_out_height),
                "illegal_actions": float(self._illegal_actions),
                "top_out": float(self._terminal_reason == "top_out"),
            }

    def describe(self) -> str:
        return (
            "A local, headless Guideline-compatible 10x20 Tetris marathon. The settled board is exposed "
            "as 20 rows top-to-bottom of integers (0 empty, 1..7 I/J/L/O/S/T/Z), while active and ghost "
            "cells use bottom-origin x,y coordinates. Pieces use deterministic seven-bag randomization seeded "
            "by reset(seed), standard SRS rotations/kicks, hold once per piece, gravity, and a 0.5 second "
            "lock delay. Level starts at 1 for the standard scenario and rises every 10 cleared lines; "
            "the optional medium/high/recovery scenarios are selected when constructing the game.\n\n"
            "The objective is to clear lines while keeping the stack low and avoiding holes/top-out. "
            "Scoring is 100/300/500/800 times level for single/double/triple/Tetris; T-spin is "
            "400/800/1200/1600 and mini T-spin is 100/200 for zero/one line. Soft drop gives one point "
            "per cell and hard drop gives two. Combo adds 50*combo*level after the first consecutive clear; "
            "a consecutive difficult clear (Tetris or line-clearing T-spin) receives a 1.5x back-to-back "
            "bonus. There is no perfect-clear bonus or garbage/multiplayer mode.\n\n"
            "Each tick advances exactly tick_s seconds (0.05 by default), applies the supplied action before "
            "gravity, and increments ticks. The runner should call tick on its own clock while thinking; the "
            "world never waits for a model. Use an action name exactly as listed in action_spec, or {'action': "
            "name}. Malformed actions are ignored and counted in illegal_actions. Observations also contain "
            "hold/queue, timers, score components, lines, level, combo, back-to-back state, stack metrics, "
            "events from the last tick, seed, scenario, terminal reason, and survival duration. Episodes end "
            "at top-out, 200 placed pieces, or 20,000 ticks by default."
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True


def make_game(**kwargs: Any) -> _Tetris:
    """Construct a fresh local game; no credentials or external services are used."""
    return _Tetris(**kwargs)
