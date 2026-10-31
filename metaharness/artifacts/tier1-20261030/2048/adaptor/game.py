"""Offline, deterministic 2048 adaptor.

The implementation follows the original game's line compression/merge order and
90% ``2`` / 10% ``4`` random tile distribution.  It deliberately has no
external process or network dependency.
"""
from __future__ import annotations

import random
from typing import Any

from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = True
SEEDS = {
    "train": [11, 23, 37, 41, 59, 71, 83, 97],
    "heldout": [101, 113, 127, 139, 151, 163, 179, 191],
    "dev": [5, 7],
    "validation": [211, 223, 227, 229, 233, 239, 241, 251],
}
EVAL_KWARGS = {"max_moves": 200}

_ACTIONS = ("up", "down", "left", "right")
_DESCRIPTIONS = {
    "up": "slide all tiles toward the top edge, merging equal neighbors once",
    "down": "slide all tiles toward the bottom edge, merging equal neighbors once",
    "left": "slide all tiles toward the left edge, merging equal neighbors once",
    "right": "slide all tiles toward the right edge, merging equal neighbors once",
}


def _merge_line(line: list[int]) -> tuple[list[int], int]:
    """Compress and merge a line in travel order, returning (line, merge score)."""
    packed = [value for value in line if value]
    merged: list[int] = []
    gained = 0
    i = 0
    while i < len(packed):
        if i + 1 < len(packed) and packed[i] == packed[i + 1]:
            value = packed[i] * 2
            merged.append(value)
            gained += value
            i += 2
        else:
            merged.append(packed[i])
            i += 1
    return merged + [0] * (4 - len(merged)), gained


class Game2048:
    name = "2048"
    action_spec = ActionSpec(
        names=list(_ACTIONS),
        descriptions=_DESCRIPTIONS,
        notes=(
            "Each changed move spawns one tile in an empty cell: 2 with probability 0.90 "
            "or 4 with probability 0.10. A no-op is accepted, consumes no move, and "
            "spawns nothing. The game continues after creating 2048."
        ),
    )

    def __init__(self, max_moves: int = 200) -> None:
        if isinstance(max_moves, bool) or int(max_moves) < 1:
            raise ValueError("max_moves must be a positive integer")
        self.max_moves = int(max_moves)
        self._rng = random.Random()
        self._board: list[list[int]] = [[0] * 4 for _ in range(4)]
        self._score = 0
        self._moves = 0
        self._seed: int | None = None
        self._last_changed = False
        self._last_spawn: dict[str, int] | None = None
        self._reset_done = False

    def reset(self, seed: int) -> dict[str, Any]:
        if isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._board = [[0] * 4 for _ in range(4)]
        self._score = 0
        self._moves = 0
        self._last_changed = False
        self._last_spawn = None
        self._reset_done = True
        self._spawn_tile()
        self._spawn_tile()
        return self.observe()

    def observe(self) -> dict[str, Any]:
        board = [row[:] for row in self._board]
        return {
            "board": board,
            "score": self._score,
            "moves": self._moves,
            "max_tile": max(max(row) for row in board),
            "game_over": self.done,
            "seed": self._seed,
            "changed": self._last_changed,
            "spawn": dict(self._last_spawn) if self._last_spawn is not None else None,
            "legal_actions": self.legal_actions(),
        }

    def _transformed(self, action: str) -> tuple[list[list[int]], int]:
        board = self._board
        candidate = [[0] * 4 for _ in range(4)]
        gained = 0
        if action in ("left", "right"):
            for r in range(4):
                line = board[r][:]
                if action == "right":
                    line.reverse()
                out, amount = _merge_line(line)
                if action == "right":
                    out.reverse()
                candidate[r] = out
                gained += amount
        else:
            for c in range(4):
                line = [board[r][c] for r in range(4)]
                if action == "down":
                    line.reverse()
                out, amount = _merge_line(line)
                if action == "down":
                    out.reverse()
                for r, value in enumerate(out):
                    candidate[r][c] = value
                gained += amount
        return candidate, gained

    def _spawn_tile(self) -> dict[str, int] | None:
        empties = [(r, c) for r in range(4) for c in range(4) if self._board[r][c] == 0]
        if not empties:
            return None
        r, c = self._rng.choice(empties)
        value = 4 if self._rng.random() < 0.1 else 2
        self._board[r][c] = value
        return {"row": r, "col": c, "value": value}

    def step(self, action: Any) -> dict[str, Any]:
        self._last_changed = False
        self._last_spawn = None
        if not isinstance(action, str) or action not in _ACTIONS:
            # Malformed actions are harmless and visible to the harness through the
            # unchanged observation; they do not terminate an episode.
            return self.observe()
        if self.done:
            return self.observe()
        candidate, gained = self._transformed(action)
        if candidate == self._board:
            return self.observe()
        self._board = candidate
        self._score += gained
        self._moves += 1
        self._last_changed = True
        self._last_spawn = self._spawn_tile()
        return self.observe()

    def legal_actions(self) -> list[str]:
        if not self._reset_done or self.done:
            return []
        return [action for action in _ACTIONS if self._transformed(action)[0] != self._board]

    @property
    def done(self) -> bool:
        if not self._reset_done:
            return False
        return self._moves >= self.max_moves or not any(self._transformed(a)[0] != self._board for a in _ACTIONS)

    @property
    def score(self) -> float:
        # The evaluator gets one scalar. Include progress toward the objective while
        # retaining the exact merge score in score_breakdown() and observations.
        return float(self._score + max(max(row) for row in self._board))

    def score_breakdown(self) -> dict[str, float]:
        max_tile = max(max(row) for row in self._board)
        return {
            "merge_score": float(self._score),
            "max_tile": float(max_tile),
            "combined_score": float(self._score + max_tile),
        }

    def info(self) -> dict[str, Any]:
        return {
            "valid_trial": True,
            "seed": self._seed,
            "max_moves": self.max_moves,
            "score_breakdown": self.score_breakdown(),
        }

    def describe(self) -> str:
        return (
            "2048 is a deterministic seeded 4x4 turn-based puzzle. The board starts with "
            "two tiles (2 or 4). On each action, all tiles slide in that direction; equal "
            "adjacent tiles merge once, and the resulting value is added to the merge score. "
            "A changed move then places one random tile in an empty cell (90% 2, 10% 4). "
            "No-op and malformed actions leave state unchanged. The objective is to maximize "
            "merge score and the largest tile; reaching 2048 does not end play. "
            "Observations contain board (four rows of four integers), score, moves, max_tile, "
            "game_over, seed, changed, spawn, and legal_actions. The scalar evaluator score "
            "is merge score plus max_tile, with both components available in score_breakdown()."
        )

    def close(self) -> None:
        # No process, file, socket, or other external resource is owned.
        self._rng = random.Random(0)


def make_game(**kwargs: Any) -> Game2048:
    return Game2048(**kwargs)

