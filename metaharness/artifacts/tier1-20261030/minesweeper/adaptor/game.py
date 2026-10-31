"""Deterministic, headless Minesweeper TurnGame adaptor.

The implementation is deliberately self contained: the benchmark talks to the
same state a local JSON client would see, without a GUI or a network service.
"""
from __future__ import annotations

import random
from typing import Any

from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = True

# Each split contains a mixture of the same six board families.  The seed is
# the only scenario selector, so held-out boards use the same generator.
SEEDS = {
    "train": list(range(600, 612)),
    "heldout": list(range(1201, 1213)),
    "dev": [2403, 2404],
    "validation": list(range(1802, 1814)),
}
EVAL_KWARGS: dict[str, Any] = {}

# width, height, mines.  The first three are the classic beginner,
# intermediate and expert presets; the others exercise different densities.
_SCENARIOS = (
    (9, 9, 10),
    (16, 16, 40),
    (30, 16, 99),
    (12, 10, 18),
    (20, 12, 45),
    (10, 10, 15),
)


class MinesweeperGame:
    name = "Minesweeper (offline deterministic)"
    action_spec = ActionSpec(
        names=("reveal", "flag", "chord", "noop"),
        descriptions={
            "reveal": "Reveal a cell: use the string 'reveal ROW COL' (zero-based).",
            "flag": "Toggle a flag on a hidden cell: use 'flag ROW COL'.",
            "chord": "On a revealed number, reveal neighbours when adjacent flags equal it: 'chord ROW COL'.",
            "noop": "Do nothing for one turn.",
        },
        noop="noop",
        notes=(
            "Actions may also be JSON objects {'type':'reveal'|'flag'|'chord', 'row': ROW, 'col': COL}. "
            "Rows and columns are zero-based. Invalid or illegal actions are ignored and reported in the "
            "next observation. A first reveal is always safe; zero cells flood-fill their neighbours."
        ),
    )

    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        mines: int | None = None,
        first_click_safe: bool = True,
    ) -> None:
        supplied = (width is not None, height is not None, mines is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("width, height, and mines must be supplied together")
        if all(supplied):
            self._fixed = self._validate_dimensions(width, height, mines)
        else:
            self._fixed = None
        if type(first_click_safe) is not bool:
            raise ValueError("first_click_safe must be a bool")
        self.first_click_safe = first_click_safe
        self._closed = False
        self.reset(0)

    @staticmethod
    def _validate_dimensions(width: int, height: int, mines: int) -> tuple[int, int, int]:
        if any(type(x) is not int for x in (width, height, mines)):
            raise ValueError("width, height, and mines must be integers")
        if width < 3 or height < 3 or mines < 1 or mines >= width * height:
            raise ValueError("need width,height >= 3 and 1 <= mines < width*height")
        return int(width), int(height), int(mines)

    def reset(self, seed: int) -> dict[str, Any]:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._seed = int(seed)
        if self._fixed is None:
            self.width, self.height, self.mines = _SCENARIOS[self._scenario_index(seed)]
        else:
            self.width, self.height, self.mines = self._fixed
        self._status = "playing"
        self._mine_cells: set[tuple[int, int]] | None = None
        self._counts: list[list[int]] | None = None
        self._revealed: set[tuple[int, int]] = set()
        self._flags: set[tuple[int, int]] = set()
        self._safe_revealed = 0
        self._last_error: str | None = None
        self._last_action: str = "reset"
        self._turn = 0
        self._closed = False
        return self.observe()

    @staticmethod
    def _scenario_index(seed: int) -> int:
        # The declared groups are contiguous blocks with identical modulo-
        # six composition (two examples of each family per 12-seed split).
        return seed % len(_SCENARIOS)

    def _ensure_board(self, first: tuple[int, int]) -> None:
        if self._mine_cells is not None:
            return
        candidates = [(r, c) for r in range(self.height) for c in range(self.width)]
        if self.first_click_safe:
            candidates.remove(first)
        rng = random.Random(self._seed)
        self._mine_cells = set(rng.sample(candidates, self.mines))
        self._counts = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for mr, mc in self._mine_cells:
            for nr in range(max(0, mr - 1), min(self.height, mr + 2)):
                for nc in range(max(0, mc - 1), min(self.width, mc + 2)):
                    if (nr, nc) != (mr, mc):
                        self._counts[nr][nc] += 1

    def _neighbours(self, row: int, col: int):
        for nr in range(max(0, row - 1), min(self.height, row + 2)):
            for nc in range(max(0, col - 1), min(self.width, col + 2)):
                if (nr, nc) != (row, col):
                    yield nr, nc

    def _in_bounds(self, row: Any, col: Any) -> bool:
        return type(row) is int and type(col) is int and 0 <= row < self.height and 0 <= col < self.width

    def _reveal(self, row: int, col: int) -> None:
        if (row, col) in self._flags or (row, col) in self._revealed:
            return
        self._ensure_board((row, col))
        assert self._mine_cells is not None and self._counts is not None
        if (row, col) in self._mine_cells:
            self._status = "lost"
            return
        queue = [(row, col)]
        queued: set[tuple[int, int]] = set(queue)
        while queue:
            cell = queue.pop()
            if cell in self._revealed or cell in self._flags or cell in self._mine_cells:
                continue
            self._revealed.add(cell)
            self._safe_revealed += 1
            r, c = cell
            if self._counts[r][c] == 0:
                for neighbour in self._neighbours(r, c):
                    if neighbour not in queued and neighbour not in self._revealed:
                        queued.add(neighbour)
                        queue.append(neighbour)
        if self._safe_revealed >= self.width * self.height - self.mines:
            self._status = "won"

    def _parse_action(self, action: Any) -> tuple[str, Any, Any, str | None]:
        if isinstance(action, dict):
            kind = action.get("type")
            row, col = action.get("row"), action.get("col")
            return (kind, row, col, None) if isinstance(kind, str) else ("", None, None, "action type must be a string")
        if isinstance(action, str):
            parts = action.strip().split()
            if not parts:
                return "", None, None, "empty action"
            kind = parts[0].lower()
            if kind == "noop" and len(parts) == 1:
                return kind, None, None, None
            if len(parts) != 3:
                return kind, None, None, "expected 'reveal ROW COL', 'flag ROW COL', or 'chord ROW COL'"
            try:
                return kind, int(parts[1]), int(parts[2]), None
            except ValueError:
                return kind, None, None, "row and col must be integers"
        return "", None, None, "action must be a string or object"

    def step(self, action: Any) -> dict[str, Any]:
        self._last_error = None
        self._turn += 1
        kind, row, col, error = self._parse_action(action)
        self._last_action = kind or "invalid"
        if error:
            self._last_error = error
            return self.observe()
        if self._status != "playing":
            self._last_error = "game is already terminal"
            return self.observe()
        if kind == "noop":
            return self.observe()
        if kind not in {"reveal", "flag", "chord"} or not self._in_bounds(row, col):
            self._last_error = "unknown action or out-of-bounds cell"
            return self.observe()
        cell = (row, col)
        if kind == "reveal":
            self._reveal(row, col)
        elif kind == "flag":
            if cell not in self._revealed:
                if cell in self._flags:
                    self._flags.remove(cell)
                else:
                    self._flags.add(cell)
            else:
                self._last_error = "cannot flag a revealed cell"
        else:  # chord
            if self._counts is None or cell not in self._revealed:
                self._last_error = "chord requires a revealed numbered cell"
            else:
                adjacent = list(self._neighbours(row, col))
                if sum(n in self._flags for n in adjacent) != self._counts[row][col]:
                    self._last_error = "adjacent flag count does not match the number"
                else:
                    for nr, nc in adjacent:
                        if (nr, nc) not in self._flags and (nr, nc) not in self._revealed:
                            self._reveal(nr, nc)
                            if self._status == "lost":
                                break
        return self.observe()

    def observe(self) -> dict[str, Any]:
        terminal = self._status in {"lost", "won"}
        grid: list[list[str]] = []
        for r in range(self.height):
            line: list[str] = []
            for c in range(self.width):
                cell = (r, c)
                if terminal and self._mine_cells is not None and cell in self._mine_cells:
                    line.append("mine")
                elif cell in self._revealed and self._counts is not None:
                    line.append(f"revealed:{self._counts[r][c]}")
                elif cell in self._flags:
                    line.append("flagged")
                else:
                    line.append("unknown")
            grid.append(line)
        remaining = self.mines - len(self._flags)
        return {
            "width": self.width,
            "height": self.height,
            "mines": self.mines,
            "status": self._status,
            "cells": grid,
            "safe_revealed": self._safe_revealed,
            "safe_total": self.width * self.height - self.mines,
            "flags": len(self._flags),
            "remaining_mines": remaining,
            "turn": self._turn,
            "last_action": self._last_action,
            "error": self._last_error,
            "won": self._status == "won",
            "terminal_reason": self._status if terminal else None,
        }

    @property
    def done(self) -> bool:
        return self._status != "playing"

    @property
    def score(self) -> float:
        return self._safe_revealed / max(1, self.width * self.height - self.mines)

    def score_breakdown(self) -> dict[str, float]:
        return {
            "progress": float(self.score),
            "safe_revealed": float(self._safe_revealed),
            "safe_total": float(self.width * self.height - self.mines),
            "won": 1.0 if self._status == "won" else 0.0,
        }

    def info(self) -> dict[str, Any]:
        return {
            "valid_trial": True,
            "seed": self._seed,
            "status": self._status,
            "scenario": {"width": self.width, "height": self.height, "mines": self.mines},
            "turns": self._turn,
            "won": self._status == "won",
        }

    def describe(self) -> str:
        return (
            "Minesweeper is a finite turn-based puzzle. Reveal every safe cell to win; revealing a mine loses. "
            "The board uses eight-neighbour mine counts, and a zero count automatically reveals its connected "
            "zero region and its border. The first reveal is guaranteed safe. Each turn accepts reveal/flag/chord "
            "or noop. Use zero-based coordinates in strings such as 'reveal 3 4', or objects with type,row,col. "
            "Flag toggles a hidden cell and is only an annotation. Chord on a revealed number opens adjacent hidden "
            "cells when the number of adjacent flags exactly matches that number. Observations expose cells, status, "
            "safe_revealed, safe_total, flags, and remaining_mines; mine locations stay hidden until terminal. "
            "The score is safe_revealed / safe_total, with a separate won indicator in info and score_breakdown."
        )

    def close(self) -> None:
        self._closed = True


def make_game(**kwargs: Any) -> MinesweeperGame:
    """Construct a fresh local game; optional width/height/mines fix a custom board family."""
    return MinesweeperGame(**kwargs)
