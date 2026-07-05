"""Go rules engine — just enough to replay records and encode positions.

Implements: stone placement with capture removal (flood-fill liberties),
suicide rejection, and SIMPLE-ko tracking (the single point forbidden for
the move immediately after a 1-stone ko capture).

Deliberately NOT implemented (documented limitation):
  - superko / positional-superko: real server records occasionally contain
    long cycles (triple ko etc.) that simple ko allows; replaying them here
    succeeds, which is what a converter wants;
  - scoring / life-and-death: game results come from the SGF ``RE`` tag;
  - alternation enforcement: ``play`` takes an explicit color, so handicap
    games (White moves first) and dirty records replay as written.

Everything is deterministic and pure python + numpy int8.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Set, Tuple

import numpy as np

Point = Tuple[int, int]  # (row, col)

EMPTY: int = 0
BLACK: int = 1
WHITE: int = 2


def opponent(color: int) -> int:
    if color == BLACK:
        return WHITE
    if color == WHITE:
        return BLACK
    raise ValueError(f"not a stone color: {color!r}")


class IllegalMove(ValueError):
    """Occupied point, suicide, ko violation, or out-of-bounds play."""


class GoBoard:
    """Mutable Go position: int8 grid + the simple-ko forbidden point.

    ``ko_point`` is the point forbidden for the NEXT move only — every
    ``play`` (including a pass) recomputes or clears it. It is set exactly
    when a move captures a single stone and the capturing stone ends up as
    a lone stone whose only liberty is the captured point (the textbook
    ko shape).
    """

    def __init__(self, size: int):
        if size < 2:
            raise ValueError(f"board size {size} too small")
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.ko_point: Optional[Point] = None

    # ------------------------------------------------------------------

    def _neighbors(self, r: int, c: int) -> Iterator[Point]:
        if r > 0:
            yield r - 1, c
        if r < self.size - 1:
            yield r + 1, c
        if c > 0:
            yield r, c - 1
        if c < self.size - 1:
            yield r, c + 1

    def _group(self, r: int, c: int) -> Tuple[Set[Point], Set[Point]]:
        """Flood-fill the group containing (r, c) → (stones, liberties)."""
        color = self.grid[r, c]
        if color == EMPTY:
            raise ValueError(f"({r}, {c}) is empty")
        stones: Set[Point] = set()
        liberties: Set[Point] = set()
        stack = [(r, c)]
        while stack:
            p = stack.pop()
            if p in stones:
                continue
            stones.add(p)
            for q in self._neighbors(*p):
                v = self.grid[q]
                if v == EMPTY:
                    liberties.add(q)
                elif v == color and q not in stones:
                    stack.append(q)
        return stones, liberties

    # ------------------------------------------------------------------

    def setup(self, color: int, points) -> None:
        """Place setup stones (AB/AW handicap) — no captures, no ko."""
        for r, c in points:
            if not (0 <= r < self.size and 0 <= c < self.size):
                raise IllegalMove(f"setup stone ({r}, {c}) off the board")
            if self.grid[r, c] != EMPTY:
                raise IllegalMove(f"setup stone on occupied point ({r}, {c})")
            self.grid[r, c] = color

    def play(self, color: int, point: Optional[Point]) -> List[Point]:
        """Play ``color`` at ``point`` (None = pass). Returns captured points.

        Raises :class:`IllegalMove` on occupied point, ko violation,
        suicide, or out-of-bounds — the board is left unchanged in every
        failure case.
        """
        if color not in (BLACK, WHITE):
            raise IllegalMove(f"not a stone color: {color!r}")
        if point is None:                       # pass
            self.ko_point = None
            return []
        r, c = point
        if not (0 <= r < self.size and 0 <= c < self.size):
            raise IllegalMove(f"({r}, {c}) is off the {self.size}x{self.size} board")
        if self.grid[r, c] != EMPTY:
            raise IllegalMove(f"({r}, {c}) is occupied")
        if self.ko_point == (r, c):
            raise IllegalMove(f"({r}, {c}) is the ko point")

        self.grid[r, c] = color
        opp = opponent(color)

        # Capture any adjacent opponent group left without liberties.
        captured: List[Point] = []
        seen: Set[Point] = set()
        for q in self._neighbors(r, c):
            if self.grid[q] == opp and q not in seen:
                stones, liberties = self._group(*q)
                seen |= stones
                if not liberties:
                    for s in stones:
                        self.grid[s] = EMPTY
                    captured.extend(stones)

        # Suicide check (only reachable when nothing was captured).
        own_stones, own_liberties = self._group(r, c)
        if not own_liberties:
            self.grid[r, c] = EMPTY
            raise IllegalMove(f"suicide at ({r}, {c})")

        # Simple ko: 1 stone captured, capturer is a lone stone whose only
        # liberty is the vacated point.
        if (
            len(captured) == 1
            and len(own_stones) == 1
            and own_liberties == {captured[0]}
        ):
            self.ko_point = captured[0]
        else:
            self.ko_point = None
        return captured
