"""Exact gomoku rules: freestyle (win = five OR MORE in a row) and
standard (win = EXACTLY five; an overline of 6+ is not a win).

Board is a flat bytearray of ``size*size`` cells, row-major.  Freestyle
is the main Gomocup convention — no renju forbidden points, overlines
win; ``wins(..., exact_five=True)`` selects the standard rule instead.
Legality is simply "the intersection is empty"; there is no pass (the
gomoku adapter is built with ``allow_pass=False``).
"""

from __future__ import annotations

EMPTY, BLACK, WHITE = 0, 1, 2

# (drow, dcol) for the four line orientations
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


class IllegalMove(ValueError):
    """Move onto an occupied cell or off the board."""


def opponent(player: int) -> int:
    return BLACK + WHITE - player


def initial_board(size: int) -> bytearray:
    return bytearray(size * size)


def apply_move(board: bytearray, size: int, player: int, row: int, col: int) -> None:
    """Place a stone; raises :class:`IllegalMove` if off-board/occupied."""
    if not (0 <= row < size and 0 <= col < size):
        raise IllegalMove(f"({row}, {col}) off a {size}x{size} board")
    idx = row * size + col
    if board[idx] != EMPTY:
        raise IllegalMove(f"({row}, {col}) already occupied")
    board[idx] = player


def wins(board: bytearray, size: int, player: int, row: int, col: int,
         exact_five: bool = False) -> bool:
    """True if ``player``'s stone at (row, col) completes a winning run:
    ≥5 (freestyle, default) or exactly 5 (``exact_five``, standard rule —
    an overline is not a win)."""
    for dr, dc in _DIRS:
        run = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < size and 0 <= c < size \
                    and board[r * size + c] == player:
                run += 1
                r += sign * dr
                c += sign * dc
        if run == 5 or (run > 5 and not exact_five):
            return True
    return False


def board_full(board: bytearray) -> bool:
    return EMPTY not in board
