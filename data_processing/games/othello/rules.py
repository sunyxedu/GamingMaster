"""Exact Othello (Reversi) rules on an 8×8 board.

Board representation: a flat ``bytearray`` of 64 squares indexed
``row * 8 + col`` (row 0 = rank 1, col 0 = file a), values EMPTY /
BLACK / WHITE.  Black always moves first.

Rules implemented exactly:
  - a move must be on an empty square and flip ≥1 opponent disc;
  - flipping scans all 8 directions: a contiguous run of opponent
    discs terminated by an own disc is flipped;
  - a player with no legal move must pass;
  - the game ends when neither player has a legal move (which covers
    both the board-full and the wipeout case — with zero discs of one
    color nothing can be flanked, so neither side can move);
  - the final result is the disc count.
"""

from __future__ import annotations

from typing import List, Tuple

BOARD_SIZE = 8
N_SQUARES = BOARD_SIZE * BOARD_SIZE

EMPTY, BLACK, WHITE = 0, 1, 2

DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


class IllegalMoveError(ValueError):
    """Raised by :func:`apply_move` for a move that flips nothing."""


def opponent(player: int) -> int:
    return BLACK + WHITE - player


def initial_board() -> bytearray:
    """Standard start: white d4/e5, black e4/d5 (0-indexed: white (3,3)
    and (4,4), black (3,4) and (4,3)); black to move."""
    board = bytearray(N_SQUARES)
    board[3 * BOARD_SIZE + 3] = WHITE
    board[4 * BOARD_SIZE + 4] = WHITE
    board[3 * BOARD_SIZE + 4] = BLACK
    board[4 * BOARD_SIZE + 3] = BLACK
    return board


def flips(board: bytearray, player: int, row: int, col: int) -> List[int]:
    """Flat indices of every disc flipped if ``player`` plays (row, col).

    Empty list ⇔ the move is illegal (occupied square or nothing flipped).
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return []
    if board[row * BOARD_SIZE + col] != EMPTY:
        return []
    opp = opponent(player)
    flipped: List[int] = []
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        run: List[int] = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            idx = r * BOARD_SIZE + c
            if board[idx] == opp:
                run.append(idx)
            elif board[idx] == player:
                flipped.extend(run)
                break
            else:  # EMPTY — unterminated run
                break
            r += dr
            c += dc
    return flipped


def legal_moves(board: bytearray, player: int) -> List[Tuple[int, int]]:
    return [
        (r, c)
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
        if flips(board, player, r, c)
    ]


def has_legal_move(board: bytearray, player: int) -> bool:
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if flips(board, player, r, c):
                return True
    return False


def apply_move(board: bytearray, player: int, row: int, col: int) -> None:
    """Place ``player``'s disc on (row, col) and flip, in place."""
    flipped = flips(board, player, row, col)
    if not flipped:
        raise IllegalMoveError(
            f"player {player} cannot play ({row}, {col})"
        )
    board[row * BOARD_SIZE + col] = player
    for idx in flipped:
        board[idx] = player


def is_game_over(board: bytearray) -> bool:
    """Neither player has a legal move (double pass / full / wipeout)."""
    return not (has_legal_move(board, BLACK) or has_legal_move(board, WHITE))


def disc_counts(board: bytearray) -> Tuple[int, int]:
    """(black discs, white discs)."""
    black = 0
    white = 0
    for v in board:
        if v == BLACK:
            black += 1
        elif v == WHITE:
            white += 1
    return black, white


def board_planes(board: bytearray, mover: int):
    """[3, 8, 8] float32 from ``mover``'s perspective (own / opp / empty) — the
    observation layout of the othello adapter (and of `gmdata othello convert`)."""
    import torch
    grid = torch.tensor(list(board), dtype=torch.uint8).reshape(BOARD_SIZE, BOARD_SIZE)
    return torch.stack([grid == mover, grid == opponent(mover), grid == EMPTY]).float()
