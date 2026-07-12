"""Convert Gomocup ``.psq`` records into training shards for the
"gomoku" adapter (BoardEncoder + PlacementDecoder, 3 planes,
``allow_pass=False``; board side = ``--board_size``, default 15 — the
main Gomocup freestyle league plays 20×20).

    uv run python -m data_processing.games.gomoku.convert \
        --input data/gomoku/gomocup/ --output_dir data/gomoku/train \
        --board_size 20 --shard_size 512

One sample = one full game seen from the SIDE-TO-MOVE perspective at
every frame; frame t is the position BEFORE the t-th half-move.
Perspective is a channel swap only — no geometric flip (same convention
as the othello converter).

    obs["board_seq"]  [T, 3, N, N] float32 — plane 0 mover's stones,
                      plane 1 opponent's stones, plane 2 empty
    tgt["place_idx"]  [T] long   row*N+col
    tgt["passed"]     [T] bool   all False (gomoku has no pass)
    tgt["value"]      [T] float  ±1 from the frame mover's perspective

The board size used is written to shard-level ``meta`` so the trainer
can validate/auto-configure the model's board_h/board_w.  ``--rule``
selects the win condition: freestyle (five or more) or standard
(exactly five; overlines are not wins).

``.psq`` parsing is defensive: the header line gives the board size,
subsequent lines matching ``x,y[,time]`` (1-based, x = column) are
moves, and everything after the first non-move line is ignored
(engine names / metadata trailers vary by Gomocup year).

Outcome: only games whose LAST recorded move completes five-or-more in
a row get a decisive label (winner = last mover).  Games that simply
stop (resignation, timeout, adjudication — the .psq itself carries no
reliable result) are skipped by default; ``--keep_undecided`` labels
them winner-=-last-mover, the usual resignation convention, at the cost
of mislabeling timeouts.  A stone played onto an occupied cell, moves
continuing after a completed five, or a wrong board size skip the game.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from data_processing.games.dataset import save_shard
from data_processing.games.gomoku.rules import (
    BLACK,
    EMPTY,
    IllegalMove,
    apply_move,
    initial_board,
    opponent,
    wins,
)

BOARD_SIZE = 15  # matches the registry's gomoku preset (board_h = board_w = 15)

_MOVE_RE = re.compile(r"^\s*(\d+),(\d+)(?:,(-?\d+))?\s*$")
_HEADER_RE = re.compile(r"(\d+)x(\d+)", re.ASCII)


class SkipGame(ValueError):
    """Record unusable: illegal move, wrong size, undecided, ..."""


def parse_psq(text: str) -> Tuple[int, int, List[Tuple[int, int]]]:
    """One .psq file → (width, height, moves as 0-based (row, col)).

    Move lines are ``x,y[,time]`` with 1-based x = column, y = row.
    Parsing stops at the first non-move line after moves began (trailing
    engine names / metadata).
    """
    lines = text.splitlines()
    if not lines:
        raise SkipGame("empty file")
    m = _HEADER_RE.search(lines[0])
    if not m:
        raise SkipGame(f"unrecognized header {lines[0]!r}")
    w, h = int(m.group(1)), int(m.group(2))

    moves: List[Tuple[int, int]] = []
    for line in lines[1:]:
        mm = _MOVE_RE.match(line)
        if mm is None:
            if moves:
                break           # metadata trailer
            continue            # tolerate blank/comment lines before moves
        x, y = int(mm.group(1)), int(mm.group(2))
        moves.append((y - 1, x - 1))
    return w, h, moves


def board_planes(board: bytearray, size: int, mover: int) -> torch.Tensor:
    """[3, size, size] float32 from ``mover``'s perspective (own/opp/empty)."""
    grid = torch.tensor(list(board), dtype=torch.uint8).reshape(size, size)
    return torch.stack(
        [grid == mover, grid == opponent(mover), grid == EMPTY]
    ).float()


def game_to_sample(
    moves: Sequence[Tuple[int, int]],
    *,
    board_size: int = BOARD_SIZE,
    exact_five: bool = False,
    keep_undecided: bool = False,
    min_moves: int = 9,
    meta: Optional[Dict] = None,
) -> Dict:
    """Replay one game → shard sample.  Raises :class:`SkipGame`."""
    if len(moves) < max(min_moves, 1):
        raise SkipGame("too_short")

    board = initial_board(board_size)
    player = BLACK
    frames: List[torch.Tensor] = []
    movers: List[int] = []
    place_idx: List[int] = []
    winner: Optional[int] = None

    for n, (row, col) in enumerate(moves):
        if winner is not None:
            raise SkipGame(f"moves_after_win (move {n})")
        frames.append(board_planes(board, board_size, player))
        movers.append(player)
        try:
            apply_move(board, board_size, player, row, col)
        except IllegalMove as e:
            raise SkipGame(f"illegal_move ({e})") from None
        place_idx.append(row * board_size + col)
        if wins(board, board_size, player, row, col, exact_five=exact_five):
            winner = player
        player = opponent(player)

    if winner is None:
        if not keep_undecided:
            raise SkipGame("undecided")
        winner = movers[-1]     # resignation convention: last mover won

    T = len(frames)
    win_sign = [1.0 if m == winner else -1.0 for m in movers]
    return {
        "obs": {"board_seq": torch.stack(frames)},                # [T,3,H,W]
        "tgt": {
            "place_idx": torch.tensor(place_idx, dtype=torch.long),
            "passed": torch.zeros(T, dtype=torch.bool),
            "value": torch.tensor(win_sign, dtype=torch.float32),
        },
        "meta": {**(meta or {}), "plies": T,
                 "winner": "black" if winner == BLACK else "white"},
    }


def convert(
    input_dir: str,
    output_dir: str,
    shard_size: int = 512,
    max_games: Optional[int] = None,
    keep_undecided: bool = False,
    min_moves: int = 9,
    board_size: int = BOARD_SIZE,
    rule: str = "freestyle",
) -> Dict[str, int]:
    """Convert every ``*.psq`` under ``input_dir`` (recursively).

    Returns counters: seen / kept / shards plus a skip-reason histogram.
    """
    if not 5 <= board_size <= 25:
        raise ValueError(f"board_size must be within [5, 25], got {board_size}")
    if rule not in ("freestyle", "standard"):
        raise ValueError(f"unknown rule {rule!r}")
    exact_five = rule == "standard"
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.psq"),
                             recursive=True))
    samples: List[Dict] = []
    stats: Dict[str, int] = {"seen": 0, "kept": 0, "shards": 0}
    shard_i = 0

    def flush():
        nonlocal samples, shard_i
        if samples:
            save_shard(
                os.path.join(output_dir, f"gomoku_{shard_i:05d}.pt"),
                game="gomoku", samples=samples,
                meta={"board_size": board_size},
            )
            shard_i += 1
            stats["shards"] = shard_i
            samples = []

    for path in paths:
        stats["seen"] += 1
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                w, h, moves = parse_psq(f.read())
            if (w, h) != (board_size, board_size):
                raise SkipGame("wrong_board_size")
            samples.append(game_to_sample(
                moves, board_size=board_size, exact_five=exact_five,
                keep_undecided=keep_undecided, min_moves=min_moves,
                meta={"source": os.path.basename(path)},
            ))
        except SkipGame as e:
            reason = str(e).split(" ")[0]
            stats[reason] = stats.get(reason, 0) + 1
            continue
        stats["kept"] += 1
        if len(samples) >= shard_size:
            flush()
        if max_games is not None and stats["kept"] >= max_games:
            break
    flush()
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="directory with .psq files")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_size", type=int, default=512)
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--min_moves", type=int, default=9)
    p.add_argument("--board_size", type=int, default=BOARD_SIZE,
                   help="board side length (5..25); .psq files with any "
                        "other header size are skipped.  Gomocup's main "
                        "freestyle league plays 20x20 (default: 15)")
    p.add_argument("--rule", choices=("freestyle", "standard"),
                   default="freestyle",
                   help="win condition: freestyle = five or more in a row "
                        "(Gomocup default), standard = exactly five (an "
                        "overline of 6+ is not a win).  Caro (blocked-ends) "
                        "rules are NOT supported; do not convert caro "
                        "archives with this tool")
    p.add_argument("--keep_undecided", action="store_true",
                   help="label games without a completed five as won by "
                        "the last mover (resignation convention)")
    args = p.parse_args()
    stats = convert(
        args.input, args.output_dir, shard_size=args.shard_size,
        max_games=args.max_games, keep_undecided=args.keep_undecided,
        min_moves=args.min_moves, board_size=args.board_size,
        rule=args.rule,
    )
    print(f"[gomoku] {stats}")


if __name__ == "__main__":
    main()
