"""SGF → shard converter for the ``go`` adapter (BoardEncoder + PlacementDecoder).

    uv run python -m data_processing.games.go.convert \\
        --input sgf_dir_or_file --output_dir shards/ \\
        --shard_size 256 --board_size 19 --max_games 100000 --min_moves 30

Frame convention:
    Frame t is the position BEFORE move t, encoded from the perspective of
    the side to move at t; the label is that move. Every move is replayed
    through :mod:`data_processing.games.go.rules`, so captures are applied
    and a record containing an illegal move (dirty SGFs exist) skips the
    whole game with a counted warning.

Observation — ``obs["board_seq"]`` [T, 4, H, W] float32, side-to-move POV:
    plane 0  own stones
    plane 1  opponent stones
    plane 2  empty points          (== 1 − plane0 − plane1)
    plane 3  simple-ko forbidden point (one-hot, or all zeros)
    No globals: ``board_global_features=0`` for the go preset.

    POV note: Go needs NO geometric flip — the board is symmetric and both
    players place stones under identical rules, so swapping the own/opp
    channels IS the full perspective change (unlike e.g. generals, where
    the mover's spatial frame matters).

Targets:
    place_idx [T] long   row * W + col (0 on pass frames — masked out of
                         the placement CE by ``passed`` in il_loss)
    passed    [T] bool
    value     [T] float  final result from the frame's mover perspective:
                         +1 win / −1 loss from the RE tag (komi is already
                         folded into RE), 0 for jigo/draw.

Skipping policy (each with its own counter):
    wrong board size; mid-game AB/AW/AE setup (always — pre-placing it
    would corrupt earlier frames); handicap games (HA>0 or AB/AW setup)
    unless --allow_handicap (root setup stones are then pre-placed; HA>0
    needs len(AB)==HA, or no AB with the placements carried as the first
    HA Black moves); unknown or unfinished RE unless --allow_no_result
    (value 0); games shorter than --min_moves; games with an
    illegal/unreplayable move.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data_processing.games.dataset import save_shard
from data_processing.games.go.rules import (
    BLACK,
    EMPTY,
    WHITE,
    GoBoard,
    IllegalMove,
    opponent,
)
from data_processing.games.go.sgf import SGFError, SGFGame, parse_sgf

CHANNELS = 4  # own / opp / empty / ko — must match the go preset's piece_channels

SKIP_REASONS = (
    "wrong_board_size",
    "handicap",
    "no_result",
    "too_short",
    "illegal_move",
)


def parse_result(re_tag: Optional[str]) -> Optional[int]:
    """RE tag → +1 (Black wins) / −1 (White wins) / 0 (draw) / None (unknown).

    Komi is already folded into RE ("B+2.5" means Black leads after komi).
    "B+?"-style unknown margins still count as a Black win; "Void", "?"
    and missing tags are unknown.
    """
    if not re_tag:
        return None
    s = re_tag.strip().upper()
    if s.startswith("B+") and len(s) > 2:
        return +1
    if s.startswith("W+") and len(s) > 2:
        return -1
    if s in ("0", "DRAW", "JIGO"):
        return 0
    return None


def encode_planes(board: GoBoard, color: int) -> np.ndarray:
    """Current position → [4, H, W] float32 from ``color``'s perspective."""
    planes = np.zeros((CHANNELS, board.size, board.size), dtype=np.float32)
    planes[0] = board.grid == color
    planes[1] = board.grid == opponent(color)
    planes[2] = board.grid == EMPTY
    if board.ko_point is not None:
        planes[3][board.ko_point] = 1.0
    return planes


def game_to_sample(
    game: SGFGame,
    board_size: int,
    *,
    allow_handicap: bool = False,
    allow_no_result: bool = False,
    min_moves: int = 0,
    source: str = "",
) -> Tuple[Optional[Dict], str]:
    """One parsed SGF game → shard sample, or (None, skip_reason)."""
    try:
        if game.board_size != board_size:
            return None, "wrong_board_size"
    except SGFError:
        return None, "wrong_board_size"

    # Mid-game AB/AW/AE setup would be applied before move 0 (the flat
    # setup lists lose the ordering), corrupting every frame before the
    # setup node — unreplayable regardless of --allow_handicap.
    if game.setup_after_move:
        return None, "handicap"
    if (game.handicap > 0 or game.setup_black or game.setup_white) \
            and not allow_handicap:
        return None, "handicap"
    if allow_handicap and game.handicap > 0 \
            and len(game.setup_black) != game.handicap:
        # No AB at all can still be complete: free-placement servers record
        # the handicap stones as the first HA consecutive Black moves, and
        # those replay correctly as-is (go rules don't enforce alternation).
        # Any other mismatch means the server implied placements the record
        # does not carry.
        free_placement = (
            not game.setup_black
            and len(game.moves) >= game.handicap
            and all(c == "B" for c, _ in game.moves[:game.handicap])
        )
        if not free_placement:
            return None, "handicap"
    if len(game.moves) < max(min_moves, 1):
        return None, "too_short"

    result = parse_result(game.result)
    if result is None:
        if not allow_no_result:
            return None, "no_result"
        result = 0

    board = GoBoard(board_size)
    try:
        board.setup(BLACK, game.setup_black)
        board.setup(WHITE, game.setup_white)
    except IllegalMove:
        return None, "illegal_move"

    T = len(game.moves)
    frames = np.zeros((T, CHANNELS, board_size, board_size), dtype=np.float32)
    place_idx = torch.zeros(T, dtype=torch.long)
    passed = torch.zeros(T, dtype=torch.bool)
    value = torch.zeros(T, dtype=torch.float32)

    for t, (color_ch, point) in enumerate(game.moves):
        color = BLACK if color_ch == "B" else WHITE
        frames[t] = encode_planes(board, color)          # position BEFORE move t
        if point is None:
            passed[t] = True                             # place_idx stays 0
        else:
            place_idx[t] = point[0] * board_size + point[1]
        if result != 0:
            mover_is_black = color == BLACK
            value[t] = 1.0 if (result > 0) == mover_is_black else -1.0
        try:
            board.play(color, point)                     # captures update grid
        except IllegalMove:
            return None, "illegal_move"

    sample = {
        "obs": {"board_seq": torch.from_numpy(frames)},
        "tgt": {"place_idx": place_idx, "passed": passed, "value": value},
        "meta": {
            "source": source,
            "players": {"black": game.player_black, "white": game.player_white},
            "ranks": {"black": game.rank_black, "white": game.rank_white},
            "result": game.result,
            "komi": game.komi,
            "handicap": game.handicap,
            "board_size": board_size,
            "n_moves": T,
        },
    }
    return sample, "ok"


def find_sgf_files(input_path: str) -> List[str]:
    """One file, or every *.sgf under a directory (recursive, sorted)."""
    if os.path.isfile(input_path):
        return [input_path]
    out: List[str] = []
    for root, _, files in os.walk(input_path):
        out.extend(
            os.path.join(root, f) for f in files if f.lower().endswith(".sgf")
        )
    return sorted(out)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="one .sgf file, or a directory searched recursively")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_size", type=int, default=256,
                   help="games per output shard")
    p.add_argument("--board_size", type=int, default=19,
                   help="games of any other size are skipped")
    p.add_argument("--max_games", type=int, default=None,
                   help="stop after converting this many games")
    p.add_argument("--min_moves", type=int, default=30,
                   help="skip games shorter than this")
    p.add_argument("--allow_handicap", action="store_true",
                   help="convert handicap games (AB/AW stones pre-placed)")
    p.add_argument("--allow_no_result", action="store_true",
                   help="convert games with unknown RE (value target = 0)")
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, int]:
    args = parse_args(argv)
    files = find_sgf_files(args.input)
    if not files:
        raise FileNotFoundError(f"no .sgf files under {args.input!r}")
    os.makedirs(args.output_dir, exist_ok=True)

    stats: Dict[str, int] = {"converted": 0, "unparseable_files": 0}
    stats.update({r: 0 for r in SKIP_REASONS})
    buffer: List[Dict] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx
        if not buffer:
            return
        path = os.path.join(args.output_dir, f"go_{shard_idx:05d}.pt")
        save_shard(path, game="go", samples=list(buffer))
        print(f"[shard] {path}: {len(buffer)} games")
        buffer.clear()
        shard_idx += 1

    done = False
    for path in files:
        if done:
            break
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                games = parse_sgf(fh.read())
        except (OSError, SGFError) as e:
            print(f"[warn] unparseable file {path}: {e}")
            stats["unparseable_files"] += 1
            continue
        for gi, game in enumerate(games):
            source = path if len(games) == 1 else f"{path}#{gi}"
            sample, reason = game_to_sample(
                game, args.board_size,
                allow_handicap=args.allow_handicap,
                allow_no_result=args.allow_no_result,
                min_moves=args.min_moves,
                source=source,
            )
            if sample is None:
                stats[reason] += 1
                if reason == "illegal_move":
                    print(f"[warn] illegal move in {source}: game skipped")
                continue
            buffer.append(sample)
            stats["converted"] += 1
            if len(buffer) >= args.shard_size:
                flush()
            if args.max_games is not None and stats["converted"] >= args.max_games:
                done = True
                break
    flush()

    print("[done] " + " ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    return stats


if __name__ == "__main__":
    main()
