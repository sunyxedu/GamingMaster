"""Convert WTHOR ``.wtb`` databases into training shards for the
"othello" adapter (BoardEncoder + PlacementDecoder, 8×8, 3 planes).

    uv run python -m data_processing.games.othello.convert \
        --input data/othello/wthor/ --output_dir data/othello/train \
        --shard_size 512 --max_games 100000

One sample = one full game seen from the SIDE-TO-MOVE perspective at
every frame; frame t is the position BEFORE the t-th half-move,
INCLUDING the pass frames the replayer inserts (WTHOR does not store
passes).  Perspective is a channel swap only — no geometric flip.

    obs["board_seq"]  [T, 3, 8, 8] float32 — plane 0 mover's discs,
                      plane 1 opponent's discs, plane 2 empty
    tgt["place_idx"]  [T] long   row*8+col (0 on pass frames)
    tgt["passed"]     [T] bool
    tgt["value"]      [T] float  sign(final own discs − opp discs) from
                      the frame mover's perspective, computed from the
                      REPLAYED final board

A recorded move that is illegal under exact rules aborts the game
(skipped + counted); the recorded black score is compared against the
replayed one (empties-awarded-to-winner convention) as a self-check —
mismatches are counted, never fatal.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from data_processing.games.dataset import save_shard
from data_processing.games.othello.rules import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    WHITE,
    apply_move,
    disc_counts,
    flips,
    has_legal_move,
    initial_board,
    opponent,
)
from data_processing.games.othello.wthor import WthorGame, parse_wtb_file


class IllegalGameError(ValueError):
    """A recorded move is illegal under exact rules (or moves remain
    after the game is over) — the whole game record is unusable."""


def board_planes(board: bytearray, mover: int) -> torch.Tensor:
    """[3, 8, 8] float32 from ``mover``'s perspective (own / opp / empty)."""
    grid = torch.tensor(list(board), dtype=torch.uint8).reshape(
        BOARD_SIZE, BOARD_SIZE
    )
    return torch.stack(
        [grid == mover, grid == opponent(mover), grid == EMPTY]
    ).float()


def replay_frames(
    moves: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[torch.Tensor, int, int, bool]], bytearray]:
    """Replay recorded moves, inserting the unstored pass frames.

    Returns (frames, final_board) where each frame is
    (planes [3,8,8], mover, place_idx, passed) — the position BEFORE the
    half-move, from the mover's perspective.  Raises
    :class:`IllegalGameError` on any inconsistency.
    """
    board = initial_board()
    player = BLACK
    frames: List[Tuple[torch.Tensor, int, int, bool]] = []
    for n, (row, col) in enumerate(moves):
        # WTHOR stores no passes: if the side to move is blocked, insert
        # the forced pass and hand the recorded move to the other player.
        while not has_legal_move(board, player):
            if not has_legal_move(board, opponent(player)):
                raise IllegalGameError(
                    f"move {n}: game already over (double pass) but "
                    f"{len(moves) - n} recorded moves remain"
                )
            frames.append((board_planes(board, player), player, 0, True))
            player = opponent(player)
        if not flips(board, player, row, col):
            raise IllegalGameError(
                f"move {n}: ({row}, {col}) is illegal for player {player}"
            )
        frames.append(
            (board_planes(board, player), player, row * BOARD_SIZE + col, False)
        )
        apply_move(board, player, row, col)
        player = opponent(player)
    return frames, board


def adjusted_black_score(black: int, white: int) -> int:
    """WTHOR "real score" convention: black's disc count with the empty
    squares awarded to the winner (split on a draw), so scores sum to 64."""
    empties = BOARD_SIZE * BOARD_SIZE - black - white
    if black > white:
        return black + empties
    if white > black:
        return black
    return black + empties // 2


def game_to_sample(
    game: WthorGame, source: str, game_index: int,
) -> Tuple[Dict, bool]:
    """(sample dict for save_shard, recorded-vs-replayed score matches)."""
    frames, final_board = replay_frames(game.moves)
    if not frames:
        raise IllegalGameError("empty move list")
    black, white = disc_counts(final_board)

    planes = torch.stack([f[0] for f in frames])                    # [T,3,8,8]
    movers = [f[1] for f in frames]
    place_idx = torch.tensor([f[2] for f in frames], dtype=torch.long)
    passed = torch.tensor([f[3] for f in frames], dtype=torch.bool)

    black_sign = float(black > white) - float(black < white)        # +1/0/−1
    value = torch.tensor(
        [black_sign if m == BLACK else -black_sign for m in movers],
        dtype=torch.float32,
    )

    sample = {
        "obs": {"board_seq": planes},
        "tgt": {"place_idx": place_idx, "passed": passed, "value": value},
        "meta": {
            "source": source,
            "game_index": game_index,
            "tournament_id": game.tournament_id,
            "black_player_id": game.black_player_id,
            "white_player_id": game.white_player_id,
            "recorded_black_score": game.black_real_score,
            "theoretical_black_score": game.black_theoretical_score,
            "replayed_black": black,
            "replayed_white": white,
            "n_recorded_moves": len(game.moves),
            "n_pass_frames": int(passed.sum()),
        },
    }
    score_ok = adjusted_black_score(black, white) == game.black_real_score
    return sample, score_ok


def input_files(path: str) -> List[str]:
    if os.path.isdir(path):
        found = sorted(
            glob.glob(os.path.join(path, "**", "*.wtb"), recursive=True)
        ) or sorted(glob.glob(os.path.join(path, "**", "*.WTB"), recursive=True))
        if not found:
            raise FileNotFoundError(f"no .wtb files under {path}")
        return found
    return [path]


def convert(
    input_path: str,
    output_dir: str,
    shard_size: int = 512,
    max_games: Optional[int] = None,
) -> Dict[str, int]:
    stats = {
        "games": 0, "converted": 0, "illegal": 0,
        "score_mismatch": 0, "pass_frames": 0, "shards": 0,
    }
    pending: List[Dict] = []

    def flush() -> None:
        if not pending:
            return
        path = os.path.join(output_dir, f"othello_{stats['shards']:05d}.pt")
        save_shard(path, "othello", list(pending))
        stats["shards"] += 1
        pending.clear()

    done = False
    for path in input_files(input_path):
        if done:
            break
        _, games = parse_wtb_file(path)
        name = os.path.basename(path)
        for i, game in enumerate(games):
            if max_games is not None and stats["games"] >= max_games:
                done = True
                break
            stats["games"] += 1
            try:
                sample, score_ok = game_to_sample(game, name, i)
            except IllegalGameError as e:
                stats["illegal"] += 1
                print(f"[warn] {name}#{i}: skipped — {e}")
                continue
            stats["converted"] += 1
            stats["pass_frames"] += sample["meta"]["n_pass_frames"]
            if not score_ok:
                stats["score_mismatch"] += 1
            pending.append(sample)
            if len(pending) >= shard_size:
                flush()
    flush()
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help=".wtb file, or directory scanned recursively")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_size", type=int, default=512)
    p.add_argument("--max_games", type=int, default=None)
    args = p.parse_args()

    stats = convert(args.input, args.output_dir,
                    shard_size=args.shard_size, max_games=args.max_games)
    print(
        f"[done] games={stats['games']} converted={stats['converted']} "
        f"illegal={stats['illegal']} score_mismatch={stats['score_mismatch']} "
        f"pass_frames={stats['pass_frames']} shards={stats['shards']} "
        f"→ {args.output_dir}"
    )


if __name__ == "__main__":
    main()
