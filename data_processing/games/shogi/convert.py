"""Floodgate CSA records → shogi training shards.

Converts one-game-per-file CSA records (e.g. wdoor floodgate archives,
see ``fetch_floodgate.sh``) into the shared shard format consumed by
``data_processing.games.dataset`` and the shogi adapter
(:class:`model.games.shogi.ShogiEncoder` / ``ShogiDecoder``).

Frame convention
----------------
Frame ``t`` is the position **before** move ``t``, always from the
side-to-move's perspective: when white (gote) is to move the board is
rotated 180° so the mover always plays "up" (toward row 0).  One helper,
:func:`pov_square`, maps a python-shogi square index (row-major 9×9,
row 0 = rank 'a' = gote's back rank) to the POV index used both for the
board planes AND for the from/to/drop_to labels:

    black to move:  pov = sq
    white to move:  pov = 80 - sq          (180° rotation)

Observation layout (matches ShogiEncoder's obs keys)
----------------------------------------------------
``board_seq`` [T, 28, 9, 9] float32 — one-hot piece planes.  Plane order
is python-shogi piece_type − 1, i.e.::

    0..13   own pieces:      P L N S G B R K +P +L +N +S +B +R
    14..27  opponent pieces: same order

``hand_seq`` [T, 14, 4] float32 — slots 0..6 = own hand P L N S G B R,
slots 7..13 = opponent's, matching the encoder's slot layout.  Per-slot
features: ``[count, count > 0, min(count / 4, 1), 1.0]``.

``globals_seq`` [T, 4] float32:
``[min(move_number / 512, 1), mover_is_black, mover_in_check, 1.0]``.

Targets (match ShogiDecoder.il_loss / train_il_game LABEL_KEYS)
---------------------------------------------------------------
    dropped     [T] long   1 = drop, 0 = board move
    from_idx    [T] long   POV from-square; 0 on drop frames (unused branch)
    to_idx      [T] long   POV to-square;   0 on drop frames (unused branch)
    promoted    [T] bool   move promotes;   False on drop frames
    piece_idx   [T] long   own-hand slot 0..6 of the dropped piece type;
                           0 on move frames (unused branch)
    drop_to_idx [T] long   POV drop square; 0 on move frames (unused branch)
    value       [T] float  game result from the frame's mover perspective
                           (+1 mover wins / −1 loses / 0 draw)

Unused-branch labels are 0, the same convention the adapter's il_loss
tests use — il_loss gates every branch head by the ``dropped`` label, so
those zeros never contribute to the loss.

CLI
---
    uv run python -m data_processing.games.shogi.convert \
        --input floodgate_csa/ --output_dir shards/shogi \
        --shard_size 256 --max_games 50000 --min_moves 30

Games without a decisive result are skipped unless ``--allow_no_result``
(they get value 0 everywhere, like draws).  Every move is replayed and
validated with ``shogi.Board``; games containing an illegal move are
skipped with a warning count.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import shogi
import shogi.CSA
import torch

from data_processing.games.dataset import save_shard

BOARD_H = BOARD_W = 9
N_SQUARES = 81
PIECE_PLANES = 14          # P L N S G B R K +P +L +N +S +B +R (= piece_type − 1)
N_HAND_TYPES = 7           # droppable types P L N S G B R (= piece_type − 1)
HAND_FEATURE_DIM = 4
GLOBAL_DIM = 4
MOVE_NUMBER_SCALE = 512.0
HAND_COUNT_SCALE = 4.0

# CSA end-of-game tokens → result semantics.  The side to move when the
# token appears loses on LOSS tokens and wins on %KACHI (declaration).
_STM_LOSES = {"%TORYO", "%TIME_UP", "%ILLEGAL_MOVE", "%TSUMI"}
_STM_WINS = {"%KACHI"}
_DRAW = {"%SENNICHITE", "%JISHOGI", "%HIKIWAKE"}

RESULT_BLACK, RESULT_WHITE, RESULT_DRAW, RESULT_NONE = "b", "w", "draw", "none"


def pov_square(sq: int, mover_is_black: bool) -> int:
    """python-shogi square index → POV index (board planes AND labels).

    python-shogi squares are row-major with square 0 = 9a (gote's back
    rank, top-left from sente's view); sente (black) already plays "up",
    so black frames use the index unchanged and white frames rotate 180°.
    """
    return sq if mover_is_black else (N_SQUARES - 1) - sq


def encode_frame(board: shogi.Board) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Position → (board planes [28,9,9], hand [14,4], globals [4])."""
    mover = board.turn
    mover_is_black = mover == shogi.BLACK
    opponent = shogi.WHITE if mover_is_black else shogi.BLACK

    planes = torch.zeros(2 * PIECE_PLANES, BOARD_H, BOARD_W)
    for sq in shogi.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pov = pov_square(sq, mover_is_black)
        plane = (piece.piece_type - 1) + (0 if piece.color == mover else PIECE_PLANES)
        planes[plane, pov // BOARD_W, pov % BOARD_W] = 1.0

    hand = torch.zeros(2 * N_HAND_TYPES, HAND_FEATURE_DIM)
    for offset, color in ((0, mover), (N_HAND_TYPES, opponent)):
        in_hand = board.pieces_in_hand[color]
        for piece_type in range(shogi.PAWN, shogi.KING):  # P L N S G B R
            count = float(in_hand[piece_type])
            slot = offset + piece_type - 1
            hand[slot, 0] = count
            hand[slot, 1] = 1.0 if count > 0 else 0.0
            hand[slot, 2] = min(count / HAND_COUNT_SCALE, 1.0)
            hand[slot, 3] = 1.0

    glb = torch.tensor([
        min(board.move_number / MOVE_NUMBER_SCALE, 1.0),
        1.0 if mover_is_black else 0.0,
        1.0 if board.is_check() else 0.0,
        1.0,
    ])
    return planes, hand, glb


def move_labels(move: shogi.Move, mover_is_black: bool) -> Dict[str, float]:
    """One move → target scalars (unused-branch labels are 0)."""
    if move.drop_piece_type is not None:
        return {
            "dropped": 1,
            "from_idx": 0,
            "to_idx": 0,
            "promoted": False,
            "piece_idx": move.drop_piece_type - 1,      # own-hand slot 0..6
            "drop_to_idx": pov_square(move.to_square, mover_is_black),
        }
    return {
        "dropped": 0,
        "from_idx": pov_square(move.from_square, mover_is_black),
        "to_idx": pov_square(move.to_square, mover_is_black),
        "promoted": bool(move.promotion),
        "piece_idx": 0,
        "drop_to_idx": 0,
    }


def game_result(csa_text: str, start_turn: int, n_moves: int) -> str:
    """Result from the CSA end token (python-shogi's ``win`` field folds
    %KACHI / %TSUMI / draws into '-', so we classify the token ourselves).
    """
    token = None
    for line in csa_text.splitlines():
        line = line.strip()
        if line.startswith("%"):
            token = line.split()[0]
            break
    if token == "%+ILLEGAL_ACTION":                     # black acted illegally
        return RESULT_WHITE
    if token == "%-ILLEGAL_ACTION":
        return RESULT_BLACK
    if token in _DRAW:
        return RESULT_DRAW
    if token in _STM_LOSES or token in _STM_WINS:
        stm = (start_turn + n_moves) % 2                # side to move at the end
        winner = stm if token in _STM_WINS else 1 - stm
        return RESULT_BLACK if winner == shogi.BLACK else RESULT_WHITE
    return RESULT_NONE                                  # %CHUDAN / missing / unknown


def result_value(result: str, mover_is_black: bool) -> float:
    if result == RESULT_BLACK:
        return 1.0 if mover_is_black else -1.0
    if result == RESULT_WHITE:
        return -1.0 if mover_is_black else 1.0
    return 0.0                                          # draw / no result


class SkipGame(Exception):
    """Raised with a reason key when a game cannot be converted."""


def convert_game(
    csa_text: str,
    min_moves: int = 30,
    allow_no_result: bool = False,
    self_check: bool = False,
) -> Dict:
    """One CSA record → one shard sample; raises :class:`SkipGame`."""
    try:
        parsed = shogi.CSA.Parser.parse_str(csa_text)[0]
    except Exception as exc:                            # malformed record
        raise SkipGame("parse_error") from exc

    usi_moves: List[str] = parsed["moves"]
    if len(usi_moves) < min_moves:
        raise SkipGame("too_short")

    board = shogi.Board(parsed["sfen"])
    result = game_result(csa_text, board.turn, len(usi_moves))
    if result == RESULT_NONE and not allow_no_result:
        raise SkipGame("no_result")

    boards, hands, globals_, tgt_rows = [], [], [], []
    for usi in usi_moves:
        move = shogi.Move.from_usi(usi)
        if not board.is_legal(move):
            raise SkipGame("illegal_move")
        mover_is_black = board.turn == shogi.BLACK
        planes, hand, glb = encode_frame(board)
        labels = move_labels(move, mover_is_black)
        labels["value"] = result_value(result, mover_is_black)

        if self_check:
            if labels["dropped"]:
                slot = labels["piece_idx"]
                if hand[slot, 0] < 1.0:
                    raise AssertionError(
                        f"self-check: drop of hand slot {slot} but count "
                        f"{hand[slot, 0].item()} < 1 ({usi})"
                    )
            else:
                f = labels["from_idx"]
                if planes[:PIECE_PLANES, f // BOARD_W, f % BOARD_W].sum() < 1.0:
                    raise AssertionError(
                        f"self-check: from_idx {f} holds no own piece ({usi})"
                    )

        boards.append(planes)
        hands.append(hand)
        globals_.append(glb)
        tgt_rows.append(labels)
        board.push(move)

    sample = {
        "obs": {
            "board_seq": torch.stack(boards),
            "hand_seq": torch.stack(hands),
            "globals_seq": torch.stack(globals_),
        },
        "tgt": {
            "dropped": torch.tensor([r["dropped"] for r in tgt_rows], dtype=torch.long),
            "from_idx": torch.tensor([r["from_idx"] for r in tgt_rows], dtype=torch.long),
            "to_idx": torch.tensor([r["to_idx"] for r in tgt_rows], dtype=torch.long),
            "promoted": torch.tensor([r["promoted"] for r in tgt_rows], dtype=torch.bool),
            "piece_idx": torch.tensor([r["piece_idx"] for r in tgt_rows], dtype=torch.long),
            "drop_to_idx": torch.tensor([r["drop_to_idx"] for r in tgt_rows], dtype=torch.long),
            "value": torch.tensor([r["value"] for r in tgt_rows], dtype=torch.float32),
        },
        "meta": {
            "players": parsed["names"],
            "result": result,
            "plies": len(usi_moves),
        },
    }
    return sample


def collect_inputs(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "**", "*.csa"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .csa files under {path}")
    return files


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="dir of .csa files or one file")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_size", type=int, default=256, help="games per shard")
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--min_moves", type=int, default=30)
    p.add_argument("--allow_no_result", action="store_true",
                   help="keep games with no decisive result (value 0)")
    p.add_argument("--self_check_games", type=int, default=16,
                   help="verify plane/hand consistency on the first N games")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, int]:
    args = parse_args(argv)
    files = collect_inputs(args.input)

    stats = {"converted": 0, "shards": 0, "parse_error": 0, "too_short": 0,
             "no_result": 0, "illegal_move": 0}
    pending: List[Dict] = []

    def flush() -> None:
        if not pending:
            return
        path = os.path.join(args.output_dir, f"shogi-{stats['shards']:05d}.pt")
        save_shard(path, game="shogi", samples=list(pending))
        stats["shards"] += 1
        pending.clear()

    for i, path in enumerate(files):
        if args.max_games is not None and stats["converted"] >= args.max_games:
            break
        with open(path, encoding="utf-8", errors="replace") as f:
            csa_text = f.read()
        try:
            sample = convert_game(
                csa_text,
                min_moves=args.min_moves,
                allow_no_result=args.allow_no_result,
                self_check=stats["converted"] < args.self_check_games,
            )
        except SkipGame as skip:
            reason = skip.args[0]
            stats[reason] += 1
            if reason == "illegal_move":
                print(f"[warn] illegal move, skipping {path}", file=sys.stderr)
            continue
        sample["meta"]["source"] = os.path.basename(path)
        pending.append(sample)
        stats["converted"] += 1
        if len(pending) >= args.shard_size:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"[progress] {i + 1}/{len(files)} files, "
                  f"{stats['converted']} games converted")
    flush()

    print(f"[done] converted={stats['converted']} shards={stats['shards']} "
          f"skipped: parse_error={stats['parse_error']} "
          f"too_short={stats['too_short']} no_result={stats['no_result']} "
          f"illegal_move={stats['illegal_move']}")
    return stats


if __name__ == "__main__":
    main()
