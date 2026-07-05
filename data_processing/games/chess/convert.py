"""Lichess PGN → training shards for the chess adapter.

Streams games from a ``.pgn`` or ``.pgn.zst`` file (never loads the whole
file) and emits shards in the shared format of
:mod:`data_processing.games.dataset` via :func:`save_shard`.

One sample per game.  Every frame ``t`` is the position BEFORE move ``t``
observed from the SIDE TO MOVE (the Maia convention): the board is rotated
180° when black is to move so "own" pieces always advance toward row 0.
The label is that player's move — an IL policy trained on this always
predicts the mover.

Observation layout
------------------
``obs["board_seq"]`` [T, 14, 8, 8] float32, side-to-move POV:

    planes 0-5   own pieces   P, N, B, R, Q, K
    planes 6-11  opponent     P, N, B, R, Q, K
    plane  12    en-passant target square one-hot (POV frame; the square
                 python-chess reports after any double pawn push)
    plane  13    all ones (bias / on-board plane)

POV index mapping (single source of truth: :func:`square_to_pov_idx`,
used for board planes, the ep square AND the from/to labels so they can
never disagree):

    white to move (flip=False):  row = 7 - rank, col = file
    black to move (flip=True):   row = rank,     col = 7 - file   (180°)

so the mover's back rank is row 7 (e.g. white's e2 → row 6, col 4 →
index 52) and its pawns advance toward row 0.

``obs["globals_seq"]`` [T, 8] float32:

    [own kingside castling, own queenside castling,
     opp kingside castling, opp queenside castling,
     side_to_move_is_white, fullmove/100 (clipped to 1),
     halfmove_clock/100 (clipped to 1), is_check]

Targets
-------
``tgt["from_idx"]`` / ``tgt["to_idx"]`` [T] long — POV square indices
(row*8 + col).  ``tgt["promotion"]`` [T] long with

    0 = none, 1 = knight, 2 = bishop, 3 = rook, 4 = queen

(matches the adapter default ``n_promotions=5``; index 0 = no promotion).
``tgt["value"]`` [T] float — final game outcome from the frame's mover
perspective: +1 the mover went on to win, 0 draw, −1 loss.

CLI
---
    uv run python -m data_processing.games.chess.convert \
        --input games.pgn[.zst] --output_dir shards/ \
        --shard_size 256 --min_elo 0 --max_games N --min_plies 10 \
        [--skip_bullet]

Filters: both players ≥ ``min_elo`` (games without Elo tags are skipped
unless ``min_elo == 0``), at least ``min_plies`` mainline plies, standard
chess only (variants and FEN-setup games skipped), decisive/drawn result
tag required, and optionally no bullet (estimated duration
base + 40·increment < 180 s, per the Lichess speed classification).
"""

from __future__ import annotations

import argparse
import io
import os
from typing import Dict, Iterator, Optional, TextIO

import chess
import chess.pgn
import torch

from data_processing.games.dataset import save_shard

# promotion piece_type → label index (0 = no promotion)
PROMO_TO_IDX: Dict[Optional[int], int] = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
IDX_TO_PROMO: Dict[int, Optional[int]] = {v: k for k, v in PROMO_TO_IDX.items()}

RESULT_TO_WHITE_VALUE = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

N_PLANES = 14
EP_PLANE = 12
ONES_PLANE = 13


# ----------------------------------------------------------------------
# POV mapping — the single helper shared by planes, ep square and labels.
# ----------------------------------------------------------------------

def square_to_pov_idx(square: int, flip: bool) -> int:
    """0..63 square → POV index (row*8 + col).

    flip=False (white to move): row = 7 - rank, col = file.
    flip=True  (black to move): 180° rotation — row = rank, col = 7 - file.
    """
    file, rank = chess.square_file(square), chess.square_rank(square)
    if flip:
        return rank * 8 + (7 - file)
    return (7 - rank) * 8 + file


def pov_idx_to_square(idx: int, flip: bool) -> int:
    """Inverse of :func:`square_to_pov_idx` (the mapping is an involution
    on (row, col) → (rank, file), so this mirrors it exactly)."""
    row, col = divmod(idx, 8)
    if flip:
        return chess.square(7 - col, row)
    return chess.square(col, 7 - row)


# ----------------------------------------------------------------------
# Frame encoding
# ----------------------------------------------------------------------

def encode_frame(board: chess.Board) -> tuple:
    """(planes [14, 8, 8], globals [8]) for the side to move."""
    flip = board.turn == chess.BLACK
    own = board.turn
    planes = torch.zeros(N_PLANES, 8, 8)
    for square, piece in board.piece_map().items():
        base = 0 if piece.color == own else 6
        row, col = divmod(square_to_pov_idx(square, flip), 8)
        planes[base + piece.piece_type - 1, row, col] = 1.0
    if board.ep_square is not None:
        row, col = divmod(square_to_pov_idx(board.ep_square, flip), 8)
        planes[EP_PLANE, row, col] = 1.0
    planes[ONES_PLANE] = 1.0

    glb = torch.tensor(
        [
            float(board.has_kingside_castling_rights(own)),
            float(board.has_queenside_castling_rights(own)),
            float(board.has_kingside_castling_rights(not own)),
            float(board.has_queenside_castling_rights(not own)),
            float(own == chess.WHITE),
            min(board.fullmove_number / 100.0, 1.0),
            min(board.halfmove_clock / 100.0, 1.0),
            float(board.is_check()),
        ],
        dtype=torch.float32,
    )
    return planes, glb


# ----------------------------------------------------------------------
# Game filtering + encoding
# ----------------------------------------------------------------------

def _elo(headers: chess.pgn.Headers, key: str) -> Optional[int]:
    try:
        return int(headers[key])
    except (KeyError, ValueError):
        return None


def _is_bullet(headers: chess.pgn.Headers) -> bool:
    """Lichess speed rule: estimated duration base + 40·inc < 180 s.

    Unknown/correspondence time controls ("-", missing) are NOT bullet.
    """
    tc = headers.get("TimeControl", "-")
    if "+" not in tc:
        return False
    try:
        base, inc = (int(x) for x in tc.split("+", 1))
    except ValueError:
        return False
    return base + 40 * inc < 180


def game_passes_filters(
    game: chess.pgn.Game,
    min_elo: int,
    min_plies: int,
    skip_bullet: bool,
) -> bool:
    h = game.headers
    if h.get("Variant", "Standard") != "Standard":
        return False  # Chess960 / Atomic / "From Position" / ...
    if "FEN" in h or h.get("SetUp") == "1":
        return False  # non-initial setup
    if h.get("Result") not in RESULT_TO_WHITE_VALUE:
        return False  # "*" / unknown
    if min_elo > 0:
        we, be = _elo(h, "WhiteElo"), _elo(h, "BlackElo")
        if we is None or be is None or we < min_elo or be < min_elo:
            return False
    if skip_bullet and _is_bullet(h):
        return False
    if sum(1 for _ in game.mainline_moves()) < min_plies:
        return False
    return True


def encode_game(game: chess.pgn.Game, self_check: bool = False) -> Dict:
    """One shard sample from a filtered game (side-to-move POV frames)."""
    white_value = RESULT_TO_WHITE_VALUE[game.headers["Result"]]

    boards, globals_, from_idx, to_idx, promotion, value = [], [], [], [], [], []
    board = game.board()
    for move in game.mainline_moves():
        flip = board.turn == chess.BLACK
        planes, glb = encode_frame(board)
        f = square_to_pov_idx(move.from_square, flip)
        t = square_to_pov_idx(move.to_square, flip)
        if self_check:
            # Invariant: the from-square must hold an own piece in the
            # emitted planes — planes and labels share one POV mapping.
            row, col = divmod(f, 8)
            assert planes[0:6, row, col].sum() > 0, (
                f"from_idx {f} ({move.uci()}) empty in own planes"
            )
        boards.append(planes)
        globals_.append(glb)
        from_idx.append(f)
        to_idx.append(t)
        promotion.append(PROMO_TO_IDX[move.promotion])
        value.append(white_value if board.turn == chess.WHITE else -white_value)
        board.push(move)

    h = game.headers
    return {
        "obs": {
            "board_seq": torch.stack(boards),                      # [T,14,8,8]
            "globals_seq": torch.stack(globals_),                  # [T,8]
        },
        "tgt": {
            "from_idx": torch.tensor(from_idx, dtype=torch.long),  # [T]
            "to_idx": torch.tensor(to_idx, dtype=torch.long),      # [T]
            "promotion": torch.tensor(promotion, dtype=torch.long),
            "value": torch.tensor(value, dtype=torch.float32),
        },
        "meta": {
            "white": h.get("White", "?"),
            "black": h.get("Black", "?"),
            "elo": (_elo(h, "WhiteElo"), _elo(h, "BlackElo")),
            "result": h.get("Result"),
        },
    }


# ----------------------------------------------------------------------
# Streaming input
# ----------------------------------------------------------------------

def open_pgn(path: str) -> TextIO:
    """Text handle over a .pgn or .pgn.zst — streamed, never fully loaded."""
    if path.endswith(".zst"):
        import zstandard

        fh = open(path, "rb")
        reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_games(handle: TextIO) -> Iterator[chess.pgn.Game]:
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            return
        if game.errors:  # malformed movetext — never emit a broken sample
            continue
        yield game


# ----------------------------------------------------------------------
# Conversion driver
# ----------------------------------------------------------------------

def convert(
    input_path: str,
    output_dir: str,
    shard_size: int = 256,
    min_elo: int = 0,
    max_games: Optional[int] = None,
    min_plies: int = 10,
    skip_bullet: bool = False,
    self_check_games: int = 16,
) -> Dict[str, int]:
    """Stream ``input_path`` into ``output_dir/chess_XXXXX.pt`` shards.

    Returns counters: {"seen", "kept", "shards"}.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples, seen, kept, shard_i = [], 0, 0, 0

    def flush():
        nonlocal samples, shard_i
        if samples:
            save_shard(
                os.path.join(output_dir, f"chess_{shard_i:05d}.pt"),
                game="chess",
                samples=samples,
            )
            shard_i += 1
            samples = []

    handle = open_pgn(input_path)
    try:
        for game in iter_games(handle):
            seen += 1
            if not game_passes_filters(game, min_elo, min_plies, skip_bullet):
                continue
            samples.append(encode_game(game, self_check=kept < self_check_games))
            kept += 1
            if len(samples) >= shard_size:
                flush()
            if max_games is not None and kept >= max_games:
                break
    finally:
        handle.close()
    flush()
    return {"seen": seen, "kept": kept, "shards": shard_i}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help=".pgn or .pgn.zst file")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_size", type=int, default=256)
    p.add_argument("--min_elo", type=int, default=0,
                   help="both players ≥ this; >0 also skips untagged games")
    p.add_argument("--max_games", type=int, default=None,
                   help="stop after emitting this many samples")
    p.add_argument("--min_plies", type=int, default=10)
    p.add_argument("--skip_bullet", action="store_true",
                   help="skip time controls with base + 40*inc < 180 s")
    args = p.parse_args()

    stats = convert(
        args.input, args.output_dir,
        shard_size=args.shard_size, min_elo=args.min_elo,
        max_games=args.max_games, min_plies=args.min_plies,
        skip_bullet=args.skip_bullet,
    )
    print(f"[chess] seen={stats['seen']} kept={stats['kept']} "
          f"shards={stats['shards']} → {args.output_dir}")


if __name__ == "__main__":
    main()
