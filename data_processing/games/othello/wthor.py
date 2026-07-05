"""WTHOR ``.wtb`` binary parser (Fédération Française d'Othello game base).

File layout (verified against the FFO spec ``Format_WThor.pdf`` and two
independent parsers — LimeEng/wthor ``src/game_archive.rs`` and Edax
``src/base.c`` / ``src/game.c``):

16-byte header (all files share it; little-endian):
    offset 0   uint8   century of file creation (e.g. 20)
    offset 1   uint8   year of file creation (year % 100)
    offset 2   uint8   month
    offset 3   uint8   day
    offset 4   uint32  n1 — number of game records (.wtb)
    offset 8   uint16  n2 — number of name records (.jou/.trn; 0 for .wtb)
    offset 10  uint16  year of the games
    offset 12  uint8   p1 — board size: 0 or 8 → 8×8, 10 → 10×10
    offset 13  uint8   p2 — game type (0 = real games, 1 = solitaires)
    offset 14  uint8   p3 — depth used for the theoretical score
    offset 15  uint8   reserved

Then n1 game records of 68 bytes each (8×8 boards):
    offset 0   uint16  tournament label number (index into WTHOR.TRN)
    offset 2   uint16  black player number (index into WTHOR.JOU)
    offset 4   uint16  white player number
    offset 6   uint8   black real score (black discs, empties awarded to
                       the winner — so black+white always sums to 64)
    offset 7   uint8   black theoretical score (perfect play from depth p3)
    offset 8   60 × uint8 move bytes

Move byte convention (the part worth double-checking — VERIFIED against
both parsers, which agree):
    byte = 10 * (row + 1) + (col + 1)      (rows/cols 0-indexed here)
so the TENS digit is the row (rank 1..8) and the UNITS digit is the
column (file a..h): a1 = 11, h8 = 88, f5 = 56.
    LimeEng/wthor:  rank = byte/10 - 1, file = byte%10 - 1
    Edax game.c:    move_from_wthor(x) = 8*((x-11)/10) + ((x-11)%10)
Byte 0 is the filler after the game's last move.  PASSES ARE NOT STORED:
the move list is only the discs actually placed, so a replayer must
detect forced passes itself and attribute the next recorded move to the
other player.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

HEADER_SIZE = 16
RECORD_SIZE_8X8 = 68
N_MOVE_BYTES = 60

_HEADER_STRUCT = struct.Struct("<4BIHH4B")
_RECORD_HEAD_STRUCT = struct.Struct("<HHHBB")


@dataclass
class WthorHeader:
    century: int
    year: int
    month: int
    day: int
    n_games: int          # n1
    n_records: int        # n2 (0 for .wtb)
    game_year: int
    board_size: int       # p1: 0 or 8 → 8×8
    game_type: int        # p2: 0 = real games
    depth: int            # p3
    reserved: int = 0


@dataclass
class WthorGame:
    tournament_id: int
    black_player_id: int
    white_player_id: int
    black_real_score: int
    black_theoretical_score: int
    moves: List[Tuple[int, int]] = field(default_factory=list)  # (row, col) 0-indexed


def decode_move(byte: int) -> Tuple[int, int]:
    """WTHOR move byte → (row, col), both 0-indexed.  See module docstring
    for the verified ``byte = 10*(row+1) + (col+1)`` convention."""
    row, col = divmod(byte, 10)
    row -= 1
    col -= 1
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"invalid WTHOR 8×8 move byte {byte}")
    return row, col


def encode_move(row: int, col: int) -> int:
    """(row, col) 0-indexed → WTHOR move byte (inverse of decode_move)."""
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"square ({row}, {col}) off the 8×8 board")
    return 10 * (row + 1) + (col + 1)


def parse_wtb(data: bytes) -> Tuple[WthorHeader, List[WthorGame]]:
    """Parse a whole ``.wtb`` byte blob into (header, games).

    Move lists stop at the first 0 byte (filler): early-ended games
    (wipeouts) legitimately have fewer than 60 moves and trailing zeros.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"file too short for WTHOR header: {len(data)} bytes")
    (century, year, month, day, n1, n2, game_year,
     p1, p2, p3, reserved) = _HEADER_STRUCT.unpack_from(data, 0)
    header = WthorHeader(century, year, month, day, n1, n2, game_year,
                         p1, p2, p3, reserved)

    if header.board_size not in (0, 8):
        raise ValueError(
            f"unsupported WTHOR board size p1={header.board_size} (only 8×8)"
        )
    expected = HEADER_SIZE + RECORD_SIZE_8X8 * header.n_games
    if len(data) != expected:
        raise ValueError(
            f"size mismatch: n1={header.n_games} games need {expected} bytes, "
            f"file has {len(data)}"
        )

    games: List[WthorGame] = []
    for i in range(header.n_games):
        off = HEADER_SIZE + i * RECORD_SIZE_8X8
        (tournament_id, black_id, white_id,
         real_score, theo_score) = _RECORD_HEAD_STRUCT.unpack_from(data, off)
        moves: List[Tuple[int, int]] = []
        for b in data[off + 8: off + RECORD_SIZE_8X8]:
            if b == 0:  # filler — game ended before 60 discs were placed
                break
            moves.append(decode_move(b))
        games.append(WthorGame(tournament_id, black_id, white_id,
                               real_score, theo_score, moves))
    return header, games


def parse_wtb_file(path: str) -> Tuple[WthorHeader, List[WthorGame]]:
    with open(path, "rb") as f:
        return parse_wtb(f.read())


def build_wtb(
    games: Sequence[WthorGame],
    game_year: int = 2024,
    depth: int = 22,
) -> bytes:
    """Encode games back into ``.wtb`` bytes (inverse of :func:`parse_wtb`).

    Used by the round-trip tests; also handy for writing filtered subsets.
    """
    out = bytearray(
        _HEADER_STRUCT.pack(
            game_year // 100, game_year % 100, 1, 1,
            len(games), 0, game_year, 8, 0, depth, 0,
        )
    )
    for g in games:
        if len(g.moves) > N_MOVE_BYTES:
            raise ValueError(f"game has {len(g.moves)} moves > {N_MOVE_BYTES}")
        out += _RECORD_HEAD_STRUCT.pack(
            g.tournament_id, g.black_player_id, g.white_player_id,
            g.black_real_score, g.black_theoretical_score,
        )
        move_bytes = bytes(encode_move(r, c) for r, c in g.moves)
        out += move_bytes + bytes(N_MOVE_BYTES - len(move_bytes))
    return bytes(out)
