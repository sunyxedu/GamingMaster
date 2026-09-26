"""Reference data from Fairy-Stockfish (pyffish) for the differential tests of the
gmrules game ``chess_variants`` (tests/test_chess_variants_reference.py).  Standalone:
runs in its own throw-away uv environment, never in the project one:

    # Fairy-Stockfish's own random playouts, every variant
    uv run --no-project --python 3.12 --with pyffish \\
        python scripts/reference/fairy_stockfish_dump.py --playouts 4 --seed 0 --out fsf.json
    # replay move lists (gmrules' random playouts, real lichess games) through it
    uv run --no-project --python 3.12 --with pyffish \\
        python scripts/reference/fairy_stockfish_dump.py --replay ours.json --out fsf.json

``--replay`` input: ``{"trajectories": [{"key", "variant", "start", "moves"}]}`` —
``variant`` a gmrules name, ``start`` the chess960 Scharnagl number (else null),
``moves`` UCI strings.  Output: ``{"source", "trajectories": [...]}``, each trajectory
with its ``fen`` (start position), ``moves``, ``steps`` (one per position BEFORE a
move: Fairy-Stockfish's legal moves ``legal`` — sorted UCI —, ``o`` = an optional game
end (fifty moves, repetition) is claimable there, ``ok`` = 0 when the next move is
illegal (replay); with ``--compact`` the string ``"n:hash[:o][:x]"`` — the legal moves'
count and :func:`legal_hash`, see :func:`parse_step`) and
``final`` (the position after the last move: its legal moves, ``immediate`` / ``optional``
game end ``[flag, value]``, ``insufficient`` ``[white, black]``, ``result`` = the game
result when no move is legal; values from the side to move's view: -32000 lost, 0 draw,
+32000 won).  A playout stops at the game end, at a claimable draw or at
``--max-plies``.

Notation: UCI with ``UCI_Chess960`` (castling = king to own rook square), drops
``P@e4``, promotions ``e7e8q`` / antichess ``e7e8k`` — gmrules' canonical action ids
(``gmrules::games::chess_variants::uci``) print the same strings.

Fairy-Stockfish (GPL-3.0, https://github.com/fairy-stockfish/Fairy-Stockfish; pyffish
on PyPI) is not a dependency of the project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from typing import Dict, List, Optional

try:
    import pyffish as sf
except ImportError:  # the project env imports the pure helpers (tests)
    sf = None

#: gmrules variant name → Fairy-Stockfish variant name
FSF = {
    "chess960": "chess",
    "crazyhouse": "crazyhouse",
    "atomic": "atomic",
    "antichess": "antichess",
    "threecheck": "3check",
    "kingofthehill": "kingofthehill",
    "horde": "horde",
    "racingkings": "racingkings",
}

KNIGHTS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


def chess960_back_rank(n: int) -> str:
    """White's back rank of chess960 start position ``n`` (Scharnagl numbering: 518 =
    RNBQKBNR)."""
    rank: List[Optional[str]] = [None] * 8
    n, b = divmod(n, 4)
    rank[2 * b + 1] = "B"                       # light-squared bishop: b, d, f, h
    n, b = divmod(n, 4)
    rank[2 * b] = "B"                           # dark-squared bishop: a, c, e, g
    n, q = divmod(n, 6)
    empty = [i for i in range(8) if rank[i] is None]
    rank[empty[q]] = "Q"
    k1, k2 = KNIGHTS[n]
    empty = [i for i in range(8) if rank[i] is None]
    rank[empty[k1]] = rank[empty[k2]] = "N"
    empty = [i for i in range(8) if rank[i] is None]
    for i, piece in zip(empty, "RKR"):
        rank[i] = piece
    return "".join(rank)


def start_fen(variant: str, start: Optional[int]) -> str:
    if variant == "chess960":
        white = chess960_back_rank(start)
        return f"{white.lower()}/pppppppp/8/8/8/8/PPPPPPPP/{white} w KQkq - 0 1"
    return sf.start_fen(FSF[variant])


def legal_hash(moves: List[str]) -> str:
    """Hash of a legal move set (the compact fixture form)."""
    return hashlib.sha1(" ".join(sorted(moves)).encode()).hexdigest()[:10]


def parse_step(step) -> Dict:
    """A step as a dict: full form as is; compact form ``"n:hash[:o][:x]"`` (o = a draw
    is claimable, x = the next move is illegal) → ``{"n", "h", "o", "ok"}``."""
    if isinstance(step, dict):
        return step
    n, h, *flags = step.split(":")
    return {"n": int(n), "h": h, "o": int("o" in flags), "ok": int("x" not in flags)}


def _legal(v: str, fen: str, moves: List[str]) -> List[str]:
    return sorted(sf.legal_moves(v, fen, moves, True))


def _step(legal: List[str], optional: bool, compact: bool, ok: bool = True):
    if compact:
        return f"{len(legal)}:{legal_hash(legal)}" + (":o" if optional else "") + ("" if ok else ":x")
    st: Dict = {"legal": legal}
    if optional:
        st["o"] = 1
    if not ok:
        st["ok"] = 0
    return st


def _final(v: str, fen: str, moves: List[str], compact: bool) -> Dict:
    legal = _legal(v, fen, moves)
    out = parse_step(_step(legal, False, compact))
    out.pop("o", None)
    out.pop("ok", None)
    out["immediate"] = list(sf.is_immediate_game_end(v, fen, moves, True))
    out["optional"] = list(sf.is_optional_game_end(v, fen, moves, True))
    out["insufficient"] = list(sf.has_insufficient_material(v, fen, moves, True))
    out["result"] = sf.game_result(v, fen, moves, True) if not legal else None
    out["halfmoves"] = int(sf.get_fen(v, fen, moves, True).split()[-2])
    return out


def playout(variant: str, rng: random.Random, max_plies: int, compact: bool) -> Dict:
    """Fairy-Stockfish's own uniformly random playout."""
    v = FSF[variant]
    start = rng.randrange(960) if variant == "chess960" else None
    fen = start_fen(variant, start)
    moves: List[str] = []
    steps = []
    while len(moves) < max_plies:
        legal = _legal(v, fen, moves)
        ended = not legal or sf.is_immediate_game_end(v, fen, moves, True)[0]
        claim = sf.is_optional_game_end(v, fen, moves, True)[0]
        if ended or claim:
            break
        steps.append(_step(legal, False, compact))
        moves.append(rng.choice(legal))
    return {"variant": variant, "start": start, "fen": fen, "moves": moves, "steps": steps,
            "final": _final(v, fen, moves, compact)}


def replay(t: Dict, compact: bool) -> Dict:
    """A given move list through Fairy-Stockfish (stops at the first move it rejects)."""
    variant = t["variant"]
    v = FSF[variant]
    fen = start_fen(variant, t.get("start"))
    moves: List[str] = []
    steps = []
    for m in t["moves"]:
        legal = _legal(v, fen, moves)
        steps.append(_step(legal, sf.is_optional_game_end(v, fen, moves, True)[0], compact, m in legal))
        if m not in legal:
            break
        moves.append(m)
    return {**t, "fen": fen, "moves": moves, "steps": steps, "final": _final(v, fen, moves, compact)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--playouts", type=int, default=0, help="random playouts per variant")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--replay", help="JSON of move lists to replay")
    p.add_argument("--compact", action="store_true", help="legal-move hashes, not lists")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = []
    rng = random.Random(a.seed)
    for variant in FSF:
        for i in range(a.playouts):
            t = playout(variant, rng, a.max_plies, a.compact)
            t["key"] = f"fsf:{variant}:{a.seed}:{i}"
            out.append(t)
    if a.replay:
        with open(a.replay) as fh:
            for t in json.load(fh)["trajectories"]:
                out.append(replay(t, a.compact))
    with open(a.out, "w") as fh:
        json.dump({"source": f"pyffish {'.'.join(map(str, sf.version()))}", "trajectories": out}, fh)


if __name__ == "__main__":
    main()
