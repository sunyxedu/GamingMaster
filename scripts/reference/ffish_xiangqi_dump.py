"""Reference trajectories of xiangqi from Fairy-Stockfish (pyffish) for the
differential tests of the gmrules xiangqi engine (tests/test_xiangqi_reference.py).
Standalone: runs in its own throw-away uv environment, never in the project one:

    uv run --no-project --python 3.12 --with pyffish==0.0.90 \\
        python scripts/reference/ffish_xiangqi_dump.py --episodes 40 --seed 0 --out ref.json
    # replay move lists produced by gmrules (JSON {"trajectories": [{"fen": ..., "moves":
    # [uci, ...]}, ...]}) through Fairy-Stockfish and record what it says
    uv run --no-project --python 3.12 --with pyffish==0.0.90 \\
        python scripts/reference/ffish_xiangqi_dump.py --replay ours.json --out ref.json

Every step records the position as Fairy-Stockfish sees it: its FEN (``get_fen``: the
transition), the sorted legal moves (UCI coordinates, ``legal_moves``), whether the
side to move is in check (``gives_check``), and at the end of the trajectory the game
end by the procedure the gmrules engine adjudicates after every move:

  1. ``is_optional_game_end`` (the natural move limit and the repetition rules:
     perpetual check / AXF chase lose, other repetitions draw) — a result for the side
     to move;
  2. no legal move: ``game_result`` (checkmate and stalemate lose in xiangqi);
  3. ``has_insufficient_material`` for both sides: draw.

``result`` is from Red's point of view (+1 / 0 / -1), ``reason`` the rule that ended
the game.  Playouts are random with biases towards repetitions (a side undoes its
previous move with probability ``--undo``) and checks (a side picks a checking move
with probability ``--check``), from the initial position and from endgame / chase
positions; the AXF examples of Fairy-Stockfish's own test suite are played first as
scripted openings (perpetual checks and chases, won / lost / drawn), so that every end
rule occurs.

pyffish (Fairy-Stockfish, GPL-3.0, https://github.com/fairy-stockfish/Fairy-Stockfish)
is not a dependency of the project.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from typing import Dict, List, Optional, Tuple

import pyffish as sf

V = "xiangqi"
START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

#: start positions of the random playouts (name → FEN)
STARTS: Dict[str, str] = {
    "start": START,
    # few pieces: checkmates, stalemates, bare kings, the move limit
    "rook_vs_advisors": "3ak4/4a4/9/9/9/9/9/9/9/R3K4 w - - 0 1",
    "horse_cannon": "4ka3/9/4b4/9/2p6/6P2/9/4B4/4A4/1NC1K4 w - - 0 1",
    "cannons": "3akab2/9/4b4/4n4/2p1c1p2/9/2P1C1P2/4B4/4A4/2BAK4 w - - 0 1",
    "soldiers": "4k4/4a4/9/2p1p1p2/9/9/2P1P1P2/9/4A4/4K4 b - - 0 1",
    "late_clock": "3k5/9/9/9/9/9/9/9/R8/4K4 w - - 90 80",
    # many checks late in the no-capture span: the AXF extension of the move limit
    "checks_clock": "4k4/9/9/9/9/9/9/9/R7R/3K5 w - - 70 60",
    "one_from_stalemate": "3k5/9/9/9/9/9/9/9/R8/4RK3 w - - 0 1",
    "stalemate_now": "3k5/R8/9/9/9/9/9/9/9/4RK3 b - - 0 1",
    # Fairy-Stockfish's AXF test positions (chases, perpetual checks)
    "axf_cannon_chase": "2bakabnr/9/r1n1c4/2p1p1p1p/PP7/9/4P1P1P/2C3NC1/9/1NBAKAB1R w - - 0 1",
    "axf_discovered": "2bakabr1/9/9/r1p1p1p2/p7R/P8/9/9/9/CC1AKA3 w - - 0 1",
    "axf_perpetual": "9/3kc4/3a5/3P5/9/4p4/9/4K4/9/3C5 w - - 0 1",
    "axf_mutual_chase": "4k4/7n1/9/4pR3/9/9/4P4/9/9/4K4 w - - 0 1",
    "axf_pins": "5k3/9/9/9/9/4c4/3n5/3NBA3/4A4/4K4 w - - 0 1",
    "axf_fake_cannon": "4ka3/c2R1R2c/4b4/9/9/9/9/9/9/4K4 w - - 0 1",
}


#: Fairy-Stockfish's AXF examples (pyffish test.py) as scripted openings: (FEN, moves)
SCRIPTED: Dict[str, Tuple[str, List[str]]] = {
    "axf_direct_chase": ("2bakabnr/9/r1n1c4/2p1p1p1p/PP7/9/4P1P1P/2C3NC1/9/1NBAKAB1R w - - 0 1",
                         ["c3a3", "a8b8", "a3b3", "b8a8", "b3a3", "a8b8", "a3b3", "b8a8", "b3a3"]),
    "axf_discovered_chase": ("2bakabr1/9/9/r1p1p1p2/p7R/P8/9/9/9/CC1AKA3 w - - 0 1",
                             ["a5a6", "a7b7", "a6b6", "b7a7", "b6a6", "a7b7", "a6b6", "b7a7",
                              "b6a6"]),
    "axf_anti_discovered": ("5k3/9/9/5C3/5c3/5C3/9/9/5p3/4K4 w - - 0 1",
                            2 * ["f5d5", "f6d6", "d5f5", "d6f6"]),
    "axf_soldier_check": ("3k5/9/9/9/9/5p3/9/5p3/5K3/5C3 w - - 0 1",
                          2 * ["f2e2", "f3e3", "e2f2", "e3f3"]),
    "axf_mutual_check": ("9/4c4/3k5/3r5/9/9/4C4/9/4K4/3R5 w - - 0 1",
                         2 * ["e4d4", "d7e7", "d4e4", "e7d7"]),
    "axf_mutual_chase": ("4k4/7n1/9/4pR3/9/9/4P4/9/9/4K4 w - - 0 1",
                         ["f7h7"] + 2 * ["h9f8", "h7h8", "f8g6", "h8g8", "g6i7", "g8g7", "i7h9",
                                         "g7h7"]),
    "axf_undermine_root": ("4k4/4c4/9/4p4/9/9/3rn4/3NR4/4K4/9 b - - 0 1",
                           2 * ["e4g5", "e2f2", "g5e4", "f2e2"]),
    "axf_discovered_check_threat": ("5k3/9/9/9/9/1N2P1C2/9/4BC3/9/cr1RK4 w - - 0 1",
                                    2 * ["b5c3", "b1c1", "c3b5", "c1b1"]),
    "axf_king_overprotection": ("3k5/9/9/9/9/9/3r5/9/9/3NK4 w - - 0 1",
                                2 * ["d1c3", "d4c4", "c3d1", "c4d4"]),
    "axf_fake_protection": ("5k3/9/9/9/9/1C7/1r7/9/1C7/4K4 w - - 0 1",
                            2 * ["b5c5", "b4c4", "c5b5", "c4b4"]),
}


def side_to_move(fen: str) -> int:
    return 0 if fen.split()[1] == "w" else 1


def game_end(fen0: str, moves: List[str], legal: List[str]) -> Optional[Tuple[int, str]]:
    """(result for Red, reason) or None, by the gmrules adjudication order."""
    fen = sf.get_fen(V, fen0, moves)
    red = 1 if side_to_move(fen) == 0 else -1
    end, val = sf.is_optional_game_end(V, fen0, moves)
    if end:
        v = (val > 0) - (val < 0)
        reason = "move_limit" if int(fen.split()[4]) > 99 and v == 0 else "repetition"
        return red * v, reason
    if not legal:
        val = sf.game_result(V, fen0, moves)
        v = (val > 0) - (val < 0)
        return red * v, "checkmate" if sf.gives_check(V, fen0, moves) else "stalemate"
    w, b = sf.has_insufficient_material(V, fen0, moves)
    if w and b:
        return 0, "material"
    return None


def step_record(fen0: str, moves: List[str]) -> Dict:
    legal = sorted(sf.legal_moves(V, fen0, moves))
    return {"fen": sf.get_fen(V, fen0, moves), "legal": " ".join(legal),
            "check": bool(sf.gives_check(V, fen0, moves))}, legal


def checking(fen0: str, moves: List[str], legal: List[str]) -> List[str]:
    return [m for m in legal if sf.gives_check(V, fen0, moves + [m])]


def playout(fen0: str, rng: random.Random, undo: float, check: float, max_plies: int,
            opening: Optional[List[str]] = None) -> Dict:
    moves: List[str] = []
    steps = []
    opening = list(opening or [])
    while True:
        rec, legal = step_record(fen0, moves)
        steps.append(rec)
        end = game_end(fen0, moves, legal)
        if end is not None:
            return {"fen": fen0, "moves": moves, "steps": steps,
                    "result": end[0], "reason": end[1]}
        if len(moves) >= max_plies:
            return {"fen": fen0, "moves": moves, "steps": steps, "result": None,
                    "reason": "max_plies"}
        mv = None
        if opening:
            mv = opening.pop(0)
            assert mv in legal, (fen0, moves, mv)
        elif rng.random() < check:
            checks = checking(fen0, moves, legal)
            if checks:
                mv = rng.choice(checks)
        if mv is None and len(moves) >= 2 and rng.random() < undo:
            back = moves[-2]
            i = next(k for k in range(1, len(back)) if back[k].isalpha())
            rev = back[i:] + back[:i]
            if rev in legal:
                mv = rev
        moves.append(mv if mv is not None else rng.choice(legal))


def replay(fen0: str, moves: List[str]) -> Dict:
    steps = []
    for t in range(len(moves) + 1):
        rec, legal = step_record(fen0, moves[:t])
        steps.append(rec)
        if t < len(moves) and moves[t] not in legal:
            return {"fen": fen0, "moves": moves[:t], "steps": steps, "result": None,
                    "reason": f"illegal:{moves[t]}"}
    end = game_end(fen0, moves, sorted(sf.legal_moves(V, fen0, moves)))
    return {"fen": fen0, "moves": moves, "steps": steps,
            "result": None if end is None else end[0],
            "reason": "running" if end is None else end[1]}


def write(out: str, data: Dict) -> None:
    raw = json.dumps(data, separators=(",", ":")).encode()
    if out.endswith(".gz"):
        with gzip.GzipFile(out, "wb", mtime=0) as fh:
            fh.write(raw)
    else:
        with open(out, "wb") as fh:
            fh.write(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="JSON (or .json.gz) output")
    p.add_argument("--episodes", type=int, default=2, help="playouts per start position")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--undo", type=float, default=0.4,
                   help="probability that a side undoes its previous move (repetitions)")
    p.add_argument("--check", type=float, default=0.3,
                   help="probability that a side plays a checking move when it has one")
    p.add_argument("--max_plies", type=int, default=400)
    p.add_argument("--replay", help="JSON {'trajectories': [{'fen', 'moves'}]} to replay")
    a = p.parse_args()
    info = {"pyffish": ".".join(map(str, sf.version())), "engine": sf.info()}
    if a.replay:
        with open(a.replay) as fh:
            todo = json.load(fh)
        trajs = [replay(t["fen"], t["moves"]) for t in todo["trajectories"]]
    else:
        rng = random.Random(a.seed)
        trajs = []
        for name, fen in STARTS.items():
            for k in range(a.episodes):
                clock = name.endswith("_clock")
                # alternate plain random play and the repetition / check biases
                undo = 0.0 if clock or k % 2 == 1 else a.undo
                check = a.check if not clock or k % 2 == 0 else 0.0
                t = playout(fen, rng, undo, check, a.max_plies)
                t["start"] = name
                trajs.append(t)
        for name, (fen, opening) in SCRIPTED.items():
            t = playout(fen, rng, a.undo, a.check, a.max_plies, opening)
            t["start"] = name
            trajs.append(t)
    write(a.out, {"reference": info, "variant": V, "trajectories": trajs})


if __name__ == "__main__":
    main()
