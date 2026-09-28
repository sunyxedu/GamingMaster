"""Engine self-play game records: a UCI engine plays both sides of full games under the
gmrules game's own rules (gmsim.State decides every legal move and the result), written in
the game's converter input format — superhuman games that the IL pipeline treats like any
other records (converted, then labelled by the same engine: ``gmdata label <game>``).

Xiangqi with Pikafish (GPL-3.0; the ``~/gm-engines`` install of scripts/engines/
fetch_engines.sh): PGN files with coordinate moves (``[Format "UCI"]`` style: files a-i,
ranks 1-10 from Red's side — ``gmdata xiangqi convert`` reads them; Pikafish's own ranks are
0-9).  Openings: the first ``--sample_plies`` plies draw from the engine's MultiPV lines
within ``--window`` centipawns of the best (softmax at ``--temperature``), so no two games
are alike; the rest are the engine's best move.  A game ends by the rules (mate, stalemate,
the rules' repetition / move-count draws), or is adjudicated: a side the engine scores at
<= -``--resign_cp`` for ``--resign_plies`` plies in a row loses, and a game past
``--max_plies`` is a draw (``[Termination]`` says which).

    uv run python scripts/engine_selfplay.py --game xiangqi --engine ~/gm-engines/bin/pikafish \\
        --games 1000 --nodes 50000 --workers 16 --out ~/gm-data/xiangqi-selfplay/raw
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class Uci:
    """A minimal UCI client (one engine process)."""

    def __init__(self, path: str, options: dict):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait("readyok")

    def send(self, line: str) -> None:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def wait(self, token: str) -> List[str]:
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def search(self, moves: List[str], nodes: int, multipv: int) -> Tuple[str, List[Tuple[str, int]]]:
        """(bestmove, [(first move, score cp from the mover's side) per MultiPV line])."""
        self.send(f"setoption name MultiPV value {multipv}")
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        self.send(f"go nodes {nodes}")
        lines = self.wait("bestmove")
        best = lines[-1].split()[1]
        pv = {}
        for line in lines:
            f = line.split()
            if f[:1] != ["info"] or "pv" not in f or "score" not in f:
                continue
            k = int(f[f.index("multipv") + 1]) if "multipv" in f else 1
            i = f.index("score")
            kind, val = f[i + 1], int(f[i + 2])
            cp = val if kind == "cp" else (100000 - abs(val)) * (1 if val > 0 else -1)
            pv[k] = (f[f.index("pv") + 1], cp)
        return best, [pv[k] for k in sorted(pv)]

    def close(self) -> None:
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.p.kill()


# -- xiangqi -----------------------------------------------------------------

def _split(m: str) -> Tuple[str, int, str, int]:
    # a1a2 / a10a9 / i9i10: letter, digits, letter, digits
    j = 1
    while m[j].isdigit():
        j += 1
    return m[0], int(m[1:j]), m[j], int(m[j + 1:])


def to_engine_move(m: str) -> str:
    f0, r0, f1, r1 = _split(m)
    return f"{f0}{r0 - 1}{f1}{r1 - 1}"


def from_engine_move(m: str) -> str:
    return f"{m[0]}{int(m[1]) + 1}{m[2]}{int(m[3]) + 1}"


def play_xiangqi(engine: Uci, rng: random.Random, a) -> Tuple[List[str], str, str]:
    import gmsim

    st = gmsim.State("xiangqi", {})
    moves: List[str] = []
    losing = [0, 0]
    termination = "rules"
    while True:
        kind, seats = st.turn()
        if kind != "act":
            break
        seat = seats[0]
        legal = {st.action_string(seat, x): x for x in st.legal_actions(seat)}
        if len(moves) >= a.max_plies:
            termination = "max_plies"
            break
        sample = len(moves) < a.sample_plies
        best, lines = engine.search([to_engine_move(m) for m in moves], a.nodes, a.multipv if sample else 1)
        choice = from_engine_move(best)
        if sample and len(lines) > 1:
            top = lines[0][1]
            cands = [(from_engine_move(mv), cp) for mv, cp in lines if top - cp <= a.window]
            w = [math.exp((cp - top) / a.temperature) for _, cp in cands]
            choice = rng.choices([c for c, _ in cands], weights=w)[0]
        if choice not in legal:
            termination = "engine_move_illegal"
            break
        score = lines[0][1] if lines else 0
        losing[seat] = losing[seat] + 1 if score <= -a.resign_cp else 0
        if losing[seat] >= a.resign_plies:
            termination = "resign"
            return moves, ("0-1" if seat == 0 else "1-0"), termination
        st.apply_action(seat, legal[choice])
        moves.append(choice)
    ret = st.returns() if st.turn()[0] != "act" else [0.0, 0.0]
    result = "1-0" if ret[0] > ret[1] else "0-1" if ret[1] > ret[0] else "1/2-1/2"
    return moves, result, termination


def pgn(moves: List[str], result: str, termination: str, meta: dict) -> str:
    head = [f'[Event "{meta["event"]}"]', f'[Site "{meta["site"]}"]', f'[Round "{meta["index"]}"]',
            '[Red "Pikafish"]', '[Black "Pikafish"]', f'[Result "{result}"]', '[Format "UCI"]',
            f'[Termination "{termination}"]', f'[Annotator "{meta["annotator"]}"]']
    body = []
    for i in range(0, len(moves), 2):
        body.append(f"{i // 2 + 1}. " + " ".join(moves[i:i + 2]))
    return "\n".join(head) + "\n\n" + " ".join(body) + f" {result}\n"


_E: dict = {}


def _init(a) -> None:
    _E["engine"] = Uci(a.engine, {"Threads": 1, "Hash": a.hash})
    _E["args"] = a


def _part(job) -> Tuple[str, int, dict, float]:
    out, part, lo, hi = job
    a = _E["args"]
    path = Path(out) / f"part-{part:05d}.pgn"
    t0 = time.time()
    stats: dict = {}
    texts = []
    for i in range(lo, hi):
        rng = random.Random(f"selfplay:{a.game}:{a.seed}:{i}")
        moves, result, term = play_xiangqi(_E["engine"], rng, a)
        stats[term] = stats.get(term, 0) + 1
        stats[result] = stats.get(result, 0) + 1
        texts.append(pgn(moves, result, term, {"event": f"Pikafish self-play nodes {a.nodes}", "site": "GamingMaster",
                                              "index": i, "annotator": f"engine_selfplay.py seed {a.seed}"}))
    tmp = path.with_suffix(".pgn.part")
    tmp.write_text("\n".join(texts))
    tmp.replace(path)
    return str(path), hi - lo, stats, time.time() - t0


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game", default="xiangqi", choices=("xiangqi",))
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--part_size", type=int, default=50)
    ap.add_argument("--nodes", type=int, default=50000)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_plies", type=int, default=12)
    ap.add_argument("--multipv", type=int, default=4)
    ap.add_argument("--window", type=int, default=40, help="centipawns below the best a sampled move may be")
    ap.add_argument("--temperature", type=float, default=20.0)
    ap.add_argument("--resign_cp", type=int, default=1500)
    ap.add_argument("--resign_plies", type=int, default=8)
    ap.add_argument("--max_plies", type=int, default=300)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for lo in range(a.first, a.first + a.games, a.part_size):
        part = lo // a.part_size
        if (out / f"part-{part:05d}.pgn").exists():
            continue
        jobs.append((str(out), part, lo, min(lo + a.part_size, a.first + a.games)))
    t0, done, tot = time.time(), 0, {}
    if a.workers <= 1:
        _init(a)
        results = map(_part, jobs)
    else:
        pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a,))
        results = pool.imap_unordered(_part, jobs)
    for path, n, stats, dt in results:
        done += n
        for k, v in stats.items():
            tot[k] = tot.get(k, 0) + v
        print(json.dumps({"part": path, "games": n, "seconds": round(dt, 1), "total": done,
                          "games_per_s": round(done / max(time.time() - t0, 1e-9), 3)}), flush=True)
    print(json.dumps({"out": str(out), "games": done, "stats": tot, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
