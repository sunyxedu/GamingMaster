"""Decision quality by rating: sample positions of rated human games per (rating bucket,
speed), score the move played against a UCI engine's best, and tabulate — the evidence behind
a rating threshold (docs/hq-data-and-simulators.md, principle 1: measure decision quality
directly where an engine exists).

Chess, Lichess's database (Hugging Face ``Lichess/standard-chess-games`` parquet shards:
WhiteElo / BlackElo / TimeControl / movetext).  A game's bucket is the LOWER of the two
ratings (both players at least that), its speed Lichess's own (estimated duration base +
40 × increment: < 29 s ultrabullet, < 180 bullet, < 480 blitz, < 1500 rapid, else classical).
Games with a BOT account (Lichess's ``BOT`` title: engines play rated games there too) are
left out and counted — at the top of the blitz / rapid / classical ladders they are most of
the games.  Per position: the engine's evaluation of the best move and of the move played (the position
after it, from the mover's side), both at ``--nodes``; the loss in centipawns (mates at
±1000, capped at 1000) and in Lichess's win probability (their accuracy model: win% = 50 + 50
· (2 / (1 + exp(−0.00368208 · cp)) − 1)); a blunder = a win-probability loss ≥ 30 points
(Lichess's threshold), a mistake ≥ 20.

    uv run --with chess --with pyarrow python scripts/quality_by_rating.py \\
        --parquet train-00000-of-00065.parquet --engine ~/gm-engines/bin/stockfish \\
        --workers 16 --out quality_chess.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import random
import re
import sys

BUCKETS = (1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800)
SPEEDS = ("ultrabullet", "bullet", "blitz", "rapid", "classical")


def speed_of(tc: str) -> str:
    if not tc or tc == "-" or "+" not in tc:
        return "correspondence"
    base, inc = tc.split("+", 1)
    try:
        t = int(base) + 40 * int(inc)
    except ValueError:
        return "correspondence"
    return "ultrabullet" if t < 29 else "bullet" if t < 180 else "blitz" if t < 480 else "rapid" if t < 1500 else "classical"


def bucket_of(r: int) -> int:
    b = BUCKETS[0]
    for x in BUCKETS:
        if r >= x:
            b = x
    return b


def moves_of(movetext: str):
    """SAN moves of a Lichess movetext (comments, clocks, evals and move numbers dropped)."""
    text = re.sub(r"\{[^}]*\}", " ", movetext)
    text = re.sub(r"\([^)]*\)", " ", text)
    out = []
    for tok in text.split():
        if re.fullmatch(r"\d+\.+", tok) or tok in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        tok = re.sub(r"^\d+\.+", "", tok)
        tok = tok.rstrip("!?")
        if tok:
            out.append(tok)
    return out


def win_pct(cp: float) -> float:
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


_ENGINE = None


def _init(engine_path: str, nodes: int):
    global _ENGINE, _NODES
    import chess.engine

    # a slow start on a loaded node (the default 10 s timeout is not enough there)
    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=300)
    _ENGINE.configure({"Threads": 1, "Hash": 64})
    _NODES = nodes


def _score(board) -> float:
    import chess.engine

    info = _ENGINE.analyse(board, chess.engine.Limit(nodes=_NODES))
    s = info["score"].pov(board.turn)
    return float(max(-1000, min(1000, s.score(mate_score=1000))))


def _eval(job):
    """(key, cp loss, win% loss) of one position: the best move's value vs the move played."""
    import chess

    key, fen, uci = job
    board = chess.Board(fen)
    best = _score(board)
    board.push(chess.Move.from_uci(uci))
    if board.is_checkmate():
        played = 1000.0
    elif board.is_game_over():
        played = 0.0
    else:
        played = -_score(board)
    loss = max(0.0, best - played)
    return key, min(loss, 1000.0), max(0.0, win_pct(best) - win_pct(played))


def collect(paths, per_cell: int, per_game: int, seed: int, keep_bots: bool = False):
    import chess
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    games = collections.defaultdict(list)
    seen = collections.Counter()
    bots = collections.Counter()
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["WhiteElo", "BlackElo", "TimeControl", "movetext", "Termination",
                                               "WhiteTitle", "BlackTitle"])
            for r in t.to_pylist():
                try:
                    lo = min(int(r["WhiteElo"]), int(r["BlackElo"]))
                except (TypeError, ValueError):
                    continue
                key = (bucket_of(lo), speed_of(r["TimeControl"]))
                if key[1] not in SPEEDS:
                    continue
                if "BOT" in (r.get("WhiteTitle"), r.get("BlackTitle")):
                    bots[key] += 1
                    if not keep_bots:
                        continue
                seen[key] += 1
                # reservoir sample per cell
                cell = games[key]
                if len(cell) < per_cell:
                    cell.append(r["movetext"])
                else:
                    j = rng.randrange(seen[key])
                    if j < per_cell:
                        cell[j] = r["movetext"]
    jobs = []
    for key, texts in games.items():
        for mt in texts:
            sans = moves_of(mt)
            if len(sans) < 20:
                continue
            board = chess.Board()
            pos = []
            try:
                for i, san in enumerate(sans):
                    mv = board.parse_san(san)
                    if 12 <= i < len(sans) - 4:
                        pos.append((board.fen(), mv.uci()))
                    board.push(mv)
            except ValueError:
                continue
            for fen, uci in rng.sample(pos, min(per_game, len(pos))):
                jobs.append((f"{key[0]}|{key[1]}", fen, uci))
    return jobs, seen, bots


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--parquet", action="append", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--nodes", type=int, default=200_000)
    p.add_argument("--per-cell", type=int, default=150, help="games sampled per (bucket, speed)")
    p.add_argument("--per-game", type=int, default=4, help="positions per game")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-bots", action="store_true", help="keep games with a BOT account")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    jobs, seen, bots = collect(a.parquet, a.per_cell, a.per_game, a.seed, a.keep_bots)
    print(f"[quality] {len(jobs)} positions from {sum(seen.values())} games", file=sys.stderr)
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0])
    with mp.Pool(a.workers, initializer=_init, initargs=(a.engine, a.nodes)) as pool:
        for i, (key, cp, wp) in enumerate(pool.imap_unordered(_eval, jobs, chunksize=8)):
            c = agg[key]
            c[0] += 1
            c[1] += cp
            c[2] += wp
            c[3] += wp >= 30.0
            c[4] += wp >= 20.0
            if i % 2000 == 0:
                print(f"[quality] {i}/{len(jobs)}", file=sys.stderr)
    table = {}
    for key, (n, cp, wp, blunders, mistakes) in sorted(agg.items()):
        b, s = key.split("|")
        table.setdefault(s, {})[b] = {"positions": n, "acpl": round(cp / n, 1), "win_loss": round(wp / n, 2),
                                      "blunder_rate": round(blunders / n, 4), "mistake_rate": round(mistakes / n, 4),
                                      "games_in_data": seen[(int(b), s)], "bot_games": bots[(int(b), s)]}
    out = {"engine": a.engine, "nodes": a.nodes, "parquet": a.parquet, "keep_bots": a.keep_bots, "table": table}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for s in SPEEDS:
        if s in table:
            row = " ".join(f"{b}:{v['acpl']}/{v['blunder_rate']:.3f}({v['positions']}; bots {v['bot_games']}/{v['games_in_data'] + v['bot_games']})"
                           for b, v in sorted(table[s].items(), key=lambda kv: int(kv[0])))
            print(s, row)


if __name__ == "__main__":
    main()
