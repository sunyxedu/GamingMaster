"""Full-strength engines judge every learner decision of archived evaluation episodes.

train_rl_joint (--trace_every) archives each evaluation wave's episodes as gmrules trajectories,
``<run>/traces/eval_<game>_<opponent>_it<N>_r<rank>_w<wave>.npz``.  This script replays them and
asks a strong engine, at every position where the learner moved, for the value of the best move
and of the move played: the learner's **loss per decision** against the engine — centipawns from
the mover's side for chess / xiangqi / shogi (UCI / USI engines, mate scores capped), winrate
points for go (KataGo's analysis engine).  A policy far weaker than the engine (win rate 0) still
gets a fine-grained number, and the number keeps meaning as the policy improves — the referee
win rates of the run do not (a referee is calibrated to the start and steps up once beaten).

    uv run python scripts/analyze_strength.py --run ~/gm-runs/rl/final-main \\
        --engine chess=~/gm-engines/bin/stockfish --go "chess=go depth 16" \\
        --engine xiangqi=~/gm-engines/bin/pikafish --go "xiangqi=go depth 14" \\
        --engine shogi=~/gm-engines/bin/yaneuraou --go "shogi=go depth 14" \\
        --engine "go=~/gm-engines/katago/v1.18.1-eigenavx2/katago analysis -model ... -config ... \\
                  -override-config reportAnalysisWinratesAs=SIDETOMOVE" \\
        --go "go=visits 200" --max_positions 300 --threads 8

Appends one record per archive file to ``<run>/strength.jsonl`` (files already judged are
skipped): game, opponent, iteration, positions judged, mean / median loss, the share of moves
that were the engine's best, the engine and its limit — and in games against the --init network
the same for the init network's moves (``init_*``): the policy and its starting point judged by
the same engine in the same games.  ``--summary`` prints the per-game trend.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.engine_policy import GTP_RULES, TO_ENGINE, EngineError, UciEngine, start_string  # noqa: E402

UCI_GAMES = {"chess": "uci", "xiangqi": "uci", "shogi": "usi", "chess_variants": "uci"}
MATE_CP = 1000          # a mate score as centipawns (capped loss scale)
LOSS_CAP = 1000


# ---------------------------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------------------------

class ScoringUci(UciEngine):
    """A UCI / USI engine asked for scores: the best move's and a given move's (``searchmoves``),
    both from the side to move, in centipawns (mate N → ±(MATE_CP − N))."""

    _SCORE = re.compile(r"\bscore (cp|mate) (-?\d+)")

    def _score(self, lines: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
        best: Optional[str] = None
        score: Optional[float] = None
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                best = parts[1] if len(parts) > 1 else None
            elif line.startswith("info"):
                m = self._SCORE.search(line)
                if m:
                    v = int(m.group(2))
                    score = float(v) if m.group(1) == "cp" else float(
                        (MATE_CP - abs(v)) * (1 if v > 0 else -1))
        return score, best

    def judge(self, start: str, moves: Sequence[str], played: str, go: str) -> Tuple[float, float, bool]:
        """(best score, played move's score, played == best) at the position after ``moves``."""
        kind = "sfen" if self.protocol == "usi" else "fen"
        pos = f"position {kind} {start}" + (" moves " + " ".join(moves) if moves else "")
        self._send(pos)
        self._send(go)
        best_score, best = self._score(self._until("bestmove"))
        if best_score is None or best is None:
            raise EngineError(f"{self.name}: no score for the best move")
        if best == played:
            return best_score, best_score, True
        self._send(pos)
        self._send(f"{go} searchmoves {played}")
        played_score, _ = self._score(self._until("bestmove"))
        if played_score is None:
            raise EngineError(f"{self.name}: no score for {played!r}")
        return best_score, played_score, False


class KataGoAnalysis:
    """KataGo's analysis engine (``katago analysis -model ... -config ...``): the root winrate at
    given turns of a game.  Start it with ``-override-config reportAnalysisWinratesAs=SIDETOMOVE``
    (analysis_example.cfg reports from Black's side): the loss below takes the winrate as the
    mover's."""

    def __init__(self, command: str, visits: int, rules: str = GTP_RULES["go"], komi: float = 7.5,
                 size: int = 19, timeout: float = 600.0):
        argv = [os.path.expanduser(a) for a in shlex.split(command)]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.visits, self.rules, self.komi, self.size, self.timeout = visits, rules, komi, size, timeout
        self.name = f"katago-analysis visits={visits}"
        self._n = 0

    def winrates(self, moves: Sequence[str], turns: Sequence[int]) -> Dict[int, float]:
        """turn → winrate of the player to move at that turn (after ``moves[:turn]``)."""
        self._n += 1
        colours = ("B", "W")
        q = {"id": f"q{self._n}", "moves": [[colours[i % 2], m] for i, m in enumerate(moves)],
             "rules": self.rules, "komi": self.komi, "boardXSize": self.size, "boardYSize": self.size,
             "analyzeTurns": list(turns), "maxVisits": self.visits, "includePolicy": False}
        self.proc.stdin.write(json.dumps(q) + "\n")
        self.proc.stdin.flush()
        out: Dict[int, float] = {}
        timer = threading.Timer(self.timeout, self.proc.kill)
        timer.start()
        try:
            while len(out) < len(turns):
                line = self.proc.stdout.readline()
                if not line:
                    raise EngineError(f"{self.name} exited ({self.proc.poll()})")
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("id") != q["id"]:
                    continue
                if "error" in r:
                    raise EngineError(f"{self.name}: {r['error']}")
                out[int(r["turnNumber"])] = float(r["rootInfo"]["winrate"])
        finally:
            timer.cancel()
        return out

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:   # noqa: BLE001
                self.proc.kill()


# ---------------------------------------------------------------------------------------------
# replay + judging
# ---------------------------------------------------------------------------------------------

def decisions(game: str, traj: np.ndarray, learner_seats: Sequence[bool]) -> List[Tuple[int, List[str], str]]:
    """The learner's decisions of one episode: (turn, moves before it, the move played) with the
    moves in the engine's notation."""
    import gmsim._native as gm
    to = TO_ENGINE.get(game)
    st = gm.State(game)
    moves: List[str] = []
    out = []
    for seat, a in np.asarray(traj).reshape(-1, 2).tolist():
        if seat < 0:
            st.apply_chance(int(a))
            continue
        m = st.action_string(int(seat), int(a))
        m = to(m) if to else m
        if learner_seats[seat]:
            out.append((len(moves), list(moves), m))
        moves.append(m)
        st.apply_action(int(seat), int(a))
    return out


ARCHIVE = re.compile(r"^eval_(?P<game>.+)_(?P<opponent>[^_]+)_it(?P<iteration>\d+)_r\d+_w\d+\.npz$")


def archive_game(base: str) -> Optional[str]:
    """The game of an archive file name (``eval_<game>_<opponent>_it<N>_r<rank>_w<wave>.npz``;
    game names hold underscores: ``eval_chess_variants_init_it40_r0_w0.npz`` is chess_variants)."""
    m = ARCHIVE.match(base)
    return m.group("game") if m else None


def sample(items: List[Any], k: int, seed: int = 0) -> List[Any]:
    if k <= 0 or len(items) <= k:
        return items
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), size=k, replace=False))
    return [items[i] for i in idx]


def judge_file(path: str, engine: Any, game: str, go: str, max_positions: int, verbose: bool = False) -> Dict[str, Any]:
    """One archive: the learner's decisions judged; in a game against the --init network also
    the init network's decisions of the same games (``init_*`` fields) — a paired comparison of
    the policy with its starting point under the same engine, positions and opponents."""
    z = np.load(path)
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    seats = z["learner_seats"]
    n = seats.shape[0]
    rec = {"file": os.path.basename(path), "game": game, "opponent": meta.get("opponent"),
           "iteration": meta.get("iteration"), "rank": meta.get("rank"), "wave": meta.get("wave"),
           "episodes": n, "engine": getattr(engine, "name", "?"), "go": go}
    sides = [("", lambda row: [bool(x) for x in row])]
    if meta.get("opponent") == "init":
        sides.append(("init_", lambda row: [not bool(x) for x in row]))
    for prefix, pick in sides:
        todo: List[Tuple[int, int, List[str], str]] = []
        for e in range(n):
            for turn, before, played in decisions(game, z[f"traj{e}"], pick(seats[e])):
                todo.append((e, turn, before, played))
        todo = sample(todo, max_positions, seed=int(meta.get("iteration", 0)) + len(prefix))
        losses, agree = _judge(z, seats.shape[1], todo, engine, game, go)
        rec.update({f"{prefix}positions": len(losses),
                    f"{prefix}mean_loss": statistics.fmean(losses) if losses else None,
                    f"{prefix}median_loss": statistics.median(losses) if losses else None,
                    f"{prefix}agree": (agree / len(losses)) if losses and agree >= 0 else None})
    if verbose:
        print(json.dumps(rec), flush=True)
    return rec


def _judge(z: Any, n_seats: int, todo: List[Tuple[int, int, List[str], str]], engine: Any, game: str,
           go: str) -> Tuple[List[float], int]:
    """(losses, best-move count; -1 when not measured) of the decisions ``todo``."""
    losses: List[float] = []
    agree = 0
    if game in UCI_GAMES:
        start = start_string(game)
        for e, turn, before, played in todo:
            best, got, same = engine.judge(start, before, played, go)
            losses.append(min(max(best - got, 0.0), LOSS_CAP))
            agree += int(same)
    elif game == "go":
        # the loss as winrate points: the mover's winrate before the move minus their winrate
        # after it (the opponent's root winrate at the next turn, inverted)
        by_episode: Dict[int, List[Tuple[int, List[str], str]]] = {}
        for e, turn, before, played in todo:
            by_episode.setdefault(e, []).append((turn, before, played))
        for e, items in by_episode.items():
            full = z[f"traj{e}"]
            all_moves = decisions(game, full, [True] * n_seats)
            moves = [m for _, _, m in all_moves]
            turns = sorted({t for t, _, _ in items} | {t + 1 for t, _, _ in items})
            wr = engine.winrates(moves, [t for t in turns if t <= len(moves)])
            for turn, _, _ in items:
                if turn in wr and turn + 1 in wr:
                    losses.append(min(max(wr[turn] - (1.0 - wr[turn + 1]), 0.0), 1.0) * 100)
                elif turn in wr:            # the game ended with this move: no loss measured
                    continue
        agree = -1                          # not measured for go (no single best move queried)
    else:
        raise SystemExit(f"{game}: no engine protocol known")
    return losses, agree


def make_engine(game: str, command: str, go: str, threads: int):
    if game in UCI_GAMES:
        proto = UCI_GAMES[game]
        opts = {"Threads": str(threads)} if threads > 1 else {}
        if proto == "usi":
            opts = {"Threads": str(threads), "USI_Hash": "256"} if threads > 1 else {"USI_Hash": "256"}
        else:
            opts["Hash"] = "256"
        return ScoringUci(command, proto, opts, timeout=600.0)
    if game == "go":
        m = re.search(r"visits\s+(\d+)", go)
        return KataGoAnalysis(command, int(m.group(1)) if m else 100)
    raise SystemExit(f"{game}: no engine protocol known")


def summary(run: str) -> None:
    path = os.path.join(run, "strength.jsonl")
    if not os.path.exists(path):
        print("no strength.jsonl yet")
        return
    rows = [json.loads(l) for l in open(path)]
    by: Dict[Tuple[str, str], Dict[int, List[Dict[str, Any]]]] = {}
    for r in rows:
        by.setdefault((r["game"], r.get("opponent") or "?"), {}).setdefault(int(r["iteration"]), []).append(r)
    for (g, opp), its in sorted(by.items()):
        parts = []
        for it in sorted(its):
            rs = [r for r in its[it] if r["positions"]]
            if not rs:
                continue
            n = sum(r["positions"] for r in rs)
            loss = sum(r["mean_loss"] * r["positions"] for r in rs) / n
            ag = [(r["agree"], r["positions"]) for r in rs if r["agree"] is not None]
            agree = (sum(a * k for a, k in ag) / sum(k for _, k in ag)) if ag else None
            ri = [r for r in rs if r.get("init_positions")]
            ni = sum(r["init_positions"] for r in ri)
            init = (f" | init {sum(r['init_mean_loss'] * r['init_positions'] for r in ri) / ni:.1f}"
                    if ni else "")
            parts.append(f"@{it}: loss {loss:.1f}" + (f" agree {agree:.2f}" if agree is not None else "")
                         + init + f" (n={n})")
        unit = "winrate pts" if g == "go" else "cp"
        print(f"{g:14s} vs {opp:6s} [{unit}] " + "  ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--engine", action="append", default=[], metavar="GAME=COMMAND")
    ap.add_argument("--go", action="append", default=[], metavar="GAME=LIMIT",
                    help="UCI/USI: the go command (e.g. 'go depth 16'); go: 'visits N'")
    ap.add_argument("--max_positions", type=int, default=300, help="per archive file and side (sampled evenly)")
    ap.add_argument("--max_positions_game", action="append", default=[], metavar="GAME=N",
                    help="a game's own --max_positions (go's engine is the slow one)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated games")
    ap.add_argument("--summary", action="store_true", help="print the trend and exit")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(a.run)
        return
    engines = dict(x.split("=", 1) for x in a.engine)
    limits = dict(x.split("=", 1) for x in a.go)
    budget = {g: int(v) for g, v in (x.split("=", 1) for x in a.max_positions_game)}
    out_path = os.path.join(a.run, "strength.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["file"] for l in open(out_path)}
    files = sorted(glob.glob(os.path.join(a.run, "traces", "eval_*_it*_r*_w*.npz")),
                   key=lambda p: (int(re.search(r"_it(\d+)_", p).group(1)), p))
    only = set(a.only.split(",")) if a.only else None
    live: Dict[str, Any] = {}
    dead: Dict[str, str] = {}       # games whose engine does not start (e.g. an AVX2 build on an
    n_new = 0                       # SSE2-only node): skipped, their files stay unjudged
    try:
        for path in files:
            base = os.path.basename(path)
            if base in done:
                continue
            game = archive_game(base)
            if game not in engines or game in dead or (only and game not in only):
                continue
            if game not in live:
                try:
                    live[game] = make_engine(game, engines[game], limits.get(game, ""), a.threads)
                except (EngineError, OSError) as e:
                    dead[game] = f"{type(e).__name__}: {e}"
                    print(f"{game}: the engine does not start ({dead[game][:300]}) — skipped",
                          file=sys.stderr, flush=True)
                    continue
            try:
                rec = judge_file(path, live[game], game, limits.get(game, ""),
                                 budget.get(game, a.max_positions), a.verbose)
            except EngineError as e:
                print(f"{base}: {e}", file=sys.stderr, flush=True)
                eng = live.pop(game, None)
                if eng is not None:
                    eng.close()
                continue
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_new += 1
    finally:
        for eng in live.values():
            try:
                eng.close()
            except Exception:   # noqa: BLE001
                pass
    print(f"judged {n_new} new archive(s) -> {out_path}")
    summary(a.run)


if __name__ == "__main__":
    main()
