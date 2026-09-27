"""Othello arena: the model's pure policy against Egaroucid for Console at
several levels (training/arena.py is the game-agnostic runner).

    scripts/engines/fetch_engines.sh egaroucid          # once per machine
    uv run python -m training.arena othello --ckpt ~/gm-runs/multi/RUN/multi_final.pt \\
        --levels 1 3 5 7 9 11 13 15 --openings 100 --device cuda --out arena_othello.json

Rules and frames: training/othello_probe.py — the bitboard rules and the
training-frame semantics of ``gmdata othello convert`` (pinned bit-for-bit to
the converter by tests/test_dial_probe.py): one stream per game holding
every position BEFORE a half-move, from the side to move (planes: mover /
opponent / empty), both colours alternating; a side without a legal move
while the opponent has one gets a PASS frame (the rules' forced move: the
model never decides it) and the stream continues with the opponent's frame of
the same board; the terminal position is never a frame.  The model moves on
its decision frames with the legal-masked place logits (``--temperature``);
the pass head is never read.

Engine: Egaroucid for Console (GPL-3.0, a separate process; built by
``scripts/engines/fetch_engines.sh egaroucid`` into ``~/gm-engines`` — the
patched build with a fixed RNG seed, so every process searches identically).
``--engine`` (default ``$GM_EGAROUCID``, else ``$GM_ENGINES/bin/egaroucid``,
else ``~/gm-engines/bin/egaroucid``) is started once per worker as

    <engine> [--engine_arg ...] -quiet -thread 1 -hash 20 -nobook

(a raw binary next to its ``resources/`` directory runs in that directory,
like the fetch_engines.sh wrapper) and speaks the console protocol the
labeller uses (data_processing/rs/gmdata/src/label/othello.rs): per move

    level <L>                     Egaroucid's own level table (``levelinfo``):
                                  depth-L midgame search, its endgame schedule;
                                  sent when the process's level changes
    clearcache                    a fresh transposition table: the move depends
                                  on the position and level only
    setboard <64 x X/O/-><X|O>    a1 b1 ... h8 row-major (square row*8 + col, the
                                  tgt.place_idx encoding), black X, side to move
    go                            Egaroucid plays: one line with its move ("f5")
    gm-sync-<n>                   unknown command -> "[ERROR] command `gm-sync-<n>`
                                  not found" on stdout: the end-of-answer marker

(prompts and warnings go to stderr, drained and kept for error messages).
One thread, no book, a cache cleared before every move and the patched RNG
make every answer a function of (position, level): games are reproducible and
a restarted process plays on identically.  A request that times out
(``--engine_timeout``), dies, or answers garbage / an illegal move restarts
its process and is asked again (``--engine_retries``); after that the game is
recorded as failed and its opening pair is left out.  Egaroucid is never asked
in a forced position (a pass or a single legal move).  The processes serve
every level (a pool of ``--workers``, ~0.8 GB resident each).

Calibration: unless ``--no_calibrate``, the arena also plays Egaroucid against
itself between adjacent levels of ``--levels`` (``--calibration all``: every
pair) on the same openings, so the ladder fit places every level — and the
model — on one Elo scale (anchor: ``--anchor``, default the lowest level at
0) and reports the model's equivalent (fractional) level.  These games do not
depend on the model: they are cached in ``--ladder_cache`` (JSONL keyed by the
engine build, its settings and the openings) and reused by every later run.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import os
import queue
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from training import arena
from training import othello_probe as op
from training.othello_probe import BLACK, PASS

State = Tuple[int, int, int]                   # (black bitboard, white bitboard, mover)

_SQUARE = re.compile(r"^[a-h][1-8]$")


def square_name(sq: int) -> str:
    """Square ``row*8 + col`` → "f5" (column letter, row number from 1)."""
    return "abcdefgh"[sq % 8] + str(sq // 8 + 1)


def parse_square(name: str) -> int:
    s = name.strip().lower()
    if not _SQUARE.match(s):
        raise ValueError(f"not a square: {name!r}")
    return (int(s[1]) - 1) * 8 + "abcdefgh".index(s[0])


# ---------------------------------------------------------------------------
# Rules and frames (training/othello_probe.py semantics)
# ---------------------------------------------------------------------------

class OthelloRules:
    """Othello for the arena on othello_probe's bitboards.  State = (black,
    white, mover); moves = squares 0..63 and PASS (-1), the single legal
    move of a side that cannot place while the opponent can.  Seat 0 =
    black (moves first), seat 1 = white; a game is over when neither side
    can move, the side with more discs wins (empties never change the
    sign)."""

    name = "othello"
    n_seats = 2
    seat_names = ("black", "white")
    max_plies = 200                             # 60 placements + passes, with room

    def initial_state(self) -> State:
        return (op.INITIAL_BLACK, op.INITIAL_WHITE, BLACK)

    def seat_to_move(self, s: State) -> int:
        return 0 if s[2] == BLACK else 1

    def legal_moves(self, s: State) -> List[int]:
        own, opp = op.sides(*s)
        moves = op.legal_moves(own, opp)
        if moves:
            return list(op.iter_bits(moves))
        return [PASS] if op.legal_moves(opp, own) else []

    def play(self, s: State, move: int) -> State:
        b, w, p = s
        own, opp = op.sides(b, w, p)
        if move == PASS:
            if op.legal_moves(own, opp) or not op.legal_moves(opp, own):
                raise ValueError("a pass is only legal without a legal placement "
                                 "while the opponent has one")
            return (b, w, op.other(p))
        b2, w2 = op.colours(*op.play(own, opp, int(move)), p)
        return (b2, w2, op.other(p))

    def is_terminal(self, s: State) -> bool:
        return op.is_game_over(s[0], s[1])

    def seat_scores(self, s: State) -> List[float]:
        z = op.outcome_sign(s[0], s[1])             # black's view
        return [0.5 * (1 + z), 0.5 * (1 - z)]

    def position_key(self, s: State) -> State:
        return s

    def move_str(self, m: int) -> str:
        return "pass" if m == PASS else square_name(int(m))

    def margin(self, s: State, seat: int) -> float:
        """Final disc difference of ``seat``, empties to the winner."""
        own, opp = (s[0], s[1]) if seat == 0 else (s[1], s[0])
        return float(op.final_diff(own, opp))

    def details(self, s: State) -> Dict[str, Any]:
        return {"discs": [s[0].bit_count(), s[1].bit_count()]}


class OthelloCodec:
    """Frames and decisions of the othello network (BoardEncoder on 3 planes
    + PlacementDecoder): obs ``board_seq`` [B, 3, 8, 8] = own / opponent /
    empty planes of the side to move (othello_probe.planes_batch, the
    converter's layout); a decision scores the legal squares by their place
    logits.  No legal mask is fed (as training/othello_probe.py's evaluator:
    the placement decoder's logits do not depend on it)."""

    layout = "interleaved"
    output_keys = ("place_logits",)

    def encode(self, states: Sequence[State]) -> Dict[str, torch.Tensor]:
        return {"board_seq": torch.from_numpy(op.planes_batch([op.sides(*s) for s in states]))}

    def legal(self, states: Sequence[State]) -> None:
        return None

    def move_scores(self, out, row: int, state: State, legal: Sequence[int]):
        logits = out["place_logits"][row]
        cands = [int(m) for m in legal if m != PASS]
        if not cands:
            raise ValueError("a decision frame without a legal placement")
        top1 = int(np.argmax(logits))
        return cands, logits[cands], top1 in cands


def check_othello_network(net) -> None:
    dec = type(getattr(net, "action_decoder", None)).__name__
    cfg = getattr(net, "cfg", None)
    if (dec != "PlacementDecoder" or getattr(cfg, "piece_channels", None) != 3
            or (getattr(cfg, "board_h", 8), getattr(cfg, "board_w", 8)) != (8, 8)):
        raise SystemExit(f"not an othello network (BoardEncoder on 3 planes 8x8 + "
                         f"PlacementDecoder): decoder {dec}, "
                         f"piece_channels {getattr(cfg, 'piece_channels', None)}")


# ---------------------------------------------------------------------------
# Egaroucid for Console
# ---------------------------------------------------------------------------

class EngineError(RuntimeError):
    """The engine timed out, died or answered something unusable."""


def default_engine() -> str:
    """``$GM_EGAROUCID``, else ``$GM_ENGINES/bin/egaroucid``, else
    ``~/gm-engines/bin/egaroucid`` (fetch_engines.sh's layout)."""
    return os.environ.get("GM_EGAROUCID") or os.path.join(
        os.environ.get("GM_ENGINES", os.path.expanduser("~/gm-engines")), "bin", "egaroucid")


def egaroucid_board(s: State) -> str:
    """The ``setboard`` argument: 64 cells a1 ... h8 (X black, O white, -
    empty) and the side to move."""
    b, w, p = s
    cells = ["X" if b >> i & 1 else "O" if w >> i & 1 else "-" for i in range(64)]
    return "".join(cells) + ("X" if p == BLACK else "O")


def egaroucid_command(engine: str, engine_args: Sequence[str] = (), hash_level: int = 20,
                      threads: int = 1, book: bool = False) -> Tuple[List[str], Optional[str]]:
    """(argv, working directory) of one engine process."""
    cmd = [engine, *engine_args, "-quiet", "-thread", str(int(threads)),
           "-hash", str(int(hash_level))]
    if not book:
        cmd.append("-nobook")
    parent = os.path.dirname(os.path.abspath(engine))
    cwd = parent if os.path.isdir(os.path.join(parent, "resources")) else None
    return cmd, cwd


class EgaroucidProcess:
    """One Egaroucid for Console process spoken to over stdin / stdout (see
    the module docstring).  Not thread-safe: one caller at a time (the pool
    checks processes out)."""

    def __init__(self, command: Sequence[str], cwd: Optional[str] = None,
                 startup_timeout: float = 120.0):
        self.command = list(command)
        self.cwd = cwd
        self.startup_timeout = float(startup_timeout)
        self.proc: Optional[subprocess.Popen] = None
        self.version = ""                      # "Egaroucid for Console 7.8.0 ..." line
        self.build = ""                        # the "@build ..." line (identifies the binary)
        self.level: Optional[int] = None
        self._sync = 0
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._err = collections.deque(maxlen=64)
        self.start()

    # -- process ------------------------------------------------------------

    def start(self) -> None:
        self.kill()
        try:
            self.proc = subprocess.Popen(
                self.command, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                bufsize=1)
        except OSError as e:
            raise EngineError(f"cannot start {self.command[0]}: {e}") from e
        self._lines = queue.Queue()
        self._err = collections.deque(maxlen=64)
        self._sync = 0
        self.level = None
        threading.Thread(target=self._pump_out, args=(self.proc, self._lines),
                         daemon=True).start()
        threading.Thread(target=self._pump_err, args=(self.proc, self._err),
                         daemon=True).start()
        lines = self.exchange(["version"], self.startup_timeout)
        ver = [ln.strip() for ln in lines if ln.strip().startswith("Egaroucid")]
        if not ver:
            raise EngineError(f"{self.command[0]} did not answer `version` like Egaroucid: "
                              f"{lines[:5]}{self.stderr_tail()}")
        self.version = ver[0]
        self.build = next((ln.strip() for ln in lines if ln.strip().startswith("@build")), "")

    @staticmethod
    def _pump_out(proc: subprocess.Popen, q: "queue.Queue[Optional[str]]") -> None:
        try:
            for line in proc.stdout:
                q.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        q.put(None)                               # EOF: the process is gone

    @staticmethod
    def _pump_err(proc: subprocess.Popen, tail: collections.deque) -> None:
        # drain continuously (a full stderr pipe would block the engine; its
        # prompts carry no newline), keep the last chunks for error messages
        raw = proc.stderr.buffer
        try:
            while True:
                chunk = raw.read1(4096)
                if not chunk:
                    break
                tail.append(chunk.decode("utf-8", "replace"))
        except (OSError, ValueError):
            pass

    def stderr_tail(self, n: int = 400) -> str:
        t = "".join(self._err).replace("> ", "").strip()
        return f" (stderr: {t[-n:]!r})" if t else ""

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def kill(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        with contextlib.suppress(OSError):
            p.kill()
        with contextlib.suppress(Exception):
            p.wait(timeout=10)
        for f in (p.stdin, p.stdout, p.stderr):
            with contextlib.suppress(Exception):
                f.close()

    def close(self) -> None:
        p = self.proc
        if p is not None and p.poll() is None:
            with contextlib.suppress(Exception):
                p.stdin.write("quit\n")
                p.stdin.flush()
                p.wait(timeout=5)
        self.kill()

    # -- protocol -----------------------------------------------------------

    def exchange(self, cmds: Sequence[str], timeout: float) -> List[str]:
        """Send ``cmds`` and a sync marker; the stdout lines before the
        marker's echo.  EngineError on a timeout or a dead process."""
        if not self.alive():
            raise EngineError(f"engine not running{self.stderr_tail()}")
        self._sync += 1
        marker = f"gm-sync-{self._sync}"
        echo = f"[ERROR] command `{marker}` not found"
        try:
            self.proc.stdin.write("".join(f"{c}\n" for c in [*cmds, marker]))
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise EngineError(f"engine stdin closed: {e}{self.stderr_tail()}") from e
        deadline = time.monotonic() + timeout
        out: List[str] = []
        while True:
            left = deadline - time.monotonic()
            try:
                line = (self._lines.get(timeout=left) if left > 0
                        else self._lines.get_nowait())
            except queue.Empty:
                raise EngineError(f"no answer to {list(cmds)} within {timeout:.0f} s"
                                  f"{self.stderr_tail()}") from None
            if line is None:
                code = self.proc.poll() if self.proc is not None else None
                raise EngineError(f"engine exited (code {code}) during {list(cmds)}"
                                  f"{self.stderr_tail()}")
            if line.strip() == echo:
                return out
            out.append(line)

    def best_move(self, board: str, level: int, timeout: float) -> str:
        """Egaroucid's move at ``level`` in the ``setboard`` position (a
        square name)."""
        cmds = [] if self.level == level else [f"level {int(level)}"]
        cmds += ["clearcache", f"setboard {board}", "go"]
        lines = self.exchange(cmds, timeout)
        self.level = int(level)
        moves = [ln.strip().lower() for ln in lines if _SQUARE.match(ln.strip().lower())]
        if len(moves) != 1:
            raise EngineError(f"expected one move from `go`, got {lines[:6]}")
        return moves[0]


class EgaroucidPool:
    """``size`` identical engine processes shared by every level
    (thread-safe: a caller checks a process out for one request).  A failed
    request restarts its process and is retried ``retries`` times.
    ``stats``: requests, restarts, and failed requests (after the retries)."""

    def __init__(self, command: Sequence[str], cwd: Optional[str] = None, size: int = 1,
                 timeout: float = 120.0, retries: int = 2, startup_timeout: float = 120.0):
        self.command = list(command)
        self.cwd = cwd
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.startup_timeout = float(startup_timeout)
        self._lock = threading.Lock()
        self._free: "queue.Queue[EgaroucidProcess]" = queue.Queue()
        self.engines: List[EgaroucidProcess] = []
        self.stats = {"requests": 0, "restarts": 0, "failed": 0}
        try:
            for _ in range(max(1, int(size))):
                e = EgaroucidProcess(self.command, cwd, startup_timeout)
                self.engines.append(e)
                self._free.put(e)
        except BaseException:
            self.close()
            raise
        self.version = self.engines[0].version
        self.build = self.engines[0].build

    @property
    def identity(self) -> Dict[str, str]:
        return {"version": self.version, "build": self.build}

    @contextlib.contextmanager
    def _checkout(self):
        e = self._free.get()
        try:
            yield e
        finally:
            self._free.put(e)

    def best_move(self, board: str, level: int, legal: Sequence[str] = ()) -> str:
        """Egaroucid's move (square name) at ``level``; with ``legal`` (square
        names) an answer outside it counts as a failed request."""
        with self._lock:
            self.stats["requests"] += 1
        last: Optional[Exception] = None
        with self._checkout() as e:
            for attempt in range(self.retries + 1):
                try:
                    if not e.alive():
                        raise EngineError(f"engine not running{e.stderr_tail()}")
                    mv = e.best_move(board, level, self.timeout)
                    if legal and mv not in legal:
                        raise EngineError(f"level {level} played {mv}, illegal in {board} "
                                          f"(legal: {list(legal)})")
                    return mv
                except EngineError as err:
                    last = err
                    if attempt == self.retries:
                        break
                    with self._lock:
                        self.stats["restarts"] += 1
                    try:
                        e.start()
                    except EngineError as err2:
                        last = err2
                        break
        with self._lock:
            self.stats["failed"] += 1
        raise EngineError(f"{last} (after {self.retries} restart(s))")

    def close(self) -> None:
        for e in self.engines:
            e.close()


class EgaroucidOpponent:
    """Egaroucid at one level as an :class:`training.arena.Opponent`
    (stateless: the position is rebuilt from the move history)."""

    def __init__(self, pool: EgaroucidPool, level: int, rules: Optional[OthelloRules] = None):
        self.pool = pool
        self.level = int(level)
        self.rules = rules or OthelloRules()
        self.name = f"egaroucid-L{self.level}"

    def reset(self, gid: int) -> None:
        pass

    def act(self, gid: int, seat: int, history: Sequence[int]) -> int:
        s = arena.replay(self.rules, history)
        if self.rules.is_terminal(s):
            raise ValueError("asked to move in a finished game")
        if self.rules.seat_to_move(s) != seat:
            raise ValueError(f"asked for seat {seat}, seat {self.rules.seat_to_move(s)} to move")
        legal = self.rules.legal_moves(s)
        if legal == [PASS]:
            return PASS
        mv = self.pool.best_move(egaroucid_board(s), self.level,
                                 [square_name(m) for m in legal])
        return parse_square(mv)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Egaroucid")
    g.add_argument("--levels", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11, 13, 15],
                   help="Egaroucid levels to play (0-60; `levelinfo` of the engine)")
    g.add_argument("--engine", default=None,
                   help="Egaroucid for Console (default: $GM_EGAROUCID, else "
                        "$GM_ENGINES/bin/egaroucid, else ~/gm-engines/bin/egaroucid)")
    g.add_argument("--engine_arg", action="append", default=[], metavar="ARG",
                   help="extra argument before the arena's flags (repeatable)")
    g.add_argument("--engine_hash", type=int, default=20,
                   help="-hash level (2^n transposition-table entries)")
    g.add_argument("--engine_threads", type=int, default=1,
                   help="search threads per process (> 1 is not deterministic)")
    g.add_argument("--engine_timeout", type=float, default=120.0, help="seconds per move")
    g.add_argument("--engine_retries", type=int, default=2,
                   help="restarts of a failed request before its game counts as failed")
    g = p.add_argument_group("ladder calibration")
    g.add_argument("--no_calibrate", action="store_true",
                   help="no engine-vs-engine games between levels (the ladder fit then only "
                        "relates each level to the model)")
    g.add_argument("--calibration", choices=("adjacent", "all"), default="adjacent",
                   help="level pairs played for the calibration")
    g.add_argument("--ladder_cache", default=None,
                   help="JSONL cache of calibration games (default: "
                        "arena_othello_ladder.jsonl next to --out; '' disables)")
    g.add_argument("--anchor", action="append", default=[], metavar="LEVEL=ELO",
                   help="fixed ladder ratings (repeatable; default: the lowest level = 0)")


def opponent_name(level: int) -> str:
    return f"egaroucid-L{int(level)}"


def calibration_pairs(levels: Sequence[int], mode: str = "adjacent") -> List[Tuple[int, int]]:
    ls = sorted(set(int(x) for x in levels))
    if mode == "all":
        return [(a, b) for i, a in enumerate(ls) for b in ls[i + 1:]]
    return list(zip(ls, ls[1:]))


def parse_anchors(items: Sequence[str], levels: Sequence[int]) -> Dict[str, float]:
    if not items:
        return {opponent_name(min(levels)): 0.0}
    out = {}
    for it in items:
        lv, sep, elo = it.partition("=")
        if not sep:
            raise SystemExit(f"--anchor {it!r}: expected LEVEL=ELO")
        if int(lv) not in set(levels):
            raise SystemExit(f"--anchor level {lv} is not among --levels {list(levels)}")
        out[opponent_name(int(lv))] = float(elo)
    return out


def _openings_match(rec: arena.GameRecord, opening: Sequence[str]) -> bool:
    return rec.moves[:len(opening)] == list(opening)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    levels = sorted(set(int(x) for x in args.levels))
    if not levels or min(levels) < 0 or max(levels) > 60:
        raise SystemExit(f"--levels {args.levels}: Egaroucid levels are 0-60")
    engine = args.engine or default_engine()
    if not (os.path.isfile(engine) or os.path.sep not in engine):
        raise SystemExit(f"no Egaroucid at {engine}: run scripts/engines/fetch_engines.sh "
                         f"egaroucid, or pass --engine / set GM_EGAROUCID")
    rules = OthelloRules()
    openings = arena.random_openings(rules, args.openings, args.opening_plies,
                                     args.openings_seed)
    op_str = [[rules.move_str(m) for m in o] for o in openings]
    device = arena.pick_device(args.device)
    torch.manual_seed(args.seed)
    net, net_info = arena.load_policy_network(args.ckpt, args.game, device, raw=args.raw)
    check_othello_network(net)
    player = arena.NetworkPlayer(net, OthelloCodec(), device=device, depth=args.depth,
                                 temperature=args.temperature, seed=args.seed,
                                 bf16=args.bf16, name="model")
    if args.bf16 and not player.bf16:
        print(f"[warn] --bf16 ignored on {device.type} (CUDA only)", flush=True)
    cmd, cwd = egaroucid_command(engine, args.engine_arg, args.engine_hash,
                                 args.engine_threads, book=False)
    print(f"[arena] starting {args.workers} Egaroucid process(es): {' '.join(cmd)}", flush=True)
    pool = EgaroucidPool(cmd, cwd, size=args.workers, timeout=args.engine_timeout,
                         retries=args.engine_retries)
    try:
        return _run(args, t0, levels, rules, openings, op_str, net, net_info, player, pool,
                    engine, cmd)
    finally:
        pool.close()


def _run(args, t0, levels, rules, openings, op_str, net, net_info, player, pool, engine, cmd):
    engine_cfg = {"engine": pool.version, "build": pool.build, "hash": args.engine_hash,
                  "threads": args.engine_threads, "book": False, "cache": "cleared every move",
                  "extra_args": list(args.engine_arg)}
    openings_cfg = {"plies": args.opening_plies, "seed": args.openings_seed}
    model_cfg = {"checkpoint": arena.file_identity(args.ckpt), "game": args.game,
                 "weights": net_info["weights"], "depth": player.depth,
                 "temperature": args.temperature, "bf16": player.bf16,
                 "seed": args.seed if args.temperature > 0 else None}
    run_key = arena.config_key({"arena": "othello", "model": model_cfg, "engine": engine_cfg,
                                "openings": openings_cfg})
    cal_key = arena.config_key({"arena": "othello", "engine": engine_cfg,
                                "openings": openings_cfg})
    log_path = arena.games_log_path(args)
    run_log = arena.GameLog(log_path, run_key, fresh=not args.resume)
    cache_path = args.ladder_cache
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                  "arena_othello_ladder.jsonl")
    cal_log = arena.GameLog(cache_path or None, cal_key)

    opponents = {opponent_name(lv): EgaroucidOpponent(pool, lv, rules) for lv in levels}
    players: Dict[str, Any] = {"model": player, **opponents}
    model_specs = {lv: arena.match_specs("model", opponent_name(lv), len(openings))
                   for lv in levels}
    pairs = [] if args.no_calibrate else calibration_pairs(levels, args.calibration)
    cal_specs = {(a, b): arena.match_specs(opponent_name(a), opponent_name(b), len(openings))
                 for a, b in pairs}

    def reuse(log: arena.GameLog, specs: Sequence[arena.GameSpec]):
        have = log.load()
        done, todo = {}, []
        for s in specs:
            r = have.get(s.key)
            if (r is not None and tuple(r.seats) == s.seats
                    and _openings_match(r, op_str[s.opening])):
                done[s.key] = r
            else:
                todo.append(s)
        return done, todo

    all_model = [s for lst in model_specs.values() for s in lst]
    all_cal = [s for lst in cal_specs.values() for s in lst]
    done_model, todo_model = reuse(run_log, all_model) if args.resume else ({}, all_model)
    done_cal, todo_cal = reuse(cal_log, all_cal)
    todo = arena.interleave(todo_model, todo_cal)
    print(f"[arena] {pool.version} ({pool.build}); levels {levels}; {len(openings)} openings "
          f"x 2 seats ({args.opening_plies} random plies, seed {args.openings_seed}); "
          f"model {args.game} {net_info['weights']} r={player.depth} {player.mode}"
          + (f" T={args.temperature}" if args.temperature > 0 else ""), flush=True)
    print(f"[arena] games: {len(all_model)} model ({len(done_model)} resumed), "
          f"{len(all_cal)} calibration ({len(done_cal)} cached); playing {len(todo)}; "
          f"log {log_path}", flush=True)

    cal_names = {match for match in (arena.match_name(opponent_name(a), opponent_name(b))
                                     for a, b in pairs)}

    def on_record(rec: arena.GameRecord) -> None:
        (cal_log if rec.match in cal_names else run_log).append(rec)

    t_play = time.time()
    new = arena.run_games(rules, todo, openings, players, parallel=args.parallel,
                          workers=args.workers, seed=args.seed, on_record=on_record,
                          progress_s=args.progress_s,
                          log=lambda m: print(m, flush=True))
    play_s = time.time() - t_play
    by_key = {**done_model, **done_cal, **{r.key: r for r in new}}

    def records(specs):
        return [by_key[s.key] for s in specs if s.key in by_key]

    level_rows = []
    for lv in levels:
        m = arena.match_summary(records(model_specs[lv]), "model", opponent_name(lv),
                                seat_names=rules.seat_names, n_boot=args.n_boot, seed=args.seed)
        level_rows.append({"level": lv, "opponent": opponent_name(lv), **m})
    cal_rows = []
    for (a, b), specs in cal_specs.items():
        m = arena.match_summary(records(specs), opponent_name(a), opponent_name(b),
                                seat_names=rules.seat_names, n_boot=args.n_boot, seed=args.seed)
        cal_rows.append({"level_a": a, "level_b": b, **m})

    # ladder: every game of a complete pair, all matches jointly
    fit_games = arena.ladder_games(records(all_model) + records(all_cal))
    fixed = parse_anchors(args.anchor, levels)
    ladder: Optional[Dict[str, Any]] = None
    ladder_note = None
    if fit_games:
        try:
            ladder = arena.ladder_fit(fit_games, fixed, prior_draws=args.prior_draws,
                                      n_boot=min(args.n_boot, 1000), seed=args.seed)
        except ValueError as e:
            ladder_note = str(e)
    if ladder is not None:
        lv_pts = [(lv, ladder["ratings"][opponent_name(lv)]["elo"]) for lv in levels
                  if opponent_name(lv) in ladder["ratings"]]
        mr = ladder["ratings"].get("model")
        if mr is not None:
            ladder["model"] = mr
            ladder["equivalent_level"] = arena.interpolate_level(lv_pts, mr["elo"])
        ladder["calibrated"] = bool(pairs)
        if not pairs:
            ladder_note = ("no calibration games: each level's rating is only relative to the "
                           "model, and the model's rating is its Elo against the anchor level")

    title = (f"Othello arena: {os.path.basename(args.ckpt)} [{args.game}, {net_info['weights']}, "
             f"r={player.depth}, {player.mode}"
             + (f" T={args.temperature:g}" if args.temperature > 0 else "") + f"] vs {pool.version}"
             f" (1 thread, hash {args.engine_hash}, no book); {len(openings)} openings x 2 "
             f"({args.opening_plies} random plies, seed {args.openings_seed})")
    table = [title, "", "model vs level (model's view; Elo = model - level):",
             arena.format_match_table([(f"level {r['level']}", r) for r in level_rows])]
    if cal_rows:
        table += ["", "calibration (lower level's view):",
                  arena.format_match_table([(f"L{r['level_a']} vs L{r['level_b']}", r)
                                            for r in cal_rows])]
    if ladder is not None:
        table += ["", arena.format_ladder(ladder, [opponent_name(lv) for lv in levels] + ["model"],
                                          {opponent_name(lv): f"level {lv}" for lv in levels})]
        eq = ladder.get("equivalent_level") or {}
        if eq.get("level") is not None:
            table.append(f"  model ~ level {eq['level']:.1f} (between levels "
                         f"{eq['between'][0]} and {eq['between'][1]})")
        elif eq.get("above") is not None:
            table.append(f"  model above the strongest level played ({eq['above']})")
        elif eq.get("below") is not None:
            table.append(f"  model below the weakest level played ({eq['below']})")
    if ladder_note:
        table.append(f"  note: {ladder_note}")
    st = player.stats
    table.append("")
    table.append(f"model: {st['decisions']} decisions, {st['frames']} frames in {st['calls']} "
                 f"network calls ({st['seconds']:.0f} s); unmasked top-1 illegal on "
                 f"{st['top1_illegal']} decisions; engine: {pool.stats['requests']} requests, "
                 f"{pool.stats['restarts']} restarts, {pool.stats['failed']} failed; "
                 f"{play_s:.0f} s playing")
    text = "\n".join(table)
    print("\n" + text, flush=True)

    result = {
        "arena": "othello",
        "config": {
            "checkpoint": os.path.abspath(args.ckpt), "game": args.game,
            "weights": net_info["weights"], "checkpoint_format": net_info["format"],
            "policy": {"mode": player.mode, "temperature": args.temperature,
                       "depth": player.depth, "bf16": player.bf16},
            "openings": {"n": len(openings), "plies": args.opening_plies,
                         "seed": args.openings_seed},
            "levels": levels, "seed": args.seed, "parallel": args.parallel,
            "workers": args.workers, "device": str(player.device),
            "git_commit": arena.git_commit(),
            "core_type": getattr(net, "core_type", None),
            "temporal_mixer": getattr(net.cfg, "temporal_mixer", None),
        },
        "engine": {**engine_cfg, "command": cmd, "path": os.path.abspath(engine)
                   if os.path.isfile(engine) else engine,
                   "timeout_s": args.engine_timeout, "retries": args.engine_retries,
                   "stats": dict(pool.stats)},
        "levels": level_rows,
        "calibration": cal_rows,
        "ladder": ladder,
        "ladder_note": ladder_note,
        "model_stats": dict(player.stats),
        "openings": op_str,
        "games_log": log_path,
        "ladder_cache": cache_path or None,
        "config_key": run_key,
        "table": text,
        "seconds": round(time.time() - t0, 1),
    }
    arena.write_json(args.out, result)
    print(f"[arena] wrote {args.out}", flush=True)
    return result
