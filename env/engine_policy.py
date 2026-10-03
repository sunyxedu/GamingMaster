"""An external engine (UCI / USI) as a :class:`env.league.League` opponent on a gmrules env
(env/vec_gmsim.py): a referee of fixed strength that self-play reward cannot fool.

    pol = EnginePolicy(env, "~/gm-engines/bin/stockfish", go="go nodes 20000")
    league = League(env, {"stockfish": pol}, mirror_prob=0.0)

Per acting row the engine gets the WHOLE game — ``position fen <start> moves ...`` (USI:
``position sfen <start> moves ...``), the moves being gmrules' own ``action_string`` of every
decision of the env's trajectory, so repetitions and the fifty-move count are the engine's
as well — and its ``bestmove`` is mapped back to the legal action with that string; the
action's keys (``gmsim.State.target``) are the env's replay-format actions.  A best move
that is no legal action of ours raises (a rules / notation mismatch is a bug, never a
silent substitute).

Games: whatever gmrules game prints UCI-style moves and a FEN / SFEN start (``str(State)``):
chess and chess_variants (Stockfish), xiangqi (Pikafish), shogi (YaneuraOu, ``usi``); and go
over GTP (KataGo: ``clear_board`` / ``boardsize`` / ``komi 7.5`` / our rules, :data:`GTP_RULES`
— area scoring, positional superko, no suicide, no friendly pass — then every ``play`` and
``genmove``; the strength is the
command's, e.g. ``katago gtp -model M -config C -override-config maxVisits=64``; a
``resign`` is played as a pass, since our rules have no resignation and the game is scored).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
from typing import Dict, List, Mapping, Optional, Sequence

import gmsim._native as gm
import numpy as np
import torch

#: protocol by game (the engine family that plays it)
PROTOCOL = {"chess": "uci", "chess_variants": "uci", "xiangqi": "uci", "shogi": "usi", "go": "gtp"}


class EngineError(RuntimeError):
    pass


_XQ = re.compile(r"^([a-i])(\d+)([a-i])(\d+)$")


def _xq_shift(move: str, d: int) -> str:
    m = _XQ.match(move)
    if not m:
        raise EngineError(f"not a xiangqi move: {move!r}")
    return f"{m[1]}{int(m[2]) + d}{m[3]}{int(m[4]) + d}"


#: our move string → the engine's, and back (xiangqi: gmrules ranks 1-10, the UCI
#: engines' (Pikafish, Fairy-Stockfish) 0-9)
TO_ENGINE = {"xiangqi": lambda m: _xq_shift(m, -1)}
#: the GTP engine's rules, set to what the env scores: gmrules go is Tromp-Taylor area with
#: positional superko and suicide illegal (KataGo's own "tromp-taylor" allows suicide, which
#: our env rejects; "chinese" lets it pass with dead stones left on the board, which
#: Tromp-Taylor counts as alive)
GTP_RULES = {"go": "koPOSITIONALscoreAREAtaxNONEsui0"}
FROM_ENGINE = {"xiangqi": lambda m: _xq_shift(m, +1)}


def _spawn(argv: Sequence[str], cwd: Optional[str] = None, text: bool = True):
    """An engine process with its stderr in a temporary file (read by :func:`_err_tail`
    for error messages: why an engine did not start is otherwise lost)."""
    import tempfile
    err = tempfile.TemporaryFile(mode="w+b")
    proc = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=err, text=text, bufsize=1, cwd=cwd)
    proc._gm_err = err
    return proc


def _err_tail(proc, n: int = 600) -> str:
    """The last ``n`` characters of a :func:`_spawn` process's stderr ('' if none)."""
    err = getattr(proc, "_gm_err", None)
    if err is None:
        return ""
    try:
        err.flush()
        size = err.seek(0, 2)
        err.seek(max(0, size - n))
        tail = err.read().decode(errors="replace").strip()
    except (OSError, ValueError):
        return ""
    return f"; stderr: {tail!r}" if tail else ""


class UciEngine:
    """One engine process over UCI or USI (``protocol``); ``options`` go out as
    ``setoption name K value V`` before the first game."""

    def __init__(self, command: str, protocol: str = "uci", options: Optional[Mapping[str, str]] = None,
                 timeout: float = 120.0):
        if protocol not in ("uci", "usi"):
            raise ValueError(f"protocol must be uci / usi, got {protocol!r}")
        self.protocol, self.timeout = protocol, float(timeout)
        argv = [os.path.expanduser(a) for a in shlex.split(command)]
        self.proc = _spawn(argv, cwd=os.path.dirname(argv[0]) or None)
        self.name = argv[0]
        self._send(protocol)
        for line in self._until(f"{protocol}ok"):
            if line.startswith("id name "):
                self.name = line[len("id name "):]
        for k, v in (options or {}).items():
            self._send(f"setoption name {k} value {v}")
        self._send("usinewgame" if protocol == "usi" else "ucinewgame")
        self.ready()

    def _send(self, line: str) -> None:
        if self.proc.poll() is not None:
            raise EngineError(f"{self.name} exited ({self.proc.returncode}){_err_tail(self.proc)}")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _until(self, prefix: str) -> List[str]:
        """Lines up to and including the first one starting with ``prefix``."""
        out: List[str] = []
        timer = threading.Timer(self.timeout, self.proc.kill)
        timer.start()
        try:
            for line in self.proc.stdout:
                line = line.strip()
                out.append(line)
                if line.startswith(prefix):
                    return out
        finally:
            timer.cancel()
        raise EngineError(f"{self.name}: no {prefix!r} (exited {self.proc.poll()}); last {out[-3:]}"
                          f"{_err_tail(self.proc)}")

    def ready(self) -> None:
        self._send("isready")
        self._until("readyok")

    def best_move(self, start: str, moves: Sequence[str], go: str) -> str:
        kind = "sfen" if self.protocol == "usi" else "fen"
        self._send(f"position {kind} {start}" + (" moves " + " ".join(moves) if moves else ""))
        self._send(go)
        line = self._until("bestmove")[-1]
        return line.split()[1]

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=5)
            except Exception:           # noqa: BLE001 — a dead engine is closed too
                self.proc.kill()


class GtpEngine:
    """One GTP engine process (go); the game is rebuilt on every move."""

    def __init__(self, command: str, board_size: int = 19, komi: float = 7.5, rules: str = "chinese",
                 timeout: float = 300.0):
        argv = [os.path.expanduser(a) for a in shlex.split(command)]
        self.proc = _spawn(argv)
        self.timeout, self.size, self.komi, self.rules = float(timeout), board_size, komi, rules
        self.name = self._cmd("name") + " " + self._cmd("version")

    def _cmd(self, line: str) -> str:
        if self.proc.poll() is not None:
            raise EngineError(f"gtp engine exited ({self.proc.returncode}){_err_tail(self.proc)}")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        timer = threading.Timer(self.timeout, self.proc.kill)
        timer.start()
        try:
            resp: List[str] = []
            for raw in self.proc.stdout:
                raw = raw.rstrip("\n")
                if not raw.strip():
                    if resp:
                        break
                    continue
                resp.append(raw)
        finally:
            timer.cancel()
        if not resp or resp[0][:1] not in "=?":
            raise EngineError(f"gtp {line!r}: no response ({resp[:2]}; exited {self.proc.poll()})"
                              f"{_err_tail(self.proc)}")
        if resp[0].startswith("?"):
            raise EngineError(f"gtp {line!r}: {' '.join(resp)}")
        return " ".join([resp[0][1:].strip()] + resp[1:]).strip()

    def best_move(self, start: str, moves: Sequence[str], go: str = "") -> str:
        self._cmd(f"boardsize {self.size}")
        self._cmd("clear_board")
        self._cmd(f"komi {self.komi}")
        try:
            self._cmd(f"kata-set-rules {self.rules}")
        except EngineError:
            pass                        # not KataGo: the engine's own rules
        colours = ("B", "W")
        for i, m in enumerate(moves):
            self._cmd(f"play {colours[i % 2]} {m}")
        mv = self._cmd(f"genmove {colours[len(moves) % 2]}")
        return "pass" if mv.lower() in ("pass", "resign") else mv.upper()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._cmd("quit")
                self.proc.wait(timeout=5)
            except Exception:           # noqa: BLE001
                self.proc.kill()


def start_string(game: str, config: str = "") -> str:
    """The FEN / SFEN of the game's initial position (``str(State)`` up to its first ' |')."""
    return str(gm.State(game, config)).split(" |")[0].strip()


def replay_state(game: str, trajectory: np.ndarray, config: str = ""):
    """(gmsim State after the trajectory, the move strings of its decisions)."""
    st = gm.State(game, config)
    moves: List[str] = []
    for seat, a in np.asarray(trajectory).reshape(-1, 2).tolist():
        if seat < 0:
            st.apply_chance(int(a))
        else:
            moves.append(st.action_string(int(seat), int(a)))
            st.apply_action(int(seat), int(a))
    return st, moves


def action_of(state, seat: int, move: str) -> int:
    for a in state.legal_actions(seat):
        if state.action_string(seat, a) == move:
            return int(a)
    raise EngineError(f"engine move {move!r} is no legal action of seat {seat} "
                      f"(legal: {[state.action_string(seat, a) for a in state.legal_actions(seat)][:12]} ...)")


def action_keys(state, seat: int, action: int, keys: Sequence[str]) -> Dict[str, int]:
    """The env's replay-format keys of ``action`` (its IL target indices)."""
    tgt = state.target(seat, action)
    out = {}
    for k in keys:
        raw = bytes(tgt[k])
        out[k] = int(np.frombuffer(raw, dtype=np.int64)[0]) if len(raw) == 8 else int(raw[0])
    return out


class EnginePolicy:
    """A UCI / USI engine as a league :class:`~env.league.Policy` on a gmsim env (see the
    module docstring).  ``go``: the search limit of every move (``go nodes N``,
    ``go depth D``, ``go movetime MS``, ...); ``options``: engine options (Threads, Hash,
    Skill Level, UCI_Elo, ...)."""

    #: League asks only for this policy's own moves (it keeps no stream state)
    acts_only = True

    def __init__(self, env, command: str, *, go: str = "go nodes 20000",
                 options: Optional[Mapping[str, str]] = None, protocol: Optional[str] = None,
                 config: str = "", workers: int = 1):
        game = env.game
        self.env, self.go, self.config = env, go, config
        self.protocol = protocol or PROTOCOL.get(game)
        if self.protocol is None:
            raise ValueError(f"no engine protocol for {game!r} (pass protocol=)")
        self.start = start_string(game, config)
        self.keys = tuple(env.action_keys)
        # ``workers`` engine processes, one thread each: the games of a wave are answered in
        # parallel (one process answering them in turn made a 32-game KataGo evaluation take
        # most of an hour)
        self.engines: List = []
        try:
            for _ in range(max(1, int(workers))):
                if self.protocol == "gtp":
                    size = int(np.sqrt(len(gm.State(game, config).legal_actions(0)) - 1))
                    self.engines.append(GtpEngine(command, board_size=size,
                                                  rules=GTP_RULES.get(game, "chinese")))
                else:
                    self.engines.append(UciEngine(command, self.protocol, options))
        except BaseException:
            self.close()
            raise
        self.engine = self.engines[0]
        self.name = self.engine.name + (f" x{len(self.engines)}" if len(self.engines) > 1 else "")
        self._pool = None
        self.moves = 0

    def reset_streams(self, streams: Sequence[int]) -> None:
        pass

    def release_streams(self, streams: Sequence[int]) -> None:
        pass

    def _solve(self, engine, e: int, seat: int, traj) -> Dict[str, int]:
        state, moves = replay_state(self.env.game, traj, self.config)
        to, back = TO_ENGINE.get(self.env.game), FROM_ENGINE.get(self.env.game)
        mv = engine.best_move(self.start, [to(m) for m in moves] if to else moves, self.go)
        mv = back(mv) if back else mv
        return action_keys(state, seat, action_of(state, seat, mv), self.keys)

    def __call__(self, obs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        out = {k: torch.zeros(len(rows), dtype=torch.long) for k in self.keys}
        tasks = [(i, int(obs.env_id[r]), int(obs.to_play[r]))
                 for i, r in enumerate(rows.tolist()) if bool(obs.active[r])]
        tasks = [(i, e, seat, self.env.trajectory(e)) for i, e, seat in tasks]
        W = min(len(self.engines), len(tasks))
        if W <= 1:
            results = [(i, self._solve(self.engine, e, seat, tr)) for i, e, seat, tr in tasks]
        else:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(len(self.engines))

            def run(j):     # engine j answers every W-th task, in order
                return [(i, self._solve(self.engines[j], e, seat, tr))
                        for i, e, seat, tr in tasks[j::W]]
            results = [x for part in self._pool.map(run, range(W)) for x in part]
        for i, keys in results:
            for k, v in keys.items():
                out[k][i] = v
        self.moves += len(results)
        return out

    def close(self) -> None:
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for eng in getattr(self, "engines", []):
            eng.close()


class OthelloEnginePolicy:
    """Egaroucid at one ``level`` as a league opponent on env/vec_othello.py: the position
    comes from the env's own bitboards (``black`` / ``white`` / ``mover``, seat 0 = black)
    and goes to the arena's Egaroucid pool (training/arena_othello.py) as its ``setboard``
    string; the reply square is ``place_idx`` (row * 8 + col).  A forced pass is answered
    with ``passed``, as the env expects."""

    #: League asks only for this policy's own moves (it keeps no stream state)
    acts_only = True

    def __init__(self, env, command: Optional[str] = None, *, level: int = 5, workers: int = 1):
        from training import arena_othello as ao
        from training.othello_probe import BLACK, WHITE

        self.env, self.level, self._ao = env, int(level), ao
        self._colour = (BLACK, WHITE)
        argv, cwd = ao.egaroucid_command(os.path.expanduser(command or ao.default_engine()))
        self.pool = ao.EgaroucidPool(argv, cwd, size=max(1, int(workers)))   # thread-safe
        self._threads = None
        self.name = f"egaroucid-L{self.level} {self.pool.version}"
        self.go = f"level {self.level}"
        self.moves = 0

    def reset_streams(self, streams: Sequence[int]) -> None:
        pass

    def release_streams(self, streams: Sequence[int]) -> None:
        pass

    def __call__(self, obs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        ao, env = self._ao, self.env
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        out = {"place_idx": torch.zeros(len(rows), dtype=torch.long),
               "passed": torch.zeros(len(rows), dtype=torch.bool)}
        tasks = []
        for i, r in enumerate(rows.tolist()):
            if not bool(obs.active[r]):
                continue
            if bool(obs.forced[r]):
                out["passed"][i] = True
                continue
            e = int(obs.env_id[r])
            tasks.append((i, (int(env.black[e]), int(env.white[e]), self._colour[int(env.mover[e])])))

        def solve(task):
            i, state = task
            legal = ao.OthelloRules().legal_moves(state)
            mv = self.pool.best_move(ao.egaroucid_board(state), self.level,
                                     [ao.square_name(m) for m in legal])
            sq = ao.parse_square(mv)
            if sq not in legal:
                raise EngineError(f"egaroucid move {mv!r} is not legal here")
            return i, sq

        n = len(self.pool.engines)
        if n > 1 and len(tasks) > 1:        # the pool hands each request a free process
            if self._threads is None:
                from concurrent.futures import ThreadPoolExecutor
                self._threads = ThreadPoolExecutor(n)
            results = list(self._threads.map(solve, tasks))
        else:
            results = [solve(t) for t in tasks]
        for i, sq in results:
            out["place_idx"][i] = sq
        self.moves += len(results)
        return out

    def close(self) -> None:
        if getattr(self, "_threads", None) is not None:
            self._threads.shutdown(wait=True)
            self._threads = None
        self.pool.close()


# ---------------------------------------------------------------------------
# backgammon: gnubg through scripts/reference/gnubg_bridge.py
# ---------------------------------------------------------------------------

_BG_HEAD = re.compile(r"on_roll=(\d+) roll=(\d+)-(\d+) .*cube=(\d+) owner=(-?\d+) score=\[(\d+), (\d+)\] "
                      r"crawford=(\w+)")
_BG_SIDE = re.compile(r"seat(\d) \[([^\]]*)\] bar (\d+) off (\d+)")


def bg_parse(text: str) -> Dict:
    """gmrules backgammon ``str(State)`` → on_roll, roll, cube, owner, score, crawford and each
    seat's 25-vector (index i = point i + 1 in that seat's own numbering, 24 = the bar)."""
    h = _BG_HEAD.search(text)
    if h is None:
        raise EngineError(f"unreadable backgammon state: {text[:120]!r}")
    sides = {}
    for m in _BG_SIDE.finditer(text):
        v = [0] * 25
        for pc in m[2].split():
            p, c = pc.split(":")
            v[int(p) - 1] = int(c)
        v[24] = int(m[3])
        sides[int(m[1])] = v
    return {"on_roll": int(h[1]), "roll": (int(h[2]), int(h[3])), "cube": int(h[4]),
            "owner": int(h[5]), "score": (int(h[6]), int(h[7])), "crawford": h[8] == "true",
            "board": [sides[0], sides[1]]}


def bg_apply(mover: List[int], opp: List[int], pairs: Sequence[int]):
    """gnubg's flat (from, to, ...) move (25 = bar, 0 = off) on the mover's / opponent's
    25-vectors; a checker landing on an opponent blot sends it to the bar."""
    me, op = list(mover), list(opp)
    for f, t in zip(pairs[0::2], pairs[1::2]):
        me[24 if f == 25 else f - 1] -= 1
        if t > 0:
            me[t - 1] += 1
            j = 24 - t                      # the mover's point t is the opponent's 25 - t
            if op[j] == 1:
                op[j], op[24] = 0, op[24] + 1
    return me, op


class BackgammonEnginePolicy:
    """gnubg (``plies``-ply, cubeful, match play) as a league opponent on the gmsim
    backgammon env: the position from the replayed trajectory (``bg_parse``), gnubg's best
    move matched to OUR legal move with the same resulting position (both sides, bar
    included — a hit on the way changes it), cube decisions from the bridge's ``cube``
    reply (double if it says double; take if a double should be taken)."""

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    #: League asks only for this policy's own moves (it keeps no stream state)
    acts_only = True

    def __init__(self, env, command: Optional[str] = None, *, plies: int = 0, timeout: float = 300.0):
        import json as _json
        self._json = _json
        self.env, self.plies, self.timeout = env, int(plies), float(timeout)
        self.config = json_config = getattr(env, "config", {}) or {}
        self.match = int(json_config.get("match_length", 7))
        cmd = command or (f"uv run --no-project --python 3.12 --with gnubg==1.1.0a16 python "
                          f"{os.path.join(self.ROOT, 'scripts', 'reference', 'gnubg_bridge.py')} "
                          f"--plies {self.plies}")
        self.proc = _spawn(shlex.split(os.path.expanduser(cmd)))
        hello = self._ask({"op": "hello"})
        self.name = f"{hello.get('engine', 'gnubg')} {self.plies}-ply"
        self.go = f"{self.plies}-ply"
        self.cfg_json = self._json.dumps(json_config)
        self.moves = 0

    def _ask(self, req: Dict) -> Dict:
        if self.proc.poll() is not None:
            raise EngineError(f"gnubg bridge exited ({self.proc.returncode}){_err_tail(self.proc)}")
        self.proc.stdin.write(self._json.dumps(req) + "\n")
        self.proc.stdin.flush()
        timer = threading.Timer(self.timeout, self.proc.kill)
        timer.start()
        try:
            line = self.proc.stdout.readline()
        finally:
            timer.cancel()
        if not line:
            raise EngineError(f"gnubg bridge: no reply (exited {self.proc.poll()}){_err_tail(self.proc)}")
        rep = self._json.loads(line)
        if "error" in rep:
            raise EngineError(f"gnubg bridge: {rep['error']} (request {req})")
        return rep

    def _cube(self, st: Dict, doubler: int) -> Dict:
        owner = st["owner"]
        return {"value": st["cube"], "owner": -1 if owner < 0 else (0 if owner == doubler else 1),
                "match": self.match, "score": [st["score"][doubler], st["score"][1 - doubler]],
                "crawford": st["crawford"], "jacoby": False}

    def _target(self, state, seat: int) -> Dict[int, List[int]]:
        """Both seats' 25-vectors after gnubg's best move from ``state`` (start of a turn)."""
        st = bg_parse(str(state))
        roller = st["on_roll"]
        board = [st["board"][1 - roller], st["board"][roller]]
        rep = self._ask({"op": "move", "board": board, "dice": list(st["roll"]),
                         "cube": self._cube(st, roller)})
        me, op = bg_apply(board[1], board[0], rep["moves"][0][0])
        return {roller: me, 1 - roller: op}

    @staticmethod
    def _reaches(state, seat: int, want: Dict[int, List[int]], depth: int = 4) -> Optional[int]:
        """A legal (sub-)move of ``seat`` from which ``want`` is reachable this turn.  A move
        that bears off the mover's last checker ends the game: in a match the state then
        shows the NEXT game's opening position, never ``want``'s empty board — such a move
        is recognised by the match score changing (only the mover can end the game now)."""
        score0 = bg_parse(str(state))["score"]
        bears_all = not any(want.get(seat, [1]))
        for a in state.legal_actions(seat):
            nxt = state.clone()
            nxt.apply_action(seat, int(a))
            after = bg_parse(str(nxt))
            if all(after["board"][k] == v for k, v in want.items()):
                return int(a)
            if bears_all and after["score"] != score0:
                return int(a)
            t = nxt.turn()
            # the rest of the turn: more sub-moves, or (after the first die of a double) the
            # remaining dice as full moves — the same seat still to act
            if depth > 1 and t[0] == "act" and seat in t[1]:
                if BackgammonEnginePolicy._reaches(nxt, seat, want, depth - 1) is not None:
                    return int(a)
        return None

    def choose(self, state, seat: int, turn_start=None) -> int:
        legal = state.legal_actions(seat)
        names = {state.action_string(seat, a): a for a in legal}
        st = bg_parse(str(state))
        roller = st["on_roll"]
        board = [st["board"][1 - roller], st["board"][roller]]      # [opponent, mover]
        if set(names) <= {"no double", "double"}:
            rep = self._ask({"op": "cube", "board": board, "cube": self._cube(st, roller)})
            return names["double" if rep.get("double") and "double" in names else "no double"]
        if set(names) <= {"take", "drop"}:
            rep = self._ask({"op": "cube", "board": board, "cube": self._cube(st, roller)})
            return names["take" if rep.get("take") else "drop"]
        if len(legal) == 1:
            return int(legal[0])
        # doubles are played one die at a time (sub-moves "(from/to)"): gnubg's best whole
        # move from the turn's start, then a sub-move that can still reach it
        want = self._target(turn_start if turn_start is not None else state, seat)
        a = self._reaches(state, seat, want)
        if a is None:
            raise EngineError(f"gnubg's move (target {want}) is reachable by none of our legal "
                              f"moves {list(names)[:8]} ...")
        return a

    def reset_streams(self, streams: Sequence[int]) -> None:
        pass

    def release_streams(self, streams: Sequence[int]) -> None:
        pass

    def __call__(self, obs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        keys = tuple(self.env.action_keys)
        out = {k: torch.zeros(len(rows), dtype=torch.long) for k in keys}
        for i, r in enumerate(rows.tolist()):
            if not bool(obs.active[r]):
                continue
            e, seat = int(obs.env_id[r]), int(obs.to_play[r])
            traj = np.asarray(self.env.trajectory(e)).reshape(-1, 2)
            state, _ = replay_state(self.env.game, traj, self.cfg_json)
            chance = np.nonzero(traj[:, 0] < 0)[0]
            start = (replay_state(self.env.game, traj[:chance[-1] + 1], self.cfg_json)[0]
                     if len(chance) else None)
            a = self.choose(state, seat, turn_start=start)
            for k, v in action_keys(state, seat, a, keys).items():
                out[k][i] = v
            self.moves += 1
        return out

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()


class NoisyReferee:
    """A referee made weaker by chance: each of its decisions is, with probability ``eps``,
    a uniformly random legal action (the env's ``random_actions``) instead of the wrapped
    policy's — a strength knob below an engine's weakest setting (the IL network scored 0.09
    against Egaroucid's lowest level).  The rows are split first, so the engine is asked
    only for the rows it plays.  Not for backgammon (gnubg plans a whole turn; a random
    sub-move leaves the plan unreachable)."""

    acts_only = True

    def __init__(self, base, env, eps: float, seed: int = 0):
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        if env.game == "backgammon" and eps > 0:
            raise ValueError("NoisyReferee: backgammon's engine plans whole turns")
        self.base, self.env, self.eps = base, env, float(eps)
        self.generator = torch.Generator().manual_seed(seed)
        self.name = f"{getattr(base, 'name', type(base).__name__)} eps={eps:g}"
        self.go = f"{getattr(base, 'go', '')} eps={eps:g}".strip()

    @property
    def moves(self) -> int:
        return getattr(self.base, "moves", 0)

    def reset_streams(self, streams: Sequence[int]) -> None:
        self.base.reset_streams(streams)

    def release_streams(self, streams: Sequence[int]) -> None:
        self.base.release_streams(streams)

    def __call__(self, obs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        rnd = torch.rand(len(rows), generator=self.generator) < self.eps
        parts = []
        if bool((~rnd).any()):
            parts.append((~rnd, self.base(obs, rows[~rnd])))
        if bool(rnd.any()):
            parts.append((rnd, dict(self.env.random_actions(obs.select(rows[rnd]),
                                                            generator=self.generator))))
        out: Dict[str, torch.Tensor] = {}
        for mask, acts in parts:
            for k, v in acts.items():
                v = torch.as_tensor(v)
                if k not in out:
                    out[k] = v.new_zeros((len(rows),) + tuple(v.shape[1:]))
                out[k][mask.to(v.device)] = v.to(out[k].dtype)
        return out

    def close(self) -> None:
        self.base.close()
