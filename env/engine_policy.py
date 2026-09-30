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
over GTP (KataGo: ``clear_board`` / ``boardsize`` / ``komi 7.5`` / chinese rules — area
scoring, no suicide, like ours — then every ``play`` and ``genmove``; the strength is the
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
FROM_ENGINE = {"xiangqi": lambda m: _xq_shift(m, +1)}


class UciEngine:
    """One engine process over UCI or USI (``protocol``); ``options`` go out as
    ``setoption name K value V`` before the first game."""

    def __init__(self, command: str, protocol: str = "uci", options: Optional[Mapping[str, str]] = None,
                 timeout: float = 120.0):
        if protocol not in ("uci", "usi"):
            raise ValueError(f"protocol must be uci / usi, got {protocol!r}")
        self.protocol, self.timeout = protocol, float(timeout)
        argv = [os.path.expanduser(a) for a in shlex.split(command)]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                     cwd=os.path.dirname(argv[0]) or None)
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
            raise EngineError(f"{self.name} exited ({self.proc.returncode})")
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
        raise EngineError(f"{self.name}: no {prefix!r} (exited {self.proc.poll()}); last {out[-3:]}")

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
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.timeout, self.size, self.komi, self.rules = float(timeout), board_size, komi, rules
        self.name = self._cmd("name") + " " + self._cmd("version")

    def _cmd(self, line: str) -> str:
        if self.proc.poll() is not None:
            raise EngineError(f"gtp engine exited ({self.proc.returncode})")
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
            raise EngineError(f"gtp {line!r}: no response ({resp[:2]})")
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

    def __init__(self, env, command: str, *, go: str = "go nodes 20000",
                 options: Optional[Mapping[str, str]] = None, protocol: Optional[str] = None,
                 config: str = ""):
        game = env.game
        self.env, self.go, self.config = env, go, config
        self.protocol = protocol or PROTOCOL.get(game)
        if self.protocol is None:
            raise ValueError(f"no engine protocol for {game!r} (pass protocol=)")
        self.start = start_string(game, config)
        self.keys = tuple(env.action_keys)
        if self.protocol == "gtp":
            size = int(np.sqrt(len(gm.State(game, config).legal_actions(0)) - 1))
            self.engine = GtpEngine(command, board_size=size)
        else:
            self.engine = UciEngine(command, self.protocol, options)
        self.name = self.engine.name
        self.moves = 0

    def reset_streams(self, streams: Sequence[int]) -> None:
        pass

    def release_streams(self, streams: Sequence[int]) -> None:
        pass

    def __call__(self, obs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        out = {k: torch.zeros(len(rows), dtype=torch.long) for k in self.keys}
        for i, r in enumerate(rows.tolist()):
            if not bool(obs.active[r]):
                continue
            e, seat = int(obs.env_id[r]), int(obs.to_play[r])
            state, moves = replay_state(self.env.game, self.env.trajectory(e), self.config)
            to, back = TO_ENGINE.get(self.env.game), FROM_ENGINE.get(self.env.game)
            mv = self.engine.best_move(self.start, [to(m) for m in moves] if to else moves, self.go)
            mv = back(mv) if back else mv
            a = action_of(state, seat, mv)
            for k, v in action_keys(state, seat, a, self.keys).items():
                out[k][i] = v
            self.moves += 1
        return out

    def close(self) -> None:
        self.engine.close()


class OthelloEnginePolicy:
    """Egaroucid at one ``level`` as a league opponent on env/vec_othello.py: the position
    comes from the env's own bitboards (``black`` / ``white`` / ``mover``, seat 0 = black)
    and goes to the arena's Egaroucid pool (training/arena_othello.py) as its ``setboard``
    string; the reply square is ``place_idx`` (row * 8 + col).  A forced pass is answered
    with ``passed``, as the env expects."""

    def __init__(self, env, command: Optional[str] = None, *, level: int = 5):
        from training import arena_othello as ao
        from training.othello_probe import BLACK, WHITE

        self.env, self.level, self._ao = env, int(level), ao
        self._colour = (BLACK, WHITE)
        argv, cwd = ao.egaroucid_command(os.path.expanduser(command or ao.default_engine()))
        self.pool = ao.EgaroucidPool(argv, cwd, size=1)
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
        for i, r in enumerate(rows.tolist()):
            if not bool(obs.active[r]):
                continue
            if bool(obs.forced[r]):
                out["passed"][i] = True
                continue
            e = int(obs.env_id[r])
            state = (int(env.black[e]), int(env.white[e]), self._colour[int(env.mover[e])])
            legal = ao.OthelloRules().legal_moves(state)
            mv = self.pool.best_move(ao.egaroucid_board(state), self.level,
                                     [ao.square_name(m) for m in legal])
            sq = ao.parse_square(mv)
            if sq not in legal:
                raise EngineError(f"egaroucid move {mv!r} is not legal here")
            out["place_idx"][i] = sq
            self.moves += 1
        return out

    def close(self) -> None:
        self.pool.close()
