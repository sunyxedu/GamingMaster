"""Arena: the playing strength of a trained policy against external engines
at calibrated levels — where is the model superhuman?

    uv run python -m training.arena othello --ckpt ~/gm-runs/multi/RUN/multi_final.pt \\
        --levels 1 3 5 7 9 11 13 15 --openings 100 --device cuda --out arena_othello.json

What is measured
----------------
The model plays as a PURE POLICY: one ``GameNetwork.step`` per frame at the
model's inference depth (``--depth``, default its own ``default_depth``), the
move taken greedily from the legal-masked policy (``--temperature 0``, the
default) or sampled at a temperature — no search, no value head.  Frames are
exactly the game's training frames (each game adapter builds them with the
converter-pinned code of its IL data), streamed through ONE Stage-1 history
per game as a training sample would be: every frame from the initial
position on, the opening's book plies, the opponent's frames and the rules'
forced moves included (``env.league.StreamStepper``; ``layout`` below).

Protocol: paired, seat-swapped games from a SHARED set of opening positions.
N distinct positions reached by a few random legal plies from the start
(seeded, deduplicated by position) are each played once per seating (two
player games: A as seat 0, then A as seat 1), against every opponent, so the
colour and the opening's luck cancel within a pair and opponents are compared
on the same positions.  An opening pair is the independent unit of the
statistics; a pair with a failed game is dropped whole.

Statistics (per match, from A's view): W/D/L, score (win 1, draw 1/2), a 95%
Wilson interval on the score (games as trials; draws make it conservative), a
95% percentile bootstrap over opening pairs, and the Elo difference
400·log10(s / (1 − s)) of both (clamped at ε = 0.5 / games so a sweep stays
finite).  Ladder: a Bradley–Terry (logistic) maximum-likelihood fit over ALL
games jointly — the model against every level and, when the adapter plays
them, calibration games between the engine levels — with some players' ratings
fixed (the anchor), ``prior_draws`` virtual draws per played pairing (keeps
sweeps finite), a 95% CI from the observed Fisher information and one from a
bootstrap over opening positions (every match resampled with the same
openings, then refitted).

Interfaces (the extension point)
--------------------------------
A game is a module ``training/arena_<game>.py`` registered in
:data:`ARENA_GAMES`; ``python -m training.arena <game> ...`` builds its
subcommand from :func:`add_common_arguments` plus the module's
``add_arguments(parser)`` and calls its ``run(args)``.  The module provides:

  rules     :class:`GameRules` — the reference rules: ``initial_state``,
            ``seat_to_move``, ``legal_moves`` (a forced move such as othello's
            pass is the single legal move), ``play``, ``is_terminal``,
            ``seat_scores`` (per seat in [0, 1]), ``position_key`` (opening
            dedupe), ``move_str``.  Optional: ``chance_outcome(state, rng)``
            (a chance node's outcome — dice, a deal — drawn from the game's RNG,
            seeded per opening so both games of a pair see the same outcomes as
            long as they reach the same chance nodes; the outcome is a move of
            the history, no player is asked and no frame is fed),
            ``margin(state, seat)`` (e.g. the final disc difference),
            ``details(state)``, ``seatings`` (who plays which seat per game of a
            pair: default ``(("A", "B"), ("B", "A"))``; e.g. Dou Dizhu
            ``(("A", "B", "B"), ("B", "A", "A"))`` — the model as landlord, then
            as both peasants), ``max_plies``.
  codec     :class:`FrameCodec` — state → the network's obs dict (the training
            frame, from the seat to move), the decoder output → a score per
            legal move; ``layout`` "interleaved" (one stream per game holding
            every frame from the mover's view: the board games' IL samples,
            env/vec_game.py) or "per_seat" (one stream per model seat holding
            only that seat's own frames: hidden-information games, like
            env/vec_gmsim.py's per-seat rows).
  opponents :class:`Opponent` implementations — an engine over its own
            protocol, fed the canonical move history (every move from the
            initial position) and the seat to move.  Stateless engines rebuild
            the position from the history on every call (``act`` may run
            concurrently from ``workers`` threads for different games); a
            stateful one keeps per-game state between ``reset`` and the
            optional ``release``.

Planned adapters (not implemented): backgammon vs gnubg (``chance_outcome``
= the dice; gnubg's external/socket interface fed the position), xiangqi vs
Fairy-Stockfish (UCI ``position startpos moves ...`` from the history; the
gmsim xiangqi frames), Dou Dizhu vs the DouZero agents (per-seat layout,
landlord / peasant seatings, the deal as a chance outcome; the agent rebuilds
its own infoset from the history), chess / shogi / Go vs Stockfish / YaneuraOu
/ KataGo once their simulators exist (UCI / USI / GTP from the move history).

Throughput: games run in lockstep rounds, up to ``--parallel`` at once (a new
game starts as soon as one ends); per round the network makes one batched
step per distinct frame index while the opponents' moves are computed
concurrently by ``--workers`` threads.  Finished games are appended to a JSONL
log next to ``--out`` (``--resume`` continues a killed run from it).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import contextlib
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
import torch

Move = Any
State = Any

#: game → adapter module (``add_arguments(parser)``, ``run(args)``)
ARENA_GAMES: Dict[str, str] = {
    "othello": "training.arena_othello",
}

#: default seatings of a two-player pair: A plays seat 0, then seat 1
TWO_SEATINGS: Tuple[Tuple[str, ...], ...] = (("A", "B"), ("B", "A"))

ELO_PER_NAT = 400.0 / math.log(10.0)       # Elo = ELO_PER_NAT · (logit difference)


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class GameRules(Protocol):
    """A game's reference rules (see the module docstring for the optional
    hooks).  States are immutable (``play`` returns a new one)."""

    name: str
    n_seats: int
    seat_names: Tuple[str, ...]

    def initial_state(self) -> State: ...

    def seat_to_move(self, state: State) -> int: ...

    def legal_moves(self, state: State) -> Sequence[Move]: ...

    def play(self, state: State, move: Move) -> State: ...

    def is_terminal(self, state: State) -> bool: ...

    def seat_scores(self, state: State) -> Sequence[float]: ...

    def position_key(self, state: State) -> Hashable: ...

    def move_str(self, move: Move) -> str: ...


class Opponent(Protocol):
    """An external player (engine) that sees a game only through its move
    history; ``act`` returns the move of ``seat`` (the seat to move) in the
    position the history leads to.  May be called concurrently for different
    games (one call per game at a time)."""

    name: str

    def reset(self, gid: int) -> None: ...

    def act(self, gid: int, seat: int, history: Sequence[Move]) -> Move: ...

    def close(self) -> None: ...


class FrameCodec(Protocol):
    """A game's network interface: the training frame of a state (from the
    seat to move) and a score per legal move from the decoder output.

    ``output_keys``: the step outputs :meth:`move_scores` reads (moved to the
    CPU as float32 numpy once per batched call)."""

    layout: str                                  # "interleaved" | "per_seat"
    output_keys: Tuple[str, ...]

    def encode(self, states: Sequence[State]) -> Dict[str, torch.Tensor]: ...

    def legal(self, states: Sequence[State]) -> Optional[Dict[str, torch.Tensor]]: ...

    def move_scores(self, out: Mapping[str, np.ndarray], row: int, state: State,
                    legal: Sequence[Move]) -> Tuple[Sequence[Move], np.ndarray, Optional[bool]]:
        """(candidate moves, their scores — logits / log-probabilities, a
        higher score is a more likely move —, whether the unmasked top-1
        of the network is a legal move (None: not applicable))."""


def replay(rules: GameRules, history: Sequence[Move]) -> State:
    """The state a move history leads to from the initial position."""
    s = rules.initial_state()
    for m in history:
        s = rules.play(s, m)
    return s


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------

def random_openings(rules: GameRules, n: int, plies: int, seed: int = 0,
                    max_attempts: Optional[int] = None) -> List[List[Move]]:
    """``n`` move sequences of ``plies`` uniformly random legal moves from the
    start (``random.Random(seed)``; forced moves count as plies, chance
    outcomes — ``rules.sample_chance`` — are part of the sequence but no
    plies) leading to DISTINCT non-terminal positions
    (``rules.position_key``).  ValueError when fewer exist or are found
    within ``max_attempts`` playouts."""
    rng = random.Random(seed)
    is_chance = getattr(rules, "is_chance", None)
    seen: set = set()
    out: List[List[Move]] = []
    limit = max_attempts if max_attempts is not None else 200 * n + 1000
    attempts = 0
    while len(out) < n and attempts < limit:
        attempts += 1
        s, moves, k = rules.initial_state(), [], 0
        while not rules.is_terminal(s):
            if is_chance is not None and is_chance(s):
                c = rules.sample_chance(s, rng)
                moves.append(c)
                s = rules.play(s, c)
                continue
            if k == plies:
                break
            m = rng.choice(list(rules.legal_moves(s)))
            moves.append(m)
            s = rules.play(s, m)
            k += 1
        if k < plies or rules.is_terminal(s):
            continue
        key = rules.position_key(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(moves)
    if len(out) < n:
        raise ValueError(f"{rules.name}: only {len(out)} distinct {plies}-ply opening "
                         f"positions in {attempts} random playouts (asked for {n}; "
                         f"use more --opening_plies)")
    return out


# ---------------------------------------------------------------------------
# Games, records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameSpec:
    """One game of a match: opening ``opening`` of the shared set, seating
    ``seating`` of the match's seatings, ``seats`` = the player at every seat."""
    match: str
    opening: int
    seating: int
    seats: Tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.match}|{self.opening}|{self.seating}"

    def seed(self, base: int) -> int:
        """A per-game seed that depends on the game's identity only (not on
        its position in the schedule)."""
        return zlib.crc32(f"{base}|{self.key}".encode())

    def chance_seed(self, base: int) -> int:
        """The seed of the game's chance events: shared by every game played
        from the same opening (common random numbers across the pair)."""
        return zlib.crc32(f"{base}|chance|{self.opening}".encode())


@dataclass
class GameRecord:
    """A finished (or failed) game.  ``scores`` per seat, None when the game
    failed (``error``); ``moves`` as ``rules.move_str``; ``think`` / ``calls``
    = per opponent the seconds spent in and the number of ``act`` calls."""
    match: str
    opening: int
    seating: int
    seats: Tuple[str, ...]
    moves: List[str]
    scores: Optional[List[float]]
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    think: Dict[str, float] = field(default_factory=dict)
    calls: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.match}|{self.opening}|{self.seating}"

    @property
    def ok(self) -> bool:
        return self.error is None and self.scores is not None

    def score_of(self, player: str) -> float:
        """Mean score of ``player``'s seats."""
        vals = [self.scores[s] for s, n in enumerate(self.seats) if n == player]
        if not vals:
            raise KeyError(f"{player!r} does not play in {self.key}")
        return float(sum(vals) / len(vals))

    def to_json(self) -> Dict[str, Any]:
        return {"match": self.match, "opening": self.opening, "seating": self.seating,
                "seats": list(self.seats), "moves": self.moves, "scores": self.scores,
                "error": self.error, "details": self.details, "think": self.think,
                "calls": self.calls, "seconds": round(self.seconds, 3)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "GameRecord":
        return cls(match=d["match"], opening=int(d["opening"]), seating=int(d["seating"]),
                   seats=tuple(d["seats"]), moves=list(d["moves"]),
                   scores=None if d.get("scores") is None else [float(x) for x in d["scores"]],
                   error=d.get("error"), details=dict(d.get("details") or {}),
                   think={k: float(v) for k, v in (d.get("think") or {}).items()},
                   calls={k: int(v) for k, v in (d.get("calls") or {}).items()},
                   seconds=float(d.get("seconds", 0.0)))


def match_name(a: str, b: str) -> str:
    return f"{a} vs {b}"


def match_specs(a: str, b: str, n_openings: int,
                seatings: Sequence[Tuple[str, ...]] = TWO_SEATINGS) -> List[GameSpec]:
    """Every game of the match a vs b: each opening once per seating."""
    return [GameSpec(match_name(a, b), i, j, tuple(a if lab == "A" else b for lab in seating))
            for i in range(n_openings) for j, seating in enumerate(seatings)]


def interleave(*spec_lists: Sequence[GameSpec]) -> List[GameSpec]:
    """Schedule several matches together, opening by opening (the pairs of
    every match complete early and the opponents' load is mixed)."""
    return sorted((s for lst in spec_lists for s in lst),
                  key=lambda s: (s.opening, s.match, s.seating))


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """A frame of a game for a batched player: the state, the seat to move,
    its legal moves, and whether the player must choose the move."""
    gid: int
    state: State
    mover: int
    legal: Tuple[Move, ...]
    acting: bool


class BatchedPlayer:
    """A player that sees every frame of the games it takes part in, all
    concurrent games in one :meth:`step` call (the network side)."""

    batched = True
    name = "batched"

    def begin(self, gid: int, seats: Sequence[int], seed: int) -> None:
        """Game ``gid`` starts; the player controls ``seats``."""

    def end(self, gid: int) -> None:
        """Game ``gid`` is over (or failed): drop its state."""

    def step(self, frames: Sequence[Frame]) -> Dict[int, Move]:
        """The moves of the ``acting`` frames, keyed by game id."""
        raise NotImplementedError


class RandomOpponent:
    """Uniformly random legal moves (per-game RNG seeded from ``seed`` and
    the game id) — a sanity baseline and a test opponent."""

    def __init__(self, rules: GameRules, seed: int = 0, name: str = "random"):
        self.rules = rules
        self.seed = int(seed)
        self.name = name
        self._rng: Dict[int, random.Random] = {}

    def reset(self, gid: int) -> None:
        self._rng[gid] = random.Random(zlib.crc32(f"{self.seed}|{gid}".encode()))

    def act(self, gid: int, seat: int, history: Sequence[Move]) -> Move:
        s = replay(self.rules, history)
        if self.rules.seat_to_move(s) != seat:
            raise ValueError(f"asked for seat {seat}, but seat {self.rules.seat_to_move(s)} "
                             f"is to move")
        rng = self._rng.setdefault(gid, random.Random(gid))
        return rng.choice(list(self.rules.legal_moves(s)))

    def release(self, gid: int) -> None:
        self._rng.pop(gid, None)

    def close(self) -> None:
        self._rng.clear()


class NetworkPlayer(BatchedPlayer):
    """A ``GameNetwork`` as a pure policy over many concurrent games.

    Per game one Stage-1 stream (``codec.layout`` "interleaved": every frame
    of the game, from the mover's view) or one per controlled seat
    ("per_seat": that seat's own frames), stepped by an
    :class:`env.league.StreamStepper` — every frame in order, one batched
    call per distinct frame index.  On an acting frame the move is the
    legal candidate with the largest score (``temperature`` 0; ties → the
    first candidate) or sampled ∝ exp(score / temperature) with a per-game
    RNG seeded from the game's identity (reproducible whatever the batch).

    ``depth``: recurrent depth r of the core (None: the network's own
    default).  ``bf16`` (CUDA): bf16 autocast for the trunk with the value
    head / decoder in fp32 (training.othello_probe.fp32_heads).  ``stats``:
    frames fed, network calls, decisions, and how often the unmasked top-1
    was illegal (a frame-semantics canary: a trained policy rarely points at
    an occupied square)."""

    def __init__(self, net: torch.nn.Module, codec: FrameCodec, *,
                 device: Optional[torch.device] = None, depth: Optional[int] = None,
                 temperature: float = 0.0, seed: int = 0, bf16: bool = False,
                 name: str = "model", stepper=None):
        from env.league import StreamStepper

        if codec.layout not in ("interleaved", "per_seat"):
            raise ValueError(f"unknown codec layout {codec.layout!r}")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.net = net.eval()
        self.codec = codec
        self.device = torch.device(device) if device is not None else next(net.parameters()).device
        self.stepper = stepper if stepper is not None else StreamStepper(
            net, act_steps=depth, device=self.device)
        self.depth = depth if depth is not None else getattr(net, "default_depth", None)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.bf16 = bool(bf16) and self.device.type == "cuda"
        self.name = name
        self._streams: Dict[int, Dict[Optional[int], int]] = {}
        self._rng: Dict[int, np.random.Generator] = {}
        self._next = 0
        self.stats = {"frames": 0, "calls": 0, "decisions": 0, "top1_illegal": 0,
                      "nonfinite": 0, "seconds": 0.0}

    @property
    def mode(self) -> str:
        return "greedy" if self.temperature == 0 else "sample"

    def begin(self, gid: int, seats: Sequence[int], seed: int) -> None:
        keys: List[Optional[int]] = ([None] if self.codec.layout == "interleaved"
                                     else sorted(set(int(s) for s in seats)))
        self._streams[gid] = {}
        for k in keys:
            self._streams[gid][k] = self._next
            self._next += 1
        self.stepper.reset(list(self._streams[gid].values()))
        self._rng[gid] = np.random.default_rng([self.seed, int(seed)])

    def end(self, gid: int) -> None:
        streams = self._streams.pop(gid, {})
        if streams:
            self.stepper.release(list(streams.values()))
        self._rng.pop(gid, None)

    def _autocast(self):
        if not self.bf16:
            return contextlib.nullcontext()
        from training.othello_probe import fp32_heads

        stack = contextlib.ExitStack()
        stack.enter_context(torch.autocast(self.device.type, dtype=torch.bfloat16))
        stack.enter_context(fp32_heads(self.net, self.device.type))
        return stack

    def _choose(self, gid: int, cands: Sequence[Move], scores: np.ndarray) -> Move:
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(s) != len(cands) or not len(cands):
            raise ValueError(f"codec returned {len(s)} scores for {len(cands)} candidates")
        if not np.isfinite(s).all():
            self.stats["nonfinite"] += 1
            s = np.where(np.isfinite(s), s, -np.inf)
            if not np.isfinite(s).any():
                s = np.zeros_like(s)
        if self.temperature == 0:
            return cands[int(np.argmax(s))]
        z = (s - s.max()) / self.temperature
        p = np.exp(z)
        p /= p.sum()
        return cands[int(self._rng[gid].choice(len(cands), p=p))]

    def step(self, frames: Sequence[Frame]) -> Dict[int, Move]:
        rows: List[Tuple[Frame, int]] = []
        for f in frames:
            streams = self._streams[f.gid]
            key = None if self.codec.layout == "interleaved" else f.mover
            if key in streams:
                rows.append((f, streams[key]))
            elif f.acting:
                raise RuntimeError(f"game {f.gid}: acting for seat {f.mover} without a stream")
        if not rows:
            return {}
        t0 = time.perf_counter()
        states = [f.state for f, _ in rows]
        sids = [s for _, s in rows]
        obs = self.codec.encode(states)
        legal = self.codec.legal(states)
        with torch.no_grad(), self._autocast():
            out = self.stepper.step(obs, legal, sids,
                                    [self.stepper.frames_seen(s) for s in sids])
        npo = {k: out[k].detach().float().cpu().numpy() for k in self.codec.output_keys}
        moves: Dict[int, Move] = {}
        for i, (f, _) in enumerate(rows):
            if not f.acting:
                continue
            cands, scores, top1_legal = self.codec.move_scores(npo, i, f.state, f.legal)
            moves[f.gid] = self._choose(f.gid, cands, scores)
            self.stats["decisions"] += 1
            self.stats["top1_illegal"] += int(top1_legal is False)
        self.stats["frames"] += len(rows)
        self.stats["calls"] += 1
        self.stats["seconds"] += time.perf_counter() - t0
        return moves


def _players_of(spec: GameSpec) -> List[str]:
    return list(dict.fromkeys(spec.seats))


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

class _Live:
    __slots__ = ("gid", "idx", "spec", "opening", "state", "history", "ply", "rng",
                 "t0", "think", "calls", "legal", "batched")

    def __init__(self, gid, idx, spec, opening, state, rng, batched):
        self.gid, self.idx, self.spec, self.opening = gid, idx, spec, list(opening)
        self.state, self.rng, self.batched = state, rng, batched
        self.history: List[Move] = []
        self.ply = 0
        self.t0 = time.time()
        self.think: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        self.legal: Tuple[Move, ...] = ()


def _timed_act(player, gid: int, seat: int, history: List[Move]) -> Tuple[Move, float]:
    t0 = time.perf_counter()
    m = player.act(gid, seat, history)
    return m, time.perf_counter() - t0


def run_games(rules: GameRules, specs: Sequence[GameSpec], openings: Sequence[Sequence[Move]],
              players: Mapping[str, Any], *, parallel: int = 256, workers: int = 1,
              seed: int = 0, on_record: Optional[Callable[[GameRecord], None]] = None,
              progress_s: float = 30.0, log: Callable[[str], None] = print) -> List[GameRecord]:
    """Play every game of ``specs`` to the end (see the module docstring).

    ``players``: name → a :class:`BatchedPlayer` or an :class:`Opponent`.
    Per game: the opening's moves are played as book (every player's
    streams see their frames), a single legal move is applied without
    asking anyone, chance outcomes come from the game's RNG
    (:meth:`GameSpec.chance_seed`), every other move is asked from the
    seat's player — batched players in one call per round, opponents in
    ``workers`` threads meanwhile.  A game whose opponent raises or returns
    an illegal move is recorded with ``error`` (scores None) and the run
    goes on; an error of a batched player is fatal.  ``on_record`` is called
    for every finished game (in finishing order); the result is in
    ``specs`` order."""
    batched = {n for n, p in players.items() if getattr(p, "batched", False)}
    unknown = sorted({n for s in specs for n in s.seats} - set(players))
    if unknown:
        raise KeyError(f"no player named {unknown}")
    is_chance = getattr(rules, "is_chance", None)
    max_plies = int(getattr(rules, "max_plies", 100_000))
    margin = getattr(rules, "margin", None)
    details = getattr(rules, "details", None)
    pending = collections.deque(enumerate(specs))
    live: Dict[int, _Live] = {}
    out: List[Optional[GameRecord]] = [None] * len(specs)
    gid_next = 0
    n_done = n_err = 0
    t_start = t_log = time.time()
    pool = cf.ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def start(idx: int, spec: GameSpec) -> None:
        nonlocal gid_next
        gid = gid_next
        gid_next += 1
        g = _Live(gid, idx, spec, openings[spec.opening], rules.initial_state(),
                  random.Random(spec.chance_seed(seed)),
                  [n for n in _players_of(spec) if n in batched])
        live[gid] = g
        for name in _players_of(spec):
            p = players[name]
            if name in batched:
                seats = [s for s, n in enumerate(spec.seats) if n == name]
                p.begin(gid, seats, spec.seed(seed))
            else:
                p.reset(gid)

    def finish(g: _Live, error: Optional[str] = None) -> None:
        nonlocal n_done, n_err
        live.pop(g.gid, None)
        rec = GameRecord(match=g.spec.match, opening=g.spec.opening, seating=g.spec.seating,
                         seats=g.spec.seats, moves=[rules.move_str(m) for m in g.history],
                         scores=None, error=error,
                         think={k: round(v, 4) for k, v in g.think.items()},
                         calls=dict(g.calls), seconds=time.time() - g.t0)
        if error is None:
            rec.scores = [float(x) for x in rules.seat_scores(g.state)]
            if details is not None:
                rec.details.update(details(g.state))
            if margin is not None:
                rec.details["margins"] = [float(margin(g.state, s)) for s in range(rules.n_seats)]
        for name in _players_of(g.spec):
            p = players[name]
            if name in batched:
                p.end(g.gid)
            else:
                rel = getattr(p, "release", None)
                if rel is not None:
                    rel(g.gid)
        out[g.idx] = rec
        n_done += 1
        n_err += error is not None
        if error is not None:
            log(f"[arena] game {g.spec.key} failed: {error}")
        if on_record is not None:
            on_record(rec)

    def advance(g: _Live, move: Move) -> None:
        g.state = rules.play(g.state, move)
        g.history.append(move)
        g.ply += 1

    try:
        while pending or live:
            while pending and len(live) < max(1, parallel):
                start(*pending.popleft())
            frames: Dict[str, List[Frame]] = collections.defaultdict(list)
            asks: List[Tuple[_Live, str, int]] = []
            chosen: Dict[int, Move] = {}
            for g in list(live.values()):
                try:
                    while (is_chance is not None and not rules.is_terminal(g.state)
                           and is_chance(g.state)):
                        # the opening's own outcomes are replayed, later ones drawn
                        advance(g, g.opening[len(g.history)] if len(g.history) < len(g.opening)
                                else rules.sample_chance(g.state, g.rng))
                    if rules.is_terminal(g.state):
                        finish(g)
                        continue
                    if g.ply >= max_plies:
                        finish(g, f"no result after {g.ply} plies")
                        continue
                    mover = rules.seat_to_move(g.state)
                    g.legal = tuple(rules.legal_moves(g.state))
                    if not g.legal:
                        finish(g, "a non-terminal position without legal moves")
                        continue
                except Exception as e:                      # rules error: this game only
                    finish(g, f"rules: {type(e).__name__}: {e}")
                    continue
                acting: Optional[str] = None
                if len(g.history) < len(g.opening):
                    chosen[g.gid] = g.opening[len(g.history)]
                elif len(g.legal) == 1:
                    chosen[g.gid] = g.legal[0]
                else:
                    acting = g.spec.seats[mover]
                for name in g.batched:
                    frames[name].append(Frame(g.gid, g.state, mover, g.legal, name == acting))
                if acting is not None and acting not in batched:
                    asks.append((g, acting, mover))
            futures = {}
            if pool is not None:
                for g, name, seat in asks:
                    futures[g.gid] = (g, name, pool.submit(_timed_act, players[name], g.gid,
                                                           seat, list(g.history)))
            for name, fr in frames.items():
                chosen.update(players[name].step(fr))
            results: List[Tuple[_Live, str, Any]] = []
            if pool is not None:
                for gid, (g, name, fut) in futures.items():
                    try:
                        results.append((g, name, fut.result()))
                    except Exception as e:
                        results.append((g, name, e))
            else:
                for g, name, seat in asks:
                    try:
                        results.append((g, name, _timed_act(players[name], g.gid, seat,
                                                            list(g.history))))
                    except Exception as e:
                        results.append((g, name, e))
            for g, name, res in results:
                if isinstance(res, Exception):
                    finish(g, f"{name}: {type(res).__name__}: {res}")
                    continue
                move, dt = res
                g.think[name] = g.think.get(name, 0.0) + dt
                g.calls[name] = g.calls.get(name, 0) + 1
                chosen[g.gid] = move
            for g in list(live.values()):
                if g.gid not in chosen:
                    finish(g, "no move chosen")         # defensive: a player skipped a frame
                    continue
                m = chosen[g.gid]
                if m not in g.legal:
                    who = ("the opening" if len(g.history) < len(g.opening)
                           else g.spec.seats[rules.seat_to_move(g.state)])
                    finish(g, f"{who} played the illegal move {m!r} "
                              f"(legal: {[rules.move_str(x) for x in g.legal]})")
                    continue
                try:
                    advance(g, m)
                except Exception as e:
                    finish(g, f"rules: {type(e).__name__}: {e}")
            now = time.time()
            if progress_s and now - t_log >= progress_s:
                t_log = now
                rate = n_done / max(now - t_start, 1e-9) * 60
                log(f"[arena] {n_done}/{len(specs)} games done ({n_err} failed), "
                    f"{len(live)} running, {rate:.1f} games/min")
    finally:
        for g in list(live.values()):                    # aborted: release every player
            for name in _players_of(g.spec):
                p = players[name]
                with contextlib.suppress(Exception):     # never mask the original error
                    if name in batched:
                        p.end(g.gid)
                    elif getattr(p, "release", None) is not None:
                        p.release(g.gid)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_interval(score: float, n: int, z: float = 1.96) -> Optional[List[float]]:
    """Wilson score interval of a proportion ``score`` observed over ``n``
    trials (draws as half points: conservative, a draw has less variance
    than a coin flip)."""
    if n <= 0:
        return None
    p = min(max(float(score), 0.0), 1.0)
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def elo_from_score(s: float, eps: float = 0.0) -> float:
    """400·log10(s / (1 − s)), with s clamped to [eps, 1 − eps]."""
    s = min(max(float(s), eps), 1.0 - eps)
    if s <= 0.0:
        return -math.inf
    if s >= 1.0:
        return math.inf
    return 400.0 * math.log10(s / (1.0 - s))


def complete_pairs(records: Iterable[GameRecord], n_seatings: int
                   ) -> Tuple[Dict[int, List[GameRecord]], int, int]:
    """Openings whose every seating finished without error → their records
    (by seating), the number of failed games and of incomplete pairs."""
    by_open: Dict[int, Dict[int, GameRecord]] = collections.defaultdict(dict)
    failed = 0
    for r in records:
        by_open[r.opening][r.seating] = r
        failed += not r.ok
    good = {i: [d[j] for j in range(n_seatings)] for i, d in by_open.items()
            if len(d) == n_seatings and all(x.ok for x in d.values())}
    return dict(sorted(good.items())), failed, len(by_open) - len(good)


def match_summary(records: Sequence[GameRecord], a: str, b: str, *,
                  seatings: Sequence[Tuple[str, ...]] = TWO_SEATINGS,
                  seat_names: Optional[Sequence[str]] = None,
                  n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """W/D/L, score and Elo of ``a`` against ``b`` with 95% CIs (Wilson over
    games; percentile bootstrap over opening pairs), over the complete
    pairs; per seating the same counts, and ``a``'s mean margin when the
    rules report margins."""
    pairs, failed, incomplete = complete_pairs(records, len(seatings))
    out: Dict[str, Any] = {"a": a, "b": b, "games": 0, "pairs": len(pairs),
                           "failed": failed, "incomplete_pairs": incomplete}
    if not pairs:
        return out
    sc = np.array([[r.score_of(a) for r in rs] for rs in pairs.values()], dtype=np.float64)
    flat = sc.ravel()
    n = len(flat)
    eps = 0.5 / n
    score = float(flat.mean())
    out.update(games=n, W=int((flat > 0.5).sum()), D=int((flat == 0.5).sum()),
               L=int((flat < 0.5).sum()), score=score,
               score_ci95_wilson=wilson_interval(score, n))
    pm = sc.mean(axis=1)
    if len(pm) > 1:
        idx = np.random.default_rng(seed).integers(0, len(pm), size=(n_boot, len(pm)))
        boot = pm[idx].mean(axis=1)
        out["score_ci95_bootstrap"] = [float(np.percentile(boot, 2.5)),
                                       float(np.percentile(boot, 97.5))]
    else:
        out["score_ci95_bootstrap"] = None
    out["elo"] = elo_from_score(score, eps)
    for kind in ("wilson", "bootstrap"):
        ci = out[f"score_ci95_{kind}"]
        out[f"elo_ci95_{kind}"] = None if ci is None else [elo_from_score(ci[0], eps),
                                                           elo_from_score(ci[1], eps)]
    per = []
    for j, seating in enumerate(seatings):
        col = sc[:, j]
        a_seats = [s for s, lab in enumerate(seating) if lab == "A"]
        label = ("+".join(seat_names[s] for s in a_seats) if seat_names is not None
                 else "+".join(map(str, a_seats)))
        per.append({"a_seats": label, "games": int(len(col)), "W": int((col > 0.5).sum()),
                    "D": int((col == 0.5).sum()), "L": int((col < 0.5).sum()),
                    "score": float(col.mean())})
    out["by_seating"] = per
    rs = [r for rr in pairs.values() for r in rr]
    if all("margins" in r.details for r in rs):
        m = [float(np.mean([r.details["margins"][s] for s, n_ in enumerate(r.seats) if n_ == a]))
             for r in rs]
        out["mean_margin"] = float(np.mean(m))
    spm = {}
    for p in (a, b):
        calls = sum(r.calls.get(p, 0) for r in rs)
        if calls:
            spm[p] = float(sum(r.think.get(p, 0.0) for r in rs) / calls)
    if spm:
        out["seconds_per_move"] = spm
    return out


def ladder_games(records: Iterable[GameRecord],
                 seatings: Sequence[Tuple[str, ...]] = TWO_SEATINGS
                 ) -> List[Tuple[str, str, float, int]]:
    """The games of complete opening pairs (per match) as ladder-fit input
    (player a, player b, a's score, opening index); a and b = the game's two
    players in seat order."""
    by_match: Dict[str, List[GameRecord]] = collections.defaultdict(list)
    for r in records:
        by_match[r.match].append(r)
    out = []
    for recs in by_match.values():
        pairs, _, _ = complete_pairs(recs, len(seatings))
        for rs in pairs.values():
            for r in rs:
                names = list(dict.fromkeys(r.seats))
                if len(names) != 2:
                    raise ValueError(f"{r.key}: the ladder fit needs two players per game, "
                                     f"got {names}")
                out.append((names[0], names[1], r.score_of(names[0]), r.opening))
    return out


def _pair_table(games: Sequence[Tuple[str, str, float, int]], players: Sequence[str]):
    """(pairs [(i, j)], per opening n [O, P] and score-of-i [O, P], openings)."""
    pidx = {p: k for k, p in enumerate(players)}
    pairs: Dict[Tuple[int, int], int] = {}
    opens: Dict[int, int] = {}
    rows = []
    for a, b, s, o in games:
        i, j = pidx[a], pidx[b]
        if i == j:
            raise ValueError(f"{a!r} plays itself")
        if i > j:
            i, j, s = j, i, 1.0 - s
        k = pairs.setdefault((i, j), len(pairs))
        oi = opens.setdefault(o, len(opens))
        rows.append((oi, k, s))
    N = np.zeros((len(opens), len(pairs)))
    S = np.zeros((len(opens), len(pairs)))
    for oi, k, s in rows:
        N[oi, k] += 1.0
        S[oi, k] += s
    return list(pairs), N, S, list(opens)


def _bt_newton(pairs: Sequence[Tuple[int, int]], n: np.ndarray, s: np.ndarray,
               free: np.ndarray, x0: np.ndarray, max_iter: int = 100, tol: float = 1e-10
               ) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Maximise Σ_pairs s·log σ(x_i − x_j) + (n − s)·log σ(x_j − x_i) over the
    ``free`` coordinates of x (natural units), Newton with backtracking.
    Returns (x, observed information matrix over the free players at x,
    log-likelihood, iterations)."""
    ia = np.array([p[0] for p in pairs], dtype=int)
    ib = np.array([p[1] for p in pairs], dtype=int)
    fidx = np.flatnonzero(free)
    pos = -np.ones(len(x0), dtype=int)
    pos[fidx] = np.arange(len(fidx))
    pa, pb = pos[ia], pos[ib]

    def loglik(x):
        d = x[ia] - x[ib]
        return float(np.sum(-s * np.logaddexp(0.0, -d) - (n - s) * np.logaddexp(0.0, d)))

    def derivs(x):
        d = x[ia] - x[ib]
        p = 0.5 * (1.0 + np.tanh(0.5 * d))                # σ(d), overflow-free
        r = s - n * p                                     # ∂ll/∂x_a per pair
        w = n * p * (1.0 - p)
        g = np.zeros(len(x))
        np.add.at(g, ia, r)
        np.add.at(g, ib, -r)
        info = np.zeros((len(fidx), len(fidx)))           # −Hessian (a weighted Laplacian)
        for k in range(len(pairs)):
            a, b = pa[k], pb[k]
            if a >= 0:
                info[a, a] += w[k]
            if b >= 0:
                info[b, b] += w[k]
            if a >= 0 and b >= 0:
                info[a, b] -= w[k]
                info[b, a] -= w[k]
        return g[fidx], info

    x = x0.astype(np.float64).copy()
    cur = loglik(x)
    it = 0
    if len(fidx):
        for it in range(1, max_iter + 1):
            g, info = derivs(x)
            try:
                step = np.linalg.solve(info, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(info, g, rcond=None)[0]
            step = np.clip(step, -3.0, 3.0)               # ≤ ~520 Elo per iteration
            t = 1.0
            while True:                                   # backtracking: never decrease ll
                y = x.copy()
                y[fidx] += t * step
                new = loglik(y)
                if new >= cur - 1e-12 or t < 1e-6:
                    break
                t *= 0.5
            gain = new - cur
            x, cur = y, new
            if np.max(np.abs(t * step)) < tol or abs(gain) < tol * max(1.0, abs(cur)):
                break
    return x, derivs(x)[1], cur, it


def fit_ratings(games: Sequence[Tuple[str, str, float, int]], fixed: Mapping[str, float],
                prior_draws: float = 1.0) -> Dict[str, Any]:
    """Bradley–Terry maximum likelihood (Elo scale) of the players of
    ``games`` = (a, b, score of a in [0, 1], opening index), the players in
    ``fixed`` held at their given ratings.  ``prior_draws`` virtual draws are
    added to every played pairing (finite ratings for sweeps).  Every free
    player must be connected through games to a fixed one.

    Returns ``ratings`` {player: Elo}, ``se`` {free player: standard error
    from the inverse observed information}, ``cov`` (free players, Elo²),
    ``loglik``, ``iterations``, ``players``, ``free``."""
    players = sorted({p for g in games for p in g[:2]} | set(fixed))
    if not fixed:
        raise ValueError("fit_ratings needs at least one fixed (anchor) rating")
    pairs, N, S, _ = _pair_table(games, players)
    n, s = N.sum(axis=0), S.sum(axis=0)
    _check_connected(players, pairs, fixed)
    return _fit_from_totals(players, pairs, n, s, fixed, prior_draws)


def _check_connected(players, pairs, fixed) -> None:
    adj: Dict[int, set] = collections.defaultdict(set)
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    seen = {k for k, p in enumerate(players) if p in fixed}
    todo = list(seen)
    while todo:
        k = todo.pop()
        for m in adj[k]:
            if m not in seen:
                seen.add(m)
                todo.append(m)
    lost = [p for k, p in enumerate(players) if k not in seen]
    if lost:
        raise ValueError(f"no games connect {lost} to an anchor ({sorted(fixed)})")


def _fit_from_totals(players, pairs, n, s, fixed, prior_draws, x0=None) -> Dict[str, Any]:
    free = np.array([p not in fixed for p in players])
    base = np.array([fixed.get(p, 0.0) / ELO_PER_NAT for p in players])
    if x0 is None:
        x0 = base.copy()
        x0[free] = np.mean([v for v in base[~free]])
    else:
        x0 = np.where(free, x0, base)
    n_ = n + prior_draws
    s_ = s + 0.5 * prior_draws
    x, info, ll, it = _bt_newton(pairs, n_, s_, free, x0)
    ratings = {p: float(x[k] * ELO_PER_NAT) for k, p in enumerate(players)}
    fidx = np.flatnonzero(free)
    cov = np.linalg.pinv(info) * ELO_PER_NAT ** 2 if len(fidx) else np.zeros((0, 0))
    se = {players[k]: float(math.sqrt(max(cov[a, a], 0.0))) for a, k in enumerate(fidx)}
    return {"ratings": ratings, "se": se, "cov": cov.tolist(), "loglik": ll, "iterations": it,
            "players": list(players), "free": [players[k] for k in fidx],
            "x": x}


def ladder_fit(games: Sequence[Tuple[str, str, float, int]], fixed: Mapping[str, float], *,
               prior_draws: float = 1.0, n_boot: int = 1000, seed: int = 0,
               z: float = 1.96) -> Dict[str, Any]:
    """:func:`fit_ratings` plus per free player a 95% CI from the Fisher
    information (± z·se) and a percentile bootstrap over opening positions:
    each replicate resamples the opening indices with replacement and keeps
    every game played from them (all matches share the openings), then
    refits.  Output ratings: {player: {elo, se, fixed, ci95_fisher,
    ci95_bootstrap}}."""
    players = sorted({p for g in games for p in g[:2]} | set(fixed))
    pairs, N, S, opens = _pair_table(games, players)
    _check_connected(players, pairs, fixed)
    fit = _fit_from_totals(players, pairs, N.sum(axis=0), S.sum(axis=0), fixed, prior_draws)
    boot = []
    if n_boot > 0 and len(opens) > 1:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            c = np.bincount(rng.integers(0, len(opens), size=len(opens)),
                            minlength=len(opens)).astype(np.float64)
            f = _fit_from_totals(players, pairs, c @ N, c @ S, fixed, prior_draws, x0=fit["x"])
            boot.append([f["ratings"][p] for p in players])
    B = np.array(boot) if boot else None
    ratings = {}
    for k, p in enumerate(players):
        e = fit["ratings"][p]
        if p in fixed:
            ratings[p] = {"elo": e, "fixed": True}
            continue
        se = fit["se"][p]
        ratings[p] = {"elo": e, "se": se, "fixed": False,
                      "ci95_fisher": [e - z * se, e + z * se],
                      "ci95_bootstrap": (None if B is None else
                                         [float(np.percentile(B[:, k], 2.5)),
                                          float(np.percentile(B[:, k], 97.5))])}
    pairings = []
    for (i, j), nn, ss in zip(pairs, N.sum(axis=0), S.sum(axis=0)):
        pairings.append({"a": players[i], "b": players[j], "games": int(nn),
                         "score_a": float(ss / nn) if nn else None})
    return {"method": (f"Bradley-Terry MLE (Elo scale), draws as half points, "
                       f"{prior_draws:g} virtual draw(s) per played pairing"),
            "fixed": dict(fixed), "prior_draws": prior_draws, "games": len(games),
            "openings": len(opens), "n_boot": len(boot), "ratings": ratings,
            "pairings": pairings, "loglik": fit["loglik"], "iterations": fit["iterations"]}


def interpolate_level(levels: Sequence[Tuple[float, float]], rating: float) -> Dict[str, Any]:
    """Where ``rating`` falls on a ladder of (level, Elo) points: the
    linearly interpolated fractional level within the first pair of
    adjacent levels (by level) whose ratings bracket it, or ``below`` /
    ``above`` the whole ladder."""
    pts = sorted(levels)
    if not pts:
        return {"level": None}
    elos = [e for _, e in pts]
    if rating < min(elos):
        return {"level": None, "below": pts[int(np.argmin(elos))][0]}
    if rating > max(elos):
        return {"level": None, "above": pts[int(np.argmax(elos))][0]}
    if len(pts) == 1:
        return {"level": float(pts[0][0]), "between": [pts[0][0], pts[0][0]]}
    for (l0, e0), (l1, e1) in zip(pts, pts[1:]):
        if min(e0, e1) <= rating <= max(e0, e1):
            f = 0.0 if e1 == e0 else (rating - e0) / (e1 - e0)
            return {"level": float(l0 + f * (l1 - l0)), "between": [l0, l1]}
    return {"level": None}                  # non-monotone ladder with a gap around rating


# ---------------------------------------------------------------------------
# Game log (JSONL; resume and caches)
# ---------------------------------------------------------------------------

def config_key(cfg: Mapping[str, Any]) -> str:
    """A short stable hash of a JSON-able configuration."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


class GameLog:
    """Append-only JSONL of finished games, one record per line tagged with a
    configuration key: records of another configuration are ignored (kept
    in the file).  ``fresh`` truncates the file first."""

    def __init__(self, path: Optional[str], key: str, fresh: bool = False):
        self.path = path
        self.key = key
        if path and fresh and os.path.exists(path):
            os.remove(path)

    def load(self) -> Dict[str, GameRecord]:
        """Records of this configuration by game key (the last one wins)."""
        out: Dict[str, GameRecord] = {}
        if not self.path or not os.path.exists(self.path):
            return out
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue                                # a torn last line
                if d.get("config") != self.key:
                    continue
                r = GameRecord.from_json(d)
                out[r.key] = r
        return out

    def append(self, rec: GameRecord) -> None:
        if not self.path:
            return
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"config": self.key, **rec.to_json()}) + "\n")
            fh.flush()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_policy_network(path: str, game: str, device: torch.device, raw: bool = False
                        ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """One game's network from a train_multi_game checkpoint (the EMA
    weights ``model_state`` unless ``raw``: ``train_model_state``;
    :func:`training.eval_multi_game.load_multi`) or from a single-game one
    (``config`` + ``model_state``, e.g. scripts/export_game_checkpoint.py).
    Returns (net in eval mode on ``device``, info)."""
    try:                                    # the tensors are memory-mapped, not read
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:                    # not a zipfile checkpoint: no mmap
        ck = torch.load(path, map_location="cpu", weights_only=False)
    if "games" in ck:
        from training.eval_multi_game import load_multi

        del ck
        model = load_multi(path, device, raw=raw)
        if game not in model.games:
            raise SystemExit(f"{path}: no {game} network (games: {model.games})")
        return model[game], {"format": "multi_game", "games": list(model.games),
                             "weights": "raw" if raw else "ema"}
    from model.config import checkpoint_config_dict
    from training.train_il_game import build_net_tolerant

    ck_game = ck.get("game", game)
    if ck_game != game:
        raise SystemExit(f"{path}: a {ck_game} checkpoint, not {game}")
    key = "train_model_state" if raw else "model_state"
    if key not in ck:
        raise SystemExit(f"{path}: no {key}")
    net, _ = build_net_tolerant(game, checkpoint_config_dict(ck))
    net.load_state_dict(ck[key])
    return net.to(device).eval(), {"format": "single_game", "games": [game],
                                   "weights": "raw" if raw else "ema"}


def file_identity(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    return {"path": os.path.abspath(path), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=10, cwd=os.path.dirname(os.path.abspath(__file__)))
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def pick_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    from training.train_il_game import pick_device as _pick

    return _pick()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_ci(ci, fmt: str = "{:+.0f}") -> str:
    if not ci:
        return "n/a"
    return "[" + ", ".join(fmt.format(x) for x in ci) + "]"


def format_match_table(rows: Sequence[Tuple[str, Mapping[str, Any]]]) -> str:
    """One line per (label, match_summary)."""
    head = (f"{'':22s} {'games':>5s} {'W':>4s} {'D':>3s} {'L':>4s} {'score':>6s} "
            f"{'95% CI Wilson':>15s} {'Elo':>6s} {'Elo 95% CI Wilson':>17s} "
            f"{'Elo 95% CI pairs':>17s}")
    lines = [head]
    for label, m in rows:
        if not m.get("games"):
            lines.append(f"{label:22s} {0:5d}   (no complete pairs; {m.get('failed', 0)} failed)")
            continue
        lines.append(
            f"{label:22s} {m['games']:5d} {m['W']:4d} {m['D']:3d} {m['L']:4d} "
            f"{m['score']:6.3f} {_fmt_ci(m['score_ci95_wilson'], '{:.3f}'):>15s} "
            f"{m['elo']:+6.0f} {_fmt_ci(m['elo_ci95_wilson']):>17s} "
            f"{_fmt_ci(m['elo_ci95_bootstrap']):>17s}"
            + (f"  ({m['failed']} failed)" if m.get("failed") else ""))
    return "\n".join(lines)


def format_ladder(fit: Mapping[str, Any], order: Sequence[str],
                  labels: Optional[Mapping[str, str]] = None) -> str:
    lines = [f"ladder: {fit['method']}; {fit['games']} games from {fit['openings']} openings, "
             f"{fit['n_boot']} bootstrap replicates"]
    for p in order:
        r = fit["ratings"].get(p)
        if r is None:
            continue
        name = (labels or {}).get(p, p)
        if r["fixed"]:
            lines.append(f"  {name:22s} {r['elo']:+7.0f}   (anchor)")
        else:
            lines.append(f"  {name:22s} {r['elo']:+7.0f} ± {1.96 * r['se']:4.0f}   "
                         f"Fisher {_fmt_ci(r['ci95_fisher'])}  "
                         f"bootstrap {_fmt_ci(r['ci95_bootstrap'])}")
    return "\n".join(lines)


def write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def default_workers() -> int:
    return max(1, min(16, (os.cpu_count() or 2) - 1))


def add_common_arguments(p: argparse.ArgumentParser, game: str) -> None:
    """Arguments every arena game takes (model, openings, schedule, output)."""
    g = p.add_argument_group("model")
    g.add_argument("--ckpt", required=True,
                   help="train_multi_game checkpoint (or a single-game one)")
    g.add_argument("--game", default=game,
                   help=f"which network of the checkpoint plays (default: {game})")
    g.add_argument("--raw", action="store_true", help="raw weights instead of the EMA")
    g.add_argument("--depth", type=int, default=None,
                   help="recurrent depth r of the core (default: the model's default depth)")
    g.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (default); > 0 = sample the legal moves "
                        "∝ exp(logit / T), per-game RNG from --seed")
    g.add_argument("--bf16", action="store_true",
                   help="bf16 autocast for the trunk, value head / decoder in fp32 (CUDA only)")
    g.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    g = p.add_argument_group("openings and schedule")
    g.add_argument("--openings", type=int, default=100,
                   help="distinct opening positions, each played once per seating "
                        "(two-player games: twice, colours swapped)")
    g.add_argument("--opening_plies", type=int, default=6,
                   help="random legal plies from the start that make an opening")
    g.add_argument("--openings_seed", type=int, default=0)
    g.add_argument("--parallel", type=int, default=256, help="games in flight")
    g.add_argument("--workers", type=int, default=default_workers(),
                   help="opponent threads (engine processes)")
    g.add_argument("--seed", type=int, default=0, help="sampling and bootstrap seed")
    g = p.add_argument_group("statistics and output")
    g.add_argument("--n_boot", type=int, default=2000, help="bootstrap replicates")
    g.add_argument("--prior_draws", type=float, default=1.0,
                   help="virtual draws per played pairing in the ladder fit")
    g.add_argument("--out", default=f"arena_{game}.json", help="results JSON")
    g.add_argument("--games_log", default=None,
                   help="JSONL of finished games (default: <out without .json>.games.jsonl)")
    g.add_argument("--resume", action="store_true",
                   help="reuse the finished games of the same configuration from --games_log")
    g.add_argument("--progress_s", type=float, default=30.0,
                   help="seconds between progress lines (0: none)")


def games_log_path(args) -> str:
    if args.games_log:
        return args.games_log
    stem = args.out[:-5] if args.out.endswith(".json") else args.out
    return f"{stem}.games.jsonl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m training.arena",
                                description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="arena_game", required=True, metavar="GAME")
    for game, module in ARENA_GAMES.items():
        mod = importlib.import_module(module)
        sp = sub.add_parser(game, help=(mod.__doc__ or game).strip().split("\n")[0],
                            description=mod.__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        add_common_arguments(sp, game)
        mod.add_arguments(sp)
        sp.set_defaults(_run=mod.run)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
