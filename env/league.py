"""Seat control for Stage-2 RL: which seat of which env the learner plays, and
who plays the others (opponents = callables or network snapshots).

Policy protocol
---------------
A :class:`Policy` is called with the current :class:`~env.vec_game.VecObs` and
the rows it must process, and returns one action per row (the decoder's
``sample()`` output or the replay format — :meth:`VecGameEnv.replay_actions`
normalises both).  A stateful (network) policy must see EVERY frame of the
streams it plays on, also the frames where the other side moves (othello's
interleaved stream): the league therefore calls it on all rows of its streams
and only uses its actions where it is to play.  ``reset_streams`` tells it a
stream starts a new episode (rows with ``new_episode`` also imply it).

League
------
Per env and episode the :class:`League` decides mirror self-play (the learner
plays every seat; both movers' frames carry a policy loss) or learner vs one
opponent (the learner's seat uniform or fixed — seat matters: generals'
move-priority tiebreak favours seat 0, othello's black moves first).
``learner_rows`` are the rows the learner's network must process,
``learner_acting`` the rows where its action is used (= the policy-loss mask:
"loss on learner seats"), ``act`` merges the opponents' actions in.

Snapshot league: :class:`SnapshotPool` (frozen copies of past learners, in the
full-state checkpoint via ``state_dict``) + :class:`PoolPolicy` (one snapshot
drawn per episode; uniform now, PFSP later through ``SnapshotPool.weights``).
"""

from __future__ import annotations

import collections
import contextlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn

from env.vec_game import VecGameEnv, VecObs


class Policy(Protocol):
    def reset_streams(self, streams: Sequence[int]) -> None:
        """``streams`` start new episodes: drop their temporal state."""

    def __call__(self, obs: VecObs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Actions for ``rows`` (LongTensor), aligned with ``rows``."""


def scatter_rows(actions: Mapping[str, torch.Tensor], rows: torch.Tensor,
                 n_rows: int) -> Dict[str, torch.Tensor]:
    """Row-aligned actions → full ``[n_rows]`` tensors (other rows zero)."""
    out = {}
    for k, v in actions.items():
        v = torch.as_tensor(v).detach().cpu()
        full = torch.zeros((n_rows, *v.shape[1:]), dtype=v.dtype)
        full[rows] = v
        out[k] = full
    return out


class CallablePolicy:
    """A stateless policy from ``fn(obs_subset: VecObs) -> actions``."""

    def __init__(self, fn: Callable[[VecObs], Mapping[str, torch.Tensor]]):
        self.fn = fn

    def reset_streams(self, streams: Sequence[int]) -> None:
        pass

    def __call__(self, obs: VecObs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        return dict(self.fn(obs.select(rows)))


class RandomPolicy(CallablePolicy):
    """Uniformly random legal actions (the env's ``random_actions``)."""

    def __init__(self, env: VecGameEnv, seed: int = 0):
        self.generator = torch.Generator().manual_seed(seed)
        super().__init__(lambda o: env.random_actions(o, generator=self.generator))


# ---------------------------------------------------------------------------
# Network policies
# ---------------------------------------------------------------------------

class _Block:
    """Batched Stage-1 caches of the streams stepped together last."""

    __slots__ = ("streams", "cache", "live")

    def __init__(self, streams: List[int], cache: List[torch.Tensor]):
        self.streams = streams
        self.cache = cache
        self.live = len(streams)


class StreamStepper:
    """``GameNetwork.step`` over any subset of temporal streams, each stream
    keeping its own Stage-1 state.

    Streams at the same frame index share one batched call (``step`` takes one
    ``step_idx`` per call — generals' pass head reads it); a group that is
    exactly the previous call's group (lockstep rollouts) reuses its batched
    cache as is, other groups gather their rows from where they live.  Works
    for both temporal mixers (the attention KV cache grows per frame, the
    recurrent state is fixed-size).  Every frame of a stream must be fed, in
    order: ``step_idx`` is checked against the frames seen.
    """

    def __init__(self, net: nn.Module, *, act_steps: Optional[int] = None,
                 device: Optional[torch.device] = None):
        self.net = net
        self.act_steps = act_steps
        self.device = torch.device(device) if device is not None else next(net.parameters()).device
        self._blocks: Dict[int, _Block] = {}
        self._where: Dict[int, Tuple[int, int]] = {}      # stream → (block id, row)
        self._t: Dict[int, int] = {}                      # stream → frames seen
        self._next = 0

    def frames_seen(self, stream: int) -> int:
        return self._t.get(int(stream), 0)

    def _release(self, stream: int) -> None:
        loc = self._where.pop(stream, None)
        if loc is not None:
            blk = self._blocks[loc[0]]
            blk.live -= 1
            if blk.live == 0:
                del self._blocks[loc[0]]

    def reset(self, streams: Sequence[int]) -> None:
        for s in streams:
            s = int(s)
            self._release(s)
            self._t.pop(s, None)

    def _gather(self, streams: List[int]) -> List[torch.Tensor]:
        locs = [self._where[s] for s in streams]
        b0 = locs[0][0]
        blk = self._blocks[b0]
        if blk.streams == streams and all(b == b0 for b, _ in locs):
            return blk.cache
        by_block: Dict[int, Tuple[List[int], List[int]]] = collections.defaultdict(lambda: ([], []))
        for pos, (b, row) in enumerate(locs):
            by_block[b][0].append(pos)
            by_block[b][1].append(row)
        n = len(streams)
        out = []
        for layer in range(len(blk.cache)):
            ref = blk.cache[layer]
            S = ref.shape[0] // len(blk.streams)
            rest = tuple(ref.shape[1:])
            dst = ref.new_empty((n, S) + rest)
            for b, (pos, rows) in by_block.items():
                c = self._blocks[b].cache[layer]
                src = c.view((len(self._blocks[b].streams), S) + rest).index_select(
                    0, torch.tensor(rows, device=c.device))
                dst.index_copy_(0, torch.tensor(pos, device=c.device), src)
            out.append(dst.view((n * S,) + rest))
        return out

    def _store(self, streams: List[int], cache: List[torch.Tensor], t: int) -> None:
        for s in streams:
            self._release(s)
        bid = self._next
        self._next += 1
        self._blocks[bid] = _Block(list(streams), cache)
        for row, s in enumerate(streams):
            self._where[s] = (bid, row)
            self._t[s] = t

    @torch.no_grad()
    def step(self, obs: Mapping[str, torch.Tensor], legal: Optional[Mapping[str, torch.Tensor]],
             streams: Sequence[int], step_idx: Sequence[int]) -> Dict[str, torch.Tensor]:
        """One frame for each stream (row-aligned obs / legal); returns the
        network's step outputs (``st_cache`` excluded), row-aligned."""
        streams = [int(s) for s in streams]
        step_idx = [int(t) for t in step_idx]
        if len(set(streams)) != len(streams):
            raise ValueError("a stream appears twice in one step")
        groups: Dict[int, List[int]] = collections.defaultdict(list)
        for k, (s, t) in enumerate(zip(streams, step_idx)):
            if t != self._t.get(s, 0):
                raise ValueError(f"stream {s}: frame {t} but {self._t.get(s, 0)} frames seen "
                                 f"(feed every frame of a stream, in order; reset on new episodes)")
            groups[t].append(k)
        n = len(streams)
        merged: Dict[str, torch.Tensor] = {}
        for t, ks in sorted(groups.items()):
            ss = [streams[k] for k in ks]
            idx = torch.tensor(ks, dtype=torch.long)
            o = {key: v[idx].to(self.device) for key, v in obs.items()}
            lg = ({key: v[idx].to(self.device) for key, v in legal.items()}
                  if legal else None)
            cache = None if t == 0 else self._gather(ss)
            out = self.net.step(o, legal=lg, st_cache=cache, step_idx=t,
                                act_steps=self.act_steps)
            self._store(ss, out.pop("st_cache"), t + 1)
            for key, v in out.items():
                if not torch.is_tensor(v) or v.dim() == 0 or v.shape[0] != len(ks):
                    continue
                if key not in merged:
                    merged[key] = v.new_zeros((n,) + tuple(v.shape[1:]))
                merged[key][idx.to(v.device)] = v
        return merged


@contextlib.contextmanager
def _eval_mode(net: nn.Module, greedy: Optional[bool]):
    was_training = net.training
    dec = getattr(net, "action_decoder", None)
    prev = getattr(dec, "greedy", None)
    net.eval()
    if greedy is not None and dec is not None:
        dec.greedy = greedy
    try:
        yield
    finally:
        net.train(was_training)
        if greedy is not None and dec is not None:
            dec.greedy = prev


class NetworkPolicy:
    """A game network (e.g. ``multi_game_model[game]`` or a frozen snapshot)
    as a policy: eval-mode ``step`` per frame with per-stream Stage-1 state.

    ``greedy`` toggles the decoder's argmax mode during the call (decoders
    that honour it); ``output_keys`` are returned besides the ``action_*``
    ones (e.g. ``("value",)``).
    """

    def __init__(self, net: nn.Module, *, act_steps: Optional[int] = None,
                 device: Optional[torch.device] = None, greedy: Optional[bool] = None,
                 output_keys: Sequence[str] = ()):
        self.net = net
        self.stepper = StreamStepper(net, act_steps=act_steps, device=device)
        self.greedy = greedy
        self.output_keys = tuple(output_keys)

    def reset_streams(self, streams: Sequence[int]) -> None:
        self.stepper.reset(streams)

    def __call__(self, obs: VecObs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        streams = obs.stream_id[rows].tolist()
        restart = [s for s, ne in zip(streams, obs.new_episode[rows].tolist()) if ne]
        if restart:
            self.stepper.reset(restart)
        with _eval_mode(self.net, self.greedy):
            out = self.stepper.step({k: v[rows] for k, v in obs.obs.items()},
                                    {k: v[rows] for k, v in obs.legal.items()},
                                    streams, obs.step_idx[rows].tolist())
        return {k: v.detach().cpu() for k, v in out.items()
                if k.startswith("action_") or k in self.output_keys}


class SnapshotPool:
    """Frozen learner snapshots (CPU state dicts), oldest evicted past ``max_size``.

    ``weights`` (optional: ``fn(ids, meta) -> weights``) turns uniform sampling
    into e.g. prioritized fictitious self-play later."""

    def __init__(self, max_size: int = 16, seed: int = 0,
                 weights: Optional[Callable[[List[int], List[Dict]], Sequence[float]]] = None):
        self.max_size = int(max_size)
        self.weights = weights
        self._states: "collections.OrderedDict[int, Tuple[Dict[str, torch.Tensor], Dict]]" = \
            collections.OrderedDict()
        self._next = 0
        self._gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self._states)

    def ids(self) -> List[int]:
        return list(self._states)

    def add(self, state_dict: Mapping[str, torch.Tensor], meta: Optional[Dict] = None) -> int:
        sid = self._next
        self._next += 1
        self._states[sid] = ({k: v.detach().cpu().clone() for k, v in state_dict.items()},
                             dict(meta or {}))
        while len(self._states) > self.max_size:
            self._states.popitem(last=False)
        return sid

    def get(self, sid: int) -> Dict[str, torch.Tensor]:
        return self._states[sid][0]

    def meta(self, sid: int) -> Dict:
        return self._states[sid][1]

    def sample(self) -> int:
        if not self._states:
            raise RuntimeError("snapshot pool is empty")
        ids = self.ids()
        if self.weights is None:
            return ids[int(torch.randint(len(ids), (1,), generator=self._gen))]
        w = torch.as_tensor(list(self.weights(ids, [self._states[i][1] for i in ids])),
                            dtype=torch.float64)
        return ids[int(torch.multinomial(w, 1, generator=self._gen))]

    def state_dict(self) -> Dict[str, Any]:
        return {"max_size": self.max_size, "next": self._next, "gen": self._gen.get_state(),
                "snapshots": [(sid, st, meta) for sid, (st, meta) in self._states.items()]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.max_size = int(state["max_size"])
        self._next = int(state["next"])
        self._gen.set_state(state["gen"])
        self._states = collections.OrderedDict(
            (int(sid), (st, dict(meta))) for sid, st, meta in state["snapshots"])


class PoolPolicy:
    """Opponent playing a :class:`SnapshotPool` member drawn once per episode
    (per ``reset_streams`` call).  ``load_net(state_dict) -> nn.Module`` builds
    a ready network for a snapshot; at most ``max_nets`` networks are kept
    (the least recently used idle one is dropped)."""

    def __init__(self, pool: SnapshotPool, load_net: Callable[[Dict[str, torch.Tensor]], nn.Module],
                 *, act_steps: Optional[int] = None, device: Optional[torch.device] = None,
                 greedy: Optional[bool] = None, max_nets: int = 4):
        self.pool = pool
        self.load_net = load_net
        self.kw = dict(act_steps=act_steps, device=device, greedy=greedy)
        self.max_nets = int(max_nets)
        self._sid: Dict[int, int] = {}                  # stream → snapshot id
        self._nets: "collections.OrderedDict[int, NetworkPolicy]" = collections.OrderedDict()

    def reset_streams(self, streams: Sequence[int]) -> None:
        sid = self.pool.sample()
        for s in streams:
            s = int(s)
            old = self._sid.get(s)
            if old is not None and old in self._nets:
                self._nets[old].reset_streams([s])
            self._sid[s] = sid

    def snapshot_of(self, stream: int) -> Optional[int]:
        return self._sid.get(int(stream))

    def _policy(self, sid: int) -> NetworkPolicy:
        if sid in self._nets:
            self._nets.move_to_end(sid)
            return self._nets[sid]
        live = set(self._sid.values())
        for old in list(self._nets):
            if len(self._nets) < self.max_nets:
                break
            if old not in live:
                del self._nets[old]
        pol = NetworkPolicy(self.load_net(self.pool.get(sid)), **self.kw)
        self._nets[sid] = pol
        return pol

    def __call__(self, obs: VecObs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        streams = obs.stream_id[rows].tolist()
        missing = [s for s in streams if s not in self._sid]
        if missing:
            raise RuntimeError(f"streams {missing[:4]} have no snapshot (reset_streams first)")
        by_sid: Dict[int, List[int]] = collections.defaultdict(list)
        for k, s in enumerate(streams):
            by_sid[self._sid[s]].append(k)
        parts = {}
        for sid, ks in by_sid.items():
            parts[sid] = (ks, self._policy(sid)(obs, rows[ks]))
        return _merge_parts(parts.values(), len(streams))


def _merge_parts(parts, n: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for ks, acts in parts:
        idx = torch.tensor(ks, dtype=torch.long)
        for k, v in acts.items():
            if k not in out:
                out[k] = torch.zeros((n,) + tuple(v.shape[1:]), dtype=v.dtype)
            out[k][idx] = v
    return out


# ---------------------------------------------------------------------------
# League
# ---------------------------------------------------------------------------

class League:
    """Per env and episode: mirror self-play or learner vs an opponent.

    ``opponents`` maps names to policies; each non-mirror episode draws one
    by ``opponent_weights`` (uniform by default).  ``mirror_prob`` is the
    probability of a mirror episode (1.0, the default, or no opponents: pure
    mirror).  ``learner_seat`` pins the learner's seat in non-mirror episodes
    (None: uniform per episode).

    Call :meth:`update` with every VecObs the env returns (it assigns the
    envs whose rows start a new episode), then per step: run the learner on
    :meth:`learner_rows`, and :meth:`act` to merge the opponents' actions.
    """

    def __init__(self, env: VecGameEnv, opponents: Optional[Mapping[str, Policy]] = None, *,
                 opponent_weights: Optional[Mapping[str, float]] = None,
                 mirror_prob: float = 1.0, learner_seat: Optional[int] = None, seed: int = 0):
        self.env = env
        self.opponents: Dict[str, Policy] = dict(opponents or {})
        names = list(self.opponents)
        w = dict(opponent_weights or {n: 1.0 for n in names})
        unknown = set(w) - set(names)
        if unknown:
            raise KeyError(f"weights for unknown opponents {sorted(unknown)}")
        self._names = [n for n in names if w.get(n, 0.0) > 0]
        self._weights = torch.tensor([float(w[n]) for n in self._names], dtype=torch.float64)
        self.mirror_prob = float(mirror_prob)
        if learner_seat is not None and not 0 <= learner_seat < env.n_seats:
            raise ValueError(f"learner_seat {learner_seat} out of range")
        self.learner_seat = learner_seat
        self._gen = torch.Generator().manual_seed(seed)
        self._alloc(env.num_envs)

    def _alloc(self, n: int) -> None:
        self.learner_seats = torch.ones(n, self.env.n_seats, dtype=torch.bool)
        self.opponent_of: List[Optional[str]] = [None] * n

    # -- assignment ----------------------------------------------------------

    def _draw(self) -> float:
        return float(torch.rand((), generator=self._gen))

    def _assign(self, i: int, obs: VecObs) -> None:
        S = self.env.n_seats
        if not self._names or self._draw() < self.mirror_prob:
            self.learner_seats[i] = True
            self.opponent_of[i] = None
            return
        seat = (self.learner_seat if self.learner_seat is not None
                else int(torch.randint(S, (1,), generator=self._gen)))
        name = self._names[int(torch.multinomial(self._weights, 1, generator=self._gen))]
        self.learner_seats[i] = False
        self.learner_seats[i, seat] = True
        self.opponent_of[i] = name
        rows = (obs.env_id == i) & (obs.stream_seats & ~self.learner_seats[i]).any(-1)
        self.opponents[name].reset_streams(obs.stream_id[rows].tolist())

    def update(self, obs: VecObs) -> List[int]:
        """Assign the envs whose rows start a new episode; returns their ids."""
        if self.learner_seats.shape[0] != self.env.num_envs:
            self._alloc(self.env.num_envs)
        new = sorted(set(obs.env_id[obs.new_episode].tolist()))
        for i in new:
            self._assign(i, obs)
        return new

    def is_mirror(self) -> torch.Tensor:
        """[N] bool: mirror episode (learner on every seat)."""
        return self.learner_seats.all(-1)

    # -- routing ---------------------------------------------------------------

    def learner_rows(self, obs: VecObs) -> torch.Tensor:
        """[R] bool: active rows whose stream the learner plays on (its network
        must process them, acting or not)."""
        mine = self.learner_seats[obs.env_id]
        return obs.active & (obs.stream_seats & mine).any(-1)

    def learner_acting(self, obs: VecObs) -> torch.Tensor:
        """[R] bool: active rows where the learner is to play — its action is
        used and the policy loss applies."""
        mine = self.learner_seats[obs.env_id]
        return obs.active & mine.gather(1, obs.to_play.view(-1, 1)).squeeze(1)

    def opponent_rows(self, obs: VecObs, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """([R] bool rows opponent ``name`` must process, [R] bool rows it plays)."""
        of = torch.tensor([self.opponent_of[i] == name for i in range(len(self.opponent_of))],
                          dtype=torch.bool)
        env_mask = of[obs.env_id] & obs.active
        theirs = ~self.learner_seats[obs.env_id]
        part = env_mask & (obs.stream_seats & theirs).any(-1)
        acting = env_mask & theirs.gather(1, obs.to_play.view(-1, 1)).squeeze(1)
        return part, acting

    def act(self, obs: VecObs, learner_actions: Optional[Mapping[str, torch.Tensor]]
            ) -> Dict[str, torch.Tensor]:
        """Full [R] replay-format actions for ``env.step``: the learner's
        (``[R]``-aligned, e.g. :func:`scatter_rows`) where it plays, each
        opponent's (run on all its rows) where it plays."""
        R = obs.n_rows
        out = {k: v.clone() for k, v in self.env.replay_actions(learner_actions or {}).items()}
        for name, pol in self.opponents.items():
            part, acting = self.opponent_rows(obs, name)
            rows = part.nonzero().squeeze(1)
            if rows.numel() == 0:
                continue
            acts = self.env.replay_actions(pol(obs, rows))
            use = acting[rows]
            for k, v in acts.items():
                if k not in out:
                    out[k] = torch.zeros((R,) + tuple(v.shape[1:]), dtype=v.dtype)
                out[k][rows[use]] = v[use]
        return out

    def learner_score(self, winner: torch.Tensor) -> torch.Tensor:
        """[N] learner result per env from ``info["winner"]``: 1 win, 0.5 draw,
        0 loss; NaN for mirror episodes (the learner plays both sides)."""
        w = winner.long()
        won = torch.zeros(len(w), dtype=torch.bool)
        ok = w >= 0
        won[ok] = self.learner_seats[ok.nonzero().squeeze(1), w[ok]]
        score = torch.where(ok, won.double(), torch.full((len(w),), 0.5, dtype=torch.float64))
        return torch.where(self.is_mirror(), torch.full_like(score, float("nan")), score)

    # -- checkpointing -----------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {"gen": self._gen.get_state(), "learner_seats": self.learner_seats.clone(),
                "opponent_of": list(self.opponent_of)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._gen.set_state(state["gen"])
        self.learner_seats = state["learner_seats"].clone()
        self.opponent_of = list(state["opponent_of"])


def play_episodes(env: VecGameEnv, league: League, learner: Policy, *,
                  max_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Reset every env and play the episodes out (evaluation / smoke runs;
    needs ``autoreset`` off): the learner policy on its seats, the league's
    opponents on theirs.  Returns ``winner`` [N] (seat, −1 none), ``score``
    [N] (:meth:`League.learner_score`), ``learner_seats`` [N, S],
    ``episode_len`` [N] and ``steps``."""
    if env.autoreset:
        raise ValueError("play_episodes needs autoreset off")
    obs = env.reset()
    learner.reset_streams(obs.stream_id.tolist())
    league.update(obs)          # resets the drawn opponents' streams
    N = env.num_envs
    winner = torch.full((N,), -1, dtype=torch.long)
    episode_len = torch.zeros(N, dtype=torch.long)
    steps = 0
    while bool(obs.active.any()) and (max_steps is None or steps < max_steps):
        rows = league.learner_rows(obs).nonzero().squeeze(1)
        la = scatter_rows(learner(obs, rows), rows, obs.n_rows) if rows.numel() else {}
        obs, _, done, info = env.step(league.act(obs, la))
        winner = torch.where(done, info["winner"], winner)
        episode_len = torch.where(done, info["episode_len"], episode_len)
        league.update(obs)
        steps += 1
    return {"winner": winner, "score": league.learner_score(winner),
            "learner_seats": league.learner_seats.clone(), "episode_len": episode_len,
            "steps": torch.tensor(steps)}
