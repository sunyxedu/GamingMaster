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
stream starts a new episode (rows with ``new_episode`` also imply it);
``release_streams`` that it no longer plays a stream (episode over, env
handed to another controller): it drops the stream's state.  Actions come
back on the device of the VecObs.

League
------
Per env and episode the :class:`League` decides mirror self-play (the learner
plays every seat; both movers' frames carry a policy loss) or learner vs one
opponent (the learner's seat uniform or fixed — seat matters: generals'
move-priority tiebreak favours seat 0, othello's black moves first).
``learner_rows`` are the rows the learner's network must process,
``learner_acting`` the rows where its action is sent, ``learner_policy_mask``
the rows that take the policy loss ("loss on learner seats"): learner_acting
minus the ``forced`` frames, where the env applies its own action whatever
was sampled (those still carry the value loss).  ``act`` merges the
opponents' actions in.  The league releases its opponents' streams when an
env ends or changes hands; the learner's own streams are the caller's.

Rollouts are meant to run in lockstep (see env/vec_game.py): then each
network policy makes ONE batched ``step`` call per env step.

Snapshot league: :class:`SnapshotPool` (frozen copies of past learners; with
``store_dir`` one file each and only ids / meta in ``state_dict``) +
:class:`PoolPolicy` (per iteration an active set of at most ``max_nets``
snapshots, one of them drawn per episode; uniform now, PFSP later through
``SnapshotPool.weights``).
"""

from __future__ import annotations

import collections
import contextlib
import os
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn

from env.vec_game import VecGameEnv, VecObs


class Policy(Protocol):
    def reset_streams(self, streams: Sequence[int]) -> None:
        """``streams`` start new episodes: drop their temporal state."""

    def release_streams(self, streams: Sequence[int]) -> None:
        """This policy no longer plays ``streams``: drop their state."""

    def __call__(self, obs: VecObs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Actions for ``rows`` (LongTensor), aligned with ``rows``."""


def release(policy: Policy, streams: Sequence[int]) -> None:
    """``policy.release_streams`` when the policy has it (plain callables
    written against the older protocol hold no per-stream state)."""
    fn = getattr(policy, "release_streams", None)
    if fn is not None and len(streams):
        fn(streams)


def restart_streams(obs: VecObs, rows: torch.Tensor) -> List[int]:
    """Streams of ``rows`` whose temporal state must restart: ``new_episode``, or
    the stream's first frame (``step_idx`` 0) — per-seat streams of alternating
    candidate games get their first frame after the env's reset step."""
    ne = obs.new_episode[rows] | (obs.step_idx[rows] == 0)
    return obs.stream_id[rows][ne].tolist()


def env_streams(obs: VecObs, envs) -> List[int]:
    """Every stream of the envs ``envs`` (iterable of env ids)."""
    envs = torch.as_tensor(sorted(envs), dtype=torch.long)
    return obs.stream_id[torch.isin(obs.env_id, envs.to(obs.env_id.device))].tolist()


def live_envs(obs: VecObs) -> set:
    """Envs whose episode is still running (some row has a frame)."""
    return set(obs.env_id[obs.active].tolist())


def scatter_rows(actions: Mapping[str, torch.Tensor], rows: torch.Tensor,
                 n_rows: int) -> Dict[str, torch.Tensor]:
    """Row-aligned actions → full ``[n_rows]`` tensors (other rows zero),
    on the actions' device."""
    out = {}
    for k, v in actions.items():
        v = torch.as_tensor(v).detach()
        full = v.new_zeros((n_rows, *v.shape[1:]))
        full[rows] = v
        out[k] = full
    return out


class CallablePolicy:
    """A stateless policy from ``fn(obs_subset: VecObs) -> actions``."""

    def __init__(self, fn: Callable[[VecObs], Mapping[str, torch.Tensor]]):
        self.fn = fn

    def reset_streams(self, streams: Sequence[int]) -> None:
        pass

    def release_streams(self, streams: Sequence[int]) -> None:
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
    """Batched Stage-1 caches of the streams stepped together last
    (``streams[row]`` is None once that row's stream moved on or was
    released; a block is compacted once half its rows are dead)."""

    __slots__ = ("streams", "cache", "live")

    def __init__(self, streams: List[Optional[int]], cache: List[torch.Tensor]):
        self.streams = streams
        self.cache = cache
        self.live = sum(s is not None for s in streams)


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

    Memory stays O(live streams): a stream's state is freed by
    :meth:`reset` (= release), and a batched block whose rows moved on or
    were released is compacted once at most half of it is still live, so
    the held cache is at most twice what the live streams need.
    ``n_calls`` / ``n_rows`` count the ``net.step`` calls and the rows they
    carried (mean batch = n_rows / n_calls: 1 call per env step in lockstep).
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
        self.n_calls = 0
        self.n_rows = 0

    def frames_seen(self, stream: int) -> int:
        return self._t.get(int(stream), 0)

    def live_streams(self) -> List[int]:
        return list(self._where)

    def cache_numel(self) -> int:
        """Elements held by all cached blocks (tests / memory canaries)."""
        return sum(c.numel() for b in self._blocks.values() for c in b.cache)

    def _drop_row(self, stream: int) -> Optional[int]:
        """Detach ``stream`` from its block; returns the block id."""
        loc = self._where.pop(stream, None)
        if loc is None:
            return None
        blk = self._blocks[loc[0]]
        blk.streams[loc[1]] = None
        blk.live -= 1
        return loc[0]

    def _compact(self, bid: int) -> None:
        blk = self._blocks.get(bid)
        if blk is None:
            return
        if blk.live == 0:
            del self._blocks[bid]
            return
        n = len(blk.streams)
        if 2 * blk.live > n:
            return
        keep = [r for r, s in enumerate(blk.streams) if s is not None]
        cache = []
        for c in blk.cache:
            S = c.shape[0] // n
            rest = tuple(c.shape[1:])
            idx = torch.tensor(keep, device=c.device)
            cache.append(c.view((n, S) + rest).index_select(0, idx)
                         .reshape((len(keep) * S,) + rest))
        blk.cache = cache
        blk.streams = [blk.streams[r] for r in keep]
        for row, s in enumerate(blk.streams):
            self._where[s] = (bid, row)

    def reset(self, streams: Sequence[int]) -> None:
        """Drop the streams' state (new episode, or no longer played)."""
        touched = set()
        for s in streams:
            s = int(s)
            bid = self._drop_row(s)
            if bid is not None:
                touched.add(bid)
            self._t.pop(s, None)
        for bid in touched:
            self._compact(bid)

    release = reset

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
        touched = {self._drop_row(s) for s in streams} - {None}
        for bid in touched:
            self._compact(bid)
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
            self.n_calls += 1
            self.n_rows += len(ks)
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


def _is_logits(key: str) -> bool:
    return key.endswith("_logits") or key.endswith("_logit")


class NetworkPolicy:
    """A game network (e.g. ``multi_game_model[game]`` or a frozen snapshot)
    as a policy: eval-mode ``step`` per frame with per-stream Stage-1 state.

    Returns every row-aligned step output except the logits: the decoder's
    whole ``sample()`` output (actions under whatever keys the decoder uses,
    ``action_log_prob``, the conditional ``legal_*`` masks its exact replay
    needs), ``value``, ...; ``output_keys`` adds logits back (e.g.
    ``("place_logits",)``).  ``greedy`` toggles the decoder's argmax mode
    during the call (decoders that honour it).
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

    def release_streams(self, streams: Sequence[int]) -> None:
        self.stepper.reset(streams)

    def __call__(self, obs: VecObs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        streams = obs.stream_id[rows].tolist()
        restart = restart_streams(obs, rows)
        if restart:
            self.stepper.reset(restart)
        with _eval_mode(self.net, self.greedy):
            out = self.stepper.step({k: v[rows] for k, v in obs.obs.items()},
                                    {k: v[rows] for k, v in obs.legal.items()},
                                    streams, obs.step_idx[rows].tolist())
        dev = obs.to_play.device
        return {k: v.detach().to(dev) for k, v in out.items()
                if not _is_logits(k) or k in self.output_keys}


class SnapshotPool:
    """Frozen learner snapshots, oldest evicted past ``max_size``.

    In memory (CPU state dicts, written inline by ``state_dict``) by default;
    with ``store_dir`` each snapshot is one file there (``torch.save``),
    loaded on :meth:`get`, deleted on eviction, and ``state_dict`` keeps only
    ids, file names (relative to ``store_dir``: a resume may point
    ``store_dir`` elsewhere) and meta — snapshots of large networks stay out
    of host RAM and of every full-state checkpoint.  The trainer decides
    what a snapshot holds (e.g. a per-game slice of the multi-game model).

    ``weights`` (optional: ``fn(ids, meta) -> weights``) turns uniform sampling
    into e.g. prioritized fictitious self-play later.

    ``defer_delete``: evicted snapshot files are NOT deleted on eviction but
    queued until :meth:`flush_deletes` — call it after writing the
    checkpoint a resume will start from, so that checkpoint never refers to
    a file a later eviction removed (a requeue then restores the same pool)."""

    def __init__(self, max_size: int = 16, seed: int = 0,
                 weights: Optional[Callable[[List[int], List[Dict]], Sequence[float]]] = None,
                 store_dir: Optional[str] = None, defer_delete: bool = False):
        self.max_size = int(max_size)
        self.weights = weights
        self.store_dir = store_dir
        self.defer_delete = bool(defer_delete)
        self._pending: List[str] = []
        if store_dir is not None:
            os.makedirs(store_dir, exist_ok=True)
        # sid → (state dict, or its file name under store_dir; meta)
        self._states: "collections.OrderedDict[int, Tuple[Any, Dict]]" = \
            collections.OrderedDict()
        self._next = 0
        self._gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, sid: int) -> bool:
        return sid in self._states

    def ids(self) -> List[int]:
        return list(self._states)

    def reseed(self, seed: int) -> None:
        """Restart the sampling stream (e.g. from a per-iteration seed, so a
        resumed run draws the same opponents without restoring RNG state)."""
        self._gen.manual_seed(int(seed))

    def _path(self, name: str) -> str:
        if self.store_dir is None:
            raise RuntimeError("file-backed snapshot but the pool has no store_dir")
        return os.path.join(self.store_dir, name)

    def add(self, state_dict: Mapping[str, torch.Tensor], meta: Optional[Dict] = None) -> int:
        sid = self._next
        self._next += 1
        if self.store_dir is None:
            entry: Any = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        else:
            entry = f"snapshot_{sid:06d}.pt"
            torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, self._path(entry))
        self._states[sid] = (entry, dict(meta or {}))
        while len(self._states) > self.max_size:
            _, (old, _) = self._states.popitem(last=False)
            if isinstance(old, str):
                if self.defer_delete:
                    self._pending.append(old)
                else:
                    self._remove(old)
        return sid

    def _remove(self, name: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(self._path(name))

    @property
    def pending_deletes(self) -> List[str]:
        """Evicted snapshot files still on disk (``defer_delete``)."""
        return list(self._pending)

    def flush_deletes(self) -> None:
        """Delete the files of snapshots evicted since the last flush."""
        live = {e for e, _ in self._states.values() if isinstance(e, str)}
        for name in self._pending:
            if name not in live:
                self._remove(name)
        self._pending = []

    def get(self, sid: int) -> Dict[str, torch.Tensor]:
        entry = self._states[sid][0]
        if isinstance(entry, str):
            return torch.load(self._path(entry), map_location="cpu", weights_only=True)
        return entry

    def meta(self, sid: int) -> Dict:
        return self._states[sid][1]

    def sample(self, among: Optional[Sequence[int]] = None, exclude: Sequence[int] = (),
               weighted: bool = True) -> int:
        """One snapshot id by ``weights`` (uniform without them, or with
        ``weighted=False``), restricted to ``among`` and not in ``exclude``."""
        ids = self.ids()
        if among is not None:
            allowed = set(among)
            ids = [i for i in ids if i in allowed]
        if exclude:
            ids = [i for i in ids if i not in set(exclude)]
        if not ids:
            raise RuntimeError("snapshot pool is empty" if not self._states
                               else "no snapshot left to sample")
        if self.weights is None or not weighted:
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
        self._states = collections.OrderedDict()
        for sid, st, meta in state["snapshots"]:
            if isinstance(st, str) and not os.path.exists(self._path(st)):
                # evicted (file deleted) after this checkpoint was written
                warnings.warn(f"snapshot {sid}: {st} missing under {self.store_dir}; dropped")
                continue
            self._states[int(sid)] = (st, dict(meta))


class PoolPolicy:
    """Opponent playing :class:`SnapshotPool` members.

    :meth:`refresh` draws the ACTIVE SET — at most ``max_nets`` distinct
    snapshots, a uniform subset (the whole pool when it fits) — and each
    episode (``reset_streams`` call) plays one snapshot drawn from it by the
    pool's ``weights``; with no active set (first use, or all of it evicted
    from the pool) it is drawn on demand.  Call
    ``refresh`` once per iteration (``League.refresh``; ``play_episodes``
    does): with lockstep rollouts every stream is released in between, so at
    most ``max_nets`` networks are loaded and each env step makes at most
    ``max_nets`` batched calls.  (Refreshing while episodes run — autoreset
    — keeps the running episodes' snapshots loaded until they end: the cap
    is then soft.)  ``load_net(state_dict) -> nn.Module`` builds a ready
    network for a snapshot; networks outside the active set and not played
    by any stream are dropped least recently used first."""

    def __init__(self, pool: SnapshotPool, load_net: Callable[[Dict[str, torch.Tensor]], nn.Module],
                 *, act_steps: Optional[int] = None, device: Optional[torch.device] = None,
                 greedy: Optional[bool] = None, max_nets: int = 4):
        if max_nets < 1:
            raise ValueError("max_nets must be >= 1")
        self.pool = pool
        self.load_net = load_net
        self.kw = dict(act_steps=act_steps, device=device, greedy=greedy)
        self.max_nets = int(max_nets)
        self._active: List[int] = []
        self._sid: Dict[int, int] = {}                  # stream → snapshot id
        self._nets: "collections.OrderedDict[int, NetworkPolicy]" = collections.OrderedDict()

    @property
    def active(self) -> List[int]:
        return list(self._active)

    def refresh(self) -> List[int]:
        """Draw the active set (at most ``max_nets`` distinct snapshots)."""
        k = min(self.max_nets, len(self.pool))
        chosen: List[int] = []
        while len(chosen) < k:
            chosen.append(self.pool.sample(exclude=chosen, weighted=False))
        self._active = chosen
        self._evict()
        return list(chosen)

    def _drop(self, stream: int) -> None:
        old = self._sid.pop(stream, None)
        if old is not None and old in self._nets:
            self._nets[old].release_streams([stream])

    def reset_streams(self, streams: Sequence[int]) -> None:
        self._active = [i for i in self._active if i in self.pool]
        if not self._active:
            self.refresh()
        sid = self.pool.sample(among=self._active)
        for s in streams:
            s = int(s)
            self._drop(s)
            self._sid[s] = sid

    def release_streams(self, streams: Sequence[int]) -> None:
        for s in streams:
            self._drop(int(s))

    def snapshot_of(self, stream: int) -> Optional[int]:
        return self._sid.get(int(stream))

    def loaded(self) -> List[int]:
        return list(self._nets)

    def _evict(self, room: int = 0) -> None:
        keep = set(self._active) | set(self._sid.values())
        for old in list(self._nets):
            if len(self._nets) + room <= self.max_nets:
                break
            if old not in keep:
                del self._nets[old]

    def _policy(self, sid: int) -> NetworkPolicy:
        if sid in self._nets:
            self._nets.move_to_end(sid)
            return self._nets[sid]
        self._evict(room=1)
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
        parts = []
        for sid, ks in by_sid.items():
            parts.append((ks, self._policy(sid)(obs, rows[ks])))
        return _merge_parts(parts, len(streams))


def _merge_parts(parts, n: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for ks, acts in parts:
        for k, v in acts.items():
            if k not in out:
                out[k] = v.new_zeros((n,) + tuple(v.shape[1:]))
            out[k][torch.tensor(ks, dtype=torch.long, device=v.device)] = v
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
    envs whose rows start a new episode and releases the opponents' streams
    of envs that ended), then per step: run the learner on
    :meth:`learner_rows`, and :meth:`act` to merge the opponents' actions.

    Episode results: ``learner_seats`` / ``opponent_of`` /
    ``opponent_snapshot`` describe the CURRENT episode of each env.  With
    autoreset an env that ended on a step is already reassigned by the
    ``update`` of that step's VecObs: its finished episode's assignment is
    in ``ended`` ({env: {"learner_seats", "opponent", "snapshot"}}, rebuilt
    by every update) and ``prev_learner_seats`` — score it with
    ``learner_score(info["winner"], league.prev_learner_seats)``.
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
        self.opponent_snapshot: List[Optional[int]] = [None] * n
        self.prev_learner_seats = self.learner_seats.clone()
        self.ended: Dict[int, Dict[str, Any]] = {}
        self._held: Dict[int, List[int]] = {}      # env → opponent streams holding state

    # -- assignment ----------------------------------------------------------

    def _draw(self) -> float:
        return float(torch.rand((), generator=self._gen))

    def _release_env(self, i: int) -> None:
        streams = self._held.pop(i, None)
        if streams and self.opponent_of[i] is not None:
            release(self.opponents[self.opponent_of[i]], streams)

    def _assign(self, i: int, obs: VecObs) -> None:
        self._release_env(i)
        S = self.env.n_seats
        self.opponent_snapshot[i] = None
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
        theirs = ~self.learner_seats[i].to(obs.stream_seats.device)
        rows = (obs.env_id == i) & (obs.stream_seats & theirs).any(-1)
        streams = obs.stream_id[rows].tolist()
        pol = self.opponents[name]
        pol.reset_streams(streams)
        self._held[i] = streams
        snap = getattr(pol, "snapshot_of", None)
        if snap is not None and streams:
            self.opponent_snapshot[i] = snap(streams[0])

    def release_all(self) -> None:
        """Release every opponent stream still held (end of a lockstep wave,
        or an aborted one)."""
        for i in list(self._held):
            self._release_env(i)

    def refresh(self) -> None:
        """Iteration boundary: opponents with an active set (PoolPolicy)
        redraw it."""
        for pol in self.opponents.values():
            fn = getattr(pol, "refresh", None)
            if fn is not None:
                fn()

    def update(self, obs: VecObs) -> List[int]:
        """Assign the envs whose rows start a new episode (their finished
        episode's assignment goes to ``ended``) and release the opponents'
        streams of envs with no active row; returns the reassigned env ids."""
        if self.learner_seats.shape[0] != self.env.num_envs:      # env resized
            for i in list(self._held):
                self._release_env(i)
            self._alloc(self.env.num_envs)
        self.prev_learner_seats = self.learner_seats.clone()
        self.ended = {}
        new = sorted(set(obs.env_id[obs.new_episode].tolist()))
        for i in new:
            self.ended[i] = {"learner_seats": self.learner_seats[i].clone(),
                             "opponent": self.opponent_of[i],
                             "snapshot": self.opponent_snapshot[i]}
            self._assign(i, obs)
        if self._held:
            live = set(obs.env_id[obs.active].tolist())
            for i in [i for i in self._held if i not in live]:
                self._release_env(i)
        return new

    def is_mirror(self) -> torch.Tensor:
        """[N] bool: mirror episode (learner on every seat)."""
        return self.learner_seats.all(-1)

    # -- routing ---------------------------------------------------------------

    def _seats(self, obs: VecObs) -> torch.Tensor:
        """[R, S] learner seats of each row's env (on the VecObs device)."""
        return self.learner_seats.to(obs.env_id.device)[obs.env_id]

    def learner_streams(self, obs: VecObs) -> torch.Tensor:
        """[R] bool: rows whose stream the learner plays on, active or not (a
        per-seat stream of an alternating game is idle on the other seats' turns
        but still earns its seat's rewards)."""
        return (obs.stream_seats & self._seats(obs)).any(-1)

    def learner_rows(self, obs: VecObs) -> torch.Tensor:
        """[R] bool: active rows whose stream the learner plays on (its network
        must process them, acting or not)."""
        return obs.active & self.learner_streams(obs)

    def learner_acting(self, obs: VecObs) -> torch.Tensor:
        """[R] bool: active rows where the learner is to play — its action is
        sent (forced rows included: see :meth:`learner_policy_mask`)."""
        return obs.active & self._seats(obs).gather(1, obs.to_play.view(-1, 1)).squeeze(1)

    def learner_policy_mask(self, obs: VecObs) -> torch.Tensor:
        """[R] bool: the policy-loss rows — learner_acting minus ``forced``
        frames (othello passes, generals with no movable army), where the
        env applied its own action whatever was sampled.  Forced learner
        frames still carry the value loss."""
        return self.learner_acting(obs) & ~obs.forced

    def opponent_rows(self, obs: VecObs, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """([R] bool rows opponent ``name`` must process, [R] bool rows it plays)."""
        of = torch.tensor([self.opponent_of[i] == name for i in range(len(self.opponent_of))],
                          dtype=torch.bool, device=obs.env_id.device)
        env_mask = of[obs.env_id] & obs.active
        theirs = ~self._seats(obs)
        part = env_mask & (obs.stream_seats & theirs).any(-1)
        acting = env_mask & theirs.gather(1, obs.to_play.view(-1, 1)).squeeze(1)
        return part, acting

    def act(self, obs: VecObs, learner_actions: Optional[Mapping[str, torch.Tensor]]
            ) -> Dict[str, torch.Tensor]:
        """Full [R] replay-format actions for ``env.step`` (on the VecObs
        device): the learner's (``[R]``-aligned, e.g. :func:`scatter_rows`)
        where it plays, each opponent's (run on all its rows) where it plays."""
        R = obs.n_rows
        dev = obs.to_play.device
        out = {k: v.to(dev).clone()
               for k, v in self.env.replay_actions(learner_actions or {}).items()}
        for name, pol in self.opponents.items():
            part, acting = self.opponent_rows(obs, name)
            rows = part.nonzero().squeeze(1)
            if rows.numel() == 0:
                continue
            acts = self.env.replay_actions(pol(obs, rows))
            use = acting[rows]
            for k, v in acts.items():
                v = v.to(dev)
                if k not in out:
                    out[k] = v.new_zeros((R,) + tuple(v.shape[1:]))
                out[k][rows[use]] = v[use]
        return out

    def learner_score(self, winner: torch.Tensor,
                      learner_seats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[N] learner result per env from ``info["winner"]``: 1 win, 0.5 draw,
        0 loss; NaN for mirror episodes (the learner plays both sides).
        ``learner_seats`` defaults to the current assignment (pass
        ``prev_learner_seats`` after an autoreset ``update``)."""
        seats = self.learner_seats if learner_seats is None else learner_seats.cpu()
        w = winner.long().cpu()
        won = torch.zeros(len(w), dtype=torch.bool)
        ok = w >= 0
        won[ok] = seats[ok.nonzero().squeeze(1), w[ok]]
        score = torch.where(ok, won.double(), torch.full((len(w),), 0.5, dtype=torch.float64))
        return torch.where(seats.all(-1), torch.full_like(score, float("nan")), score)

    # -- checkpointing -----------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {"gen": self._gen.get_state(), "learner_seats": self.learner_seats.clone(),
                "opponent_of": list(self.opponent_of),
                "opponent_snapshot": list(self.opponent_snapshot)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._gen.set_state(state["gen"])
        self.learner_seats = state["learner_seats"].clone()
        self.opponent_of = list(state["opponent_of"])
        self.opponent_snapshot = list(state.get("opponent_snapshot",
                                                [None] * len(self.opponent_of)))


def play_episodes(env: VecGameEnv, league: League, learner: Policy, *,
                  max_steps: Optional[int] = None) -> Dict[str, Any]:
    """Reset every env and play the episodes out in lockstep (evaluation /
    smoke runs; needs ``autoreset`` off): the learner policy on its seats,
    the league's opponents on theirs.  Streams are released as their envs
    end.  Returns ``winner`` [N] (seat, −1 none), ``score`` [N]
    (:meth:`League.learner_score`), ``learner_seats`` [N, S], ``opponent``
    (names, None for mirror), ``opponent_snapshot`` (PoolPolicy snapshot
    ids), ``episode_len`` [N] and ``steps``."""
    if env.autoreset:
        raise ValueError("play_episodes needs autoreset off")
    obs = env.reset()
    learner.reset_streams(obs.stream_id.tolist())
    league.refresh()
    league.update(obs)          # resets the drawn opponents' streams
    N = env.num_envs
    winner = torch.full((N,), -1, dtype=torch.long)
    episode_len = torch.zeros(N, dtype=torch.long)
    steps = 0
    try:
        while bool(obs.active.any()) and (max_steps is None or steps < max_steps):
            rows = league.learner_rows(obs).nonzero().squeeze(1)
            la = scatter_rows(learner(obs, rows), rows, obs.n_rows) if rows.numel() else {}
            was_live = live_envs(obs)
            obs, _, done, info = env.step(league.act(obs, la))
            winner = torch.where(done, info["winner"], winner)
            episode_len = torch.where(done, info["episode_len"], episode_len)
            league.update(obs)
            # streams of envs whose episode ended (a per-seat row that merely waits
            # for another seat's turn keeps its state)
            ended = was_live - live_envs(obs)
            if ended:
                release(learner, env_streams(obs, ended))
            steps += 1
    finally:
        # max_steps cut-offs / errors: nothing keeps per-stream state
        release(learner, obs.stream_id.tolist())
        for i in list(league._held):
            league._release_env(i)
    return {"winner": winner, "score": league.learner_score(winner),
            "learner_seats": league.learner_seats.clone(),
            "opponent": list(league.opponent_of),
            "opponent_snapshot": list(league.opponent_snapshot),
            "episode_len": episode_len, "steps": torch.tensor(steps)}
