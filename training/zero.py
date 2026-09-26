"""ZeRO-1-style sharding of the optimizer state and the EMA over the
data-parallel ranks (training/train_multi_game.py ``--shard_optimizer``).

Plain data parallelism keeps on EVERY rank the fp32 weights (4 B per
parameter), their gradients (4), AdamATan2's two moments (8) and the EMA
shadow (4): 20 B per parameter.  The forward/backward needs only the first
two; the optimizer state and the EMA are touched elementwise, one parameter
at a time.  Here the trainable parameters are partitioned over the ranks
(stage 1 of ZeRO, Rajbhandari et al. 2020, arXiv 1910.02054) and each rank
keeps the moments, step counters and EMA shadows of ITS parameters only:
8 + 12/W B per parameter instead of 20.

One optimizer step, after model.rl_utils.sync_gradients and the collective
grad-norm clip (both unchanged: every rank holds the full averaged, clipped
gradient):

1. every rank drops the gradients of the parameters it does not own
   (``grad = None``) and runs the unchanged AdamATan2 step, i.e. the same
   elementwise kernel on the same whole tensors as the unsharded run, on its
   own partition; a parameter whose gradient is None on all ranks is skipped
   by its owner exactly as single-process (no state, no step count, no
   weight decay);
2. every stepped parameter is broadcast from its owner (bucketed per owner,
   dtype and ``bucket_mb``; broadcast is gloo-capable on CUDA tensors, the
   MIG-slice mode), so every rank ends the step with identical full weights;
3. the EMA update runs on each rank's own shadows (``ModelEMA(only=...)``).

The owner computes exactly what every rank computes without sharding, and a
broadcast copies bytes: weights, optimizer state and EMA (gathered) are
bit-identical to the unsharded data-parallel run.  gloo has no
reduce_scatter, so the gradient sync stays an all-reduce; a reduce-to-owner
would also sum in a different order than the all-reduce for W > 2.

Gathering (every rank must call these, in the same order):

- :meth:`Zero1.ema_weights` — evaluation on the full EMA: a context manager
  that swaps the EMA into the live model (each owner stashes its partition
  of the training weights, the owners broadcast their shadows into the
  parameters) and restores the training weights and the buffers exactly on
  exit — instead of a deep copy of the model;
- :meth:`Zero1.gather_ema` / :meth:`Zero1.gather_optimizer_state` — the full
  ``ModelEMA.state_dict()`` / ``optimizer.state_dict()`` in the unsharded
  layout (the checkpoint format is unchanged), assembled on one rank, on the
  CPU, one bucket at a time;
- :meth:`Zero1.load_optimizer_state` loads a full (unsharded-layout) state
  dict keeping this rank's partition, so sharded and unsharded runs resume
  each other at any world size.

Partition (:func:`partition_by_numel`): whole tensors, greedy
largest-first onto the least-loaded rank, over the optimizer's parameters
(every shared core tensor once).  It depends only on the tensor sizes and the
world size: stable across runs and resumes, independent of the games a step
drew.  (The production core's largest tensor, one layer's routed-expert
``w_gate_up``, is 12.6M of 212M parameters at 32 experts and 25.2M of 364M at
64, below the 17.7M / 30.3M per-rank mean on 12 ranks, so whole tensors
balance within a few percent; see scripts/zero1_memory.py.)
"""

from __future__ import annotations

import contextlib
import heapq
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from model.optim import ModelEMA

#: AdamATan2's per-parameter state tensors (model.optim), besides "step"
MOMENT_KEYS = ("exp_avg", "exp_avg_sq")


def partition_by_numel(numels: Sequence[int], world: int) -> List[int]:
    """Owner rank of each of the tensors of sizes ``numels``: largest first
    (equal sizes in index order), each onto the least-loaded rank (ties: the
    lower rank) — the greedy LPT schedule.  Every rank's load is at most the
    mean plus the largest tensor.  Deterministic: a function of ``numels``
    and ``world`` only."""
    if world < 1:
        raise ValueError(f"world size must be >= 1, got {world}")
    owner = [0] * len(numels)
    heap = [(0, r) for r in range(world)]
    for i in sorted(range(len(numels)), key=lambda i: (-int(numels[i]), i)):
        load, r = heapq.heappop(heap)
        owner[i] = r
        heapq.heappush(heap, (load + int(numels[i]), r))
    return owner


def rank_loads(numels: Sequence[int], owner: Sequence[int], world: int) -> List[int]:
    """Parameters (elements) owned by each rank."""
    loads = [0] * world
    for n, r in zip(numels, owner):
        loads[r] += int(n)
    return loads


def state_bytes_per_param(ema: bool, element_size: int = 4) -> int:
    """Bytes of optimizer state + EMA per parameter: AdamATan2's two moments,
    plus the EMA shadow."""
    return (len(MOMENT_KEYS) + int(ema)) * element_size


class Zero1:
    """The partition of ``optimizer``'s parameters over the ``world`` ranks
    and the sharded step / gather / load operations (module docstring).

    Construct it on every rank right after the optimizer (before any step
    or state load) and create the EMA with :meth:`make_ema`.  Every
    parameter of ``optimizer`` must be a parameter of ``model``.
    ``bucket_mb`` bounds each broadcast's flat buffer (0 = one broadcast per
    tensor); a tensor larger than a bucket is broadcast in place, alone."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, rank: int,
                 world: int, bucket_mb: float = 25.0):
        if world < 2:
            raise ValueError("ZeRO-1 sharding needs a data-parallel run (world size > 1)")
        self.optimizer, self.rank, self.world = optimizer, rank, world
        #: the optimizer's parameters in state_dict index order
        self.params: List[nn.Parameter] = [p for g in optimizer.param_groups
                                           for p in g["params"]]
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError("a parameter appears twice in the optimizer's param groups")
        self.index = {id(p): i for i, p in enumerate(self.params)}
        self.owner = partition_by_numel([p.numel() for p in self.params], world)
        self.owned = [i for i, r in enumerate(self.owner) if r == rank]
        self.cap_bytes = int(bucket_mb * 2 ** 20)
        # EMA names: ModelEMA keys its shadows by each parameter's first name
        # in model.named_parameters() (shared core tensors once) and maps every
        # alias of it to that name
        canon: Dict[int, str] = {}
        for n, p in model.named_parameters():
            if id(p) in self.index:
                canon.setdefault(id(p), n)
        missing = [i for i, p in enumerate(self.params) if id(p) not in canon]
        if missing:
            raise ValueError(f"{len(missing)} optimizer parameter(s) are not parameters "
                             f"of the model")
        self.names = [canon[id(p)] for p in self.params]
        #: (EMA name, parameter index) in ModelEMA.state_dict() order
        self.ema_order: List[Tuple[str, int]] = [
            (n, self.index[id(p)]) for n, p in model.named_parameters()
            if p.requires_grad and id(p) in self.index]
        self.alias: Dict[str, str] = {
            n: canon[id(p)] for n, p in model.named_parameters(remove_duplicate=False)
            if id(p) in canon}

    # ------------------------------------------------------------------
    # memory
    # ------------------------------------------------------------------

    def loads(self) -> List[int]:
        """Parameters (elements) whose state each rank keeps."""
        return rank_loads([p.numel() for p in self.params], self.owner, self.world)

    def describe(self, ema: bool) -> str:
        """One log line: the per-rank state sizes, sharded vs unsharded."""
        per = state_bytes_per_param(ema)
        loads = self.loads()
        total = sum(loads)
        mib = 2 ** 20
        return (f"[zero1] optimizer state{' + EMA' if ema else ''} sharded over "
                f"{self.world} ranks: {min(loads):,}-{max(loads):,} of {total:,} parameters "
                f"per rank, {max(loads) * per / mib:.1f} MiB max per rank instead of "
                f"{total * per / mib:.1f} MiB (weights + gradients stay full: "
                f"{total * 8 / mib:.1f} MiB)")

    def local_state_bytes(self, ema: Optional[ModelEMA] = None) -> int:
        """Bytes of the optimizer state tensors (and EMA shadows) this rank
        actually holds."""
        n = sum(t.numel() * t.element_size() for st in self.optimizer.state.values()
                for t in st.values() if torch.is_tensor(t))
        if ema is not None:
            n += sum(t.numel() * t.element_size() for t in ema.shadow.values())
        return n

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def make_ema(self, model: nn.Module, mu: float) -> ModelEMA:
        """The EMA of this rank's partition (a partial ModelEMA)."""
        ema = ModelEMA(model, mu=mu, only={id(self.params[i]) for i in self.owned})
        want = {self.names[i] for i in self.owned}
        if set(ema.shadow) != want:
            raise ValueError("the EMA shadows do not match this rank's partition "
                             "(a frozen parameter in the optimizer?)")
        return ema

    def ema_model_state(self, model: nn.Module,
                        shadow: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """``ModelEMA.ema_model_state`` from a gathered full ``shadow``
        (:meth:`gather_ema`): ``model.state_dict()`` with the EMA values of
        the trainable parameters.  Every alias of a shared core tensor refers
        to the same (CPU) tensor instead of a clone per alias."""
        return {k: shadow[self.alias[k]] if k in self.alias else v
                for k, v in model.state_dict().items()}

    @contextlib.contextmanager
    def ema_weights(self, model: nn.Module, ema: ModelEMA) -> Iterator[nn.Module]:
        """Within the block ``model`` holds the full EMA weights (every rank);
        on exit the training weights and every buffer are restored bit for
        bit.  Extra memory: this rank's partition of the weights, a copy of
        the buffers, one broadcast bucket (no copy of the model; call it
        after ``zero_grad(set_to_none=True)``, the gradients are not needed).
        An exception inside the block
        leaves the EMA weights in place (the run is over anyway: the ranks
        are no longer in step)."""
        stash = {i: self.params[i].detach().clone() for i in self.owned}
        buffers = [b.detach().clone() for b in model.buffers()]
        with torch.no_grad():
            for i in self.owned:
                self.params[i].copy_(ema.shadow[self.names[i]])
        self._broadcast_params(range(len(self.params)))
        yield model
        with torch.no_grad():
            for i in self.owned:
                self.params[i].copy_(stash.pop(i))
            for b, saved in zip(model.buffers(), buffers):
                b.copy_(saved)
        del buffers
        self._broadcast_params(range(len(self.params)))

    def gather_ema(self, ema: ModelEMA, dst: int = 0) -> Optional[Dict[str, torch.Tensor]]:
        """The full ``ModelEMA.state_dict()`` (CPU tensors) on rank ``dst``,
        None on the others."""
        got = self._gather(range(len(self.params)), lambda i: ema.shadow[self.names[i]], dst)
        if self.rank != dst:
            return None
        return {n: got[i] for n, i in self.ema_order}

    # ------------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self) -> None:
        """The sharded optimizer step (module docstring, 1-2).  Call it on
        every rank where the unsharded run calls ``optimizer.step()``: after
        ``sync_gradients``, which leaves every parameter with a gradient on
        all ranks or None on all, so every rank broadcasts the same set."""
        stepped = [i for i, p in enumerate(self.params) if p.grad is not None]
        for i in stepped:
            if self.owner[i] != self.rank:
                self.params[i].grad = None
        self.optimizer.step()
        self._broadcast_params(stepped)

    def gather_optimizer_state(self, dst: int = 0) -> Optional[Dict]:
        """The full ``optimizer.state_dict()`` of the unsharded run (state of
        every parameter the optimizer ever stepped, on the CPU) on rank
        ``dst``, None on the others."""
        opt = self.optimizer
        code = [0] * len(self.params)          # step + 1 where this rank holds state
        for i in self.owned:
            st = opt.state.get(self.params[i])
            if st:
                if set(st) != {"step", *MOMENT_KEYS}:
                    raise RuntimeError(f"unexpected optimizer state keys {sorted(st)}")
                code[i] = int(st["step"]) + 1
        dev = self.params[0].device if self.params else torch.device("cpu")
        codes = torch.tensor(code, dtype=torch.int64, device=dev)
        dist.all_reduce(codes, op=dist.ReduceOp.SUM)   # one owner per entry
        code = codes.tolist()
        has = [i for i, c in enumerate(code) if c > 0]
        moments = {k: self._gather(has, lambda i, k=k: opt.state[self.params[i]][k], dst)
                   for k in MOMENT_KEYS}
        if self.rank != dst:
            return None
        sd = opt.state_dict()        # param_groups: identical on every rank
        sd["state"] = {i: {"step": code[i] - 1, **{k: moments[k][i] for k in MOMENT_KEYS}}
                       for i in has}
        return sd

    def load_optimizer_state(self, state_dict: Dict) -> None:
        """Load a full (unsharded-layout) ``optimizer.state_dict()``,
        keeping this rank's partition only."""
        mine = set(self.owned)
        self.optimizer.load_state_dict(
            {**state_dict, "state": {int(i): s for i, s in state_dict["state"].items()
                                     if int(i) in mine}})

    # ------------------------------------------------------------------
    # collectives
    # ------------------------------------------------------------------

    def _buckets(self, idxs: Iterable[int]) -> List[Tuple[int, List[int]]]:
        """(owner, parameter indices) buckets of ``idxs``: per owner in rank
        order, same-dtype runs in index order of at most ``cap_bytes`` (a
        larger tensor alone).  The layout depends only on ``idxs`` and the
        partition, so it is the same on every rank."""
        by_owner: Dict[int, List[int]] = {r: [] for r in range(self.world)}
        for i in sorted(set(idxs)):
            by_owner[self.owner[i]].append(i)
        out: List[Tuple[int, List[int]]] = []
        size: Dict[int, int] = {}                  # bucket (index into out) -> bytes
        for r in range(self.world):
            open_: Dict[torch.dtype, int] = {}     # dtype -> its open bucket
            for i in by_owner[r]:
                p = self.params[i]
                nbytes = p.numel() * p.element_size()
                j = open_.get(p.dtype)
                if j is None or size[j] + nbytes > self.cap_bytes:
                    j = open_[p.dtype] = len(out)
                    out.append((r, []))
                    size[j] = 0
                out[j][1].append(i)
                size[j] += nbytes
        return out

    def _recv_buffer(self, bucket: List[int]) -> torch.Tensor:
        p = self.params[bucket[0]]
        return torch.empty(sum(self.params[i].numel() for i in bucket), dtype=p.dtype,
                           device=p.device)

    @torch.no_grad()
    def _broadcast_params(self, idxs: Iterable[int]) -> None:
        """Every rank's parameters ``idxs`` ← their owners' values."""
        for r, bucket in self._buckets(idxs):
            ps = [self.params[i] for i in bucket]
            if len(ps) == 1 and ps[0].is_contiguous():
                dist.broadcast(ps[0].data, src=r)          # in place, no staging copy
                continue
            flat = (_flatten_dense_tensors([p.data for p in ps]) if self.rank == r
                    else self._recv_buffer(bucket))
            dist.broadcast(flat, src=r)
            if self.rank != r:
                for p, v in zip(ps, _unflatten_dense_tensors(flat, ps)):
                    p.data.copy_(v)
            del flat                    # one bucket alive at a time

    @torch.no_grad()
    def _gather(self, idxs: Iterable[int], get: Callable[[int], torch.Tensor],
                dst: int) -> Dict[int, torch.Tensor]:
        """{i: CPU copy of owner(i)'s ``get(i)``} on rank ``dst`` (empty
        elsewhere).  ``get(i)`` has parameter i's shape and dtype.  Buckets
        owned by ``dst`` need no transfer; the others are broadcast from
        their owners (gloo-capable on CUDA tensors), one at a time."""
        out: Dict[int, torch.Tensor] = {}
        for r, bucket in self._buckets(idxs):
            if r == dst:
                if self.rank == dst:
                    out.update((i, get(i).detach().to("cpu", copy=True)) for i in bucket)
                continue
            if self.rank == r:
                ts = [get(i).detach() for i in bucket]
                # 1-D on every rank; a lone contiguous tensor is sent as a view
                flat = (ts[0].view(-1) if len(ts) == 1 and ts[0].is_contiguous()
                        else _flatten_dense_tensors(ts))
            else:
                flat = self._recv_buffer(bucket)
            dist.broadcast(flat, src=r)
            if self.rank == dst:
                ps = [self.params[i] for i in bucket]
                out.update((i, v.to("cpu", copy=True))
                           for i, v in zip(bucket, _unflatten_dense_tensors(flat, ps)))
            del flat                    # one bucket alive at a time
        return out
