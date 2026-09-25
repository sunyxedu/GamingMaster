"""Single-node data parallelism for trainers that synchronise gradients by
hand (training/train_multi_game.py) instead of wrapping the model in DDP.

Launch with ``torchrun --standalone --nproc_per_node=N -m training.<module>``.
torchrun exports RANK / LOCAL_RANK / WORLD_SIZE (and MASTER_ADDR/PORT for the
default ``env://`` rendezvous).  With WORLD_SIZE unset or 1 nothing here
creates a process group, so the single-process code path stays untouched.

Why not DistributedDataParallel: a joint multi-game step runs a different
subset of the model on every rank (only the games that rank drew), recurrent
cores sample their depth per forward, and a step is several micro-batches of
different games — the reducer's fixed bucket order / ``find_unused_parameters``
machinery fits none of that.  One explicit, rank-invariant gradient sync per
optimizer step (:func:`model.rl_utils.sync_gradients`) is correct for any graph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

#: seed offset between data-parallel ranks, a large prime: the ranks of the
#: runs ``--seed s`` and ``--seed s+1`` (a seed sweep) share no stream, as
#: they would with ``seed + rank`` (rank 0 keeps ``seed`` itself, i.e. the
#: single-process streams)
RANK_SEED_STRIDE = 1_000_003
#: extra offset per optimizer step of a resumed run (fresh streams after a
#: resume instead of a replay of the first steps' game draws and data order)
RESUME_SEED_STRIDE = 1_000_000_007


def rank_seed(seed: int, rank: int, resume_step: int = 0) -> int:
    """Seed of data-parallel rank ``rank`` (``rank_seed(s, 0) == s``)."""
    return seed + RANK_SEED_STRIDE * rank + RESUME_SEED_STRIDE * resume_step


@dataclass(frozen=True)
class DistEnv:
    """This process's place in the job (``world == 1``: no process group)."""

    rank: int = 0
    local_rank: int = 0
    world: int = 1

    @property
    def enabled(self) -> bool:
        return self.world > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def destroy(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def dist_env_from_environ() -> DistEnv:
    """Read the torchrun environment (WORLD_SIZE > 1 means data parallel)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return DistEnv()
    rank = int(os.environ["RANK"])
    return DistEnv(rank=rank, local_rank=int(os.environ.get("LOCAL_RANK", rank)), world=world)


def init_distributed(device: torch.device, timeout_min: float = 30.0
                     ) -> Tuple[DistEnv, torch.device]:
    """Join the torchrun job, if any.

    ``device`` is the device the caller would use single-process; with a
    process group it becomes ``cuda:LOCAL_RANK`` (NCCL) or stays ``cpu``
    (gloo).  Other device types cannot run a process group here.
    """
    env = dist_env_from_environ()
    if not env.enabled:
        return env, device
    if device.type == "cuda":
        device = torch.device("cuda", env.local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    elif device.type == "cpu":
        backend = "gloo"
    else:
        raise SystemExit(f"data-parallel training runs on cuda (NCCL) or cpu (gloo), "
                         f"not {device.type}; pass --device cpu or cuda")
    dist.init_process_group(backend, timeout=timedelta(minutes=timeout_min),
                            device_id=device if device.type == "cuda" else None)
    return env, device


@torch.no_grad()
def broadcast_module_state(module: nn.Module, src: int = 0) -> None:
    """Overwrite every parameter and buffer with rank ``src``'s values
    (shared/aliased tensors are visited once; the order is the module's
    registration order, identical on every rank)."""
    for t in chain(module.parameters(), module.buffers()):
        dist.broadcast(t.data, src=src)


class BatchNormStatSync:
    """Keep BatchNorm running statistics identical across ranks.

    Running statistics are forward-side state: each rank updates them on its
    own micro-batches only (and only for the games it drew), so without a
    sync they drift apart and the checkpoint would carry rank 0's view.
    Called once per optimizer step on every rank, this merges them.

    Per layer, let P be the statistics at the step start (identical on every
    rank), n_r the forwards rank r ran through the layer during the step,
    N = Σ_r n_r, and d(n) the weight P keeps after n updates: (1-m)^n for
    momentum m, c0/(c0+n) for the cumulative average (``momentum=None``,
    c0 = ``num_batches_tracked`` at the step start).  Rank r's local
    statistics are L_r = d(n_r)·P + (1-d(n_r))·μ_r, μ_r a weighted mean of
    its batch statistics; the merge recovers each μ_r and sets

        running ← d(N)·P + (1-d(N))·Σ_r n_r·μ_r / N
        num_batches_tracked ← c0 + N

    i.e. what one process holds after the same N forwards — exactly for the
    cumulative average, up to the order of the forwards for an EMA (within
    the step's innovation the batches weigh uniformly instead of
    geometrically, an order no parallel schedule reproduces).  P keeps
    weight (1-m)^N, as single-process, so the momentum per micro-batch is
    unchanged whatever the world size.  With ONE updating rank the result is
    that rank's statistics verbatim (bit-exact single-process); with none
    they stay P.  One all-reduce, no host synchronisation.  Only eval-mode
    forwards (validation, inference) read these statistics; the training
    loss uses batch statistics.
    """

    def __init__(self, module: nn.Module):
        self.bns = [m for m in module.modules()
                    if isinstance(m, nn.modules.batchnorm._BatchNorm)
                    and m.track_running_stats and m.running_mean is not None]
        self._snapshot()

    def _snapshot(self) -> None:
        self.prev = [(m.running_mean.to(torch.float64, copy=True),
                      m.running_var.to(torch.float64, copy=True),
                      m.num_batches_tracked.clone()) for m in self.bns]

    @staticmethod
    def _keep(m: nn.Module, n: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        """d(n): the weight the step-start statistics keep after n updates."""
        if m.momentum is None:
            c0 = c0.to(torch.float64)
            return c0 / (c0 + n).clamp(min=1.0)
        return torch.pow(torch.full_like(n, 1.0 - m.momentum), n)

    @torch.no_grad()
    def __call__(self) -> None:
        if not self.bns:
            return
        parts = []
        for m, (p_mean, p_var, c0) in zip(self.bns, self.prev):
            n = (m.num_batches_tracked - c0).to(torch.float64).view(1)
            act = n > 0
            d = self._keep(m, n, c0)
            w = 1.0 - d                       # 0 only for momentum 0 (no update)
            ok = act & (w > 0)
            safe_w = torch.where(ok, w, torch.ones_like(w))
            loc_mean = m.running_mean.to(torch.float64)
            loc_var = m.running_var.to(torch.float64)
            # where, not a 0-multiply: a non-updating rank contributes an exact 0
            parts += [torch.where(act, loc_mean, 0.0),
                      torch.where(act, loc_var, 0.0),
                      torch.where(ok, n * (loc_mean - d * p_mean) / safe_w, 0.0),
                      torch.where(ok, n * (loc_var - d * p_var) / safe_w, 0.0),
                      act.to(torch.float64), n]
        flat = torch.cat(parts)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        i = 0
        for m, (p_mean, p_var, c0) in zip(self.bns, self.prev):
            c = m.running_mean.numel()
            s_mean, s_var, s_imean, s_ivar = (flat[i + k * c:i + (k + 1) * c] for k in range(4))
            n_act, n_tot = flat[i + 4 * c], flat[i + 4 * c + 1]
            i += 4 * c + 2
            d_tot = self._keep(m, n_tot.view(1), c0)
            n_div = n_tot.clamp(min=1.0)
            for buf, s_loc, s_inno, prev in ((m.running_mean, s_mean, s_imean, p_mean),
                                             (m.running_var, s_var, s_ivar, p_var)):
                merged = d_tot * prev + (1.0 - d_tot) * (s_inno / n_div)
                buf.copy_(torch.where(n_act == 1, s_loc,
                                      torch.where(n_act > 1, merged, buf.to(torch.float64))))
            m.num_batches_tracked.copy_(c0 + n_tot.to(torch.long))
        self._snapshot()


@torch.no_grad()
def collective_clip_grad_norm_(params: Iterable[torch.Tensor], max_norm: float,
                               device: torch.device) -> torch.Tensor:
    """``clip_grad_norm_`` for gradients that were just synchronised: the
    total norm is agreed by one MAX all-reduce (a non-finite norm on any rank
    becomes +inf everywhere), so the step-drop decision and the clip
    coefficient are identical on every rank by construction, not only
    because identical inputs give identical kernels.  Returns the agreed norm
    (float32 on ``device``); clips only when it is finite."""
    params = list(params)
    grads = [p.grad for p in params if p.grad is not None]
    norm = (torch.nn.utils.get_total_norm(grads) if grads
            else torch.zeros((), device=device))
    norm = norm.to(device=device, dtype=torch.float32)
    agreed = torch.where(torch.isfinite(norm), norm, torch.full_like(norm, float("inf")))
    dist.all_reduce(agreed, op=dist.ReduceOp.MAX)
    if torch.isfinite(agreed):
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, agreed)
    return agreed
