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
    Called once per optimizer step on every rank, this merges them:

        running_mean/var ← mean over the ranks that updated this layer during
                           the step (unchanged if none did — then they are
                           still identical everywhere)
        num_batches_tracked ← previous value + Σ_r forwards on rank r

    With one updating rank this is exactly that rank's statistics (what a
    single process would hold after the same forwards); with several it is
    the convex average of their EMAs (an EMA over the step's world·K
    micro-batches up to the within-step ordering, which EMAs are sensitive
    to and no parallel schedule can reproduce).  One all-reduce, no host
    synchronisation.  Only eval-mode forwards (validation, inference) read
    these statistics; the training loss uses batch statistics.
    """

    def __init__(self, module: nn.Module):
        self.bns = [m for m in module.modules()
                    if isinstance(m, nn.modules.batchnorm._BatchNorm)
                    and m.track_running_stats and m.running_mean is not None]
        self.counts = [m.num_batches_tracked.clone() for m in self.bns]

    @torch.no_grad()
    def __call__(self) -> None:
        if not self.bns:
            return
        parts = []
        for m, c0 in zip(self.bns, self.counts):
            n = (m.num_batches_tracked - c0).to(torch.float64).view(1)
            act = n > 0
            # where, not a 0-multiply: a non-updating rank contributes an exact 0
            parts += [torch.where(act, m.running_mean.to(torch.float64), 0.0),
                      torch.where(act, m.running_var.to(torch.float64), 0.0),
                      act.to(torch.float64), n]
        flat = torch.cat(parts)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        i = 0
        for j, m in enumerate(self.bns):
            c = m.running_mean.numel()
            s_mean, s_var = flat[i:i + c], flat[i + c:i + 2 * c]
            n_act, n_tot = flat[i + 2 * c], flat[i + 2 * c + 1]
            i += 2 * c + 2
            denom = n_act.clamp(min=1.0)
            m.running_mean.copy_(torch.where(n_act > 0, s_mean / denom,
                                             m.running_mean.to(torch.float64)))
            m.running_var.copy_(torch.where(n_act > 0, s_var / denom,
                                            m.running_var.to(torch.float64)))
            m.num_batches_tracked.copy_(self.counts[j] + n_tot.to(torch.long))
            self.counts[j] = m.num_batches_tracked.clone()


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
