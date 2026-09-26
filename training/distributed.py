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

MIG slices.  A CUDA process can use ONE MIG instance only, and NCCL refuses
two ranks on instances of the same physical GPU ("Duplicate GPU detected").
So on a MIG allocation (CUDA_VISIBLE_DEVICES=MIG-<uuid>,MIG-<uuid>,...) every
torchrun rank first narrows CUDA_VISIBLE_DEVICES to its own entry
(:func:`pin_rank_device`, before anything initialises CUDA — the trainers
call it first thing), uses ``cuda:0``, and the ranks talk over gloo, which
runs every collective of the trainers on CUDA tensors (staged through pinned
host memory) — ``--dist_backend auto`` picks it.  Collectives in use, all
gloo-capable: all_reduce SUM / MAX of float32/float64/int tensors (CUDA or
CPU), broadcast (module state, a 1-element flag), all_gather_object /
broadcast_object_list (pickled, always on the CPU), barrier; no ReduceOp.AVG,
reduce_scatter, coalesced or async collectives, so no op needs a fallback.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from typing import Iterable, List, MutableMapping, Optional, Sequence, Tuple

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


# ----------------------------------------------------------------------
# MIG slices and the backend choice
# ----------------------------------------------------------------------

#: CUDA_VISIBLE_DEVICES entry prefix of a MIG instance ("MIG-<uuid>", or the
#: legacy "MIG-GPU-<gpu uuid>/<gi>/<ci>")
MIG_PREFIX = "MIG-"
#: --dist_backend choices (auto: gloo on MIG slices or without CUDA, else NCCL)
DIST_BACKENDS = ("auto", "nccl", "gloo")
#: set by :func:`pin_rank_device`: the allocation's full CUDA_VISIBLE_DEVICES
MIG_ALL_DEVICES_ENV = "GM_MIG_VISIBLE_DEVICES"


def visible_devices(value: Optional[str]) -> List[str]:
    """The entries of a CUDA_VISIBLE_DEVICES value (unset or empty: [])."""
    return [e.strip() for e in (value or "").split(",") if e.strip()]


def is_mig(entries: Sequence[str]) -> bool:
    """True when a CUDA_VISIBLE_DEVICES entry list names a MIG instance."""
    return any(e.startswith(MIG_PREFIX) for e in entries)


def rank_visible_devices(cvd: Optional[str], local_rank: Optional[str]) -> Optional[str]:
    """The CUDA_VISIBLE_DEVICES value torchrun rank ``local_rank`` must use,
    or None to keep ``cvd`` as it is.

    Only a list of MORE than one device with a MIG instance among them is
    narrowed, to entry ``local_rank`` (a process can use one MIG instance:
    left alone, every rank would run on the first).  Unset / empty values,
    plain GPU indices, GPU UUIDs, a single MIG instance (already narrowed:
    the call is idempotent) or no ``local_rank`` (not a torchrun rank) are
    kept.  A ``local_rank`` that is not an index of the list raises
    ValueError."""
    entries = visible_devices(cvd)
    if local_rank is None or not local_rank.strip() or len(entries) <= 1 or not is_mig(entries):
        return None
    try:
        r = int(local_rank)
    except ValueError:
        raise ValueError(f"LOCAL_RANK={local_rank!r} is not an integer") from None
    if not 0 <= r < len(entries):
        raise ValueError(
            f"LOCAL_RANK={r} but CUDA_VISIBLE_DEVICES lists {len(entries)} device(s) with "
            f"MIG instances ({cvd}): a process can use one MIG instance, so run at most "
            f"{len(entries)} rank(s) (torchrun --nproc_per_node) on this allocation")
    return entries[r]


def _cuda_driver_initialised() -> bool:
    """Best effort: True if this process already ran cuInit (then the driver
    has read CUDA_VISIBLE_DEVICES for good).  Side-effect free: libcuda is
    only inspected if something already loaded it (RTLD_NOLOAD), and
    cuDeviceGetCount returns CUDA_ERROR_NOT_INITIALIZED until cuInit."""
    if not hasattr(os, "RTLD_NOLOAD"):
        return False
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(name, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError:
            continue                    # never loaded: nothing initialised it
        count = ctypes.c_int(0)
        try:
            return lib.cuDeviceGetCount(ctypes.byref(count)) == 0
        except AttributeError:
            return False
    return False


def cuda_initialised() -> Optional[str]:
    """Why changing CUDA_VISIBLE_DEVICES can no longer take effect in this
    process, or None while it still can."""
    if torch.cuda.is_initialized():
        return "torch.cuda is already initialised"
    if _cuda_driver_initialised():
        return ("the CUDA driver is already initialised (cuInit ran, e.g. through "
                "torch.cuda.is_available() / device_count() on MIG devices)")
    return None


def pin_rank_device(environ: Optional[MutableMapping[str, str]] = None) -> Optional[str]:
    """Narrow CUDA_VISIBLE_DEVICES to this torchrun rank's own MIG instance
    (:func:`rank_visible_devices` of CUDA_VISIBLE_DEVICES and LOCAL_RANK);
    returns the entry it pinned, or None when nothing changes.

    Call it before anything touches CUDA — the trainers do, first thing in
    ``main`` (``import torch`` alone does not initialise CUDA; the first
    ``torch.cuda.is_available()`` does).  Too late, the rewrite would be
    silently ignored and every rank would share the first slice, so that
    raises instead.  The full list stays in ``$GM_MIG_VISIBLE_DEVICES``.
    ``environ`` (tests) replaces ``os.environ`` and skips the CUDA check."""
    env = os.environ if environ is None else environ
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    local_rank = env.get("LOCAL_RANK")
    try:
        new = rank_visible_devices(cvd, local_rank)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if new is None:
        return None
    if environ is None:
        why = cuda_initialised()
        if why is not None:
            raise RuntimeError(
                f"cannot pin LOCAL_RANK={local_rank} to its MIG instance: {why} with "
                f"CUDA_VISIBLE_DEVICES={cvd}, so every rank would run on the first slice. "
                f"Call training.distributed.pin_rank_device() before anything touches CUDA.")
    env[MIG_ALL_DEVICES_ENV] = cvd
    env["CUDA_VISIBLE_DEVICES"] = new
    return new


def visible_mig(environ: Optional[MutableMapping[str, str]] = None) -> bool:
    """True when this process's CUDA devices are MIG instances (pinned or not)."""
    env = os.environ if environ is None else environ
    return is_mig(visible_devices(env.get("CUDA_VISIBLE_DEVICES")))


def resolve_backend(requested: str, device_type: str, mig: bool) -> str:
    """The process-group backend for ``--dist_backend requested`` on a
    ``device_type`` device: ``auto`` is NCCL on CUDA — unless the devices are
    MIG instances (NCCL refuses two ranks on one physical GPU) — and gloo on
    the CPU; ``gloo`` also runs on CUDA tensors; ``nccl`` needs CUDA."""
    if requested not in DIST_BACKENDS:
        raise SystemExit(f"--dist_backend {requested!r}: expected one of {DIST_BACKENDS}")
    if device_type not in ("cuda", "cpu"):
        raise SystemExit(f"data-parallel training runs on cuda or cpu, not {device_type}; "
                         f"pass --device cpu or cuda")
    if requested == "auto":
        return "nccl" if device_type == "cuda" and not mig else "gloo"
    if requested == "nccl" and device_type != "cuda":
        raise SystemExit("--dist_backend nccl needs CUDA (--device cuda); use gloo on the CPU")
    return requested


def init_distributed(device: torch.device, timeout_min: float = 30.0,
                     backend: str = "auto") -> Tuple[DistEnv, torch.device]:
    """Join the torchrun job, if any.

    ``device`` is the device the caller would use single-process; with a
    process group it becomes ``cuda:LOCAL_RANK`` — ``cuda:0`` on MIG
    instances, where every rank sees only its own (:func:`pin_rank_device`)
    — or stays ``cpu``.  ``backend`` is ``--dist_backend``
    (:func:`resolve_backend`).  Other device types cannot run a process
    group here.
    """
    env = dist_env_from_environ()
    if not env.enabled:
        return env, device
    pin_rank_device()           # a no-op when the caller pinned already (it should)
    mig = visible_mig()
    name = resolve_backend(backend, device.type, mig)
    if device.type == "cuda":
        index = 0 if mig else env.local_rank
        count = torch.cuda.device_count()
        if index >= count:
            raise SystemExit(f"rank {env.rank} (LOCAL_RANK {env.local_rank}) needs cuda:{index} "
                             f"but sees {count} CUDA device(s) "
                             f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    dist.init_process_group(name, timeout=timedelta(minutes=timeout_min),
                            device_id=device if name == "nccl" else None)
    what = (f" {torch.cuda.get_device_name(device)} "
            f"[{os.environ.get('CUDA_VISIBLE_DEVICES')}]" if device.type == "cuda" else "")
    print(f"[dist] rank {env.rank}/{env.world} (local {env.local_rank}): backend={name} "
          f"device={device}{what}", flush=True)
    if name == "nccl" and mig:
        print("[dist] WARNING: NCCL on MIG instances fails with 'Duplicate GPU detected' when "
              "two ranks share a physical GPU; --dist_backend auto (gloo) works", flush=True)
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
