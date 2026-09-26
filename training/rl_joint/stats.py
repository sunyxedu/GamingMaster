"""Masked reductions and rank-invariant per-game statistics.

Every stateful component of the library (advantage normaliser, PopArt) keeps
its statistics as Python floats (float64, device-agnostic: normalising a
tensor on any device needs no host copy) and updates them from per-game
moments of the iteration's data.  Under torchrun the moments must be GLOBAL
and the update identical on every rank, so :func:`per_game_moments` packs
the (count, sum, sum of squares, #non-finite) of EVERY configured game into
one [G, 4] float64 tensor and all-reduces it once: a rank that drew no data for a game
contributes zeros, so every rank posts exactly one collective of the same
shape no matter which games it rolled out.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

#: a per-game batch: tensor, or (tensor, validity mask) — mask None = all valid
GameBatch = Mapping[str, object]


def dist_active() -> bool:
    """True inside a process group of more than one rank."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def collective_device() -> torch.device:
    """Device a collective's tensor must live on for the default group:
    the current CUDA device under NCCL, the CPU otherwise (gloo)."""
    if dist_active() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def as_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """``mask`` as a bool tensor of ``x``'s shape (all True when None)."""
    if mask is None:
        return torch.ones_like(x, dtype=torch.bool)
    if mask.shape != x.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != tensor shape {tuple(x.shape)}")
    return mask.bool()


def masked_sum(x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """(sum of ``x`` over valid entries, number of valid entries) — both
    0-d tensors on ``x``'s device, the count in ``x``'s float dtype.

    Masked entries are selected out (``torch.where``), never multiplied by
    zero, so a NaN / inf in a padded slot cannot leak into the sum or its
    gradient."""
    m = as_mask(x, mask)
    total = torch.where(m, x, torch.zeros((), dtype=x.dtype, device=x.device)).sum()
    return total, m.sum().to(x.dtype if x.is_floating_point() else torch.float32)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean of ``x`` over valid entries (0 when there are none)."""
    total, count = masked_sum(x, mask)
    return total / count.clamp(min=1)


def _unpack(item) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(item, torch.Tensor):
        return item, None
    x, mask = item
    return x, mask


@torch.no_grad()
def per_game_moments(batches: GameBatch, games: Sequence[str], *, sync: bool = True
                     ) -> Dict[str, Tuple[float, float, float]]:
    """Per-game (count, mean, mean of squares) of ``batches`` over ``games``.

    ``batches`` maps a game to a tensor or a ``(tensor, mask)`` pair; games
    missing from it (or with no valid entry) count zero.  With ``sync`` and
    an active process group the statistics are summed over all ranks with a
    single all-reduce (call on EVERY rank with the same ``games``).  Games
    with a zero global count map to ``(0.0, 0.0, 0.0)``.  A non-finite value
    in any rank's batch raises ``ValueError`` on every rank.
    """
    unknown = set(batches) - set(games)
    if unknown:
        raise KeyError(f"statistics for unknown games {sorted(unknown)}")
    # columns: count, sum, sum of squares, non-finite count
    stats = torch.zeros(len(games), 4, dtype=torch.float64)
    for i, g in enumerate(games):
        if g not in batches:
            continue
        x, mask = _unpack(batches[g])
        m = as_mask(x, mask)
        # float64 accumulation where the device has it (MPS does not)
        v = x.detach().to(torch.float32 if x.device.type == "mps" else torch.float64)[m]
        if v.numel():
            fin = torch.isfinite(v)
            v = v[fin]
            stats[i] = torch.stack([
                torch.tensor(float(v.numel()), dtype=v.dtype, device=v.device),
                v.sum(), v.square().sum(),
                (~fin).sum().to(v.dtype)]).cpu().double()
    if sync and dist_active():
        buf = stats.to(collective_device())
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        stats = buf.cpu()
    # raised AFTER the collective, so every rank raises (no rank is left
    # waiting in a later collective)
    bad = [g for i, g in enumerate(games) if stats[i, 3] > 0]
    if bad:
        raise ValueError(f"non-finite values in the statistics batch of {bad}")
    out = {}
    for i, g in enumerate(games):
        n, s, sq, _ = stats[i].tolist()
        out[g] = (n, s / n, sq / n) if n > 0 else (0.0, 0.0, 0.0)
    return out
