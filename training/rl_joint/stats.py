"""Masked reductions and rank-invariant per-game statistics.

Every stateful component of the library (advantage normaliser, PopArt) keeps
its statistics as Python floats (float64, device-agnostic: normalising a
tensor on any device needs no host copy) and updates them from per-game
moments of the iteration's data.  Under torchrun the moments must be GLOBAL
and the update identical on every rank, so :func:`per_game_moments` packs
the (count, sum, sum of squares, #non-finite) of EVERY configured game into
one [G + 1, 4] float64 tensor (the extra row flags a local input error) and
all-reduces it once: a rank that drew no data for a game contributes zeros,
so every rank posts exactly one collective of the same shape no matter
which games it rolled out — and a rank whose input is bad still posts it,
then every rank raises.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

#: a per-game batch: tensor, or (tensor, validity mask) — mask None = all valid
GameBatch = Mapping[str, object]


def dist_active(group=None) -> bool:
    """True inside a process group (default: the world) of more than one rank."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1


def collective_device(group=None) -> torch.device:
    """Device a collective's tensor must live on for ``group``: the current
    CUDA device under NCCL, the CPU otherwise (gloo)."""
    if dist_active(group) and dist.get_backend(group) == "nccl":
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
    0-d tensors on ``x``'s device, in ``x``'s float dtype promoted to at
    least float32 (a half-precision count would overflow at 65504 tokens).

    Masked entries are selected out (``torch.where``), never multiplied by
    zero, so a NaN / inf in a padded slot cannot leak into the sum or its
    gradient."""
    m = as_mask(x, mask)
    dt = torch.promote_types(x.dtype, torch.float32) if x.is_floating_point() else torch.float32
    x = x.to(dt)
    total = torch.where(m, x, torch.zeros((), dtype=dt, device=x.device)).sum()
    return total, m.sum().to(dt)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean of ``x`` over valid entries (0 when there are none)."""
    total, count = masked_sum(x, mask)
    return total / count.clamp(min=1)


def _unpack(item) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(item, torch.Tensor):
        return item, None
    x, mask = item
    return x, mask


def _local_stats(batches: GameBatch, games: Sequence[str]) -> torch.Tensor:
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
    return stats


@torch.no_grad()
def per_game_moments(batches: GameBatch, games: Sequence[str], *, sync: bool = True,
                     group=None) -> Dict[str, Tuple[float, float, float]]:
    """Per-game (count, mean, mean of squares) of ``batches`` over ``games``.

    ``batches`` maps a game to a tensor or a ``(tensor, mask)`` pair; games
    missing from it (or with no valid entry) count zero.  With ``sync`` and
    an active process group (``group``, default the world) the statistics
    are summed over its ranks with a single all-reduce (call on EVERY rank
    of the group with the same ``games``).  Games with a zero global count
    map to ``(0.0, 0.0, 0.0)``.  Every failure is raised AFTER the
    collective and on every rank, so no rank is left blocked in it: a
    non-finite value in any rank's batch raises ``ValueError``; a bad input
    (unknown game, mask shape) re-raises its error on that rank and a
    ``RuntimeError`` on the others.
    """
    G = len(games)
    buf = torch.zeros(G + 1, 4, dtype=torch.float64)
    err: Optional[BaseException] = None
    try:
        buf[:G] = _local_stats(batches, games)
    except Exception as e:          # re-raised below, after the collective
        err = e
        buf.zero_()
        buf[G, 0] = 1.0
    if sync and dist_active(group):
        dev_buf = buf.to(collective_device(group))
        dist.all_reduce(dev_buf, op=dist.ReduceOp.SUM, group=group)
        buf = dev_buf.cpu()
    if err is not None:
        raise err
    if buf[G, 0] > 0:
        raise RuntimeError(f"per_game_moments: bad input on {int(buf[G, 0])} other rank(s)")
    stats = buf[:G]
    bad = [g for i, g in enumerate(games) if stats[i, 3] > 0]
    if bad:
        raise ValueError(f"non-finite values in the statistics batch of {bad}")
    out = {}
    for i, g in enumerate(games):
        n, s, sq, _ = stats[i].tolist()
        out[g] = (n, s / n, sq / n) if n > 0 else (0.0, 0.0, 0.0)
    return out
