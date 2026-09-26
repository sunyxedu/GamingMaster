"""Scalar diagnostics for the mixed-game learner's logs and canaries."""

from __future__ import annotations

import math
from typing import Optional

import torch

from training.rl_joint.stats import masked_mean


@torch.no_grad()
def explained_variance(pred: torch.Tensor, target: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> float:
    """1 − Var[target − pred] / Var[target] over valid entries (1: perfect
    critic, 0: no better than a constant, < 0: worse).  NaN when the
    targets are constant."""
    dt = torch.float32 if pred.device.type == "mps" else torch.float64
    p, t = pred.detach().to(dt), target.detach().to(dt)
    var_t = masked_mean((t - masked_mean(t, mask)).square(), mask)
    err = t - p
    var_e = masked_mean((err - masked_mean(err, mask)).square(), mask)
    if float(var_t) == 0.0:
        return float("nan")
    return float(1.0 - var_e / var_t)


@torch.no_grad()
def expert_load_entropy(loads: torch.Tensor) -> torch.Tensor:
    """Normalised entropy H(load / Σ load) / log E of MoE expert loads over
    the last dim ([E] or [L, E] per layer): 1 = perfectly balanced, 0 = one
    expert takes every token.  All-zero rows (no tokens) give NaN."""
    x = loads.detach().to(torch.float32 if loads.device.type == "mps" else torch.float64)
    n_exp = x.shape[-1]
    if n_exp < 2:
        return torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    total = x.sum(-1, keepdim=True)
    p = x / total.clamp(min=torch.finfo(x.dtype).tiny)
    zero = torch.zeros((), dtype=p.dtype, device=p.device)
    h = -torch.where(p > 0, p * p.log(), zero).sum(-1) / math.log(n_exp)
    return torch.where(total.squeeze(-1) > 0, h, torch.full_like(h, float("nan")))
