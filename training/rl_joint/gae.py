"""Generalized Advantage Estimation with per-game discounting and exact
episode-boundary handling.

One call handles one game's rollout (or a mixed batch: ``gamma`` / ``lam``
may be per-row tensors).  Rows are continuous streams of steps; an episode
can end inside a row in two ways, which bootstrap differently:

  terminated[t]   the environment ended the episode at step t — the
                  successor state has value 0 (no bootstrap);
  truncated[t]    the episode was CUT after step t (turn limit, rollout
                  horizon of a pooled env, ...) — its successor state's value
                  is ``trunc_values[t]`` (the critic's value of the final
                  observation), not the next step of the row, which already
                  belongs to the next episode.

Either way the advantage recursion does not carry across the boundary.
After the last valid step of a row the successor value is ``bootstrap``
(the critic's value of the state the rollout stopped in; 0 when every row
ends terminated).  ``valid`` marks tail padding: it must be a per-row
prefix, padded steps get zero advantage and return.

Values are from the ACTING player's perspective, so an interleaved stream
of several seats (a board game played by both movers) is split into one
row per seat before calling this — each seat's successor is its own next
decision, and the final outcome has to be credited to both seats' last
steps.  Non-learner seats are then masked out of the losses, not here.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

Scalarish = Union[float, torch.Tensor]


def _full(x: Scalarish, like: torch.Tensor, name: str) -> torch.Tensor:
    """``x`` (scalar, per-row [...] or per-step [..., T]) broadcast to ``like``."""
    t = torch.as_tensor(x, dtype=like.dtype, device=like.device)
    if t.dim() == like.dim():
        return t.expand_as(like)
    try:
        return t.unsqueeze(-1).expand_as(like) if t.dim() else t.expand_as(like)
    except RuntimeError:
        raise ValueError(f"{name} of shape {tuple(t.shape)} broadcasts neither per row "
                         f"nor per step to {tuple(like.shape)}") from None


@torch.no_grad()
def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: Scalarish,
    lam: Scalarish,
    truncated: Optional[torch.Tensor] = None,
    trunc_values: Optional[torch.Tensor] = None,
    bootstrap: Optional[Scalarish] = None,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE(γ, λ) advantages and λ-returns (Schulman et al. 2016).

    Args:
        rewards:      [..., T] reward received after step t
        values:       [..., T] critic value V(s_t) (UNnormalised: with PopArt,
                      denormalise the head's output first)
        terminated:   [..., T] bool, episode terminated at step t
        gamma, lam:   float, per-row tensor [...], or per-step tensor [..., T]
        truncated:    [..., T] bool, episode cut after step t (optional)
        trunc_values: [..., T] V(final observation) at truncated steps; read
                      only where ``truncated`` (required if any is set)
        bootstrap:    [...] V(s) after each row's last valid step (default 0)
        valid:        [..., T] bool tail-padding mask (default all valid)

    Returns:
        (advantages, returns) — [..., T], float32 or wider, zero on padding;
        ``returns = advantages + values`` on valid steps.
    """
    if rewards.shape != values.shape or terminated.shape != rewards.shape:
        raise ValueError(f"rewards {tuple(rewards.shape)}, values {tuple(values.shape)} and "
                         f"terminated {tuple(terminated.shape)} must share one shape")
    if rewards.dim() == 0:
        raise ValueError("gae needs at least a time dimension")
    dtype = torch.promote_types(torch.promote_types(rewards.dtype, values.dtype), torch.float32)
    r = rewards.to(dtype)
    v = values.to(dtype)
    term = terminated.bool()
    T = r.shape[-1]
    lead = r.shape[:-1]

    trunc = torch.zeros_like(term) if truncated is None else truncated.bool()
    if trunc.shape != r.shape:
        raise ValueError(f"truncated shape {tuple(trunc.shape)} != {tuple(r.shape)}")
    if bool(trunc.any()):
        if trunc_values is None:
            raise ValueError("truncated steps need trunc_values (V of the final observation)")
        tv = trunc_values.to(dtype)
    else:
        tv = torch.zeros_like(r)
    if bool((trunc & term).any()):
        raise ValueError("a step cannot be both terminated and truncated")

    ok = torch.ones_like(term) if valid is None else valid.bool()
    if ok.shape != r.shape:
        raise ValueError(f"valid shape {tuple(ok.shape)} != {tuple(r.shape)}")
    if T > 1 and bool((ok[..., 1:] & ~ok[..., :-1]).any()):
        raise ValueError("valid must be a per-row prefix (tail padding only)")

    boot = (torch.zeros(lead, dtype=dtype, device=r.device) if bootstrap is None
            else torch.as_tensor(bootstrap, dtype=dtype, device=r.device).expand(lead))
    g = _full(gamma, r, "gamma")
    gl = g * _full(lam, r, "lam")

    zero = torch.zeros((), dtype=dtype, device=r.device)
    adv = torch.zeros_like(r)
    next_adv = torch.zeros(lead, dtype=dtype, device=r.device)
    next_v = boot
    for t in range(T - 1, -1, -1):
        end = term[..., t] | trunc[..., t]
        succ = torch.where(term[..., t], zero,
                           torch.where(trunc[..., t], tv[..., t], next_v))
        delta = r[..., t] + g[..., t] * succ - v[..., t]
        a = delta + gl[..., t] * torch.where(end, zero, next_adv)
        a = torch.where(ok[..., t], a, zero)
        adv[..., t] = a
        next_adv = a
        next_v = torch.where(ok[..., t], v[..., t], boot)
    returns = torch.where(ok, adv + v, zero)
    return adv, returns
