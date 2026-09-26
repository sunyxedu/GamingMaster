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

Values are from the ACTING player's perspective.  An interleaved stream of
several seats (a board game played by both movers, one sequence per game)
goes through :func:`seat_gae` instead: each seat's successor is its own next
decision, and every seat is credited its own rewards / outcome.
Non-learner seats are masked out of the losses, not here.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

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
    if trunc_values is not None and trunc_values.shape != r.shape:
        raise ValueError(f"trunc_values shape {tuple(trunc_values.shape)} != "
                         f"{tuple(r.shape)} (one value per step, read at truncated steps)")
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


@torch.no_grad()
def seat_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    seat: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: Scalarish,
    lam: Scalarish,
    n_seats: Optional[int] = None,
    truncated: Optional[torch.Tensor] = None,
    trunc_values: Optional[torch.Tensor] = None,
    bootstrap: Optional[Scalarish] = None,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE over INTERLEAVED multi-seat streams (one sequence per game, every
    mover's frames in ply order — the board-game / mahjong layout).

    Each seat's frames form their own GAE chain: the successor of seat s's
    decision at ply t is seat s's NEXT decision (its value from s's own
    perspective), not ply t+1, which belongs to another mover; per-decision
    discounting as in ``training/train_rl_game.seat_gae``.  One reverse scan
    with [..., S] per-seat state, batched over rows (no per-seat gather /
    pad / scatter in the trainer).

    Args:
        rewards:      [..., T, S] reward each seat receives after ply t (the
                      final outcome of every seat at the episode's last ply;
                      shaped rewards anywhere) — summed per seat over the plies
                      between its decisions; rewards before a seat's first
                      decision of an episode are not attributable and dropped.
                      Or [..., T]: already credited to the ACTING seat of ply t
                      (outcome on each seat's last frame).
        values:       [..., T] V(s_t) from the acting seat's perspective
                      (reward units)
        seat:         [..., T] int acting seat in [0, S)
        terminated:   [..., T] bool, the episode terminated after ply t — for
                      EVERY seat (stream-level: flag the episode's last ply)
        gamma, lam:   float, per-row [...] or per-step [..., T] (the acting
                      decision's)
        n_seats:      S (default: rewards' last dim, else max seat + 1)
        truncated:    [..., T] bool, the episode was cut after ply t
        trunc_values: [..., T, S] V(final observation) from each seat's
                      perspective, read at truncated plies
        bootstrap:    [..., S] (or scalar) each seat's V after the row's last
                      valid ply, if the row stops mid-episode (default 0)
        valid:        [..., T] bool tail-padding mask

    Returns (advantages, returns) [..., T] as :func:`gae`; with S = 1 this
    is exactly :func:`gae`.
    """
    if values.dim() == 0:
        raise ValueError("seat_gae needs at least a time dimension")
    if seat.shape != values.shape or terminated.shape != values.shape:
        raise ValueError(f"values {tuple(values.shape)}, seat {tuple(seat.shape)} and "
                         f"terminated {tuple(terminated.shape)} must share one shape")
    per_seat = rewards.dim() == values.dim() + 1
    if per_seat:
        if rewards.shape[:-1] != values.shape:
            raise ValueError(f"per-seat rewards {tuple(rewards.shape)} must be "
                             f"{tuple(values.shape)} + [S]")
    elif rewards.shape != values.shape:
        raise ValueError(f"rewards {tuple(rewards.shape)} must be {tuple(values.shape)} "
                         f"(acting-seat credit) or {tuple(values.shape)} + [S]")
    dtype = torch.promote_types(torch.promote_types(rewards.dtype, values.dtype), torch.float32)
    v = values.to(dtype)
    T = v.shape[-1]
    lead = v.shape[:-1]
    dev = v.device

    ok = torch.ones_like(terminated, dtype=torch.bool) if valid is None else valid.bool()
    if ok.shape != v.shape:
        raise ValueError(f"valid shape {tuple(ok.shape)} != {tuple(v.shape)}")
    if T > 1 and bool((ok[..., 1:] & ~ok[..., :-1]).any()):
        raise ValueError("valid must be a per-row prefix (tail padding only)")
    st = seat.long()
    if n_seats is None:
        n_seats = rewards.shape[-1] if per_seat else (int(st[ok].max()) + 1 if ok.any() else 1)
    S = int(n_seats)
    if per_seat and rewards.shape[-1] != S:
        raise ValueError(f"rewards have {rewards.shape[-1]} seats, n_seats={S}")
    if bool(((st < 0) | (st >= S))[ok].any()):
        raise ValueError(f"seat ids must be in [0, {S}) on valid plies")
    st = torch.where(ok, st, torch.zeros_like(st))

    term = terminated.bool() & ok
    trunc = (torch.zeros_like(term) if truncated is None else truncated.bool()) & ok
    if truncated is not None and truncated.shape != v.shape:
        raise ValueError(f"truncated shape {tuple(truncated.shape)} != {tuple(v.shape)}")
    if bool((trunc & term).any()):
        raise ValueError("a ply cannot be both terminated and truncated")
    if trunc_values is not None and trunc_values.shape != (*v.shape, S):
        raise ValueError(f"trunc_values shape {tuple(trunc_values.shape)} != "
                         f"{(*v.shape, S)} (per ply, per seat)")
    if bool(trunc.any()):
        if trunc_values is None:
            raise ValueError("truncated plies need trunc_values (V of the final observation)")
        tv = trunc_values.to(dtype)
    else:
        tv = torch.zeros(*v.shape, S, dtype=dtype, device=dev)
    r = rewards.to(dtype)

    if bootstrap is None:
        boot = torch.zeros(*lead, S, dtype=dtype, device=dev)
    else:
        b = torch.as_tensor(bootstrap, dtype=dtype, device=dev)
        if b.dim() and b.shape != (*lead, S):
            raise ValueError(f"bootstrap shape {tuple(b.shape)} != {(*lead, S)} (per seat)")
        boot = b.expand(*lead, S).clone()
    g = _full(gamma, v, "gamma")
    gl = g * _full(lam, v, "lam")

    zero = torch.zeros((), dtype=dtype, device=dev)
    next_v = boot                                     # [..., S] successor value per seat
    next_a = torch.zeros(*lead, S, dtype=dtype, device=dev)
    racc = torch.zeros(*lead, S, dtype=dtype, device=dev)   # rewards since the seat's decision
    adv = torch.zeros_like(v)
    for t in range(T - 1, -1, -1):
        ok_t = ok[..., t]
        # an episode ending after ply t cuts every seat's chain there
        end = (term[..., t] | trunc[..., t]).unsqueeze(-1)
        next_v = torch.where(end, torch.where(trunc[..., t].unsqueeze(-1), tv[..., t, :], zero),
                             next_v)
        next_a = torch.where(end, zero, next_a)
        racc = torch.where(end, zero, racc)
        s_t = st[..., t]
        onehot = F.one_hot(s_t, S).bool() & ok_t.unsqueeze(-1)
        if per_seat:
            racc = racc + torch.where(ok_t.unsqueeze(-1), r[..., t, :], zero)
        else:
            racc = racc + torch.where(onehot, r[..., t].unsqueeze(-1), zero)
        idx = s_t.unsqueeze(-1)
        delta = (racc.gather(-1, idx).squeeze(-1) + g[..., t] * next_v.gather(-1, idx).squeeze(-1)
                 - v[..., t])
        a = torch.where(ok_t, delta + gl[..., t] * next_a.gather(-1, idx).squeeze(-1), zero)
        adv[..., t] = a
        next_v = torch.where(onehot, v[..., t].unsqueeze(-1), next_v)
        next_a = torch.where(onehot, a.unsqueeze(-1), next_a)
        racc = torch.where(onehot, zero, racc)
    returns = torch.where(ok, adv + v, zero)
    return adv, returns
