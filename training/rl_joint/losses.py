"""Per-game loss terms of mixed-game PPO.

Every function takes ONE game's tensors (any shape, typically [B, T]) plus a
bool validity mask (padding, non-learner seats, forced moves → False) and
returns a dict whose ``"loss"`` is the masked mean (``reduction="mean"``)
or masked sum (``reduction="sum"``, for a global token normaliser across
games / ranks, see :func:`combine_game_losses`), ``"count"`` the number of
tokens it averaged over, and detached diagnostics.

Masked entries are SELECTED out (``torch.where``) before any exp / product /
log-softmax, so garbage in padded slots (−inf log-probs, NaN advantages or
logits) never reaches the loss or its gradient.  (:func:`categorical_kl` is
unreduced: mask its rows yourself.)
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Union

import torch
import torch.nn.functional as F

from model.losses import hl_gauss_cross_entropy
from training.rl_joint.stats import as_mask, masked_mean, masked_sum

Coef = Union[float, torch.Tensor]

#: cap on log(π_ref/π) inside the k3 estimator's exp
_K3_LOG_RHO_MAX = 20.0


def _reduce(x: torch.Tensor, mask: torch.Tensor, reduction: str) -> Dict[str, torch.Tensor]:
    total, count = masked_sum(x, mask)
    if reduction == "mean":
        return {"loss": total / count.clamp(min=1), "count": count.detach()}
    if reduction == "sum":
        return {"loss": total, "count": count.detach()}
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def _zero_where(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, x, torch.zeros((), dtype=x.dtype, device=x.device))


# ======================================================================
# Policy
# ======================================================================

def decoupled_ppo_loss(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    logp_mu: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    policy_mask: Optional[torch.Tensor] = None,
    clip: float = 0.2,
    clip_high: Optional[float] = None,
    is_clip: Optional[float] = 1.0,
    dual_clip: Optional[float] = None,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Decoupled PPO clipped surrogate (Hilton et al. 2021, "Batch
    size-invariance for policy optimization", arXiv 2110.00641) with a
    truncated importance weight on the behaviour policy:

        L = −E_μ[ min(1, π_old/μ)_trunc · min(r·A, clip(r, 1−ε, 1+ε_hi)·A) ],
        r = π / π_old

    μ is the behaviour policy the actors sampled from (their stored
    log-probs), π_old the learner's recomputation of the same actions with
    the iteration-start weights (proximal policy), π the current policy.
    The clip trust region is centred on π_old — not on the stale μ — and the
    off-policy correction is the separate weight ``min(is_clip, π_old/μ)``
    (``is_clip=None``: untruncated).  With μ = π_old the weight is 1 and this
    is exactly standard PPO.

    Args:
        logp:        [...] log π(a|s), differentiable
        logp_old:    [...] log π_old(a|s) (detached internally)
        logp_mu:     [...] log μ(a|s) from the actors
        advantages:  [...] normalised advantages
        mask:        [...] bool valid learner tokens
        policy_mask: [...] bool subset the surrogate trains on (top-q
                     advantage filter); diagnostics use the full ``mask``
        clip / clip_high: ε below / above 1 (``clip_high`` default ``clip``)
        is_clip:     truncation level of π_old/μ (V-trace's ρ̄)
        dual_clip:   c > 1 bounds the loss of negative-advantage tokens by
                     −c·A (Ye et al. 2020, arXiv 1912.09729); None disables

    Returns ``loss``, ``count`` and detached diagnostics over ``mask``:
    ``approx_kl`` (k3 estimate of KL(π_old‖π)), ``clip_frac``, ``is_weight``
    (mean truncated weight), ``is_trunc_frac``, and the two canaries
    ``abs_log_old_mu`` = mean |log π_old/μ| (actor/learner mismatch: ~0 up to
    numerics when the rollout is replayed exactly) and ``abs_log_ratio`` =
    mean |log π/π_old| (0 on the first minibatch of epoch 1).
    """
    m = as_mask(logp, mask)
    pm = m if policy_mask is None else m & policy_mask.bool()
    eps_lo = float(clip)
    eps_hi = float(clip if clip_high is None else clip_high)
    if dual_clip is not None and dual_clip <= 1.0:
        raise ValueError(f"dual_clip must be > 1, got {dual_clip}")

    ftype = torch.promote_types(logp.dtype, torch.float32)
    lp = logp.to(ftype)
    lp_old = logp_old.detach().to(ftype)
    lp_mu = logp_mu.detach().to(ftype)
    adv = _zero_where(advantages.detach().to(ftype), m)

    log_ratio = _zero_where(lp - lp_old, m)
    ratio = log_ratio.exp()
    log_old_mu = _zero_where(lp_old - lp_mu, m)
    w = log_old_mu.exp()
    if is_clip is not None:
        w = w.clamp(max=float(is_clip))

    surr = torch.min(ratio * adv, ratio.clamp(1.0 - eps_lo, 1.0 + eps_hi) * adv)
    if dual_clip is not None:
        surr = torch.where(adv < 0, torch.max(surr, float(dual_clip) * adv), surr)
    out = _reduce(-(w * surr), pm, reduction)

    with torch.no_grad():
        r = ratio.detach()
        lr = log_ratio.detach()
        out.update(
            approx_kl=masked_mean((r - 1.0) - lr, m),
            clip_frac=masked_mean(((r < 1.0 - eps_lo) | (r > 1.0 + eps_hi)).to(ftype), m),
            is_weight=masked_mean(w, m),
            is_trunc_frac=(masked_mean((log_old_mu.exp() > float(is_clip)).to(ftype), m)
                           if is_clip is not None else torch.zeros((), device=lp.device)),
            abs_log_old_mu=masked_mean(log_old_mu.abs(), m),
            abs_log_ratio=masked_mean(lr.abs(), m),
            policy_frac=pm.sum().to(ftype) / m.sum().clamp(min=1).to(ftype),
        )
    return out


def entropy_loss(entropy: torch.Tensor, mask: Optional[torch.Tensor] = None, *,
                 coef: Coef, reduction: str = "mean") -> Dict[str, torch.Tensor]:
    """``−coef · H`` (a bonus for coef > 0, a suppressor for coef < 0), plus
    the detached mean entropy the adaptive controller consumes."""
    m = as_mask(entropy, mask)
    h = entropy.to(torch.promote_types(entropy.dtype, torch.float32))
    out = _reduce(-coef * h, m, reduction)
    out["entropy"] = masked_mean(h.detach(), m)
    return out


# ======================================================================
# Value
# ======================================================================

def value_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    old_pred: Optional[torch.Tensor] = None,
    clip: Optional[float] = None,
    kind: str = "mse",
    huber_delta: float = 1.0,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Scalar critic loss, optionally PPO-clipped around the rollout value.

    ``kind="mse"``: (v − G)²; ``kind="huber"``: 2·Huber_δ, which equals (v − G)²
    inside |v − G| ≤ δ and grows linearly beyond (the generals trainer's
    critic loss at δ = 1).  With ``clip`` and ``old_pred`` the loss is
    max(ℓ(v), ℓ(v_old + clip(v − v_old, ±clip))).

    Under PopArt every argument is in the head's NORMALISED units:
    ``target = popart.normalize(g, G)``, ``old_pred = popart.normalize(g,
    V_rollout)`` (after the statistics update — equal to the rescaled head's
    output on the same input).
    """
    m = as_mask(pred, mask)
    ftype = torch.promote_types(pred.dtype, torch.float32)
    v = pred.to(ftype)
    tgt = _zero_where(target.detach().to(ftype), m)

    def ell(x: torch.Tensor) -> torch.Tensor:
        if kind == "mse":
            return (x - tgt).square()
        if kind == "huber":
            return 2.0 * F.huber_loss(x, tgt, reduction="none", delta=huber_delta)
        raise ValueError(f"value loss kind must be 'mse' or 'huber', got {kind!r}")

    v_safe = _zero_where(v, m)
    loss = ell(v_safe)
    clip_frac = None
    if clip is not None:
        if old_pred is None:
            raise ValueError("value clipping needs old_pred")
        old = _zero_where(old_pred.detach().to(ftype), m)
        v_clip = old + (v_safe - old).clamp(-float(clip), float(clip))
        clipped = ell(v_clip)
        loss = torch.max(loss, clipped)
        with torch.no_grad():
            clip_frac = masked_mean(((v_safe - old).abs() > float(clip)).to(ftype), m)
    out = _reduce(loss, m, reduction)
    if clip_frac is not None:
        out["clip_frac"] = clip_frac
    return out


def hl_gauss_value_loss(logits: torch.Tensor, target: torch.Tensor,
                        mask: Optional[torch.Tensor] = None, *, vmax: float,
                        sigma: float, reduction: str = "mean") -> Dict[str, torch.Tensor]:
    """HL-Gauss critic loss (cross-entropy against the Gaussian-smeared
    histogram of G, ``model.losses.hl_gauss_cross_entropy``) for
    :class:`~model.components.heads.HLGaussValueHead` games — the PopArt
    passthrough games; targets stay in reward units.

    ``logits`` [..., n_bins], ``target`` / ``mask`` [...].  Targets outside
    [−vmax, vmax] are clamped onto the support; their share is reported as
    ``out_of_support`` (non-zero means the support is too narrow for the
    game's returns)."""
    m = as_mask(target, mask)
    tgt = _zero_where(target.detach().float(), m)
    # masked rows zeroed BEFORE the log-softmax: its backward multiplies the
    # (zero) incoming gradient by softmax(row), NaN for a garbage row
    safe = torch.where(m.unsqueeze(-1), logits, torch.zeros((), dtype=logits.dtype,
                                                            device=logits.device))
    ce = hl_gauss_cross_entropy(safe, tgt, vmax, sigma)
    out = _reduce(ce, m, reduction)
    out["out_of_support"] = masked_mean((tgt.abs() > vmax).float(), m)
    return out


# ======================================================================
# KL to the frozen reference
# ======================================================================

def kl_penalty(logp: torch.Tensor, logp_ref: torch.Tensor,
               mask: Optional[torch.Tensor] = None, *, beta: Coef,
               estimator: str = "k3", reduction: str = "mean") -> Dict[str, torch.Tensor]:
    """``beta · KL̂`` from the log-probs of the taken actions under the
    current policy and the frozen reference (sample estimators, Schulman
    2020 "Approximating KL divergence"), with ρ = π_ref/π:

      k1 = −log ρ          unbiased value of KL(π‖π_ref); as a LOSS its
                           gradient is zero in expectation — monitoring only
      k2 = ½ (log ρ)²      loss gradient = ∇ KL(π‖π_ref) (reverse / mode-seeking)
      k3 = (ρ − 1) − log ρ unbiased, ≥ 0 value of KL(π‖π_ref); loss gradient
                           = ∇ KL(π_ref‖π) (mass-covering — the direction the
                           generals trainer anchors with)

    The gradient statements hold for actions sampled from π; with the
    rollout's actions they hold up to the (small, clipped) π/μ mismatch.
    Use :func:`categorical_kl` when a head's full logits are available.

    Returns ``loss``, ``count``, ``kl`` (mean of the chosen estimator) and
    ``kl_k3`` (the unbiased non-negative value estimate, for the beta
    controller)."""
    m = as_mask(logp, mask)
    ftype = torch.promote_types(logp.dtype, torch.float32)
    log_rho = _zero_where(logp_ref.detach().to(ftype) - logp.to(ftype), m)
    if estimator == "k1":
        k = -log_rho
    elif estimator == "k2":
        k = 0.5 * log_rho.square()
    elif estimator == "k3":
        # ρ capped at e^20: beyond it the value (and gradient) only overflow
        k = log_rho.clamp(max=_K3_LOG_RHO_MAX).exp() - 1.0 - log_rho
    else:
        raise ValueError(f"estimator must be 'k1', 'k2' or 'k3', got {estimator!r}")
    out = _reduce(beta * k, m, reduction)
    with torch.no_grad():
        lr = log_rho.detach()
        out["kl"] = masked_mean(k.detach(), m)
        out["kl_k3"] = masked_mean(lr.clamp(max=_K3_LOG_RHO_MAX).exp() - 1.0 - lr, m)
    return out


def categorical_kl(logits: torch.Tensor, ref_logits: torch.Tensor,
                   legal: Optional[torch.Tensor] = None, *,
                   direction: str = "ref||pi") -> torch.Tensor:
    """Exact per-row KL between the categoricals of ``logits`` (current,
    differentiable) and ``ref_logits`` (frozen) over the last dim, both
    restricted to ``legal`` (bool, same shape; rows with no legal entry
    count as unmasked; logits the decoder already set to −inf are fine
    without it).  ``direction="ref||pi"``: KL(π_ref‖π) (the anchor
    the generals trainer uses); ``"pi||ref"``: KL(π‖π_ref).  Returns [...]
    (unreduced; mask rows yourself), computed in ≥ fp32, finite gradients at
    illegal entries."""
    ftype = torch.promote_types(logits.dtype, torch.float32)
    cur = logits.to(ftype)
    ref = ref_logits.detach().to(ftype)
    if legal is not None:
        ok = legal.bool()
        ok = ok | ~ok.any(-1, keepdim=True)
        neg = torch.tensor(float("-inf"), dtype=ftype, device=cur.device)
        cur = torch.where(ok, cur, neg)
        ref = torch.where(ok, ref, neg)
    else:
        ok = torch.ones_like(cur, dtype=torch.bool)
    lp_c = F.log_softmax(cur, dim=-1)
    lp_r = F.log_softmax(ref, dim=-1)
    if direction == "ref||pi":
        p, lp_p, lp_q = lp_r.exp(), lp_r, lp_c
    elif direction == "pi||ref":
        p, lp_p, lp_q = lp_c.exp(), lp_c, lp_r
    else:
        raise ValueError(f"direction must be 'ref||pi' or 'pi||ref', got {direction!r}")
    # the log-ratio is selected BEFORE the product: at a zero-probability
    # entry (illegal, or a −inf logit the decoder already masked) it is
    # NaN / ±inf, and a product's backward would turn 0·inf into NaN
    diff = _zero_where(lp_p - lp_q, ok & (p > 0))
    return (p * diff).sum(-1)


# ======================================================================
# Combining games
# ======================================================================

def combine_game_losses(
    terms: Mapping[str, Mapping[str, torch.Tensor]],
    weights: Optional[Mapping[str, float]] = None,
    *,
    mode: str = "token",
    counts: Optional[Mapping[str, Union[float, torch.Tensor]]] = None,
) -> torch.Tensor:
    """One scalar from per-game loss terms computed with ``reduction="sum"``
    (each a dict with ``loss`` and ``count``).

      mode="token": Σ_g w_g·sum_g / Σ_g w_g·n_g — the (weighted) mean over
                    every token of the mixed batch; with unit weights the
                    games' influence is their token share, i.e. whatever the
                    quota scheduler allotted.
      mode="game":  Σ_g w_g·(sum_g / n_g) / Σ_g w_g over games with n_g > 0 —
                    each game's own mean, games weighted explicitly.

    ``counts`` overrides the per-game token counts n_g of the denominators.
    Under data parallelism with gradient AVERAGING across ranks pass
    ``all_reduce(n_g) / world_size``: every rank then divides by the same
    per-rank-average count and the averaged gradient is exactly the global
    token mean (local counts would weight every rank equally instead).
    """
    if not terms:
        raise ValueError("no loss terms to combine")
    w = {g: float((weights or {}).get(g, 1.0)) for g in terms}
    n = {g: (counts[g] if counts is not None else terms[g]["count"]) for g in terms}
    n = {g: torch.as_tensor(v, dtype=torch.float32) for g, v in n.items()}
    losses = {g: t["loss"] for g, t in terms.items()}
    if mode == "token":
        denom = sum(w[g] * n[g] for g in terms)
        num = sum(w[g] * losses[g] for g in terms)
        return num / torch.as_tensor(denom).to(num.device).clamp(min=1e-12)
    if mode == "game":
        live = [g for g in terms if float(n[g]) > 0]
        if not live:
            return sum(0.0 * losses[g] for g in terms)
        num = sum(w[g] * losses[g] / n[g].to(losses[g].device) for g in live)
        return num / sum(w[g] for g in live)
    raise ValueError(f"mode must be 'token' or 'game', got {mode!r}")
