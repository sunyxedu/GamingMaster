"""Per-game loss terms of mixed-game PPO.

Every function takes ONE game's tensors (any shape, typically [B, T]) plus a
bool validity mask (padding, non-learner seats, forced moves → False) and
returns a dict whose ``"loss"`` is the masked mean (``reduction="mean"``)
or masked sum (``reduction="sum"``, for a global token normaliser across
games / ranks, see :func:`combine_game_losses`), ``"count"`` the number of
tokens it averaged over, and detached diagnostics.

Masked entries are SELECTED out (``torch.where``) before any exp / product /
log-softmax, so garbage in padded slots (−inf log-probs, NaN advantages or
logits) never reaches the loss or its gradient.  (:func:`categorical_kl` and
:func:`bernoulli_kl` are unreduced: mask their rows yourself;
:func:`head_terms_kl` composes them with a decoder's IL head weights.)
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Union

import torch
import torch.nn.functional as F

from model.losses import hl_gauss_cross_entropy
from training.rl_joint.stats import as_mask, masked_mean, masked_sum

Coef = Union[float, torch.Tensor]

#: cap on log(π_ref/π) inside the k3 estimator's exp; beyond it k3 continues
#: linearly (value and slope continuous), see :func:`_k3`
_K3_LOG_RHO_MAX = 20.0


def _k3(log_rho: torch.Tensor) -> torch.Tensor:
    """k3 = (ρ − 1) − log ρ, overflow-safe.  Past ``log ρ = C`` the exp is
    replaced by its tangent ``e^C·(1 + log ρ − C)``: value and slope stay
    continuous and the slope in log ρ stays ``e^C − 1 > 0``, so the loss
    keeps pulling π(a) back up toward the reference.  (Clamping inside the
    exp alone would leave slope −1 there — the anchor would push a collapsed
    action FURTHER below the reference.)"""
    lc = log_rho.clamp(max=_K3_LOG_RHO_MAX)
    return lc.exp() * (1.0 + (log_rho - lc)) - 1.0 - log_rho


def _reduce(x: torch.Tensor, mask: torch.Tensor, reduction: str) -> Dict[str, torch.Tensor]:
    total, count = masked_sum(x, mask)
    if reduction == "mean":
        return {"loss": total / count.clamp(min=1), "count": count.detach()}
    if reduction == "sum":
        return {"loss": total, "count": count.detach()}
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def _zero_where(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, x, torch.zeros((), dtype=x.dtype, device=x.device))


def _two_class(logit: torch.Tensor) -> torch.Tensor:
    """Bernoulli logit l → 2-class logits [0, l]: softmax([0, l])[1] = σ(l)."""
    return torch.stack([torch.zeros_like(logit), logit], dim=-1)


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
        k = _k3(log_rho)
    else:
        raise ValueError(f"estimator must be 'k1', 'k2' or 'k3', got {estimator!r}")
    out = _reduce(beta * k, m, reduction)
    with torch.no_grad():
        lr = log_rho.detach()
        out["kl"] = masked_mean(k.detach(), m)
        out["kl_k3"] = masked_mean(_k3(lr), m)
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


def bernoulli_kl(logits: torch.Tensor, ref_logits: torch.Tensor, *,
                 direction: str = "ref||pi") -> torch.Tensor:
    """Exact element-wise KL between Bernoulli(σ(``logits``)) (current,
    differentiable) and Bernoulli(σ(``ref_logits``)) (frozen) — e.g. a
    pass / promote / drop head.  ``direction`` as in :func:`categorical_kl`.
    Returns the input shape, ≥ fp32, via log-sigmoids (stable at large
    |logit|)."""
    ftype = torch.promote_types(logits.dtype, torch.float32)
    return categorical_kl(_two_class(logits.to(ftype)),
                          _two_class(ref_logits.detach().to(ftype)), direction=direction)


def head_terms_kl(
    cur_terms: Mapping[str, Mapping[str, object]],
    ref_terms: Mapping[str, Mapping[str, object]],
    *,
    beta: Coef = 1.0,
    direction: str = "ref||pi",
    frame_dims: int = 2,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Full-distribution KL to the frozen reference over EVERY action head,
    composed exactly like the decoders' IL loss (``ActionDecoder.il_head_terms``):

        KL = Σ_h scale_h · Σ(kl_h · weight_h) / norm_h

    — the multi-head generalisation of the generals trainer's anchor
    (``model.rl_utils.compute_policy_kl_heads``: every head, the pass
    Bernoulli included, averaged over the frames it is supervised on).
    Exact per-state KLs, so unlike :func:`kl_penalty` it has no sampling
    variance and is valid on the IL replay stream, whose actions do not come
    from π.

    ``cur_terms`` / ``ref_terms``: ``il_head_terms`` dicts of the current
    policy and the reference, from forwards teacher-forced on the SAME
    actions (conditional heads condition on the same choices).  Per head:
    ``logits`` (categorical [..., N] / bernoulli [...]), ``kind``
    (default "categorical"), ``weight`` (per-element, 0 = not supervised),
    ``norm`` (the head's denominator), optional ``mask`` (legality, applied
    to both sides; a bernoulli ``mask`` [..., 2] lists its legal classes
    [0, 1]) and ``scale``.  The current side's weight / norm / mask / scale
    are used.  Unweighted elements are zeroed BEFORE the softmaxes, so
    garbage logits there reach neither the loss nor its gradient.

    Returns ``loss`` (β·KL; ``reduction="sum"``: β·KL·count, for
    :func:`combine_game_losses`), ``count`` = frames (the first
    ``frame_dims`` dims) with any supervised head, ``kl`` (detached KL, the
    beta controller's measurement) and ``kl_<head>`` (each head's weighted
    mean KL)."""
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    missing = set(cur_terms) - set(ref_terms)
    if missing:
        raise KeyError(f"reference terms missing heads {sorted(missing)}")
    if not cur_terms:
        raise ValueError("no heads to take the KL of")
    total = None
    frames = None
    diags: Dict[str, torch.Tensor] = {}
    for h, term in cur_terms.items():
        ref = ref_terms[h]
        kind = term.get("kind", "categorical")
        logits, ref_logits = term["logits"], ref["logits"]
        weight = term["weight"].detach().float()
        on = weight > 0
        mask = term.get("mask")
        if kind == "categorical":
            if weight.shape != logits.shape[:-1]:
                raise ValueError(f"{h!r}: weight {tuple(weight.shape)} != logits "
                                 f"{tuple(logits.shape)}[:-1]")
            sel = on.unsqueeze(-1)
            cur = torch.where(sel, logits, torch.zeros((), dtype=logits.dtype,
                                                       device=logits.device))
            rl = torch.where(sel, ref_logits.detach(),
                             torch.zeros((), dtype=ref_logits.dtype, device=ref_logits.device))
            kl = categorical_kl(cur, rl, mask, direction=direction)
        elif kind == "bernoulli":
            if weight.shape != logits.shape:
                raise ValueError(f"{h!r}: weight {tuple(weight.shape)} != logits "
                                 f"{tuple(logits.shape)}")
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            cur = torch.where(on, logits, zero)
            rl = torch.where(on, ref_logits.detach(), zero.to(ref_logits.dtype))
            if mask is None:
                kl = bernoulli_kl(cur, rl, direction=direction)
            else:
                kl = categorical_kl(_two_class(cur), _two_class(rl), mask, direction=direction)
        else:
            raise ValueError(f"{h!r}: unknown head kind {kind!r}")
        kl = _zero_where(kl, on)
        norm = torch.as_tensor(term["norm"], dtype=kl.dtype, device=kl.device)
        part = float(term.get("scale", 1.0)) * (kl * weight.to(kl.dtype)).sum() / norm
        total = part if total is None else total + part
        with torch.no_grad():
            diags[f"kl_{h}"] = (kl * weight).sum() / weight.sum().clamp(min=1e-12)
            f = on.reshape(*on.shape[:frame_dims], -1).any(-1) if on.dim() > frame_dims else on
            frames = f if frames is None else frames | f
    count = frames.sum().to(total.dtype)
    loss = beta * total
    out = {"loss": loss * count if reduction == "sum" else loss, "count": count,
           "kl": total.detach()}
    out.update(diags)
    return out


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

    ``counts`` (game → n_g) replaces the local token counts, and its KEYS
    define the games of the denominators: a game in ``counts`` but not in
    ``terms`` contributes a zero numerator and still counts in the
    denominator (token mode) / in the set of live games (game mode).  Under
    data parallelism with gradient AVERAGING across ranks pass
    ``{g: all_reduce(n_g) / world_size}`` over EVERY configured game (zeros
    included) on every rank — also for games this rank drew no tokens of:
    every rank then divides by the same denominators and the averaged
    gradient is exactly the global token mean (game mode: the mean of the
    global per-game means).  Local counts, or counts over only the local
    games, would weight ranks rather than tokens.  A game in ``terms`` must
    appear in ``counts``.
    """
    if not terms:
        raise ValueError("no loss terms to combine")
    if mode not in ("token", "game"):
        raise ValueError(f"mode must be 'token' or 'game', got {mode!r}")
    if counts is not None:
        missing = set(terms) - set(counts)
        if missing:
            raise KeyError(f"counts missing for games with loss terms {sorted(missing)}")
        games = list(counts)
        n = {g: torch.as_tensor(counts[g], dtype=torch.float32) for g in games}
    else:
        games = list(terms)
        n = {g: torch.as_tensor(terms[g]["count"], dtype=torch.float32) for g in games}
    w = {g: float((weights or {}).get(g, 1.0)) for g in games}
    losses = {g: t["loss"] for g, t in terms.items()}
    # a graph-connected zero: a rank whose terms all drop out still backprops
    zero = sum(0.0 * v for v in losses.values())
    device = zero.device
    if mode == "token":
        denom = sum(w[g] * n[g].to(device) for g in games)
        num = zero + sum(w[g] * losses[g] for g in terms)
        return num / torch.as_tensor(denom, dtype=num.dtype, device=device).clamp(min=1e-12)
    live = [g for g in games if float(n[g]) > 0]
    if not live:
        return zero
    num = zero + sum(w[g] * losses[g] / n[g].to(device) for g in live if g in terms)
    return num / sum(w[g] for g in live)
