"""Attention / output-logit stability diagnostics for the IL trainers.

What a diverging run needs logged to tell its failure modes apart
(QK-norm, ``cfg.qk_norm``, only fixes the first):

* attention-logit growth (Wortsman et al. 2023): the max pre-softmax
  logit climbs and the attention entropy falls toward one-hot rows —
  :func:`attention_probe`, per calling module;
* the QK-norm bound itself: every logit is ≤ √D·max|g_q|·max|g_k|, and
  with the scales out of weight decay only the learning rate limits it —
  :func:`qk_caps` (from the parameters, free);
* output-logit divergence (what z-loss fixes): the decoder logits'
  log-partition drifts — :func:`logit_lse_absmax`.

Everything here is read-only and meant for one eval batch at a time (the
probe materialises attention matrices in fp32, in chunks).
"""

from __future__ import annotations

import contextlib
import math
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def qk_caps(model: nn.Module) -> Dict[str, float]:
    """``{module name: √D·max|g_q|·max|g_k|}`` for every QK-norm module
    (empty for qk_norm=False models).  One host sync."""
    from model.components.core import QKNorm

    names, caps = [], []
    for name, m in model.named_modules():
        if isinstance(m, QKNorm):
            d = m.q_norm.weight.numel()
            names.append(name)
            caps.append(math.sqrt(d) * m.q_norm.weight.detach().abs().max().float()
                        * m.k_norm.weight.detach().abs().max().float())
    if not caps:
        return {}
    return dict(zip(names, torch.stack(caps).tolist()))


@torch.no_grad()
def _attn_stats(q: torch.Tensor, k: torch.Tensor, attn_mask: Optional[torch.Tensor],
                is_causal: bool, scale: Optional[float],
                max_elems: int) -> Tuple[float, float, int]:
    """(max |logit| over attended pairs, Σ normalised row entropy, #rows) of
    one SDPA call.  Row entropy is divided by log(#attended keys) (1 =
    uniform, 0 = one-hot); rows with a single key are skipped."""
    D, Sq, Sk = q.shape[-1], q.shape[-2], k.shape[-2]
    s = scale if scale is not None else D ** -0.5
    lead = q.shape[:-2]
    qf, kf = q.detach().reshape(-1, Sq, D), k.detach().reshape(-1, Sk, D)
    keep = bias = None
    if is_causal:          # SDPA's is_causal: top-left aligned lower triangle
        keep = torch.ones(Sq, Sk, dtype=torch.bool, device=q.device).tril()
    if attn_mask is not None:
        m = attn_mask.detach().expand(*lead, Sq, Sk).reshape(-1, Sq, Sk)
        if m.dtype == torch.bool:
            keep = m if keep is None else keep & m
        else:
            bias = m.float()
            fin = bias > float("-inf")
            keep = fin if keep is None else keep & fin
    top, ent_sum, rows = 0.0, 0.0, 0
    step = max(1, max_elems // max(1, Sq * Sk))
    for i in range(0, qf.shape[0], step):
        logits = qf[i:i + step].float() @ kf[i:i + step].float().transpose(-2, -1) * s
        kp = None
        if bias is not None:
            logits = logits + bias[i:i + step]
        if keep is not None:
            kp = keep if keep.dim() == 2 else keep[i:i + step]
            kp = kp.expand_as(logits)
            top = max(top, float(logits.masked_fill(~kp, 0.0).abs().amax()))
            logits = logits.masked_fill(~kp, float("-inf"))
            n_keys = kp.sum(-1).float()
        else:
            top = max(top, float(logits.abs().amax()))
            n_keys = torch.full(logits.shape[:-1], float(Sk), device=logits.device)
        p = logits.softmax(-1)
        ent = -torch.special.xlogy(p, p).sum(-1)
        ok = n_keys > 1
        ent_sum += float((ent[ok] / n_keys[ok].log()).sum())
        rows += int(ok.sum())
    return top, ent_sum, rows


@contextlib.contextmanager
def attention_probe(model: nn.Module, max_elems: int = 1 << 24) -> Iterator[Dict[str, Dict]]:
    """Record every ``F.scaled_dot_product_attention`` call made inside
    ``model``'s forward while the context is open.

    Yields a dict filled on exit: ``{caller: {"max_logit", "entropy",
    "calls"}}`` — the calling module's name (the innermost module running
    when SDPA is called; ``[causal]`` appended for causal calls, i.e. the
    Stage-1 temporal attention), the max |pre-softmax logit| over its calls,
    and the mean normalised attention entropy (1 = uniform, 0 = one-hot).
    Recurrent-depth blocks run once per iteration: their calls aggregate.
    Attention that does not go through SDPA (nn.MultiheadAttention.forward)
    is not seen.
    """
    stats: Dict[str, List[float]] = {}
    stack: List[str] = []

    # hooks must return None (a forward hook's return value replaces the
    # module output)
    def enter(name: str):
        def pre_hook(_m, _a) -> None:
            stack.append(name)
        return pre_hook

    def leave(_m, _a, _o) -> None:
        stack.pop()

    handles = []
    for name, m in model.named_modules():
        handles.append(m.register_forward_pre_hook(enter(name)))
        handles.append(m.register_forward_hook(leave))
    orig = F.scaled_dot_product_attention

    def spy(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=None, **kw):
        name = (stack[-1] if stack else "") or "<root>"
        if is_causal:
            name += "[causal]"
        top, ent, rows = _attn_stats(query, key, attn_mask, is_causal, scale, max_elems)
        s = stats.setdefault(name, [0.0, 0.0, 0, 0])
        s[0] = max(s[0], top)
        s[1] += ent
        s[2] += rows
        s[3] += 1
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, scale=scale, **kw)

    out: Dict[str, Dict] = {}
    F.scaled_dot_product_attention = spy
    try:
        yield out
    finally:
        F.scaled_dot_product_attention = orig
        for h in handles:
            h.remove()
        out.update({n: {"max_logit": s[0], "entropy": s[1] / max(s[2], 1), "calls": s[3]}
                    for n, s in stats.items()})


@torch.no_grad()
def logit_lse_absmax(pred: Mapping, frame_mask: torch.Tensor) -> Optional[float]:
    """max |logsumexp| of the decoder output logits on valid frames: every
    ``*logits`` tensor shaped [B, T, ...] (the dims after T flattened into
    one distribution; a [B, T] Bernoulli logit is its own value).  Fully
    masked rows (lse ≤ −1e4) are skipped.  None if there is nothing."""
    valid = frame_mask.bool()
    vals = []
    for k, v in pred.items():
        if not (k.endswith("logits") and torch.is_tensor(v) and v.is_floating_point()):
            continue
        if v.dim() < 2 or tuple(v.shape[:2]) != tuple(valid.shape):
            continue
        lse = v.float().reshape(*v.shape[:2], -1).logsumexp(-1)[valid]
        lse = lse[torch.isfinite(lse) & (lse > -1e4)]
        if lse.numel():
            vals.append(lse.abs().max())
    return float(torch.stack(vals).max()) if vals else None


def format_attention(stats: Mapping[str, Mapping], lse: Optional[float] = None) -> str:
    """One log line: the worst layer (max logit / min entropy), the output
    log-partition, then every caller as ``name=max_logit/entropy``."""
    parts = []
    if stats:
        hi = max(stats, key=lambda n: stats[n]["max_logit"])
        lo = min(stats, key=lambda n: stats[n]["entropy"])
        parts.append(f"max_logit={stats[hi]['max_logit']:.2f}@{hi} "
                     f"min_entropy={stats[lo]['entropy']:.3f}@{lo}")
    if lse is not None:
        parts.append(f"lse_absmax={lse:.2f}")
    if stats:
        parts.append(" ".join(f"{n}={s['max_logit']:.1f}/{s['entropy']:.2f}"
                              for n, s in stats.items()))
    return "  ".join(parts)
