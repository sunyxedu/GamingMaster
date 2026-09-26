"""Top-q advantage filter (AverageJoe's top-25%).

The policy surrogate trains only on the tokens with the largest |A| — the
decisive moves — which also mutes the noise-dominated gradient of long,
flat stretches.  The critic, the KL term and the diagnostics keep the full
mask; pass the result as ``policy_mask`` to
:func:`training.rl_joint.losses.decoupled_ppo_loss`.
"""

from __future__ import annotations

from typing import Optional

import torch

from training.rl_joint.stats import as_mask


@torch.no_grad()
def top_q_mask(
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    q: float = 0.25,
    per_row: bool = False,
    min_count: int = 8,
    key: str = "abs",
) -> torch.Tensor:
    """Bool mask of the valid tokens whose score is within the top ``q``
    fraction.

    Args:
        advantages: [...] (typically normalised)
        mask:       [...] bool valid tokens (default all)
        q:          kept fraction in (0, 1]; ``q >= 1`` keeps every valid token
        per_row:    rank within each row (last dim = time, e.g. per episode,
                    as the generals trainer does) instead of over all tokens
        min_count:  a group with fewer valid tokens keeps all of them
        key:        "abs" ranks by |A|, "signed" by A itself (largest first)

    The kept count per group is ``max(1, round(q · n_valid))``; ties at the
    threshold are all kept (the kept fraction can exceed q).
    """
    if not 0.0 < q:
        raise ValueError(f"q must be positive, got {q}")
    valid = as_mask(advantages, mask)
    if q >= 1.0:
        return valid.clone()
    if key == "abs":
        score = advantages.detach().float().abs()
    elif key == "signed":
        score = advantages.detach().float()
    else:
        raise ValueError(f"key must be 'abs' or 'signed', got {key!r}")
    neg_inf = torch.tensor(float("-inf"), device=score.device)
    score = torch.where(valid, score, neg_inf)

    if per_row:
        if score.dim() == 0:
            raise ValueError("per_row needs at least one dimension")
        flat = score.reshape(-1, score.shape[-1])
        vflat = valid.reshape(-1, score.shape[-1])
    else:
        flat = score.reshape(1, -1)
        vflat = valid.reshape(1, -1)
    n = vflat.sum(-1)                                              # [G]
    k = torch.clamp(torch.round(n.float() * q).long(), min=1)       # [G]
    sorted_desc = flat.sort(dim=-1, descending=True).values
    idx = (k - 1).clamp(max=flat.shape[-1] - 1).unsqueeze(-1)
    thresh = sorted_desc.gather(-1, idx)                            # [G, 1]
    keep = (flat >= thresh) & vflat
    small = (n < min_count).unsqueeze(-1)
    keep = torch.where(small, vflat, keep)
    return keep.reshape(valid.shape)
