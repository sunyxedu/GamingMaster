"""
Reinforcement-learning utilities.

- GAE (Generalized Advantage Estimation)
- RolloutBuffer (trajectory storage for PPO)
- KL-divergence between policies (regularisation)
- Encoder freeze / unfreeze helpers
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# GAE
# ======================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    Args:
        rewards      [T]  rewards collected during rollout
        values       [T]  value-head predictions (detached)
        dones        [T]  bool — True if episode ended at this step
        gamma             discount factor
        lam               GAE lambda (bias–variance trade-off)
        last_value        V(s_{T+1}) bootstrap; 0 if final step is terminal

    Returns:
        advantages   [T]
        returns      [T]  (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        # If episode ended at step t, no bootstrapping
        if dones[t]:
            next_value = 0.0
            last_gae = 0.0

        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ======================================================================
# Rollout buffer
# ======================================================================

class RolloutBuffer:
    """Stores a single trajectory for one player during self-play.

    Usage::

        buf = RolloutBuffer()
        for each step:
            buf.add(obs, action, log_prob, value, reward, done)
        buf.compute_returns_and_advantages(last_value=…)
        batch = buf.to_tensor_batch(device)
    """

    def __init__(self) -> None:
        self.grids: List[torch.Tensor] = []          # [C, H, W]
        self.globals_: List[torch.Tensor] = []        # [G]
        self.legal_masks: List[torch.Tensor] = []    # [1, H, W]
        self.source_idxs: List[int] = []
        self.directions: List[int] = []
        self.splits: List[int] = []
        self.passes: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    # ---- collection ----------------------------------------------------

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: Dict[str, int],
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.grids.append(obs["grid"])
        self.globals_.append(obs["globals"])
        self.legal_masks.append(obs["legal_mask"])
        self.source_idxs.append(action["source_idx"])
        self.directions.append(action["direction"])
        self.splits.append(action["split"])
        self.passes.append(action.get("passed", False))
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    # ---- advantage computation -----------------------------------------

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        last_value: float = 0.0,
    ) -> None:
        self.advantages, self.returns = compute_gae(
            rewards=torch.tensor(self.rewards, dtype=torch.float32),
            values=torch.tensor(self.values, dtype=torch.float32),
            dones=torch.tensor(self.dones, dtype=torch.bool),
            gamma=gamma,
            lam=lam,
            last_value=last_value,
        )

    # ---- convert to tensors for PPO update -----------------------------

    def to_tensor_batch(self, device: torch.device) -> Dict:
        """Stack trajectory into a single-sample batch ``[1, T, …]``."""
        assert self.advantages is not None, "call compute_returns_and_advantages first"
        T = len(self.grids)
        return {
            "grid_seq": torch.stack(self.grids).unsqueeze(0).to(device),
            "globals_seq": torch.stack(self.globals_).unsqueeze(0).to(device),
            "legal_mask": torch.stack(self.legal_masks).unsqueeze(0).to(device),
            "actions": {
                "source_idx": torch.tensor(
                    self.source_idxs, dtype=torch.long, device=device,
                ).unsqueeze(0),
                "direction": torch.tensor(
                    self.directions, dtype=torch.long, device=device,
                ).unsqueeze(0),
                "split": torch.tensor(
                    self.splits, dtype=torch.long, device=device,
                ).unsqueeze(0),
                "passed": torch.tensor(
                    self.passes, dtype=torch.bool, device=device,
                ).unsqueeze(0),
            },
            "old_log_probs": torch.tensor(
                self.log_probs, dtype=torch.float32, device=device,
            ).unsqueeze(0),
            "advantages": self.advantages.to(device).unsqueeze(0),
            "returns": self.returns.to(device).unsqueeze(0),
            "length": T,
        }

    # ---- housekeeping --------------------------------------------------

    def clear(self) -> None:
        self.__init__()  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.rewards)


# ======================================================================
# KL divergence between policies (source-only, for regularisation)
# ======================================================================

def compute_policy_kl(
    current_source_logits: torch.Tensor,
    reference_source_logits: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL( π_ref ‖ π_current ) over the source-action distribution.

    Used as regularisation to prevent the RL policy from drifting too far
    from the IL-pretrained policy.  Only computed over the source head
    because it dominates the action space (625 positions vs 4+2).

    Args:
        current_source_logits    [B, T, N]
        reference_source_logits  [B, T, N]  (detached)
        legal_mask               [B, T, 1, H, W] or None
        mask                     [B, T] bool – valid timesteps

    Returns:
        scalar KL divergence (mean over valid positions)
    """
    B, T, N = current_source_logits.shape

    # Apply legal mask
    if legal_mask is not None:
        lm = legal_mask.reshape(B, T, N)
        cur_masked = current_source_logits.masked_fill(~lm, float("-inf"))
        ref_masked = reference_source_logits.masked_fill(~lm, float("-inf"))
    else:
        cur_masked = current_source_logits
        ref_masked = reference_source_logits

    # Handle fully-masked rows
    all_inf = cur_masked.isinf() & (cur_masked < 0)
    bad = all_inf.all(dim=-1)
    if bad.any():
        cur_masked = cur_masked.clone()
        ref_masked = ref_masked.clone()
        cur_masked[bad, 0] = 0.0
        ref_masked[bad, 0] = 0.0

    log_p_ref = F.log_softmax(ref_masked, dim=-1)
    log_p_cur = F.log_softmax(cur_masked, dim=-1)
    p_ref = torch.exp(log_p_ref)

    # KL = Σ p_ref * (log p_ref - log p_cur)
    kl = (p_ref * (log_p_ref - log_p_cur)).nan_to_num(0.0).sum(-1)  # [B, T]

    if mask is not None:
        n_valid = mask.float().sum().clamp(min=1)
        kl = (kl * mask.float()).sum() / n_valid
    else:
        kl = kl.mean()

    return kl


def compute_policy_kl_rl(
    current_source_logits: torch.Tensor,
    reference_source_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL( π_ref ‖ π_current ) without legal mask (RL mode).

    Same as ``compute_policy_kl`` but operates on raw logits over all 625
    positions without masking.
    """
    log_p_ref = F.log_softmax(reference_source_logits, dim=-1)
    log_p_cur = F.log_softmax(current_source_logits, dim=-1)
    p_ref = torch.exp(log_p_ref)

    kl = (p_ref * (log_p_ref - log_p_cur)).nan_to_num(0.0).sum(-1)

    if mask is not None:
        n_valid = mask.float().sum().clamp(min=1)
        kl = (kl * mask.float()).sum() / n_valid
    else:
        kl = kl.mean()

    return kl


# ======================================================================
# Encoder freeze / unfreeze
# ======================================================================

def freeze_encoder(model: nn.Module) -> None:
    """Freeze Stage-1 encoder weights (SpatialEncoder + global_proj + ST layers).

    Useful during early RL fine-tuning to stabilise representations
    learned during IL pre-training.
    """
    for param in model.spatial_encoder.parameters():
        param.requires_grad = False
    for param in model.global_proj.parameters():
        param.requires_grad = False
    if hasattr(model, "st_layers"):
        for param in model.st_layers.parameters():
            param.requires_grad = False
    if hasattr(model, "history_encoder"):
        for param in model.history_encoder.parameters():
            param.requires_grad = False
    # Positional embeddings
    model.row_embed.requires_grad = False
    model.col_embed.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
