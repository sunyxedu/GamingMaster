"""
Loss functions for Imitation Learning (IL) and Reinforcement Learning (RL).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImitationLoss(nn.Module):
    """
    Imitation Learning loss for the generals.io bot.

    Components (all masked to valid moves only):
      - Source CE    : cross-entropy over H*W positions
      - Direction CE : cross-entropy over 4 directions
      - Split CE     : cross-entropy over 2 split options
      - Value MSE    : mean squared error on game-value estimate
    """

    def __init__(
        self,
        source_weight: float = 1.0,
        direction_weight: float = 1.0,
        split_weight: float = 0.5,
        value_weight: float = 0.5,
        grid_w: int = 25,
    ):
        super().__init__()
        self.source_weight = source_weight
        self.direction_weight = direction_weight
        self.split_weight = split_weight
        self.value_weight = value_weight
        self.grid_w = grid_w

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        lengths: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: model output dict with keys:
                source_logits    [B, T, N]
                direction_logits [B, T, 4]
                split_logits     [B, T, 2]
                value            [B, T]
            tgt: target dict with keys:
                count    [B, T]       – number of actions (0 or 1)
                src_y    [B, T, K]    – source row
                src_x    [B, T, K]    – source col
                dir_id   [B, T, K]    – direction
                ratio_id [B, T, K]    – split option
                value    [B, T]       – estimated game value
            lengths: list[int] – actual sequence lengths per sample

        Returns:
            dict with individual loss components and total loss
        """
        B, T, N = pred["source_logits"].shape
        device = pred["source_logits"].device

        # --- Build masks ------------------------------------------------
        # move_mask: True where a valid move exists (count > 0)
        move_mask = tgt["count"] > 0  # [B, T]

        # seq_mask: True where the timestep is within the actual sequence
        if lengths is not None:
            seq_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for i, length in enumerate(lengths):
                seq_mask[i, :length] = True
        else:
            seq_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Combined: valid move AND within sequence
        valid = move_mask & seq_mask  # [B, T]
        n_valid = valid.sum().clamp(min=1).float()

        # seq-only mask for value loss (always predicted, even without move)
        n_seq = seq_mask.sum().clamp(min=1).float()

        # --- Source loss -------------------------------------------------
        # Target: flat index = src_y * W + src_x  (first action only, K=0)
        tgt_src = (tgt["src_y"][:, :, 0] * self.grid_w + tgt["src_x"][:, :, 0]).long()
        tgt_src = tgt_src.clamp(0, N - 1)

        source_loss_all = F.cross_entropy(
            pred["source_logits"].reshape(-1, N),
            tgt_src.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        source_loss_all = torch.nan_to_num(source_loss_all, nan=0.0)
        source_loss = (source_loss_all * valid).sum() / n_valid

        # --- Direction loss ----------------------------------------------
        tgt_dir = tgt["dir_id"][:, :, 0].long()
        dir_loss_all = F.cross_entropy(
            pred["direction_logits"].reshape(-1, pred["direction_logits"].size(-1)),
            tgt_dir.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        dir_loss_all = torch.nan_to_num(dir_loss_all, nan=0.0)
        dir_loss = (dir_loss_all * valid).sum() / n_valid

        # --- Split loss --------------------------------------------------
        tgt_split = tgt["ratio_id"][:, :, 0].long()
        split_loss_all = F.cross_entropy(
            pred["split_logits"].reshape(-1, pred["split_logits"].size(-1)),
            tgt_split.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        split_loss_all = torch.nan_to_num(split_loss_all, nan=0.0)
        split_loss = (split_loss_all * valid).sum() / n_valid

        # --- Value loss --------------------------------------------------
        value_loss_all = F.mse_loss(
            pred["value"], tgt["value"], reduction="none",
        )
        value_loss = (value_loss_all * seq_mask.float()).sum() / n_seq

        # --- Total -------------------------------------------------------
        total = (
            self.source_weight * source_loss
            + self.direction_weight * dir_loss
            + self.split_weight * split_loss
            + self.value_weight * value_loss
        )

        return {
            "loss": total,
            "source_loss": source_loss.detach(),
            "direction_loss": dir_loss.detach(),
            "split_loss": split_loss.detach(),
            "value_loss": value_loss.detach(),
        }


class RLLoss(nn.Module):
    """
    Placeholder for future RL losses (PPO / IMPALA style).

    Includes:
      - Policy gradient loss
      - Value loss
      - Entropy bonus
      - (Optional) World-model prediction error as intrinsic reward
    """

    def __init__(
        self,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        returns: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """PPO-style loss (to be implemented when RL training begins)."""
        B, T, N = pred["source_logits"].shape

        # Source log-probs
        source_log_probs = F.log_softmax(pred["source_logits"], dim=-1)
        source_action = actions["source_idx"]  # [B, T]
        source_lp = source_log_probs.gather(-1, source_action.unsqueeze(-1)).squeeze(-1)

        # Direction log-probs
        dir_log_probs = F.log_softmax(pred["direction_logits"], dim=-1)
        dir_action = actions["direction"]  # [B, T]
        dir_lp = dir_log_probs.gather(-1, dir_action.unsqueeze(-1)).squeeze(-1)

        # Split log-probs
        split_log_probs = F.log_softmax(pred["split_logits"], dim=-1)
        split_action = actions["split"]  # [B, T]
        split_lp = split_log_probs.gather(-1, split_action.unsqueeze(-1)).squeeze(-1)

        # Total log-prob
        total_lp = source_lp + dir_lp + split_lp

        # PPO clipped objective
        ratio = torch.exp(total_lp - old_log_probs)
        clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(pred["value"], returns)

        # Entropy bonus (source only for now)
        source_probs = F.softmax(pred["source_logits"], dim=-1)
        entropy = -(source_probs * source_log_probs).sum(-1).mean()

        total = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy

        return {
            "loss": total,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
        }
