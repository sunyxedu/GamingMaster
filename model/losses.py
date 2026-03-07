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

    Components:
      - Source CE    : cross-entropy over H*W positions      (valid mask: count>0)
      - Direction CE : cross-entropy over 4 directions       (valid mask: count>0)
      - Split CE     : cross-entropy over 2 options          (valid mask: count>0)
                       0=all-but-1, 1=50%
      - Pass BCE     : binary cross-entropy on pass head     (seq_mask: all timesteps)
                       target=1 when count==0 (no move)
      - Value MSE    : mean squared error on game-value      (seq_mask: all timesteps)
    """

    def __init__(
        self,
        source_weight: float = 1.0,
        direction_weight: float = 1.0,
        split_weight: float = 0.5,
        pass_weight: float = 10.0,
        value_weight: float = 0.5,
        q_halt_weight: float = 0.5,
        move_match_weight: float = 2.0,
        grid_w: int = 25,
    ):
        super().__init__()
        self.source_weight = source_weight
        self.direction_weight = direction_weight
        self.split_weight = split_weight
        self.pass_weight = pass_weight
        self.value_weight = value_weight
        self.q_halt_weight = q_halt_weight
        self.move_match_weight = move_match_weight
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
                split_logits     [B, T, 2]  (0=all-but-1, 1=50%)
                pass_logits      [B, T]     (>0 → pass)
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

        # --- Split loss (2-class: 0=all-but-1, 1=50%) ----------------------
        # Only on valid (count>0) timesteps — split is meaningless without a move.
        n_split = pred["split_logits"].size(-1)
        tgt_split = tgt["ratio_id"][:, :, 0].long()
        # Clamp invalid positions to 0 so CE doesn't error; they're masked out anyway.
        tgt_split = torch.where(valid, tgt_split, torch.zeros_like(tgt_split))

        split_loss_all = F.cross_entropy(
            pred["split_logits"].reshape(-1, n_split),
            tgt_split.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        split_loss_all = torch.nan_to_num(split_loss_all, nan=0.0)
        split_loss = (split_loss_all * valid).sum() / n_valid

        # --- Pass loss (binary: should the agent pass this turn?) ----------
        # target=1 when count==0 (no move), target=0 when count>0
        tgt_pass = (~move_mask).float()  # [B, T]
        pass_loss_all = F.binary_cross_entropy_with_logits(
            pred["pass_logits"], tgt_pass, reduction="none",
        )
        pass_loss = (pass_loss_all * seq_mask.float()).sum() / n_seq

        # --- Move-mismatch loss (extra penalty for incomplete moves) ------
        # Only on valid (count>0) timesteps.  A correct move requires
        # src + dir + split all correct AND pass_head says "don't pass".
        move_match_loss = source_loss.new_tensor(0.0)
        if valid.any():
            with torch.no_grad():
                pred_src = pred["source_logits"].argmax(dim=-1)   # [B, T]
                pred_dir = pred["direction_logits"].argmax(dim=-1)
                pred_split = pred["split_logits"].argmax(dim=-1)
                pred_pass = pred["pass_logits"] > 0  # True = pass
                all_match = (
                    (pred_src == tgt_src)
                    & (pred_dir == tgt_dir)
                    & (pred_split == tgt_split)
                    & ~pred_pass  # must NOT pass on a valid-move timestep
                )
                mismatch_mask = ~all_match & valid
            n_mismatch = mismatch_mask.sum().clamp(min=1).float()
            combined_per_step = source_loss_all + dir_loss_all + split_loss_all
            move_match_loss = (combined_per_step * mismatch_mask.float()).sum() / n_mismatch

        # --- Value loss --------------------------------------------------
        value_loss_all = F.mse_loss(
            pred["value"], tgt["value"], reduction="none",
        )
        value_loss = (value_loss_all * seq_mask.float()).sum() / n_seq

        # --- Q-halt loss (TRM halting) ------------------------------------
        q_halt_loss = source_loss.new_tensor(0.0)
        if "q_halt" in pred and valid.any():
            with torch.no_grad():
                predicted_src = pred["source_logits"].argmax(dim=-1)  # [B, T]
                correct = (predicted_src == tgt_src).float()
            q_halt_loss = F.binary_cross_entropy_with_logits(
                pred["q_halt"], correct, reduction="none",
            )
            q_halt_loss = (q_halt_loss * valid.float()).sum() / n_valid

        # --- Total -------------------------------------------------------
        total = (
            self.source_weight * source_loss
            + self.direction_weight * dir_loss
            + self.split_weight * split_loss
            + self.pass_weight * pass_loss
            + self.value_weight * value_loss
            + self.q_halt_weight * q_halt_loss
            + self.move_match_weight * move_match_loss
        )

        return {
            "loss": total,
            "source_loss": source_loss.detach(),
            "direction_loss": dir_loss.detach(),
            "split_loss": split_loss.detach(),
            "pass_loss": pass_loss.detach(),
            "value_loss": value_loss.detach(),
            "q_halt_loss": q_halt_loss.detach() if isinstance(q_halt_loss, torch.Tensor) else q_halt_loss,
            "move_match_loss": move_match_loss.detach() if isinstance(move_match_loss, torch.Tensor) else move_match_loss,
        }


class RLLoss(nn.Module):
    """PPO loss for RL training.

    Takes **pre-computed** log-probabilities and entropy produced by
    ``GeneralsNetwork.compute_action_log_prob()``.  This cleanly
    separates action-space logic (in the model) from the RL objective
    (here).

    Components:
      - Clipped policy-gradient loss
      - Value-function MSE
      - Entropy bonus (maximised)

    Extra diagnostics returned (detached):
      - approx_kl:      0.5 * mean((log_prob - old_log_prob)^2)
      - clip_fraction:   fraction of samples where ratio was clipped
    """

    def __init__(
        self,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
        clip_range: float = 0.2,
    ):
        super().__init__()
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.clip_range = clip_range

    def forward(
        self,
        log_probs: torch.Tensor,
        entropy: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            log_probs      [B, T]  current-policy log π(a|s)  (differentiable)
            entropy        [B, T]  current-policy entropy     (differentiable)
            values         [B, T]  value-head predictions     (differentiable)
            returns        [B, T]  discounted returns from GAE
            advantages     [B, T]  GAE advantages (should be normalized)
            old_log_probs  [B, T]  log π_old(a|s) from rollout
            mask           [B, T]  bool — True for valid timesteps
        """
        if mask is None:
            mask = torch.ones_like(log_probs, dtype=torch.bool)
        mask_f = mask.float()
        n_valid = mask_f.sum().clamp(min=1)

        # --- PPO clipped policy loss ---
        ratio = torch.exp(log_probs - old_log_probs)
        clip_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(
            ratio * advantages,
            clip_ratio * advantages,
        )
        policy_loss = (policy_loss * mask_f).sum() / n_valid

        # --- value loss ---
        value_loss = F.mse_loss(values, returns, reduction="none")
        value_loss = (value_loss * mask_f).sum() / n_valid

        # --- entropy bonus (negative → we *maximise* entropy) ---
        entropy_bonus = (entropy * mask_f).sum() / n_valid

        total = (
            policy_loss
            + self.value_weight * value_loss
            - self.entropy_weight * entropy_bonus
        )

        return {
            "loss": total,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy_bonus.detach(),
            "approx_kl": (
                0.5 * (log_probs - old_log_probs).pow(2) * mask_f
            ).sum().detach() / n_valid,
            "clip_fraction": (
                ((ratio - 1.0).abs() > self.clip_range).float() * mask_f
            ).sum().detach() / n_valid,
        }
