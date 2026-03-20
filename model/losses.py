"""
Loss functions for Imitation Learning (IL) and Reinforcement Learning (RL).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Original ImitationLoss (kept for reference)
# ======================================================================
# class ImitationLoss(nn.Module):
#     """
#     Original IL loss with move-mismatch penalty.
#     Weights: source=1.5, direction=1.0, split=0.8, pass=10.0,
#              value=0.5, q_halt=0.5, move_match=1.5
#     """
#     def __init__(self, source_weight=1.5, direction_weight=1.0,
#                  split_weight=0.8, pass_weight=10.0, value_weight=0.5,
#                  q_halt_weight=0.5, move_match_weight=1.5, grid_w=25):
#         super().__init__()
#         self.source_weight = source_weight
#         self.direction_weight = direction_weight
#         self.split_weight = split_weight
#         self.pass_weight = pass_weight
#         self.value_weight = value_weight
#         self.q_halt_weight = q_halt_weight
#         self.move_match_weight = move_match_weight
#         self.grid_w = grid_w
#
#     def forward(self, pred, tgt, lengths=None):
#         B, T, N = pred["source_logits"].shape
#         device = pred["source_logits"].device
#         move_mask = tgt["count"] > 0
#         if lengths is not None:
#             seq_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
#             for i, length in enumerate(lengths):
#                 seq_mask[i, :length] = True
#         else:
#             seq_mask = torch.ones(B, T, dtype=torch.bool, device=device)
#         valid = move_mask & seq_mask
#         n_valid = valid.sum().clamp(min=1).float()
#         n_seq = seq_mask.sum().clamp(min=1).float()
#         tgt_src = (tgt["src_y"][:,:,0] * self.grid_w + tgt["src_x"][:,:,0]).long().clamp(0, N-1)
#         tgt_dir = tgt["dir_id"][:,:,0].long()
#         tgt_split = torch.where(valid, tgt["ratio_id"][:,:,0].long(), torch.zeros_like(tgt["ratio_id"][:,:,0]))
#
#         source_loss_all = torch.nan_to_num(F.cross_entropy(
#             pred["source_logits"].reshape(-1,N), tgt_src.reshape(-1), reduction="none").reshape(B,T), nan=0.0)
#         source_loss = (source_loss_all * valid).sum() / n_valid
#
#         dir_loss_all = torch.nan_to_num(F.cross_entropy(
#             pred["direction_logits"].reshape(-1,4), tgt_dir.reshape(-1), reduction="none").reshape(B,T), nan=0.0)
#         dir_loss = (dir_loss_all * valid).sum() / n_valid
#
#         split_loss_all = torch.nan_to_num(F.cross_entropy(
#             pred["split_logits"].reshape(-1,2), tgt_split.reshape(-1), reduction="none").reshape(B,T), nan=0.0)
#         split_loss = (split_loss_all * valid).sum() / n_valid
#
#         tgt_pass = (~move_mask).float()
#         pass_loss = (F.binary_cross_entropy_with_logits(
#             pred["pass_logits"], tgt_pass, reduction="none") * seq_mask.float()).sum() / n_seq
#
#         # Move-mismatch: re-apply src+dir+split CE on steps where full move is wrong
#         with torch.no_grad():
#             mismatch = ~((pred["source_logits"].argmax(-1)==tgt_src)
#                 & (pred["direction_logits"].argmax(-1)==tgt_dir)
#                 & (pred["split_logits"].argmax(-1)==tgt_split)
#                 & ~(pred["pass_logits"]>0)) & valid
#         move_match_loss = ((source_loss_all+dir_loss_all+split_loss_all) * mismatch.float()).sum() / n_seq
#
#         value_loss = (F.mse_loss(pred["value"], tgt["value"], reduction="none") * seq_mask.float()).sum() / n_seq
#         # ... q_halt_loss omitted for brevity ...
#         total = (self.source_weight*source_loss + self.direction_weight*dir_loss
#                  + self.split_weight*split_loss + self.pass_weight*pass_loss
#                  + self.value_weight*value_loss + self.move_match_weight*move_match_loss)
#         return {"loss": total, "source_loss": source_loss.detach(), ...}
# ======================================================================


class ImitationLoss(nn.Module):
    """
    Imitation Learning loss for the generals.io bot.

    Components:
      - Source: Focal CE over H*W positions (auto-focuses on hard samples)
      - Direction: Soft-conditional CE (weighted by source confidence)
      - Split: Soft-conditional CE (weighted by source confidence)
      - Pass BCE: binary cross-entropy on pass head
      - Value MSE: mean squared error on game-value
      - Q-halt: BCE for TRM halting supervision

    Accuracy metrics (returned detached):
      - src_acc:  source top-1 accuracy on valid timesteps
      - dir_acc:  direction accuracy (only where source is correct)
      - full_acc: full move match (src + dir + split + pass all correct)
    """

    def __init__(
        self,
        source_weight: float = 3.0,
        direction_weight: float = 1.5,
        split_weight: float = 0.3,
        pass_weight: float = 1.0,
        value_weight: float = 0.2,
        q_halt_weight: float = 0.2,
        focal_gamma: float = 1.5,
        grid_w: int = 25,
    ):
        super().__init__()
        self.source_weight = source_weight
        self.direction_weight = direction_weight
        self.split_weight = split_weight
        self.pass_weight = pass_weight
        self.value_weight = value_weight
        self.q_halt_weight = q_halt_weight
        self.focal_gamma = focal_gamma
        self.grid_w = grid_w

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        lengths: Optional[list] = None,
        tf_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, N = pred["source_logits"].shape
        device = pred["source_logits"].device

        # --- Build masks ------------------------------------------------
        move_mask = tgt["count"] > 0  # [B, T]

        if lengths is not None:
            seq_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for i, length in enumerate(lengths):
                seq_mask[i, :length] = True
        else:
            seq_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        valid = move_mask & seq_mask
        n_valid = valid.sum().clamp(min=1).float()
        n_seq = seq_mask.sum().clamp(min=1).float()

        # --- Targets -----------------------------------------------------
        tgt_src = (tgt["src_y"][:, :, 0] * self.grid_w + tgt["src_x"][:, :, 0]).long()
        tgt_src = tgt_src.clamp(0, N - 1)
        tgt_dir = tgt["dir_id"][:, :, 0].long()
        tgt_split = tgt["ratio_id"][:, :, 0].long()
        tgt_split = torch.where(valid, tgt_split, torch.zeros_like(tgt_split))

        # --- Source loss: Focal CE ----------------------------------------
        source_ce_all = F.cross_entropy(
            pred["source_logits"].reshape(-1, N),
            tgt_src.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        source_ce_all = torch.nan_to_num(source_ce_all, nan=0.0)

        # Focal modulation: (1 - p_t)^gamma
        with torch.no_grad():
            p_t = torch.exp(-source_ce_all)  # prob assigned to correct class
            focal_weight = (1.0 - p_t).pow(self.focal_gamma)
        source_loss_all = focal_weight * source_ce_all
        source_loss = (source_loss_all * valid).sum() / n_valid

        # --- Dir/split weight: TF steps get full weight, others get src_conf ---
        with torch.no_grad():
            src_prob = F.softmax(pred["source_logits"], dim=-1)
            src_conf = src_prob.gather(-1, tgt_src.unsqueeze(-1)).squeeze(-1)  # [B, T]

            if tf_mask is not None:
                # TF steps: direction sees correct source features → weight=1.0
                # Non-TF steps: direction sees predicted source → weight=src_conf
                cond_weight = torch.where(tf_mask, torch.ones_like(src_conf), src_conf)
            else:
                # No TF info (e.g. inference) → always soft-conditional
                cond_weight = src_conf

            cond_weight_valid = cond_weight * valid.float()
            n_cond = cond_weight_valid.sum().clamp(min=1.0)

        # --- Direction loss: conditional CE -------------------------------
        dir_loss_all = F.cross_entropy(
            pred["direction_logits"].reshape(-1, pred["direction_logits"].size(-1)),
            tgt_dir.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        dir_loss_all = torch.nan_to_num(dir_loss_all, nan=0.0)
        dir_loss = (dir_loss_all * cond_weight_valid).sum() / n_cond

        # --- Split loss: conditional CE -----------------------------------
        n_split = pred["split_logits"].size(-1)
        split_loss_all = F.cross_entropy(
            pred["split_logits"].reshape(-1, n_split),
            tgt_split.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        split_loss_all = torch.nan_to_num(split_loss_all, nan=0.0)
        split_loss = (split_loss_all * cond_weight_valid).sum() / n_cond

        # --- Pass loss ----------------------------------------------------
        tgt_pass = (~move_mask).float()
        pass_loss_all = F.binary_cross_entropy_with_logits(
            pred["pass_logits"], tgt_pass, reduction="none",
        )
        pass_loss = (pass_loss_all * seq_mask.float()).sum() / n_seq

        # --- Value loss ---------------------------------------------------
        value_loss_all = F.mse_loss(
            pred["value"], tgt["value"], reduction="none",
        )
        value_loss = (value_loss_all * seq_mask.float()).sum() / n_seq

        # --- Q-halt loss --------------------------------------------------
        if "q_halt" in pred and valid.any():
            with torch.no_grad():
                predicted_src = pred["source_logits"].argmax(dim=-1)
                correct = (predicted_src == tgt_src).float()
            q_halt_loss = F.binary_cross_entropy_with_logits(
                pred["q_halt"], correct, reduction="none",
            )
            q_halt_loss = (q_halt_loss * valid.float()).sum() / n_valid
        elif "q_halt" in pred:
            q_halt_loss = 0.0 * pred["q_halt"].sum()
        else:
            q_halt_loss = source_loss.new_tensor(0.0)

        # --- Total --------------------------------------------------------
        total = (
            self.source_weight * source_loss
            + self.direction_weight * dir_loss
            + self.split_weight * split_loss
            + self.pass_weight * pass_loss
            + self.value_weight * value_loss
            + self.q_halt_weight * q_halt_loss
        )

        # --- Accuracy metrics (detached) ----------------------------------
        with torch.no_grad():
            pred_src = pred["source_logits"].argmax(dim=-1)
            pred_dir = pred["direction_logits"].argmax(dim=-1)
            pred_split = pred["split_logits"].argmax(dim=-1)
            pred_pass = pred["pass_logits"] > 0

            src_ok = (pred_src == tgt_src) & valid
            src_acc = src_ok.sum().float() / n_valid

            # Direction accuracy only where source is correct
            n_src_ok = src_ok.sum().clamp(min=1).float()
            dir_acc = ((pred_dir == tgt_dir) & src_ok).sum().float() / n_src_ok

            # Full move: src + dir + split correct AND didn't pass (move timesteps only)
            full_match = src_ok & (pred_dir == tgt_dir) & (pred_split == tgt_split) & ~pred_pass
            full_acc = full_match.sum().float() / n_valid

            # Complete: correct on ALL timesteps (moves + passes)
            pass_ok = pred_pass & (~move_mask) & seq_mask    # correctly passed
            complete_ok = full_match | pass_ok               # move correct OR pass correct
            complete_acc = complete_ok.sum().float() / n_seq

        return {
            "loss": total,
            "source_loss": source_loss.detach(),
            "direction_loss": dir_loss.detach(),
            "split_loss": split_loss.detach(),
            "pass_loss": pass_loss.detach(),
            "value_loss": value_loss.detach(),
            "q_halt_loss": q_halt_loss.detach() if isinstance(q_halt_loss, torch.Tensor) else q_halt_loss,
            "src_acc": src_acc,
            "dir_acc": dir_acc,
            "full_acc": full_acc,
            "complete_acc": complete_acc,
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
