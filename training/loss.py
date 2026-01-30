"""
Loss functions for training.

Includes:
- TRM-style halting loss from "Less is More" paper (arXiv:2510.04871)
- Illegal action penalty for generalization (no masking approach)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from data_processing.processing.game_env import is_legal_batch


def compute_illegal_action_penalty(
    move_id: torch.Tensor,
    source_idx: torch.Tensor,
    dir_idx: torch.Tensor,
    grid: torch.Tensor,
    grid_height: int = 25,
    grid_width: int = 25,
    penalty_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute penalty for illegal actions (for generalization without masking).

    Uses is_legal_batch() from game_env to check validity based on grid state.
    Only applies when move_id == 1 (move action) and action is invalid.

    Args:
        move_id: Selected move (0=pass, 1=move) [B] or [B, T]
        source_idx: Selected source positions [B] or [B, T]
        dir_idx: Selected directions [B] or [B, T]
        grid: Grid state [B, C, H, W] or [B, T, C, H, W]
        grid_height: Grid height
        grid_width: Grid width
        penalty_weight: Weight for illegal action penalty

    Returns:
        Penalty tensor [B] or [B, T]
    """
    original_shape = move_id.shape

    # Use is_legal_batch to check validity based on grid
    # Note: is_legal_batch handles both [B, C, H, W] and [B, T, C, H, W] formats
    is_legal = is_legal_batch(
        grid=grid,
        source_idx=source_idx,
        dir_idx=dir_idx,
        ratio_idx=torch.zeros_like(source_idx),  # ratio not checked
        grid_height=grid_height,
        grid_width=grid_width
    )

    # Action is invalid if move=1 AND action is not legal
    is_move = (move_id == 1)
    is_invalid = is_move & ~is_legal

    # Penalty: 1.0 for invalid actions, 0.0 for valid
    penalty = is_invalid.float() * penalty_weight

    return penalty


class ImitationLoss(nn.Module):
    """
    Imitation learning loss for supervised pre-training on replay data.

    Computes:
    - Cross-entropy loss between model predictions and expert actions
    - MSE loss for value prediction
    - TRM halting loss: BCE(halt_prob, is_correct)
    - Illegal action penalty (for no-masking generalization approach)
    """

    def __init__(
        self,
        value_weight: float = 0.5,
        halting_weight: float = 0.1,
        illegal_penalty_weight: float = 1.0  # Penalty for selecting illegal actions
    ):
        super().__init__()
        self.value_weight = value_weight
        self.halting_weight = halting_weight
        self.illegal_penalty_weight = illegal_penalty_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        grid: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute imitation learning loss with TRM halting loss and illegal action penalty.

        Args:
            outputs: Model outputs with keys:
                - move_logits: [B, T, 2]
                - source_logits: [B, T, N]
                - dir_logits: [B, T, 4]
                - ratio_logits: [B, T, 2]
                - value: [B, T]
                - halt_probs: [B, T, max_cycles] - halting probabilities
                - step_source_preds: [B, T, n_cycles] - predictions at each supervision step
            targets: Expert action targets with keys:
                - move_id: [B, T] - ground truth move IDs (0=pass, 1=move)
                - source_idx: [B, T] - ground truth source indices
                - dir_id: [B, T] - ground truth direction IDs
                - ratio_id: [B, T] - ground truth ratio IDs
                - value: [B, T] or [B] - game outcome
                - action_mask: [B, T] - which timesteps have valid actions
            mask: Optional sequence mask [B, T]
            grid: Grid state [B, T, C, H, W] - for illegal action penalty (uses is_legal_batch)

        Returns:
            Loss dictionary with 'total', 'policy', 'value', 'halting', 'illegal_penalty' keys
        """
        B, T = outputs['source_logits'].shape[:2]
        device = outputs['source_logits'].device

        action_mask = targets.get('action_mask', torch.ones(B, T, device=device).bool())
        if mask is None:
            mask = torch.ones(B, T, device=device).bool()
        valid_mask = action_mask & mask

        # Replace -inf with large negative value to avoid NaN in cross_entropy
        source_logits = outputs['source_logits'].clone()
        source_logits = torch.where(
            torch.isinf(source_logits),
            torch.full_like(source_logits, -100.0),
            source_logits
        )

        dir_logits = outputs['dir_logits'].clone()
        dir_logits = torch.where(
            torch.isinf(dir_logits),
            torch.full_like(dir_logits, -100.0),
            dir_logits
        )

        # Policy losses
        # Move loss (0=pass, 1=move)
        move_target = targets.get('move_id', torch.ones(B, T, dtype=torch.long, device=device))
        move_loss = F.cross_entropy(
            outputs['move_logits'].view(-1, 2),
            move_target.view(-1),
            reduction='none'
        ).view(B, T)

        source_loss = F.cross_entropy(
            source_logits.view(-1, source_logits.size(-1)),
            targets['source_idx'].view(-1),
            reduction='none'
        ).view(B, T)

        dir_loss = F.cross_entropy(
            dir_logits.view(-1, 4),
            targets['dir_id'].view(-1),
            reduction='none'
        ).view(B, T)

        ratio_loss = F.cross_entropy(
            outputs['ratio_logits'].view(-1, 2),
            targets['ratio_id'].view(-1),
            reduction='none'
        ).view(B, T)

        # Total policy loss includes move_loss
        # Note: source/dir/ratio losses only apply when move_id == 1
        # For simplicity, we always compute them but they will be masked by action_mask
        policy_loss = (move_loss + source_loss + dir_loss + ratio_loss) * valid_mask.float()
        policy_loss = policy_loss.sum() / valid_mask.sum().clamp(min=1)

        # Value loss
        value_target = targets['value']
        if value_target.dim() == 1:
            value_target = value_target.unsqueeze(1).expand(-1, T)

        value_loss = F.mse_loss(
            outputs['value'], value_target, reduction='none'
        )
        value_loss = (value_loss * mask.float()).sum() / mask.sum().clamp(min=1)

        # TRM Halting loss: BCE(halt_prob, is_correct)
        # The model learns to predict when its answer is correct
        halting_loss = torch.tensor(0.0, device=device)

        # FIX: Key name was 'cycle_source_preds' but model outputs 'step_source_preds'
        if 'step_source_preds' in outputs and isinstance(outputs['step_source_preds'], torch.Tensor):
            step_preds = outputs['step_source_preds']  # [B, T, n_supervision_steps]
            halt_probs = outputs['halt_probs']  # [B, T, n_supervision_steps]

            # Get target source indices
            target_source = targets['source_idx']  # [B, T]

            # Compute is_correct for each supervision step: 1 if prediction matches target
            # [B, T, n_steps] == [B, T, 1] -> [B, T, n_steps]
            is_correct = (step_preds == target_source.unsqueeze(-1)).float()

            # Match dimensions (step_preds should match halt_probs)
            n_steps = step_preds.shape[-1]
            halt_probs_matched = halt_probs[:, :, :n_steps]  # [B, T, n_steps]

            # BCE loss: model should output high halt_prob when prediction is correct
            halting_loss = F.binary_cross_entropy(
                halt_probs_matched,
                is_correct,
                reduction='none'
            )

            # Apply mask and average
            # Expand valid_mask to [B, T, n_steps]
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(halting_loss)
            halting_loss = (halting_loss * valid_mask_expanded.float()).sum()
            halting_loss = halting_loss / (valid_mask_expanded.sum().clamp(min=1))

        # Illegal action penalty (for no-masking generalization approach)
        # Penalizes the model when it predicts move=1 but selects invalid source/direction
        # Uses is_legal_batch() to check validity based on grid state only
        illegal_penalty = torch.tensor(0.0, device=device)

        if grid is not None and self.illegal_penalty_weight > 0:
            # Get model's predicted actions (argmax)
            pred_source_idx = outputs['source_logits'].argmax(dim=-1)  # [B, T]
            pred_dir_idx = outputs['dir_logits'].argmax(dim=-1)  # [B, T]

            # For imitation learning, we also check if the model would select move=1
            # If move_logits available, use them; otherwise assume move=1 for all
            if 'move_logits' in outputs:
                pred_move_id = outputs['move_logits'].argmax(dim=-1)  # [B, T]
            else:
                pred_move_id = torch.ones(B, T, dtype=torch.long, device=device)

            # Compute penalty using is_legal_batch (no legal_mask needed)
            penalty = compute_illegal_action_penalty(
                move_id=pred_move_id,
                source_idx=pred_source_idx,
                dir_idx=pred_dir_idx,
                grid=grid,
                penalty_weight=self.illegal_penalty_weight
            )

            # Apply mask and average
            illegal_penalty = (penalty * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)

        total_loss = (
            policy_loss +
            self.value_weight * value_loss +
            self.halting_weight * halting_loss +
            illegal_penalty
        )

        return {
            'total': total_loss,
            'policy': policy_loss,
            'value': value_loss,
            'halting': halting_loss,
            'illegal_penalty': illegal_penalty
        }
