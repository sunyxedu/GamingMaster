"""
Policy Heads for generals.io action prediction.

Implements separate heads for:
- Move decision (whether to move or pass)
- Source selection (which tile to move from)
- Direction selection (which direction to move)
- Ratio selection (all-but-1 vs 50% split)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .components import RMSNorm


class MoveHead(nn.Module):
    """
    Move decision head.

    Predicts whether to move (1) or pass (0).
    Uses global features aggregated from all tokens.

    Args:
        d_model: Input dimension (default 512)
    """

    def __init__(self, d_model: int = 512):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_model, 128, bias=False),
            RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, 2, bias=False)  # 0=pass, 1=move
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute move/pass logits.

        Args:
            z: Token representations [B, N, D]

        Returns:
            Move logits [B, 2] where index 0=pass, 1=move
        """
        # Global pooling: mean over all tokens
        z_global = z.mean(dim=1)  # [B, D]

        # Compute logits
        logits = self.proj(z_global)  # [B, 2]

        return logits


class SourceHead(nn.Module):
    """
    Source position selection head.

    Predicts logits for each grid position (25x25 = 625 positions).
    NO masking applied - model must learn valid positions through loss penalty.

    Args:
        d_model: Input dimension (default 512)
    """

    def __init__(self, d_model: int = 512):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_model, 256, bias=False),
            RMSNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1, bias=False)
        )

    def forward(
        self,
        z: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None  # Kept for API compatibility but unused
    ) -> torch.Tensor:
        """
        Compute source position logits.

        Args:
            z: Token representations [B, N, D] where N = 625
            legal_mask: UNUSED - kept for API compatibility

        Returns:
            Source logits [B, N] - NO masking, model learns valid positions
        """
        # Project each token to a single logit
        logits = self.proj(z).squeeze(-1)  # [B, N]

        # NO masking - model must learn which positions are valid
        return logits


class DirectionHead(nn.Module):
    """
    Direction selection head.

    Predicts logits for 4 directions (down, right, up, left).
    Conditioned on the selected source position.
    NO masking applied - model must learn valid directions through loss penalty.

    Direction encoding:
        0: down  (row + 1)
        1: right (col + 1)
        2: up    (row - 1)
        3: left  (col - 1)

    Args:
        d_model: Input dimension (default 512)
        grid_height: Grid height (default 25)
        grid_width: Grid width (default 25)
    """

    def __init__(
        self,
        d_model: int = 512,
        grid_height: int = 25,
        grid_width: int = 25
    ):
        super().__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width

        # Direction offsets: [dy, dx] for each direction (kept for validity checking)
        self.register_buffer('dir_offsets', torch.tensor([
            [1, 0],   # down
            [0, 1],   # right
            [-1, 0],  # up
            [0, -1]   # left
        ]))

        self.proj = nn.Sequential(
            nn.Linear(d_model, 128, bias=False),
            RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, 4, bias=False)
        )

    def forward(
        self,
        z: torch.Tensor,
        source_idx: torch.Tensor,
        grid: Optional[torch.Tensor] = None  # Kept for API compatibility but unused for masking
    ) -> torch.Tensor:
        """
        Compute direction logits.

        Args:
            z: Token representations [B, N, D]
            source_idx: Selected source indices [B]
            grid: UNUSED for masking - kept for API compatibility

        Returns:
            Direction logits [B, 4] - NO masking, model learns valid directions
        """
        B = z.shape[0]
        device = z.device

        # Extract features at source positions
        batch_indices = torch.arange(B, device=device)
        source_features = z[batch_indices, source_idx.long()]  # [B, D]

        # Compute direction logits
        logits = self.proj(source_features)  # [B, 4]

        # NO masking - model must learn which directions are valid
        return logits

    def check_direction_validity(
        self,
        source_idx: torch.Tensor,
        dir_idx: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Check if selected directions are valid (for loss penalty computation).

        Args:
            source_idx: Source positions [B]
            dir_idx: Selected directions [B]
            grid: Optional grid [B, C, H, W] for obstacle checking

        Returns:
            Validity mask [B] (True = valid, False = invalid)
        """
        B = source_idx.shape[0]
        device = source_idx.device

        # Convert flat index to 2D coordinates
        source_y = source_idx // self.grid_width  # [B]
        source_x = source_idx % self.grid_width   # [B]

        # Get direction offsets for selected directions
        dy = self.dir_offsets[dir_idx, 0]  # [B]
        dx = self.dir_offsets[dir_idx, 1]  # [B]

        # Compute destinations
        dest_y = source_y + dy
        dest_x = source_x + dx

        # Check bounds
        out_of_bounds = (
            (dest_y < 0) | (dest_y >= self.grid_height) |
            (dest_x < 0) | (dest_x >= self.grid_width)
        )

        valid = ~out_of_bounds

        # Check mountains if grid provided
        if grid is not None and valid.any():
            mountain_channel = 4
            # For valid destinations, check if mountain
            for i in range(B):
                if valid[i]:
                    dy_i = dest_y[i].item()
                    dx_i = dest_x[i].item()
                    if grid[i, mountain_channel, dy_i, dx_i] > 0.5:
                        valid[i] = False

        return valid


class RatioHead(nn.Module):
    """
    Ratio selection head.

    Predicts logits for 2 ratio options:
        0: all-but-1 (move all troops except 1)
        1: 50% (move half the troops)

    Args:
        d_model: Input dimension (default 512)
    """

    def __init__(self, d_model: int = 512):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_model, 64, bias=False),
            RMSNorm(64),
            nn.SiLU(),
            nn.Linear(64, 2, bias=False)
        )

    def forward(
        self,
        z: torch.Tensor,
        source_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ratio logits.

        Args:
            z: Token representations [B, N, D]
            source_idx: Selected source indices [B]

        Returns:
            Ratio logits [B, 2]
        """
        B = z.shape[0]
        device = z.device

        # Extract features at source positions
        batch_indices = torch.arange(B, device=device)
        source_features = z[batch_indices, source_idx.long()]  # [B, D]

        # Compute ratio logits
        logits = self.proj(source_features)  # [B, 2]

        return logits


class CombinedPolicyHead(nn.Module):
    """
    Combined policy head with all four action components.

    NO masking applied - model must learn valid actions through loss penalty.
    This enables better generalization.

    Provides unified interface for:
    - Move decision (whether to move or pass)
    - Source selection (which tile to move from)
    - Direction selection (conditioned on source)
    - Ratio selection (conditioned on source)

    Args:
        d_model: Input dimension (default 512)
        grid_height: Grid height (default 25)
        grid_width: Grid width (default 25)
    """

    def __init__(
        self,
        d_model: int = 512,
        grid_height: int = 25,
        grid_width: int = 25
    ):
        super().__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width

        self.move_head = MoveHead(d_model)
        self.source_head = SourceHead(d_model)
        self.direction_head = DirectionHead(d_model, grid_height, grid_width)
        self.ratio_head = RatioHead(d_model)

    def forward(
        self,
        z: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,  # UNUSED - kept for API compatibility
        grid: Optional[torch.Tensor] = None,
        sample: bool = False,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full policy distribution. NO masking applied.

        Args:
            z: Token representations [B, N, D]
            legal_mask: UNUSED - kept for API compatibility
            grid: Optional grid [B, C, H, W] - only used for validity checking, not masking
            sample: If True, sample actions; else use argmax
            temperature: Sampling temperature

        Returns:
            Dictionary with all logits, probs, and selected actions
        """
        # Move decision
        move_logits = self.move_head(z)  # [B, 2]
        move_probs = F.softmax(move_logits / temperature, dim=-1)

        if sample:
            move_idx = torch.multinomial(move_probs, 1).squeeze(-1)
        else:
            move_idx = move_logits.argmax(dim=-1)

        # Source selection - NO masking
        source_logits = self.source_head(z)  # [B, N]
        source_probs = F.softmax(source_logits / temperature, dim=-1)

        if sample:
            source_idx = torch.multinomial(source_probs, 1).squeeze(-1)
        else:
            source_idx = source_logits.argmax(dim=-1)

        # Direction selection - NO masking
        dir_logits = self.direction_head(z, source_idx, grid)  # [B, 4]
        dir_probs = F.softmax(dir_logits / temperature, dim=-1)

        if sample:
            dir_idx = torch.multinomial(dir_probs, 1).squeeze(-1)
        else:
            dir_idx = dir_logits.argmax(dim=-1)

        # Ratio selection
        ratio_logits = self.ratio_head(z, source_idx)  # [B, 2]
        ratio_probs = F.softmax(ratio_logits / temperature, dim=-1)

        if sample:
            ratio_idx = torch.multinomial(ratio_probs, 1).squeeze(-1)
        else:
            ratio_idx = ratio_logits.argmax(dim=-1)

        # Convert source index to y, x coordinates
        source_y = source_idx // self.grid_width
        source_x = source_idx % self.grid_width

        return {
            'move_logits': move_logits,
            'move_probs': move_probs,
            'source_logits': source_logits,
            'source_probs': source_probs,
            'dir_logits': dir_logits,
            'dir_probs': dir_probs,
            'ratio_logits': ratio_logits,
            'ratio_probs': ratio_probs,
            'action': {
                'move_id': move_idx,
                'source_idx': source_idx,
                'source_y': source_y,
                'source_x': source_x,
                'dir_id': dir_idx,
                'ratio_id': ratio_idx
            }
        }

    def forward_training(
        self,
        z: torch.Tensor,
        target_source_idx: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with given target source. NO masking applied.

        Args:
            z: Token representations [B, N, D]
            target_source_idx: Ground truth source indices [B]
            grid: Optional grid [B, C, H, W]

        Returns:
            Dictionary with all logits and probabilities
        """
        # Move logits
        move_logits = self.move_head(z)
        move_probs = F.softmax(move_logits, dim=-1)

        # Source logits - NO masking
        source_logits = self.source_head(z)
        source_probs = F.softmax(source_logits, dim=-1)

        # Direction logits using target source
        dir_logits = self.direction_head(z, target_source_idx, grid)
        dir_probs = F.softmax(dir_logits, dim=-1)

        # Ratio logits using target source
        ratio_logits = self.ratio_head(z, target_source_idx)
        ratio_probs = F.softmax(ratio_logits, dim=-1)

        return {
            'move_logits': move_logits,
            'move_probs': move_probs,
            'source_logits': source_logits,
            'source_probs': source_probs,
            'dir_logits': dir_logits,
            'dir_probs': dir_probs,
            'ratio_logits': ratio_logits,
            'ratio_probs': ratio_probs
        }

    def check_action_validity(
        self,
        source_idx: torch.Tensor,
        dir_idx: torch.Tensor,
        legal_mask: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Check if selected actions are valid (for loss penalty computation).

        Args:
            source_idx: Selected source positions [B]
            dir_idx: Selected directions [B]
            legal_mask: Legal source mask [B, 1, H, W] or [B, N]
            grid: Optional grid [B, C, H, W] for obstacle checking

        Returns:
            Dictionary with:
                - source_valid: [B] bool - whether source is legal
                - dir_valid: [B] bool - whether direction is valid
                - action_valid: [B] bool - overall validity
        """
        B = source_idx.shape[0]
        device = source_idx.device

        # Check source validity
        if legal_mask.dim() == 4:
            legal_mask_flat = legal_mask.view(B, -1)  # [B, N]
        else:
            legal_mask_flat = legal_mask

        # Gather validity at selected source positions
        source_valid = legal_mask_flat.gather(1, source_idx.unsqueeze(1)).squeeze(1)  # [B]

        # Check direction validity
        dir_valid = self.direction_head.check_direction_validity(source_idx, dir_idx, grid)  # [B]

        # Overall validity: both source and direction must be valid
        action_valid = source_valid & dir_valid

        return {
            'source_valid': source_valid,
            'dir_valid': dir_valid,
            'action_valid': action_valid
        }
