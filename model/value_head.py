"""
Value Head for predicting game outcome.

Predicts win probability in range [-1, 1]:
    -1 = certain loss
     0 = even/draw
    +1 = certain win
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .components import RMSNorm


class ValueHead(nn.Module):
    """
    Value prediction head.

    Architecture:
    1. Spatial pooling of token representations
    2. Global features integration
    3. MLP to single value output
    4. Tanh activation to bound in [-1, 1]

    Args:
        d_model: Token dimension (default 512)
        d_global: Global features dimension (default 4)
    """

    def __init__(
        self,
        d_model: int = 512,
        d_global: int = 4
    ):
        super().__init__()

        # Spatial feature processing
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_model, 256, bias=False),
            RMSNorm(256),
            nn.SiLU()
        )

        # Global features processing
        self.global_proj = nn.Linear(d_global, 64, bias=False)

        # Value MLP
        self.value_mlp = nn.Sequential(
            nn.Linear(256 + 64, 128, bias=False),
            RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64, bias=False),
            RMSNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(
        self,
        z: torch.Tensor,
        globals_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value estimate.

        Args:
            z: Token representations [B, N, D]
            globals_vec: Global features [B, G]

        Returns:
            Value estimate [B] in range [-1, 1]
        """
        # Spatial aggregation via mean pooling
        z_pooled = z.mean(dim=1)  # [B, D]
        spatial_features = self.spatial_proj(z_pooled)  # [B, 256]

        # Global features
        global_features = self.global_proj(globals_vec)  # [B, 64]

        # Combine and predict value
        combined = torch.cat([spatial_features, global_features], dim=-1)
        value = self.value_mlp(combined).squeeze(-1)  # [B]

        # Bound to [-1, 1]
        value = torch.tanh(value)

        return value


class ValueHeadWithUncertainty(nn.Module):
    """
    Value head that also predicts uncertainty.

    Outputs both value prediction and a confidence score.
    Useful for exploration.

    Args:
        d_model: Token dimension (default 512)
        d_global: Global features dimension (default 4)
    """

    def __init__(
        self,
        d_model: int = 512,
        d_global: int = 4
    ):
        super().__init__()

        # Spatial feature processing
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_model, 256, bias=False),
            RMSNorm(256),
            nn.SiLU()
        )

        # Global features processing
        self.global_proj = nn.Linear(d_global, 64, bias=False)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(256 + 64, 128, bias=False),
            RMSNorm(128),
            nn.SiLU()
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            RMSNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1, bias=False)
        )

        # Uncertainty head (log variance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 32, bias=False),
            RMSNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1, bias=False)
        )

    def forward(
        self,
        z: torch.Tensor,
        globals_vec: torch.Tensor
    ) -> tuple:
        """
        Compute value estimate with uncertainty.

        Args:
            z: Token representations [B, N, D]
            globals_vec: Global features [B, G]

        Returns:
            value: Value estimate [B] in range [-1, 1]
            uncertainty: Uncertainty estimate [B] (positive)
        """
        # Spatial aggregation
        z_pooled = z.mean(dim=1)
        spatial_features = self.spatial_proj(z_pooled)

        # Global features
        global_features = self.global_proj(globals_vec)

        # Trunk
        combined = torch.cat([spatial_features, global_features], dim=-1)
        trunk_out = self.trunk(combined)

        # Value prediction
        value = self.value_head(trunk_out).squeeze(-1)
        value = torch.tanh(value)

        # Uncertainty (softplus to ensure positive)
        log_var = self.uncertainty_head(trunk_out).squeeze(-1)
        uncertainty = F.softplus(log_var)

        return value, uncertainty
