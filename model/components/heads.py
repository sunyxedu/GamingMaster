"""Policy heads (autoregressive: Source → Direction / Split) and Value head."""

from typing import Optional

import torch
import torch.nn as nn


class SourceHead(nn.Module):
    """Predict which tile to select as movement source.  [B, H*W] logits."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_tokens: [B, N, d]
            legal_mask:     [B, N]  bool – True where move is legal
        Returns:
            logits: [B, N]
        """
        logits = self.proj(spatial_tokens).squeeze(-1)  # [B, N]
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return logits


class DirectionHead(nn.Module):
    """Predict movement direction conditioned on source embedding."""

    def __init__(self, d_model: int, n_dirs: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_dirs),
        )

    def forward(
        self, source_embed: torch.Tensor, decision_token: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([source_embed, decision_token], dim=-1)
        return self.net(combined)


class SplitHead(nn.Module):
    """Predict split ratio (all-but-1 vs 50%) conditioned on source embedding."""

    def __init__(self, d_model: int, n_split: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_split),
        )

    def forward(
        self, source_embed: torch.Tensor, decision_token: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([source_embed, decision_token], dim=-1)
        return self.net(combined)


class ValueHead(nn.Module):
    """Predict game value from the high-level decision token."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def forward(self, hl_token: torch.Tensor) -> torch.Tensor:
        return self.net(hl_token).squeeze(-1)  # [B]
