"""Policy heads (autoregressive: Source → Direction / Split) and Value head."""

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

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_tokens: [B, N, d]
        Returns:
            logits: [B, N]  (raw, no masking — mask applied externally during inference)
        """
        return self.proj(spatial_tokens).squeeze(-1)  # [B, N]


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
    """Predict split ratio (0=all-but-1, 1=50%) conditioned on source embedding."""

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


class PassHead(nn.Module):
    """Decide whether to pass (no move) given action context, proposed move, and timing.

    Inputs: ll_token, source_embed, direction_logits, split_logits, time_pe.
    Output: single logit (>0 → pass, ≤0 → execute move).
    """

    def __init__(self, d_model: int, n_directions: int = 4, n_split: int = 2,
                 time_pe_dim: int = 32):
        super().__init__()
        self.time_pe_dim = time_pe_dim
        # ll(d) + source_embed(d) + direction(n_dir) + split(n_split) + time_pe
        in_dim = d_model * 2 + n_directions + n_split + time_pe_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        ll_token: torch.Tensor,
        source_embed: torch.Tensor,
        direction_logits: torch.Tensor,
        split_logits: torch.Tensor,
        time_pe: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([ll_token, source_embed, direction_logits,
                              split_logits, time_pe], dim=-1)
        return self.net(combined).squeeze(-1)


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
