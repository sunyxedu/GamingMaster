"""Spatial encoder (Create Embeddings) and History encoder (GRU)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.config import ModelConfig


class ResBlock(nn.Module):
    """Pre-activation residual block for the spatial encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SpatialEncoder(nn.Module):
    """
    Create Embeddings — converts raw grid channels into per-cell feature vectors.

    Input : [B, C_in, H, W]  (C_in = 12 = 11 type one-hot + 1 army)
    Output: [B, d_model, H, W]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c_mid = cfg.encoder_channels
        self.net = nn.Sequential(
            nn.Conv2d(cfg.grid_channels, c_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            ResBlock(c_mid),
            nn.Conv2d(c_mid, cfg.d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.d_model),
            nn.ReLU(inplace=True),
            ResBlock(cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HistoryEncoder(nn.Module):
    """
    GRU-based temporal encoder.

    Receives the spatially-pooled embedding at each timestep and produces
    a temporal-context vector that is broadcast-added back into the spatial
    feature map.

    Input  : [B, T, d_model]   (pooled spatial)
    Output : [B, T, d_model],  hidden state
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gru = nn.GRU(
            cfg.d_model, cfg.d_model,
            num_layers=cfg.gru_layers,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.gru(x, hidden)
        return out, hidden
