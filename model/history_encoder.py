"""
History Encoder for compressing temporal sequences into a latent state.

This module encodes the history of game states into a fixed-size latent
representation that captures temporal patterns. During training, it sees
the full sequence; during inference, it can be updated incrementally.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .components import RMSNorm, SwiGLUFFN, TRMAttention


class SpatialEncoder(nn.Module):
    """
    Encodes a single spatial frame into a vector representation.

    Uses convolutional layers to compress the 25x25 grid into a feature vector.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_dim: int = 256
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            # [B, 12, 25, 25] -> [B, 64, 25, 25]
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            # [B, 64, 25, 25] -> [B, 128, 13, 13]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            # [B, 128, 13, 13] -> [B, 256, 7, 7]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),

            # Global average pooling -> [B, 256]
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # Project to output dim
            nn.Linear(256, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial frame.

        Args:
            x: Grid tensor of shape [B, C, H, W]

        Returns:
            Feature vector of shape [B, out_dim]
        """
        return self.encoder(x)


class TemporalTransformer(nn.Module):
    """
    Causal Transformer for processing temporal sequences.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 1024,
        max_seq_len: int = 512
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Learnable position embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process temporal sequence with causal attention.

        Args:
            x: Sequence tensor of shape [B, T, D]
            attention_mask: Optional mask of shape [B, T]

        Returns:
            Output tensor of shape [B, T, D]
        """
        B, T, D = x.shape

        # Add positional embeddings
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embedding(positions)

        # Apply transformer layers with causal masking
        for layer in self.layers:
            x = layer(x, attention_mask, causal=True)

        return self.norm(x)


class TransformerLayer(nn.Module):
    """Single Transformer layer with attention and FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()

        self.attention = TRMAttention(d_model, n_heads)
        self.ffn = SwiGLUFFN(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False
    ) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x), attention_mask, causal)
        x = x + self.ffn(self.norm2(x))
        return x


class HistoryEncoder(nn.Module):
    """
    Encodes history sequence into a fixed-size latent state.

    During training:
        - Receives full sequence [B, T, C, H, W]
        - Outputs latent state [B, D_latent]

    During inference:
        - Can update incrementally using update_latent()

    Args:
        grid_channels: Number of input channels (default 12)
        global_features: Number of global features (default 4)
        d_model: Internal dimension (default 256)
        n_heads: Number of attention heads (default 4)
        n_layers: Number of transformer layers (default 2)
        max_seq_len: Maximum sequence length (default 512)
    """

    def __init__(
        self,
        grid_channels: int = 12,
        global_features: int = 4,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 512
    ):
        super().__init__()

        self.d_model = d_model

        # Spatial encoder for each frame
        self.spatial_encoder = SpatialEncoder(grid_channels, d_model)

        # Global features projection
        self.global_proj = nn.Linear(global_features, d_model, bias=False)

        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_model * 4,
            max_seq_len=max_seq_len
        )

        # GRU-style recurrent update for inference
        self.gate_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.update_proj = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(
        self,
        grid_seq: torch.Tensor,
        globals_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode full history sequence.

        Args:
            grid_seq: Grid sequence [B, T, C, H, W]
            globals_seq: Global features sequence [B, T, G]
            attention_mask: Optional mask [B, T]

        Returns:
            Latent state [B, D]
        """
        B, T, C, H, W = grid_seq.shape

        # Encode each frame spatially
        # Reshape to process all frames at once
        grid_flat = grid_seq.reshape(B * T, C, H, W)
        spatial_features = self.spatial_encoder(grid_flat)  # [B*T, D]
        spatial_features = spatial_features.reshape(B, T, -1)  # [B, T, D]

        # Add global features
        global_features = self.global_proj(globals_seq)  # [B, T, D]
        frame_features = spatial_features + global_features  # [B, T, D]

        # Process through temporal transformer
        temporal_output = self.temporal_transformer(frame_features, attention_mask)

        # Take the last hidden state as latent
        latent = temporal_output[:, -1, :]  # [B, D]

        return latent

    def update_latent(
        self,
        latent_prev: torch.Tensor,
        current_grid: torch.Tensor,
        current_globals: torch.Tensor
    ) -> torch.Tensor:
        """
        Update latent state with new observation (for inference).

        Uses GRU-style gating for incremental updates.

        Args:
            latent_prev: Previous latent state [B, D]
            current_grid: Current frame [B, C, H, W]
            current_globals: Current global features [B, G]

        Returns:
            Updated latent state [B, D]
        """
        # Encode current frame
        spatial_emb = self.spatial_encoder(current_grid)  # [B, D]
        global_emb = self.global_proj(current_globals)  # [B, D]
        frame_emb = spatial_emb + global_emb  # [B, D]

        # GRU-style update
        combined = torch.cat([latent_prev, frame_emb], dim=-1)  # [B, 2D]

        gate = torch.sigmoid(self.gate_proj(combined))  # [B, D]
        update = torch.tanh(self.update_proj(combined))  # [B, D]

        # New latent: gate * prev + (1 - gate) * update
        latent_new = gate * latent_prev + (1 - gate) * update

        return latent_new

    def get_initial_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get initial latent state (zeros).

        Args:
            batch_size: Batch size
            device: Target device

        Returns:
            Zero latent state [B, D]
        """
        return torch.zeros(batch_size, self.d_model, device=device)
