"""
Input Embedding module for the TRM model.

Combines current frame, global features, history latent, and positional
information into unified tokens for the Transformer backbone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .components import RMSNorm


class PositionalEncoding2D(nn.Module):
    """
    2D Sinusoidal Positional Encoding.

    Creates position embeddings for a 2D grid using sinusoidal functions.
    """

    def __init__(self, d_model: int, height: int = 25, width: int = 25):
        super().__init__()

        self.d_model = d_model
        self.height = height
        self.width = width

        # Create 2D position encoding
        pe = self._create_encoding(d_model, height, width)
        self.register_buffer('pe', pe)

    def _create_encoding(
        self,
        d_model: int,
        height: int,
        width: int
    ) -> torch.Tensor:
        """Create 2D sinusoidal position encoding."""
        # Half the dimensions for y, half for x
        d_half = d_model // 2

        # Create position indices
        y_pos = torch.arange(height).unsqueeze(1).expand(height, width)
        x_pos = torch.arange(width).unsqueeze(0).expand(height, width)

        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half)
        )

        # Compute sin/cos encodings for y
        pe_y = torch.zeros(height, width, d_half)
        pe_y[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe_y[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div_term)

        # Compute sin/cos encodings for x
        pe_x = torch.zeros(height, width, d_half)
        pe_x[:, :, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe_x[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)

        # Concatenate y and x encodings
        pe = torch.cat([pe_y, pe_x], dim=-1)  # [H, W, D]

        return pe.reshape(height * width, d_model)  # [N, D]

    def forward(self) -> torch.Tensor:
        """Return position encoding [N, D]."""
        return self.pe


class InputEmbedding(nn.Module):
    """
    Input Embedding for the TRM model.

    Combines:
    - Current frame grid (25x25, 12 channels)
    - Global features (4 dimensions)
    - History latent state (256 dimensions)
    - 2D positional encoding
    - Tick phase embedding

    Output: [B, N, D] where N = 625 (25x25), D = 512

    Args:
        grid_channels: Number of input grid channels (default 12)
        global_features: Number of global features (default 4)
        history_dim: Dimension of history latent (default 256)
        d_model: Output embedding dimension (default 512)
        height: Grid height (default 25)
        width: Grid width (default 25)
        n_tick_buckets: Number of tick phase buckets (default 26)
    """

    def __init__(
        self,
        grid_channels: int = 12,
        global_features: int = 4,
        history_dim: int = 256,
        d_model: int = 512,
        height: int = 25,
        width: int = 25,
        n_tick_buckets: int = 26
    ):
        super().__init__()

        self.d_model = d_model
        self.height = height
        self.width = width
        self.n_tokens = height * width

        # Grid embedding via convolutions
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(grid_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, d_model, kernel_size=1, bias=False),
        )

        # Global features projection (broadcast to all tokens)
        self.global_proj = nn.Linear(global_features, d_model, bias=False)

        # History latent projection (broadcast to all tokens)
        self.history_proj = nn.Linear(history_dim, d_model, bias=False)

        # 2D positional encoding (sinusoidal + learnable)
        self.pos_encoding = PositionalEncoding2D(d_model // 2, height, width)
        self.pos_learned = nn.Embedding(self.n_tokens, d_model // 2)

        # Tick phase embedding
        self.tick_embedding = nn.Embedding(n_tick_buckets, d_model)

        # Final normalization
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        grid: torch.Tensor,
        globals_vec: torch.Tensor,
        history_latent: torch.Tensor,
        tick_bucket: torch.Tensor
    ) -> torch.Tensor:
        """
        Create input embeddings for the Transformer backbone.

        Args:
            grid: Current frame [B, C, H, W]
            globals_vec: Global features [B, G]
            history_latent: History latent state [B, D_history]
            tick_bucket: Tick phase bucket [B] (values 0-25)

        Returns:
            Token embeddings [B, N, D] where N = H*W = 625
        """
        B = grid.shape[0]
        device = grid.device

        # Grid embedding: [B, C, H, W] -> [B, D, H, W] -> [B, N, D]
        grid_emb = self.grid_encoder(grid)  # [B, D, H, W]
        grid_emb = grid_emb.flatten(2).transpose(1, 2)  # [B, N, D]

        # Global features embedding: [B, G] -> [B, 1, D] -> broadcast to [B, N, D]
        global_emb = self.global_proj(globals_vec).unsqueeze(1)  # [B, 1, D]

        # History latent embedding: [B, D_h] -> [B, 1, D] -> broadcast to [B, N, D]
        history_emb = self.history_proj(history_latent).unsqueeze(1)  # [B, 1, D]

        # Positional encoding: sinusoidal + learned
        pos_sin = self.pos_encoding()  # [N, D/2]
        pos_indices = torch.arange(self.n_tokens, device=device)
        pos_learned = self.pos_learned(pos_indices)  # [N, D/2]
        pos_emb = torch.cat([pos_sin, pos_learned], dim=-1)  # [N, D]
        pos_emb = pos_emb.unsqueeze(0)  # [1, N, D]

        # Tick embedding: [B] -> [B, 1, D] -> broadcast to [B, N, D]
        tick_emb = self.tick_embedding(tick_bucket).unsqueeze(1)  # [B, 1, D]

        # Combine all embeddings
        tokens = grid_emb + global_emb + history_emb + pos_emb + tick_emb

        # Final normalization
        tokens = self.norm(tokens)

        return tokens  # [B, N, D]

    def forward_batch_timesteps(
        self,
        grid_seq: torch.Tensor,
        globals_seq: torch.Tensor,
        history_latent: torch.Tensor,
        tick_bucket_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Process a batch of timesteps efficiently.

        Args:
            grid_seq: Grid sequence [B, T, C, H, W]
            globals_seq: Global features sequence [B, T, G]
            history_latent: History latent (shared) [B, D_history]
            tick_bucket_seq: Tick buckets [B, T]

        Returns:
            Token embeddings [B, T, N, D]
        """
        B, T, C, H, W = grid_seq.shape

        # Process all frames at once
        grid_flat = grid_seq.reshape(B * T, C, H, W)

        # Temporarily expand history_latent for all timesteps
        history_expanded = history_latent.unsqueeze(1).expand(B, T, -1)
        history_flat = history_expanded.reshape(B * T, -1)

        globals_flat = globals_seq.reshape(B * T, -1)
        tick_flat = tick_bucket_seq.reshape(B * T)

        # Get embeddings for all frames
        tokens_flat = self.forward(
            grid_flat,
            globals_flat,
            history_flat,
            tick_flat
        )  # [B*T, N, D]

        # Reshape back
        tokens = tokens_flat.reshape(B, T, self.n_tokens, self.d_model)

        return tokens
