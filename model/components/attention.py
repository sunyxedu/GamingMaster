"""
TRM-style Multi-Head Self-Attention.

No bias terms, following TRM paper design.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TRMAttention(nn.Module):
    """
    Multi-Head Self-Attention without bias terms.

    TRM-style attention:
    - No bias in Q, K, V, and output projections
    - Standard scaled dot-product attention
    - Optional causal masking for temporal sequences

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability (default 0.0 per TRM)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Combined QKV projection (no bias per TRM)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection (no bias per TRM)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape [B, N, D]
            attention_mask: Optional mask of shape [B, N] or [B, N, N]
                True/1 = attend, False/0 = mask
            causal: If True, apply causal (lower triangular) mask

        Returns:
            Output tensor of shape [B, N, D]
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H, N, head_dim]

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale  # [B, H, N, N]

        # Apply causal mask if requested
        if causal:
            causal_mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, N] -> [B, 1, 1, N]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # [B, N, N] -> [B, 1, N, N]
                attention_mask = attention_mask.unsqueeze(1)

            attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        out = attn_weights @ v  # [B, H, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]

        return self.out_proj(out)

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, n_heads={self.n_heads}, head_dim={self.head_dim}'
