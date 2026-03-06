"""
Transformer blocks.

- TransformerBlock        : spatial-only (shared-weight pondering, ×k)
- SpatiotemporalBlock     : factored spatial + causal temporal attention
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig

# ---------------------------------------------------------------------------
# Spatial-only block (original pondering design)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Self-Attention → Add & Norm → MLP → Add & Norm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Spatiotemporal factored block
# ---------------------------------------------------------------------------

def _sinusoidal_pe(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal positional encoding [length, d_model]."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe


class SpatiotemporalBlock(nn.Module):
    """
    Factored spatiotemporal attention block.

        Spatial Self-Attention   (within each timestep, 627 tokens)
      → Add & Norm
      → Temporal Self-Attention  (across timesteps, per token position, causal)
      → Add & Norm
      → MLP
      → Add & Norm
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(cfg.d_model)

        self.temporal_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(cfg.d_model)

        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.ff_norm = nn.LayerNorm(cfg.d_model)

    def _causal_temporal_attn(self, x: torch.Tensor) -> torch.Tensor:
        """Temporal self-attention with causal masking via F.scaled_dot_product_attention.

        Bypasses nn.MultiheadAttention.forward() but reuses its parameters,
        so existing checkpoints load without any changes.
        """
        B, T, d = x.shape
        n_heads = self.temporal_attn.num_heads
        d_head = d // n_heads

        # Project Q, K, V using temporal_attn's weights
        qkv = F.linear(x, self.temporal_attn.in_proj_weight, self.temporal_attn.in_proj_bias)
        qkv = qkv.reshape(B, T, 3, n_heads, d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, n_heads, T, d_head]

        dropout_p = self.temporal_attn.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=True,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, d)
        return self.temporal_attn.out_proj(attn_out)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x : [B, T, S, d]   S = N+2 (spatial tokens + decision tokens)
        Returns:
            [B, T, S, d]
        """
        B, T, S, d = x.shape

        # ── spatial self-attention (within each timestep) ──
        x_flat = x.reshape(B * T, S, d)
        sa_out, _ = self.spatial_attn(x_flat, x_flat, x_flat, need_weights=False)
        x_flat = self.spatial_norm(x_flat + sa_out)
        x = x_flat.reshape(B, T, S, d)

        # ── temporal self-attention (across timesteps, per position, causal) ──
        x_perm = x.permute(0, 2, 1, 3).reshape(B * S, T, d)
        ta_out = self._causal_temporal_attn(x_perm)
        x_perm = self.temporal_norm(x_perm + ta_out)
        x = x_perm.reshape(B, S, T, d).permute(0, 2, 1, 3)

        # ── feed-forward ──
        x = self.ff_norm(x + self.ff(x))
        return x

    def forward_step(
        self,
        x_cur: torch.Tensor,
        temporal_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step inference with KV cache for temporal attention.

        Args:
            x_cur          : [B, 1, S, d]   current timestep only
            temporal_cache  : [B*S, t_prev, d] or None  (past inputs to temporal attn)
        Returns:
            output         : [B, 1, S, d]
            new_cache      : [B*S, t_prev+1, d]
        """
        B, _, S, d = x_cur.shape

        # ── spatial (only current timestep) ──
        x_flat = x_cur.reshape(B, S, d)
        sa_out, _ = self.spatial_attn(x_flat, x_flat, x_flat, need_weights=False)
        x_flat = self.spatial_norm(x_flat + sa_out)

        # ── temporal (query = current, key/value = cache + current) ──
        q = x_flat.reshape(B * S, 1, d)
        if temporal_cache is not None:
            kv = torch.cat([temporal_cache, q], dim=1)  # [B*S, t_prev+1, d]
        else:
            kv = q  # [B*S, 1, d]
        new_cache = kv

        ta_out, _ = self.temporal_attn(q, kv, kv, need_weights=False)
        q = self.temporal_norm(q + ta_out)

        x_out = q.reshape(B, S, d)

        # ── feed-forward ──
        x_out = self.ff_norm(x_out + self.ff(x_out))
        return x_out.unsqueeze(1), new_cache
