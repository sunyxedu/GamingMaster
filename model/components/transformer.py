"""
Transformer blocks.

- TransformerBlock        : spatial-only (shared-weight pondering, ×k)
- SpatiotemporalBlock     : factored spatial + causal temporal attention
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.core import QKNorm
from model.components.recurrence import GatedRecurrence
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

def _pe_from_positions(pos: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal encoding of float32 positions [L, 1] → [L, d_model]."""
    device = pos.device
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe = torch.zeros(pos.shape[0], d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe


def _sinusoidal_pe(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal positional encoding [length, d_model]."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    return _pe_from_positions(pos, d_model)


def sinusoidal_pe_rows(t0, length: int, d_model: int,
                       device: torch.device) -> torch.Tensor:
    """Rows t0 .. t0+length-1 of :func:`_sinusoidal_pe`: [length, d_model].

    ``t0`` is a Python int (the historical path, unchanged) or a
    single-element integer tensor on ``device``.  The tensor form keeps the
    absolute time a runtime VALUE instead of a trace-time constant, so a
    CUDA-graph-captured / compiled step can advance time without
    re-capturing, and it never reads the value on the host (no sync).
    Positions are exact integers in float32 (< 2^24), so both forms encode
    the same angles.
    """
    if not isinstance(t0, torch.Tensor):
        return _sinusoidal_pe(t0 + length, d_model, device)[t0: t0 + length]
    pos = (t0.reshape(()).to(device=device, dtype=torch.float32)
           + torch.arange(length, device=device, dtype=torch.float32)).unsqueeze(1)
    return _pe_from_positions(pos, d_model)


class SpatiotemporalBlock(nn.Module):
    """
    Factored spatiotemporal block.

        Spatial Self-Attention   (within each timestep, 627 tokens)
      → Add & Norm
      → Temporal mixer           (across timesteps, per token position)
      → Add & Norm
      → MLP
      → Add & Norm

    The temporal mixer is either causal self-attention over the full
    history (cfg.temporal_mixer="attention", KV cache at inference) or a
    gated linear recurrence (cfg.temporal_mixer="recurrent", fixed-size
    state at inference — episode memory lives in the neurons).

    ``cfg.qk_norm`` adds per-head q/k RMSNorm (``QKNorm``) to the spatial
    attention and to the temporal attention: the NEW submodules
    ``spatial_qk_norm`` / ``temporal_qk_norm`` exist only when it is on, so
    a qk_norm=False block (every checkpoint saved before the field) has
    exactly the historical parameters and forward code.  With it on,
    forward_step runs the same SDPA code as forward instead of
    ``nn.MultiheadAttention.forward``.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mixer = getattr(cfg, "temporal_mixer", "attention")
        if self.mixer not in ("attention", "recurrent"):
            raise ValueError(f"unknown temporal_mixer: {self.mixer!r}")

        self.spatial_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(cfg.d_model)

        if self.mixer == "recurrent":
            self.temporal_rnn = GatedRecurrence(cfg.d_model)
        else:
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

        # QK-norm (registered last: qk_norm=False keeps the legacy
        # parameter set and order)
        qk = getattr(cfg, "qk_norm", False)
        d_head = cfg.d_model // cfg.n_heads
        self.spatial_qk_norm = QKNorm(d_head) if qk else None
        self.temporal_qk_norm = (QKNorm(d_head)
                                 if qk and self.mixer == "attention" else None)

    def _spatial_sdpa(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial self-attention via F.scaled_dot_product_attention.

        Bypasses nn.MultiheadAttention.forward() but reuses its parameters,
        avoiding checkpoint recomputation mismatches caused by non-deterministic
        dispatch inside nn.MHA.
        """
        B, S, d = x.shape
        n_heads = self.spatial_attn.num_heads
        d_head = d // n_heads

        qkv = F.linear(x, self.spatial_attn.in_proj_weight, self.spatial_attn.in_proj_bias)
        qkv = qkv.reshape(B, S, 3, n_heads, d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.spatial_qk_norm is not None:
            q, k = self.spatial_qk_norm(q, k)

        # Pin the kernel set so the re-entrant checkpoint recompute cannot
        # dispatch to a different backend (they consume dropout RNG
        # differently, which would silently corrupt gradients).
        dropout_p = self.spatial_attn.dropout if self.training else 0.0
        with torch.nn.attention.sdpa_kernel([
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]):
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, d)
        return self.spatial_attn.out_proj(attn_out)

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
        if self.temporal_qk_norm is not None:
            q, k = self.temporal_qk_norm(q, k)

        dropout_p = self.temporal_attn.dropout if self.training else 0.0
        with torch.nn.attention.sdpa_kernel([
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]):
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True,
            )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, d)
        return self.temporal_attn.out_proj(attn_out)

    def _temporal_attn_step(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """One query frame attending to the cached history + itself —
        the last row of :meth:`_causal_temporal_attn` (same projection
        weights, same QK-norm).  Used by forward_step when qk_norm is on.

        x_q [N, 1, d] (current frame), x_kv [N, t, d] (history + current).
        """
        N, t, d = x_kv.shape
        n_heads = self.temporal_attn.num_heads
        d_head = d // n_heads
        w, b = self.temporal_attn.in_proj_weight, self.temporal_attn.in_proj_bias
        w_q, w_kv = w.split([d, 2 * d])
        b_q, b_kv = b.split([d, 2 * d]) if b is not None else (None, None)
        q = F.linear(x_q, w_q, b_q).reshape(N, 1, n_heads, d_head).transpose(1, 2)
        kv = F.linear(x_kv, w_kv, b_kv).reshape(N, t, 2, n_heads, d_head).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)                                  # each [N, n_heads, t, d_head]
        q, k = self.temporal_qk_norm(q, k)
        dropout_p = self.temporal_attn.dropout if self.training else 0.0
        with torch.nn.attention.sdpa_kernel([
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]):
            # the query is the newest frame: every key is in its past
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        return self.temporal_attn.out_proj(attn_out.transpose(1, 2).reshape(N, 1, d))

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
        sa_out = self._spatial_sdpa(x_flat)
        x_flat = self.spatial_norm(x_flat + sa_out)
        x = x_flat.reshape(B, T, S, d)

        # ── temporal mixing (across timesteps, per position) ──
        x_perm = x.permute(0, 2, 1, 3).reshape(B * S, T, d)
        if self.mixer == "recurrent":
            ta_out = self.temporal_rnn(x_perm)
        else:
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
        Single-step inference.

        Args:
            x_cur          : [B, 1, S, d]   current timestep only
            temporal_cache : mixer="attention": [B*S, t_prev, d] KV cache
                             (grows each step) or None;
                             mixer="recurrent": [B*S, d] float32 recurrent
                             state (fixed size) or None.
        Returns:
            output         : [B, 1, S, d]
            new_cache      : same convention as ``temporal_cache``
        """
        B, _, S, d = x_cur.shape

        # ── spatial (only current timestep) ──
        x_flat = x_cur.reshape(B, S, d)
        if self.spatial_qk_norm is not None:
            sa_out = self._spatial_sdpa(x_flat)
        else:
            sa_out, _ = self.spatial_attn(x_flat, x_flat, x_flat, need_weights=False)
        x_flat = self.spatial_norm(x_flat + sa_out)

        if self.mixer == "recurrent":
            # ── temporal: one recurrence step on a fixed-size state ──
            q = x_flat.reshape(B * S, d)
            ta_out, new_cache = self.temporal_rnn.forward_step(q, temporal_cache)
            q = self.temporal_norm(q + ta_out)
            x_out = q.reshape(B, S, d)
        else:
            # ── temporal (query = current, key/value = cache + current) ──
            q = x_flat.reshape(B * S, 1, d)
            if temporal_cache is not None:
                kv = torch.cat([temporal_cache, q], dim=1)  # [B*S, t_prev+1, d]
            else:
                kv = q  # [B*S, 1, d]
            new_cache = kv

            if self.temporal_qk_norm is not None:
                ta_out = self._temporal_attn_step(q, kv)
            else:
                ta_out, _ = self.temporal_attn(q, kv, kv, need_weights=False)
            q = self.temporal_norm(q + ta_out)
            x_out = q.reshape(B, S, d)

        # ── feed-forward ──
        x_out = self.ff_norm(x_out + self.ff(x_out))
        return x_out.unsqueeze(1), new_cache
