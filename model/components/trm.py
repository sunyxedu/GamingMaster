"""
Tiny Recursive Model (TRM) — faithful implementation based on
https://github.com/SamsungSAILMontreal/TinyRecursiveModels
https://arxiv.org/html/2510.04871v1

Core idea: two state vectors (z_H, z_L) are iteratively refined by a
single shared reasoning module.  z_L is the low-level "reasoning
scratchpad"; z_H is the high-level "answer" state.  Nested loops:

    for h in H_cycles:
        for l in L_cycles:
            z_L = reasoning(z_L, injection=z_H + x)   # refine reasoning
        z_H = reasoning(z_H, injection=z_L)            # update answer

Only the last H_cycle propagates gradients (the rest are detached for
memory efficiency, following the paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from model.config import ModelConfig


class TRMBlock(nn.Module):
    """Single transformer block: Self-Attention → Add & Norm → MLP → Add & Norm."""

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

    def _sdpa(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention via F.scaled_dot_product_attention.

        Bypasses nn.MultiheadAttention.forward() to avoid checkpoint
        recomputation mismatches caused by non-deterministic dispatch.
        """
        B, S, d = x.shape
        n_heads = self.attn.num_heads
        d_head = d // n_heads

        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        qkv = qkv.reshape(B, S, 3, n_heads, d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        dropout_p = self.attn.dropout if self.training else 0.0
        with torch.nn.attention.sdpa_kernel([
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]):
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, d)
        return self.attn.out_proj(attn_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self._sdpa(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TRMReasoningModule(nn.Module):
    """
    Shared reasoning module: additive input injection → TransformerBlock × N.

    This single module is reused for both z_L and z_H updates.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TRMBlock(cfg) for _ in range(cfg.trm_block_layers)]
        )

    def forward(
        self, z: torch.Tensor, injection: torch.Tensor,
    ) -> torch.Tensor:
        z = z + injection
        for layer in self.layers:
            z = layer(z)
        return z


class TRM(nn.Module):
    """
    Tiny Recursive Model — two-state recursive pondering.

    Args (from config):
        trm_H_cycles    : outer loop iterations (refine answer)
        trm_L_cycles    : inner loop iterations (refine reasoning)
        trm_block_layers: transformer layers inside the shared reasoning module
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Shared reasoning module (used for both z_L and z_H updates)
        self.reasoning = TRMReasoningModule(cfg)

        # Fixed initial states (not trainable, following the paper)
        self.register_buffer("H_init", torch.randn(cfg.d_model))
        self.register_buffer("L_init", torch.randn(cfg.d_model))

        # Q-halt head: predicts whether the current answer is good enough
        self.q_head = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

    def _reasoning_with_injection(
        self,
        z: torch.Tensor,
        z_H: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Checkpoint-friendly wrapper that builds ``z_H + x`` inside the segment."""
        return self.reasoning(z, z_H + x)

    def forward(
        self,
        x: torch.Tensor,
        carry: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:     [B, S, d]  input embeddings (from spatiotemporal encoder)
            carry: optional (z_H, z_L) from a previous ACT step

        Returns:
            z_H:       [B, S, d]  final high-level state (→ decode to heads)
            q_halt:    [B]        halt Q-logit (positive = confident, should halt)
            new_carry: (z_H, z_L) detached, for next ACT step
        """
        B, S, d = x.shape
        H_cycles = self.cfg.trm_H_cycles
        L_cycles = self.cfg.trm_L_cycles

        if carry is not None:
            z_H, z_L = carry
        else:
            z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(B, S, d)
            z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(B, S, d)

        # H_cycles-1 without gradient (memory efficiency, following the paper)
        if H_cycles > 1:
            with torch.no_grad():
                for _ in range(H_cycles - 1):
                    for _ in range(L_cycles):
                        z_L = self.reasoning(z_L, z_H + x)
                    z_H = self.reasoning(z_H, z_L)
            z_H = z_H.detach().requires_grad_(True)
            z_L = z_L.detach().requires_grad_(True)

        # Last H_cycle with gradient. Keep checkpoint granularity per step so
        # backward only recomputes one reasoning step at a time, but avoid
        # materialising ``z_H + x`` outside the checkpoint.
        for _ in range(L_cycles):
            z_L = checkpoint(
                self._reasoning_with_injection,
                z_L,
                z_H,
                x,
                use_reentrant=True,
            )
        z_H = checkpoint(self.reasoning, z_H, z_L, use_reentrant=True)

        # Q-halt from the first token position (following paper)
        q_halt = self.q_head(z_H[:, 0]).squeeze(-1)

        new_carry = (z_H.detach(), z_L.detach())
        return z_H, q_halt, new_carry
