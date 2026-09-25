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

Block layout (``cfg.trm_arch``): "trm" reproduces the reference block
(models/recursive_reasoning/trm.py + models/layers.py): post-norm
RMSNorm without weight, bias-free attention and SwiGLU, no dropout,
truncated LeCun-normal init.  "legacy" is the earlier LayerNorm / GELU /
dropout block, kept so existing checkpoints load unchanged.
"""

import math

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


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0,
                       lower: float = -2.0, upper: float = 2.0) -> torch.Tensor:
    """JAX-style truncated normal whose std really is ``std`` (reference
    models/common.py; torch's trunc_normal_ under-shoots the std)."""
    with torch.no_grad():
        if std == 0:
            return tensor.zero_()
        sqrt2 = math.sqrt(2)
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * lower ** 2)
        pdf_l = c * math.exp(-0.5 * upper ** 2)
        comp_std = std / math.sqrt(
            1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        tensor.uniform_(a, b)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMSNorm computed in fp32 (reference models/layers.py)."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x.to(dtype)


class TRMLinear(nn.Module):
    """Linear layer with truncated LeCun-normal init (reference CastedLinear).

    Deliberately NOT an ``nn.Linear`` subclass so the network-wide
    ``_init_weights`` pass (std 0.02) leaves it alone."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(
            torch.empty(out_features, in_features), std=in_features ** -0.5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden * 2 / 3), 256)
        self.gate_up_proj = TRMLinear(hidden, inter * 2)
        self.down_proj = TRMLinear(inter, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class ReferenceTRMBlock(nn.Module):
    """Reference TRM block: x = rms(x + attn(x)); x = rms(x + swiglu(x)).

    Non-causal multi-head attention without biases.  Positions are already
    in the input (the network adds learned PEs before Stage 1), matching
    the reference's ``pos_encodings: learned`` mode."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        self.eps = cfg.trm_rms_eps
        self.qkv_proj = TRMLinear(d, 3 * d)
        self.o_proj = TRMLinear(d, d)
        self.mlp = SwiGLU(d, cfg.trm_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, d = x.shape
        qkv = self.qkv_proj(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v)
        a = self.o_proj(a.permute(0, 2, 1, 3).reshape(B, S, d))
        x = rms_norm(x + a, self.eps)
        return rms_norm(x + self.mlp(x), self.eps)


class TRMReasoningModule(nn.Module):
    """
    Shared reasoning module: additive input injection → TransformerBlock × N.

    This single module is reused for both z_L and z_H updates.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        arch = getattr(cfg, "trm_arch", "trm")
        if arch not in ("trm", "legacy"):
            raise ValueError(f"unknown trm_arch {arch!r} (expected 'trm' or 'legacy')")
        block = ReferenceTRMBlock if arch == "trm" else TRMBlock
        self.layers = nn.ModuleList(
            [block(cfg) for _ in range(cfg.trm_block_layers)]
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

        # Fixed initial states (not trainable, following the paper;
        # reference: trunc_normal std 1)
        if getattr(cfg, "trm_arch", "trm") == "trm":
            self.register_buffer("H_init", trunc_normal_init_(torch.empty(cfg.d_model)))
            self.register_buffer("L_init", trunc_normal_init_(torch.empty(cfg.d_model)))
        else:
            self.register_buffer("H_init", torch.randn(cfg.d_model))
            self.register_buffer("L_init", torch.randn(cfg.d_model))

        # Q-halt head: predicts whether the current answer is good enough
        self.q_head = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

        # Read q_halt from the HL decision token (index N = grid_h*grid_w in
        # the network's token layout).  Falls back to token 0 when the
        # sequence is shorter (e.g. unit tests running TRM standalone).
        self.q_token_idx = cfg.grid_h * cfg.grid_w

    def initial_carry(self, B: int, S: int, like: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fresh (z_H, z_L) = (H_init, L_init) broadcast to [B, S, d]."""
        d = self.cfg.d_model
        z_H = self.H_init.to(like.dtype).expand(B, S, d)
        z_L = self.L_init.to(like.dtype).expand(B, S, d)
        return z_H, z_L

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

        # Re-entrant checkpointing propagates grads only when its inputs
        # require grad — this must run for H_cycles == 1 (init/carry states)
        # too, not just after the no-grad cycles above.
        if torch.is_grad_enabled():
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

        q_idx = self.q_token_idx if S > self.q_token_idx else 0
        q_halt = self.q_head(z_H[:, q_idx]).squeeze(-1)

        new_carry = (z_H.detach(), z_L.detach())
        return z_H, q_halt, new_carry
