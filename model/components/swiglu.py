"""
SwiGLU activation and FFN implementation.

SwiGLU is a gated activation function used in TRM and modern LLMs.
Reference: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish-Gated Linear Unit.

    Computes: SwiGLU(x, W, V) = Swish(xW) * xV
    Where Swish(x) = x * sigmoid(x) = SiLU(x)
    """

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.

        Args:
            x: Main input tensor
            gate: Gate tensor (same shape as x)

        Returns:
            Gated output
        """
        return F.silu(gate) * x


class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.

    TRM-style: No bias terms in any linear layers.

    Architecture:
        Input -> [W1, W3] (parallel) -> SwiGLU(W1*x, W3*x) -> W2 -> Output

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (typically 4 * d_model)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        # No bias per TRM paper
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection

        self.swiglu = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU FFN.

        Args:
            x: Input tensor of shape [B, N, D]

        Returns:
            Output tensor of shape [B, N, D]
        """
        # Compute gate and value in parallel
        gate = self.w3(x)  # [B, N, d_ff]
        value = self.w1(x)  # [B, N, d_ff]

        # Apply SwiGLU: SiLU(gate) * value
        hidden = self.swiglu(value, gate)  # [B, N, d_ff]

        # Down projection
        return self.w2(hidden)  # [B, N, D]

    def extra_repr(self) -> str:
        return f'd_model={self.w1.in_features}, d_ff={self.w1.out_features}'
