"""
RMSNorm implementation (Root Mean Square Layer Normalization).

Used in TRM instead of LayerNorm for better training stability.
Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm:
    - Does not center the activations (no mean subtraction)
    - Only rescales using root mean square
    - Has no bias term (per TRM paper)

    Args:
        dim: Dimension of the input features
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        return (x / rms) * self.weight

    def extra_repr(self) -> str:
        return f'{self.weight.shape[0]}, eps={self.eps}'
