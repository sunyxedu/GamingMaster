"""
TRM-style Recursive Transformer Backbone.

Implements the Tiny Recursive Model architecture from:
"Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)

Key features:
- Dual state recursion: z (latent reasoning) and y (answer)
- Deep recursion with no_grad for efficiency
- Adaptive halting mechanism
- Single tiny network (2 layers) with cycle embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .components import RMSNorm, SwiGLUFFN, TRMAttention


class TRMLayer(nn.Module):
    """
    Single TRM Transformer layer.

    Pre-norm architecture with:
    - Self-attention (no bias)
    - SwiGLU FFN (no bias)
    - RMSNorm
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()

        self.attention = TRMAttention(d_model, n_heads, dropout)
        self.ffn = SwiGLUFFN(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply transformer layer.

        Args:
            x: Input tensor [B, N, D]
            attention_mask: Optional attention mask [B, N]

        Returns:
            Output tensor [B, N, D]
        """
        # Pre-norm attention
        x = x + self.attention(self.norm1(x), attention_mask)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))

        return x


class HaltHead(nn.Module):
    """
    Halting mechanism head for adaptive computation.

    Predicts whether to halt the recursive reasoning at this cycle.
    Trained via BCE loss: BCE(halt_prob, is_correct)
    """

    def __init__(self, d_model: int = 512):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=False)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute halt probability.

        Args:
            z: Token representations [B, N, D]

        Returns:
            Halt probability [B]
        """
        # Global pool then predict
        pooled = z.mean(dim=1)  # [B, D]
        logit = self.proj(pooled).squeeze(-1)  # [B]
        return torch.sigmoid(logit)


class AnswerHead(nn.Module):
    """
    Answer refinement head for TRM.

    Updates the answer state y given latent z.
    y = net(y, z) in the TRM formulation.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()

        # Combine y and z, then refine
        self.combine = nn.Linear(d_model * 2, d_model, bias=False)
        self.refine = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False)
        )

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Update answer state.

        Args:
            y: Current answer state [B, N, D]
            z: Latent reasoning state [B, N, D]

        Returns:
            Updated answer state [B, N, D]
        """
        combined = torch.cat([y, z], dim=-1)  # [B, N, 2D]
        combined = self.combine(combined)  # [B, N, D]
        return y + self.refine(combined)  # Residual update


class TRMBackbone(nn.Module):
    """
    TRM-style Recursive Transformer Backbone.

    Implements the full TRM architecture from the paper:
    - Dual state: z (latent reasoning) and y (answer)
    - Recursive update: z ← net(x, y, z), y ← net(y, z)
    - Deep recursion: T supervision steps, n inner recursions each
    - Adaptive halting mechanism

    Args:
        d_model: Model dimension (default 512)
        n_heads: Number of attention heads (default 8)
        d_ff: FFN hidden dimension (default 2048)
        n_layers: Number of transformer layers (default 2)
        n_inner_cycles: Inner recursion cycles per supervision step (default 6)
        n_supervision_steps: Number of supervision steps (default 3)
        dropout: Dropout probability (default 0.0)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_layers: int = 2,
        n_inner_cycles: int = 6,
        n_supervision_steps: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()

        self.d_model = d_model
        self.n_inner_cycles = n_inner_cycles
        self.n_supervision_steps = n_supervision_steps
        self.total_cycles = n_inner_cycles * n_supervision_steps

        # Transformer layers for z update: z ← net(x, y, z)
        self.z_layers = nn.ModuleList([
            TRMLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Answer refinement head: y ← net(y, z)
        self.answer_head = AnswerHead(d_model)

        # Cycle embedding to distinguish recursion steps
        self.cycle_embedding = nn.Embedding(self.total_cycles, d_model)

        # Halting head
        self.halt_head = HaltHead(d_model)

        # Final normalization
        self.final_norm = RMSNorm(d_model)

        # Projection for combining x with y and z
        self.xyz_combine = nn.Linear(d_model * 2, d_model, bias=False)

    def _single_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        cycle_idx: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single recursion step: z ← net(x, y, z), y ← net(y, z)

        Args:
            x: Input tokens [B, N, D]
            y: Answer state [B, N, D]
            z: Latent reasoning state [B, N, D]
            cycle_idx: Current cycle index
            attention_mask: Optional attention mask

        Returns:
            Updated (y, z)
        """
        device = x.device

        # Add cycle embedding to z
        cycle_emb = self.cycle_embedding(
            torch.tensor([cycle_idx], device=device)
        ).unsqueeze(1)  # [1, 1, D]
        z_input = z + cycle_emb

        # Combine x, y with z for context: z ← net(x, y, z)
        # We concatenate y with z and project
        yz_combined = torch.cat([y, z_input], dim=-1)  # [B, N, 2D]
        z_input = self.xyz_combine(yz_combined)  # [B, N, D]
        z_input = z_input + x  # Add input context

        # Pass through transformer layers
        for layer in self.z_layers:
            z_input = layer(z_input, attention_mask)

        # Update z with residual
        z = z + z_input

        # Update y: y ← net(y, z)
        y = self.answer_head(y, z)

        return y, z

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_all_steps: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        TRM forward pass with deep recursion.

        During training:
        - For each supervision step t:
            - Run n_inner_cycles with no_grad (except last)
            - Compute halt probability
            - Return intermediate y for loss computation

        Args:
            x: Input tokens [B, N, D]
            attention_mask: Optional attention mask [B, N]
            deterministic: If True, use hard halting threshold
            return_all_steps: If True, return y at each supervision step

        Returns:
            y: Final answer state [B, N, D]
            z: Final latent state [B, N, D]
            halt_probs: Halting probabilities [B, n_supervision_steps]
            all_y: (Optional) Answer states at each supervision step
        """
        B, N, D = x.shape
        device = x.device

        # Initialize y and z
        y = x.clone()  # Initial answer = input
        z = torch.zeros_like(x)  # Initial latent = zeros

        halt_probs = []
        all_y = [] if return_all_steps else None

        global_cycle = 0

        for t in range(self.n_supervision_steps):
            # Deep recursion: n_inner_cycles per supervision step
            for i in range(self.n_inner_cycles):
                is_last_inner = (i == self.n_inner_cycles - 1)

                if is_last_inner or not self.training:
                    # Last inner cycle: compute gradients
                    y, z = self._single_recursion(x, y, z, global_cycle, attention_mask)
                else:
                    # Earlier cycles: no gradients for efficiency
                    with torch.no_grad():
                        y, z = self._single_recursion(x, y, z, global_cycle, attention_mask)

                global_cycle += 1

            # Compute halt probability at end of each supervision step
            halt_prob = self.halt_head(z)
            halt_probs.append(halt_prob)

            # Store intermediate y for supervision
            if return_all_steps:
                all_y.append(y.clone())

            # Early stopping during inference
            if deterministic and (halt_prob > 0.5).all():
                break

        # Pad halt_probs if stopped early
        while len(halt_probs) < self.n_supervision_steps:
            halt_probs.append(torch.ones(B, device=device))

        halt_probs = torch.stack(halt_probs, dim=1)  # [B, n_supervision_steps]

        # Final normalization
        y = self.final_norm(y)
        z = self.final_norm(z)

        return y, z, halt_probs, all_y

    def inference(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode with adaptive halting.

        Args:
            x: Input tokens [B, N, D]
            attention_mask: Optional attention mask
            max_steps: Maximum supervision steps (default: n_supervision_steps)

        Returns:
            y: Final answer state [B, N, D]
            z: Final latent state [B, N, D]
        """
        y, z, _, _ = self.forward(
            x, attention_mask,
            deterministic=True,
            return_all_steps=False
        )
        return y, z


