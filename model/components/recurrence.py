"""Token-wise gated linear recurrence — the temporal memory core.

Replaces causal temporal attention with a fixed-size recurrent state so
episode memory lives in persistent activations ("in the neurons") instead
of a growing KV cache:

    a_t = sigmoid(W_a x_t + b_a)            retention gate, per channel
    h_t = a_t * h_{t-1} + (1 - a_t) * v_t   v_t = W_v x_t
    y_t = W_o (h_t * silu(W_g x_t))         gated readout

Properties this module guarantees (and tests assert):
  - forward(x) over T steps is EXACTLY the same math as chaining
    forward_step() — required for rollout/update policy identity (PPO).
  - Gradients flow through the full recurrence (true BPTT, analytic
    backward, O(T) time / O(T) memory) so the network can learn WHAT to
    write into memory, not just how to read it.
  - The scan runs in float32 regardless of autocast: products of up to
    thousands of gates are numerically fragile in bf16.
  - b_a is initialized so retention half-lives are geometrically spread
    across channels (multi-timescale memory from step one).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GatedScan(torch.autograd.Function):
    """h_t = a_t * h_{t-1} + (1 - a_t) * v_t, with analytic backward.

    a, v: [B, T, d] (float32), h0: [B, d] (float32) → h_all: [B, T, d].

    The backward recurrence, derived from the forward:
        dL/dh_t      = grad_out_t + a_{t+1} * dL/dh_{t+1}
        dL/da_t      = dL/dh_t * (h_{t-1} - v_t)
        dL/dv_t      = dL/dh_t * (1 - a_t)
        dL/dh0       = a_1 * dL/dh_1
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, v: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        T = a.shape[1]
        h = h0
        hs = []
        for t in range(T):
            h = a[:, t] * h + (1.0 - a[:, t]) * v[:, t]
            hs.append(h)
        h_all = torch.stack(hs, dim=1)
        ctx.save_for_backward(a, v, h0, h_all)
        return h_all

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, v, h0, h_all = ctx.saved_tensors
        T = a.shape[1]
        da = torch.empty_like(a)
        dv = torch.empty_like(v)
        dh = torch.zeros_like(h0)
        for t in range(T - 1, -1, -1):
            dh = dh + grad_out[:, t]
            h_prev = h_all[:, t - 1] if t > 0 else h0
            da[:, t] = dh * (h_prev - v[:, t])
            dv[:, t] = dh * (1.0 - a[:, t])
            dh = dh * a[:, t]
        return da, dv, dh


class GatedRecurrence(nn.Module):
    """Diagonal gated linear recurrence over the time axis of one token
    position (callers flatten [B, S] positions into the batch dim).

    State is a single [batch, d] float32 tensor — O(1) per step, no growth.
    """

    def __init__(
        self,
        d_model: int,
        max_half_life: float = 1024.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.a_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Retention bias: half-lives geometrically spread over channels,
        # L in [1, max_half_life]  →  a = 2^(-1/L),  b = logit(a).
        # Raw Parameter (not a Linear bias) so GeneralsNetwork._init_weights
        # cannot zero it.
        half_lives = torch.logspace(0, math.log10(max_half_life), d_model)
        a_init = torch.exp(-math.log(2.0) / half_lives)
        self.retention_bias = nn.Parameter(torch.log(a_init / (1.0 - a_init)))

        # Learnable initial state (float32 semantics; stored as a Parameter).
        self.h_init = nn.Parameter(torch.zeros(d_model))

    def _gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retention gate and write candidate, always in float32.

        The gate is clamped below exact 1.0: fp32 sigmoid saturates at
        logits ≳ 16.6, which would zero the write path (1-a)·v AND its
        gradient, permanently closing the channel. The clamp caps the
        half-life at ~7e5 steps instead — effectively infinite, but alive.
        """
        a = torch.sigmoid((self.a_proj(x) + self.retention_bias).float())
        a = a.clamp(max=1.0 - 1e-6)
        v = self.v_proj(x).float()
        return a, v

    def _readout(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(h.to(x.dtype) * F.silu(self.g_proj(x)))

    def initial_state(self, batch: int) -> torch.Tensor:
        return self.h_init.float().unsqueeze(0).expand(batch, -1)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B_, T, d] → y: [B_, T, d] (full-sequence training path)."""
        a, v = self._gates(x)
        if h0 is None:
            h0 = self.initial_state(x.shape[0])
        h_all = _GatedScan.apply(a, v, h0)
        return self._readout(h_all, x)

    def forward_step(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B_, d], h_prev: [B_, d] float32 (or None at t=0).

        Returns (y, h_new). Exactly one scan step of forward().
        """
        a, v = self._gates(x)
        if h_prev is None:
            h_prev = self.initial_state(x.shape[0])
        h = a * h_prev + (1.0 - a) * v
        return self._readout(h, x), h
