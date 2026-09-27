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


# Scan implementation selection.
#
# Measured on this codebase's real shape ([627·B, 1000, 384] fp32): each
# sequential step is already a ~240k-element tensor op, so the loop is
# BANDWIDTH-bound, not launch-bound — the parallel scan's log2(chunk)×
# extra memory traffic makes it SLOWER on CPU (0.6×) and MPS (0.2×).
# The parallel path is kept, tested for exact equivalence, and selectable
# for devices where kernel-launch overhead dominates instead; benchmark
# before flipping this on CUDA.
_USE_PARALLEL_SCAN = False

# Chunk length for the parallel scan: T is processed in sequential chunks
# (bounded transient memory), each scanned with log2(chunk) parallel steps.
_SCAN_CHUNK = 256


def _hillis_steele(a: torch.Tensor, u: torch.Tensor):
    """Inclusive parallel scan of the linear recurrence x_t = a_t x_{t-1} + u_t.

    Returns (A, U) per position t such that x_t = A_t * x_0 + U_t, i.e. the
    composed segment coefficients over 1..t. log2(T) parallel steps; the
    composition (a2,u2)∘(a1,u1) = (a1*a2, a2*u1 + u2) is associative.
    """
    A, U = a, u
    offset = 1
    T = a.shape[1]
    while offset < T:
        A_tail = A[:, offset:] * A[:, :-offset]
        U_tail = A[:, offset:] * U[:, :-offset] + U[:, offset:]
        A = torch.cat([A[:, :offset], A_tail], dim=1)
        U = torch.cat([U[:, :offset], U_tail], dim=1)
        offset *= 2
    return A, U


def _scan_parallel(a: torch.Tensor, u: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Chunked Hillis-Steele evaluation of h_t = a_t h_{t-1} + u_t."""
    T = a.shape[1]
    hs = []
    h = h0
    for s in range(0, T, _SCAN_CHUNK):
        A, U = _hillis_steele(a[:, s:s + _SCAN_CHUNK], u[:, s:s + _SCAN_CHUNK])
        hc = A * h.unsqueeze(1) + U
        hs.append(hc)
        h = hc[:, -1]
    return torch.cat(hs, dim=1)


def _scan_forward(a: torch.Tensor, u: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """h_t = a_t h_{t-1} + u_t. Plain math (no autograd assumptions) —
    used by BOTH directions of _GatedScan. Dispatches between the
    bandwidth-optimal sequential loop and the launch-optimal parallel scan
    (see _USE_PARALLEL_SCAN)."""
    if _USE_PARALLEL_SCAN and a.device.type != "cpu":
        return _scan_parallel(a, u, h0)

    h = h0
    hs = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + u[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)


def _scan_sequential(a: torch.Tensor, v: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Step-by-step reference of the gated scan (kept for tests/benchmarks)."""
    h = h0
    hs = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + (1.0 - a[:, t]) * v[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)


class _GatedScan(torch.autograd.Function):
    """h_t = a_t * h_{t-1} + (1 - a_t) * v_t, with analytic backward.

    a, v: [B, T, d] (float32), h0: [B, d] (float32) → h_all: [B, T, d].

    Both directions are linear scans and both run as chunked parallel scans:
      forward:   h_t  = a_t * h_{t-1} + u_t,          u_t = (1 - a_t) v_t
      backward:  dH_t = a_{t+1} * dH_{t+1} + g_t      (reverse scan)
    then  dL/da_t = dH_t (h_{t-1} - v_t),  dL/dv_t = dH_t (1 - a_t),
          dL/dh0 = a_1 dH_1.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, v: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        h_all = _scan_forward(a, (1.0 - a) * v, h0)
        ctx.save_for_backward(a, v, h0, h_all)
        return h_all

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, v, h0, h_all = ctx.saved_tensors

        # Reverse scan: flip time, coefficient at t becomes a_{t+1} (0 at T).
        coef = torch.cat([a[:, 1:], torch.zeros_like(a[:, :1])], dim=1)
        dH = torch.flip(
            _scan_forward(
                torch.flip(coef, dims=[1]),
                torch.flip(grad_out, dims=[1]),
                torch.zeros_like(h0),
            ),
            dims=[1],
        )

        h_prev = torch.cat([h0.unsqueeze(1), h_all[:, :-1]], dim=1)
        da = dH * (h_prev - v)
        dv = dH * (1.0 - a)
        dh0 = dH[:, 0] * a[:, 0]
        return da, dv, dh0


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

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None,
                return_state: bool = False):
        """x: [B_, T, d] → y: [B_, T, d] (full-sequence training path); from state
        ``h0`` [B_, d] float32 (None = h_init), with ``return_state`` also the state
        after the last frame, (y, h_T) — a longer sequence in chunks."""
        a, v = self._gates(x)
        if h0 is None:
            h0 = self.initial_state(x.shape[0])
        h_all = _GatedScan.apply(a, v, h0)
        y = self._readout(h_all, x)
        return (y, h_all[:, -1]) if return_state else y

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
