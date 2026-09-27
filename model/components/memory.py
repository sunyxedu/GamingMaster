"""Episode memory inside the Stage-2 core — sub-layers between core blocks.

The Stage-1 spatiotemporal layers are the network's only other memory: a gated
recurrence per token position at the bottom of the network, so what is carried from
frame to frame are shallow features; the 12-layer core recomputes every conclusion
from the current frame.  These sub-layers carry the core's deep features across frames:

* :class:`GatedMemory` — the Stage-1 recurrence (model.components.recurrence), per token
  position, after a core block: ``h_t = a_t ⊙ h_{t−1} + (1 − a_t) ⊙ v_t`` over the
  block's output, read out gated; a spatially anchored memory (every cell / unit / card
  slot keeps its own history of deep features).
* :class:`DeltaMemory` — a GLOBAL associative memory per record (gated delta rule,
  Yang et al. 2024 arXiv 2412.06464): per head a matrix M (d_v × d_k).  Each frame
  pools its tokens into ``writes`` key / value pairs (learned attention pooling) and
  updates ``M ← α M + Σ_w β_w (v_w − α M k_w) k_wᵀ`` (a key's old value is replaced,
  not piled up); every token reads ``M_{t−1} q`` — content-addressed recall of past
  frames by any position.  State: heads × d_v × d_k per record, fixed.

Both are residual (``x + out(...)``, pre-norm) with a ZERO-initialised output
projection: a network with memory sub-layers added computes exactly what it computed
without them, so they can join a trained network and continue from its weights.
Both are causal and linear in their state (scan over frames), so training processes
every frame of a sequence together while inference steps one frame at a time with a
fixed-size state (``step``; the Stage-1 cache list of ``GameNetwork.step`` carries it);
``forward`` over T frames equals T chained ``step`` calls.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.recurrence import GatedRecurrence


def memory_layers(spec, n_blocks: int) -> List[int]:
    """Block indices of a ``"2,5,8"`` spec (after which block a sub-layer sits); also
    an int or a sequence (a ``--override core_gated_memory=2,5,8`` parses as a tuple)."""
    if spec is None or spec == "" or spec == ():
        return []
    if isinstance(spec, str):
        items = [s for s in spec.replace(" ", "").split(",") if s != ""]
    elif isinstance(spec, int):
        items = [spec]
    else:
        items = list(spec)
    out = sorted({int(s) for s in items})
    bad = [i for i in out if not 0 <= i < n_blocks]
    if bad:
        raise ValueError(f"core memory layers {bad} outside the core's {n_blocks} blocks")
    return out


class GatedMemory(nn.Module):
    """Per-position gated recurrence over a core block's output (module docstring)."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.rnn = GatedRecurrence(d)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """The zero output (identity sub-layer); call after a network-wide re-init."""
        with torch.no_grad():
            self.rnn.out_proj.weight.zero_()

    def initial_state(self, batch: int, tokens: int) -> torch.Tensor:
        """[batch · tokens, d] float32 — the learned h_init of every position."""
        return self.rnn.initial_state(batch * tokens).clone(memory_format=torch.contiguous_format)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, S, d] → same shape."""
        B, T, S, d = x.shape
        xs = x.permute(0, 2, 1, 3).reshape(B * S, T, d)
        y = self.rnn(self.norm(xs))
        return x + y.reshape(B, S, T, d).permute(0, 2, 1, 3).to(x.dtype)

    def step(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """x [B, S, d], h [B·S, d] (None = h_init) → (x', h')."""
        B, S, d = x.shape
        y, h = self.rnn.forward_step(self.norm(x.reshape(B * S, d)), h)
        return x + y.reshape(B, S, d).to(x.dtype), h


class DeltaMemory(nn.Module):
    """Global gated-delta associative memory of a record (module docstring)."""

    def __init__(self, d: int, heads: int = 8, dk: int = 64, dv: int = 64, writes: int = 4,
                 max_half_life: float = 1024.0):
        super().__init__()
        self.d, self.heads, self.dk, self.dv, self.writes = d, heads, dk, dv, writes
        self.norm = nn.LayerNorm(d)
        self.pool = nn.Linear(d, writes, bias=False)             # write slots' attention
        self.k_proj = nn.Linear(d, heads * dk, bias=False)
        self.v_proj = nn.Linear(d, heads * dv, bias=False)
        self.q_proj = nn.Linear(d, heads * dk, bias=False)
        self.beta_proj = nn.Linear(d, heads)                     # write strength per slot
        self.alpha_proj = nn.Linear(d, heads)                    # retention per frame
        self.out_proj = nn.Linear(heads * dv, d, bias=False)
        self.max_half_life = max_half_life
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Zero output (identity sub-layer) and the retention prior: half-lives spread
        over the heads, 8 … max_half_life frames; call after a network-wide re-init."""
        hl = torch.logspace(math.log10(8.0), math.log10(self.max_half_life), self.heads)
        a0 = torch.exp(-math.log(2.0) / hl)
        with torch.no_grad():
            self.out_proj.weight.zero_()
            self.alpha_proj.weight.zero_()
            self.alpha_proj.bias.copy_(torch.log(a0 / (1.0 - a0)))
            self.beta_proj.bias.zero_()

    def initial_state(self, batch: int, tokens: int = 1) -> torch.Tensor:
        """[batch, heads, dv, dk] float32 zeros (an empty memory)."""
        p = self.k_proj.weight
        return torch.zeros(batch, self.heads, self.dv, self.dk, device=p.device, dtype=torch.float32)

    def _frame_terms(self, x: torch.Tensor):
        """x [..., S, d] (normed) → per frame: keys / values [..., W, H, dk|dv], β
        [..., W, H], α [..., H] (float32), queries [..., S, H, dk]."""
        H, W = self.heads, self.writes
        att = torch.softmax(self.pool(x).float(), dim=-2)                  # [..., S, W]
        pooled = torch.einsum("...sw,...sd->...wd", att, x.float())        # [..., W, d]
        pooled = pooled.to(x.dtype)
        k = F.normalize(self.k_proj(pooled).float().unflatten(-1, (H, self.dk)), dim=-1)
        v = self.v_proj(pooled).float().unflatten(-1, (H, self.dv))
        beta = torch.sigmoid(self.beta_proj(pooled).float()) / W            # [..., W, H]
        alpha = torch.sigmoid(self.alpha_proj(x.mean(-2)).float()).clamp(max=1.0 - 1e-6)
        q = F.normalize(self.q_proj(x).float().unflatten(-1, (H, self.dk)), dim=-1)
        return k, v, beta, alpha, q

    @staticmethod
    def _update(M: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        """One frame: M [B, H, dv, dk], k [B, W, H, dk], v [B, W, H, dv], β [B, W, H],
        α [B, H] → α M + Σ_w β_w (v_w − α M k_w) k_wᵀ."""
        Ma = M * alpha[..., None, None]
        pred = torch.einsum("bhvk,bwhk->bwhv", Ma, k)
        err = (v - pred) * beta[..., None]
        return Ma + torch.einsum("bwhv,bwhk->bhvk", err, k)

    def _read(self, M: torch.Tensor, q: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """M [B, H, dv, dk], q [B, S, H, dk] → out [B, S, d]."""
        o = torch.einsum("bhvk,bshk->bshv", M, q).flatten(-2)
        return self.out_proj(o.to(dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, S, d] → same shape; frame t reads the memory of frames < t."""
        B, T, S, d = x.shape
        k, v, beta, alpha, q = self._frame_terms(self.norm(x))
        M = self.initial_state(B)
        before = []
        for t in range(T):
            before.append(M)
            M = self._update(M, k[:, t], v[:, t], beta[:, t], alpha[:, t])
        Ms = torch.stack(before, dim=1)                                    # [B, T, H, dv, dk]
        o = torch.einsum("bthvk,btshk->btshv", Ms, q).flatten(-2)
        return x + self.out_proj(o.to(x.dtype))

    def step(self, x: torch.Tensor, M: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """x [B, S, d], M [B, H, dv, dk] (None = empty) → (x', M')."""
        B = x.shape[0]
        if M is None:
            M = self.initial_state(B)
        k, v, beta, alpha, q = self._frame_terms(self.norm(x))
        y = self._read(M, q, x.dtype)
        return x + y, self._update(M, k, v, beta, alpha)


class CoreMemory(nn.Module):
    """The memory sub-layers of a core: after block i, ``gated[i]`` then ``delta[i]``."""

    def __init__(self, d: int, n_blocks: int, gated: str = "", delta: str = "",
                 heads: int = 8, dk: int = 64, writes: int = 4):
        super().__init__()
        self.gated_at = memory_layers(gated, n_blocks)
        self.delta_at = memory_layers(delta, n_blocks)
        self.gated = nn.ModuleDict({str(i): GatedMemory(d) for i in self.gated_at})
        self.delta = nn.ModuleDict({str(i): DeltaMemory(d, heads, dk, dk, writes)
                                    for i in self.delta_at})

    def __bool__(self) -> bool:
        return bool(self.gated_at or self.delta_at)

    def modules_after(self, i: int) -> List[nn.Module]:
        out: List[nn.Module] = []
        if str(i) in self.gated:
            out.append(self.gated[str(i)])
        if str(i) in self.delta:
            out.append(self.delta[str(i)])
        return out

    def ordered(self) -> List[nn.Module]:
        """Every sub-layer in forward order (the order of their states in a cache)."""
        return [m for i in sorted(set(self.gated_at) | set(self.delta_at))
                for m in self.modules_after(i)]

    def initial_states(self, batch: int, tokens: int) -> List[torch.Tensor]:
        return [m.initial_state(batch, tokens) for m in self.ordered()]

    def reset_parameters(self) -> None:
        for m in self.ordered():
            m.reset_parameters()


def split_blocks(blocks: Sequence[nn.Module], cuts: Sequence[int]) -> List[Tuple[int, int]]:
    """[start, end) block ranges ending after each cut (and the tail)."""
    out, start = [], 0
    for c in sorted(set(cuts)):
        out.append((start, c + 1))
        start = c + 1
    if start < len(blocks):
        out.append((start, len(blocks)))
    return out
