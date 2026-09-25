"""Stage-2 cores: a plain deep transformer and a recurrent-depth transformer,
both with optional Mixture-of-Experts feed-forward layers.

Recurrent depth — Geiping et al. 2025, "Scaling up Test-Time Compute with
Latent Reasoning: A Recurrent Depth Approach" (arXiv 2502.05171):

    e   = Prelude(x)                          l_P blocks
    s_0 ~ TruncNormal(0, σ_s² = 2/5)          i.i.d. per element
    s_i = R(A([s_{i-1}; e])),  i = 1..r       A: R^{2d} → R^d, R: l_R blocks
    y   = norm(Coda(s_r))                     l_C blocks

* Sandwich normalisation in every block (the paper: "required to train the
  recurrence at scale").
* Training draws r per forward from the log-normal Poisson
  τ ~ N(log r̄ − σ²/2, σ), r = Poisson(e^τ) + 1, σ = 1/2, and backpropagates
  through the last k iterations only (truncated BPTT; memory independent
  of r).  The input e is re-injected at every iteration, so the iteration
  cannot drift away from the observation ("recall").
* Inference uses a FIXED r and a FIXED initial state: one draw from the
  same s_0 distribution, generated deterministically from (tokens, width)
  and shared by every row.  Inference is therefore in-distribution AND
  deterministic, which keeps RL rollouts exactly replayable by the update.

MoE — DeepSeek-V3 routing (arXiv 2412.19437 §2.1.1-2.1.2):

    s_{t,e}  = sigmoid(u_t · c_e)                        affinity
    top-k selection on s_{t,e} + b_e                     b: balance bias
    g_{t,e}  = s_{t,e} / Σ_{selected} s_{t,e'}           gates (bias-free)
    y_t      = Σ_shared FFN(u_t) + Σ_selected g_{t,e} FFN_e(u_t)

Auxiliary-loss-free balancing: b is not trained by gradient; after every
optimizer step b_e ← b_e + γ·sign(mean load − load_e), with the loads summed
over EVERY micro-batch of the step.  In joint multi-game training the
balance is thus enforced on the global mixture, not per game — a per-batch
balancing loss would force every single game to spread uniformly over all
experts and forbid game specialisation.  Padded tokens are excluded from
the loads, and loads are counted once per real forward (never during an
activation-checkpoint recompute).
"""

from __future__ import annotations

import contextlib
import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.components.trm import _find_multiple, trunc_normal_init_
from model.config import ModelConfig

# True while torch.utils.checkpoint recomputes a forward during backward:
# MoE load counting must not see the same tokens twice.
_RECOMPUTING = False


@contextlib.contextmanager
def _recompute_flag():
    global _RECOMPUTING
    prev, _RECOMPUTING = _RECOMPUTING, True
    try:
        yield
    finally:
        _RECOMPUTING = prev


def _ckpt_contexts():
    return contextlib.nullcontext(), _recompute_flag()


def ckpt(fn, *args):
    """Non-reentrant activation checkpoint whose recompute is flagged."""
    return checkpoint(fn, *args, use_reentrant=False, context_fn=_ckpt_contexts)


class CoreLinear(nn.Module):
    """Bias-free linear with truncated-normal init of a chosen std.  Not an
    ``nn.Linear`` so GameNetwork's blanket std-0.02 init leaves it alone."""

    def __init__(self, d_in: int, d_out: int, std: float):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(d_out, d_in), std=std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


def _inter(d: int, expansion: float) -> int:
    return _find_multiple(round(expansion * d * 2 / 3), 64)


class SwiGLUFFN(nn.Module):
    def __init__(self, d: int, expansion: float, std: float, out_std: float):
        super().__init__()
        inter = _inter(d, expansion)
        self.gate_up = CoreLinear(d, 2 * inter, std)
        self.down = CoreLinear(inter, d, out_std)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class MoEFFN(nn.Module):
    """DeepSeek-V3 MoE feed-forward (see module docstring).

    forward(x [B, S, d], valid [B] bool | None) → y [B, S, d]
    """

    def __init__(self, d: int, n_experts: int, top_k: int, n_shared: int,
                 expansion: float, std: float, out_std: float):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"moe_top_k={top_k} must be in [1, moe_experts={n_experts}]")
        self.n_experts, self.top_k = n_experts, top_k
        inter = _inter(d, expansion)
        self.centroids = CoreLinear(d, n_experts, std=d ** -0.5)
        self.w_gate_up = nn.Parameter(
            trunc_normal_init_(torch.empty(n_experts, d, 2 * inter), std=std))
        self.w_down = nn.Parameter(
            trunc_normal_init_(torch.empty(n_experts, inter, d), std=out_std))
        self.shared = nn.ModuleList(
            [SwiGLUFFN(d, expansion, std, out_std) for _ in range(n_shared)])
        # balance bias (not a gradient parameter) and the step's load tally
        self.register_buffer("balance_bias", torch.zeros(n_experts))
        self.register_buffer("load", torch.zeros(n_experts), persistent=False)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        shape = x.shape
        d = shape[-1]
        xf = x.reshape(-1, d)
        N = xf.shape[0]
        affinity = torch.sigmoid(self.centroids(xf).float())            # [N, E]
        _, top_i = (affinity + self.balance_bias).topk(self.top_k, dim=-1)
        top_s = affinity.gather(-1, top_i)
        gates = (top_s / top_s.sum(-1, keepdim=True)).to(xf.dtype)       # [N, k]

        if self.training and not _RECOMPUTING:
            with torch.no_grad():
                sel = top_i
                if valid is not None:
                    tok_valid = valid.reshape(-1, 1).expand(shape[0], shape[1]).reshape(-1)
                    sel = top_i[tok_valid]
                self.load += torch.bincount(sel.reshape(-1),
                                            minlength=self.n_experts).float()

        out = xf.new_zeros(N, d)
        for e in range(self.n_experts):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            h = xf.index_select(0, rows)
            g, u = (h @ self.w_gate_up[e].to(h.dtype)).chunk(2, dim=-1)
            h = (F.silu(g) * u) @ self.w_down[e].to(h.dtype)
            out.index_add_(0, rows, h * gates[rows, slot].unsqueeze(-1))
        for sh in self.shared:
            out = out + sh(xf)
        return out.reshape(shape)

    @torch.no_grad()
    def update_balance(self, gamma: float) -> Optional[torch.Tensor]:
        """b_e += γ·sign(mean load − load_e) over the tallied step, then
        reset the tally.  Returns the loads used (None if nothing routed)."""
        load = self.load.clone()
        self.load.zero_()
        if load.sum() <= 0:
            return None
        if not any(p.requires_grad for p in self.parameters()):
            return None     # a frozen layer (e.g. frozen core) keeps its routing
        self.balance_bias += gamma * torch.sign(load.mean() - load)
        return load


class CoreBlock(nn.Module):
    """Sandwich-normalised transformer block (arXiv 2502.05171 §3):

        x = x + n2(Attn(n1(x)));   x = x + n4(FFN(n3(x)))

    Non-causal multi-head attention over one frame's tokens (positions are
    already in the input)."""

    def __init__(self, cfg: ModelConfig, out_std: float, moe: bool):
        super().__init__()
        d = cfg.d_model
        std = math.sqrt(2.0 / (5.0 * d))
        self.n_heads = cfg.n_heads
        self.qkv = CoreLinear(d, 3 * d, std)
        self.o = CoreLinear(d, d, out_std)
        self.norms = nn.ModuleList([nn.RMSNorm(d, eps=1e-6) for _ in range(4)])
        self.ffn = (MoEFFN(d, cfg.moe_experts, cfg.moe_top_k, cfg.moe_shared_experts,
                           cfg.moe_expansion, std, out_std)
                    if moe else SwiGLUFFN(d, cfg.core_expansion, std, out_std))

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, d = x.shape
        h = self.norms[0](x)
        q, k, v = self.qkv(h).view(B, S, 3, self.n_heads, d // self.n_heads).permute(
            2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.norms[1](self.o(a.permute(0, 2, 1, 3).reshape(B, S, d)))
        return x + self.norms[3](self.ffn(self.norms[2](x), valid))


def _run_blocks(blocks: Iterable[nn.Module], x: torch.Tensor,
                valid: Optional[torch.Tensor]) -> torch.Tensor:
    for blk in blocks:
        if torch.is_grad_enabled() and x.requires_grad:
            x = ckpt(blk, x, valid)
        else:
            x = blk(x, valid)
    return x


def _out_std(cfg: ModelConfig, effective_layers: float) -> float:
    """Output-projection init std √(1 / (5·d·L_eff)) (arXiv 2502.05171)."""
    return math.sqrt(1.0 / (5.0 * cfg.d_model * max(effective_layers, 1.0)))


def _moe_on(cfg: ModelConfig) -> bool:
    return getattr(cfg, "moe_experts", 0) > 0


class DeepCore(nn.Module):
    """``core_layers`` untied sandwich blocks; forward → (y, depth=1)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        out_std = _out_std(cfg, cfg.core_layers)
        self.blocks = nn.ModuleList(
            [CoreBlock(cfg, out_std, _moe_on(cfg)) for _ in range(cfg.core_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)

    def default_iters(self) -> int:
        return 1

    def forward(self, x: torch.Tensor, steps: Optional[int] = None,
                valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        return self.final_norm(_run_blocks(self.blocks, x, valid)), 1


def sample_recurrence(mean: float, sigma: float = 0.5,
                      max_iters: Optional[int] = None) -> int:
    """Log-normal Poisson (arXiv 2502.05171 eq. 1): τ ~ N(log r̄ − σ²/2, σ),
    r = Poisson(e^τ) + 1, so E[r] = r̄ + 1."""
    tau = torch.normal(torch.tensor(math.log(max(mean, 1e-6)) - 0.5 * sigma ** 2),
                       torch.tensor(float(sigma)))
    r = int(torch.poisson(tau.exp()).item()) + 1
    if max_iters:
        r = min(r, max_iters)
    return max(1, r)


class RecurrentDepthCore(nn.Module):
    """Prelude → r × (adapter + recurrent blocks) → coda; forward → (y, r)."""

    STATE_SEED = 20250207          # fixed inference state: one s_0 draw

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        eff = cfg.rd_prelude_layers + cfg.rd_recurrent_layers * cfg.rd_mean_iters \
            + cfg.rd_coda_layers
        out_std = _out_std(cfg, eff)
        moe = _moe_on(cfg)
        self.prelude = nn.ModuleList(
            [CoreBlock(cfg, out_std, moe) for _ in range(cfg.rd_prelude_layers)])
        self.adapter = CoreLinear(2 * d, d, math.sqrt(2.0 / (5.0 * d)))
        self.recurrent = nn.ModuleList(
            [CoreBlock(cfg, out_std, moe) for _ in range(cfg.rd_recurrent_layers)])
        self.coda = nn.ModuleList(
            [CoreBlock(cfg, out_std, moe) for _ in range(cfg.rd_coda_layers)])
        self.final_norm = nn.RMSNorm(d, eps=1e-6)
        self.state_std = math.sqrt(2.0 / 5.0)
        self._fixed_state: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def default_iters(self) -> int:
        return int(self.cfg.rd_eval_iters or self.cfg.rd_mean_iters)

    def fixed_state(self, S: int, like: torch.Tensor) -> torch.Tensor:
        """One deterministic draw from the s_0 distribution, [S, d]."""
        key = (S, like.shape[-1], str(like.device))
        if key not in self._fixed_state:
            g = torch.Generator().manual_seed(self.STATE_SEED + 7919 * S)
            z = torch.empty(S, like.shape[-1])
            # truncated normal via rejection-free inverse-CDF on a seeded generator
            sqrt2 = math.sqrt(2.0)
            lo, hi = math.erf(-2.0 / sqrt2), math.erf(2.0 / sqrt2)
            z.uniform_(lo, hi, generator=g).erfinv_().mul_(sqrt2)
            # Var of N(0,1) truncated to [-2, 2] = 1 − 2·2·φ(2) / (Φ(2) − Φ(−2)),
            # and Φ(2) − Φ(−2) = (hi − lo) / 2 in erf space
            z.mul_(self.state_std / math.sqrt(
                1 - (4 * math.exp(-2) / math.sqrt(2 * math.pi)) / ((hi - lo) / 2)))
            self._fixed_state[key] = z.to(like.device)
        return self._fixed_state[key].to(like.dtype)

    def initial_state(self, e: torch.Tensor) -> torch.Tensor:
        if self.training:
            return trunc_normal_init_(torch.empty_like(e), std=self.state_std)
        return self.fixed_state(e.shape[1], e).expand_as(e)

    def forward(self, x: torch.Tensor, steps: Optional[int] = None,
                valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        cfg = self.cfg
        if steps is not None:
            r = max(1, int(steps))
        elif self.training:
            r = sample_recurrence(cfg.rd_mean_iters, max_iters=cfg.rd_max_iters)
        else:
            r = self.default_iters()
        e = _run_blocks(self.prelude, x, valid)
        s = self.initial_state(e)
        k = min(r, cfg.rd_backprop_iters) if torch.is_grad_enabled() else r
        with torch.no_grad():
            for _ in range(r - k):
                s = _run_blocks(self.recurrent, self.adapter(torch.cat([s, e], -1)), valid)
        for _ in range(k):
            s = _run_blocks(self.recurrent, self.adapter(torch.cat([s, e], -1)), valid)
        return self.final_norm(_run_blocks(self.coda, s, valid)), r


def moe_layers(module: nn.Module):
    return [m for m in module.modules() if isinstance(m, MoEFFN)]


@torch.no_grad()
def update_moe_balance(module: nn.Module, gamma: float) -> Optional[torch.Tensor]:
    """Apply the aux-loss-free bias update to every MoE layer of ``module``
    (call once per optimizer step, after all its micro-batches).  Returns
    the per-layer relative loads [L, E] (load / mean load) for logging."""
    loads = [m.update_balance(gamma) for m in moe_layers(module)]
    loads = [ld / ld.mean() for ld in loads if ld is not None]
    return torch.stack(loads) if loads else None
