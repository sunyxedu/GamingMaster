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
over EVERY micro-batch of the step (under data parallelism also over every
rank, :func:`allreduce_moe_loads`).  In joint multi-game training the
balance is thus enforced on the global mixture, not per game — a per-batch
balancing loss would force every single game to spread uniformly over all
experts and forbid game specialisation.  Padded tokens are excluded from
the loads, and loads are counted once per real forward (never during an
activation-checkpoint recompute).

Routing replay (rollout routing replay: Ma et al. 2025, "Stabilizing MoE
Reinforcement Learning by Aligning Training and Inference Routers", arXiv
2510.11370; the "routing replay" of GSPO, arXiv 2507.18071): top-k selection
is discontinuous in its inputs, so two forwards of the same weights on the
same frame — the RL actor's batched step and the learner's full-sequence
forward, different batch shapes, kernels and precision — can pick different
experts at a near-tie, and the learner's log π_old then differs from the
actor's μ by a discrete jump.  :func:`record_moe_routes` captures the
selected expert set of every token of every MoE call of a forward; a later
forward given those ``routes`` uses them instead of its own top-k, while the
gates are still computed from the current affinities (so the router keeps
its gradient).  At the recording weights this reproduces the recorded
forward exactly, and its θ-gradient equals the unreplayed one wherever the
routing is locally constant (everywhere but the measure-zero tie set): the
replayed surrogate is the unreplayed policy to first order, without its
jumps.
"""

from __future__ import annotations

import contextlib
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
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


class RMSNorm(nn.RMSNorm):
    """nn.RMSNorm that stays on the fused kernel under bf16 autocast: the
    statistics are computed in fp32 and the (fp32) weight is cast to the
    input dtype, instead of the unfused mixed-dtype fallback."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.rms_norm(x.float(), self.normalized_shape, None, self.eps).to(x.dtype)
        return y * self.weight.to(x.dtype)


#: minimum attention-logit cap √D·g_q·g_k the QK-norm scales start at
QK_NORM_INIT_CAP = 8.0


def qk_norm_init_scale(head_dim: int) -> float:
    """Initial value of every QK-norm scale: 1 for head_dim ≥ 64, else
    (8 / √head_dim)^½, so that the cap √D·g_q·g_k starts at
    max(√D, :data:`QK_NORM_INIT_CAP`).

    With unit scales a 32-dim head caps logits at √32 ≈ 5.7: a query exactly
    aligned with its key among ~360 random ones (a go board) gets only ~1/3
    of the attention mass, and sharpening needs g_q·g_k ≈ 2 — thousands of
    steps at lr 3e-4, since each scale moves ≤ ~π/2·lr per AdamATan2 step
    and nothing but the scales can raise the cap (softening stays fast: the
    q/k directions do it).  The price of a higher start is noisier random
    attention at init (logit std = g_q·g_k): for D=32 and 363 keys, cap
    5.7 / 8 / 11.3 give aligned-key mass 0.33 / 0.75 / 0.97 and normalised
    init entropy 0.92 / 0.83 / 0.69 — 8 buys most of the sharpness for a
    modest loss of init entropy."""
    return max(1.0, (QK_NORM_INIT_CAP / math.sqrt(head_dim)) ** 0.5)


class QKNorm(nn.Module):
    """QK-normalisation (``cfg.qk_norm``): per-head RMSNorm of queries and
    keys over head_dim — one [head_dim] learnable scale for q and one for k,
    shared by all heads (the per-head form of ViT-22B, Dehghani et al. 2023,
    and Qwen3; OLMo 2 instead normalises the full-width q/k projection
    before the head split) — applied right before
    ``scaled_dot_product_attention``.  The scales start at
    :func:`qk_norm_init_scale` (1 for heads of ≥ 64 dims, 2^¼ for 32-dim
    ones: initial cap 8).

    After it every q, k row has RMS |g| (‖q̂‖ ≤ √D·max|g_q|), so with SDPA's
    1/√D scaling each attention logit obeys

        |q̂·k̂| / √D ≤ √D · max|g_q| · max|g_k|           (D = head_dim)

    whatever the scale of the activations or of the q/k projection weights.
    Unbounded logit growth — the attention collapsing onto one-hot rows, the
    late loss / grad-norm blow-up of Wortsman et al. 2023 (arXiv 2309.14322)
    — can then only come from the scales themselves, which move at the
    learning rate.

    forward(q, k [..., D]) → (q̂, k̂) in the input dtype (fp32 statistics).
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        g0 = qk_norm_init_scale(head_dim)
        with torch.no_grad():
            self.q_norm.weight.fill_(g0)
            self.k_norm.weight.fill_(g0)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


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

    Two numerically equivalent expert paths (same routing, same gates; they
    differ only in float summation order):

    * sparse — each expert runs on the tokens routed to it.  Finding those
      tokens (``nonzero``) is a host sync per expert on an accelerator, but
      compute is only the routed k/E fraction.  Used whenever autograd is
      on (training, and the eval-mode PPO update), on the CPU (no sync to
      save there, so the routed compute is simply cheaper) and for large
      inference batches.
    * dense — every expert runs on every token (one batched matmul over the
      expert axis) and a gate matrix G [N, E], zero except the top-k
      selected entries, combines them.  No data-dependent shapes, no host
      sync — capturable in a CUDA graph.  Inference only: used in eval mode
      with autograd off, on an accelerator, when tokens × experts is at most
      :attr:`DENSE_MAX_TOKEN_EXPERTS` (batch-1 inference) — or always within
      ``MoEFFN.inference_path("dense")`` (what CUDA-graph capture uses).

    Every autograd path is therefore the sparse one, bit for bit as before.
    """

    #: largest N·E (tokens × experts) the automatic choice sends down the
    #: dense inference path.  Its transient memory is ~N·E·(3·inter + 2·d)
    #: elements and its FLOPs E/k × the routed ones, so large eval batches
    #: keep the sparse path.  Class-level knob (0 disables the automatic
    #: dense path); a batch-1 generals frame is 627 tokens, so it stays
    #: dense up to 104 experts.
    DENSE_MAX_TOKEN_EXPERTS = 65536

    #: "auto" (the rule above) | "dense" | "sparse" — the no-grad inference
    #: path; set it with :meth:`inference_path`, never directly.
    _INFERENCE_PATH = "auto"

    #: tally the routed loads in eval mode too (instance flag, default off):
    #: the Stage-2 RL learner runs its PPO forwards in eval mode (fixed depth,
    #: exactly replayable) but still balances the routing between iterations.
    #: Set it with :func:`moe_load_counting`.
    count_eval_loads = False

    #: the list :func:`record_moe_routes` collects the forwards' selected
    #: expert indices into (instance attribute while recording, else None)
    _route_sink = None

    @classmethod
    @contextlib.contextmanager
    def inference_path(cls, path: str):
        """Within the block, every MoE layer's no-grad eval forward takes
        ``path`` ("dense" | "sparse" | "auto").  Training / autograd always
        stays sparse.  "dense" is the CUDA-graph-capturable path at any batch
        size (``model.cuda_graph.GraphedStep`` captures under it)."""
        if path not in ("auto", "dense", "sparse"):
            raise ValueError(f"MoE inference path must be auto|dense|sparse, got {path!r}")
        prev, cls._INFERENCE_PATH = cls._INFERENCE_PATH, path
        try:
            yield
        finally:
            cls._INFERENCE_PATH = prev

    def expert_path(self, n_tokens: int, device: torch.device) -> str:
        """The expert path ("dense" | "sparse") a forward over ``n_tokens``
        tokens on ``device`` takes in the current mode (training, grad mode,
        :meth:`inference_path`)."""
        if self.training or torch.is_grad_enabled():
            return "sparse"
        if self._INFERENCE_PATH != "auto":
            return self._INFERENCE_PATH
        if (torch.device(device).type != "cpu"
                and n_tokens * self.n_experts <= self.DENSE_MAX_TOKEN_EXPERTS):
            return "dense"
        return "sparse"

    def __init__(self, d: int, n_experts: int, top_k: int, n_shared: int,
                 expansion: float, std: float, out_std: float, bias_clip: float = 0.0):
        super().__init__()
        #: |b_e| bound after centring (0 = unbounded, the DeepSeek-V3 rule)
        self.bias_clip = float(bias_clip)
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

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None,
                route: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``route`` [..., k] integer (the leading dims flatten to x's tokens):
        replay these experts instead of the top-k selection (module
        docstring, routing replay); the gates still come from the current
        affinities."""
        shape = x.shape
        d = shape[-1]
        xf = x.reshape(-1, d)
        N = xf.shape[0]
        affinity = torch.sigmoid(self.centroids(xf).float())            # [N, E]
        if route is None:
            _, top_i = (affinity + self.balance_bias).topk(self.top_k, dim=-1)
        else:
            if route.numel() != N * self.top_k:
                raise ValueError(f"replayed MoE route has {route.numel()} entries, the "
                                 f"forward routes {N} tokens × top-{self.top_k}")
            top_i = route.reshape(N, self.top_k).long()
        if self._route_sink is not None and not _RECOMPUTING:
            self._route_sink.append(top_i)
        top_s = affinity.gather(-1, top_i)
        gates = (top_s / top_s.sum(-1, keepdim=True)).to(xf.dtype)       # [N, k]

        if (self.training or self.count_eval_loads) and not _RECOMPUTING:
            with torch.no_grad():
                sel = top_i
                if valid is not None:
                    tok_valid = valid.reshape(-1, 1).expand(shape[0], shape[1]).reshape(-1)
                    sel = top_i[tok_valid]
                self.load += torch.bincount(sel.reshape(-1),
                                            minlength=self.n_experts).float()

        if self.expert_path(N, xf.device) == "dense":
            out = self._dense_experts(xf, top_i, gates)
        else:
            out = self._sparse_experts(xf, top_i, gates)
        for sh in self.shared:
            out = out + sh(xf)
        return out.reshape(shape)

    def _sparse_experts(self, xf: torch.Tensor, top_i: torch.Tensor,
                        gates: torch.Tensor) -> torch.Tensor:
        """Σ_selected g·FFN_e(x), each expert on its routed tokens only.

        The expert weights are cast once and ``unbind``-ed into per-expert
        views: indexing ``w[e]`` in the loop instead makes autograd build a
        full-size zero gradient per expert and add the E of them up (memory
        traffic ∝ E² per layer).  Each element still receives exactly one
        gradient contribution (or none), so the result is unchanged."""
        out = xf.new_zeros(xf.shape)
        w_gate_up = self.w_gate_up.to(xf.dtype).unbind(0)
        w_down = self.w_down.to(xf.dtype).unbind(0)
        for e in range(self.n_experts):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            h = xf.index_select(0, rows)
            g, u = (h @ w_gate_up[e]).chunk(2, dim=-1)
            h = (F.silu(g) * u) @ w_down[e]
            out.index_add_(0, rows, h * gates[rows, slot].unsqueeze(-1))
        return out

    def _dense_experts(self, xf: torch.Tensor, top_i: torch.Tensor,
                       gates: torch.Tensor) -> torch.Tensor:
        """Same sum, sync-free: all experts on all tokens, combined with the
        gate matrix G [N, E] (the top-k gates scattered into zeros).

        Unselected experts are dropped with ``where`` rather than a 0 gate,
        so a non-finite activation of an expert the token was NOT routed to
        cannot leak into the output (0·inf = NaN) — the sparse path's forward
        semantics.  Forward only: its backward would form exactly that 0·inf
        in w_down.grad, which is why :meth:`expert_path` never picks it with
        autograd on.  Weight/activation dtypes mirror :meth:`_sparse_experts`
        op for op.
        """
        N, E = xf.shape[0], self.n_experts
        sel = torch.zeros(N, E, dtype=torch.bool, device=xf.device).scatter_(1, top_i, True)
        G = torch.zeros(N, E, dtype=gates.dtype, device=xf.device).scatter(1, top_i, gates)
        h = torch.matmul(xf, self.w_gate_up.to(xf.dtype))                 # [E, N, 2I]
        g, u = h.chunk(2, dim=-1)
        y = torch.matmul(F.silu(g) * u, self.w_down.to(xf.dtype))          # [E, N, d]
        y = y * G.t().unsqueeze(-1)
        return torch.where(sel.t().unsqueeze(-1), y, 0.0).sum(0)

    @torch.no_grad()
    def update_balance(self, gamma: float) -> Optional[torch.Tensor]:
        """b_e += γ·sign(mean load − load_e) over the tallied step, then
        reset the tally.  Returns the loads used (None if nothing routed).

        Top-k selection is invariant to adding a constant to every b_e, but
        Σ_e sign(mean − load_e) ≠ 0 in general, so the raw rule makes b drift
        in common mode without bound.  b is therefore re-centred after each
        update (routing-neutral).  With ``bias_clip`` > 0, b is also bounded to
        [−clip, clip] so the bias can never swamp the sigmoid affinities
        (range 0..1) and routing stays input-dependent."""
        load = self.load.clone()
        self.load.zero_()
        if load.sum() <= 0:
            return None
        if not any(p.requires_grad for p in self.parameters()):
            return None     # a frozen layer (e.g. frozen core) keeps its routing
        self.balance_bias += gamma * torch.sign(load.mean() - load)
        self.balance_bias -= self.balance_bias.mean()      # routing-neutral centring
        if self.bias_clip > 0:
            self.balance_bias.clamp_(-self.bias_clip, self.bias_clip)
        return load


class CoreBlock(nn.Module):
    """Sandwich-normalised transformer block (arXiv 2502.05171 §3):

        x = x + n2(Attn(n1(x)));   x = x + n4(FFN(n3(x)))

    Non-causal multi-head attention over one frame's tokens (positions are
    already in the input).  With ``cfg.qk_norm`` the per-head q and k are
    RMS-normalised (:class:`QKNorm`) before the dot product; the sandwich
    norms stay.  Configs without the field (old checkpoints) build no
    ``qk_norm`` submodule, so their state dicts load strictly."""

    def __init__(self, cfg: ModelConfig, out_std: float, moe: bool):
        super().__init__()
        d = cfg.d_model
        std = math.sqrt(2.0 / (5.0 * d))
        self.n_heads = cfg.n_heads
        self.qkv = CoreLinear(d, 3 * d, std)
        self.o = CoreLinear(d, d, out_std)
        self.norms = nn.ModuleList([RMSNorm(d, eps=1e-6) for _ in range(4)])
        self.ffn = (MoEFFN(d, cfg.moe_experts, cfg.moe_top_k, cfg.moe_shared_experts,
                           cfg.moe_expansion, std, out_std,
                           bias_clip=getattr(cfg, "moe_bias_clip", 0.0))
                    if moe else SwiGLUFFN(d, cfg.core_expansion, std, out_std))
        # registered last: parameter order of qk_norm=False blocks unchanged
        self.qk_norm = (QKNorm(d // cfg.n_heads, eps=1e-6)
                        if getattr(cfg, "qk_norm", False) else None)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None,
                route: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``route`` [B, S, k]: the MoE layer's replayed experts (MoE blocks only)."""
        B, S, d = x.shape
        h = self.norms[0](x)
        q, k, v = self.qkv(h).view(B, S, 3, self.n_heads, d // self.n_heads).permute(
            2, 0, 3, 1, 4).unbind(0)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.norms[1](self.o(a.permute(0, 2, 1, 3).reshape(B, S, d)))
        if route is None:
            return x + self.norms[3](self.ffn(self.norms[2](x), valid))
        if not isinstance(self.ffn, MoEFFN):
            raise ValueError("a replayed route was given to a dense (non-MoE) block")
        return x + self.norms[3](self.ffn(self.norms[2](x), valid, route))


def _run_blocks(blocks: Sequence[nn.Module], x: torch.Tensor,
                valid: Optional[torch.Tensor], routes: Optional[torch.Tensor] = None,
                first: int = 0) -> torch.Tensor:
    """The blocks in order.  ``routes`` [N, C, S, k]: the replayed routing of
    a whole core forward, block i taking call ``first + i`` (an explicit
    argument of the checkpointed call, so the backward recompute replays the
    same experts)."""
    for i, blk in enumerate(blocks):
        args = (x, valid) if routes is None else (x, valid, routes[:, first + i])
        if torch.is_grad_enabled() and x.requires_grad:
            x = ckpt(blk, *args)
        else:
            x = blk(*args)
    return x


def _check_routes(routes: Optional[torch.Tensor], x: torch.Tensor, calls: int) -> None:
    if routes is None:
        return
    if routes.dim() != 4 or routes.shape[0] != x.shape[0] or routes.shape[1] != calls \
            or routes.shape[2] != x.shape[1]:
        raise ValueError(
            f"replayed MoE routes {tuple(routes.shape)} do not fit this forward: expected "
            f"[{x.shape[0]} rows, {calls} MoE calls, {x.shape[1]} tokens, top-k] (the "
            f"recording forward must use the same core depth)")


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
        self.final_norm = RMSNorm(cfg.d_model, eps=1e-6)

    def default_iters(self) -> int:
        return 1

    def moe_calls(self, steps: Optional[int] = None) -> int:
        """MoE layer calls of one forward (the routes' C axis): one per block."""
        return len(self.blocks) if self.blocks and isinstance(self.blocks[0].ffn, MoEFFN) else 0

    def forward(self, x: torch.Tensor, steps: Optional[int] = None,
                valid: Optional[torch.Tensor] = None,
                routes: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        """``routes`` [N, C, S, k]: replay a recorded routing (:func:`record_moe_routes`)."""
        _check_routes(routes, x, self.moe_calls())
        return self.final_norm(_run_blocks(self.blocks, x, valid, routes)), 1


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
        self.final_norm = RMSNorm(d, eps=1e-6)
        self.state_std = math.sqrt(2.0 / 5.0)
        self._fixed_state: Dict[Tuple[int, int, str, torch.dtype], torch.Tensor] = {}

    def default_iters(self) -> int:
        return int(self.cfg.rd_eval_iters or self.cfg.rd_mean_iters)

    def moe_calls(self, steps: Optional[int] = None) -> int:
        """MoE layer calls of one forward at depth ``steps`` (the routes' C
        axis), in call order: prelude, r × recurrent, coda."""
        if not _moe_on(self.cfg):
            return 0
        r = max(1, int(steps)) if steps is not None else self.default_iters()
        return len(self.prelude) + r * len(self.recurrent) + len(self.coda)

    def prepare_inference(self, S: int, device: torch.device,
                          dtype: torch.dtype = torch.float32) -> None:
        """Materialise the fixed inference state for S tokens on ``device``.

        :meth:`fixed_state` builds it lazily on the CPU and copies it over on
        first use — a host→device copy that cannot happen inside a CUDA
        graph capture.  Call this (or run one warm-up step) before capture.
        """
        self.fixed_state(S, torch.empty(0, self.cfg.d_model, device=device, dtype=dtype))

    def fixed_state(self, S: int, like: torch.Tensor) -> torch.Tensor:
        """One deterministic draw from the s_0 distribution, [S, d], cached
        per (S, d, device, dtype) so the eval path is a pure lookup."""
        key = (S, like.shape[-1], str(like.device), like.dtype)
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
            self._fixed_state[key] = z.to(like.device).to(like.dtype)
        return self._fixed_state[key]

    def initial_state(self, e: torch.Tensor) -> torch.Tensor:
        if self.training:
            return trunc_normal_init_(torch.empty_like(e), std=self.state_std)
        return self.fixed_state(e.shape[1], e).expand_as(e)

    def forward(self, x: torch.Tensor, steps: Optional[int] = None,
                valid: Optional[torch.Tensor] = None,
                routes: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        """``routes`` [N, C, S, k]: replay a recorded routing (needs a fixed
        ``steps`` or eval mode: C depends on the depth)."""
        cfg = self.cfg
        if steps is not None:
            r = max(1, int(steps))
        elif self.training:
            if routes is not None:
                raise ValueError("replaying MoE routes needs a fixed depth (steps=...)")
            r = sample_recurrence(cfg.rd_mean_iters, max_iters=cfg.rd_max_iters)
        else:
            r = self.default_iters()
        _check_routes(routes, x, self.moe_calls(r))
        P, R = len(self.prelude), len(self.recurrent)
        e = _run_blocks(self.prelude, x, valid, routes, 0)
        s = self.initial_state(e)
        k = min(r, cfg.rd_backprop_iters) if torch.is_grad_enabled() else r
        with torch.no_grad():
            for i in range(r - k):
                s = _run_blocks(self.recurrent, self.adapter(torch.cat([s, e], -1)), valid,
                                routes, P + i * R)
        for i in range(r - k, r):
            s = _run_blocks(self.recurrent, self.adapter(torch.cat([s, e], -1)), valid,
                            routes, P + i * R)
        return self.final_norm(_run_blocks(self.coda, s, valid, routes, P + r * R)), r


def moe_layers(module: nn.Module):
    return [m for m in module.modules() if isinstance(m, MoEFFN)]


@contextlib.contextmanager
def moe_load_counting(module: nn.Module, enabled: bool = True):
    """Within the block every MoE layer of ``module`` tallies its routed
    loads also in eval mode (training mode always tallies).  Only this
    module's layers: a frozen reference or opponent network forwarding in
    the same block is untouched.  Checkpoint recomputes stay excluded."""
    layers = moe_layers(module)
    prev = [m.count_eval_loads for m in layers]
    for m in layers:
        m.count_eval_loads = bool(enabled)
    try:
        yield
    finally:
        for m, p in zip(layers, prev):
            m.count_eval_loads = p


@contextlib.contextmanager
def record_moe_routes(module: nn.Module):
    """Within the block every MoE layer of ``module`` appends the expert
    indices it selected for each forward's tokens ([N_tokens, k] long) to
    the yielded list, in call order — one core forward's list stacks into
    the ``routes`` a later forward replays (:func:`stack_routes`).
    Checkpoint recomputes are not recorded."""
    layers = moe_layers(module)
    sink: list = []
    prev = [m._route_sink for m in layers]
    for m in layers:
        m._route_sink = sink
    try:
        yield sink
    finally:
        for m, p in zip(layers, prev):
            m._route_sink = p


def route_dtype(n_experts: int) -> torch.dtype:
    """The narrowest integer dtype that holds expert indices 0..n_experts-1
    (routes are stored per frame for a whole RL iteration)."""
    if n_experts <= 256:
        return torch.uint8
    return torch.int16 if n_experts <= 32767 else torch.int32


def stack_routes(calls: Sequence[torch.Tensor], rows: int, tokens: int,
                 n_experts: int) -> torch.Tensor:
    """A recorded core forward (the list of :func:`record_moe_routes`, each
    call [rows·tokens, k]) → routes [rows, C, tokens, k] in :func:`route_dtype`."""
    return torch.stack([c.view(rows, tokens, -1) for c in calls], 1).to(route_dtype(n_experts))


@torch.no_grad()
def allreduce_moe_loads(module: nn.Module) -> None:
    """Sum every MoE layer's step load tally across the ranks of the default
    process group (one all-reduce), so the next :func:`update_moe_balance`
    sees the GLOBAL load of all ranks' micro-batches and moves the balance
    bias identically on every rank.  No-op without a process group or with
    one rank.  Call on every rank, right before ``update_moe_balance``."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return
    layers = moe_layers(module)
    if not layers:
        return
    flat = torch.cat([m.load for m in layers])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    for m, ld in zip(layers, flat.split([m.load.numel() for m in layers])):
        m.load.copy_(ld)


@torch.no_grad()
def update_moe_balance(module: nn.Module, gamma: float) -> Optional[torch.Tensor]:
    """Apply the aux-loss-free bias update to every MoE layer of ``module``
    (call once per optimizer step, after all its micro-batches).  Returns
    the per-layer relative loads [L, E] (load / mean load) for logging."""
    loads = [m.update_balance(gamma) for m in moe_layers(module)]
    loads = [ld / ld.mean() for ld in loads if ld is not None]
    return torch.stack(loads) if loads else None
