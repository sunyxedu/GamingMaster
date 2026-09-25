"""Game abstraction layer — the boundary between the game-agnostic core
(Stage 1 spatiotemporal encoder + Stage 2 TRM) and per-game modules.

A game plugs into the core through exactly two modules:

  ObservationEncoder   raw per-frame observation dict → token grid
                       [B, T, S_game, d].  The token layout is FIXED per
                       game (padded slots for variable-count entities) so
                       the core's positional embeddings and the TRM's
                       decision-token index stay static.

  ActionDecoder        pondered tokens (z_H) → action logits, action
                       sampling, and log-prob/entropy evaluation for the
                       stored-action replay needed by PPO.

Everything in between — decision tokens, Stage-1 blocks, TRM, value head —
is shared by all games and lives in ``model.game_network.GameNetwork``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared sampling / masking utilities
# ---------------------------------------------------------------------------

_INVALID_PROBS = "probability tensor contains either `inf`, `nan` or element < 0, " \
                 "or a row summing to 0"


def check_distribution(probs: torch.Tensor) -> None:
    """``torch.multinomial``'s input checks (no NaN / inf / negative entry,
    no all-zero row), without its host sync where the device allows.

    CPU and CUDA: ``torch._assert_async`` — on CUDA a device-side assert
    (no host read, capturable in a CUDA graph; a failure aborts the CUDA
    context at the next sync).  Other devices (MPS has no async assert):
    a host-side check, the sync multinomial always paid there.
    """
    lo, hi = probs.aminmax()
    ok = (lo >= 0) & (hi < float("inf")) & (probs.sum(-1) != 0).all()
    if probs.device.type in ("cpu", "cuda"):
        torch._assert_async(ok, _INVALID_PROBS)
    elif not ok:
        raise RuntimeError(_INVALID_PROBS)


def categorical_sample(probs: torch.Tensor) -> torch.Tensor:
    """Draw one index per row from ``probs`` [..., N] → [...].

    Exactly ``torch.multinomial(probs, 1).squeeze(-1)`` — the same
    exponential race (argmax of p / q, q ~ Exp(1)) torch uses internally
    for a single draw, consuming the same random numbers, so samples are
    bit-identical under the same generator state.  Invalid input (NaN / inf
    / negative entries, e.g. a head whose logits overflowed) still fails
    loudly like multinomial, but via :func:`check_distribution`'s async
    assert instead of multinomial's ``.item()`` syncs, which also made the
    call impossible to capture in a CUDA graph.
    """
    check_distribution(probs)
    return (probs / torch.empty_like(probs).exponential_(1)).argmax(-1)


def safe_masked_logits(
    logits: torch.Tensor, mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a legality mask to logits, guarding degenerate rows.

    - masked positions become -inf,
    - rows where EVERYTHING is masked fall back to index 0 (uniform-ish),
    - rows containing NaN (bf16 overflow) fall back to uniform.

    Always returns a detached float32 copy — callers use it for sampling
    and log-prob bookkeeping, never for gradient flow.  Branch-free (the
    fallbacks are masked writes that leave healthy rows untouched): no
    host sync, CUDA-graph capturable.
    """
    safe = logits.detach().clone().float()
    if mask is not None:
        safe = safe.masked_fill(~mask, float("-inf"))
        all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
        # Column-slice write, NOT `safe[all_masked, ..., 0]`: mixing a
        # boolean mask with a trailing index is only valid for 1-D masks,
        # and callers pass [B, T, N] logits too.
        safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
    nan_rows = safe.isnan().any(dim=-1, keepdim=True)
    return safe.masked_fill(nan_rows, 0.0)


def masked_sample(
    logits: torch.Tensor, mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample category indices from (optionally masked) logits.  [B, N] → [B]."""
    with torch.no_grad():
        safe = safe_masked_logits(logits, mask)
        probs = F.softmax(safe, dim=-1)
        return categorical_sample(probs)


def masked_log_prob(
    logits: torch.Tensor, index: torch.Tensor, mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable log-prob of ``index`` under (optionally masked) logits.

    Unlike :func:`safe_masked_logits` this keeps the gradient path; only the
    degenerate all-masked guard is applied.  Always evaluates in float32 so
    rollout bookkeeping and PPO replay see the same distribution regardless
    of the logits' compute dtype.
    """
    logits = logits.float()
    if mask is not None:
        safe = logits.masked_fill(~mask, float("-inf"))
        all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
        # Column-slice write — see safe_masked_logits (branch-free: a no-op
        # on rows with any legal entry).
        safe = safe.clone()
        safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
    else:
        safe = logits
    return F.log_softmax(safe, dim=-1).gather(
        -1, index.unsqueeze(-1),
    ).squeeze(-1)


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of a categorical distribution given (possibly -inf) logits.

    The −inf entries must be clamped BEFORE the p·log p product:
    ``nan_to_num`` on the product cleans only the forward — mul's
    backward still computes 0·(−inf) = NaN, and log_softmax's backward
    spreads it to the unmasked positions.  exp underflows to 0 long
    before the clamp value, so the entropy itself is unchanged."""
    log_p = F.log_softmax(logits, dim=-1).clamp_min(-1e4)
    return -(torch.exp(log_p) * log_p).nan_to_num(0.0).sum(-1)


def bernoulli_log_prob_entropy(
    logit: torch.Tensor, value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(log-prob, entropy) of a Bernoulli given a single logit and bool value.

    Evaluated in float32 so rollout and PPO replay agree bit-for-bit."""
    logit = logit.float()
    lp = torch.where(value, F.logsigmoid(logit), F.logsigmoid(-logit))
    p = torch.sigmoid(logit)
    ent = -(p * F.logsigmoid(logit) + (1 - p) * F.logsigmoid(-logit))
    return lp, ent


# ---------------------------------------------------------------------------
# Reusable head building blocks
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Linear → ReLU → Linear.  The standard 2-layer head used everywhere."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointerHead(nn.Module):
    """Score every token against zero or more query vectors.

    With no queries this degenerates to a per-token scorer (like the
    generals SourceHead); with queries the query vectors are broadcast
    over the token axis and concatenated before scoring (like "which
    target does this actor attack").

        tokens  [B, S, d]
        queries each [B, d]
        →       [B, S] logits (raw, unmasked)
    """

    def __init__(self, d_model: int, n_queries: int = 0, hidden: Optional[int] = None):
        super().__init__()
        self.n_queries = n_queries
        in_dim = d_model * (1 + n_queries)
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: torch.Tensor, *queries: torch.Tensor) -> torch.Tensor:
        if len(queries) != self.n_queries:
            raise ValueError(
                f"PointerHead expects {self.n_queries} queries, got {len(queries)}"
            )
        if queries:
            S = tokens.shape[-2]
            expanded = [q.unsqueeze(-2).expand(*tokens.shape[:-1], q.shape[-1])
                        for q in queries]
            tokens = torch.cat([tokens, *expanded], dim=-1)
        return self.net(tokens).squeeze(-1)


# ---------------------------------------------------------------------------
# Encoder / decoder interfaces
# ---------------------------------------------------------------------------

class ObservationEncoder(nn.Module, ABC):
    """Maps raw observations to the fixed-size token sequence the core consumes.

    Contract:
      - ``n_tokens`` is a constant: the core appends its two decision tokens
        (HL at index ``n_tokens``, LL at ``n_tokens + 1``) after the game
        tokens, and the TRM reads q_halt from the HL index.
      - Variable-count entities (cards in hand, units on the map) must be
        padded into fixed slots; encode a "present" flag into the features
        rather than shrinking the sequence.
    """

    #: number of game tokens S_game produced per frame
    n_tokens: int

    @abstractmethod
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of frame sequences.

        Args:
            obs:        game-specific tensors, each shaped [B, T, ...]
            frame_mask: [B, T] bool, True = real frame.  Encoders with
                        batch statistics (BatchNorm) must exclude padded
                        frames from those statistics.
        Returns:
            tokens [B, T, n_tokens, d_model]
        """

    def encode_step(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a single frame [B, ...] → [B, n_tokens, d].

        Default: unsqueeze a T=1 axis and reuse :meth:`forward`.
        """
        seq = {k: v.unsqueeze(1) for k, v in obs.items()}
        return self.forward(seq).squeeze(1)

    @abstractmethod
    def positional_embedding(self) -> torch.Tensor:
        """Learnable positional embedding for the game tokens: [1, n_tokens, d]."""

    # -- optional world-model support ------------------------------------

    @property
    def supports_next_obs(self) -> bool:
        """Whether this encoder can host the self-supervised next-observation
        objective (``cfg.predict_next_obs``)."""
        return False

    def next_obs_outputs(
        self, belief_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict the next frame's observation from Stage-1 belief tokens
        [B, T, n_tokens, d].  Only called when :attr:`supports_next_obs`."""
        raise NotImplementedError


class ActionDecoder(nn.Module, ABC):
    """Turns pondered tokens into an action distribution.

    Most game action spaces are hierarchical (pick an actor, then an
    argument conditioned on it).  The decoder owns that structure end to
    end: logits, sampling, and log-prob/entropy re-evaluation must all
    agree so PPO ratios are exact.

    Tensor-shape conventions:
      - :meth:`decode` receives flattened [B*T, ...] tokens and returns a
        prediction dict with leading dims restored to [B, T, ...].
      - ``legal`` and ``labels`` are game-specific dicts shaped [B, T, ...]
        (or None).  ``labels`` carries teacher-forcing targets during IL.
      - :meth:`sample` / :meth:`log_prob_entropy` consume that same
        prediction dict.
    """

    def __init__(self):
        super().__init__()
        # Greedy (argmax) action selection for deterministic evaluation.
        # Default False = stochastic sampling. Adapters MAY honor it in
        # their decode/sample paths (generals does); training code never
        # sets it — evaluation harnesses toggle it around eval games so
        # paired seat-swapped matches are reproducible.
        self.greedy: bool = False

    @abstractmethod
    def decode(
        self,
        tokens: torch.Tensor,          # [B*T, S_game, d]  pondered game tokens
        hl: torch.Tensor,              # [B*T, d]          high-level decision token
        ll: torch.Tensor,              # [B*T, d]          low-level decision token
        B: int,
        T: int,
        t0: int = 0,                   # absolute index of the first frame (step inference)
        legal: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Produce action logits (values shaped [B, T, ...])."""

    @abstractmethod
    def sample(
        self,
        pred: Dict[str, torch.Tensor],  # decode() output squeezed to [B, ...]
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample a full action for one frame (rollout).

        Must return a dict of [B]-shaped tensors including
        ``action_log_prob`` — the hierarchical log-prob of exactly the
        sampled action, computed the same way :meth:`log_prob_entropy`
        would, so rollout and PPO update define the same policy.
        All computation runs under ``torch.no_grad``.
        """

    @abstractmethod
    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],     # decode() output, [B, T, ...]
        actions: Dict[str, torch.Tensor],  # stored actions, [B, T, ...]
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Re-evaluate stored actions under the current policy.

        Returns:
            log_prob  [B, T]  differentiable hierarchical log-prob
            entropy   [B, T]  differentiable summed per-head entropy
            head_ent  dict of detached per-head mean entropies (diagnostics)
        """

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Optional per-game imitation loss (generals keeps its own
        ``ImitationLoss`` for training-script compatibility)."""
        raise NotImplementedError
