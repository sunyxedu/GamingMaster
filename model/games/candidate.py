"""Candidate-action games: ONE adapter for every game whose legal actions can be
enumerated at each decision (card games, bidding, poker with discretised sizes,
backgammon full moves, ...).  The per-decision legal actions are CANDIDATES: each
is a token with its own features, and the decoder points at one of them.

Observation (the candidate schema of gmrules games, ``FrameSpec::candidate``:
the gm-shard ``obs`` tensors written by data_processing/rs/gmdata/src/traj.rs and
emitted by the gmsim simulators — the same Rust encoder, gmrules ``encode_frame``):

    entities     [B, T, E, Fe] float   entity_mask [B, T, E] bool
    entity_type  [B, T, E] int         (only when cand_entity_types > 0)
    globals      [B, T, G] float
    actions      [B, T, A, Fa] float   action_mask [B, T, A] bool
                                       (the LEGAL candidates of the frame, in ascending
                                       canonical-id order, padded to A)

Tokens (``n_tokens = E + 1 + A``, fixed per config: static shapes, CUDA-graph
friendly):

    [0, E)          entity tokens   Linear(Fe) (+ type embedding); absent slots get
                                    a learned "empty" embedding (so padded garbage
                                    cannot leak in); positional embedding per slot
                                    (``cand_entity_slot_pe``) or one shared vector
    E               global token    MLP(G)
    [E+1, E+1+A)    candidate tokens Linear(Fa); padded slots → learned "empty
                                    candidate"; ONE shared positional embedding
                                    (a candidate's index carries no meaning)

Candidates are tokens of the core, so the shared core attends between the state
and the options before the decoder scores them.  Keep entity slots stable across
frames (slot = identity: a card, a seat, a zone) — the core's Stage-1 mixes each
token slot over time.

Decoder (:class:`CandidateDecoder`): a pointer head scores every candidate token
against the LL decision token; the legality mask is the frame's own
``action_mask`` (delivered through :meth:`CandidateEncoder.decode_context`, so
IL and RL use the same mask) ∧ ``legal["action"]`` when given.

    pred        action_logits [B, T, A] (raw), action_mask [B, T, A]
    sample()    {"action_idx": [B], "action_log_prob": [B]}
    replay      actions = {"action_idx": [B, T]}
    IL tgt      action_idx [B, T] (−1 = no label), value; il_head_terms head
                "action" → soft targets ``tgt["action_probs_idx"/"action_probs_val"]``
                over candidate indices (or dense ``action_probs`` [B, T, A])

Each game registers its own entry (model/games/gmrules/<game>.py: its config
defaults, optionally its own encoder / decoder subclasses) and reuses these
classes; the frame dimensions of a fresh run come from the shard meta
(train_il_game META_ARCH_KEYS = :data:`CANDIDATE_META_ARCH`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    ObservationEncoder,
    PointerHead,
    categorical_entropy,
    masked_log_prob,
    masked_sample,
    safe_masked_logits,
)
from model.games.gmrules import CANDIDATE_META_ARCH, resolve


@dataclass
class CandidateConfig(ModelConfig):
    """Core config + the candidate-action frame dimensions (all defaulted; a
    fresh run takes them from the shard meta, CANDIDATE_META_ARCH)."""

    #: E: entity slots per frame
    cand_max_entities: int = 8
    #: Fe: features per entity
    cand_entity_features: int = 8
    #: entity type ids (0 = no ``entity_type`` input)
    cand_entity_types: int = 0
    #: G: global features per frame
    cand_global_features: int = 4
    #: A: candidate slots per frame (max legal actions)
    cand_max_actions: int = 8
    #: Fa: features per candidate
    cand_action_features: int = 4
    #: a positional embedding per entity slot (True) or one shared (False: the
    #: entities are a set)
    cand_entity_slot_pe: bool = True


class CandidateEncoder(ObservationEncoder):
    """entities + globals + candidate actions → ``E + 1 + A`` tokens.

    ``frame_mask`` is ignored: no batch statistics (per-frame Linear / Embedding
    only), so a padded frame cannot leak into real frames.
    """

    def __init__(self, cfg: CandidateConfig):
        super().__init__()
        self.cfg = cfg
        E, A, d = cfg.cand_max_entities, cfg.cand_max_actions, cfg.d_model
        if min(E, A, cfg.cand_entity_features, cfg.cand_global_features,
               cfg.cand_action_features) < 1:
            raise ValueError("candidate frame sizes must all be >= 1")
        self.n_tokens = E + 1 + A
        self.entity_slice = slice(0, E)
        self.global_idx = E
        self.action_slice = slice(E + 1, E + 1 + A)

        self.entity_proj = nn.Linear(cfg.cand_entity_features, d)
        self.type_embed = (nn.Embedding(cfg.cand_entity_types, d)
                           if cfg.cand_entity_types > 0 else None)
        self.entity_empty = nn.Parameter(torch.randn(d) * 0.02)
        self.global_proj = nn.Sequential(
            nn.Linear(cfg.cand_global_features, d), nn.ReLU(), nn.Linear(d, d))
        self.action_proj = nn.Linear(cfg.cand_action_features, d)
        self.action_empty = nn.Parameter(torch.randn(d) * 0.02)

        # delivered through positional_embedding(): the core adds it once
        self.entity_pe = nn.Parameter(
            torch.randn(1, E if cfg.cand_entity_slot_pe else 1, d) * 0.02)
        self.global_pe = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.action_pe = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # the core only re-inits Linear / Conv / Norm; Embedding is ours to init
        if self.type_embed is not None:
            nn.init.trunc_normal_(self.type_embed.weight, std=0.02)

    def forward(self, obs: Dict[str, torch.Tensor],
                frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.entity_proj(obs["entities"].float())                  # [B,T,E,d]
        if self.type_embed is not None:
            types = obs["entity_type"].long().clamp(0, self.cfg.cand_entity_types - 1)
            x = x + self.type_embed(types).to(x.dtype)
        present = obs["entity_mask"].bool().unsqueeze(-1)
        x = torch.where(present, x, self.entity_empty.to(x.dtype))
        g = self.global_proj(obs["globals"].float()).unsqueeze(2)      # [B,T,1,d]
        a = self.action_proj(obs["actions"].float())                    # [B,T,A,d]
        legal = obs["action_mask"].bool().unsqueeze(-1)
        a = torch.where(legal, a, self.action_empty.to(a.dtype))
        return torch.cat([x, g.to(x.dtype), a.to(x.dtype)], dim=2)

    def positional_embedding(self) -> torch.Tensor:
        E, A, d = self.cfg.cand_max_entities, self.cfg.cand_max_actions, self.cfg.d_model
        return torch.cat([self.entity_pe.expand(1, E, d), self.global_pe,
                          self.action_pe.expand(1, A, d)], dim=1)

    def decode_context(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """The frame's candidate mask: the decoder's legality in IL and RL alike."""
        return {"action_mask": obs["action_mask"].bool()}


class CandidateDecoder(ActionDecoder):
    """Pointer over the candidate tokens (see the module docstring).

    legal keys:  action [B, T, A] bool (optional; ANDed with the frame's own
                 candidate mask from the decode context)
    labels:      none (a single categorical: nothing to teacher-force)
    """

    def __init__(self, cfg: CandidateConfig):
        super().__init__()
        self.cfg = cfg
        self.a0 = cfg.cand_max_entities + 1
        self.n_actions = cfg.cand_max_actions
        self.pointer = PointerHead(cfg.d_model, n_queries=1)

    def _mask(self, ctx, legal, B: int, T: int, device) -> torch.Tensor:
        A = self.n_actions
        mask = None
        if ctx is not None and "action_mask" in ctx:
            mask = ctx["action_mask"].reshape(B, T, A).bool()
        if legal is not None and legal.get("action") is not None:
            lg = legal["action"].reshape(B, T, A).bool()
            mask = lg if mask is None else mask & lg
        if mask is None:
            mask = torch.ones(B, T, A, dtype=torch.bool, device=device)
        return mask

    def decode(self, tokens: torch.Tensor, hl: torch.Tensor, ll: torch.Tensor,
               B: int, T: int, t0: int = 0,
               legal: Optional[Dict[str, torch.Tensor]] = None,
               labels: Optional[Dict[str, torch.Tensor]] = None,
               ctx: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        cand = tokens[:, self.a0:self.a0 + self.n_actions]             # [BT, A, d]
        logits = self.pointer(cand, ll).reshape(B, T, self.n_actions)
        return {"action_logits": logits,
                "action_mask": self._mask(ctx, legal, B, T, tokens.device)}

    @staticmethod
    def _full_mask(pred: Dict[str, torch.Tensor],
                   legal: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        mask = pred["action_mask"].bool()
        if legal is not None and legal.get("action") is not None:
            mask = mask & legal["action"].bool().reshape(mask.shape)
        return mask

    def sample(self, pred: Dict[str, torch.Tensor],
               legal: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = pred["action_logits"]
            mask = self._full_mask(pred, legal)
            if self.greedy:
                idx = safe_masked_logits(logits, mask).argmax(-1)
            else:
                idx = masked_sample(logits, mask)
            lp = masked_log_prob(logits, idx, mask)
        return {"action_idx": idx, "action_log_prob": lp}

    def log_prob_entropy(self, pred: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor],
                         legal: Optional[Dict[str, torch.Tensor]] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        logits = pred["action_logits"].float()
        mask = self._full_mask(pred, legal)
        lp = masked_log_prob(logits, actions["action_idx"].long(), mask)
        safe = mask | ~mask.any(-1, keepdim=True)          # all-masked rows: uniform, no NaN
        ent = categorical_entropy(logits.masked_fill(~safe, float("-inf")))
        return lp, ent, {"ent_action": ent.detach().mean()}

    # ------------------------------------------------------------------
    # imitation
    # ------------------------------------------------------------------

    def _il_parts(self, pred: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                  seq_mask: torch.Tensor) -> Dict[str, Any]:
        logits = pred["action_logits"]
        mask = pred["action_mask"].bool()
        A = logits.shape[-1]
        tgt_idx = tgt["action_idx"].long()
        safe_tgt = tgt_idx.clamp(0, A - 1)
        tgt_legal = mask.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
        valid = seq_mask.bool() & (tgt_idx >= 0) & (tgt_idx < A) & tgt_legal
        ce = -masked_log_prob(logits, safe_tgt, mask)
        # a label on an illegal / padded candidate (never written by the converters)
        # gets weight 0; nan_to_num keeps its inf out of the arithmetic
        ce = torch.where(valid, ce, torch.zeros_like(ce)).nan_to_num(0.0)
        w = valid.float()
        return dict(ce=ce, w=w, norm=w.sum().clamp(min=1), mask=mask, safe_tgt=safe_tgt,
                    valid=valid)

    def il_head_terms(self, pred: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                      seq_mask: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """One categorical head ``action`` over the candidate slots, legality-masked
        (soft targets: ``tgt["action_probs_idx"/"action_probs_val"]`` = candidate
        indices / probabilities, or dense ``tgt["action_probs"]`` [B, T, A])."""
        p = self._il_parts(pred, tgt, seq_mask)
        return {"action": {"logits": pred["action_logits"], "ce": p["ce"], "weight": p["w"],
                           "norm": p["norm"], "mask": p["mask"]}}

    def il_loss(self, pred: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                seq_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Legality-masked cross-entropy of ``tgt["action_idx"]`` (−1 = unlabelled
        frame) over the candidates."""
        p = self._il_parts(pred, tgt, seq_mask)
        loss = (p["ce"] * p["w"]).sum() / p["norm"]
        with torch.no_grad():
            best = safe_masked_logits(pred["action_logits"], p["mask"]).argmax(-1)
            correct = best == p["safe_tgt"]
            acc = (correct.float() * p["w"]).sum() / p["norm"]
            n_legal = (p["mask"].float().sum(-1) * p["w"]).sum() / p["norm"]
        return {"loss": loss, "action_loss": loss.detach(), "action_acc": acc,
                "mean_legal": n_legal, "answer_correct": correct, "answer_mask": p["valid"]}


def build_candidate_network(game: Optional[str] = None, **overrides) -> nn.Module:
    """A GameNetwork with the candidate adapter; ``game`` picks the registered
    entry's config defaults and encoder / decoder classes
    (model/games/gmrules/<game>.py)."""
    from model.game_network import GameNetwork
    from model.games.gmrules import rules_games

    spec = rules_games()[game] if game is not None else None
    if spec is not None and not spec.is_candidate:
        raise ValueError(f"{game} is a custom-schema gmrules game (its own builder)")
    # precedence: explicit overrides (incl. the shard meta of a fresh IL run, see
    # train_il_game.apply_meta_arch_defaults) > the entry's config > the
    # simulator's frame spec for its sim_config
    base = {}
    if spec is not None:
        if not all(k in spec.config or k in overrides
                   for keys in CANDIDATE_META_ARCH.values() for k in keys):
            base = spec.sim_dims()
        base.update(spec.config)
    kw = {**base, **overrides}
    cfg_cls = (resolve(spec.config_class) if spec is not None and spec.config_class is not None
               else CandidateConfig)
    cfg = cfg_cls(**kw)
    enc = resolve(spec.encoder) if spec is not None and spec.encoder is not None else CandidateEncoder
    dec = resolve(spec.decoder) if spec is not None and spec.decoder is not None else CandidateDecoder
    return GameNetwork(cfg, enc(cfg), dec(cfg))
