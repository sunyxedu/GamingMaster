"""Hearthstone game adapter — observation encoder and action decoder.

Hearthstone has no spatial grid: the state is a small set of entities
(hand cards, minions, heroes, hero power) with scalar features.  Each
entity gets a fixed token slot — variable-count zones (hand, boards) are
padded to their maximum size with a presence mask, per the
``ObservationEncoder`` contract, so the core's PE and decision-token
indices stay static.

The action space is hierarchical with per-action-type argument
applicability: an ``end_turn`` has no arguments, an ``attack`` has an
actor and a target but no position, etc.  The decoder gathers per-type
(uses_actor, uses_target, uses_position) flags and includes only the
applicable heads in the hierarchical log-prob — the same discipline that
lets generals skip move heads on pass frames, generalised to a table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    MLPHead,
    ObservationEncoder,
    PointerHead,
    categorical_entropy,
    masked_sample,
)


@dataclass
class HearthstoneConfig(ModelConfig):
    """Core config + Hearthstone-specific dimensions (all defaulted so the
    dataclass inheritance stays valid)."""

    hand_slots: int = 10
    board_slots: int = 7
    #: per-entity scalars: attack / health / cost / keyword flags ...
    entity_feature_dim: int = 24
    #: hand card, own minion, enemy minion, own hero, enemy hero, hero power
    n_entity_types: int = 6
    #: mana, turn, deck sizes, ...
    hs_global_features: int = 8
    #: 0=end_turn, 1=play_card, 2=attack, 3=hero_power
    n_action_types: int = 4


def _mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply a legality mask with the same degenerate-row guard as
    ``masked_log_prob`` so sampling and replay see the same distribution."""
    if mask is None:
        return logits
    safe = logits.masked_fill(~mask, float("-inf"))
    all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
    if all_masked.any():
        # Column-slice write — see base.safe_masked_logits.
        safe = safe.clone()
        safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
    return safe


class HearthstoneEncoder(ObservationEncoder):
    """Fixed entity-slot layout → one token per entity slot.

    Slot layout (exposed as attributes for env/feature code):
        hand_slice        hand cards
        own_board_slice   own minions
        enemy_board_slice enemy minions
        own_hero_idx / enemy_hero_idx / hero_power_idx

    obs keys:
        entities_seq [B, T, n_tokens, F]      per-entity scalar features
        entity_mask  [B, T, n_tokens] bool    True = slot occupied
        entity_types [B, T, n_tokens] long    entity-type ids
        globals_seq  [B, T, G]                global scalars

    ``frame_mask`` is intentionally ignored: this encoder has no batch
    statistics (no BatchNorm — only per-frame pointwise Linear/Embedding),
    so a padded frame cannot leak into real frames' features; padded
    frames are excluded downstream by the loss masks and the causal
    temporal mixers, exactly as documented in the base contract.
    """

    def __init__(self, cfg: HearthstoneConfig):
        super().__init__()
        self.cfg = cfg

        h, b = cfg.hand_slots, cfg.board_slots
        self.hand_slice = slice(0, h)
        self.own_board_slice = slice(h, h + b)
        self.enemy_board_slice = slice(h + b, h + 2 * b)
        self.own_hero_idx = h + 2 * b
        self.enemy_hero_idx = h + 2 * b + 1
        self.hero_power_idx = h + 2 * b + 2
        self.n_tokens = h + 2 * b + 3

        self.feat_proj = nn.Linear(cfg.entity_feature_dim, cfg.d_model)
        self.type_embed = nn.Embedding(cfg.n_entity_types, cfg.d_model)
        # Empty slots keep their token (fixed layout) but carry a dedicated
        # learnable embedding instead of stale features / garbage types.
        self.entity_empty = nn.Parameter(torch.randn(1, 1, 1, cfg.d_model) * 0.02)

        self.global_proj = nn.Sequential(
            nn.Linear(cfg.hs_global_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Per-slot embedding delivered via positional_embedding() — the
        # core adds it once, so the encoder must not add it again.
        self.slot_embed = nn.Parameter(torch.randn(1, self.n_tokens, cfg.d_model) * 0.02)

        # The core only re-inits Linear/Conv/Norm; Embedding is ours to init.
        nn.init.trunc_normal_(self.type_embed.weight, std=0.02)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        present = obs["entity_mask"].unsqueeze(-1)                     # [B,T,S,1]
        x = self.feat_proj(obs["entities_seq"]) + self.type_embed(obs["entity_types"])
        # Absent slots: zero the content, substitute the "empty" embedding
        # (zeroing also makes the output invariant to padded-slot garbage).
        x = x * present + self.entity_empty * (~present)
        return x + self.global_proj(obs["globals_seq"]).unsqueeze(2)  # [B,T,S,d]

    def positional_embedding(self) -> torch.Tensor:
        return self.slot_embed


class EntityActionDecoder(ActionDecoder):
    """Hierarchical Type → Actor → Target / Position decoder.

    legal keys:   action_type [B, T, n_action_types] bool
                  actor       [B, T, n_tokens]       bool
                  target      [B, T, n_tokens]       bool
    labels keys:  action_type / actor_idx [B, T]  teacher forcing
                  (−1 → sample; actor-less types pin the actor to the
                  own-hero token, with or without labels) — actor
                  conditions on the type embedding, target/position
                  condition on the actor embedding, so both choices must
                  be replayable.

    Hierarchical log-prob: type_lp plus each argument head's lp only where
    ARG_USES[action_type] says the argument exists for that type — an
    end_turn must not pay (or earn) probability mass for a fictitious
    target.  Sampling, replay and entropy all gather the same flags.
    """

    #: (uses_actor, uses_target, uses_position) per action type
    ARG_USES_DEFAULT = [
        (False, False, False),  # 0 end_turn
        (True, True, True),     # 1 play_card:  card, battlecry target, board position
        (True, True, False),    # 2 attack:     attacker, defender
        (False, True, False),   # 3 hero_power: target only
    ]

    def __init__(self, cfg: HearthstoneConfig, arg_uses: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        if arg_uses is None:
            if cfg.n_action_types != len(self.ARG_USES_DEFAULT):
                raise ValueError(
                    f"n_action_types={cfg.n_action_types} but the default ARG_USES "
                    f"table has {len(self.ARG_USES_DEFAULT)} rows; pass arg_uses="
                )
            arg_uses = torch.tensor(self.ARG_USES_DEFAULT, dtype=torch.bool)
        if arg_uses.shape != (cfg.n_action_types, 3):
            raise ValueError(f"arg_uses must be [{cfg.n_action_types}, 3], got {tuple(arg_uses.shape)}")
        self.register_buffer("ARG_USES", arg_uses.bool())

        self.type_head = MLPHead(cfg.d_model, cfg.n_action_types)
        self.type_embed = nn.Embedding(cfg.n_action_types, cfg.d_model)
        self.actor_head = PointerHead(cfg.d_model, n_queries=2)   # entities ⊗ (type, ll)
        self.target_head = PointerHead(cfg.d_model, n_queries=2)  # entities ⊗ (actor, ll)
        self.position_head = MLPHead(cfg.d_model * 2, cfg.board_slots + 1, hidden=cfg.d_model)

        # The core only re-inits Linear/Conv/Norm; Embedding is ours to init.
        nn.init.trunc_normal_(self.type_embed.weight, std=0.02)

    # ------------------------------------------------------------------
    # Decode: tokens → logits
    # ------------------------------------------------------------------

    def _pick(self, logits, label, mask, BT, n):
        """Teacher-forced (−1 rows → sampled) or fully sampled index [BT]."""
        if label is not None:
            gt = label.reshape(BT).long().clamp(0, n - 1)
            needs_sample = label.reshape(BT) < 0
            if needs_sample.any():
                return torch.where(needs_sample, masked_sample(logits, mask), gt)
            return gt
        return masked_sample(logits, mask)

    def decode(
        self,
        tokens: torch.Tensor,
        hl: torch.Tensor,
        ll: torch.Tensor,
        B: int,
        T: int,
        t0: int = 0,
        legal: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        BT = B * T
        S = tokens.shape[1]
        device = tokens.device

        def _legal(key, n):
            m = legal.get(key) if legal else None
            return m.reshape(BT, n) if m is not None else None

        type_logits = self.type_head(ll)                              # [BT, n_types] raw
        action_type = self._pick(
            type_logits, labels.get("action_type") if labels else None,
            _legal("action_type", cfg.n_action_types), BT, cfg.n_action_types,
        )

        actor_logits = self.actor_head(tokens, self.type_embed(action_type), ll)
        actor_label = labels.get("actor_idx") if labels else None
        # Actor-less action types (ARG_USES[type][0] == False) carry a
        # −1 label by convention, and the label-less rollout path carries
        # no actor label at all.  Either way the actor would be sampled
        # from own policy, conditioning the target head on a random
        # entity that differs between IL, rollout and replay.  Pin those
        # rows to the own-hero token instead — a fixed, always-present
        # entity — so the conditioning is deterministic in every path.
        # The pin deliberately bypasses legal["actor"]: for actor-less
        # types the actor is pure conditioning, never part of the action
        # (its log-prob is gated off by ARG_USES), and matching the
        # IL-time conditioning matters more than an actor mask written
        # for actor-USING types.  Explicit ≥0 labels still win, and
        # actor-USING types with a −1 label keep scheduled sampling.
        own_hero = cfg.hand_slots + 2 * cfg.board_slots
        uses_actor = self.ARG_USES[action_type.reshape(BT), 0]
        if actor_label is None:
            actor_label = torch.full((BT,), -1, dtype=torch.long, device=device)
        else:
            actor_label = actor_label.reshape(BT).long()
        actor_label = torch.where(
            ~uses_actor & (actor_label < 0),
            torch.full_like(actor_label, own_hero),
            actor_label,
        )
        actor_idx = self._pick(
            actor_logits, actor_label,
            _legal("actor", S), BT, S,
        )

        actor_embed = tokens[torch.arange(BT, device=device), actor_idx]
        target_logits = self.target_head(tokens, actor_embed, ll)     # [BT, S]
        position_logits = self.position_head(torch.cat([actor_embed, ll], dim=-1))

        return {
            "action_type_logits": type_logits.reshape(B, T, cfg.n_action_types),
            "actor_logits": actor_logits.reshape(B, T, S),
            "target_logits": target_logits.reshape(B, T, S),
            "position_logits": position_logits.reshape(B, T, cfg.board_slots + 1),
            "action_type_used": action_type.reshape(B, T),
            "actor_idx": actor_idx.reshape(B, T),
        }

    # ------------------------------------------------------------------
    # Shared lp/entropy core — one code path for sample() and
    # log_prob_entropy() is what makes the PPO ratio exact.
    # ------------------------------------------------------------------

    def _lp_ent(self, pred, actions, legal):
        """Hierarchical (log_prob, entropy, per-head entropies) for any
        leading shape ([B] rollout or [B, T] replay)."""
        def _m(key):
            return legal.get(key) if legal else None

        type_masked = _mask_logits(pred["action_type_logits"], _m("action_type"))
        type_lp = F.log_softmax(type_masked, dim=-1).gather(
            -1, actions["action_type"].unsqueeze(-1)).squeeze(-1)
        type_ent = categorical_entropy(type_masked)

        actor_masked = _mask_logits(pred["actor_logits"], _m("actor"))
        actor_lp = F.log_softmax(actor_masked, dim=-1).gather(
            -1, actions["actor_idx"].unsqueeze(-1)).squeeze(-1)
        actor_ent = categorical_entropy(actor_masked)

        target_masked = _mask_logits(pred["target_logits"], _m("target"))
        target_lp = F.log_softmax(target_masked, dim=-1).gather(
            -1, actions["target_idx"].unsqueeze(-1)).squeeze(-1)
        target_ent = categorical_entropy(target_masked)

        pos_masked = _mask_logits(pred["position_logits"], _m("position"))
        pos_lp = F.log_softmax(pos_masked, dim=-1).gather(
            -1, actions["position"].unsqueeze(-1)).squeeze(-1)
        pos_ent = categorical_entropy(pos_masked)

        uses = self.ARG_USES[actions["action_type"]]                  # [..., 3]
        zero = torch.zeros_like(type_lp)
        total_lp = (
            type_lp
            + torch.where(uses[..., 0], actor_lp, zero)
            + torch.where(uses[..., 1], target_lp, zero)
            + torch.where(uses[..., 2], pos_lp, zero)
        )
        total_ent = (
            type_ent
            + torch.where(uses[..., 0], actor_ent, zero)
            + torch.where(uses[..., 1], target_ent, zero)
            + torch.where(uses[..., 2], pos_ent, zero)
        )

        with torch.no_grad():
            def _mean(ent, flag):
                f = flag.float()
                return (ent.detach() * f).sum() / f.sum().clamp(min=1)
            head_entropy = {
                "ent_type": type_ent.detach().mean(),
                "ent_actor": _mean(actor_ent, uses[..., 0]),
                "ent_target": _mean(target_ent, uses[..., 1]),
                "ent_position": _mean(pos_ent, uses[..., 2]),
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------
    # Rollout sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample target/position (type/actor were sampled in decode()).

        All argument heads are sampled even when the type doesn't use them
        so the replay buffer always holds valid indices; the ARG_USES
        gather zeroes their log-prob contribution."""
        with torch.no_grad():
            target_idx = masked_sample(
                pred["target_logits"], legal.get("target") if legal else None,
            )
            position = masked_sample(
                pred["position_logits"], legal.get("position") if legal else None,
            )
            actions = {
                "action_type": pred["action_type_used"],
                "actor_idx": pred["actor_idx"],
                "target_idx": target_idx,
                "position": position,
            }
            total_lp, _, _ = self._lp_ent(pred, actions, legal)
        return {**actions, "action_log_prob": total_lp}

    # ------------------------------------------------------------------
    # PPO log-prob replay
    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return self._lp_ent(pred, actions, legal)

    # ------------------------------------------------------------------
    # Imitation loss
    # ------------------------------------------------------------------

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Masked CE per head, restricted to frames whose GT action type
        actually uses the argument, and condition-weighted by whether
        decode()'s conditioning choices matched the ground truth — logits
        computed from the wrong type/actor embedding are not a learning
        signal (same discipline as ImitationLoss's cond_weight).

        tgt keys: action_type / actor_idx / target_idx / position [B,T]
        long.  Per-INSTANCE absent arguments are −1 (untargeted plays,
        spell positions, untargeted hero powers): ARG_USES gating is per
        TYPE, so validity is additionally gated on ``tgt >= 0`` — the
        clamp in ``_ce`` is purely a crash guard, never a training label.
        """
        cfg = self.cfg
        B, T, S = pred["actor_logits"].shape
        seq_f = seq_mask.float()
        n_seq = seq_f.sum().clamp(min=1)

        tgt_type = tgt["action_type"].long().clamp(0, cfg.n_action_types - 1)
        uses = self.ARG_USES[tgt_type]                                # [B, T, 3]

        with torch.no_grad():
            type_match = pred["action_type_used"] == tgt_type
            # Actor conditioning is "transitively correct" for target and
            # position when either the actor matched or the type never had
            # an actor to begin with.
            actor_ok = (pred["actor_idx"] == tgt["actor_idx"]) | ~uses[..., 0]
            w_actor = (uses[..., 0].float() * type_match.float() * seq_f
                       * (tgt["actor_idx"] >= 0).float())
            w_target = (uses[..., 1].float() * (type_match & actor_ok).float() * seq_f
                        * (tgt["target_idx"] >= 0).float())
            w_pos = (uses[..., 2].float() * (type_match & actor_ok).float() * seq_f
                     * (tgt["position"] >= 0).float())
            n_actor = w_actor.sum().clamp(min=1)
            n_target = w_target.sum().clamp(min=1)
            n_pos = w_pos.sum().clamp(min=1)

        def _ce(logits, target, n_classes):
            safe_tgt = target.long().clamp(0, n_classes - 1)
            return F.cross_entropy(
                logits.reshape(-1, n_classes), safe_tgt.reshape(-1), reduction="none",
            ).reshape(B, T).nan_to_num(0.0), safe_tgt

        type_loss_all, _ = _ce(pred["action_type_logits"], tgt_type, cfg.n_action_types)
        type_loss = (type_loss_all * seq_f).sum() / n_seq

        actor_loss_all, safe_actor = _ce(pred["actor_logits"], tgt["actor_idx"], S)
        actor_loss = (actor_loss_all * w_actor).sum() / n_actor

        target_loss_all, safe_target = _ce(pred["target_logits"], tgt["target_idx"], S)
        target_loss = (target_loss_all * w_target).sum() / n_target

        pos_loss_all, safe_pos = _ce(pred["position_logits"], tgt["position"], cfg.board_slots + 1)
        pos_loss = (pos_loss_all * w_pos).sum() / n_pos

        loss = type_loss + actor_loss + target_loss + pos_loss

        with torch.no_grad():
            type_acc = ((pred["action_type_logits"].argmax(-1) == tgt_type).float() * seq_f).sum() / n_seq
            actor_acc = ((pred["actor_logits"].argmax(-1) == safe_actor).float() * w_actor).sum() / n_actor
            target_acc = ((pred["target_logits"].argmax(-1) == safe_target).float() * w_target).sum() / n_target
            pos_acc = ((pred["position_logits"].argmax(-1) == safe_pos).float() * w_pos).sum() / n_pos
            type_ok = pred["action_type_logits"].argmax(-1) == tgt_type

            def _arg_ok(logits, safe, tgt_raw, use):
                return (logits.argmax(-1) == safe) | ~use | (tgt_raw < 0)

            answer_ok = (type_ok
                         & _arg_ok(pred["actor_logits"], safe_actor, tgt["actor_idx"], uses[..., 0])
                         & _arg_ok(pred["target_logits"], safe_target, tgt["target_idx"], uses[..., 1])
                         & _arg_ok(pred["position_logits"], safe_pos, tgt["position"], uses[..., 2]))

        return {
            "answer_correct": answer_ok,
            "answer_mask": seq_mask.bool(),
            "loss": loss,
            "type_loss": type_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "target_loss": target_loss.detach(),
            "position_loss": pos_loss.detach(),
            "type_acc": type_acc,
            "actor_acc": actor_acc,
            "target_acc": target_acc,
            "position_acc": pos_acc,
        }
