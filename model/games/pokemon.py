"""Pokémon (Showdown singles) game adapter — observation encoder and
action decoder.

Singles v1 token layout (``n_tokens = 13``):

    slots 0-5    own team   (team-preview order)
    slots 6-11   opponent team
    slot  12     field token (weather / terrain / turn / side conditions)

Each Pokémon slot carries a hash-bucket species embedding (species names
are hashed into ``pokemon_species_buckets`` ids by the data pipeline —
id 0 is reserved for "unknown") plus a small numeric feature vector.
Slots whose species is unknown (``slot_mask`` False — e.g. no team
preview) are replaced by a dedicated learnable "empty" embedding, the
Hearthstone pattern.

The observer-view replay data cannot see hidden information (opponent
movesets, items, exact stats); the encoder deliberately only consumes
what the sim-protocol log reveals.

Numeric per-slot features (``pokemon_feature_dim = 17``; the layout is
the single source of truth shared with the converter via the
``POKE_FEAT_*`` constants):

    0      hp_frac                  current HP fraction 0..1
    1      active                   is the currently active Pokémon
    2      fainted
    3-9    status one-hot           none/brn/par/psn/tox/slp/frz
    10-14  boosts / 6               atk/def/spa/spd/spe, clipped to ±1
    15     revealed                 has entered battle at least once
    16     tera                     is terastallized

Field-token features (``pokemon_field_features = 20``):

    0-4    weather one-hot          sun/rain/sand/snow/other (all 0 = clear)
    5-8    terrain one-hot          electric/grassy/misty/psychic
    9      turn / 50                clipped to 1
    10-14  own side conditions      spikes/3, toxic spikes/2, stealth rock,
                                    sticky web, screens/3 (reflect + light
                                    screen + aurora veil count)
    15-19  opponent side conditions (same layout)

obs keys:
    species_ids [B, T, 12] long     hash-bucket species ids (0 = unknown)
    poke_feats  [B, T, 12, F]       numeric features above
    slot_mask   [B, T, 12] bool     True = species known (slot exists)
    field_seq   [B, T, 20]          field features above

``frame_mask`` is intentionally a no-op: the encoder has no batch
statistics (no BatchNorm — only per-frame pointwise Linear/Embedding),
so padded frames cannot leak into real frames; they are excluded
downstream by loss masks and the causal temporal mixers, per the base
contract.

Action space:

    type 0 = move    argument: categorical over 4 move slots.  Slot k is
                     scored by the identity of the k-th REVEALED move of
                     the active mon: the converter emits
                     ``obs["move_ids"]`` [T, 4] hash-bucket ids (0 = not
                     yet revealed) in the same revelation order the
                     ``move_idx`` label indexes, the trainer forwards
                     them via ``legal`` (LEGAL_FROM_OBS), and the head
                     scores each slot's move embedding against
                     (active_embed, type_embed, ll).  Plus an optional
                     Bernoulli terastallize flag.
    type 1 = switch  argument: pointer over the 6 own slot tokens.

Because the acting Pokémon is state, not a decision, the decoder needs
its own-slot index from outside: ``decode`` reads ``active_idx`` [B, T]
long from ``labels`` (IL: the converter stores it in ``tgt``) or from
``legal`` (rollout / PPO replay), defaulting to slot 0 if absent.
``move_ids`` travels the same way (legal or labels); without it every
slot shares the id-0 embedding and the move head degrades to the old
positional prior.
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
    bernoulli_log_prob_entropy,
    categorical_entropy,
    masked_sample,
)

# Per-slot numeric feature indices (shared with the converter).
POKE_FEAT_HP = 0
POKE_FEAT_ACTIVE = 1
POKE_FEAT_FAINTED = 2
POKE_FEAT_STATUS = 3          # 7 one-hot dims: none/brn/par/psn/tox/slp/frz
POKE_FEAT_BOOSTS = 10         # 5 dims: atk/def/spa/spd/spe, /6 clipped
POKE_FEAT_REVEALED = 15
POKE_FEAT_TERA = 16
POKE_FEATURE_DIM = 17

FIELD_FEATURE_DIM = 20

N_MOVE_SLOTS = 4
TEAM_SLOTS = 6


@dataclass
class PokemonConfig(ModelConfig):
    """Core config + Pokémon-specific dimensions (all defaulted so the
    dataclass inheritance stays valid)."""

    team_slots: int = TEAM_SLOTS
    #: numeric per-slot features (see module docstring feature table)
    pokemon_feature_dim: int = POKE_FEATURE_DIM
    #: hash buckets for the species embedding (id 0 = unknown)
    pokemon_species_buckets: int = 512
    #: hash buckets for the revealed-move embedding (id 0 = unrevealed
    #: slot) — the move head scores slots by their move identity, so the
    #: within-battle revelation-order labels refer to observable content
    pokemon_move_buckets: int = 256
    #: field-token features (weather/terrain/turn/side conditions)
    pokemon_field_features: int = FIELD_FEATURE_DIM
    #: 0 = move, 1 = switch
    n_action_types: int = 2
    n_move_slots: int = N_MOVE_SLOTS


class PokemonEncoder(ObservationEncoder):
    """12 Pokémon slot tokens + 1 field token → [B, T, 13, d].

    Slot layout attributes (for feature/env code):
        own_slice   slots 0-5
        opp_slice   slots 6-11
        field_idx   12
    """

    def __init__(self, cfg: PokemonConfig):
        super().__init__()
        self.cfg = cfg
        k = cfg.team_slots
        self.own_slice = slice(0, k)
        self.opp_slice = slice(k, 2 * k)
        self.field_idx = 2 * k
        self.n_tokens = 2 * k + 1

        self.feat_proj = nn.Linear(cfg.pokemon_feature_dim, cfg.d_model)
        self.species_embed = nn.Embedding(cfg.pokemon_species_buckets, cfg.d_model)
        # Unknown slots keep their token (fixed layout) but carry a
        # dedicated learnable embedding instead of stale features.
        self.slot_empty = nn.Parameter(torch.randn(1, 1, 1, cfg.d_model) * 0.02)

        self.field_proj = nn.Sequential(
            nn.Linear(cfg.pokemon_field_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Per-slot embedding delivered via positional_embedding() — the
        # core adds it once, so the encoder must not add it again.
        self.slot_embed = nn.Parameter(torch.randn(1, self.n_tokens, cfg.d_model) * 0.02)

        # The core only re-inits Linear/Conv/Norm; Embedding is ours to init.
        nn.init.trunc_normal_(self.species_embed.weight, std=0.02)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # frame_mask is a documented no-op: no batch statistics here.
        present = obs["slot_mask"].unsqueeze(-1)                        # [B,T,12,1]
        x = self.feat_proj(obs["poke_feats"]) + self.species_embed(obs["species_ids"])
        # Absent slots: zero the content, substitute the "empty" embedding
        # (zeroing also makes the output invariant to padded-slot garbage).
        x = x * present + self.slot_empty * (~present)
        field = self.field_proj(obs["field_seq"]).unsqueeze(2)          # [B,T,1,d]
        return torch.cat([x, field], dim=2)                            # [B,T,13,d]

    def positional_embedding(self) -> torch.Tensor:
        return self.slot_embed


class PokemonDecoder(ActionDecoder):
    """Hierarchical Type → (Move slot + Tera | Switch target) decoder.

    legal keys (all optional):
        action_type [B, T, 2] bool
        move        [B, T, 4] bool      legal move slots
        switch      [B, T, 6] bool      own bench slots (alive, not active)
        tera        [B, T] bool         may still terastallize
        active_idx  [B, T] long         own active slot (NOT a mask — the
                                        acting Pokémon is state)
    labels keys (teacher forcing, −1 → sample):
        action_type [B, T] long
        active_idx  [B, T] long         conditioning state, never sampled

    Hierarchical log-prob: type_lp plus the move head + tera Bernoulli
    only on move frames, the switch pointer only on switch frames (the
    ARG_USES table, Hearthstone discipline).  The tera Bernoulli is
    additionally gated by ``legal["tera"]`` when provided: once tera is
    spent the flag is deterministic and contributes no log-prob/entropy.
    Sampling, replay and entropy all gather the same flags.
    """

    #: (uses_move, uses_switch, uses_tera) per action type
    ARG_USES_DEFAULT = [
        (True, False, True),    # 0 move:   move slot + tera flag
        (False, True, False),   # 1 switch: target own slot
    ]

    def __init__(self, cfg: PokemonConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.n_action_types != len(self.ARG_USES_DEFAULT):
            raise ValueError(
                f"n_action_types={cfg.n_action_types} but the ARG_USES table "
                f"has {len(self.ARG_USES_DEFAULT)} rows"
            )
        self.register_buffer(
            "ARG_USES", torch.tensor(self.ARG_USES_DEFAULT, dtype=torch.bool)
        )

        d = cfg.d_model
        self.type_head = MLPHead(d, cfg.n_action_types)
        self.type_embed = nn.Embedding(cfg.n_action_types, d)
        # Move slots are scored by their revealed-move identity: slot k's
        # hash-bucket move id (0 = not yet revealed, from legal/labels
        # "move_ids") is embedded and scored against (active_embed, type,
        # ll).  Without move_ids all slots share the id-0 embedding and
        # the head degrades to a positional prior.
        self.move_embed = nn.Embedding(cfg.pokemon_move_buckets, d)
        self.move_slot_embed = nn.Parameter(
            torch.randn(1, cfg.n_move_slots, d) * 0.02,
        )
        self.move_head = PointerHead(d, n_queries=3)  # move slots ⊗ (active, type, ll)
        # Switch target: pointer over the 6 own slot tokens ⊗ (type, ll).
        self.switch_head = PointerHead(d, n_queries=2)
        self.tera_head = MLPHead(d * 2, 1, hidden=d)

        # The core only re-inits Linear/Conv/Norm; Embedding is ours to init.
        nn.init.trunc_normal_(self.type_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.move_embed.weight, std=0.02)

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
        device = tokens.device
        own_tokens = tokens[:, : cfg.team_slots]                       # [BT, 6, d]

        def _legal(key, n):
            m = legal.get(key) if legal else None
            return m.reshape(BT, n) if m is not None else None

        type_logits = self.type_head(ll)                               # [BT, 2] raw
        action_type = self._pick(
            type_logits, labels.get("action_type") if labels else None,
            _legal("action_type", cfg.n_action_types), BT, cfg.n_action_types,
        )
        te = self.type_embed(action_type)

        # The acting Pokémon is state, not a decision: read its own-slot
        # index from labels (IL) or legal (rollout/replay), default slot 0.
        active = None
        if labels is not None and labels.get("active_idx") is not None:
            active = labels["active_idx"]
        elif legal is not None and legal.get("active_idx") is not None:
            active = legal["active_idx"]
        if active is not None:
            active_idx = active.reshape(BT).long().clamp(0, cfg.team_slots - 1)
        else:
            active_idx = torch.zeros(BT, dtype=torch.long, device=device)
        active_embed = own_tokens[torch.arange(BT, device=device), active_idx]

        # Revealed-move ids of the active mon (state, like active_idx):
        # labels (IL, via LEGAL_FROM_OBS→legal in the trainer) or legal
        # (rollout/replay); absent → all slots id 0 ("unrevealed").
        move_ids = None
        if labels is not None and labels.get("move_ids") is not None:
            move_ids = labels["move_ids"]
        elif legal is not None and legal.get("move_ids") is not None:
            move_ids = legal["move_ids"]
        if move_ids is not None:
            move_ids = move_ids.reshape(BT, cfg.n_move_slots).long().clamp(
                0, cfg.pokemon_move_buckets - 1)
        else:
            move_ids = torch.zeros(BT, cfg.n_move_slots, dtype=torch.long,
                                   device=device)
        move_tokens = self.move_embed(move_ids) + self.move_slot_embed
        move_logits = self.move_head(move_tokens, active_embed, te, ll)
        switch_logits = self.switch_head(own_tokens, te, ll)           # [BT, 6]
        tera_logit = self.tera_head(
            torch.cat([active_embed, ll], dim=-1)
        ).squeeze(-1)                                                  # [BT]

        return {
            "action_type_logits": type_logits.reshape(B, T, cfg.n_action_types),
            "move_logits": move_logits.reshape(B, T, cfg.n_move_slots),
            "switch_logits": switch_logits.reshape(B, T, cfg.team_slots),
            "tera_logit": tera_logit.reshape(B, T),
            "action_type_used": action_type.reshape(B, T),
            "active_idx_used": active_idx.reshape(B, T),
        }

    # ------------------------------------------------------------------
    # Shared lp/entropy core — one code path for sample() and
    # log_prob_entropy() is what makes the PPO ratio exact.
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_logits(logits, mask):
        if mask is None:
            return logits
        safe = logits.masked_fill(~mask, float("-inf"))
        all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
        if all_masked.any():
            # Column-slice write — see base.safe_masked_logits.
            safe = safe.clone()
            safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
        return safe

    def _lp_ent(self, pred, actions, legal):
        """Hierarchical (log_prob, entropy, per-head entropies) for any
        leading shape ([B] rollout or [B, T] replay)."""
        def _m(key):
            return legal.get(key) if legal else None

        type_masked = self._mask_logits(pred["action_type_logits"], _m("action_type"))
        type_lp = F.log_softmax(type_masked, dim=-1).gather(
            -1, actions["action_type"].unsqueeze(-1)).squeeze(-1)
        type_ent = categorical_entropy(type_masked)

        move_masked = self._mask_logits(pred["move_logits"], _m("move"))
        move_lp = F.log_softmax(move_masked, dim=-1).gather(
            -1, actions["move_idx"].unsqueeze(-1)).squeeze(-1)
        move_ent = categorical_entropy(move_masked)

        switch_masked = self._mask_logits(pred["switch_logits"], _m("switch"))
        switch_lp = F.log_softmax(switch_masked, dim=-1).gather(
            -1, actions["switch_idx"].unsqueeze(-1)).squeeze(-1)
        switch_ent = categorical_entropy(switch_masked)

        tera_lp, tera_ent = bernoulli_log_prob_entropy(
            pred["tera_logit"], actions["tera"].bool(),
        )

        uses = self.ARG_USES[actions["action_type"]]                   # [..., 3]
        tera_gate = _m("tera")
        use_move = uses[..., 0]
        use_switch = uses[..., 1]
        use_tera = uses[..., 2] if tera_gate is None else uses[..., 2] & tera_gate

        zero = torch.zeros_like(type_lp)
        total_lp = (
            type_lp
            + torch.where(use_move, move_lp, zero)
            + torch.where(use_switch, switch_lp, zero)
            + torch.where(use_tera, tera_lp, zero)
        )
        total_ent = (
            type_ent
            + torch.where(use_move, move_ent, zero)
            + torch.where(use_switch, switch_ent, zero)
            + torch.where(use_tera, tera_ent, zero)
        )

        with torch.no_grad():
            def _mean(ent, flag):
                f = flag.float()
                return (ent.detach() * f).sum() / f.sum().clamp(min=1)
            head_entropy = {
                "ent_type": type_ent.detach().mean(),
                "ent_move": _mean(move_ent, use_move),
                "ent_switch": _mean(switch_ent, use_switch),
                "ent_tera": _mean(tera_ent, use_tera),
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
        """Sample move/switch/tera (the type was sampled in decode()).

        All argument heads are sampled even when the type doesn't use
        them so the replay buffer always holds valid indices; the
        ARG_USES gather zeroes their log-prob contribution.  A tera flag
        sampled while ``legal["tera"]`` is False is forced to 0 — its
        log-prob contribution is gated to zero on both the rollout and
        replay paths, so the PPO ratio stays exact."""
        with torch.no_grad():
            move_idx = masked_sample(
                pred["move_logits"], legal.get("move") if legal else None,
            )
            switch_idx = masked_sample(
                pred["switch_logits"], legal.get("switch") if legal else None,
            )
            tera = torch.bernoulli(torch.sigmoid(pred["tera_logit"])).long()
            tera_gate = legal.get("tera") if legal else None
            if tera_gate is not None:
                tera = tera * tera_gate.long()
            actions = {
                "action_type": pred["action_type_used"],
                "move_idx": move_idx,
                "switch_idx": switch_idx,
                "tera": tera,
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
        decode()'s sampled type matched the ground truth — the move and
        switch heads condition on the type embedding, so logits computed
        from the wrong type embedding are not a learning signal.  The
        tera head conditions only on state (active embed + ll), so it
        needs no condition weight; its BCE runs on all GT move frames.

        tgt keys: action_type / move_idx / switch_idx / tera [B, T] long
        (unused arguments may be −1; a −1 ``move_idx`` on a move frame
        marks an expert move outside the 4 observed slots — 5th+ reveal —
        and contributes no move loss; extra keys like value / active_idx
        are ignored).
        """
        cfg = self.cfg
        B, T = pred["action_type_used"].shape
        seq_f = seq_mask.float()
        n_seq = seq_f.sum().clamp(min=1)

        tgt_type = tgt["action_type"].long().clamp(0, cfg.n_action_types - 1)
        uses = self.ARG_USES[tgt_type]                                 # [B, T, 3]

        with torch.no_grad():
            type_match = (pred["action_type_used"] == tgt_type).float()
            # move_idx −1 on a move frame is unsupervisable (the used
            # move has no observed slot); _ce's clamp would otherwise
            # silently train slot 0.
            w_move = (uses[..., 0].float() * (tgt["move_idx"] >= 0).float()
                      * type_match * seq_f)
            w_switch = uses[..., 1].float() * type_match * seq_f
            w_tera = uses[..., 2].float() * seq_f
            n_move = w_move.sum().clamp(min=1)
            n_switch = w_switch.sum().clamp(min=1)
            n_tera = w_tera.sum().clamp(min=1)

        def _ce(logits, target, n_classes):
            safe_tgt = target.long().clamp(0, n_classes - 1)
            return F.cross_entropy(
                logits.reshape(-1, n_classes), safe_tgt.reshape(-1),
                reduction="none",
            ).reshape(B, T).nan_to_num(0.0), safe_tgt

        type_loss_all, _ = _ce(pred["action_type_logits"], tgt_type, cfg.n_action_types)
        type_loss = (type_loss_all * seq_f).sum() / n_seq

        move_loss_all, safe_move = _ce(pred["move_logits"], tgt["move_idx"], cfg.n_move_slots)
        move_loss = (move_loss_all * w_move).sum() / n_move

        switch_loss_all, safe_switch = _ce(pred["switch_logits"], tgt["switch_idx"], cfg.team_slots)
        switch_loss = (switch_loss_all * w_switch).sum() / n_switch

        tgt_tera = tgt["tera"].float().clamp(0.0, 1.0)
        tera_loss_all = F.binary_cross_entropy_with_logits(
            pred["tera_logit"], tgt_tera, reduction="none",
        ).nan_to_num(0.0)
        tera_loss = (tera_loss_all * w_tera).sum() / n_tera

        loss = type_loss + move_loss + switch_loss + tera_loss

        with torch.no_grad():
            type_acc = ((pred["action_type_logits"].argmax(-1) == tgt_type).float() * seq_f).sum() / n_seq
            move_acc = ((pred["move_logits"].argmax(-1) == safe_move).float() * w_move).sum() / n_move
            switch_acc = ((pred["switch_logits"].argmax(-1) == safe_switch).float() * w_switch).sum() / n_switch
            tera_acc = (((pred["tera_logit"] > 0).float() == tgt_tera).float() * w_tera).sum() / n_tera

        return {
            "loss": loss,
            "type_loss": type_loss.detach(),
            "move_loss": move_loss.detach(),
            "switch_loss": switch_loss.detach(),
            "tera_loss": tera_loss.detach(),
            "type_acc": type_acc,
            "move_acc": move_acc,
            "switch_acc": switch_acc,
            "tera_acc": tera_acc,
        }
