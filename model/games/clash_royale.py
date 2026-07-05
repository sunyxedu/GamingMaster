"""Clash Royale game adapter — observation encoder and action decoder.

Observations combine three modalities:
  - an arena grid (unit presence / side / hp / tower planes) encoded by the
    same conv stack shape as the generals ``SpatialEncoder``,
  - a fixed-size hand of cards (variable real count → fixed padded slots
    with a presence mask, per the ``ObservationEncoder`` contract),
  - a global scalar vector (elixir, time, tower counts, ...).

The action space is hierarchical: first decide whether to WAIT this frame
(elixir management is mostly about *not* playing), and if playing, pick a
card from the hand and then a placement cell conditioned on that card.
Sampling, log-prob replay and teacher forcing all share one code path so
PPO ratios are exact (same discipline as the generals adapter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.encoder import ResBlock
from model.components.transformer import _sinusoidal_pe
from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    MLPHead,
    ObservationEncoder,
    PointerHead,
    bernoulli_log_prob_entropy,
    categorical_entropy,
    masked_log_prob,
    masked_sample,
)


@dataclass
class ClashRoyaleConfig(ModelConfig):
    """Core config + Clash-Royale-specific dimensions (all defaulted so the
    dataclass inheritance stays valid)."""

    arena_h: int = 32
    arena_w: int = 18
    #: input planes: unit presence / side / hp / tower planes etc.
    arena_channels: int = 16
    hand_size: int = 4
    #: card-id vocabulary; index 0 = empty slot
    n_cards: int = 110
    #: per-card scalar features (elixir cost, level, ...)
    card_feature_dim: int = 8
    #: own/enemy elixir, match time, tower counts, ...
    cr_global_features: int = 6


def _mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply a legality mask with the same degenerate-row guard as
    ``masked_log_prob`` so lp/entropy and sampling see the same distribution."""
    if mask is None:
        return logits
    safe = logits.masked_fill(~mask, float("-inf"))
    all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
    if all_masked.any():
        safe = safe.clone()
        safe[all_masked, ..., 0] = 0.0
    return safe


class ClashRoyaleEncoder(ObservationEncoder):
    """Arena grid + hand slots + globals → arena_h*arena_w + hand_size tokens.

    obs keys:
        arena_seq   [B, T, C, H, W]              arena planes
        hand_ids    [B, T, hand_size] long       card ids (0 = empty)
        hand_feats  [B, T, hand_size, F]         per-card scalar features
        hand_mask   [B, T, hand_size] bool       True = real card in slot
        globals_seq [B, T, G]                    global scalars

    Token order: arena tokens first, then hand tokens — the decoder relies
    on this split (``n_arena_tokens``).
    """

    def __init__(self, cfg: ClashRoyaleConfig):
        super().__init__()
        self.cfg = cfg
        self.n_arena_tokens = cfg.arena_h * cfg.arena_w
        self.n_tokens = self.n_arena_tokens + cfg.hand_size

        # Arena conv encoder — mirrors SpatialEncoder's Conv→BN→ReLU→Res
        # structure so masked-BatchNorm frame handling transfers unchanged.
        c_mid = cfg.encoder_channels
        self.arena_encoder = nn.Sequential(
            nn.Conv2d(cfg.arena_channels, c_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            ResBlock(c_mid),
            nn.Conv2d(c_mid, cfg.d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.d_model),
            nn.ReLU(inplace=True),
            ResBlock(cfg.d_model),
        )

        # Hand tokens: card identity + card scalars, summed.
        self.card_embed = nn.Embedding(cfg.n_cards, cfg.d_model)
        self.hand_feat_proj = nn.Linear(cfg.card_feature_dim, cfg.d_model)
        # Empty hand slots keep their token (fixed layout) but carry a
        # dedicated learnable embedding instead of stale card features.
        self.hand_empty = nn.Parameter(torch.randn(1, 1, 1, cfg.d_model) * 0.02)

        self.global_proj = nn.Sequential(
            nn.Linear(cfg.cr_global_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # 2-D arena PE (learnable row + column, summed) and per-hand-slot
        # embedding.  Both are delivered through positional_embedding() —
        # the core adds it once, so the encoder must not add them again.
        self.row_embed = nn.Parameter(torch.randn(1, cfg.arena_h, 1, cfg.d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, cfg.arena_w, cfg.d_model) * 0.02)
        self.hand_slot_embed = nn.Parameter(torch.randn(1, cfg.hand_size, cfg.d_model) * 0.02)

        # The core only re-inits Linear/Conv/Norm; Embedding is ours to init.
        nn.init.trunc_normal_(self.card_embed.weight, std=0.02)

    # ------------------------------------------------------------------

    def _encode_frames(self, arena_seq, frame_mask):
        """Run the arena conv encoder, excluding padded frames from
        BatchNorm statistics when ``frame_mask`` ([B, T] bool, True = real
        frame) is given — same discipline as GeneralsEncoder: padded frames
        get zero features and, being tail-padding under causal mixers,
        cannot influence real frames."""
        B, T, C, H, W = arena_seq.shape
        BT = B * T
        flat = arena_seq.reshape(BT, C, H, W)
        if frame_mask is None or bool(frame_mask.all()):
            return self.arena_encoder(flat).reshape(B, T, -1, H, W)
        fm = frame_mask.reshape(BT)
        enc = self.arena_encoder(flat[fm])
        feat = enc.new_zeros(BT, enc.shape[1], H, W)
        feat[fm] = enc
        return feat.reshape(B, T, -1, H, W)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        arena_seq = obs["arena_seq"]
        B, T = arena_seq.shape[:2]
        d = self.cfg.d_model

        spatial = self._encode_frames(arena_seq, frame_mask)          # [B,T,d,H,W]
        spatial = spatial + self.global_proj(obs["globals_seq"]).unsqueeze(-1).unsqueeze(-1)
        arena_tokens = spatial.reshape(B, T, d, self.n_arena_tokens).permute(0, 1, 3, 2)

        hand = self.card_embed(obs["hand_ids"]) + self.hand_feat_proj(obs["hand_feats"])
        present = obs["hand_mask"].unsqueeze(-1)                      # [B,T,hand,1]
        # Absent slots: zero the card features, substitute the learnable
        # "empty" embedding (fixed token layout — never shrink the sequence).
        hand = hand * present + self.hand_empty * (~present)

        return torch.cat([arena_tokens, hand], dim=2)                 # [B,T,S,d]

    def positional_embedding(self) -> torch.Tensor:
        arena_pe = (self.row_embed + self.col_embed).reshape(1, -1, self.cfg.d_model)
        return torch.cat([arena_pe, self.hand_slot_embed], dim=1)


class CardPlacementDecoder(ActionDecoder):
    """Hierarchical Wait → Card → Placement decoder.

    legal keys:   card      [B, T, hand_size]       bool
                  placement [B, T, n_arena_tokens]  bool
    labels keys:  card_slot [B, T]  teacher-forcing card choice
                  (−1 → sample from the model's own card distribution)

    Hierarchical log-prob (must match between sample() and
    log_prob_entropy() — the PPO-exactness invariant):
        wait → wait_lp
        play → wait_lp (not-wait side) + card_lp + placement_lp
    """

    def __init__(self, cfg: ClashRoyaleConfig):
        super().__init__()
        self.cfg = cfg
        self.n_arena_tokens = cfg.arena_h * cfg.arena_w
        # Wait uses a sinusoidal time PE like generals' PassHead: whether
        # to hold elixir depends on match phase, not just the board.
        self.time_pe_dim = 32
        self.wait_head = MLPHead(cfg.d_model + self.time_pe_dim, 1, hidden=cfg.d_model)
        self.card_head = PointerHead(cfg.d_model, n_queries=1)       # hand ⊗ ll
        self.place_head = PointerHead(cfg.d_model, n_queries=2)      # arena ⊗ (card, ll)

    # ------------------------------------------------------------------
    # Decode: tokens → logits
    # ------------------------------------------------------------------

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

        arena_tok = tokens[:, : self.n_arena_tokens]                 # [BT, A, d]
        hand_tok = tokens[:, self.n_arena_tokens:]                   # [BT, hand, d]

        card_logits = self.card_head(hand_tok, ll)                   # [BT, hand] raw

        # Card conditioning: teacher-forced when labels are given (−1 rows
        # fall back to the model's own masked sample), else sampled — the
        # chosen index is returned so losses can condition-weight on it.
        card_mask = legal.get("card") if legal else None
        cm_flat = card_mask.reshape(BT, cfg.hand_size) if card_mask is not None else None
        card_label = labels.get("card_slot") if labels else None
        if card_label is not None:
            gt_slot = card_label.reshape(BT).long().clamp(0, cfg.hand_size - 1)
            needs_sample = card_label.reshape(BT) < 0
            if needs_sample.any():
                pred_slot = masked_sample(card_logits, cm_flat)
                card_slot = torch.where(needs_sample, pred_slot, gt_slot)
            else:
                card_slot = gt_slot
        else:
            card_slot = masked_sample(card_logits, cm_flat)

        card_embed = hand_tok[torch.arange(BT, device=device), card_slot]
        placement_logits = self.place_head(arena_tok, card_embed, ll)  # [BT, A]

        # Absolute-time PE (indices t0 .. t0+T-1) so step() at frame t and
        # forward() at position t see the same wait feature.
        pe = _sinusoidal_pe(t0 + T, self.time_pe_dim, device)[t0: t0 + T]
        pe = pe.unsqueeze(0).expand(B, T, self.time_pe_dim).reshape(BT, self.time_pe_dim)
        wait_logits = self.wait_head(torch.cat([ll, pe], dim=-1)).squeeze(-1)

        return {
            "wait_logits": wait_logits.reshape(B, T),
            "card_logits": card_logits.reshape(B, T, cfg.hand_size),
            "placement_logits": placement_logits.reshape(B, T, self.n_arena_tokens),
            "card_slot": card_slot.reshape(B, T),
        }

    # ------------------------------------------------------------------
    # Rollout sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample wait/placement (card was already sampled in decode()) and
        compute the hierarchical log-prob exactly as log_prob_entropy would.

        Card and placement are always sampled, even on wait frames, so the
        replay buffer has valid conditioning indices; the hierarchy zeroes
        their contribution when waited (same as generals' pass)."""
        wait_logits = pred["wait_logits"]
        card_slot = pred["card_slot"]
        card_mask = legal.get("card") if legal else None
        place_mask = legal.get("placement") if legal else None

        with torch.no_grad():
            waited = torch.bernoulli(torch.sigmoid(wait_logits)).bool()
            wait_lp, _ = bernoulli_log_prob_entropy(wait_logits, waited)

            card_lp = masked_log_prob(pred["card_logits"], card_slot, card_mask)

            place_idx = masked_sample(pred["placement_logits"], place_mask)
            place_lp = masked_log_prob(pred["placement_logits"], place_idx, place_mask)

            total_lp = torch.where(waited, wait_lp, wait_lp + card_lp + place_lp)

        return {
            "card_slot": card_slot,
            "place_idx": place_idx,
            "waited": waited,
            "action_log_prob": total_lp,
        }

    # ------------------------------------------------------------------
    # PPO log-prob replay
    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        waited = actions["waited"]                                    # [B, T] bool
        card_mask = legal.get("card") if legal else None
        place_mask = legal.get("placement") if legal else None

        wait_lp, wait_ent = bernoulli_log_prob_entropy(pred["wait_logits"], waited)

        card_masked = _mask_logits(pred["card_logits"], card_mask)
        card_lp = F.log_softmax(card_masked, dim=-1).gather(
            -1, actions["card_slot"].unsqueeze(-1),
        ).squeeze(-1)
        card_ent = categorical_entropy(card_masked)

        place_masked = _mask_logits(pred["placement_logits"], place_mask)
        place_lp = F.log_softmax(place_masked, dim=-1).gather(
            -1, actions["place_idx"].unsqueeze(-1),
        ).squeeze(-1)
        place_ent = categorical_entropy(place_masked)

        total_lp = torch.where(waited, wait_lp, wait_lp + card_lp + place_lp)
        total_ent = torch.where(waited, wait_ent, wait_ent + card_ent + place_ent)

        with torch.no_grad():
            play_f = (~waited).float()
            n_play = play_f.sum().clamp(min=1)
            head_entropy = {
                "ent_wait": wait_ent.detach().mean(),
                "ent_card": (card_ent.detach() * play_f).sum() / n_play,
                "ent_place": (place_ent.detach() * play_f).sum() / n_play,
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------
    # Imitation loss
    # ------------------------------------------------------------------

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Masked CE per head.  Placement conditions on the chosen card, so
        its loss is weighted by (decode's conditioning card == GT card) —
        the same discipline as ImitationLoss's cond_weight: a placement
        computed from the wrong card embedding is not a learning signal.

        tgt keys: waited [B,T] bool, card_slot [B,T] long, place_idx [B,T]
        long (card/place ignored — may be −1 — on waited frames).
        """
        B, T, hand = pred["card_logits"].shape
        n_arena = pred["placement_logits"].shape[-1]
        seq_f = seq_mask.float()
        n_seq = seq_f.sum().clamp(min=1)

        waited = tgt["waited"]
        play = seq_mask & ~waited
        play_f = play.float()
        n_play = play_f.sum().clamp(min=1)

        tgt_card = torch.where(play, tgt["card_slot"], 0).long().clamp(0, hand - 1)
        tgt_place = torch.where(play, tgt["place_idx"], 0).long().clamp(0, n_arena - 1)

        wait_loss_all = F.binary_cross_entropy_with_logits(
            pred["wait_logits"], waited.float(), reduction="none",
        ).nan_to_num(0.0)
        wait_loss = (wait_loss_all * seq_f).sum() / n_seq

        card_loss_all = F.cross_entropy(
            pred["card_logits"].reshape(-1, hand), tgt_card.reshape(-1),
            reduction="none",
        ).reshape(B, T).nan_to_num(0.0)
        card_loss = (card_loss_all * play_f).sum() / n_play

        with torch.no_grad():
            cond = (pred["card_slot"] == tgt_card).float() * play_f
            n_cond = cond.sum().clamp(min=1.0)

        place_loss_all = F.cross_entropy(
            pred["placement_logits"].reshape(-1, n_arena), tgt_place.reshape(-1),
            reduction="none",
        ).reshape(B, T).nan_to_num(0.0)
        place_loss = (place_loss_all * cond).sum() / n_cond

        loss = wait_loss + card_loss + place_loss

        with torch.no_grad():
            wait_acc = (((pred["wait_logits"] > 0) == waited).float() * seq_f).sum() / n_seq
            card_acc = ((pred["card_logits"].argmax(-1) == tgt_card).float() * play_f).sum() / n_play
            place_acc = ((pred["placement_logits"].argmax(-1) == tgt_place).float() * cond).sum() / n_cond

        return {
            "loss": loss,
            "wait_loss": wait_loss.detach(),
            "card_loss": card_loss.detach(),
            "place_loss": place_loss.detach(),
            "wait_acc": wait_acc,
            "card_acc": card_acc,
            "place_acc": place_acc,
        }
