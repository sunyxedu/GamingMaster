"""Riichi Mahjong game adapter — observation encoder and action decoder.

Four-player riichi mahjong, everything player-relative to the acting
seat.  The observation is a fixed 39-token layout:

    tokens  0..33   tile-kind tokens (index = 34-kind map below)
    tokens 34..37   player tokens (0 = self, 1 = shimocha/right,
                    2 = toimen/across, 3 = kamicha/left)
    token  38       table token (round context)

34-kind tile index map (shared with the data pipeline via the constants
exported here)::

    0..8    1m..9m          (red 5, "5mr"/"0m", maps to kind 4)
    9..17   1p..9p          (red 5 → kind 13)
    18..26  1s..9s          (red 5 → kind 22)
    27..30  E S W N         (winds)
    31..33  P F C           (haku, hatsu, chun — white/green/red dragons)

Per-frame feature layouts (produced by
``data_processing.games.mahjong.convert``, consumed here):

``tiles_seq`` [B, T, 34, TILE_FEATURE_DIM=16] float — per tile kind k:
    0..4   own concealed-hand count one-hot (0..4)
    5      count of k in own melds / 4
    6      count of k in opponents' melds / 4
    7      own discard count of k / 4
    8      shimocha discard count / 4
    9      toimen discard count / 4
    10     kamicha discard count / 4
    11     k is a live dora (successor of a revealed marker)
    12     visible count of k / 4 (own hand + all melds + all discards
           + dora markers, clamped to 4)
    13     k is a terminal or honor
    14     own concealed hand holds the red five of this kind
    15     focus flag: the just-drawn tile on own-turn frames, the
           claimable discard on call-opportunity frames

``players_seq`` [B, T, 4, PLAYER_FEATURE_DIM=12] float — player-relative
row r (0 self, 1 shimocha, 2 toimen, 3 kamicha):
    0      score / 100000
    1      riichi declared
    2..5   seat wind one-hot (E S W N)
    6      is dealer (oya)
    7      meld count / 4
    8      discard count / 24
    9      kan count / 4
    10     focus actor (the discarder on call-opportunity frames)
    11     1.0 bias

``table_seq`` [B, T, TABLE_FEATURE_DIM=12] float:
    0..3   bakaze (round wind) one-hot
    4      (kyoku − 1) / 4
    5      honba / 8
    6      kyotaku (riichi sticks) / 4
    7      wall tiles left / 70
    8      revealed dora-marker count / 5
    9      own-turn flag (1 = deciding after own draw, 0 = call decision)
    10     riichi-declared count / 4
    11     1.0 bias

Action space (hierarchical Type → Tile, ARG_USES-gated like the
Hearthstone adapter): 8 types, one optional tile argument pointing at
the 34 tile-kind tokens.

    0 dahai     discard            tile = discarded kind
    1 riichi    declare + discard  tile = discarded kind
    2 chi       call run           tile = base (lowest) kind of the run
    3 pon       call triplet       tile = called kind
    4 kan       any kan            tile = kan kind
    5 hora      ron / tsumo win    (no tile)
    6 ryukyoku  kyushu kyuhai      (no tile)
    7 pass      decline a call     (no tile)

The tile pointer conditions only on (type embedding, LL token), so the
sole teacher-forcing label is ``action_type`` (−1 → sample); the chosen
type is echoed as ``action_type_used``.
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

# ---------------------------------------------------------------------------
# Shared constants (the data pipeline imports these — single source of truth)
# ---------------------------------------------------------------------------

N_TILE_KINDS = 34
N_MJ_PLAYERS = 4
TILE_FEATURE_DIM = 16
PLAYER_FEATURE_DIM = 12
TABLE_FEATURE_DIM = 12

N_ACTION_TYPES = 8
(
    ACTION_DAHAI,
    ACTION_RIICHI,
    ACTION_CHI,
    ACTION_PON,
    ACTION_KAN,
    ACTION_HORA,
    ACTION_RYUKYOKU,
    ACTION_PASS,
) = range(N_ACTION_TYPES)

ACTION_NAMES = (
    "dahai", "riichi", "chi", "pon", "kan", "hora", "ryukyoku", "pass",
)


@dataclass
class MahjongConfig(ModelConfig):
    """Core config + mahjong-specific dimensions (all defaulted so the
    dataclass inheritance stays valid)."""

    mahjong_tile_feature_dim: int = TILE_FEATURE_DIM
    mahjong_player_feature_dim: int = PLAYER_FEATURE_DIM
    mahjong_table_feature_dim: int = TABLE_FEATURE_DIM
    n_action_types: int = N_ACTION_TYPES


def _mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply a legality mask with the same degenerate-row guard as
    ``masked_log_prob`` so sampling and replay see the same distribution."""
    if mask is None:
        return logits
    safe = logits.masked_fill(~mask, float("-inf"))
    all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
    if all_masked.any():
        safe = safe.clone()
        safe[all_masked, ..., 0] = 0.0
    return safe


class MahjongEncoder(ObservationEncoder):
    """34 tile-kind tokens + 4 player tokens + 1 table token = 39 tokens.

    obs keys:
        tiles_seq   [B, T, 34, tile_feature_dim]
        players_seq [B, T, 4, player_feature_dim]
        table_seq   [B, T, table_feature_dim]

    Every group gets its own Linear projection; the per-slot embedding is
    delivered via :meth:`positional_embedding` (the core adds it once).

    ``frame_mask`` is intentionally ignored: there are no batch statistics
    (no BatchNorm — only per-frame pointwise Linears), so a padded frame
    cannot leak into real frames' features; padded frames are excluded
    downstream by the loss masks and the causal temporal mixers, exactly
    as documented in the base contract.
    """

    def __init__(self, cfg: MahjongConfig):
        super().__init__()
        self.cfg = cfg

        self.tile_slice = slice(0, N_TILE_KINDS)
        self.player_slice = slice(N_TILE_KINDS, N_TILE_KINDS + N_MJ_PLAYERS)
        self.table_idx = N_TILE_KINDS + N_MJ_PLAYERS
        self.n_tokens = N_TILE_KINDS + N_MJ_PLAYERS + 1

        self.tile_proj = nn.Linear(cfg.mahjong_tile_feature_dim, cfg.d_model)
        self.player_proj = nn.Linear(cfg.mahjong_player_feature_dim, cfg.d_model)
        self.table_proj = nn.Sequential(
            nn.Linear(cfg.mahjong_table_feature_dim, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Per-slot embedding delivered via positional_embedding() — the
        # core adds it once, so the encoder must not add it again.
        # (nn.Parameter is NOT re-initialised by the core's _init_weights.)
        self.slot_embed = nn.Parameter(
            torch.randn(1, self.n_tokens, cfg.d_model) * 0.02,
        )

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tiles = self.tile_proj(obs["tiles_seq"])            # [B, T, 34, d]
        players = self.player_proj(obs["players_seq"])      # [B, T, 4, d]
        table = self.table_proj(obs["table_seq"]).unsqueeze(2)  # [B, T, 1, d]
        return torch.cat([tiles, players, table], dim=2)    # [B, T, 39, d]

    def positional_embedding(self) -> torch.Tensor:
        return self.slot_embed


class MahjongDecoder(ActionDecoder):
    """Hierarchical Type → Tile decoder with per-type argument gating.

    legal keys:   action_type [B, T, 8]  bool
                  tile        [B, T, 34] bool
    labels keys:  action_type [B, T]     teacher forcing (−1 → sample) —
                  the tile pointer conditions on the type embedding, so
                  the type choice must be replayable; the tile pointer
                  itself needs no forcing (nothing conditions on it).

    Hierarchical log-prob: type_lp plus the tile head's lp only where
    ``ARG_USES[action_type]`` says the type takes a tile — a pass must
    not pay (or earn) probability mass for a fictitious tile.  Sampling,
    replay and entropy all gather the same flag.
    """

    #: uses_tile per action type
    ARG_USES_DEFAULT = [
        True,    # 0 dahai:    discarded kind
        True,    # 1 riichi:   discarded kind
        True,    # 2 chi:      base (lowest) kind of the run
        True,    # 3 pon:      called kind
        True,    # 4 kan:      kan kind
        False,   # 5 hora
        False,   # 6 ryukyoku
        False,   # 7 pass
    ]

    def __init__(self, cfg: MahjongConfig, arg_uses: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        if arg_uses is None:
            if cfg.n_action_types != len(self.ARG_USES_DEFAULT):
                raise ValueError(
                    f"n_action_types={cfg.n_action_types} but the default ARG_USES "
                    f"table has {len(self.ARG_USES_DEFAULT)} rows; pass arg_uses="
                )
            arg_uses = torch.tensor(self.ARG_USES_DEFAULT, dtype=torch.bool)
        if arg_uses.shape != (cfg.n_action_types,):
            raise ValueError(
                f"arg_uses must be [{cfg.n_action_types}], got {tuple(arg_uses.shape)}"
            )
        self.register_buffer("ARG_USES", arg_uses.bool())

        self.type_head = MLPHead(cfg.d_model, cfg.n_action_types)
        self.type_embed = nn.Embedding(cfg.n_action_types, cfg.d_model)
        self.tile_head = PointerHead(cfg.d_model, n_queries=2)  # tiles ⊗ (type, ll)

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

        def _legal(key, n):
            m = legal.get(key) if legal else None
            return m.reshape(BT, n) if m is not None else None

        type_logits = self.type_head(ll)                     # [BT, n_types] raw
        action_type = self._pick(
            type_logits, labels.get("action_type") if labels else None,
            _legal("action_type", cfg.n_action_types), BT, cfg.n_action_types,
        )

        tile_tokens = tokens[:, :N_TILE_KINDS]               # [BT, 34, d]
        tile_logits = self.tile_head(tile_tokens, self.type_embed(action_type), ll)

        return {
            "action_type_logits": type_logits.reshape(B, T, cfg.n_action_types),
            "tile_logits": tile_logits.reshape(B, T, N_TILE_KINDS),
            "action_type_used": action_type.reshape(B, T),
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

        tile_masked = _mask_logits(pred["tile_logits"], _m("tile"))
        tile_lp = F.log_softmax(tile_masked, dim=-1).gather(
            -1, actions["tile_idx"].unsqueeze(-1)).squeeze(-1)
        tile_ent = categorical_entropy(tile_masked)

        uses = self.ARG_USES[actions["action_type"]]          # [...]
        zero = torch.zeros_like(type_lp)
        total_lp = type_lp + torch.where(uses, tile_lp, zero)
        total_ent = type_ent + torch.where(uses, tile_ent, zero)

        with torch.no_grad():
            f = uses.float()
            head_entropy = {
                "ent_type": type_ent.detach().mean(),
                "ent_tile": (tile_ent.detach() * f).sum() / f.sum().clamp(min=1),
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
        """Sample the tile (the type was sampled in decode()).

        The tile head is sampled even when the type doesn't use it so the
        replay buffer always holds valid indices; the ARG_USES gather
        zeroes its log-prob contribution."""
        with torch.no_grad():
            tile_idx = masked_sample(
                pred["tile_logits"], legal.get("tile") if legal else None,
            )
            actions = {
                "action_type": pred["action_type_used"],
                "tile_idx": tile_idx,
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
        """Masked CE per head; the tile head is restricted to frames whose
        GT action type actually uses a tile AND condition-weighted by
        whether decode()'s type choice matched the ground truth — tile
        logits computed from the wrong type embedding are not a learning
        signal (same discipline as the Hearthstone adapter's cond_weight).

        tgt keys: action_type / tile_idx [B, T] long (tile_idx may be −1
        or 0 on tile-less frames; either way the weight zeroes it out).
        """
        cfg = self.cfg
        B, T, _ = pred["tile_logits"].shape
        seq_f = seq_mask.float()
        n_seq = seq_f.sum().clamp(min=1)

        tgt_type = tgt["action_type"].long().clamp(0, cfg.n_action_types - 1)
        uses = self.ARG_USES[tgt_type]                        # [B, T]

        with torch.no_grad():
            type_match = pred["action_type_used"] == tgt_type
            w_tile = uses.float() * type_match.float() * seq_f
            n_tile = w_tile.sum().clamp(min=1)

        def _ce(logits, target, n_classes):
            safe_tgt = target.long().clamp(0, n_classes - 1)
            return F.cross_entropy(
                logits.reshape(-1, n_classes), safe_tgt.reshape(-1), reduction="none",
            ).reshape(B, T).nan_to_num(0.0), safe_tgt

        type_loss_all, _ = _ce(pred["action_type_logits"], tgt_type, cfg.n_action_types)
        type_loss = (type_loss_all * seq_f).sum() / n_seq

        tile_loss_all, safe_tile = _ce(pred["tile_logits"], tgt["tile_idx"], N_TILE_KINDS)
        tile_loss = (tile_loss_all * w_tile).sum() / n_tile

        loss = type_loss + tile_loss

        with torch.no_grad():
            type_acc = ((pred["action_type_logits"].argmax(-1) == tgt_type).float()
                        * seq_f).sum() / n_seq
            tile_acc = ((pred["tile_logits"].argmax(-1) == safe_tile).float()
                        * w_tile).sum() / n_tile

        return {
            "loss": loss,
            "type_loss": type_loss.detach(),
            "tile_loss": tile_loss.detach(),
            "type_acc": type_acc,
            "tile_acc": tile_acc,
        }
