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
``gmdata mahjong convert`` (data_processing/rs/gmdata/src/mahjong/), consumed here):

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
           claimable discard (or robbable kakan tile) on call-opportunity
           frames

``players_seq`` [B, T, 4, PLAYER_FEATURE_DIM=12] float — player-relative
row r (0 self, 1 shimocha, 2 toimen, 3 kamicha):
    0      score / 100000
    1      riichi declared
    2..5   seat wind one-hot (E S W N)
    6      is dealer (oya)
    7      meld count / 4
    8      discard count / 24
    9      kan count / 4
    10     focus actor (the discarder / kakan declarer on call-opportunity
           frames)
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

Legality: ``action_type`` [..., 8] plus a tile mask conditioned on the
committed type — ``tile_cond`` [..., 8, 34] (exact, RL: build it with
:func:`mahjong_legal_tables`) or ``tile_of`` [..., 34] (valid for the
labelled type: the converter's ``tgt.legal_tile_of``, IL legality via
``tgt.legal_<key>`` → ``legal[<key>]``); the legacy unconditional ``tile``
mask still works.  See :class:`MahjongDecoder`.

Imitation weight: ``tgt["il_weight"]`` [B, T] (converter_version ≥ 3)
scales every frame's imitation terms — 0 on a recorded pass that does not
show the seat's choice, because a call it could make was pre-empted by a
higher-priority one (another seat's ron, or a pon over a chi).  Absent
(older shards) = 1: the historical loss bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

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
        # Column-slice write — see base.safe_masked_logits.
        safe = safe.clone()
        safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
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


def _as_mask(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Legality masks may arrive as packed-bit float {0,1} (converter tgt.legal_*)."""
    if m is None or m.dtype == torch.bool:
        return m
    return m.bool()


def mahjong_legal_tables(
    legal_actions: Sequence[Tuple[int, Optional[int]]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Legal ``(action_type, tile | None)`` pairs of one decision point →
    decoder legal masks ``{"action_type": [8], "tile_cond": [8, 34]}`` (bool).

    ``tile_cond[t]`` lists the legal tile arguments of type ``t``; a type
    that takes no tile (hora / ryukyoku / pass, or a pair given with
    ``tile=None``) gets an all-True row, like the converter's
    ``legal_tile_of`` — its tile is sampled but never scored (ARG_USES).
    Stack per-frame tables along new leading dims for a batch."""
    types = torch.zeros(N_ACTION_TYPES, dtype=torch.bool, device=device)
    cond = torch.zeros(N_ACTION_TYPES, N_TILE_KINDS, dtype=torch.bool, device=device)
    uses = MahjongDecoder.ARG_USES_DEFAULT
    for t, tile in legal_actions:
        if not 0 <= t < N_ACTION_TYPES:
            raise ValueError(f"action type {t} outside [0, {N_ACTION_TYPES})")
        types[t] = True
        if tile is None or not uses[t]:
            cond[t] = True
        elif not 0 <= tile < N_TILE_KINDS:
            raise ValueError(f"tile {tile} outside [0, {N_TILE_KINDS})")
        else:
            cond[t, tile] = True
    return {"action_type": types, "tile_cond": cond}


class MahjongDecoder(ActionDecoder):
    """Hierarchical Type → Tile decoder with per-type argument gating.

    legal keys (all optional, bool or {0,1} float):
        action_type [..., 8]      legal action types
        tile        [..., 34]     legacy unconditional tile mask
        tile_cond   [..., 8, 34]  exact: row t = legal tiles of type t
                                  (RL; see :func:`mahjong_legal_tables`)
        tile_of     [..., 34]     legal tiles of the type in
                                  ``labels["action_type"]`` (the converter's
                                  ``tgt.legal_tile_of``; RL replay of a
                                  stored ``legal_tile``) — applied only where
                                  the committed type equals the label; a −1 /
                                  absent label leaves the tile unconstrained
                                  (or on the legacy ``tile`` mask)
    labels keys:  action_type [B, T]     teacher forcing (−1 → sample) —
                  the tile pointer conditions on the type embedding, so
                  the type choice must be replayable; the tile pointer
                  itself needs no forcing (nothing conditions on it).

    decode() resolves the tile mask of the COMMITTED type (tile_cond >
    tile_of > tile) and echoes the masks it used — ``action_type_mask``
    [B, T, 8] / ``tile_mask`` [B, T, 34], only when a mask was given — so
    sample(), log_prob_entropy() and il_loss() all score the distribution
    decode() committed to.  sample() also returns them as
    ``legal_action_type`` / ``legal_tile``; replaying a rollout with
    ``legal={"action_type": legal_action_type, "tile_of": legal_tile}`` and
    ``labels={"action_type": stored type}`` reproduces its log-prob exactly.

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

    @staticmethod
    def _resolve_tile_mask(
        legal: Optional[Dict[str, torch.Tensor]],
        types: torch.Tensor,
        label: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Tile mask [..., 34] for the committed ``types`` [...]:
        tile_cond row > tile_of (where types == label ≥ 0) > legacy tile."""
        if not legal:
            return None
        cond = _as_mask(legal.get("tile_cond"))
        if cond is not None:
            cond = cond.reshape(*types.shape, -1, N_TILE_KINDS)
            idx = types.long()[..., None, None].expand(*types.shape, 1, N_TILE_KINDS)
            return cond.gather(-2, idx).squeeze(-2)
        flat = _as_mask(legal.get("tile"))
        flat = flat.reshape(*types.shape, N_TILE_KINDS) if flat is not None else None
        of = _as_mask(legal.get("tile_of"))
        if of is None:
            return flat
        of = of.reshape(*types.shape, N_TILE_KINDS)
        if label is None:
            applies = torch.zeros_like(types, dtype=torch.bool)
        else:
            lbl = label.reshape(types.shape).long()
            applies = (lbl >= 0) & (types.long() == lbl)
        base = flat if flat is not None else torch.ones_like(of)
        return torch.where(applies.unsqueeze(-1), of, base)

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

        type_mask = _as_mask(legal.get("action_type")) if legal else None
        if type_mask is not None:
            type_mask = type_mask.reshape(BT, cfg.n_action_types)

        type_logits = self.type_head(ll)                     # [BT, n_types] raw
        label = labels.get("action_type") if labels else None
        action_type = self._pick(
            type_logits, label, type_mask, BT, cfg.n_action_types,
        )
        tile_mask = self._resolve_tile_mask(
            legal, action_type, label.reshape(BT) if label is not None else None,
        )

        tile_tokens = tokens[:, :N_TILE_KINDS]               # [BT, 34, d]
        tile_logits = self.tile_head(tile_tokens, self.type_embed(action_type), ll)

        out = {
            "action_type_logits": type_logits.reshape(B, T, cfg.n_action_types),
            "tile_logits": tile_logits.reshape(B, T, N_TILE_KINDS),
            "action_type_used": action_type.reshape(B, T),
        }
        if type_mask is not None:
            out["action_type_mask"] = type_mask.reshape(B, T, cfg.n_action_types)
        if tile_mask is not None:
            out["tile_mask"] = tile_mask.reshape(B, T, N_TILE_KINDS)
        return out

    # ------------------------------------------------------------------
    # Shared lp/entropy core — one code path for sample() and
    # log_prob_entropy() is what makes the PPO ratio exact.
    # ------------------------------------------------------------------

    def _masks(self, pred, types, legal):
        """(type mask, tile mask) for scoring ``types``: decode()'s echoes,
        else the ``legal`` dict (tile_cond gathered at ``types``, tile_of —
        whose row is the stored type — or the legacy tile mask)."""
        type_mask = pred.get("action_type_mask")
        if type_mask is None and legal:
            type_mask = _as_mask(legal.get("action_type"))
        tile_mask = pred.get("tile_mask")
        if tile_mask is None and legal:
            tile_mask = self._resolve_tile_mask(legal, types, types)
        return type_mask, tile_mask

    def _lp_ent(self, pred, actions, legal):
        """Hierarchical (log_prob, entropy, per-head entropies) for any
        leading shape ([B] rollout or [B, T] replay)."""
        type_mask, tile_mask = self._masks(pred, actions["action_type"], legal)

        type_masked = _mask_logits(pred["action_type_logits"].float(), type_mask)
        type_lp = F.log_softmax(type_masked, dim=-1).gather(
            -1, actions["action_type"].unsqueeze(-1)).squeeze(-1)
        type_ent = categorical_entropy(type_masked)

        tile_masked = _mask_logits(pred["tile_logits"].float(), tile_mask)
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
        """Sample the tile (the type was sampled in decode()) under the
        committed type's tile mask.

        The tile head is sampled even when the type doesn't use it so the
        replay buffer always holds valid indices; the ARG_USES gather
        zeroes its log-prob contribution.  Returns the masks used as
        ``legal_action_type`` / ``legal_tile`` (when masked) for replay."""
        with torch.no_grad():
            types = pred["action_type_used"]
            type_mask, tile_mask = self._masks(pred, types, legal)
            tile_idx = masked_sample(pred["tile_logits"], tile_mask)
            actions = {"action_type": types, "tile_idx": tile_idx}
            total_lp, _, _ = self._lp_ent(pred, actions, legal)
        out = {**actions, "action_log_prob": total_lp}
        if type_mask is not None:
            out["legal_action_type"] = type_mask
        if tile_mask is not None:
            out["legal_tile"] = tile_mask
        return out

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

    def _il_parts(self, pred, tgt, seq_mask) -> Dict[str, Any]:
        """Per-element pieces shared by il_loss and il_head_terms.

        Without decode() mask echoes: plain CE (the historical loss,
        bit-identical).  With them (IL legality via ``--il_legal_mask``):
        CE over the masked logits (degenerate-row guard), a target outside
        its mask gets weight 0 (its CE would be +inf) and is counted in
        ``illegal_target_frac``; the normalisers are unchanged.

        ``tgt["il_weight"]`` [B, T] (converter_version ≥ 3; absent = 1, the
        historical loss bit for bit) weights every frame's imitation terms
        and their normalisers and metrics: 0 on a recorded pass that does
        not show the seat's choice (a call it could make was pre-empted by
        a higher-priority one), so such frames train the value head only."""
        cfg = self.cfg
        B, T, _ = pred["tile_logits"].shape
        seq_f = seq_mask.float()
        il_weight = tgt.get("il_weight")
        if il_weight is not None:
            seq_f = seq_f * il_weight.float()
        n_seq = seq_f.sum().clamp(min=1)

        tgt_type = tgt["action_type"].long().clamp(0, cfg.n_action_types - 1)
        uses = self.ARG_USES[tgt_type]                        # [B, T]
        type_mask = pred.get("action_type_mask")
        tile_mask = pred.get("tile_mask")

        with torch.no_grad():
            type_match = pred["action_type_used"] == tgt_type
            w_tile = uses.float() * type_match.float() * seq_f
            n_tile = w_tile.sum().clamp(min=1)

        def _ce(logits, target, n_classes):
            safe_tgt = target.long().clamp(0, n_classes - 1)
            return F.cross_entropy(
                logits.reshape(-1, n_classes), safe_tgt.reshape(-1), reduction="none",
            ).reshape(B, T).nan_to_num(0.0), safe_tgt

        type_logits = pred["action_type_logits"]
        tile_logits = pred["tile_logits"]
        type_scored = _mask_logits(type_logits, type_mask)
        tile_scored = _mask_logits(tile_logits, tile_mask)

        type_ce, _ = _ce(type_scored, tgt_type, cfg.n_action_types)
        tile_ce, safe_tile = _ce(tile_scored, tgt["tile_idx"], N_TILE_KINDS)
        w_type = seq_f
        illegal = None
        if type_mask is not None:
            type_ok = type_mask.gather(-1, tgt_type.unsqueeze(-1)).squeeze(-1)
            type_ce = torch.where(type_ok, type_ce, torch.zeros_like(type_ce))
            w_type = seq_f * type_ok.float()
            illegal = ~type_ok
        if tile_mask is not None:
            tile_ok = tile_mask.gather(-1, safe_tile.unsqueeze(-1)).squeeze(-1)
            tile_ce = torch.where(tile_ok, tile_ce, torch.zeros_like(tile_ce))
            bad_tile = ~tile_ok & (w_tile > 0)
            w_tile = w_tile * tile_ok.float()
            illegal = bad_tile if illegal is None else illegal | bad_tile
        return {
            "tgt_type": tgt_type, "safe_tile": safe_tile, "uses": uses,
            "seq_f": seq_f, "n_seq": n_seq,
            "supervised": seq_mask.bool() if il_weight is None else seq_mask.bool() & (seq_f > 0),
            "type_logits": type_logits, "type_scored": type_scored, "type_ce": type_ce,
            "w_type": w_type, "type_mask": type_mask,
            "tile_logits": tile_logits, "tile_scored": tile_scored, "tile_ce": tile_ce,
            "w_tile": w_tile, "n_tile": n_tile, "tile_mask": tile_mask,
            "illegal": illegal,
        }

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
        With decode() mask echoes both heads score the legal-masked
        distributions (see :meth:`_il_parts`).

        tgt keys: action_type / tile_idx [B, T] long (tile_idx may be −1
        or 0 on tile-less frames; either way the weight zeroes it out);
        optional il_weight [B, T] (see :meth:`_il_parts`; weight-0 frames
        are also left out of ``answer_mask``, the q-halt / exact-accuracy
        target).
        """
        p = self._il_parts(pred, tgt, seq_mask)
        seq_f, n_seq, n_tile = p["seq_f"], p["n_seq"], p["n_tile"]
        tgt_type, safe_tile, uses = p["tgt_type"], p["safe_tile"], p["uses"]

        type_loss = (p["type_ce"] * p["w_type"]).sum() / n_seq
        tile_loss = (p["tile_ce"] * p["w_tile"]).sum() / n_tile
        loss = type_loss + tile_loss

        with torch.no_grad():
            type_hit = p["type_scored"].argmax(-1) == tgt_type
            tile_hit = p["tile_scored"].argmax(-1) == safe_tile
            type_acc = (type_hit.float() * seq_f).sum() / n_seq
            tile_acc = (tile_hit.float() * p["w_tile"]).sum() / n_tile
            answer_ok = type_hit & (tile_hit | ~uses.bool())

        out = {
            "answer_correct": answer_ok,
            "answer_mask": p["supervised"],
            "loss": loss,
            "type_loss": type_loss.detach(),
            "tile_loss": tile_loss.detach(),
            "type_acc": type_acc,
            "tile_acc": tile_acc,
        }
        if p["illegal"] is not None:
            out["illegal_target_frac"] = (p["illegal"].float() * seq_f).sum() / n_seq
        return out

    def il_head_terms(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, Dict[str, Any]]:
        """Per-head decomposition of :meth:`il_loss` (ActionDecoder
        contract): ``il_loss["loss"] == Σ_h (ce_h · weight_h).sum() / norm_h``.

        Soft targets: ``tgt["action_type_probs"]`` [B, T, 8] and
        ``tgt["tile_probs"]`` [B, T, 34] = p(tile | type = the labelled
        type) — the tile head is scored where decode() committed to the
        labelled type (``weight``), under the committed type's tile mask."""
        p = self._il_parts(pred, tgt, seq_mask)
        return {
            "action_type": {
                "logits": p["type_logits"], "kind": "categorical", "ce": p["type_ce"],
                "weight": p["w_type"], "norm": p["n_seq"], "mask": p["type_mask"],
                "scale": 1.0,
            },
            "tile": {
                "logits": p["tile_logits"], "kind": "categorical", "ce": p["tile_ce"],
                "weight": p["w_tile"], "norm": p["n_tile"], "mask": p["tile_mask"],
                "scale": 1.0,
            },
        }
