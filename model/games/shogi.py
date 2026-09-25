"""Shogi adapter for the game-agnostic core.

Shogi extends the chess-like from/to action space with **drops**: a
captured piece sits in the capturer's hand and may be placed back on any
legal empty square instead of moving a board piece.  The action is
therefore a two-branch hierarchy:

    mode (Bernoulli: move vs drop)
      ├─ MOVE:  from-square → to-square → promote? (Bernoulli — shogi
      │         promotion is a yes/no choice, never a piece pick)
      └─ DROP:  hand piece type → drop square

Token layout (fixed, ``n_tokens = 81 + 14``):

    [0 .. 80]   board squares (row-major 9×9)
    [81 .. 87]  own hand slots      P L N S G B R
    [88 .. 94]  opponent hand slots P L N S G B R

Hand slots are always "real" tokens: a piece type the player holds zero
of is still encoded (its features say count = 0), so no empty-slot
embedding machinery is needed — slot *identity* (piece type + owner) is
delivered through the per-slot embedding in :meth:`positional_embedding`,
exactly like the hearthstone adapter's ``slot_embed``.

Sampling, PPO replay and IL teacher forcing all evaluate the SAME masked
distributions.  BOTH branches are decoded on every frame (all logits are
always present); only the log-prob / entropy *contribution* is gated by
the mode, which is what keeps rollout-vs-replay exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.encoder import ResBlock
from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    MLPHead,
    ObservationEncoder,
    PointerHead,
    bernoulli_log_prob_entropy,
    masked_log_prob,
    masked_sample,
)
from model.games.board import (
    _log_prob_and_entropy,
    _pick_index,
)


@dataclass
class ShogiConfig(ModelConfig):
    """Shogi dimensions on top of the shared core config.

    ``hand_feature_dim`` describes each hand slot's feature vector; the
    reference layout is (count, count > 0, count / max_count,
    side-to-move-owns-this-slot) but the encoder only fixes the *width* —
    the data pipeline owns the exact semantics.
    ``shogi_global_features = 0`` disables the global-vector projection.
    """

    board_h: int = 9
    board_w: int = 9
    piece_channels: int = 28        # 14 piece types × 2 sides is the typical layout
    n_hand_piece_types: int = 7     # P L N S G B R (droppable types)
    hand_feature_dim: int = 4       # count, count>0, normalized count, side-to-move
    shogi_global_features: int = 4  # move number, repetition flags, ...


class ShogiEncoder(ObservationEncoder):
    """Board planes → one token per square, plus 14 hand-slot tokens.

    obs keys:
        board_seq   [B, T, piece_channels, H, W]
        hand_seq    [B, T, 14, hand_feature_dim]
        globals_seq [B, T, G]   (only when ``shogi_global_features > 0``)

    Hand slots 0..6 are the side-to-move's P L N S G B R counts, slots
    7..13 the opponent's.  The hand path is a shared Linear (no batch
    statistics, so padded frames cannot pollute anything and need no
    ``frame_mask`` handling); the board conv stack uses the same
    masked-BatchNorm technique as ``BoardEncoder``.
    """

    def __init__(self, cfg: ShogiConfig):
        super().__init__()
        self.cfg = cfg
        self.n_board_tokens = cfg.board_h * cfg.board_w
        self.n_hand_slots = 2 * cfg.n_hand_piece_types
        self.n_tokens = self.n_board_tokens + self.n_hand_slots

        # Board: same conv → ResBlock → conv → ResBlock stack as BoardEncoder.
        c_mid = cfg.encoder_channels
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.piece_channels, c_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            ResBlock(c_mid),
            nn.Conv2d(c_mid, cfg.d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.d_model),
            nn.ReLU(inplace=True),
            ResBlock(cfg.d_model),
        )

        # Hand: shared feature projection; slot identity (piece type +
        # owner) comes from the per-slot embedding delivered via
        # positional_embedding() — the core adds PE right after encoding,
        # so token = Linear(features) + slot_embed, as designed.
        self.hand_proj = nn.Linear(cfg.hand_feature_dim, cfg.d_model)
        self.hand_slot_embed = nn.Parameter(
            torch.randn(1, self.n_hand_slots, cfg.d_model) * 0.02,
        )

        self.global_proj: Optional[nn.Sequential] = None
        if cfg.shogi_global_features > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(cfg.shogi_global_features, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )

        # 2-D spatial PE for board squares (learnable row + col, summed).
        self.row_embed = nn.Parameter(
            torch.randn(1, cfg.board_h, 1, cfg.d_model) * 0.02,
        )
        self.col_embed = nn.Parameter(
            torch.randn(1, 1, cfg.board_w, cfg.d_model) * 0.02,
        )

    def _encode_frames(self, board_seq, frame_mask):
        """Conv stack with padded frames excluded from BatchNorm statistics
        (same technique as ``BoardEncoder._encode_frames``): padded frames
        get zero features and cannot influence real frames because collate
        pads at the sequence tail and the temporal mixers are causal."""
        B, T, C, H, W = board_seq.shape
        BT = B * T
        flat = board_seq.reshape(BT, C, H, W)
        if frame_mask is None or bool(frame_mask.all()):
            return self.conv(flat).reshape(B, T, -1, H, W)
        fm = frame_mask.reshape(BT)
        enc = self.conv(flat[fm])
        feat = enc.new_zeros(BT, enc.shape[1], H, W)
        feat[fm] = enc
        return feat.reshape(B, T, -1, H, W)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        board_seq = obs["board_seq"]
        B, T, C, H, W = board_seq.shape
        d = self.cfg.d_model

        feat = self._encode_frames(board_seq, frame_mask)
        if self.global_proj is not None:
            glb = self.global_proj(obs["globals_seq"])
            feat = feat + glb.unsqueeze(-1).unsqueeze(-1)   # broadcast to board only
        board_tokens = feat.reshape(B, T, d, H * W).permute(0, 1, 3, 2)

        hand_tokens = self.hand_proj(obs["hand_seq"])       # [B, T, 14, d]
        return torch.cat([board_tokens, hand_tokens], dim=2)

    def positional_embedding(self) -> torch.Tensor:
        board_pe = (self.row_embed + self.col_embed).reshape(
            1, self.n_board_tokens, self.cfg.d_model,
        )
        return torch.cat([board_pe, self.hand_slot_embed], dim=1)


def _pick_mode(
    mode_logits: torch.Tensor,          # [BT] raw Bernoulli logits
    label: Optional[torch.Tensor],      # [B, T] long (−1 = sample) or None
    BT: int,
) -> torch.Tensor:
    """Teacher-forced or self-sampled drop/move mode (bool [BT], no grad).

    Same −1 convention as :func:`model.games.board._pick_index`: unknown
    labels fall back to the model's own Bernoulli sample.
    """
    with torch.no_grad():
        if label is not None:
            lbl = label.reshape(BT).long()
            gt = lbl > 0
            needs_sample = lbl < 0
            if needs_sample.any():
                sampled = torch.bernoulli(torch.sigmoid(mode_logits.float())).bool()
                return torch.where(needs_sample, sampled, gt)
            return gt
        return torch.bernoulli(torch.sigmoid(mode_logits.float())).bool()


class ShogiDecoder(ActionDecoder):
    """Move-or-drop decoder.

    legal keys (all optional, bool):
        from        [B, T, 81]   squares holding a movable own piece
        to          [B, T, 81]   destinations (given the position, not the from)
        drop_piece  [B, T, 14]   hand slots droppable (side-to-move's 7)
        drop_to     [B, T, 81]   empty squares legal for a drop

    labels keys (teacher forcing, −1 = sample from own distribution):
        dropped   [B, T] long    mode; conditions nothing but is echoed so
                                 replay commits the rolled-out mode
        from_idx  [B, T] long    conditions the to- and promotion heads
        to_idx    [B, T] long    conditions the promotion head
        piece_idx [B, T] long    conditions the drop-square head

    actions keys (for :meth:`log_prob_entropy`):
        dropped / promoted  [B, T] bool
        from_idx / to_idx / piece_idx / drop_to_idx  [B, T] long

    BOTH branches are decoded on every frame; the mode only gates which
    branch's log-prob / entropy contributes, so stored-action replay under
    teacher-forced conditioning reproduces rollout log-probs exactly.
    """

    def __init__(self, cfg: ShogiConfig):
        super().__init__()
        self.cfg = cfg
        self.n_board_tokens = cfg.board_h * cfg.board_w
        self.n_hand_slots = 2 * cfg.n_hand_piece_types

        self.mode_head = MLPHead(cfg.d_model, 1)             # drop (>0) vs move
        # MOVE branch
        self.from_head = PointerHead(cfg.d_model, n_queries=1)
        self.to_head = PointerHead(cfg.d_model, n_queries=2)
        self.promo_head = MLPHead(3 * cfg.d_model, 1)        # promote yes/no
        # DROP branch
        self.piece_head = PointerHead(cfg.d_model, n_queries=1)
        self.drop_to_head = PointerHead(cfg.d_model, n_queries=2)

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
        BT = B * T
        Nb, Nh = self.n_board_tokens, self.n_hand_slots
        board_tok = tokens[:, :Nb]
        hand_tok = tokens[:, Nb:]
        arange = torch.arange(BT, device=tokens.device)

        mode_logits = self.mode_head(ll).squeeze(-1)              # [BT]
        dropped = _pick_mode(
            mode_logits, labels.get("dropped") if labels else None, BT,
        )

        # -- MOVE branch (always decoded) --
        from_logits = self.from_head(board_tok, ll)               # [BT, Nb] raw
        from_idx = _pick_index(
            from_logits,
            labels.get("from_idx") if labels else None,
            legal.get("from") if legal else None,
            BT, Nb,
        )
        from_embed = board_tok[arange, from_idx]

        to_logits = self.to_head(board_tok, from_embed, ll)       # [BT, Nb] raw
        to_idx = _pick_index(
            to_logits,
            labels.get("to_idx") if labels else None,
            legal.get("to") if legal else None,
            BT, Nb,
        )
        to_embed = board_tok[arange, to_idx]

        promo_logits = self.promo_head(
            torch.cat([from_embed, to_embed, ll], dim=-1),
        ).squeeze(-1)                                             # [BT]

        # -- DROP branch (always decoded) --
        piece_logits = self.piece_head(hand_tok, ll)              # [BT, Nh] raw
        piece_idx = _pick_index(
            piece_logits,
            labels.get("piece_idx") if labels else None,
            legal.get("drop_piece") if legal else None,
            BT, Nh,
        )
        piece_embed = hand_tok[arange, piece_idx]

        drop_to_logits = self.drop_to_head(board_tok, piece_embed, ll)  # [BT, Nb]

        return {
            "mode_logits": mode_logits.reshape(B, T),
            "dropped": dropped.reshape(B, T),
            "from_logits": from_logits.reshape(B, T, Nb),
            "to_logits": to_logits.reshape(B, T, Nb),
            "promo_logits": promo_logits.reshape(B, T),
            "piece_logits": piece_logits.reshape(B, T, Nh),
            "drop_to_logits": drop_to_logits.reshape(B, T, Nb),
            "from_idx": from_idx.reshape(B, T),
            "to_idx": to_idx.reshape(B, T),
            "piece_idx": piece_idx.reshape(B, T),
        }

    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Mode / from / to / piece were committed in :meth:`decode`; here
        we sample the remaining leaves (promotion, drop square) and build
        the hierarchical log-prob exactly as :meth:`log_prob_entropy`
        would, so rollout and PPO update define one policy."""
        mode_logits = pred["mode_logits"].float()
        dropped = pred["dropped"].bool()
        from_idx = pred["from_idx"]
        to_idx = pred["to_idx"]
        piece_idx = pred["piece_idx"]

        with torch.no_grad():
            mode_lp, _ = bernoulli_log_prob_entropy(mode_logits, dropped)

            # MOVE branch
            from_lp = masked_log_prob(
                pred["from_logits"].float(), from_idx,
                legal.get("from") if legal else None,
            )
            to_lp = masked_log_prob(
                pred["to_logits"].float(), to_idx,
                legal.get("to") if legal else None,
            )
            promo_logits = pred["promo_logits"].float()
            promoted = torch.bernoulli(torch.sigmoid(promo_logits)).bool()
            promo_lp, _ = bernoulli_log_prob_entropy(promo_logits, promoted)

            # DROP branch
            piece_lp = masked_log_prob(
                pred["piece_logits"].float(), piece_idx,
                legal.get("drop_piece") if legal else None,
            )
            drop_to_mask = legal.get("drop_to") if legal else None
            drop_to_idx = masked_sample(pred["drop_to_logits"], drop_to_mask)
            drop_to_lp = masked_log_prob(
                pred["drop_to_logits"].float(), drop_to_idx, drop_to_mask,
            )

            total_lp = mode_lp + torch.where(
                dropped, piece_lp + drop_to_lp, from_lp + to_lp + promo_lp,
            )

        return {
            "action_dropped": dropped,
            "action_from": from_idx,
            "action_to": to_idx,
            "action_promoted": promoted,
            "action_piece": piece_idx,
            "action_drop_to": drop_to_idx,
            "action_log_prob": total_lp,
        }

    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        dropped = actions["dropped"].bool()

        mode_lp, mode_ent = bernoulli_log_prob_entropy(
            pred["mode_logits"], dropped,
        )

        # MOVE branch
        from_lp, from_ent = _log_prob_and_entropy(
            pred["from_logits"], actions["from_idx"],
            legal.get("from") if legal else None,
        )
        to_lp, to_ent = _log_prob_and_entropy(
            pred["to_logits"], actions["to_idx"],
            legal.get("to") if legal else None,
        )
        promo_lp, promo_ent = bernoulli_log_prob_entropy(
            pred["promo_logits"], actions["promoted"].bool(),
        )

        # DROP branch
        piece_lp, piece_ent = _log_prob_and_entropy(
            pred["piece_logits"], actions["piece_idx"],
            legal.get("drop_piece") if legal else None,
        )
        drop_to_lp, drop_to_ent = _log_prob_and_entropy(
            pred["drop_to_logits"], actions["drop_to_idx"],
            legal.get("drop_to") if legal else None,
        )

        total_lp = mode_lp + torch.where(
            dropped, piece_lp + drop_to_lp, from_lp + to_lp + promo_lp,
        )
        total_ent = mode_ent + torch.where(
            dropped, piece_ent + drop_to_ent, from_ent + to_ent + promo_ent,
        )

        with torch.no_grad():
            move_f = (~dropped).float()
            drop_f = dropped.float()
            n_moves = move_f.sum().clamp(min=1.0)
            n_drops = drop_f.sum().clamp(min=1.0)
            head_entropy = {
                "ent_mode": mode_ent.detach().mean(),
                "ent_from": (from_ent.detach() * move_f).sum() / n_moves,
                "ent_to": (to_ent.detach() * move_f).sum() / n_moves,
                "ent_promo": (promo_ent.detach() * move_f).sum() / n_moves,
                "ent_piece": (piece_ent.detach() * drop_f).sum() / n_drops,
                "ent_drop_to": (drop_to_ent.detach() * drop_f).sum() / n_drops,
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Mode BCE on every labeled frame; branch heads are supervised on
        every labeled frame of their true mode (the mode conditions no
        branch logits), with the genuine downstream conditioning chains
        (from→to→promo, piece→drop-square) condition-weighted exactly like
        ``FromToDecoder.il_loss``.  Denominators are clamped so degenerate
        masks give a finite zero.

        ``tgt['dropped']`` may be long with −1 = unlabeled mode; such
        frames contribute to no head.
        """
        B, T, Nb = pred["from_logits"].shape
        Nh = pred["piece_logits"].shape[-1]
        valid = seq_mask.float()

        d_lbl = tgt["dropped"].long()
        labeled = valid * (d_lbl >= 0).float()
        n_labeled = labeled.sum().clamp(min=1.0)

        # -- mode BCE --
        tgt_dropped = (d_lbl > 0).float()
        mode_bce = F.binary_cross_entropy_with_logits(
            pred["mode_logits"], tgt_dropped, reduction="none",
        ).nan_to_num(0.0)
        mode_loss = (mode_bce * labeled).sum() / n_labeled

        mode_pred_drop = (pred["mode_logits"] > 0).long()
        mode_ok = (mode_pred_drop == d_lbl).float()  # metric only

        # No branch logit conditions on the mode (from/piece heads see
        # tokens + ll only; decode always evaluates both branches), so —
        # unlike the from→to→promo chains below — branch supervision must
        # NOT be gated on the mode head being right: that would zero out
        # exactly the ambiguous move-vs-drop positions.
        move_w = labeled * (d_lbl == 0).float()
        drop_w = labeled * (d_lbl == 1).float()

        def _ce(logits: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
            safe = idx.long().clamp(0, n - 1)
            return F.cross_entropy(
                logits.reshape(-1, n), safe.reshape(-1), reduction="none",
            ).reshape(B, T).nan_to_num(0.0)

        # -- MOVE branch: from → to → promo conditioning chain --
        tgt_from = tgt["from_idx"].long().clamp(0, Nb - 1)
        tgt_to = tgt["to_idx"].long().clamp(0, Nb - 1)

        n_move = move_w.sum().clamp(min=1.0)
        from_loss = (_ce(pred["from_logits"], tgt_from, Nb) * move_w).sum() / n_move

        to_w = move_w * (pred["from_idx"] == tgt_from).float()
        n_to = to_w.sum().clamp(min=1.0)
        to_loss = (_ce(pred["to_logits"], tgt_to, Nb) * to_w).sum() / n_to

        promo_w = to_w * (pred["to_idx"] == tgt_to).float()
        n_promo = promo_w.sum().clamp(min=1.0)
        tgt_promo = tgt["promoted"].float()
        promo_bce = F.binary_cross_entropy_with_logits(
            pred["promo_logits"], tgt_promo, reduction="none",
        ).nan_to_num(0.0)
        promo_loss = (promo_bce * promo_w).sum() / n_promo

        # -- DROP branch: piece → drop-square conditioning chain --
        tgt_piece = tgt["piece_idx"].long().clamp(0, Nh - 1)
        tgt_drop_to = tgt["drop_to_idx"].long().clamp(0, Nb - 1)

        n_drop = drop_w.sum().clamp(min=1.0)
        piece_loss = (_ce(pred["piece_logits"], tgt_piece, Nh) * drop_w).sum() / n_drop

        drop_to_w = drop_w * (pred["piece_idx"] == tgt_piece).float()
        n_drop_to = drop_to_w.sum().clamp(min=1.0)
        drop_to_loss = (
            _ce(pred["drop_to_logits"], tgt_drop_to, Nb) * drop_to_w
        ).sum() / n_drop_to

        loss = mode_loss + from_loss + to_loss + promo_loss + piece_loss + drop_to_loss

        with torch.no_grad():
            mode_acc = (mode_ok * labeled).sum() / n_labeled
            from_ok = pred["from_logits"].argmax(-1) == tgt_from
            to_ok = pred["to_logits"].argmax(-1) == tgt_to
            promo_ok = (pred["promo_logits"] > 0) == tgt["promoted"].bool()
            piece_ok = pred["piece_logits"].argmax(-1) == tgt_piece
            drop_to_ok = pred["drop_to_logits"].argmax(-1) == tgt_drop_to
            is_drop = d_lbl == 1
            answer_ok = (mode_pred_drop == d_lbl) & torch.where(
                is_drop, piece_ok & drop_to_ok, from_ok & to_ok & promo_ok)
            metrics = {
                "answer_correct": answer_ok,
                # frames with an unlabeled mode (−1) have no answer
                "answer_mask": seq_mask.bool() & (d_lbl >= 0),
                "mode_loss": mode_loss.detach(),
                "from_loss": from_loss.detach(),
                "to_loss": to_loss.detach(),
                "promo_loss": promo_loss.detach(),
                "piece_loss": piece_loss.detach(),
                "drop_to_loss": drop_to_loss.detach(),
                "mode_acc": mode_acc,
                "from_acc": (from_ok.float() * move_w).sum() / n_move,
                "to_acc": (to_ok.float() * to_w).sum() / n_to,
                "promo_acc": (promo_ok.float() * promo_w).sum() / n_promo,
                "piece_acc": (piece_ok.float() * drop_w).sum() / n_drop,
                "drop_to_acc": (drop_to_ok.float() * drop_to_w).sum() / n_drop_to,
            }
        return {"loss": loss, **metrics}
