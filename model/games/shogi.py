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

Legality is conditional: the to-squares depend on the from-square, the
promotion options on (from, to), the drop squares on the hand piece, and
the mode on whether any board move / drop exists.  :class:`ShogiDecoder`
takes full tables (``to_cond`` …, e.g. from :func:`shogi_legal_tables` in
an environment) or the rows for the labelled conditioning (``to_of`` …, as
the converter writes them: ``tgt.legal_*``, converter_version 2), echoes
the resolved masks, and scores every path (sampling, replay, IL loss)
under them.
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
    ``shogi_global_features = 0`` disables the global-vector projection;
    the default 4 is the converter_version 1 layout (move number, mover is
    black, in check, 1) and stays the default so old checkpoints load
    unchanged; converter_version 2 shards carry 7 (+ position occurred
    ≥1 / ≥2 / ≥3 times before — shard meta ``global_features``).
    """

    board_h: int = 9
    board_w: int = 9
    piece_channels: int = 28        # 14 piece types × 2 sides is the typical layout
    n_hand_piece_types: int = 7     # P L N S G B R (droppable types)
    hand_feature_dim: int = 4       # count, count>0, normalized count, side-to-move
    shogi_global_features: int = 4  # 4 = converter v1; v2 shards: 7 (+ repetition flags)


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


# ---------------------------------------------------------------------------
# Legality masks: resolution of conditional tables, masked 2-class Bernoullis
# ---------------------------------------------------------------------------

def _as_mask(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Legality masks may arrive as float {0, 1} (bit-packed shard tensors)."""
    if m is None:
        return None
    return m if m.dtype == torch.bool else m.bool()


def _two_class(logit: torch.Tensor) -> torch.Tensor:
    """Bernoulli logit l → 2-class logits [0, l] (class 1 = drop / promote):
    softmax([0, l])[1] = sigmoid(l), so the unmasked distribution is the
    legacy Bernoulli; a legality mask can then remove either option."""
    return torch.stack([torch.zeros_like(logit), logit], dim=-1)


def _masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Masked logits with the all-masked guard (index 0 stays finite), the
    same guard ``_log_prob_and_entropy`` / ``masked_sample`` apply."""
    if mask is None:
        return logits
    safe = logits.masked_fill(~mask, float("-inf"))
    all_masked = ~mask.any(dim=-1)
    safe = safe.clone()
    safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
    return safe


def _masked_argmax(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits.argmax(-1)
    return _masked_logits(logits.float(), mask).argmax(-1)


def _masked_ce(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(per-frame CE of ``target`` under the masked softmax, target legal).

    An illegal target (outside ``mask``) contributes 0 and no gradient: it
    is scored at a legal stand-in index so nothing is infinite, then zeroed.
    """
    safe = _masked_logits(logits.float(), mask)
    legal = mask.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    idx = torch.where(legal, target, safe.detach().argmax(-1))
    ce = -F.log_softmax(safe, dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return torch.where(legal, ce, torch.zeros_like(ce)).nan_to_num(0.0), legal


def _flat_label(labels: Optional[Dict[str, torch.Tensor]], key: str,
                n: int) -> Optional[torch.Tensor]:
    if not labels or labels.get(key) is None:
        return None
    return labels[key].reshape(n).long()


def _resolve_mask(
    legal: Optional[Dict[str, torch.Tensor]],
    name: str,
    idx: Tuple[torch.Tensor, ...],
    lbl: Tuple[Optional[torch.Tensor], ...],
    n: int,
    width: int,
    cond_dims: Tuple[int, ...],
) -> Optional[torch.Tensor]:
    """Mask [n, width] of a conditioned head, rows committed at ``idx``.

    Precedence ``<name>_cond`` (full table, gathered at ``idx``) >
    ``<name>_of`` (the row for the labelled conditioning: used where every
    committed index equals its label; elsewhere the flat ``<name>`` mask if
    given, else unconstrained) > flat ``<name>`` > None.
    """
    if not legal:
        return None
    cond = _as_mask(legal.get(f"{name}_cond"))
    if cond is not None:
        c = cond.reshape(n, *cond_dims, width)
        rows = torch.arange(n, device=c.device)
        return c[(rows, *idx)]
    flat = _as_mask(legal.get(name))
    if flat is not None:
        flat = flat.reshape(n, width)
    of = _as_mask(legal.get(f"{name}_of"))
    if of is None:
        return flat
    applies = torch.ones(n, dtype=torch.bool, device=of.device)
    for committed, label in zip(idx, lbl):
        if label is None:
            applies = torch.zeros_like(applies)
            break
        applies = applies & (label >= 0) & (committed == label)
    fallback = flat
    if fallback is None:
        fallback = torch.ones(n, width, dtype=torch.bool, device=of.device)
    return torch.where(applies.unsqueeze(-1), of.reshape(n, width), fallback)


def _flat_mask(legal: Optional[Dict[str, torch.Tensor]], key: str, n: int,
               width: int) -> Optional[torch.Tensor]:
    m = _as_mask(legal.get(key)) if legal else None
    return m.reshape(n, width) if m is not None else None


def shogi_legal_tables(
    moves,
    n_squares: int = 81,
    n_hand_slots: int = 14,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Full conditional legality tables of ONE position for :class:`ShogiDecoder`.

    ``moves``: iterable of ``(dropped, from, to, promote, piece, drop_to)``
    tuples in POV indices — board moves use (from, to, promote), drops use
    (piece = hand slot, drop_to).  Returns ``mode`` [2], ``from`` [N],
    ``to_cond`` [N, N], ``promo_cond`` [N, N, 2] ([no promotion, promotion]),
    ``drop_piece`` [n_hand_slots], ``drop_to_cond`` [n_hand_slots, N]
    (bool); stack positions along leading dims for a batch.
    """
    N, H = n_squares, n_hand_slots
    z = dict(dtype=torch.bool, device=device)
    out = {
        "mode": torch.zeros(2, **z),
        "from": torch.zeros(N, **z),
        "to_cond": torch.zeros(N, N, **z),
        "promo_cond": torch.zeros(N, N, 2, **z),
        "drop_piece": torch.zeros(H, **z),
        "drop_to_cond": torch.zeros(H, N, **z),
    }
    for dropped, frm, to, promote, piece, drop_to in moves:
        if dropped:
            out["mode"][1] = True
            out["drop_piece"][piece] = True
            out["drop_to_cond"][piece, drop_to] = True
        else:
            out["mode"][0] = True
            out["from"][frm] = True
            out["to_cond"][frm, to] = True
            out["promo_cond"][frm, to, int(bool(promote))] = True
    return out


# sample() output key → legal key of the stored-action replay
_REPLAY_LEGAL = {
    "legal_mode": "mode",
    "legal_from": "from",
    "legal_to": "to_of",
    "legal_promo": "promo_of",
    "legal_drop_piece": "drop_piece",
    "legal_drop_to": "drop_to_of",
}


def shogi_replay_legal(sample_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The legal dict that replays a rollout exactly: the masks
    :meth:`ShogiDecoder.sample` used (``legal_*`` keys, any leading dims —
    e.g. stacked over time) as ``mode`` / ``from`` / ``to_of`` / ``promo_of``
    / ``drop_piece`` / ``drop_to_of``.  Replay with labels = the stored
    ``dropped`` / ``from_idx`` / ``to_idx`` / ``piece_idx`` so every ``_of``
    row applies."""
    return {v: sample_out[k] for k, v in _REPLAY_LEGAL.items() if k in sample_out}


class ShogiDecoder(ActionDecoder):
    """Move-or-drop decoder.

    legal keys (all optional; bool, or float {0, 1} as loaded from shards):
        mode          [B, T, 2]          [board move allowed, drop allowed]
        from          [B, T, 81]         from-squares of legal board moves
        to            [B, T, 81]         legacy: destinations given the position only
        to_cond       [B, T, 81, 81]     destinations per from-square
        to_of         [B, T, 81]         destinations for labels['from_idx']
        promo_cond    [B, T, 81, 81, 2]  [no promotion, promotion] per (from, to)
        promo_of      [B, T, 2]          for (labels['from_idx'], labels['to_idx'])
        promo         [B, T, 2]          unconditional (fallback of promo_of)
        drop_piece    [B, T, 14]         droppable hand slots (side-to-move's 7)
        drop_to       [B, T, 81]         legacy: legal drop squares, any piece
        drop_to_cond  [B, T, 14, 81]     drop squares per hand slot
        drop_to_of    [B, T, 81]         drop squares for labels['piece_idx']

    Conditioned masks resolve at the committed indices with precedence
    ``_cond`` > ``_of`` > flat.  An ``_of`` row only applies where the
    committed index equals its label (label −1 / absent: the flat mask if
    given, else unconstrained) — the converter's ``tgt.legal_*_of`` rows are
    valid for the ground-truth conditioning, i.e. under teacher forcing.
    With a ``mode`` (promotion) mask the mode (promotion) Bernoulli becomes
    the 2-class categorical over [0, logit] (class 1 = drop / promote), so a
    forced option has log-prob 0 and entropy 0; without it the legacy
    Bernoulli path (and RNG stream) is unchanged.

    ``decode`` echoes the masks it resolved (only those that exist):
    ``mode_mask`` [B, T, 2], ``from_mask``, ``to_mask`` [B, T, 81],
    ``promo_mask`` [B, T, 2], ``piece_mask`` [B, T, 14], ``drop_to_mask``
    [B, T, 81]; ``sample`` / ``log_prob_entropy`` / ``il_loss`` read the
    echoes first, so all of them score one distribution.

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
    teacher-forced conditioning reproduces rollout log-probs exactly.  RL
    replay: ``legal = shogi_replay_legal(sample outputs)`` with labels = the
    stored dropped / from_idx / to_idx / piece_idx.
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

    def _masks(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]],
        from_idx: torch.Tensor,
        to_idx: torch.Tensor,
        piece_idx: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Masks for scoring committed / stored actions: decode's echoes
        first, else resolved from ``legal`` with the given indices as both
        the committed rows and the labels (every ``_of`` row applies)."""
        lead = from_idx.shape
        n = from_idx.numel()
        Nb, Nh = self.n_board_tokens, self.n_hand_slots
        f, t, p = (x.reshape(n).long() for x in (from_idx, to_idx, piece_idx))

        def shaped(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return m.reshape(*lead, m.shape[-1]) if m is not None else None

        resolved = {
            "mode": lambda: _flat_mask(legal, "mode", n, 2),
            "from": lambda: _flat_mask(legal, "from", n, Nb),
            "to": lambda: _resolve_mask(legal, "to", (f,), (f,), n, Nb, (Nb,)),
            "promo": lambda: _resolve_mask(legal, "promo", (f, t), (f, t), n, 2, (Nb, Nb)),
            "piece": lambda: _flat_mask(legal, "drop_piece", n, Nh),
            "drop_to": lambda: _resolve_mask(legal, "drop_to", (p,), (p,), n, Nb, (Nh,)),
        }
        out: Dict[str, Optional[torch.Tensor]] = {}
        for head, fn in resolved.items():
            echo = pred.get(f"{head}_mask")
            out[head] = _as_mask(echo) if echo is not None else shaped(fn())
        return out

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

        def label(key):
            return labels.get(key) if labels else None

        mode_logits = self.mode_head(ll).squeeze(-1)              # [BT]
        mode_mask = _flat_mask(legal, "mode", BT, 2)
        if mode_mask is None:
            dropped = _pick_mode(mode_logits, label("dropped"), BT)
        else:
            with torch.no_grad():
                dropped = _pick_index(
                    _two_class(mode_logits.float()), label("dropped"), mode_mask, BT, 2,
                ).bool()

        # -- MOVE branch (always decoded) --
        from_mask = _flat_mask(legal, "from", BT, Nb)
        from_logits = self.from_head(board_tok, ll)               # [BT, Nb] raw
        from_idx = _pick_index(from_logits, label("from_idx"), from_mask, BT, Nb)
        from_embed = board_tok[arange, from_idx]

        lbl_from = _flat_label(labels, "from_idx", BT)
        to_mask = _resolve_mask(legal, "to", (from_idx,), (lbl_from,), BT, Nb, (Nb,))
        to_logits = self.to_head(board_tok, from_embed, ll)       # [BT, Nb] raw
        to_idx = _pick_index(to_logits, label("to_idx"), to_mask, BT, Nb)
        to_embed = board_tok[arange, to_idx]

        lbl_to = _flat_label(labels, "to_idx", BT)
        promo_mask = _resolve_mask(
            legal, "promo", (from_idx, to_idx), (lbl_from, lbl_to), BT, 2, (Nb, Nb),
        )
        promo_logits = self.promo_head(
            torch.cat([from_embed, to_embed, ll], dim=-1),
        ).squeeze(-1)                                             # [BT]

        # -- DROP branch (always decoded) --
        piece_mask = _flat_mask(legal, "drop_piece", BT, Nh)
        piece_logits = self.piece_head(hand_tok, ll)              # [BT, Nh] raw
        piece_idx = _pick_index(piece_logits, label("piece_idx"), piece_mask, BT, Nh)
        piece_embed = hand_tok[arange, piece_idx]

        lbl_piece = _flat_label(labels, "piece_idx", BT)
        drop_to_mask = _resolve_mask(
            legal, "drop_to", (piece_idx,), (lbl_piece,), BT, Nb, (Nh,),
        )
        drop_to_logits = self.drop_to_head(board_tok, piece_embed, ll)  # [BT, Nb]

        out = {
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
        for key, m in (("mode_mask", mode_mask), ("from_mask", from_mask),
                       ("to_mask", to_mask), ("promo_mask", promo_mask),
                       ("piece_mask", piece_mask), ("drop_to_mask", drop_to_mask)):
            if m is not None:
                out[key] = m.reshape(B, T, m.shape[-1])
        return out

    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Mode / from / to / piece were committed in :meth:`decode`; here
        we sample the remaining leaves (promotion, drop square) and build
        the hierarchical log-prob exactly as :meth:`log_prob_entropy`
        would, so rollout and PPO update define one policy.  The masks used
        are returned as ``legal_*`` (see :func:`shogi_replay_legal`)."""
        mode_logits = pred["mode_logits"].float()
        dropped = pred["dropped"].bool()
        from_idx = pred["from_idx"]
        to_idx = pred["to_idx"]
        piece_idx = pred["piece_idx"]

        with torch.no_grad():
            m = self._masks(pred, legal, from_idx, to_idx, piece_idx)
            if m["mode"] is None:
                mode_lp, _ = bernoulli_log_prob_entropy(mode_logits, dropped)
            else:
                mode_lp, _ = _log_prob_and_entropy(
                    _two_class(mode_logits), dropped.long(), m["mode"],
                )

            # MOVE branch
            from_lp = masked_log_prob(pred["from_logits"].float(), from_idx, m["from"])
            to_lp = masked_log_prob(pred["to_logits"].float(), to_idx, m["to"])
            promo_logits = pred["promo_logits"].float()
            if m["promo"] is None:
                promoted = torch.bernoulli(torch.sigmoid(promo_logits)).bool()
                promo_lp, _ = bernoulli_log_prob_entropy(promo_logits, promoted)
            else:
                promo2 = _two_class(promo_logits)
                promoted = masked_sample(promo2, m["promo"]).bool()
                promo_lp, _ = _log_prob_and_entropy(promo2, promoted.long(), m["promo"])

            # DROP branch
            piece_lp = masked_log_prob(pred["piece_logits"].float(), piece_idx, m["piece"])
            drop_to_idx = masked_sample(pred["drop_to_logits"], m["drop_to"])
            drop_to_lp = masked_log_prob(
                pred["drop_to_logits"].float(), drop_to_idx, m["drop_to"],
            )

            total_lp = mode_lp + torch.where(
                dropped, piece_lp + drop_to_lp, from_lp + to_lp + promo_lp,
            )

        out = {
            "action_dropped": dropped,
            "action_from": from_idx,
            "action_to": to_idx,
            "action_promoted": promoted,
            "action_piece": piece_idx,
            "action_drop_to": drop_to_idx,
            "action_log_prob": total_lp,
        }
        for head, key in (("mode", "legal_mode"), ("from", "legal_from"),
                          ("to", "legal_to"), ("promo", "legal_promo"),
                          ("piece", "legal_drop_piece"), ("drop_to", "legal_drop_to")):
            if m[head] is not None:
                out[key] = m[head]
        return out

    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        dropped = actions["dropped"].bool()
        m = self._masks(pred, legal, actions["from_idx"], actions["to_idx"],
                        actions["piece_idx"])

        if m["mode"] is None:
            mode_lp, mode_ent = bernoulli_log_prob_entropy(
                pred["mode_logits"], dropped,
            )
        else:
            mode_lp, mode_ent = _log_prob_and_entropy(
                _two_class(pred["mode_logits"].float()), dropped.long(), m["mode"],
            )

        # MOVE branch
        from_lp, from_ent = _log_prob_and_entropy(
            pred["from_logits"], actions["from_idx"], m["from"],
        )
        to_lp, to_ent = _log_prob_and_entropy(
            pred["to_logits"], actions["to_idx"], m["to"],
        )
        if m["promo"] is None:
            promo_lp, promo_ent = bernoulli_log_prob_entropy(
                pred["promo_logits"], actions["promoted"].bool(),
            )
        else:
            promo_lp, promo_ent = _log_prob_and_entropy(
                _two_class(pred["promo_logits"].float()),
                actions["promoted"].long(), m["promo"],
            )

        # DROP branch
        piece_lp, piece_ent = _log_prob_and_entropy(
            pred["piece_logits"], actions["piece_idx"], m["piece"],
        )
        drop_to_lp, drop_to_ent = _log_prob_and_entropy(
            pred["drop_to_logits"], actions["drop_to_idx"], m["drop_to"],
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

    def _il_parts(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Per-head IL terms (shared by :meth:`il_loss` and
        :meth:`il_head_terms`) + detached metrics.

        Without echoed masks every op is the legacy one (bit-identical
        loss); a head with an echoed mask uses the masked CE (mode /
        promotion as 2-class [0, logit]) and drops frames whose target is
        outside the mask from its weight and denominator."""
        B, T, Nb = pred["from_logits"].shape
        Nh = pred["piece_logits"].shape[-1]
        valid = seq_mask.float()

        d_lbl = tgt["dropped"].long()
        labeled = valid * (d_lbl >= 0).float()

        # No branch logit conditions on the mode (from/piece heads see
        # tokens + ll only; decode always evaluates both branches), so —
        # unlike the from→to→promo chains below — branch supervision must
        # NOT be gated on the mode head being right: that would zero out
        # exactly the ambiguous move-vs-drop positions.
        move_w = labeled * (d_lbl == 0).float()
        drop_w = labeled * (d_lbl == 1).float()

        tgt_from = tgt["from_idx"].long().clamp(0, Nb - 1)
        tgt_to = tgt["to_idx"].long().clamp(0, Nb - 1)
        tgt_piece = tgt["piece_idx"].long().clamp(0, Nh - 1)
        tgt_drop_to = tgt["drop_to_idx"].long().clamp(0, Nb - 1)
        # genuine downstream conditioning chains (from→to→promo,
        # piece→drop-square), condition-weighted like FromToDecoder.il_loss
        to_w = move_w * (pred["from_idx"] == tgt_from).float()
        promo_w = to_w * (pred["to_idx"] == tgt_to).float()
        drop_to_w = drop_w * (pred["piece_idx"] == tgt_piece).float()

        masks = {h: _as_mask(pred.get(f"{h}_mask"))
                 for h in ("mode", "from", "to", "promo", "piece", "drop_to")}
        illegal = torch.zeros(B, T, dtype=torch.bool, device=d_lbl.device)

        def term(logits, ce, weight, mask):
            t = {"logits": logits, "ce": ce, "weight": weight,
                 "norm": weight.sum().clamp(min=1.0)}
            if mask is not None:
                t["mask"] = mask
            return t

        def categorical(logits, target, base_w, mask):
            nonlocal illegal
            n = logits.shape[-1]
            if mask is None:
                ce = F.cross_entropy(
                    logits.reshape(-1, n), target.reshape(-1), reduction="none",
                ).reshape(B, T).nan_to_num(0.0)
                return term(logits, ce, base_w, None)
            ce, ok = _masked_ce(logits, target, mask)
            illegal = illegal | ((base_w > 0) & ~ok)
            return term(logits, ce, base_w * ok.float(), mask)

        def binary(logit, target, base_w, mask):
            nonlocal illegal
            logits = _two_class(logit)
            if mask is None:
                ce = F.binary_cross_entropy_with_logits(
                    logit, target.float(), reduction="none",
                ).nan_to_num(0.0)
                return term(logits, ce, base_w, None)
            ce, ok = _masked_ce(logits, target.long(), mask)
            illegal = illegal | ((base_w > 0) & ~ok)
            return term(logits, ce, base_w * ok.float(), mask)

        tgt_dropped = d_lbl > 0
        tgt_promo = tgt["promoted"].bool()
        heads = {
            "mode": binary(pred["mode_logits"], tgt_dropped, labeled, masks["mode"]),
            "from": categorical(pred["from_logits"], tgt_from, move_w, masks["from"]),
            "to": categorical(pred["to_logits"], tgt_to, to_w, masks["to"]),
            "promo": binary(pred["promo_logits"], tgt_promo, promo_w, masks["promo"]),
            "piece": categorical(pred["piece_logits"], tgt_piece, drop_w, masks["piece"]),
            "drop_to": categorical(pred["drop_to_logits"], tgt_drop_to, drop_to_w,
                                   masks["drop_to"]),
        }

        with torch.no_grad():
            n_labeled = labeled.sum().clamp(min=1.0)
            n_move = move_w.sum().clamp(min=1.0)
            n_to = to_w.sum().clamp(min=1.0)
            n_promo = promo_w.sum().clamp(min=1.0)
            n_drop = drop_w.sum().clamp(min=1.0)
            n_drop_to = drop_to_w.sum().clamp(min=1.0)

            if masks["mode"] is None:
                mode_pred_drop = (pred["mode_logits"] > 0).long()
            else:
                mode_pred_drop = _masked_argmax(heads["mode"]["logits"], masks["mode"])
            if masks["promo"] is None:
                promo_ok = (pred["promo_logits"] > 0) == tgt_promo
            else:
                promo_pred = _masked_argmax(heads["promo"]["logits"], masks["promo"])
                promo_ok = promo_pred.bool() == tgt_promo
            mode_ok = (mode_pred_drop == d_lbl).float()
            from_ok = _masked_argmax(pred["from_logits"], masks["from"]) == tgt_from
            to_ok = _masked_argmax(pred["to_logits"], masks["to"]) == tgt_to
            piece_ok = _masked_argmax(pred["piece_logits"], masks["piece"]) == tgt_piece
            drop_to_ok = _masked_argmax(pred["drop_to_logits"], masks["drop_to"]) == tgt_drop_to
            is_drop = d_lbl == 1
            answer_ok = (mode_pred_drop == d_lbl) & torch.where(
                is_drop, piece_ok & drop_to_ok, from_ok & to_ok & promo_ok)
            metrics = {
                "answer_correct": answer_ok,
                # frames with an unlabeled mode (−1) have no answer
                "answer_mask": seq_mask.bool() & (d_lbl >= 0),
                "mode_acc": (mode_ok * labeled).sum() / n_labeled,
                "from_acc": (from_ok.float() * move_w).sum() / n_move,
                "to_acc": (to_ok.float() * to_w).sum() / n_to,
                "promo_acc": (promo_ok.float() * promo_w).sum() / n_promo,
                "piece_acc": (piece_ok.float() * drop_w).sum() / n_drop,
                "drop_to_acc": (drop_to_ok.float() * drop_to_w).sum() / n_drop_to,
            }
            if any(m is not None for m in masks.values()):
                # labeled frames whose ground truth lies outside a resolved
                # mask on a supervised head (0 on consistent data)
                metrics["illegal_target_frac"] = (illegal.float() * labeled).sum() / n_labeled
        return heads, metrics

    def il_head_terms(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Per-head decomposition of :meth:`il_loss`:
        ``il_loss['loss'] == Σ_head (ce · weight).sum() / norm``.

        Each head: ``logits`` [B, T, K] raw categorical logits (the mode and
        promotion Bernoullis as 2-class [0, logit], class 1 = drop /
        promote), ``ce`` [B, T] the hard-target loss il_loss uses, ``weight``
        [B, T] (0 where the head is not supervised), ``norm`` (scalar
        denominator) and, when decode resolved one, ``mask`` [B, T, K] bool.

        Heads and the soft-target distribution each is conditioned like
        (``<head>_probs`` in tgt):
            mode     [2]   (P(move), P(drop))                 weight: labeled frames
            from     [81]  p(from | move)                      move frames
            to       [81]  p(to | move, from = label)          move frames, committed from = label
            promo    [2]   p(promotion | from, to = labels)    ... and committed to = label
            piece    [14]  p(hand slot | drop)                 drop frames
            drop_to  [81]  p(square | drop, piece = label)     drop frames, committed piece = label
        """
        heads, _ = self._il_parts(pred, tgt, seq_mask)
        return heads

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

        With echoed legality masks (decode got ``legal``) each masked head
        uses the masked CE (mode / promotion as masked 2-class CE), frames
        whose target is illegal are dropped from that head
        (``illegal_target_frac``), and accuracies use the masked argmax.

        ``tgt['dropped']`` may be long with −1 = unlabeled mode; such
        frames contribute to no head.
        """
        heads, metrics = self._il_parts(pred, tgt, seq_mask)
        losses = {h: (t["ce"] * t["weight"]).sum() / t["norm"] for h, t in heads.items()}
        loss = (losses["mode"] + losses["from"] + losses["to"] + losses["promo"]
                + losses["piece"] + losses["drop_to"])
        for h, v in losses.items():
            metrics[f"{h}_loss"] = v.detach()
        return {"loss": loss, **metrics}
