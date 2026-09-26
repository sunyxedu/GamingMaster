"""Board-game adapters (chess-like and Go-like) for the game-agnostic core.

Two action-space families share one :class:`BoardEncoder`:

  FromToDecoder      chess/shogi-style "pick a square, then a destination
                     conditioned on it" (+ optional promotion) — the
                     conditioning discipline mirrors the generals
                     Source → Direction decoder.
  PlacementDecoder   Go/Gomoku-style "place a stone or pass" — a flat
                     pointer with the same pass-gated hierarchical
                     log-prob as generals.

Sampling, PPO replay and IL teacher forcing all evaluate the SAME
distributions (masked identically), which is what makes PPO ratios exact.
FromToDecoder legality can be conditional (to-squares given the chosen
from-square, promotions given the chosen (from, to)): exact tables for RL
rollouts, label-conditioned masks for IL and replay — see its docstring.
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
    masked_log_prob,
    masked_sample,
)


@dataclass
class BoardConfig(ModelConfig):
    """Board-game dimensions on top of the shared core config.

    ``board_global_features=0`` disables the global-vector projection;
    ``n_promotions`` ∈ {0, 1} disables the promotion head (a single
    "promotion" class carries no information — index 0 means "none");
    ``allow_pass=False`` removes the PlacementDecoder pass head entirely
    (games like gomoku have no pass move — the placement legal mask
    cannot suppress the pass Bernoulli, only this flag can).
    """

    board_h: int = 8
    board_w: int = 8
    piece_channels: int = 14       # input planes (piece type × color, history, ...)
    # side-to-move, castling rights, move counters, ... — 8 = chess
    # converter v2 (the legacy default); v3 shards carry 10 (+ repetition
    # flags) and say so in their shard meta ``global_features``
    board_global_features: int = 8
    n_promotions: int = 5           # 0 = none, 1..4 = promotion piece
    allow_pass: bool = True         # PlacementDecoder only; False = no pass action


def _masked_policy_logits(
    logits: torch.Tensor, mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Differentiable masked logits with the degenerate-row guard.

    Mirrors :func:`model.games.base.masked_log_prob`'s masking exactly so
    that entropy and log-prob are computed under the SAME distribution the
    sampler used — otherwise the PPO entropy bonus would push a different
    policy than the ratio measures.
    """
    if mask is None:
        return logits
    safe = logits.masked_fill(~mask, float("-inf"))
    all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
    if all_masked.any():
        # Column-slice write — see base.safe_masked_logits.
        safe = safe.clone()
        safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
    return safe


def _log_prob_and_entropy(
    logits: torch.Tensor, index: torch.Tensor, mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(log-prob of ``index``, entropy) under optionally-masked logits.

    One log_softmax serves both quantities so they cannot drift apart.
    Computed in float32 (differentiable cast) so the PPO replay path
    evaluates the same distribution as the fp32 rollout bookkeeping in
    ``sample()`` even when the logits arrive in bf16.

    The entropy term clamps −inf masked entries BEFORE the p·log p
    product: ``nan_to_num`` on the product cleans only the forward —
    mul's backward still computes 0·(−inf) = NaN and log_softmax's
    backward spreads it to the unmasked positions.  exp underflows to 0
    long before the clamp value, so the entropy itself is unchanged.
    """
    log_p = F.log_softmax(_masked_policy_logits(logits.float(), mask), dim=-1)
    lp = log_p.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    ent_log_p = log_p.clamp_min(-1e4)
    ent = -(torch.exp(ent_log_p) * ent_log_p).nan_to_num(0.0).sum(-1)
    return lp, ent


def _pick_index(
    logits: torch.Tensor,               # [BT, N]
    label: Optional[torch.Tensor],      # [B, T] long or None
    legal_mask: Optional[torch.Tensor],  # broadcastable to [BT, N] or None
    BT: int,
    N: int,
) -> torch.Tensor:
    """Teacher-forced or sampled conditioning index (no grad through sampling).

    Label −1 means "sample from the model's own distribution" — the mixed
    mode IL uses so downstream heads see realistic conditioning early in
    training (same convention as the generals source head).
    """
    lm = legal_mask.reshape(BT, N) if legal_mask is not None else None
    if label is not None:
        lbl = label.reshape(BT).long()
        gt = lbl.clamp(0, N - 1)
        needs_sample = lbl < 0
        if needs_sample.any():
            sampled = masked_sample(logits, lm)
            return torch.where(needs_sample, sampled, gt)
        return gt
    return masked_sample(logits, lm)


# ----------------------------------------------------------------------
# From → To (→ Promotion) legality: conditional masks
# ----------------------------------------------------------------------

def _legal_mask(
    legal: Optional[Dict[str, torch.Tensor]], key: str,
) -> Optional[torch.Tensor]:
    """``legal[key]`` as a bool tensor, or None.  IL masks arrive as float
    {0, 1} (gm-shard bit tensors); a bool mask passes through untouched."""
    m = legal.get(key) if legal else None
    if m is None:
        return None
    return m if m.dtype == torch.bool else m.bool()


def _gather_row(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``table[..., idx, :]``: table [..., R, K], idx [...] → [..., K]."""
    R, K = table.shape[-2:]
    i = idx.long().clamp(0, R - 1)
    return table.gather(-2, i[..., None, None].expand(*i.shape, 1, K)).squeeze(-2)


def _labelled_rows(
    committed: torch.Tensor, label: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """[...] bool: rows whose committed index IS the (non-negative) label;
    None without labels."""
    if label is None:
        return None
    lbl = label.reshape(committed.shape).long()
    return (lbl >= 0) & (committed == lbl)


def _where_labelled(mask: torch.Tensor, rows: Optional[torch.Tensor]) -> torch.Tensor:
    """An ``X_of`` mask is only valid for the labelled conditioning index:
    rows that committed to something else (or carry no label) are
    unconstrained (all True)."""
    if rows is None:
        return torch.ones_like(mask)
    return mask | ~rows.unsqueeze(-1)


def _resolve_to_mask(
    legal: Optional[Dict[str, torch.Tensor]],
    from_idx: torch.Tensor,                  # [...] committed from-squares
    from_label: Optional[torch.Tensor],      # labels["from_idx"] or None
) -> Optional[torch.Tensor]:
    """To-head mask [..., N] for the committed from-squares.

    Precedence ``to_cond`` (exact table, row = from-square) > ``to_of``
    (valid for the labelled from-square only) > ``to`` (legacy
    unconditional mask).  ``legal`` tensors carry ``from_idx``'s leading
    dims in any flattening ([B, T, ...] against [B*T] is fine)."""
    cond = _legal_mask(legal, "to_cond")
    if cond is not None:
        N = cond.shape[-1]
        return _gather_row(cond.reshape(*from_idx.shape, cond.shape[-2], N), from_idx)
    of = _legal_mask(legal, "to_of")
    if of is not None:
        of = of.reshape(*from_idx.shape, of.shape[-1])
        return _where_labelled(of, _labelled_rows(from_idx, from_label))
    flat = _legal_mask(legal, "to")
    if flat is not None:
        return flat.reshape(*from_idx.shape, flat.shape[-1])
    return None


def _resolve_promotion_mask(
    legal: Optional[Dict[str, torch.Tensor]],
    from_idx: torch.Tensor,
    to_idx: torch.Tensor,
    from_label: Optional[torch.Tensor],
    to_label: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Promotion mask [..., P] for the committed (from, to): precedence
    ``promotion_cond`` (exact [from, to] table) > ``promotion_of`` (valid
    for the labelled (from, to) only)."""
    cond = _legal_mask(legal, "promotion_cond")
    if cond is not None:
        Nf, Nt, P = cond.shape[-3:]
        flat = cond.reshape(*from_idx.shape, Nf * Nt, P)
        pair = from_idx.long().clamp(0, Nf - 1) * Nt + to_idx.long().clamp(0, Nt - 1)
        return _gather_row(flat, pair)
    of = _legal_mask(legal, "promotion_of")
    if of is not None:
        of = of.reshape(*from_idx.shape, of.shape[-1])
        rows = None
        if from_label is not None and to_label is not None:
            rows = (_labelled_rows(from_idx, from_label)
                    & _labelled_rows(to_idx, to_label))
        return _where_labelled(of, rows)
    return None


def fromto_legal_tables(
    moves, n_squares: int, n_promotions: int, device=None,
) -> Dict[str, torch.Tensor]:
    """Exact FromToDecoder legality tables of ONE position from its legal
    moves, an iterable of (from, to, promotion) indices (promotion 0 =
    none): ``from`` [N], ``to_cond`` [N, N] (row = from-square),
    ``promotion_cond`` [N, N, P] (P = max(n_promotions, 1)), all bool.
    Stack over a batch for :meth:`GameNetwork.step` (envs / tests)."""
    P = max(int(n_promotions), 1)
    frm = torch.zeros(n_squares, dtype=torch.bool)
    to_cond = torch.zeros(n_squares, n_squares, dtype=torch.bool)
    promo = torch.zeros(n_squares, n_squares, P, dtype=torch.bool)
    idx = torch.as_tensor([tuple(m) for m in moves], dtype=torch.long).reshape(-1, 3)
    if idx.numel():
        frm[idx[:, 0]] = True
        to_cond[idx[:, 0], idx[:, 1]] = True
        promo[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    out = {"from": frm, "to_cond": to_cond, "promotion_cond": promo}
    return {k: v.to(device) for k, v in out.items()} if device is not None else out


def fromto_replay_legal(sample_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The ``legal`` dict that replays a FromToDecoder rollout exactly.

    ``sample_out`` = :meth:`FromToDecoder.sample` / ``GameNetwork.step``
    outputs (one step [B, ...], or stacked over time [B, T, ...]): the masks
    the rollout sampled under become ``from`` / ``to_of`` / ``promotion_of``;
    replay with ``labels={"from_idx", "to_idx"}`` = the stored actions."""
    keys = (("legal_from", "from"), ("legal_to", "to_of"),
            ("legal_promotion", "promotion_of"))
    return {dst: sample_out[src] for src, dst in keys if src in sample_out}


def _categorical_head(
    logits: torch.Tensor,             # [B, T, K]
    target: torch.Tensor,             # [B, T] long, in range
    mask: Optional[torch.Tensor],     # [B, T, K] bool or None
    weight: torch.Tensor,             # [B, T] float
    norm: torch.Tensor,               # scalar
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """One categorical IL head → (il_head_terms entry, argmax-correct [B, T],
    target-illegal [B, T] or None).

    Unmasked: exactly the historical CE (``nan_to_num`` guarded).  Masked:
    CE of the masked logits (degenerate-row guard), 0 where the target is
    illegal under the mask (a label outside the legal set cannot be
    learned; its +inf CE would otherwise poison the batch), accuracy from
    the masked argmax."""
    B, T, K = logits.shape
    if mask is None:
        ce = F.cross_entropy(
            logits.reshape(-1, K), target.reshape(-1), reduction="none",
        ).reshape(B, T).nan_to_num(0.0)
        with torch.no_grad():
            ok = logits.argmax(-1) == target
        term = {"kind": "categorical", "logits": logits, "ce": ce,
                "weight": weight, "norm": norm}
        return term, ok, None
    mask = mask if mask.dtype == torch.bool else mask.bool()
    ml = _masked_policy_logits(logits, mask)
    ce = F.cross_entropy(
        ml.reshape(-1, K), target.reshape(-1), reduction="none",
    ).reshape(B, T)
    tgt_legal = mask.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    ce = torch.where(tgt_legal, ce, torch.zeros_like(ce)).nan_to_num(0.0)
    with torch.no_grad():
        ok = ml.argmax(-1) == target
    term = {"kind": "categorical", "logits": logits, "ce": ce,
            "weight": weight, "norm": norm, "mask": mask}
    return term, ok, ~tgt_legal


class BoardEncoder(ObservationEncoder):
    """Piece planes → one token per square.

    obs keys:
        board_seq   [B, T, piece_channels, H, W]
        globals_seq [B, T, G]   (only when ``board_global_features > 0``)
    """

    def __init__(self, cfg: BoardConfig):
        super().__init__()
        self.cfg = cfg
        self.n_tokens = cfg.board_h * cfg.board_w

        # Same shape as SpatialEncoder (conv → ResBlock → conv → ResBlock)
        # but sized for piece planes; boards are small so this is cheap.
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

        self.global_proj: Optional[nn.Sequential] = None
        if cfg.board_global_features > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(cfg.board_global_features, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )

        # 2-D spatial PE (learnable row + column embeddings, summed)
        self.row_embed = nn.Parameter(
            torch.randn(1, cfg.board_h, 1, cfg.d_model) * 0.02,
        )
        self.col_embed = nn.Parameter(
            torch.randn(1, 1, cfg.board_w, cfg.d_model) * 0.02,
        )

    def _encode_frames(self, board_seq, frame_mask):
        """Run the conv stack, excluding padded frames from BatchNorm
        statistics when ``frame_mask`` is given (same technique as
        ``GeneralsEncoder._encode_frames``): padded frames get zero
        features and cannot influence real frames because collate pads at
        the sequence tail and the temporal mixers are causal."""
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
            feat = feat + glb.unsqueeze(-1).unsqueeze(-1)
        return feat.reshape(B, T, d, H * W).permute(0, 1, 3, 2)

    def positional_embedding(self) -> torch.Tensor:
        return (self.row_embed + self.col_embed).reshape(1, -1, self.cfg.d_model)


class FromToDecoder(ActionDecoder):
    """Chess-like decoder: From-square → To-square (→ Promotion).

    legal keys (all optional, bool or float {0, 1}; N squares, P promotions):
        from            [B, T, N]        legal from-squares
        to              [B, T, N]        legacy unconditional to-mask
                                         (applied to whatever from was picked)
        to_cond         [B, T, N, N]     exact to-table, row = from-square
        to_of           [B, T, N]        destinations valid for the LABELLED
                                         from-square (labels["from_idx"])
        promotion_cond  [B, T, N, N, P]  exact promotion table [from, to]
        promotion_of    [B, T, P]        promotions valid for the labelled
                                         (from_idx, to_idx)
      Resolution precedence: cond > of > flat.  An ``X_of`` mask applies only
      on rows whose committed index equals the label; rows whose label is
      −1 or absent (no labels, tf<1 self-sampled rows) are unconstrained.
      ``*_cond`` serve RL rollouts (envs: :func:`fromto_legal_tables`),
      ``*_of`` IL (the chess converter's tgt.legal_from / legal_to_of /
      legal_promotion_of) and PPO replay (:func:`fromto_replay_legal`).
    labels keys:  from_idx / to_idx [B, T]  (−1 → sample from own dist)
    actions keys: from_idx / to_idx / promotion [B, T]

    The to-head is conditioned on the from-embedding, so ``decode`` must
    commit to a from-square (teacher-forced or self-sampled) and report it
    back (``from_idx`` / ``to_idx`` in the pred dict) — losses condition-
    weight on it and PPO replay teacher-forces it.  The masks ``decode``
    resolved are echoed as ``from_mask`` / ``to_mask`` [B, T, N] and
    ``promotion_mask`` [B, T, P] (bool, present only when a mask was given),
    and :meth:`sample`, :meth:`log_prob_entropy` and :meth:`il_loss` all use
    the echoes — one masked distribution per head everywhere.
    """

    def __init__(self, cfg: BoardConfig):
        super().__init__()
        self.cfg = cfg
        self.from_head = PointerHead(cfg.d_model, n_queries=1)
        self.to_head = PointerHead(cfg.d_model, n_queries=2)
        self.promotion_head: Optional[MLPHead] = None
        if cfg.n_promotions > 1:
            self.promotion_head = MLPHead(3 * cfg.d_model, cfg.n_promotions)

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
        N = tokens.shape[1]
        arange = torch.arange(BT, device=tokens.device)
        lbl_from = labels.get("from_idx") if labels else None
        lbl_to = labels.get("to_idx") if labels else None

        from_mask = _legal_mask(legal, "from")
        if from_mask is not None:
            from_mask = from_mask.reshape(BT, N)
        from_logits = self.from_head(tokens, ll)               # [BT, N] raw
        from_idx = _pick_index(from_logits, lbl_from, from_mask, BT, N)
        from_embed = tokens[arange, from_idx]

        to_mask = _resolve_to_mask(legal, from_idx, lbl_from)
        to_logits = self.to_head(tokens, from_embed, ll)       # [BT, N] raw
        to_idx = _pick_index(to_logits, lbl_to, to_mask, BT, N)

        out = {
            "from_logits": from_logits.reshape(B, T, N),
            "to_logits": to_logits.reshape(B, T, N),
            "from_idx": from_idx.reshape(B, T),
            "to_idx": to_idx.reshape(B, T),
        }
        if from_mask is not None:
            out["from_mask"] = from_mask.reshape(B, T, N)
        if to_mask is not None:
            out["to_mask"] = to_mask.reshape(B, T, N)
        if self.promotion_head is not None:
            P = self.cfg.n_promotions
            to_embed = tokens[arange, to_idx]
            promo = self.promotion_head(
                torch.cat([from_embed, to_embed, ll], dim=-1),
            )
            out["promotion_logits"] = promo.reshape(B, T, P)
            promo_mask = _resolve_promotion_mask(
                legal, from_idx, to_idx, lbl_from, lbl_to)
            if promo_mask is not None:
                out["promotion_mask"] = promo_mask.reshape(B, T, P)
        return out

    # ------------------------------------------------------------------

    def _replay_masks(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]],
        from_idx: torch.Tensor,
        to_idx: torch.Tensor,
        labelled: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(from, to, promotion) masks: decode's echoes, else resolved from
        ``legal`` at the committed indices (pred dicts built without
        decode, e.g. decoder-level tests).  ``labelled``: the indices are
        stored actions, so ``X_of`` masks apply (replay); otherwise (a
        rollout has no labels) they are ignored."""
        from_mask = pred.get("from_mask")
        if from_mask is None:
            from_mask = _legal_mask(legal, "from")
        if not labelled and legal:
            legal = {k: v for k, v in legal.items() if not k.endswith("_of")}
        lbl_from, lbl_to = (from_idx, to_idx) if labelled else (None, None)
        to_mask = pred.get("to_mask")
        if to_mask is None:
            to_mask = _resolve_to_mask(legal, from_idx, lbl_from)
        promo_mask = None
        if self.promotion_head is not None:
            promo_mask = pred.get("promotion_mask")
            if promo_mask is None:
                promo_mask = _resolve_promotion_mask(
                    legal, from_idx, to_idx, lbl_from, lbl_to)
        return from_mask, to_mask, promo_mask

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """From/to were committed in :meth:`decode`; here we sample the
        promotion and compute the hierarchical log-prob — identically to
        :meth:`log_prob_entropy`, so rollout and update define one policy.

        Also returns the masks sampled under (``legal_from`` / ``legal_to``
        / ``legal_promotion``, when present) for an exact PPO replay via
        :func:`fromto_replay_legal`."""
        from_idx = pred["from_idx"]
        to_idx = pred["to_idx"]
        B = from_idx.shape[0]
        device = from_idx.device

        with torch.no_grad():
            from_mask, to_mask, promo_mask = self._replay_masks(
                pred, legal, from_idx, to_idx, labelled=False)
            from_lp = masked_log_prob(
                pred["from_logits"].float(), from_idx, from_mask,
            )
            to_lp = masked_log_prob(
                pred["to_logits"].float(), to_idx, to_mask,
            )
            total_lp = from_lp + to_lp

            if self.promotion_head is not None:
                promo_idx = masked_sample(pred["promotion_logits"], promo_mask)
                total_lp = total_lp + masked_log_prob(
                    pred["promotion_logits"].float(), promo_idx, promo_mask,
                )
            else:
                promo_idx = torch.zeros(B, dtype=torch.long, device=device)

        out = {
            "action_from": from_idx,
            "action_to": to_idx,
            "action_promotion": promo_idx,
            "action_log_prob": total_lp,
        }
        for key, m in (("legal_from", from_mask), ("legal_to", to_mask),
                       ("legal_promotion", promo_mask)):
            if m is not None:
                out[key] = m
        return out

    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        from_mask, to_mask, promo_mask = self._replay_masks(
            pred, legal, actions["from_idx"], actions["to_idx"], labelled=True)
        from_lp, from_ent = _log_prob_and_entropy(
            pred["from_logits"], actions["from_idx"], from_mask,
        )
        to_lp, to_ent = _log_prob_and_entropy(
            pred["to_logits"], actions["to_idx"], to_mask,
        )
        total_lp = from_lp + to_lp
        total_ent = from_ent + to_ent

        head_entropy = {
            "ent_from": from_ent.detach().mean(),
            "ent_to": to_ent.detach().mean(),
        }
        if self.promotion_head is not None:
            promo_lp, promo_ent = _log_prob_and_entropy(
                pred["promotion_logits"], actions["promotion"], promo_mask,
            )
            total_lp = total_lp + promo_lp
            total_ent = total_ent + promo_ent
            head_entropy["ent_promo"] = promo_ent.detach().mean()
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------

    def _il_parts(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor],
               Dict[str, Optional[torch.Tensor]]]:
        """Per-head IL terms (see :meth:`il_head_terms`) + per-head
        argmax-correct and target-illegal [B, T] (None when unmasked)."""
        B, T, N = pred["from_logits"].shape
        valid = seq_mask.float()
        n_valid = valid.sum().clamp(min=1.0)

        tgt_from = tgt["from_idx"].long().clamp(0, N - 1)
        tgt_to = tgt["to_idx"].long().clamp(0, N - 1)

        terms, ok, illegal = {}, {}, {}
        terms["from"], ok["from"], illegal["from"] = _categorical_head(
            pred["from_logits"], tgt_from, pred.get("from_mask"), valid, n_valid)

        # cond_weight: only frames where the committed from-square is the
        # ground-truth one supervise the conditioned to-head.
        cond_from = (pred["from_idx"] == tgt_from).float() * valid
        n_cond_from = cond_from.sum().clamp(min=1.0)
        terms["to"], ok["to"], illegal["to"] = _categorical_head(
            pred["to_logits"], tgt_to, pred.get("to_mask"), cond_from, n_cond_from)

        if self.promotion_head is not None:
            P = self.cfg.n_promotions
            tgt_promo = tgt["promotion"].long().clamp(0, P - 1)
            cond_to = cond_from * (pred["to_idx"] == tgt_to).float()
            n_cond_to = cond_to.sum().clamp(min=1.0)
            terms["promotion"], ok["promotion"], illegal["promotion"] = _categorical_head(
                pred["promotion_logits"], tgt_promo, pred.get("promotion_mask"),
                cond_to, n_cond_to)
        return terms, ok, illegal

    def il_head_terms(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Per-head decomposition of :meth:`il_loss`:
        ``il_loss["loss"] == Σ_head (ce · weight).sum() / norm``.

        Each entry: ``kind`` "categorical", ``logits`` [B, T, K] (raw),
        ``ce`` [B, T] (the hard-label CE il_loss uses), ``weight`` [B, T],
        ``norm`` (scalar) and ``mask`` [B, T, K] bool when decode resolved a
        legality mask (apply it, with the degenerate-row guard, before a
        softmax).  Heads: ``from`` (weight = valid frames), ``to`` (frames
        whose committed from-square is tgt.from_idx), ``promotion`` (also
        the committed to-square is tgt.to_idx; only with a promotion head).

        Soft targets (teacher policies) for these heads mean:
        ``from_probs`` [B, T, N] = the teacher's marginal over from-squares;
        ``to_probs`` [B, T, N] = p(to | from = tgt.from_idx);
        ``promotion_probs`` [B, T, P] = p(promo | tgt.from_idx, tgt.to_idx)
        — the conditioning the weights select."""
        return self._il_parts(pred, tgt, seq_mask)[0]

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Masked CE per head.  The to/promotion heads only see gradient
        where their conditioning matched the ground truth (cond_weight),
        because their logits are meaningless under wrong conditioning —
        same discipline as ``model.losses.ImitationLoss``.

        With legality masks echoed by decode (``from_mask`` / ``to_mask`` /
        ``promotion_mask``) each head's CE and accuracy use the masked
        logits, a target outside its mask contributes 0, and
        ``illegal_target_frac`` reports the fraction of valid frames with an
        illegal supervised target; without them the loss is the plain
        (historical) CE."""
        terms, ok, illegal = self._il_parts(pred, tgt, seq_mask)
        head_loss = {k: (t["ce"] * t["weight"]).sum() / t["norm"]
                     for k, t in terms.items()}
        loss = head_loss["from"] + head_loss["to"]

        with torch.no_grad():
            def acc(k):
                t = terms[k]
                return (ok[k].float() * t["weight"]).sum() / t["norm"]

            out = {
                "from_loss": head_loss["from"].detach(),
                "to_loss": head_loss["to"].detach(),
                "from_acc": acc("from"),
                "to_acc": acc("to"),
            }
            answer_ok = ok["from"] & ok["to"]

        if "promotion" in terms:
            loss = loss + head_loss["promotion"]
            with torch.no_grad():
                out["promo_loss"] = head_loss["promotion"].detach()
                out["promo_acc"] = acc("promotion")
                answer_ok = answer_ok & ok["promotion"]

        if any(v is not None for v in illegal.values()):
            with torch.no_grad():
                bad = torch.zeros_like(seq_mask, dtype=torch.bool)
                for k, v in illegal.items():
                    if v is not None:
                        bad = bad | (v & (terms[k]["weight"] > 0))
                valid = seq_mask.float()
                out["illegal_target_frac"] = (
                    (bad.float() * valid).sum() / valid.sum().clamp(min=1.0))

        # whole-move exact match per frame → the TRM q_halt target
        out["answer_correct"] = answer_ok
        out["answer_mask"] = seq_mask.bool()
        out["loss"] = loss
        return out


class PlacementDecoder(ActionDecoder):
    """Go-like decoder: place a stone or pass.

    legal keys:   place [B, T, N] bool
    actions keys: place_idx [B, T] long, passed [B, T] bool

    No autoregressive conditioning exists (a single pointer + pass gate),
    so ``decode`` is fully deterministic and returns no conditioning index.
    Hierarchical log-prob is pass-gated exactly like generals: a pass frame
    is scored by the pass head alone; a move frame by pass + placement.

    ``cfg.allow_pass=False`` (gomoku) removes the pass action entirely:
    no pass head exists, sample() never passes, and log-prob / entropy /
    il_loss are pure placement — all three paths branch on the same flag,
    which is what keeps PPO replay exact.  The placement legal mask alone
    can NOT suppress the pass Bernoulli, hence the config flag.
    """

    def __init__(self, cfg: BoardConfig):
        super().__init__()
        self.cfg = cfg
        self.allow_pass = getattr(cfg, "allow_pass", True)
        self.place_head = PointerHead(cfg.d_model, n_queries=1)
        self.pass_head: Optional[MLPHead] = (
            MLPHead(cfg.d_model, 1) if self.allow_pass else None
        )

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
        N = tokens.shape[1]
        place_logits = self.place_head(tokens, ll)      # [BT, N] raw
        out = {"place_logits": place_logits.reshape(B, T, N)}
        if self.pass_head is not None:
            out["pass_logits"] = self.pass_head(ll).squeeze(-1).reshape(B, T)
        return out

    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        place_logits = pred["place_logits"]
        legal_mask = legal.get("place") if legal else None

        with torch.no_grad():
            place_idx = masked_sample(place_logits, legal_mask)
            place_lp = masked_log_prob(place_logits, place_idx, legal_mask)

            if self.pass_head is None:
                did_pass = torch.zeros_like(place_idx, dtype=torch.bool)
                total_lp = place_lp
            else:
                pass_logits = pred["pass_logits"].float()
                did_pass = torch.bernoulli(torch.sigmoid(pass_logits)).bool()
                pass_lp = torch.where(
                    did_pass,
                    F.logsigmoid(pass_logits),
                    F.logsigmoid(-pass_logits),
                )
                # Hierarchical: pass → only pass_lp; move → pass_lp + place_lp
                total_lp = torch.where(did_pass, pass_lp, pass_lp + place_lp)

        return {
            "action_place": place_idx,
            "action_pass": did_pass,
            "action_log_prob": total_lp,
        }

    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        place_lp, place_ent = _log_prob_and_entropy(
            pred["place_logits"], actions["place_idx"],
            legal.get("place") if legal else None,
        )

        if self.pass_head is None:
            with torch.no_grad():
                head_entropy = {"ent_place": place_ent.detach().mean()}
            return place_lp, place_ent, head_entropy

        pass_logits = pred["pass_logits"].float()
        passed = actions["passed"].bool()

        pass_lp = torch.where(
            passed, F.logsigmoid(pass_logits), F.logsigmoid(-pass_logits),
        )
        p_pass = torch.sigmoid(pass_logits)
        pass_ent = -(
            p_pass * F.logsigmoid(pass_logits)
            + (1 - p_pass) * F.logsigmoid(-pass_logits)
        )

        total_lp = torch.where(passed, pass_lp, pass_lp + place_lp)
        total_ent = torch.where(passed, pass_ent, pass_ent + place_ent)

        with torch.no_grad():
            move_f = (~passed).float()
            n_moves = move_f.sum().clamp(min=1.0)
            head_entropy = {
                "ent_pass": pass_ent.detach().mean(),
                "ent_place": (place_ent.detach() * move_f).sum() / n_moves,
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------

    def _il_parts(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        """(il_head_terms entries, placement argmax-correct [B, T])."""
        B, T, N = pred["place_logits"].shape
        valid = seq_mask.float()
        n_valid = valid.sum().clamp(min=1.0)

        terms: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.pass_head is None:
            move_w = valid
        else:
            tgt_pass = tgt["passed"].float()
            pass_bce = F.binary_cross_entropy_with_logits(
                pred["pass_logits"], tgt_pass, reduction="none",
            ).nan_to_num(0.0)
            move_w = valid * (1.0 - tgt_pass)
            terms["pass"] = {"kind": "bernoulli", "logits": pred["pass_logits"],
                             "ce": pass_bce, "weight": valid, "norm": n_valid}

        n_moves = move_w.sum().clamp(min=1.0)
        tgt_place = tgt["place_idx"].long().clamp(0, N - 1)
        place, place_ok, _ = _categorical_head(
            pred["place_logits"], tgt_place, None, move_w, n_moves)
        return {"place": place, **terms}, place_ok

    def il_head_terms(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Per-head decomposition of :meth:`il_loss`
        (``loss == Σ_head (ce · weight).sum() / norm``; entry format as
        :meth:`FromToDecoder.il_head_terms`): ``place`` (categorical,
        weight = move frames) and, with a pass head, ``pass`` (``kind``
        "bernoulli", ``logits`` [B, T], ``ce`` = the BCE, weight = valid
        frames).  Soft targets: ``place_probs`` [B, T, N] = p(place | not
        pass), ``pass_probs`` [B, T] = P(pass)."""
        return self._il_parts(pred, tgt, seq_mask)[0]

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Pass BCE on every valid frame; placement CE only on move frames
        (a pass frame carries no placement target).  With ``allow_pass``
        off, every valid frame is a move frame and no pass loss exists."""
        terms, place_ok = self._il_parts(pred, tgt, seq_mask)
        place = terms["place"]
        place_loss = (place["ce"] * place["weight"]).sum() / place["norm"]

        with torch.no_grad():
            metrics = {
                "place_loss": place_loss.detach(),
                "place_acc": (place_ok.float() * place["weight"]).sum() / place["norm"],
            }
        metrics["answer_mask"] = seq_mask.bool()
        if self.pass_head is None:
            metrics["answer_correct"] = place_ok
            return {"loss": place_loss, **metrics}

        ps = terms["pass"]
        pass_loss = (ps["ce"] * ps["weight"]).sum() / ps["norm"]
        with torch.no_grad():
            pass_ok = (pred["pass_logits"] > 0) == tgt["passed"].bool()
            metrics["pass_loss"] = pass_loss.detach()
            metrics["pass_acc"] = (pass_ok.float() * ps["weight"]).sum() / ps["norm"]
            passed = tgt["passed"].bool()
            pred_pass = pred["pass_logits"] > 0
            metrics["answer_correct"] = torch.where(
                passed, pred_pass, ~pred_pass & place_ok)
        return {"loss": pass_loss + place_loss, **metrics}
