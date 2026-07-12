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
    board_global_features: int = 8  # side-to-move, castling rights, move counters, ...
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
        safe = safe.clone()
        safe[all_masked, ..., 0] = 0.0
    return safe


def _log_prob_and_entropy(
    logits: torch.Tensor, index: torch.Tensor, mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(log-prob of ``index``, entropy) under optionally-masked logits.

    One log_softmax serves both quantities so they cannot drift apart.
    ``nan_to_num`` kills the 0·(−inf) terms produced by masked entries.
    Computed in float32 (differentiable cast) so the PPO replay path
    evaluates the same distribution as the fp32 rollout bookkeeping in
    ``sample()`` even when the logits arrive in bf16.
    """
    log_p = F.log_softmax(_masked_policy_logits(logits.float(), mask), dim=-1)
    lp = log_p.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    ent = -(torch.exp(log_p) * log_p).nan_to_num(0.0).sum(-1)
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

    legal keys:   from / to  [B, T, N] bool
    labels keys:  from_idx / to_idx [B, T]  (−1 → sample from own dist)
    actions keys: from_idx / to_idx / promotion [B, T]

    The to-head is conditioned on the from-embedding, so ``decode`` must
    commit to a from-square (teacher-forced or self-sampled) and report it
    back (``from_idx`` / ``to_idx`` in the pred dict) — losses condition-
    weight on it and PPO replay teacher-forces it.
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

        from_logits = self.from_head(tokens, ll)               # [BT, N] raw
        from_idx = _pick_index(
            from_logits,
            labels.get("from_idx") if labels else None,
            legal.get("from") if legal else None,
            BT, N,
        )
        from_embed = tokens[arange, from_idx]

        to_logits = self.to_head(tokens, from_embed, ll)       # [BT, N] raw
        to_idx = _pick_index(
            to_logits,
            labels.get("to_idx") if labels else None,
            legal.get("to") if legal else None,
            BT, N,
        )

        out = {
            "from_logits": from_logits.reshape(B, T, N),
            "to_logits": to_logits.reshape(B, T, N),
            "from_idx": from_idx.reshape(B, T),
            "to_idx": to_idx.reshape(B, T),
        }
        if self.promotion_head is not None:
            to_embed = tokens[arange, to_idx]
            promo = self.promotion_head(
                torch.cat([from_embed, to_embed, ll], dim=-1),
            )
            out["promotion_logits"] = promo.reshape(B, T, self.cfg.n_promotions)
        return out

    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """From/to were committed in :meth:`decode`; here we sample the
        promotion and compute the hierarchical log-prob — identically to
        :meth:`log_prob_entropy`, so rollout and update define one policy."""
        from_idx = pred["from_idx"]
        to_idx = pred["to_idx"]
        B = from_idx.shape[0]
        device = from_idx.device

        with torch.no_grad():
            from_lp = masked_log_prob(
                pred["from_logits"].float(), from_idx,
                legal.get("from") if legal else None,
            )
            to_lp = masked_log_prob(
                pred["to_logits"].float(), to_idx,
                legal.get("to") if legal else None,
            )
            total_lp = from_lp + to_lp

            if self.promotion_head is not None:
                promo_idx = masked_sample(pred["promotion_logits"])
                total_lp = total_lp + masked_log_prob(
                    pred["promotion_logits"].float(), promo_idx,
                )
            else:
                promo_idx = torch.zeros(B, dtype=torch.long, device=device)

        return {
            "action_from": from_idx,
            "action_to": to_idx,
            "action_promotion": promo_idx,
            "action_log_prob": total_lp,
        }

    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        from_lp, from_ent = _log_prob_and_entropy(
            pred["from_logits"], actions["from_idx"],
            legal.get("from") if legal else None,
        )
        to_lp, to_ent = _log_prob_and_entropy(
            pred["to_logits"], actions["to_idx"],
            legal.get("to") if legal else None,
        )
        total_lp = from_lp + to_lp
        total_ent = from_ent + to_ent

        head_entropy = {
            "ent_from": from_ent.detach().mean(),
            "ent_to": to_ent.detach().mean(),
        }
        if self.promotion_head is not None:
            promo_lp, promo_ent = _log_prob_and_entropy(
                pred["promotion_logits"], actions["promotion"], None,
            )
            total_lp = total_lp + promo_lp
            total_ent = total_ent + promo_ent
            head_entropy["ent_promo"] = promo_ent.detach().mean()
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Masked CE per head.  The to/promotion heads only see gradient
        where their conditioning matched the ground truth (cond_weight),
        because their logits are meaningless under wrong conditioning —
        same discipline as ``model.losses.ImitationLoss``."""
        B, T, N = pred["from_logits"].shape
        valid = seq_mask.float()
        n_valid = valid.sum().clamp(min=1.0)

        tgt_from = tgt["from_idx"].long().clamp(0, N - 1)
        tgt_to = tgt["to_idx"].long().clamp(0, N - 1)

        from_ce = F.cross_entropy(
            pred["from_logits"].reshape(-1, N), tgt_from.reshape(-1),
            reduction="none",
        ).reshape(B, T).nan_to_num(0.0)
        from_loss = (from_ce * valid).sum() / n_valid

        # cond_weight: only frames where the committed from-square is the
        # ground-truth one supervise the conditioned to-head.
        cond_from = (pred["from_idx"] == tgt_from).float() * valid
        n_cond_from = cond_from.sum().clamp(min=1.0)
        to_ce = F.cross_entropy(
            pred["to_logits"].reshape(-1, N), tgt_to.reshape(-1),
            reduction="none",
        ).reshape(B, T).nan_to_num(0.0)
        to_loss = (to_ce * cond_from).sum() / n_cond_from

        loss = from_loss + to_loss

        with torch.no_grad():
            from_ok = pred["from_logits"].argmax(-1) == tgt_from
            to_ok = pred["to_logits"].argmax(-1) == tgt_to
            out = {
                "from_loss": from_loss.detach(),
                "to_loss": to_loss.detach(),
                "from_acc": (from_ok.float() * valid).sum() / n_valid,
                "to_acc": (to_ok.float() * cond_from).sum() / n_cond_from,
            }

        if self.promotion_head is not None:
            P = self.cfg.n_promotions
            tgt_promo = tgt["promotion"].long().clamp(0, P - 1)
            cond_to = cond_from * (pred["to_idx"] == tgt_to).float()
            n_cond_to = cond_to.sum().clamp(min=1.0)
            promo_ce = F.cross_entropy(
                pred["promotion_logits"].reshape(-1, P), tgt_promo.reshape(-1),
                reduction="none",
            ).reshape(B, T).nan_to_num(0.0)
            promo_loss = (promo_ce * cond_to).sum() / n_cond_to
            loss = loss + promo_loss
            with torch.no_grad():
                promo_ok = pred["promotion_logits"].argmax(-1) == tgt_promo
                out["promo_loss"] = promo_loss.detach()
                out["promo_acc"] = (promo_ok.float() * cond_to).sum() / n_cond_to

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

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Pass BCE on every valid frame; placement CE only on move frames
        (a pass frame carries no placement target).  With ``allow_pass``
        off, every valid frame is a move frame and no pass loss exists."""
        B, T, N = pred["place_logits"].shape
        valid = seq_mask.float()
        n_valid = valid.sum().clamp(min=1.0)

        if self.pass_head is None:
            move_w = valid
        else:
            tgt_pass = tgt["passed"].float()
            pass_bce = F.binary_cross_entropy_with_logits(
                pred["pass_logits"], tgt_pass, reduction="none",
            ).nan_to_num(0.0)
            pass_loss = (pass_bce * valid).sum() / n_valid
            move_w = valid * (1.0 - tgt_pass)

        n_moves = move_w.sum().clamp(min=1.0)
        tgt_place = tgt["place_idx"].long().clamp(0, N - 1)
        place_ce = F.cross_entropy(
            pred["place_logits"].reshape(-1, N), tgt_place.reshape(-1),
            reduction="none",
        ).reshape(B, T).nan_to_num(0.0)
        place_loss = (place_ce * move_w).sum() / n_moves

        with torch.no_grad():
            place_ok = pred["place_logits"].argmax(-1) == tgt_place
            metrics = {
                "place_loss": place_loss.detach(),
                "place_acc": (place_ok.float() * move_w).sum() / n_moves,
            }
        if self.pass_head is None:
            return {"loss": place_loss, **metrics}

        with torch.no_grad():
            pass_ok = (pred["pass_logits"] > 0) == tgt["passed"].bool()
            metrics["pass_loss"] = pass_loss.detach()
            metrics["pass_acc"] = (pass_ok.float() * valid).sum() / n_valid
        return {"loss": pass_loss + place_loss, **metrics}
