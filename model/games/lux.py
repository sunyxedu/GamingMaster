"""Lux AI–style adapter — grid resource map + many units, ALL of which act
simultaneously each turn (Kaggle Lux AI Challenge).

Observation = full-resolution map planes (resources, ownership, lichen /
energy, ...) + a fixed number of unit slots + a global vector.  Units are
padded into ``max_units`` slots exactly like the StarCraft adapter: absent
slots get zeroed features (bias included) plus a learnable "empty"
embedding, so attention can tell "no unit here" from "unit with all-zero
features".

Action = **factored simultaneous multi-unit policy**: one forward pass
produces an independent categorical over ``n_unit_actions`` for EVERY unit
slot, and the joint log-prob of a turn is the SUM over *present* units
(independent factorization — the standard Lux baseline treatment).  There
is no hierarchical conditioning between units, so stored-action replay is
trivially exact for PPO: sampling, replay and IL all evaluate the same
per-unit masked distributions.

Sap target (``lux_sap_range`` R > 0, the frozen pre-training schema): the
sap action (``lux_sap_action``, Lux S3: 5) carries a per-unit joint
categorical over the target offset (dx, dy), |dx|, |dy| ≤ R — class
o = (dy+R)·(2R+1) + (dx+R), env action [5, dx, dy] (:func:`lux_sap_delta`,
:func:`lux_sap_legal` builds ``legal["sap"]``).  ``gmdata lux convert``
v2 writes ``tgt.sap_offset`` with R = 7 and shard meta
``{"global_features": 11, "sap_range": 7}``; the trainers take
``lux_global_features`` / ``lux_sap_range`` from that meta.  R = 0 (the
default, and every config written before the field existed) builds the
legacy decoder: no sap head, the sap action a plain class with no target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.encoder import ResBlock
from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    MLPHead,
    ObservationEncoder,
    masked_log_prob,
    masked_sample,
)
from model.games.board import _log_prob_and_entropy


@dataclass
class LuxConfig(ModelConfig):
    """Lux dimensions on top of the shared core config."""

    map_h: int = 24
    map_w: int = 24
    map_channels: int = 12         # resources, ownership, lichen/energy planes, ...
    max_units: int = 16            # fixed unit slots (padded, never shrunk)
    unit_feature_dim: int = 16     # pos, cargo, energy, ...
    # 0 = noop, 1-4 = move NESW, 5 = sap (target offset from the sap head)
    n_unit_actions: int = 6
    # team points, wins, steps (converter v1: 6) + env params (v2: 11, shard meta)
    lux_global_features: int = 6
    # sap target head over offsets |dx|, |dy| <= R ((2R+1)^2 classes); 0 = no
    # head — the legacy value, so configs/checkpoints predating it rebuild
    # identically (converter v2 shards: 7, shard meta "sap_range")
    lux_sap_range: int = 0
    lux_sap_action: int = 5        # the unit action whose target the sap head picks


class LuxEncoder(ObservationEncoder):
    """Map planes at full resolution + unit slots + globals → tokens.

    obs keys:
        map_seq     [B, T, map_channels, map_h, map_w]
        units_seq   [B, T, max_units, unit_feature_dim]
        unit_mask   [B, T, max_units] bool — True = real unit
        globals_seq [B, T, lux_global_features]

    Token layout: ``map_h*map_w`` map tokens (row-major) followed by
    ``max_units`` unit tokens.  Globals are broadcast-added to the map
    tokens (same treatment as the StarCraft economy vector).
    """

    def __init__(self, cfg: LuxConfig):
        super().__init__()
        self.cfg = cfg
        self.n_map_tokens = cfg.map_h * cfg.map_w
        self.n_tokens = self.n_map_tokens + cfg.max_units

        # Full-resolution conv stack (stride-1, BoardEncoder-shaped): the
        # Lux map is small enough that every cell keeps its own token.
        c_mid = cfg.encoder_channels
        self.map_conv = nn.Sequential(
            nn.Conv2d(cfg.map_channels, c_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            ResBlock(c_mid),
            nn.Conv2d(c_mid, cfg.d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.d_model),
            nn.ReLU(inplace=True),
            ResBlock(cfg.d_model),
        )

        self.unit_proj = nn.Linear(cfg.unit_feature_dim, cfg.d_model)
        # Absent slots: features are zeroed (bias included — the Linear
        # output is multiplied by the presence flag) and replaced by this
        # marker, a distinct learned vector, so attention can route around
        # empty slots instead of confusing them with real units.
        self.empty_embed = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        self.global_proj = nn.Sequential(
            nn.Linear(cfg.lux_global_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # PE: 2-D row/col for map tokens, per-slot embedding for unit slots.
        # (nn.Parameter is NOT re-initialised by the core's _init_weights —
        # it only touches Linear/Conv/Norm — so init here.)
        self.map_row_embed = nn.Parameter(
            torch.randn(1, cfg.map_h, 1, cfg.d_model) * 0.02,
        )
        self.map_col_embed = nn.Parameter(
            torch.randn(1, 1, cfg.map_w, cfg.d_model) * 0.02,
        )
        self.unit_slot_embed = nn.Parameter(
            torch.randn(1, cfg.max_units, cfg.d_model) * 0.02,
        )

    def _encode_frames(self, map_seq, frame_mask):
        """Conv the map, excluding padded frames from BatchNorm statistics
        (same technique as ``GeneralsEncoder._encode_frames``): padded
        frames get zero features and cannot influence real frames because
        collate pads at the sequence tail and the temporal mixers are
        causal."""
        B, T, C, H, W = map_seq.shape
        BT = B * T
        flat = map_seq.reshape(BT, C, H, W)
        if frame_mask is None or bool(frame_mask.all()):
            return self.map_conv(flat).reshape(B, T, -1, H, W)
        fm = frame_mask.reshape(BT)
        enc = self.map_conv(flat[fm])
        feat = enc.new_zeros(BT, enc.shape[1], H, W)
        feat[fm] = enc
        return feat.reshape(B, T, -1, H, W)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        map_seq = obs["map_seq"]
        units_seq = obs["units_seq"]
        unit_mask = obs["unit_mask"].bool()
        B, T = map_seq.shape[:2]
        d = self.cfg.d_model

        feat = self._encode_frames(map_seq, frame_mask)       # [B, T, d, H, W]
        feat = feat + self.global_proj(obs["globals_seq"]).unsqueeze(-1).unsqueeze(-1)
        map_tokens = feat.reshape(B, T, d, self.n_map_tokens).permute(0, 1, 3, 2)

        present = unit_mask.unsqueeze(-1).to(units_seq.dtype)  # [B, T, U, 1]
        unit_tokens = self.unit_proj(units_seq) * present      # bias zeroed too
        unit_tokens = unit_tokens + self.empty_embed * (1.0 - present)

        return torch.cat([map_tokens, unit_tokens], dim=2)

    def positional_embedding(self) -> torch.Tensor:
        map_pe = (self.map_row_embed + self.map_col_embed).reshape(
            1, self.n_map_tokens, self.cfg.d_model,
        )
        return torch.cat([map_pe, self.unit_slot_embed], dim=1)


def _mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Differentiable masked logits with the degenerate-row guard of
    ``masked_log_prob`` (an all-False row falls back to class 0)."""
    if mask is None:
        return logits
    safe = logits.masked_fill(~mask, float("-inf"))
    all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
    if all_masked.any():
        # Column-slice write — see base.safe_masked_logits.
        safe = safe.clone()
        safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
    return safe


def lux_sap_offset_grid(sap_range: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """(dx, dy) [O] of every sap class o = (dy+R)·(2R+1) + (dx+R), R = ``sap_range``."""
    n = 2 * sap_range + 1
    o = torch.arange(n * n, device=device)
    return o % n - sap_range, o // n - sap_range


def lux_sap_offset(dx, dy, sap_range: int):
    """Env sap action [5, dx, dy] → sap class (dy+R)·(2R+1) + (dx+R); ints or tensors.
    Callers must keep |dx|, |dy| ≤ R (the converter writes −1 outside)."""
    return (dy + sap_range) * (2 * sap_range + 1) + (dx + sap_range)


def lux_sap_delta(offset, sap_range: int):
    """Sap class → (dx, dy) for the env action [5, dx, dy]; ints or tensors."""
    n = 2 * sap_range + 1
    return offset % n - sap_range, offset // n - sap_range


def lux_sap_legal(
    unit_xy: torch.Tensor,
    unit_mask: torch.Tensor,
    sap_range,
    R: int,
    map_w: int,
    map_h: int,
) -> torch.Tensor:
    """Legal sap classes ``legal["sap"]`` [..., U, (2R+1)²] bool.

    ``unit_xy`` [..., U, 2] integer (x, y) cells, ``unit_mask`` [..., U]
    bool, ``sap_range`` the episode's ``unit_sap_range`` (int, or a tensor
    broadcastable to the leading dims [...]).  A class is legal when the
    unit is present, |dx| and |dy| are within the sap range (Lux S3's
    square range) and the target cell (x+dx, y+dy) lies on the map.  An
    absent unit gets an all-False row (the decoder's degenerate-row guard
    then samples class 0, which never scores: absent units are gated)."""
    dx, dy = lux_sap_offset_grid(R, device=unit_xy.device)
    x = unit_xy[..., 0:1].long() + dx                          # [..., U, O]
    y = unit_xy[..., 1:2].long() + dy
    rng = torch.as_tensor(sap_range, device=unit_xy.device)
    if rng.dim() > 0:
        rng = rng[..., None, None]
    ok = (dx.abs() <= rng) & (dy.abs() <= rng)
    ok = ok & (x >= 0) & (x < map_w) & (y >= 0) & (y < map_h)
    return ok & unit_mask.bool().unsqueeze(-1)


class LuxDecoder(ActionDecoder):
    """Factored per-unit decoder: an independent categorical per unit slot.

    Each unit token gets its own head evaluation — MLP over
    ``[unit_token ‖ ll]`` → ``n_unit_actions`` logits — so one decode
    yields ``unit_action_logits [B, T, U, A]`` for every unit at once.
    A PointerHead (score tokens against a query) would be wrong here: we
    are not picking ONE unit, every unit needs its own full distribution.

    Sap target (``lux_sap_range`` R > 0): a second per-unit head over the
    same ``[unit_token ‖ ll]`` → ``sap_logits [B, T, U, (2R+1)²]``, a joint
    categorical over the offset (dx, dy), |dx|, |dy| ≤ R, class
    o = (dy+R)·(2R+1) + (dx+R), inverse dx = o mod (2R+1) − R,
    dy = o div (2R+1) − R (:func:`lux_sap_delta`; the env action is
    [lux_sap_action, dx, dy]).  A unit's action log-prob is
    unit_lp + sap_lp where its action is the sap action (the mahjong
    ARG_USES convention): the offset is sampled for every unit so replay
    always holds a valid index, but only a sapping unit pays for it.

    legal keys:   unit_action [B, T, U, A] bool — per-unit legality
                  unit        [B, T, U] bool — which slots hold real units
                  (gates the joint log-prob; absent → all-present)
                  sap         [B, T, U, O] bool — legal sap classes
                  (:func:`lux_sap_legal`; echoed as ``sap_mask``)
    labels:       accepted and IGNORED — with no cross-head conditioning
                  there is nothing to teacher-force.
    actions keys: unit_actions [B, T, U] long (+ sap_offsets [B, T, U]
                  long when R > 0)

    Joint log-prob / entropy = Σ over present units; absent units
    contribute exactly 0 via ``torch.where``, so their (meaningless)
    logits can never leak into PPO ratios or the entropy bonus.
    """

    def __init__(self, cfg: LuxConfig):
        super().__init__()
        self.cfg = cfg
        self.n_map_tokens = cfg.map_h * cfg.map_w
        self.action_head = MLPHead(2 * cfg.d_model, cfg.n_unit_actions)
        R = int(getattr(cfg, "lux_sap_range", 0))
        if R < 0:
            raise ValueError(f"lux_sap_range must be >= 0, got {R}")
        self.sap_range = R
        self.sap_action = int(getattr(cfg, "lux_sap_action", 5))
        self.n_sap_offsets = (2 * R + 1) ** 2 if R > 0 else 0
        # R == 0 (legacy): no module at all, so old checkpoints load strictly.
        self.sap_head: Optional[MLPHead] = None
        if R > 0:
            if not 0 <= self.sap_action < cfg.n_unit_actions:
                raise ValueError(
                    f"lux_sap_action={self.sap_action} must index one of the "
                    f"n_unit_actions={cfg.n_unit_actions} unit actions")
            self.sap_head = MLPHead(2 * cfg.d_model, self.n_sap_offsets)

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
        unit_tokens = tokens[:, self.n_map_tokens:]            # [BT, U, d]
        U = unit_tokens.shape[1]

        ll_b = ll.unsqueeze(1).expand(-1, U, -1)               # [BT, U, d]
        x = torch.cat([unit_tokens, ll_b], dim=-1)
        logits = self.action_head(x)

        # Presence gates the joint log-prob.  Record the mask actually used
        # so sample()/log_prob_entropy() and the caller agree on it.
        if legal is not None and legal.get("unit") is not None:
            present = legal["unit"].reshape(B, T, U).bool()
        else:
            present = torch.ones(B, T, U, dtype=torch.bool, device=tokens.device)

        out = {
            "unit_action_logits": logits.reshape(B, T, U, cfg.n_unit_actions),
            "unit_present": present,
        }
        if self.sap_head is not None:
            n_off = self.n_sap_offsets
            out["sap_logits"] = self.sap_head(x).reshape(B, T, U, n_off)
            sap_mask = legal.get("sap") if legal else None
            if sap_mask is not None:
                out["sap_mask"] = sap_mask.reshape(B, T, U, n_off).bool()
        return out

    def _sap_mask(self, pred, legal):
        m = legal.get("sap") if legal else None
        if m is None:
            return pred.get("sap_mask")
        return m.bool()

    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample every unit's action independently (masked) and sum the
        per-unit log-probs over present units — identically to
        :meth:`log_prob_entropy`, so rollout and update define one policy.
        With a sap head every unit also samples a sap offset
        (``sap_offsets`` [B, U]) that counts only for sapping units."""
        logits = pred["unit_action_logits"]                    # [B, U, A]
        present = pred["unit_present"].bool()                  # [B, U]
        B, U, A = logits.shape
        mask = legal.get("unit_action") if legal else None

        with torch.no_grad():
            flat_mask = mask.reshape(B * U, A) if mask is not None else None
            unit_actions = masked_sample(
                logits.reshape(B * U, A), flat_mask,
            ).reshape(B, U)
            unit_lp = masked_log_prob(logits.float(), unit_actions, mask)
            out = {"unit_actions": unit_actions}
            if self.sap_head is not None:
                sap_logits = pred["sap_logits"]                # [B, U, O]
                n_off = sap_logits.shape[-1]
                sap_mask = self._sap_mask(pred, legal)
                sap = masked_sample(
                    sap_logits.reshape(B * U, n_off),
                    sap_mask.reshape(B * U, n_off) if sap_mask is not None else None,
                ).reshape(B, U)
                sap_lp = masked_log_prob(sap_logits.float(), sap, sap_mask)
                unit_lp = unit_lp + torch.where(
                    unit_actions == self.sap_action, sap_lp, torch.zeros_like(sap_lp))
                out["sap_offsets"] = sap
            total_lp = torch.where(
                present, unit_lp, torch.zeros_like(unit_lp),
            ).sum(-1)

        return {**out, "action_log_prob": total_lp}

    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        logits = pred["unit_action_logits"]                    # [B, T, U, A]
        present = pred["unit_present"].bool()                  # [B, T, U]
        unit_actions = actions["unit_actions"].long()

        unit_lp, unit_ent = _log_prob_and_entropy(
            logits, unit_actions,
            legal.get("unit_action") if legal else None,
        )                                                      # [B, T, U] each
        zeros = torch.zeros_like(unit_lp)
        lp, ent = unit_lp, unit_ent
        head_extra = {}
        if self.sap_head is not None:
            if "sap_offsets" not in actions:
                raise KeyError(
                    "LuxDecoder with a sap head (lux_sap_range > 0) needs "
                    "actions['sap_offsets'] (the offsets sample() returned)")
            sap_logits = pred["sap_logits"]
            n_off = sap_logits.shape[-1]
            sap_lp, sap_ent = _log_prob_and_entropy(
                sap_logits, actions["sap_offsets"].long().clamp(0, n_off - 1),
                self._sap_mask(pred, legal),
            )
            is_sap = unit_actions == self.sap_action
            lp = lp + torch.where(is_sap, sap_lp, zeros)
            ent = ent + torch.where(is_sap, sap_ent, zeros)
            with torch.no_grad():
                w = (present & is_sap).float()
                head_extra["ent_sap"] = (sap_ent.detach() * w).sum() / w.sum().clamp(min=1.0)
        total_lp = torch.where(present, lp, zeros).sum(-1)
        total_ent = torch.where(present, ent, zeros).sum(-1)

        with torch.no_grad():
            n_present = present.float().sum().clamp(min=1.0)
            head_entropy = {
                "ent_unit_action":
                    (unit_ent.detach() * present.float()).sum() / n_present,
                **head_extra,
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------

    def _il_parts(self, pred, tgt, seq_mask) -> Dict[str, Any]:
        """Per-element pieces shared by il_loss and il_head_terms."""
        logits = pred["unit_action_logits"]                    # [B, T, U, A]
        B, T, U, A = logits.shape
        present = pred["unit_present"].float()                 # [B, T, U]
        weight = present * seq_mask.float().unsqueeze(-1)      # [B, T, U]
        n_units = weight.sum().clamp(min=1.0)

        tgt_action = tgt["unit_actions"].long().clamp(0, A - 1)
        ce = F.cross_entropy(
            logits.reshape(-1, A), tgt_action.reshape(-1), reduction="none",
        ).reshape(B, T, U).nan_to_num(0.0)
        parts: Dict[str, Any] = {
            "logits": logits, "tgt_action": tgt_action, "ce": ce,
            "weight": weight, "n_units": n_units, "sap": None,
        }
        if self.sap_head is None:
            return parts

        sap_logits = pred["sap_logits"]                        # [B, T, U, O]
        n_off = sap_logits.shape[-1]
        sap_mask = pred.get("sap_mask")
        is_sap = (tgt_action == self.sap_action).float()
        so = tgt.get("sap_offset")
        if so is None:
            # no hard targets: the head is still addressable for soft
            # (teacher) targets on the units that sapped, additively
            w = weight * is_sap
            parts["sap"] = {"logits": sap_logits, "ce": None, "weight": w,
                            "norm": w.sum().clamp(min=1.0), "mask": sap_mask}
            return parts
        so = so.long()
        safe_so = so.clamp(0, n_off - 1)
        w_sap = weight * is_sap * (so >= 0).float()
        n_sap = w_sap.sum().clamp(min=1.0)
        scored = _mask_logits(sap_logits, sap_mask)
        sap_ce = F.cross_entropy(
            scored.reshape(-1, n_off), safe_so.reshape(-1), reduction="none",
        ).reshape(B, T, U).nan_to_num(0.0)
        if sap_mask is not None:
            # an illegal target's CE is +inf: unsupervised, not a NaN source
            ok = sap_mask.gather(-1, safe_so.unsqueeze(-1)).squeeze(-1)
            sap_ce = torch.where(ok, sap_ce, torch.zeros_like(sap_ce))
            w_sap = w_sap * ok.float()
        parts["sap"] = {"logits": sap_logits, "scored": scored, "ce": sap_ce,
                        "weight": w_sap, "norm": n_sap, "mask": sap_mask,
                        "target": safe_so}
        return parts

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Per-unit CE masked by unit presence AND the frame mask, averaged
        over present unit-steps.  A batch with no present units yields a
        finite zero (never NaN).

        With a sap head and ``tgt["sap_offset"]`` [B, T, U] (−1 = none):
        + sap CE over the present unit-steps whose target action is the sap
        action and whose offset is known (legal-masked when decode() got
        ``legal["sap"]``, an illegal target then unsupervised), averaged
        over those; ``answer_correct`` also requires their offsets."""
        p = self._il_parts(pred, tgt, seq_mask)
        logits, weight, n_units = p["logits"], p["weight"], p["n_units"]
        loss = (p["ce"] * weight).sum() / n_units
        sap = p["sap"]
        has_sap_loss = sap is not None and sap["ce"] is not None
        if has_sap_loss:
            sap_loss = (sap["ce"] * sap["weight"]).sum() / sap["norm"]
            loss_total = loss + sap_loss
        else:
            loss_total = loss

        with torch.no_grad():
            ok = logits.argmax(-1) == p["tgt_action"]
            present_b = pred["unit_present"].bool()
            metrics = {
                "answer_mask": seq_mask.bool(),
                "unit_action_loss": loss.detach(),
                "unit_action_acc": (ok.float() * weight).sum() / n_units,
            }
            if has_sap_loss:
                sap_hit = sap["scored"].argmax(-1) == sap["target"]
                w_sap = sap["weight"]
                metrics["sap_loss"] = sap_loss.detach()
                metrics["sap_acc"] = (sap_hit.float() * w_sap).sum() / sap["norm"]
                ok = ok & (sap_hit | ~(w_sap > 0))
            # every present unit's action right (no units → trivially)
            metrics["answer_correct"] = (ok | ~present_b).all(-1)
        return {"loss": loss_total, **metrics}

    def il_head_terms(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, Dict[str, Any]]:
        """Per-head decomposition of :meth:`il_loss` (ActionDecoder
        contract): ``il_loss["loss"] == Σ_h (ce_h · weight_h).sum() / norm_h``.

        ``unit_action``: logits [B, T, U, A], weight = presence × frame
        mask.  ``sap`` (sap head only): logits [B, T, U, O] over the offset
        classes, weight = the sap-target unit-steps, mask = decode()'s
        ``sap_mask`` echo; without ``tgt["sap_offset"]`` its ce is None
        (a soft target ``tgt["sap_probs"]`` then adds to the loss)."""
        p = self._il_parts(pred, tgt, seq_mask)
        terms = {
            "unit_action": {
                "logits": p["logits"], "kind": "categorical", "ce": p["ce"],
                "weight": p["weight"], "norm": p["n_units"], "mask": None, "scale": 1.0,
            },
        }
        sap = p["sap"]
        if sap is not None:
            terms["sap"] = {
                "logits": sap["logits"], "kind": "categorical", "ce": sap["ce"],
                "weight": sap["weight"], "norm": sap["norm"], "mask": sap["mask"],
                "scale": 1.0,
            }
        return terms
