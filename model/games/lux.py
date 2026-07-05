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

v1 simplification (documented): the "special" action (e.g. sap / transfer
in Lux S3) is a plain categorical class with no target-cell argument.  A
per-unit target pointer conditioned on the unit token can be added later
without breaking this factorization.
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
    # 0 = noop, 1-4 = move NESW, 5.. = special (sap / transfer, no target arg in v1)
    n_unit_actions: int = 6
    lux_global_features: int = 6   # team points, step, ...


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


class LuxDecoder(ActionDecoder):
    """Factored per-unit decoder: an independent categorical per unit slot.

    Each unit token gets its own head evaluation — MLP over
    ``[unit_token ‖ ll]`` → ``n_unit_actions`` logits — so one decode
    yields ``unit_action_logits [B, T, U, A]`` for every unit at once.
    A PointerHead (score tokens against a query) would be wrong here: we
    are not picking ONE unit, every unit needs its own full distribution.

    legal keys:   unit_action [B, T, U, A] bool — per-unit legality
                  unit        [B, T, U] bool — which slots hold real units
                  (gates the joint log-prob; absent → all-present)
    labels:       accepted and IGNORED — with no cross-head conditioning
                  there is nothing to teacher-force.
    actions keys: unit_actions [B, T, U] long

    Joint log-prob / entropy = Σ over present units; absent units
    contribute exactly 0 via ``torch.where``, so their (meaningless)
    logits can never leak into PPO ratios or the entropy bonus.
    """

    def __init__(self, cfg: LuxConfig):
        super().__init__()
        self.cfg = cfg
        self.n_map_tokens = cfg.map_h * cfg.map_w
        self.action_head = MLPHead(2 * cfg.d_model, cfg.n_unit_actions)

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
        logits = self.action_head(torch.cat([unit_tokens, ll_b], dim=-1))

        # Presence gates the joint log-prob.  Record the mask actually used
        # so sample()/log_prob_entropy() and the caller agree on it.
        if legal is not None and legal.get("unit") is not None:
            present = legal["unit"].reshape(B, T, U).bool()
        else:
            present = torch.ones(B, T, U, dtype=torch.bool, device=tokens.device)

        return {
            "unit_action_logits": logits.reshape(B, T, U, cfg.n_unit_actions),
            "unit_present": present,
        }

    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample every unit's action independently (masked) and sum the
        per-unit log-probs over present units — identically to
        :meth:`log_prob_entropy`, so rollout and update define one policy."""
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
            total_lp = torch.where(
                present, unit_lp, torch.zeros_like(unit_lp),
            ).sum(-1)

        return {
            "unit_actions": unit_actions,
            "action_log_prob": total_lp,
        }

    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        logits = pred["unit_action_logits"]                    # [B, T, U, A]
        present = pred["unit_present"].bool()                  # [B, T, U]

        unit_lp, unit_ent = _log_prob_and_entropy(
            logits, actions["unit_actions"].long(),
            legal.get("unit_action") if legal else None,
        )                                                      # [B, T, U] each
        zeros = torch.zeros_like(unit_lp)
        total_lp = torch.where(present, unit_lp, zeros).sum(-1)
        total_ent = torch.where(present, unit_ent, zeros).sum(-1)

        with torch.no_grad():
            n_present = present.float().sum().clamp(min=1.0)
            head_entropy = {
                "ent_unit_action":
                    (unit_ent.detach() * present.float()).sum() / n_present,
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Per-unit CE masked by unit presence AND the frame mask, averaged
        over present unit-steps.  A batch with no present units yields a
        finite zero (never NaN)."""
        logits = pred["unit_action_logits"]                    # [B, T, U, A]
        B, T, U, A = logits.shape
        present = pred["unit_present"].float()                 # [B, T, U]
        weight = present * seq_mask.float().unsqueeze(-1)      # [B, T, U]
        n_units = weight.sum().clamp(min=1.0)

        tgt_action = tgt["unit_actions"].long().clamp(0, A - 1)
        ce = F.cross_entropy(
            logits.reshape(-1, A), tgt_action.reshape(-1), reduction="none",
        ).reshape(B, T, U).nan_to_num(0.0)
        loss = (ce * weight).sum() / n_units

        with torch.no_grad():
            ok = logits.argmax(-1) == tgt_action
            metrics = {
                "unit_action_loss": loss.detach(),
                "unit_action_acc": (ok.float() * weight).sum() / n_units,
            }
        return {"loss": loss, **metrics}
