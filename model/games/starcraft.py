"""StarCraft-style RTS adapter — AlphaStar-lite action space.

Observation = minimap planes + a fixed number of unit slots + a global
economy vector.  Units are padded into ``max_units`` slots (never a
variable-length sequence): absent slots get zeroed features plus a
learnable "empty" embedding so attention can distinguish "no unit here"
from "unit with all-zero features".

Action = hierarchical (type → actor → target-unit / location, + queued),
where each action type declares which arguments it actually uses
(``ARG_USES``).  Only applicable heads contribute to the hierarchical
log-prob and entropy — inapplicable heads are unconstrained garbage, and
letting them into the PPO ratio would inject pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
from model.games.board import _log_prob_and_entropy, _pick_index


@dataclass
class StarCraftConfig(ModelConfig):
    """RTS dimensions on top of the shared core config."""

    minimap_h: int = 32
    minimap_w: int = 32
    minimap_channels: int = 8
    spatial_downsample: int = 4    # two stride-2 convs → 8×8 = 64 map tokens
    max_units: int = 32            # fixed unit slots (padded, never shrunk)
    unit_feature_dim: int = 24
    sc_global_features: int = 8    # minerals / gas / supply / ...
    n_action_types: int = 8


def _n_map_tokens(cfg: StarCraftConfig) -> int:
    return (cfg.minimap_h // cfg.spatial_downsample) * (
        cfg.minimap_w // cfg.spatial_downsample
    )


def default_arg_uses(n_action_types: int) -> torch.Tensor:
    """[n_types, 3] bool: (uses_actor, uses_target_unit, uses_location).

    Row 0 is always no_op.  The canonical 8-type table covers
    (no_op, move, attack, stop, gather, build, train, ability); other
    sizes cycle through the non-noop rows so every type stays sensible.
    """
    base: List[Tuple[bool, bool, bool]] = [
        (True, False, True),    # move    : actor → location
        (True, True, False),    # attack  : actor → target unit
        (True, False, False),   # stop    : actor only
        (True, True, False),    # gather  : actor → resource (a unit)
        (True, False, True),    # build   : actor → location
        (True, False, False),   # train   : actor (production building)
        (True, True, True),     # ability : actor → target unit + location
    ]
    rows = [(False, False, False)]  # no_op
    for i in range(1, n_action_types):
        rows.append(base[(i - 1) % len(base)])
    return torch.tensor(rows, dtype=torch.bool)


class StarCraftEncoder(ObservationEncoder):
    """Minimap + unit slots + globals → map tokens ‖ unit tokens.

    obs keys:
        minimap_seq [B, T, C, minimap_h, minimap_w]
        units_seq   [B, T, max_units, unit_feature_dim]
        unit_mask   [B, T, max_units] bool — True = real unit
        globals_seq [B, T, sc_global_features]
    """

    def __init__(self, cfg: StarCraftConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.spatial_downsample != 4:
            raise ValueError(
                "StarCraftEncoder uses exactly two stride-2 convs "
                f"(spatial_downsample=4), got {cfg.spatial_downsample}"
            )
        self.map_h = cfg.minimap_h // cfg.spatial_downsample
        self.map_w = cfg.minimap_w // cfg.spatial_downsample
        self.n_map_tokens = self.map_h * self.map_w
        self.n_tokens = self.n_map_tokens + cfg.max_units

        c_mid = cfg.encoder_channels
        self.minimap_conv = nn.Sequential(
            nn.Conv2d(cfg.minimap_channels, c_mid, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, cfg.d_model, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg.d_model),
            nn.ReLU(inplace=True),
            ResBlock(cfg.d_model),
        )

        self.unit_proj = nn.Linear(cfg.unit_feature_dim, cfg.d_model)
        # Absent slots: features are zeroed and replaced by this marker —
        # a distinct learned vector, not silence, so attention can route
        # around empty slots instead of confusing them with real units.
        self.empty_embed = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        self.global_proj = nn.Sequential(
            nn.Linear(cfg.sc_global_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # PE: 2-D row/col for map tokens, per-slot embedding for unit slots
        self.map_row_embed = nn.Parameter(
            torch.randn(1, self.map_h, 1, cfg.d_model) * 0.02,
        )
        self.map_col_embed = nn.Parameter(
            torch.randn(1, 1, self.map_w, cfg.d_model) * 0.02,
        )
        self.unit_slot_embed = nn.Parameter(
            torch.randn(1, cfg.max_units, cfg.d_model) * 0.02,
        )

    def _encode_frames(self, minimap_seq, frame_mask):
        """Conv the minimap, excluding padded frames from BatchNorm
        statistics (same technique as ``GeneralsEncoder._encode_frames``)."""
        B, T, C, H, W = minimap_seq.shape
        BT = B * T
        flat = minimap_seq.reshape(BT, C, H, W)
        if frame_mask is None or bool(frame_mask.all()):
            enc = self.minimap_conv(flat)
            return enc.reshape(B, T, *enc.shape[1:])
        fm = frame_mask.reshape(BT)
        enc = self.minimap_conv(flat[fm])
        feat = enc.new_zeros(BT, *enc.shape[1:])
        feat[fm] = enc
        return feat.reshape(B, T, *feat.shape[1:])

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        minimap_seq = obs["minimap_seq"]
        units_seq = obs["units_seq"]
        unit_mask = obs["unit_mask"].bool()
        B, T = minimap_seq.shape[:2]
        d = self.cfg.d_model

        feat = self._encode_frames(minimap_seq, frame_mask)   # [B, T, d, mh, mw]
        feat = feat + self.global_proj(obs["globals_seq"]).unsqueeze(-1).unsqueeze(-1)
        map_tokens = feat.reshape(B, T, d, self.n_map_tokens).permute(0, 1, 3, 2)

        present = unit_mask.unsqueeze(-1).to(units_seq.dtype)  # [B, T, U, 1]
        unit_tokens = self.unit_proj(units_seq) * present
        unit_tokens = unit_tokens + self.empty_embed * (1.0 - present)

        return torch.cat([map_tokens, unit_tokens], dim=2)

    def positional_embedding(self) -> torch.Tensor:
        map_pe = (self.map_row_embed + self.map_col_embed).reshape(
            1, self.n_map_tokens, self.cfg.d_model,
        )
        return torch.cat([map_pe, self.unit_slot_embed], dim=1)


class RTSDecoder(ActionDecoder):
    """AlphaStar-lite decoder: Type → Actor → Target-unit / Location (+ queued).

    legal keys:   action_type [B, T, n_types], actor / target_unit
                  [B, T, max_units], location [B, T, n_map_tokens] — all bool
    labels keys:  action_type / actor_idx [B, T]  (−1 → sample) — the two
                  choices downstream heads condition on
    actions keys: action_type / actor_idx / target_idx / location_idx /
                  queued [B, T]

    ``ARG_USES[type] = (uses_actor, uses_target_unit, uses_location)``
    gates which argument heads enter the hierarchical log-prob/entropy.
    """

    def __init__(self, cfg: StarCraftConfig, arg_uses: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.n_map_tokens = _n_map_tokens(cfg)

        if arg_uses is None:
            arg_uses = default_arg_uses(cfg.n_action_types)
        if tuple(arg_uses.shape) != (cfg.n_action_types, 3):
            raise ValueError(
                f"arg_uses must be [{cfg.n_action_types}, 3], got {tuple(arg_uses.shape)}"
            )
        self.register_buffer("arg_uses", arg_uses.bool())

        self.type_head = MLPHead(cfg.d_model, cfg.n_action_types)
        # nn.Embedding is NOT re-initialised by the core's _init_weights
        # (it only touches Linear/Conv/Norm) — init it here.
        self.type_embed = nn.Embedding(cfg.n_action_types, cfg.d_model)
        nn.init.trunc_normal_(self.type_embed.weight, std=0.02)

        self.actor_head = PointerHead(cfg.d_model, n_queries=2)     # (type, ll)
        self.target_head = PointerHead(cfg.d_model, n_queries=2)    # (actor, ll)
        self.location_head = PointerHead(cfg.d_model, n_queries=2)  # (actor, type)
        self.queued_head = MLPHead(2 * cfg.d_model, 1)

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
        map_tokens = tokens[:, : self.n_map_tokens]
        unit_tokens = tokens[:, self.n_map_tokens:]
        U = unit_tokens.shape[1]
        arange = torch.arange(BT, device=tokens.device)

        type_logits = self.type_head(ll)                        # [BT, n_types] raw
        action_type = _pick_index(
            type_logits,
            labels.get("action_type") if labels else None,
            legal.get("action_type") if legal else None,
            BT, cfg.n_action_types,
        )
        t_embed = self.type_embed(action_type)                  # [BT, d]

        actor_logits = self.actor_head(unit_tokens, t_embed, ll)  # [BT, U] raw
        actor_idx = _pick_index(
            actor_logits,
            labels.get("actor_idx") if labels else None,
            legal.get("actor") if legal else None,
            BT, U,
        )
        actor_embed = unit_tokens[arange, actor_idx]

        target_logits = self.target_head(unit_tokens, actor_embed, ll)
        location_logits = self.location_head(map_tokens, actor_embed, t_embed)
        queued_logits = self.queued_head(
            torch.cat([ll, t_embed], dim=-1),
        ).squeeze(-1)

        return {
            "type_logits": type_logits.reshape(B, T, cfg.n_action_types),
            "actor_logits": actor_logits.reshape(B, T, U),
            "target_logits": target_logits.reshape(B, T, U),
            "location_logits": location_logits.reshape(B, T, self.n_map_tokens),
            "queued_logits": queued_logits.reshape(B, T),
            # conditioning actually used — losses condition-weight on these
            # and PPO replay teacher-forces them
            "action_type_used": action_type.reshape(B, T),
            "actor_idx": actor_idx.reshape(B, T),
        }

    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Type and actor were committed in :meth:`decode`; sample the
        remaining arguments and score exactly the applicable heads."""
        action_type = pred["action_type_used"]
        actor_idx = pred["actor_idx"]

        with torch.no_grad():
            type_lp = masked_log_prob(
                pred["type_logits"].float(), action_type,
                legal.get("action_type") if legal else None,
            )
            queued_logits = pred["queued_logits"].float()
            queued = torch.bernoulli(torch.sigmoid(queued_logits)).bool()
            queued_lp, _ = bernoulli_log_prob_entropy(queued_logits, queued)

            actor_lp = masked_log_prob(
                pred["actor_logits"].float(), actor_idx,
                legal.get("actor") if legal else None,
            )
            target_mask = legal.get("target_unit") if legal else None
            target_idx = masked_sample(pred["target_logits"], target_mask)
            target_lp = masked_log_prob(
                pred["target_logits"].float(), target_idx, target_mask,
            )
            loc_mask = legal.get("location") if legal else None
            location_idx = masked_sample(pred["location_logits"], loc_mask)
            loc_lp = masked_log_prob(
                pred["location_logits"].float(), location_idx, loc_mask,
            )

            uses = self.arg_uses[action_type]                    # [B, 3]
            zeros = torch.zeros_like(type_lp)
            total_lp = (
                type_lp + queued_lp
                + torch.where(uses[..., 0], actor_lp, zeros)
                + torch.where(uses[..., 1], target_lp, zeros)
                + torch.where(uses[..., 2], loc_lp, zeros)
            )

        return {
            "action_type": action_type,
            "action_actor": actor_idx,
            "action_target": target_idx,
            "action_location": location_idx,
            "action_queued": queued,
            "action_log_prob": total_lp,
        }

    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        action_type = actions["action_type"].long()

        type_lp, type_ent = _log_prob_and_entropy(
            pred["type_logits"], action_type,
            legal.get("action_type") if legal else None,
        )
        queued_lp, queued_ent = bernoulli_log_prob_entropy(
            pred["queued_logits"], actions["queued"].bool(),
        )
        actor_lp, actor_ent = _log_prob_and_entropy(
            pred["actor_logits"], actions["actor_idx"],
            legal.get("actor") if legal else None,
        )
        target_lp, target_ent = _log_prob_and_entropy(
            pred["target_logits"], actions["target_idx"],
            legal.get("target_unit") if legal else None,
        )
        loc_lp, loc_ent = _log_prob_and_entropy(
            pred["location_logits"], actions["location_idx"],
            legal.get("location") if legal else None,
        )

        uses = self.arg_uses[action_type]                        # [B, T, 3]
        zeros = torch.zeros_like(type_lp)
        total_lp = (
            type_lp + queued_lp
            + torch.where(uses[..., 0], actor_lp, zeros)
            + torch.where(uses[..., 1], target_lp, zeros)
            + torch.where(uses[..., 2], loc_lp, zeros)
        )
        total_ent = (
            type_ent + queued_ent
            + torch.where(uses[..., 0], actor_ent, zeros)
            + torch.where(uses[..., 1], target_ent, zeros)
            + torch.where(uses[..., 2], loc_ent, zeros)
        )

        # Per-head mean entropy over the frames where the head applies
        with torch.no_grad():
            w = uses.float()
            n = [w[..., i].sum().clamp(min=1.0) for i in range(3)]
            head_entropy = {
                "ent_type": type_ent.detach().mean(),
                "ent_queued": queued_ent.detach().mean(),
                "ent_actor": (actor_ent.detach() * w[..., 0]).sum() / n[0],
                "ent_target": (target_ent.detach() * w[..., 1]).sum() / n[1],
                "ent_location": (loc_ent.detach() * w[..., 2]).sum() / n[2],
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Masked CE / BCE per head.  Argument heads are (a) gated by
        ARG_USES of the ground-truth type — an unused argument has no
        target — and (b) condition-weighted by whether the conditioning
        indices committed in decode matched the ground truth, because
        logits computed under wrong conditioning are not supervisable."""
        cfg = self.cfg
        B, T, K = pred["type_logits"].shape
        U = pred["actor_logits"].shape[-1]
        M = pred["location_logits"].shape[-1]
        valid = seq_mask.float()
        n_valid = valid.sum().clamp(min=1.0)

        tgt_type = tgt["action_type"].long().clamp(0, K - 1)
        tgt_actor = tgt["actor_idx"].long().clamp(0, U - 1)
        tgt_target = tgt["target_idx"].long().clamp(0, U - 1)
        tgt_loc = tgt["location_idx"].long().clamp(0, M - 1)

        def _masked_ce(logits, target, weight, denom):
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                reduction="none",
            ).reshape(B, T).nan_to_num(0.0)
            return (ce * weight).sum() / denom

        type_loss = _masked_ce(pred["type_logits"], tgt_type, valid, n_valid)

        queued_bce = F.binary_cross_entropy_with_logits(
            pred["queued_logits"], tgt["queued"].float(), reduction="none",
        ).nan_to_num(0.0)
        queued_loss = (queued_bce * valid).sum() / n_valid

        uses_gt = self.arg_uses[tgt_type].float()                # [B, T, 3]
        type_match = (pred["action_type_used"] == tgt_type).float()
        actor_match = (pred["actor_idx"] == tgt_actor).float()

        actor_w = valid * uses_gt[..., 0] * type_match
        n_actor = actor_w.sum().clamp(min=1.0)
        actor_loss = _masked_ce(pred["actor_logits"], tgt_actor, actor_w, n_actor)

        target_w = valid * uses_gt[..., 1] * actor_match
        n_target = target_w.sum().clamp(min=1.0)
        target_loss = _masked_ce(pred["target_logits"], tgt_target, target_w, n_target)

        loc_w = valid * uses_gt[..., 2] * type_match * actor_match
        n_loc = loc_w.sum().clamp(min=1.0)
        loc_loss = _masked_ce(pred["location_logits"], tgt_loc, loc_w, n_loc)

        loss = type_loss + queued_loss + actor_loss + target_loss + loc_loss

        with torch.no_grad():
            type_ok = pred["type_logits"].argmax(-1) == tgt_type
            actor_ok = pred["actor_logits"].argmax(-1) == tgt_actor
            target_ok = pred["target_logits"].argmax(-1) == tgt_target
            loc_ok = pred["location_logits"].argmax(-1) == tgt_loc
            queued_ok = (pred["queued_logits"] > 0) == tgt["queued"].bool()
            metrics = {
                "type_loss": type_loss.detach(),
                "queued_loss": queued_loss.detach(),
                "actor_loss": actor_loss.detach(),
                "target_loss": target_loss.detach(),
                "location_loss": loc_loss.detach(),
                "type_acc": (type_ok.float() * valid).sum() / n_valid,
                "queued_acc": (queued_ok.float() * valid).sum() / n_valid,
                "actor_acc": (actor_ok.float() * actor_w).sum() / n_actor,
                "target_acc": (target_ok.float() * target_w).sum() / n_target,
                "location_acc": (loc_ok.float() * loc_w).sum() / n_loc,
            }
        return {"loss": loss, **metrics}
