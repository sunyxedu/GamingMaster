"""
Generals.io neural network — the generals adapter of the game-agnostic
two-stage core (see ``model.game_network.GameNetwork``):

  Stage 1 — History Encoder
            SpatiotemporalBlock × L (separate weights)

  Stage 2 — TRM (Tiny Recursive Model)
            Two-state recursive pondering with shared weights.

The game-specific parts (grid conv encoder, 2-D PE, world-model head,
Source → Direction / Split / Pass decoder) live in
``model.games.generals``; this class only adapts the historical public
API (tensor-argument ``forward``/``step``/``step_rl`` signatures, legacy
attribute names, and old checkpoint state-dict keys) onto the generic
core so existing training scripts, tools and checkpoints keep working.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.game_network import GameNetwork
from model.games.generals import GeneralsDecoder, GeneralsEncoder


class GeneralsNetwork(GameNetwork):
    """
    Two-stage architecture:
        SpatialEncoder → (+globals, +PE) →
        Stage 1: History Encoding (spatiotemporal) →
        Stage 2: TRM (two-state recursive pondering) →
        Decode → Source / Direction / Split / Value heads
    """

    #: state-dict key prefixes from before the game-adapter refactor →
    #: their current location.  Old checkpoints are remapped on load.
    LEGACY_KEY_MAP = {
        "spatial_encoder.": "obs_encoder.spatial_encoder.",
        "global_proj.": "obs_encoder.global_proj.",
        "row_embed": "obs_encoder.row_embed",
        "col_embed": "obs_encoder.col_embed",
        "next_obs_head.": "obs_encoder.next_obs_head.",
        "source_head.": "action_decoder.source_head.",
        "direction_head.": "action_decoder.direction_head.",
        "split_head.": "action_decoder.split_head.",
        "pass_head.": "action_decoder.pass_head.",
    }

    def __init__(self, cfg: ModelConfig):
        if getattr(cfg, "joint_decoder", False):
            from model.games.generals import JointGeneralsDecoder
            decoder = JointGeneralsDecoder(cfg)
        else:
            decoder = GeneralsDecoder(cfg)
        super().__init__(cfg, GeneralsEncoder(cfg), decoder)

    # ------------------------------------------------------------------
    # Legacy attribute access (freeze helpers, tests, tools)
    # ------------------------------------------------------------------

    @property
    def spatial_encoder(self):
        return self.obs_encoder.spatial_encoder

    @property
    def global_proj(self):
        return self.obs_encoder.global_proj

    @property
    def row_embed(self):
        return self.obs_encoder.row_embed

    @property
    def col_embed(self):
        return self.obs_encoder.col_embed

    @property
    def next_obs_head(self):
        return self.obs_encoder.next_obs_head

    @property
    def source_head(self):
        return self.action_decoder.source_head

    @property
    def direction_head(self):
        return self.action_decoder.direction_head

    @property
    def split_head(self):
        return self.action_decoder.split_head

    @property
    def pass_head(self):
        return self.action_decoder.pass_head

    def _build_spatial_pe(self) -> torch.Tensor:
        return self._build_pe()

    # ------------------------------------------------------------------
    # Legacy checkpoint compatibility
    # ------------------------------------------------------------------

    @classmethod
    def remap_legacy_keys(cls, state_dict: Dict) -> Dict:
        """Translate pre-refactor state-dict keys to the current layout."""
        remapped = {}
        for k, v in state_dict.items():
            for old, new in cls.LEGACY_KEY_MAP.items():
                if k.startswith(old):
                    k = new + k[len(old):]
                    break
            remapped[k] = v
        return remapped

    @classmethod
    def remap_to_legacy_keys(cls, state_dict: Dict) -> Dict:
        """Inverse of :meth:`remap_legacy_keys`: rewrite a current-format
        state dict with pre-refactor key names, for exporting checkpoints
        to tools pinned to the old layout."""
        remapped = {}
        for k, v in state_dict.items():
            for old, new in cls.LEGACY_KEY_MAP.items():
                if k.startswith(new):
                    k = old + k[len(new):]
                    break
            remapped[k] = v
        return remapped

    #: input-layer weights that may legitimately be WIDER in the current
    #: model than in a checkpoint: enabling derive_army_channels /
    #: turn_globals widens conv1 / global_proj along dim 1. Zero-padding
    #: the checkpoint slice is exactly function-preserving (the new input
    #: channels contribute 0 until trained), so resumed policies, the KL
    #: anchor and every pool snapshot keep their behavior bit-for-bit.
    _EXPANDABLE_DIM1 = (
        "obs_encoder.spatial_encoder.net.0.weight",
        "obs_encoder.global_proj.0.weight",
    )

    def _adapt_input_shapes(self, sd: Dict) -> Dict:
        own = self.state_dict()
        for key in self._EXPANDABLE_DIM1:
            if key not in sd or key not in own:
                continue
            v, target = sd[key], own[key]
            if (v.dim() == target.dim() and v.shape[0] == target.shape[0]
                    and v.shape[2:] == target.shape[2:]
                    and v.shape[1] < target.shape[1]):
                pad = torch.zeros(
                    v.shape[0], target.shape[1] - v.shape[1], *v.shape[2:],
                    dtype=v.dtype, device=v.device)
                sd[key] = torch.cat([v, pad], dim=1)
        return sd

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(
            self._adapt_input_shapes(self.remap_legacy_keys(dict(state_dict))),
            strict=strict, assign=assign,
        )

    # ------------------------------------------------------------------
    # Legacy public API — tensor arguments instead of obs/legal/label dicts
    # ------------------------------------------------------------------

    @staticmethod
    def _pack(
        grid_seq: torch.Tensor,
        globals_seq: torch.Tensor,
        legal_mask: Optional[torch.Tensor],
        source_y: Optional[torch.Tensor] = None,
        source_x: Optional[torch.Tensor] = None,
        direction_mask: Optional[torch.Tensor] = None,
    ):
        obs = {"grid_seq": grid_seq, "globals_seq": globals_seq}
        legal = None
        if legal_mask is not None or direction_mask is not None:
            legal = {}
            if legal_mask is not None:
                legal["source"] = legal_mask
            if direction_mask is not None:
                legal["direction"] = direction_mask
        labels = None
        if source_y is not None and source_x is not None:
            labels = {"source_y": source_y, "source_x": source_x}
        return obs, legal, labels

    def forward(
        self,
        grid_seq: torch.Tensor,
        globals_seq: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        source_y: Optional[torch.Tensor] = None,
        source_x: Optional[torch.Tensor] = None,
        act_steps: Optional[int] = None,
        frame_mask: Optional[torch.Tensor] = None,
        direction_mask: Optional[torch.Tensor] = None,
        carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        active: Optional[torch.Tensor] = None,
        return_carry: bool = False,
        reset: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        obs, legal, labels = self._pack(
            grid_seq, globals_seq, legal_mask, source_y, source_x,
            direction_mask,
        )
        return super().forward(
            obs, legal=legal, labels=labels,
            act_steps=act_steps, frame_mask=frame_mask,
            carry=carry, active=active, return_carry=return_carry, reset=reset,
        )

    # ------------------------------------------------------------------
    # Single-step inference
    # ------------------------------------------------------------------

    def step(
        self,
        grid: torch.Tensor,
        globals_vec: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        st_cache: Optional[List[torch.Tensor]] = None,
        step_idx: int = 0,
        act_steps: Optional[int] = None,
        direction_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        obs, legal, _ = self._pack(grid, globals_vec, legal_mask,
                                   direction_mask=direction_mask)
        return super().step(
            obs, legal=legal, st_cache=st_cache,
            step_idx=step_idx, act_steps=act_steps,
        )

    def step_rl(
        self,
        grid: torch.Tensor,
        globals_vec: torch.Tensor,
        st_cache: Optional[List[torch.Tensor]] = None,
        step_idx: int = 0,
        act_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Single-step inference without legal mask (RL mode).

        Same as ``step()`` but the model can select any position, and the
        source log-prob uses the full unmasked distribution.

        ``act_steps`` forces a fixed TRM depth so the PPO update can
        recompute the exact same policy with ``forward(act_steps=...)``.
        """
        return self.step(
            grid, globals_vec, legal_mask=None,
            st_cache=st_cache, step_idx=step_idx, act_steps=act_steps,
        )

    # ------------------------------------------------------------------
    # RL: log-probability computation for stored actions
    # ------------------------------------------------------------------

    def compute_action_log_prob(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions under the **current** policy.

        Used during PPO update.  ``pred`` must have been produced by
        ``forward()`` with ``source_y / source_x`` set to the stored source
        action so that direction/split logits are conditioned on the correct
        source embedding.

        Args:
            pred: model output with
                source_logits    [B, T, N]  (raw, unmasked)
                direction_logits [B, T, 4]
                split_logits     [B, T, 2]  (0=all-but-1, 1=50%)
                pass_logits      [B, T]     (>0 → pass)
            actions: dict with
                source_idx  [B, T]   flat index 0..N-1
                direction   [B, T]   0..3
                split       [B, T]   0..1
                passed      [B, T]   bool — True if action was pass
            legal_mask: [B, T, 1, H, W] or ``None``

        Returns:
            log_prob  [B, T]  hierarchical log-prob including pass (differentiable)
            entropy   [B, T]  sum of per-head entropies (differentiable)
        """
        legal = {"source": legal_mask} if legal_mask is not None else None
        log_prob, entropy, _ = self.compute_log_prob(pred, actions, legal)
        return log_prob, entropy

    def compute_action_log_prob_rl(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Re-evaluate stored actions without legal mask (RL mode).

        Same as ``compute_action_log_prob`` but source logits are not masked,
        and per-head mean entropies are returned for logging.
        """
        return self.compute_log_prob(pred, actions, legal=None)
