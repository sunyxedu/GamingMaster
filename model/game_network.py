"""Game-agnostic core network — two-stage architecture:

  Stage 1 — History Encoder
            SpatiotemporalBlock × L (separate weights)

  Stage 2 — TRM (Tiny Recursive Model)
            Two-state recursive pondering with shared weights.
            z_L (reasoning scratchpad) and z_H (answer state) are
            iteratively refined by a single shared reasoning module.

The core never changes across games.  Per-game modules plug in at the
boundary defined in ``model.games.base``:

    obs dict → ObservationEncoder → game tokens [B, T, S, d]
             → (+ HL/LL decision tokens, + PE)
             → Stage 1 (spatiotemporal) → Stage 2 (TRM pondering)
             → ActionDecoder → action logits / samples / log-probs
             → ValueHead (from the HL decision token, shared)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.components.heads import ValueHead
from model.components.transformer import SpatiotemporalBlock, _sinusoidal_pe
from model.components.trm import TRM
from model.config import ModelConfig
from model.games.base import ActionDecoder, ObservationEncoder


class GameNetwork(nn.Module):
    """Generic two-stage network parameterised by a game adapter.

        ObservationEncoder → (+decision tokens, +PE) →
        Stage 1: History Encoding (spatiotemporal) →
        Stage 2: TRM (two-state recursive pondering) →
        ActionDecoder + ValueHead
    """

    def __init__(
        self,
        cfg: ModelConfig,
        obs_encoder: ObservationEncoder,
        action_decoder: ActionDecoder,
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.temporal_type != "spatiotemporal":
            raise ValueError(
                f"unsupported temporal_type {cfg.temporal_type!r}; the GRU "
                "path was removed — only 'spatiotemporal' exists"
            )

        # --- Game adapter modules ---
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder

        # --- Stage 1: History Encoder ---
        self.st_layers = nn.ModuleList(
            [SpatiotemporalBlock(cfg) for _ in range(cfg.n_st_layers)]
        )

        # --- Stage 2: TRM ---
        self.trm = TRM(cfg)
        # q_halt is read from the HL decision token, which sits directly
        # after the game tokens.  (For generals this equals the config
        # default grid_h*grid_w; other games have other token counts.)
        self.trm.q_token_idx = obs_encoder.n_tokens

        # --- Learnable decision tokens ---
        self.hl_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.ll_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.hl_pos = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.ll_pos = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # --- Norms ---
        self.pre_norm = nn.LayerNorm(cfg.d_model)
        self.mid_norm = nn.LayerNorm(cfg.d_model)
        self.post_norm = nn.LayerNorm(cfg.d_model)

        # --- Value head (reads the HL decision token; shared by all games) ---
        self.value_head = ValueHead(cfg.d_model)

        if cfg.predict_next_obs and not obs_encoder.supports_next_obs:
            raise ValueError(
                f"predict_next_obs=True but {type(obs_encoder).__name__} "
                "does not implement next-observation prediction"
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.trm.q_head:
                    continue  # TRM q_head has custom init
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Shared building blocks
    # ------------------------------------------------------------------

    def _build_pe(self) -> torch.Tensor:
        """Game-token PE (from the encoder) + decision-token PE: [1, S+2, d]."""
        return torch.cat(
            [self.obs_encoder.positional_embedding(), self.hl_pos, self.ll_pos],
            dim=1,
        )

    def _trm_ponder(
        self, x: torch.Tensor, act_steps: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run ``act_steps`` ACT steps of the TRM with detached carry.

        The first ``act_steps - 1`` steps run without gradient; only the
        final step is differentiated (deep-supervision one-step gradient,
        matching the TRM recursion itself).  ``act_steps=None`` → 1 step.
        """
        steps = max(1, act_steps or 1)
        carry = None
        if steps > 1:
            with torch.no_grad():
                for _ in range(steps - 1):
                    _, _, carry = self.trm(x, carry)
        z_H, q_halt, _ = self.trm(x, carry)
        return z_H, q_halt

    def _trm_act_loop(
        self,
        x: torch.Tensor,
        act_steps: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-time ACT: fixed depth when ``act_steps`` is given
        (needed for exact policy replay in RL), q_halt-gated otherwise."""
        z_H, q_halt, carry = self.trm(x)
        if act_steps is not None:
            for _ in range(max(1, act_steps) - 1):
                z_H, q_halt, carry = self.trm(x, carry)
        else:
            for _ in range(self.cfg.halt_max_steps - 1):
                if (q_halt > 0).all():
                    break
                z_H, q_halt, carry = self.trm(x, carry)
        return z_H, q_halt

    # ------------------------------------------------------------------
    # Full-sequence forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        act_steps: Optional[int] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        d = cfg.d_model

        # 1. Game-specific encoding → [B, T, S, d]
        tokens = self.obs_encoder(obs, frame_mask)
        B, T, S_game, _ = tokens.shape
        BT = B * T

        # 2. Decision tokens
        hl = self.hl_token.expand(B, T, -1, -1)
        ll = self.ll_token.expand(B, T, -1, -1)
        tokens = torch.cat([tokens, hl, ll], dim=2)  # [B, T, S, d]

        # 3. PE — absolute time PE only for the attention mixer; the
        # recurrence carries temporal order itself and absolute positions
        # would only reintroduce step-index dependence.
        tokens = tokens + self._build_pe().unsqueeze(0)
        if cfg.temporal_mixer != "recurrent":
            tokens = tokens + _sinusoidal_pe(T, d, tokens.device).unsqueeze(0).unsqueeze(2)

        # ── Stage 1: Spatiotemporal encoding (checkpointed) ──
        tokens = self.pre_norm(tokens)
        for layer in self.st_layers:
            tokens = checkpoint(layer, tokens, use_reentrant=True)
        tokens = self.mid_norm(tokens)

        # ── Stage 2: TRM pondering ──
        S = tokens.shape[2]
        x = tokens.reshape(BT, S, d)                  # input embeddings for TRM
        z_H, q_halt = self._trm_ponder(x, act_steps)  # [BT, S, d], [BT]
        z_H = self.post_norm(z_H)

        # ── Decode ──
        game_out = z_H[:, :S_game]
        hl_out = z_H[:, S_game]
        ll_out = z_H[:, S_game + 1]

        out = self.action_decoder.decode(
            game_out, hl_out, ll_out, B=B, T=T, t0=0, legal=legal, labels=labels,
        )
        out["value"] = self.value_head(hl_out).reshape(B, T)
        out["q_halt"] = q_halt.reshape(B, T)

        # Self-supervised next-observation prediction from belief tokens
        if cfg.predict_next_obs:
            out.update(self.obs_encoder.next_obs_outputs(tokens[:, :, :S_game]))
        return out

    # ------------------------------------------------------------------
    # Single-step inference (rollout, with Stage-1 cache)
    # ------------------------------------------------------------------

    def step(
        self,
        obs: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
        st_cache: Optional[List[torch.Tensor]] = None,
        step_idx: int = 0,
        act_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Single-frame inference with Stage-1 temporal cache.

        ``legal`` restricts sampling (IL evaluation); pass ``None`` for RL
        mode where the model may select anything.  Returns the decoder's
        logits (T squeezed away), the sampled action dict, value, q_halt
        and the updated ``st_cache``.
        """
        cfg = self.cfg
        d = cfg.d_model

        tokens = self.obs_encoder.encode_step(obs)    # [B, S_game, d]
        B, S_game, _ = tokens.shape

        tokens = torch.cat([tokens, self.hl_token.expand(B, -1, -1),
                            self.ll_token.expand(B, -1, -1)], dim=1)
        tokens = tokens + self._build_pe()

        if cfg.temporal_mixer == "recurrent":
            # No absolute time PE on tokens (matches forward); the decoder's
            # time feature below still uses the caller-supplied step index.
            # The fixed-size cache carries no step count, so a caller that
            # threads the cache but forgets step_idx would silently freeze
            # the time feature at 0 — catch it.
            if st_cache is not None and step_idx == 0:
                raise ValueError(
                    "recurrent mode: pass step_idx to step()/step_rl() "
                    "(cache present but step_idx is still 0)"
                )
            t_cur = step_idx
        else:
            t_cur = 0 if st_cache is None else st_cache[0].shape[1]
            t_pe = _sinusoidal_pe(t_cur + 1, d, tokens.device)[t_cur]
            tokens = tokens + t_pe.unsqueeze(0).unsqueeze(0)

        # Stage 1: spatiotemporal with temporal cache
        tokens = self.pre_norm(tokens).unsqueeze(1)
        new_caches: List[torch.Tensor] = []
        for i, layer in enumerate(self.st_layers):
            prev = st_cache[i] if st_cache is not None else None
            tokens, cache_i = layer.forward_step(tokens, prev)
            new_caches.append(cache_i)
        tokens = self.mid_norm(tokens.squeeze(1))

        # Stage 2: TRM with ACT
        z_H, q_halt = self._trm_act_loop(tokens, act_steps)
        z_H = self.post_norm(z_H)

        game_out = z_H[:, :S_game]
        hl_out = z_H[:, S_game]
        ll_out = z_H[:, S_game + 1]

        # Decode as a T=1 sequence at absolute time t_cur, then squeeze
        legal_seq = (
            {k: v.unsqueeze(1) for k, v in legal.items()} if legal else None
        )
        pred = self.action_decoder.decode(
            game_out, hl_out, ll_out, B=B, T=1, t0=t_cur,
            legal=legal_seq, labels=None,
        )
        pred = {k: v.squeeze(1) for k, v in pred.items()}

        # Sample the remaining action components + log-prob for RL rollout
        rl_out = self.action_decoder.sample(pred, legal)

        return {
            **pred,
            "value": self.value_head(hl_out),
            "q_halt": q_halt,
            "st_cache": new_caches,
            **rl_out,
        }

    # ------------------------------------------------------------------
    # PPO: re-evaluate stored actions under the current policy
    # ------------------------------------------------------------------

    def compute_log_prob(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Delegate to the decoder: (log_prob [B,T], entropy [B,T], head_ent)."""
        return self.action_decoder.log_prob_entropy(pred, actions, legal)
