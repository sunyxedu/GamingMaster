"""
Generals.io neural network — two-stage architecture:

  Stage 1 — History Encoder (spatiotemporal attention × L, separate weights)
            Builds a rich spatiotemporal embedding where every cell knows
            what happened at its position across all past timesteps.

  Stage 2 — TRM / Tiny Recursive Model (spatial-only shared block × k, with Halt?)
            Ponders the encoded state to produce the final decision.
            Same weights are reused k times; optional ACT halting.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.encoder import HistoryEncoder, SpatialEncoder
from model.components.heads import DirectionHead, SourceHead, SplitHead, ValueHead
from model.components.transformer import (
    SpatiotemporalBlock,
    TransformerBlock,
    _sinusoidal_pe,
)
from model.components.world_model import WorldModel
from model.config import ModelConfig

# ---------------------------------------------------------------------------
# ACT — Adaptive Computation Time halting
# ---------------------------------------------------------------------------

class ACTHalt(nn.Module):
    """Per-position halting probability for the TRM pondering stage."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class GeneralsNetwork(nn.Module):
    """
    Two-stage architecture matching the diagram:

        SpatialEncoder → (+globals, +PE)
        → Stage 1: SpatiotemporalBlock × L  (history encoding)
        → Stage 2: TransformerBlock × k     (TRM pondering, shared weights, Halt?)
        → Decode → Source / Direction / Split / Value heads
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.is_st = cfg.temporal_type == "spatiotemporal"

        # --- Create Embeddings ---
        self.spatial_encoder = SpatialEncoder(cfg)
        self.global_proj = nn.Sequential(
            nn.Linear(cfg.global_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # --- Stage 1: History Encoder ---
        if self.is_st:
            self.st_layers = nn.ModuleList(
                [SpatiotemporalBlock(cfg) for _ in range(cfg.n_st_layers)]
            )
        else:
            self.history_encoder = HistoryEncoder(cfg)

        # --- Stage 2: TRM (shared-weight pondering block) ---
        self.trm_block = TransformerBlock(cfg)

        # --- ACT halting (for the TRM stage) ---
        self.halt: Optional[ACTHalt] = None
        if cfg.use_act:
            self.halt = ACTHalt(cfg.d_model)

        # --- Learnable decision tokens ---
        self.hl_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.ll_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # --- 2-D spatial PE (row + col) ---
        self.row_embed = nn.Parameter(
            torch.randn(1, cfg.grid_h, 1, cfg.d_model) * 0.02,
        )
        self.col_embed = nn.Parameter(
            torch.randn(1, 1, cfg.grid_w, cfg.d_model) * 0.02,
        )
        self.hl_pos = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.ll_pos = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # --- Norms ---
        self.pre_norm = nn.LayerNorm(cfg.d_model)
        self.mid_norm = nn.LayerNorm(cfg.d_model)  # between stage 1 and 2
        self.post_norm = nn.LayerNorm(cfg.d_model)

        # --- Policy heads ---
        self.source_head = SourceHead(cfg.d_model)
        self.direction_head = DirectionHead(cfg.d_model, cfg.n_directions)
        self.split_head = SplitHead(cfg.d_model, cfg.n_split)
        self.value_head = ValueHead(cfg.d_model)

        # --- Optional world model ---
        self.world_model: Optional[WorldModel] = None
        if cfg.use_world_model:
            self.world_model = WorldModel(cfg)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu",
                )
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_spatial_pe(self) -> torch.Tensor:
        sp = (self.row_embed + self.col_embed).reshape(
            1, -1, self.cfg.d_model,
        )
        return torch.cat([sp, self.hl_pos, self.ll_pos], dim=1)

    # ------------------------------------------------------------------
    # TRM pondering with optional ACT
    # ------------------------------------------------------------------

    def _ponder_trm(
        self,
        tokens: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self.trm_block k times (shared weights).
        If ACT is on, weight-average intermediate states and return
        the ACT remainder loss.
        """
        if self.halt is None:
            for _ in range(k):
                tokens = self.trm_block(tokens)
            return tokens, tokens.new_tensor(0.0)

        shape = tokens.shape[:-1]
        device = tokens.device
        halted = torch.zeros(shape, device=device)
        output = torch.zeros_like(tokens)
        remainders = torch.zeros(shape, device=device)
        threshold = self.cfg.act_threshold

        for step in range(k):
            tokens = self.trm_block(tokens)
            p = self.halt(tokens)

            still_running = (halted < threshold).float()
            if step == k - 1:
                p_use = still_running * (1.0 - halted)
            else:
                p_use = still_running * p

            halted = halted + p_use
            remainders = remainders + still_running * (1.0 - halted)
            output = output + p_use.unsqueeze(-1) * tokens

        return output, remainders.mean()

    # ------------------------------------------------------------------
    # Forward — spatiotemporal mode (two-stage)
    # ------------------------------------------------------------------

    def _forward_spatiotemporal(
        self,
        grid_seq: torch.Tensor,
        globals_seq: torch.Tensor,
        legal_mask: Optional[torch.Tensor],
        source_y: Optional[torch.Tensor],
        source_x: Optional[torch.Tensor],
        world_model_embed: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T, C, H, W = grid_seq.shape
        N = H * W
        d = cfg.d_model

        # 1. Spatial encoder
        spatial = self.spatial_encoder(
            grid_seq.reshape(B * T, C, H, W),
        ).reshape(B, T, d, H, W)

        # 2. Global features
        g = self.global_proj(globals_seq)
        spatial = spatial + g.unsqueeze(-1).unsqueeze(-1)

        # 3. Flatten → [B, T, N, d], append decision tokens → [B, T, S, d]
        tokens = spatial.reshape(B, T, d, N).permute(0, 1, 3, 2)
        hl = self.hl_token.expand(B, T, -1, -1)
        ll = self.ll_token.expand(B, T, -1, -1)
        tokens = torch.cat([tokens, hl, ll], dim=2)

        # 4. Spatial PE + sinusoidal temporal PE
        spatial_pe = self._build_spatial_pe()
        temporal_pe = _sinusoidal_pe(T, d, tokens.device)
        tokens = tokens + spatial_pe.unsqueeze(0)
        tokens = tokens + temporal_pe.unsqueeze(0).unsqueeze(2)

        # 5. Optional world-model embedding
        if world_model_embed is not None:
            tokens[:, :, N:N + 1] += world_model_embed.unsqueeze(2)

        # ── Stage 1: Spatiotemporal encoding (L layers, separate weights) ──
        tokens = self.pre_norm(tokens)
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        for layer in self.st_layers:
            tokens = layer(tokens, causal)
        tokens = self.mid_norm(tokens)

        # ── Stage 2: TRM pondering (shared block × k, spatial only) ──
        BT = B * T
        S = tokens.shape[2]
        tokens_flat = tokens.reshape(BT, S, d)
        tokens_flat, act_loss = self._ponder_trm(tokens_flat, cfg.n_ponder_steps)
        tokens_flat = self.post_norm(tokens_flat)

        # Decode & heads
        out = self._decode_heads(
            tokens_flat, B, T, N, W,
            legal_mask, source_y, source_x,
        )
        out["act_loss"] = act_loss
        return out

    # ------------------------------------------------------------------
    # Forward — GRU mode
    # ------------------------------------------------------------------

    def _forward_gru(
        self,
        grid_seq: torch.Tensor,
        globals_seq: torch.Tensor,
        legal_mask: Optional[torch.Tensor],
        source_y: Optional[torch.Tensor],
        source_x: Optional[torch.Tensor],
        hidden: Optional[torch.Tensor],
        world_model_embed: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T, C, H, W = grid_seq.shape
        N = H * W
        d = cfg.d_model
        BT = B * T

        spatial = self.spatial_encoder(
            grid_seq.reshape(BT, C, H, W),
        ).reshape(B, T, d, H, W)

        g = self.global_proj(globals_seq)
        spatial = spatial + g.unsqueeze(-1).unsqueeze(-1)

        pooled = spatial.mean(dim=(-2, -1))
        temporal, hidden = self.history_encoder(pooled, hidden)
        spatial = spatial + temporal.unsqueeze(-1).unsqueeze(-1)

        tokens = spatial.reshape(BT, d, N).permute(0, 2, 1)
        hl = self.hl_token.expand(BT, -1, -1)
        ll = self.ll_token.expand(BT, -1, -1)
        tokens = torch.cat([tokens, hl, ll], dim=1)

        tokens = tokens + self._build_spatial_pe()

        if world_model_embed is not None:
            tokens[:, N:N + 1] += world_model_embed.reshape(BT, 1, d)

        # TRM pondering (shared block × k)
        tokens = self.pre_norm(tokens)
        tokens, act_loss = self._ponder_trm(tokens, cfg.n_ponder_steps)
        tokens = self.post_norm(tokens)

        out = self._decode_heads(
            tokens, B, T, N, W, legal_mask, source_y, source_x,
        )
        out["hidden"] = hidden
        out["act_loss"] = act_loss
        return out

    # ------------------------------------------------------------------
    # Shared head logic
    # ------------------------------------------------------------------

    def _decode_heads(
        self,
        tokens: torch.Tensor,
        B: int, T: int, N: int, W: int,
        legal_mask: Optional[torch.Tensor],
        source_y: Optional[torch.Tensor],
        source_x: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        BT = B * T
        spatial_out = tokens[:, :N]
        hl_out = tokens[:, N]
        ll_out = tokens[:, N + 1]

        lm = legal_mask.reshape(BT, N) if legal_mask is not None else None
        source_logits = self.source_head(spatial_out, lm)

        if source_y is not None and source_x is not None:
            src_idx = (
                source_y.reshape(BT) * W + source_x.reshape(BT)
            ).long().clamp(0, N - 1)
        else:
            with torch.no_grad():
                probs = F.softmax(source_logits.detach(), dim=-1)
                src_idx = torch.multinomial(probs, 1).squeeze(-1)

        source_embed = spatial_out[
            torch.arange(BT, device=spatial_out.device), src_idx,
        ]

        return {
            "source_logits": source_logits.reshape(B, T, N),
            "direction_logits": self.direction_head(
                source_embed, ll_out,
            ).reshape(B, T, cfg.n_directions),
            "split_logits": self.split_head(
                source_embed, ll_out,
            ).reshape(B, T, cfg.n_split),
            "value": self.value_head(hl_out).reshape(B, T),
        }

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        grid_seq: torch.Tensor,
        globals_seq: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        source_y: Optional[torch.Tensor] = None,
        source_x: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        world_model_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.is_st:
            return self._forward_spatiotemporal(
                grid_seq, globals_seq, legal_mask,
                source_y, source_x, world_model_embed,
            )
        return self._forward_gru(
            grid_seq, globals_seq, legal_mask,
            source_y, source_x, hidden, world_model_embed,
        )

    # ------------------------------------------------------------------
    # Single-step inference
    # ------------------------------------------------------------------

    def step(
        self,
        grid: torch.Tensor,
        globals_vec: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        world_model_embed: Optional[torch.Tensor] = None,
        st_cache: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.is_st:
            return self._step_spatiotemporal(
                grid, globals_vec, legal_mask,
                world_model_embed, st_cache,
            )
        return self._step_gru(
            grid, globals_vec, legal_mask, hidden, world_model_embed,
        )

    def _step_gru(self, grid, globals_vec, legal_mask, hidden, wm_embed):
        out = self._forward_gru(
            grid.unsqueeze(1),
            globals_vec.unsqueeze(1),
            legal_mask.unsqueeze(1) if legal_mask is not None else None,
            None, None, hidden,
            wm_embed.unsqueeze(1) if wm_embed is not None else None,
        )
        squeezed = {}
        for k, v in out.items():
            if k == "hidden":
                squeezed[k] = v
            elif isinstance(v, torch.Tensor) and v.dim() > 1:
                squeezed[k] = v.squeeze(1)
            else:
                squeezed[k] = v
        return squeezed

    def _step_spatiotemporal(
        self, grid, globals_vec, legal_mask, wm_embed, st_cache,
    ):
        cfg = self.cfg
        B = grid.shape[0]
        d = cfg.d_model
        H, W = cfg.grid_h, cfg.grid_w
        N = H * W

        spatial = self.spatial_encoder(grid)
        g = self.global_proj(globals_vec)
        spatial = spatial + g.unsqueeze(-1).unsqueeze(-1)

        tokens = spatial.reshape(B, d, N).permute(0, 2, 1)
        hl = self.hl_token.expand(B, -1, -1)
        ll = self.ll_token.expand(B, -1, -1)
        tokens = torch.cat([tokens, hl, ll], dim=1)
        tokens = tokens + self._build_spatial_pe()

        t_cur = 0 if st_cache is None else st_cache[0].shape[1]
        t_pe = _sinusoidal_pe(t_cur + 1, d, tokens.device)[t_cur]
        tokens = tokens + t_pe.unsqueeze(0).unsqueeze(0)

        if wm_embed is not None:
            tokens[:, N:N + 1] += wm_embed.unsqueeze(1)

        # Stage 1: spatiotemporal layers with KV cache
        tokens = self.pre_norm(tokens).unsqueeze(1)  # [B, 1, S, d]
        new_caches: List[torch.Tensor] = []
        for i, layer in enumerate(self.st_layers):
            prev = st_cache[i] if st_cache is not None else None
            tokens, cache_i = layer.forward_step(tokens, prev)
            new_caches.append(cache_i)

        tokens = self.mid_norm(tokens.squeeze(1))  # [B, S, d]

        # Stage 2: TRM pondering (spatial only, no cache needed)
        tokens, _ = self._ponder_trm(tokens, cfg.n_ponder_steps)
        tokens = self.post_norm(tokens)

        spatial_out = tokens[:, :N]
        hl_out = tokens[:, N]
        ll_out = tokens[:, N + 1]

        lm = legal_mask.reshape(B, N) if legal_mask is not None else None
        source_logits = self.source_head(spatial_out, lm)

        with torch.no_grad():
            probs = F.softmax(source_logits.detach(), dim=-1)
            src_idx = torch.multinomial(probs, 1).squeeze(-1)
        source_embed = spatial_out[torch.arange(B, device=tokens.device), src_idx]

        return {
            "source_logits": source_logits,
            "direction_logits": self.direction_head(source_embed, ll_out),
            "split_logits": self.split_head(source_embed, ll_out),
            "value": self.value_head(hl_out),
            "st_cache": new_caches,
        }
