"""
Generals.io neural network — AlphaStar-inspired architecture.

Two temporal modes (selected via ModelConfig.temporal_type):

  "gru"             – GRU history encoder  →  spatial-only TransformerBlock ×k
  "spatiotemporal"  – factored SpatiotemporalBlock ×n_layers
                      (spatial self-attn + causal temporal self-attn + MLP)
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


class GeneralsNetwork(nn.Module):
    """
    Full network for the generals.io bot.

    Supports both temporal modes and both IL / RL training.
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

        # --- Temporal backbone ---
        if self.is_st:
            self.st_blocks = nn.ModuleList(
                [SpatiotemporalBlock(cfg) for _ in range(cfg.n_layers)]
            )
        else:
            self.history_encoder = HistoryEncoder(cfg)
            self.transformer_block = TransformerBlock(cfg)

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
        """[1, N+2, d] spatial + decision-token positional encoding."""
        sp = (self.row_embed + self.col_embed).reshape(
            1, -1, self.cfg.d_model,
        )  # [1, N, d]
        return torch.cat([sp, self.hl_pos, self.ll_pos], dim=1)  # [1, N+2, d]

    # ------------------------------------------------------------------
    # Forward — spatiotemporal mode
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
        tokens = torch.cat([tokens, hl, ll], dim=2)  # [B, T, N+2, d]
        S = tokens.shape[2]

        # 4. Spatial PE + sinusoidal temporal PE
        spatial_pe = self._build_spatial_pe()  # [1, S, d]
        temporal_pe = _sinusoidal_pe(T, d, tokens.device)  # [T, d]
        tokens = tokens + spatial_pe.unsqueeze(0)                # broadcast [1,1,S,d]
        tokens = tokens + temporal_pe.unsqueeze(0).unsqueeze(2)  # broadcast [1,T,1,d]

        # 5. Optional world-model embedding
        if world_model_embed is not None:
            wm = world_model_embed.unsqueeze(2)  # [B, T, 1, d]
            tokens[:, :, N:N + 1] += wm

        # 6. Pre-norm
        tokens = self.pre_norm(tokens)

        # 7. Causal mask for temporal attention
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tokens.device),
            diagonal=1,
        )

        # 8. Spatiotemporal blocks
        for block in self.st_blocks:
            tokens = block(tokens, causal)

        # 9. Post-norm
        tokens = self.post_norm(tokens)

        # 10. Decode & heads
        return self._decode_heads(
            tokens.reshape(B * T, S, d), B, T, N, W,
            legal_mask, source_y, source_x,
        )

    # ------------------------------------------------------------------
    # Forward — GRU + spatial pondering mode
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

        spatial_pe = self._build_spatial_pe()
        tokens = tokens + spatial_pe

        if world_model_embed is not None:
            wm = world_model_embed.reshape(BT, 1, d)
            tokens[:, N:N + 1] += wm

        tokens = self.pre_norm(tokens)
        for _ in range(cfg.n_ponder_steps):
            tokens = self.transformer_block(tokens)
        tokens = self.post_norm(tokens)

        out = self._decode_heads(
            tokens, B, T, N, W, legal_mask, source_y, source_x,
        )
        out["hidden"] = hidden
        return out

    # ------------------------------------------------------------------
    # Shared head logic
    # ------------------------------------------------------------------

    def _decode_heads(
        self,
        tokens: torch.Tensor,  # [BT, S, d]
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
        """
        Args:
            grid_seq          : [B, T, C, H, W]
            globals_seq       : [B, T, G]
            legal_mask        : [B, T, 1, H, W] bool
            source_y / source_x : [B, T] int  (ground-truth source for training)
            hidden            : GRU hidden (gru mode only)
            world_model_embed : [B, T, d]
        """
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
        """
        Single-timestep forward for inference.

        For spatiotemporal mode, pass *st_cache* (list of per-block KV
        tensors) and receive the updated cache back.  For gru mode, pass
        *hidden* and receive it back.
        """
        if self.is_st:
            return self._step_spatiotemporal(
                grid, globals_vec, legal_mask,
                world_model_embed, st_cache,
            )
        return self._step_gru(
            grid, globals_vec, legal_mask, hidden, world_model_embed,
        )

    # ── gru step ──

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

    # ── spatiotemporal step (with per-block KV cache) ──

    def _step_spatiotemporal(
        self, grid, globals_vec, legal_mask, wm_embed, st_cache,
    ):
        cfg = self.cfg
        B = grid.shape[0]
        d = cfg.d_model
        H, W = cfg.grid_h, cfg.grid_w
        N = H * W

        # encode current timestep
        spatial = self.spatial_encoder(grid)  # [B, d, H, W]
        g = self.global_proj(globals_vec)     # [B, d]
        spatial = spatial + g.unsqueeze(-1).unsqueeze(-1)

        tokens = spatial.reshape(B, d, N).permute(0, 2, 1)  # [B, N, d]
        hl = self.hl_token.expand(B, -1, -1)
        ll = self.ll_token.expand(B, -1, -1)
        tokens = torch.cat([tokens, hl, ll], dim=1)  # [B, S, d]

        # spatial PE
        spatial_pe = self._build_spatial_pe()
        tokens = tokens + spatial_pe

        # temporal PE (figure out current step from cache)
        t_cur = 0 if st_cache is None else st_cache[0].shape[1]
        t_pe = _sinusoidal_pe(t_cur + 1, d, tokens.device)[t_cur]  # [d]
        tokens = tokens + t_pe.unsqueeze(0).unsqueeze(0)

        if wm_embed is not None:
            tokens[:, N:N + 1] += wm_embed.unsqueeze(1)

        tokens = self.pre_norm(tokens).unsqueeze(1)  # [B, 1, S, d]

        new_cache: List[torch.Tensor] = []
        for i, block in enumerate(self.st_blocks):
            prev = st_cache[i] if st_cache is not None else None
            tokens, cache_i = block.forward_step(tokens, prev)
            new_cache.append(cache_i)

        tokens = self.post_norm(tokens.squeeze(1))  # [B, S, d]

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
            "st_cache": new_cache,
        }
