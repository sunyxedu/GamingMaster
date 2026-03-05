"""
Generals.io neural network — AlphaStar-inspired Transformer architecture.

Architecture (matching the diagram):
  Input (Grid Types 11ch + Grid Number 1ch + Globals 4-dim)
    → Create Embeddings (SpatialEncoder)
    → History Encoder (GRU)
    → [spatial tokens] + [High-level Decision token] + [Low-level Decision token]
    → Transformer Block ×k  (shared-weight pondering)
    → Decode
    → Policy Heads: Source, Direction, Split
    → Value Head
    → (optional) World Model Prediction Error
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.encoder import HistoryEncoder, SpatialEncoder
from model.components.heads import DirectionHead, SourceHead, SplitHead, ValueHead
from model.components.transformer import TransformerBlock
from model.components.world_model import WorldModel
from model.config import ModelConfig


class GeneralsNetwork(nn.Module):
    """
    Full network for the generals.io bot.

    Supports Imitation Learning (IL) and is structured for future
    Reinforcement Learning (RL) with an optional world-model head.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # --- Create Embeddings ---
        self.spatial_encoder = SpatialEncoder(cfg)

        # Global feature projection (broadcast to spatial grid)
        self.global_proj = nn.Sequential(
            nn.Linear(cfg.global_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # --- History Encoder ---
        self.history_encoder = HistoryEncoder(cfg)

        # --- Learnable decision tokens ---
        self.hl_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.ll_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # --- 2-D positional encoding (row + col, additive) ---
        self.row_embed = nn.Parameter(
            torch.randn(1, cfg.grid_h, 1, cfg.d_model) * 0.02,
        )
        self.col_embed = nn.Parameter(
            torch.randn(1, 1, cfg.grid_w, cfg.d_model) * 0.02,
        )

        # Separate positional embeddings for the two decision tokens
        self.hl_pos = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.ll_pos = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # --- Transformer block (shared weights, applied k times) ---
        self.transformer_block = TransformerBlock(cfg)
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

    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------

    def _build_spatial_pe(self, BT: int) -> torch.Tensor:
        """Return [BT, H*W, d] positional encoding from row+col embeddings."""
        pe = self.row_embed + self.col_embed  # [1, H, W, d]
        return pe.reshape(1, -1, self.cfg.d_model).expand(BT, -1, -1)

    # ---------------------------------------------------------------

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
            grid_seq     : [B, T, C, H, W]   C=12
            globals_seq  : [B, T, G]          G=4
            legal_mask   : [B, T, 1, H, W]   bool (optional)
            source_y     : [B, T]  int – ground-truth source row (training)
            source_x     : [B, T]  int – ground-truth source col (training)
            hidden       : GRU hidden state (for online inference)
            world_model_embed : [B, T, d]  (zeros for IL, prediction for RL)

        Returns dict:
            source_logits    : [B, T, H*W]
            direction_logits : [B, T, n_dirs]
            split_logits     : [B, T, n_split]
            value            : [B, T]
            hidden           : GRU hidden state
        """
        cfg = self.cfg
        B, T, C, H, W = grid_seq.shape
        N = H * W
        d = cfg.d_model
        BT = B * T

        # ---- 1. Create Embeddings (spatial encoder) --------------------
        spatial = self.spatial_encoder(
            grid_seq.reshape(BT, C, H, W),
        )  # [BT, d, H, W]
        spatial = spatial.reshape(B, T, d, H, W)

        # ---- 2. Broadcast global features into spatial map -------------
        g = self.global_proj(globals_seq)  # [B, T, d]
        spatial = spatial + g.unsqueeze(-1).unsqueeze(-1)

        # ---- 3. History Encoder ----------------------------------------
        pooled = spatial.mean(dim=(-2, -1))  # [B, T, d]
        temporal, hidden = self.history_encoder(pooled, hidden)
        spatial = spatial + temporal.unsqueeze(-1).unsqueeze(-1)

        # ---- 4. Flatten to token sequence ------------------------------
        tokens = spatial.reshape(BT, d, N).permute(0, 2, 1)  # [BT, N, d]

        # ---- 5. Add learnable decision tokens --------------------------
        hl = self.hl_token.expand(BT, -1, -1)
        ll = self.ll_token.expand(BT, -1, -1)
        tokens = torch.cat([tokens, hl, ll], dim=1)  # [BT, N+2, d]

        # ---- 6. Positional encoding ------------------------------------
        spatial_pe = self._build_spatial_pe(BT)
        hl_pe = self.hl_pos.expand(BT, -1, -1)
        ll_pe = self.ll_pos.expand(BT, -1, -1)
        pe = torch.cat([spatial_pe, hl_pe, ll_pe], dim=1)
        tokens = tokens + pe

        # ---- (optional) World-model embedding --------------------------
        if world_model_embed is not None:
            wm = world_model_embed.reshape(BT, 1, d)
            tokens[:, N:N + 1] += wm

        # ---- 7. Transformer pondering (shared block × k) ---------------
        tokens = self.pre_norm(tokens)
        for _ in range(cfg.n_ponder_steps):
            tokens = self.transformer_block(tokens)
        tokens = self.post_norm(tokens)

        # ---- 8. Decode -------------------------------------------------
        spatial_out = tokens[:, :N]       # [BT, N, d]
        hl_out = tokens[:, N]             # [BT, d]
        ll_out = tokens[:, N + 1]         # [BT, d]

        # ---- 9. Policy heads -------------------------------------------
        if legal_mask is not None:
            lm = legal_mask.reshape(BT, N)
        else:
            lm = None

        source_logits = self.source_head(spatial_out, lm)

        # Get source embedding (autoregressive conditioning)
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

        dir_logits = self.direction_head(source_embed, ll_out)
        split_logits = self.split_head(source_embed, ll_out)
        value = self.value_head(hl_out)

        # ---- reshape back to [B, T, …] --------------------------------
        return {
            "source_logits": source_logits.reshape(B, T, N),
            "direction_logits": dir_logits.reshape(B, T, cfg.n_directions),
            "split_logits": split_logits.reshape(B, T, cfg.n_split),
            "value": value.reshape(B, T),
            "hidden": hidden,
        }

    # ---------------------------------------------------------------
    # Convenience: single-step forward for inference / RL rollouts
    # ---------------------------------------------------------------

    def step(
        self,
        grid: torch.Tensor,
        globals_vec: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        world_model_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single-timestep forward (wraps self.forward with T=1).

        Args:
            grid        : [B, C, H, W]
            globals_vec : [B, G]
            legal_mask  : [B, 1, H, W] bool
            hidden      : GRU hidden
        """
        out = self.forward(
            grid.unsqueeze(1),
            globals_vec.unsqueeze(1),
            legal_mask=(
                legal_mask.unsqueeze(1) if legal_mask is not None else None
            ),
            hidden=hidden,
            world_model_embed=(
                world_model_embed.unsqueeze(1)
                if world_model_embed is not None else None
            ),
        )
        squeezed = {}
        for k, v in out.items():
            if k == "hidden":
                squeezed[k] = v  # keep GRU hidden shape [n_layers, B, d]
            elif isinstance(v, torch.Tensor) and v.dim() > 1:
                squeezed[k] = v.squeeze(1)
            else:
                squeezed[k] = v
        return squeezed
