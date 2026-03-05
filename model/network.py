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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Pre-activation residual block for the spatial encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SpatialEncoder(nn.Module):
    """
    Create Embeddings — converts raw grid channels into per-cell feature vectors.

    Input : [B, C_in, H, W]  (C_in = 12 = 11 type one-hot + 1 army)
    Output: [B, d_model, H, W]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c_mid = cfg.encoder_channels
        self.net = nn.Sequential(
            nn.Conv2d(cfg.grid_channels, c_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            ResBlock(c_mid),
            nn.Conv2d(c_mid, cfg.d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.d_model),
            nn.ReLU(inplace=True),
            ResBlock(cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HistoryEncoder(nn.Module):
    """
    GRU-based temporal encoder.

    Receives the spatially-pooled embedding at each timestep and produces
    a temporal-context vector that is broadcast-added back into the spatial
    feature map.

    Input  : [B, T, d_model]   (pooled spatial)
    Output : [B, T, d_model],  hidden state
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gru = nn.GRU(
            cfg.d_model, cfg.d_model,
            num_layers=cfg.gru_layers,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.gru(x, hidden)
        return out, hidden


# ---------------------------------------------------------------------------
# Transformer core  (shared-weight pondering, applied k times)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One transformer block: Self-Attention → Add & Norm → MLP → Add & Norm.

    In the pondering setup the *same* block is applied k times.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual + norm
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # MLP + residual + norm
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Policy heads  (autoregressive: source → direction / split)
# ---------------------------------------------------------------------------

class SourceHead(nn.Module):
    """Predict which tile to select as movement source.  [B, H*W] logits."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_tokens: [B, N, d]
            legal_mask:     [B, N]  bool – True where move is legal
        Returns:
            logits: [B, N]
        """
        logits = self.proj(spatial_tokens).squeeze(-1)  # [B, N]
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return logits


class DirectionHead(nn.Module):
    """Predict movement direction conditioned on source embedding."""

    def __init__(self, d_model: int, n_dirs: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_dirs),
        )

    def forward(self, source_embed: torch.Tensor, decision_token: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([source_embed, decision_token], dim=-1)
        return self.net(combined)


class SplitHead(nn.Module):
    """Predict split ratio (all-but-1 vs 50%) conditioned on source embedding."""

    def __init__(self, d_model: int, n_split: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_split),
        )

    def forward(self, source_embed: torch.Tensor, decision_token: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([source_embed, decision_token], dim=-1)
        return self.net(combined)


class ValueHead(nn.Module):
    """Predict game value from the high-level decision token."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def forward(self, hl_token: torch.Tensor) -> torch.Tensor:
        return self.net(hl_token).squeeze(-1)  # [B]


# ---------------------------------------------------------------------------
# World model (optional – for RL curiosity reward)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """
    Predicts next-state embedding from (current embedding, action).

    During IL training this module is unused (caller feeds zeros).
    During RL it provides intrinsic reward via prediction error.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        action_dim = cfg.grid_h * cfg.grid_w + cfg.n_directions + cfg.n_split
        self.encoder = nn.Sequential(
            nn.Linear(cfg.d_model + action_dim, cfg.world_model_hidden),
            nn.ReLU(),
            nn.Linear(cfg.world_model_hidden, cfg.d_model),
        )

    def forward(
        self,
        state_embed: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state_embed, action_onehot], dim=-1)
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

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
        self.row_embed = nn.Parameter(torch.randn(1, cfg.grid_h, 1, cfg.d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, cfg.grid_w, cfg.d_model) * 0.02)

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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------

    def _build_spatial_pe(self, BT: int) -> torch.Tensor:
        """Return [1, H*W, d] positional encoding from row+col embeddings."""
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
        spatial = self.spatial_encoder(grid_seq.reshape(BT, C, H, W))  # [BT, d, H, W]
        spatial = spatial.reshape(B, T, d, H, W)

        # ---- 2. Broadcast global features into spatial map -------------
        g = self.global_proj(globals_seq)  # [B, T, d]
        spatial = spatial + g.unsqueeze(-1).unsqueeze(-1)  # broadcast [B,T,d,1,1]

        # ---- 3. History Encoder ----------------------------------------
        pooled = spatial.mean(dim=(-2, -1))  # [B, T, d]
        temporal, hidden = self.history_encoder(pooled, hidden)  # [B, T, d]
        spatial = spatial + temporal.unsqueeze(-1).unsqueeze(-1)

        # ---- 4. Flatten to token sequence ------------------------------
        tokens = spatial.reshape(BT, d, N).permute(0, 2, 1)  # [BT, N, d]

        # ---- 5. Add learnable decision tokens --------------------------
        hl = self.hl_token.expand(BT, -1, -1)  # [BT, 1, d]
        ll = self.ll_token.expand(BT, -1, -1)  # [BT, 1, d]
        tokens = torch.cat([tokens, hl, ll], dim=1)  # [BT, N+2, d]

        # ---- 6. Positional encoding ------------------------------------
        spatial_pe = self._build_spatial_pe(BT)  # [BT, N, d]
        hl_pe = self.hl_pos.expand(BT, -1, -1)
        ll_pe = self.ll_pos.expand(BT, -1, -1)
        pe = torch.cat([spatial_pe, hl_pe, ll_pe], dim=1)  # [BT, N+2, d]
        tokens = tokens + pe

        # ---- (optional) World-model embedding --------------------------
        if world_model_embed is not None:
            wm = world_model_embed.reshape(BT, 1, d)
            tokens[:, N:N + 1] += wm  # add to HL token

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
        # Flatten legal mask
        if legal_mask is not None:
            lm = legal_mask.reshape(BT, N)  # [BT, N]
        else:
            lm = None

        source_logits = self.source_head(spatial_out, lm)  # [BT, N]

        # Get source embedding (autoregressive conditioning)
        if source_y is not None and source_x is not None:
            src_idx = (source_y.reshape(BT) * W + source_x.reshape(BT)).long()
            src_idx = src_idx.clamp(0, N - 1)
        else:
            # Inference: sample from source distribution
            with torch.no_grad():
                probs = F.softmax(source_logits.detach(), dim=-1)
                src_idx = torch.multinomial(probs, 1).squeeze(-1)  # [BT]

        source_embed = spatial_out[torch.arange(BT, device=spatial_out.device), src_idx]  # [BT, d]

        dir_logits = self.direction_head(source_embed, ll_out)    # [BT, n_dirs]
        split_logits = self.split_head(source_embed, ll_out)      # [BT, n_split]
        value = self.value_head(hl_out)                           # [BT]

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
            legal_mask=legal_mask.unsqueeze(1) if legal_mask is not None else None,
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
