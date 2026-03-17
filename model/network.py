"""
Generals.io neural network — two-stage architecture:

  Stage 1 — History Encoder
            spatiotemporal: SpatiotemporalBlock × L (separate weights)
            gru:            GRU over pooled spatial embeddings

  Stage 2 — TRM (Tiny Recursive Model)
            Two-state recursive pondering with shared weights.
            z_L (reasoning scratchpad) and z_H (answer state) are
            iteratively refined by a single shared reasoning module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.components.encoder import HistoryEncoder, SpatialEncoder
from model.components.heads import DirectionHead, PassHead, SourceHead, SplitHead, ValueHead
from model.components.transformer import SpatiotemporalBlock, _sinusoidal_pe
from model.components.trm import TRM
from model.components.world_model import WorldModel
from model.config import ModelConfig


class GeneralsNetwork(nn.Module):
    """
    Two-stage architecture:
        SpatialEncoder → (+globals, +PE) →
        Stage 1: History Encoding (spatiotemporal / GRU) →
        Stage 2: TRM (two-state recursive pondering) →
        Decode → Source / Direction / Split / Value heads
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

        # --- Stage 2: TRM ---
        self.trm = TRM(cfg)

        # --- Learnable decision tokens ---
        self.hl_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.ll_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # --- 2-D spatial PE ---
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
        self.mid_norm = nn.LayerNorm(cfg.d_model)
        self.post_norm = nn.LayerNorm(cfg.d_model)

        # --- Policy heads ---
        self.source_head = SourceHead(cfg.d_model)
        self.direction_head = DirectionHead(cfg.d_model, cfg.n_directions)
        self.split_head = SplitHead(cfg.d_model, cfg.n_split)
        self.pass_head = PassHead(cfg.d_model, cfg.n_directions, cfg.n_split)
        self.value_head = ValueHead(cfg.d_model)

        # --- Optional world model ---
        self.world_model: Optional[WorldModel] = None
        if cfg.use_world_model:
            self.world_model = WorldModel(cfg)

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

    def _build_spatial_pe(self) -> torch.Tensor:
        sp = (self.row_embed + self.col_embed).reshape(1, -1, self.cfg.d_model)
        return torch.cat([sp, self.hl_pos, self.ll_pos], dim=1)

    # ------------------------------------------------------------------
    # Forward — spatiotemporal + TRM
    # ------------------------------------------------------------------

    def _forward_spatiotemporal(
        self, grid_seq, globals_seq, legal_mask, source_y, source_x, wm_embed,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T, C, H, W = grid_seq.shape
        N = H * W
        d = cfg.d_model
        BT = B * T

        # 1. Spatial encoder
        spatial = self.spatial_encoder(
            grid_seq.reshape(BT, C, H, W),
        ).reshape(B, T, d, H, W)

        # 2. Global features
        spatial = spatial + self.global_proj(globals_seq).unsqueeze(-1).unsqueeze(-1)

        # 3. Flatten + decision tokens
        tokens = spatial.reshape(B, T, d, N).permute(0, 1, 3, 2)
        hl = self.hl_token.expand(B, T, -1, -1)
        ll = self.ll_token.expand(B, T, -1, -1)
        tokens = torch.cat([tokens, hl, ll], dim=2)  # [B, T, S, d]

        # 4. PE
        tokens = tokens + self._build_spatial_pe().unsqueeze(0)
        tokens = tokens + _sinusoidal_pe(T, d, tokens.device).unsqueeze(0).unsqueeze(2)

        # 5. World model embedding
        if wm_embed is not None:
            tokens[:, :, N:N + 1] += wm_embed.unsqueeze(2)

        # ── Stage 1: Spatiotemporal encoding (checkpointed) ──
        tokens = self.pre_norm(tokens)
        for layer in self.st_layers:
            tokens = checkpoint(layer, tokens, use_reentrant=True)
        tokens = self.mid_norm(tokens)

        # ── Stage 2: TRM pondering ──
        S = tokens.shape[2]
        x = tokens.reshape(BT, S, d)          # input embeddings for TRM
        z_H, q_halt, _ = self.trm(x)          # [BT, S, d], [BT]
        z_H = self.post_norm(z_H)

        # Decode
        out = self._decode_heads(z_H, B, T, N, W, legal_mask, source_y, source_x)
        out["q_halt"] = q_halt.reshape(B, T)
        return out

    # ------------------------------------------------------------------
    # Forward — GRU + TRM
    # ------------------------------------------------------------------

    def _forward_gru(
        self, grid_seq, globals_seq, legal_mask, source_y, source_x, hidden, wm_embed,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T, C, H, W = grid_seq.shape
        N = H * W
        d = cfg.d_model
        BT = B * T

        spatial = self.spatial_encoder(
            grid_seq.reshape(BT, C, H, W),
        ).reshape(B, T, d, H, W)

        spatial = spatial + self.global_proj(globals_seq).unsqueeze(-1).unsqueeze(-1)

        pooled = spatial.mean(dim=(-2, -1))
        temporal, hidden = self.history_encoder(pooled, hidden)
        spatial = spatial + temporal.unsqueeze(-1).unsqueeze(-1)

        tokens = spatial.reshape(BT, d, N).permute(0, 2, 1)
        hl = self.hl_token.expand(BT, -1, -1)
        ll = self.ll_token.expand(BT, -1, -1)
        tokens = torch.cat([tokens, hl, ll], dim=1)
        tokens = tokens + self._build_spatial_pe()

        if wm_embed is not None:
            tokens[:, N:N + 1] += wm_embed.reshape(BT, 1, d)

        # TRM pondering
        x = self.pre_norm(tokens)
        z_H, q_halt, _ = self.trm(x)
        z_H = self.post_norm(z_H)

        out = self._decode_heads(z_H, B, T, N, W, legal_mask, source_y, source_x)
        out["hidden"] = hidden
        out["q_halt"] = q_halt.reshape(B, T)
        return out

    # ------------------------------------------------------------------
    # Shared head logic
    # ------------------------------------------------------------------

    def _sample_source(self, source_logits, legal_mask, BT, N):
        """Sample source index from model's own predictions (no grad)."""
        with torch.no_grad():
            logits_safe = source_logits.detach().clone().float()
            if legal_mask is not None:
                lm = legal_mask.reshape(BT, N)
                logits_safe = logits_safe.masked_fill(~lm, float("-inf"))
            all_masked = logits_safe.isinf() & (logits_safe < 0)
            bad_rows = all_masked.all(dim=-1)
            if bad_rows.any():
                logits_safe[bad_rows, 0] = 0.0
            # Handle NaN from bf16 overflow: fall back to uniform sampling
            nan_rows = logits_safe.isnan().any(dim=-1)
            if nan_rows.any():
                logits_safe[nan_rows] = 0.0
            probs = F.softmax(logits_safe, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)

    def _decode_heads(self, tokens, B, T, N, W, legal_mask, source_y, source_x):
        cfg = self.cfg
        BT = B * T
        device = tokens.device
        spatial_out = tokens[:, :N]
        hl_out = tokens[:, N]
        ll_out = tokens[:, N + 1]

        source_logits = self.source_head(spatial_out)  # raw logits, no mask

        if source_y is not None and source_x is not None:
            sy = source_y.reshape(BT)
            sx = source_x.reshape(BT)
            gt_idx = (sy * W + sx).long().clamp(0, N - 1)
            needs_sample = (sy < 0)
            if needs_sample.any():
                pred_idx = self._sample_source(source_logits, legal_mask, BT, N)
                src_idx = torch.where(needs_sample, pred_idx, gt_idx)
            else:
                src_idx = gt_idx
        else:
            src_idx = self._sample_source(source_logits, legal_mask, BT, N)

        source_embed = spatial_out[
            torch.arange(BT, device=device), src_idx,
        ]

        direction_logits = self.direction_head(
            source_embed, spatial_out, ll_out,
        )  # [BT, n_dir]
        split_logits = self.split_head(
            source_embed, ll_out,
        )  # [BT, n_split]

        # Time PE for PassHead: [T, pe_dim] → [BT, pe_dim]
        pe_dim = self.pass_head.time_pe_dim
        time_pe = _sinusoidal_pe(T, pe_dim, device)  # [T, pe_dim]
        time_pe = time_pe.unsqueeze(0).expand(B, T, pe_dim).reshape(BT, pe_dim)

        pass_logits = self.pass_head(
            ll_out, source_embed, direction_logits, split_logits, time_pe,
        )  # [BT]

        return {
            "source_logits": source_logits.reshape(B, T, N),
            "direction_logits": direction_logits.reshape(B, T, cfg.n_directions),
            "split_logits": split_logits.reshape(B, T, cfg.n_split),
            "pass_logits": pass_logits.reshape(B, T),
            "value": self.value_head(hl_out).reshape(B, T),
            "src_idx": src_idx.reshape(B, T),
        }

    # ------------------------------------------------------------------
    # RL: action sampling and log-probability computation
    # ------------------------------------------------------------------

    def _sample_actions(
        self,
        source_logits: torch.Tensor,
        direction_logits: torch.Tensor,
        split_logits: torch.Tensor,
        pass_logits: torch.Tensor,
        legal_mask: Optional[torch.Tensor],
        src_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Sample direction/split/pass and compute log-probs for all action heads.

        Called during ``step()`` for RL rollout.  Source has already been
        sampled (``src_idx`` provided); direction and split are sampled here.

        All computations run under ``torch.no_grad`` — gradients are not needed
        during data-collection rollouts.

        Returns dict with keys:
            action_source     [B]
            action_direction  [B]
            action_split      [B]
            action_pass       [B]   bool — True if model chose to pass
            action_log_prob   [B]   (hierarchical log-prob including pass)
        """
        B = source_logits.shape[0]
        N = source_logits.shape[-1]
        device = source_logits.device
        arange_B = torch.arange(B, device=device)

        with torch.no_grad():
            # --- pass decision ---
            did_pass = pass_logits > 0  # [B]
            pass_lp = torch.where(
                did_pass,
                F.logsigmoid(pass_logits),
                F.logsigmoid(-pass_logits),
            )

            # --- source log-prob (masked distribution) ---
            sl_masked = source_logits.detach().clone()
            if legal_mask is not None:
                lm = legal_mask.reshape(B, N)
                sl_masked = sl_masked.masked_fill(~lm, float("-inf"))
            all_inf = sl_masked.isinf() & (sl_masked < 0)
            bad_rows = all_inf.all(dim=-1)
            if bad_rows.any():
                sl_masked[bad_rows, 0] = 0.0
            source_lp = F.log_softmax(sl_masked, dim=-1)[arange_B, src_idx]

            # --- direction ---
            dir_probs = F.softmax(direction_logits, dim=-1)
            dir_idx = torch.multinomial(dir_probs, 1).squeeze(-1)
            dir_lp = F.log_softmax(direction_logits, dim=-1)[arange_B, dir_idx]

            # --- split ---
            spl_probs = F.softmax(split_logits, dim=-1)
            spl_idx = torch.multinomial(spl_probs, 1).squeeze(-1)
            spl_lp = F.log_softmax(split_logits, dim=-1)[arange_B, spl_idx]

            # Hierarchical: pass → only pass_lp; move → pass_lp + move heads
            move_lp = source_lp + dir_lp + spl_lp
            total_lp = torch.where(did_pass, pass_lp, pass_lp + move_lp)

        return {
            "action_source": src_idx,
            "action_direction": dir_idx,
            "action_split": spl_idx,
            "action_pass": did_pass,
            "action_log_prob": total_lp,
        }

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
        source_logits = pred["source_logits"]          # [B, T, N]
        B, T, N = source_logits.shape

        # --- apply legal mask to source logits ---
        if legal_mask is not None:
            lm = legal_mask.reshape(B, T, N)
            sl_masked = source_logits.masked_fill(~lm, float("-inf"))
            all_inf = sl_masked.isinf() & (sl_masked < 0)
            bad_rows = all_inf.all(dim=-1)                       # [B, T]
            if bad_rows.any():
                sl_masked = sl_masked.clone()
                sl_masked[bad_rows, 0] = 0.0
        else:
            sl_masked = source_logits

        # --- pass ---
        pass_logits = pred["pass_logits"]              # [B, T]
        passed = actions["passed"]                     # [B, T] bool
        pass_lp = torch.where(
            passed,
            F.logsigmoid(pass_logits),
            F.logsigmoid(-pass_logits),
        )
        p_pass = torch.sigmoid(pass_logits)
        pass_ent = -(
            p_pass * F.logsigmoid(pass_logits)
            + (1 - p_pass) * F.logsigmoid(-pass_logits)
        )

        # --- source ---
        src_log_p = F.log_softmax(sl_masked, dim=-1)             # [B, T, N]
        src_lp = src_log_p.gather(
            -1, actions["source_idx"].unsqueeze(-1),
        ).squeeze(-1)                                            # [B, T]
        src_probs = torch.exp(src_log_p)
        src_ent = -(src_probs * src_log_p).nan_to_num(0.0).sum(-1)

        # --- direction ---
        dir_log_p = F.log_softmax(pred["direction_logits"], dim=-1)
        dir_lp = dir_log_p.gather(
            -1, actions["direction"].unsqueeze(-1),
        ).squeeze(-1)
        dir_probs = torch.exp(dir_log_p)
        dir_ent = -(dir_probs * dir_log_p).sum(-1)

        # --- split ---
        spl_log_p = F.log_softmax(pred["split_logits"], dim=-1)
        spl_lp = spl_log_p.gather(
            -1, actions["split"].unsqueeze(-1),
        ).squeeze(-1)
        spl_probs = torch.exp(spl_log_p)
        spl_ent = -(spl_probs * spl_log_p).sum(-1)

        # --- hierarchical combination ---
        move_lp = src_lp + dir_lp + spl_lp
        total_lp = torch.where(passed, pass_lp, pass_lp + move_lp)

        move_ent = src_ent + dir_ent + spl_ent
        total_ent = torch.where(passed, pass_ent, pass_ent + move_ent)

        return total_lp, total_ent

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
        step_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        if self.is_st:
            return self._step_st(grid, globals_vec, legal_mask, world_model_embed, st_cache)
        return self._step_gru(grid, globals_vec, legal_mask, hidden, world_model_embed, step_idx)

    def _step_gru(self, grid, globals_vec, legal_mask, hidden, wm_embed, step_idx):
        cfg = self.cfg
        B, d = grid.shape[0], cfg.d_model
        N = cfg.grid_h * cfg.grid_w
        W = cfg.grid_w

        spatial = self.spatial_encoder(grid)
        spatial = spatial + self.global_proj(globals_vec).unsqueeze(-1).unsqueeze(-1)

        pooled = spatial.mean(dim=(-2, -1)).unsqueeze(1)  # [B, 1, d]
        temporal, hidden = self.history_encoder(pooled, hidden)
        spatial = spatial + temporal.squeeze(1).unsqueeze(-1).unsqueeze(-1)

        tokens = spatial.reshape(B, d, N).permute(0, 2, 1)
        tokens = torch.cat([tokens, self.hl_token.expand(B, -1, -1),
                            self.ll_token.expand(B, -1, -1)], dim=1)
        tokens = tokens + self._build_spatial_pe()

        if wm_embed is not None:
            tokens[:, N:N + 1] += wm_embed.unsqueeze(1)

        # TRM with ACT
        x = self.pre_norm(tokens)
        z_H, q_halt, carry = self.trm(x)
        for _ in range(cfg.halt_max_steps - 1):
            if (q_halt > 0).all():
                break
            z_H, q_halt, carry = self.trm(x, carry)
        z_H = self.post_norm(z_H)

        spatial_out = z_H[:, :N]
        hl_out = z_H[:, N]
        ll_out = z_H[:, N + 1]

        source_logits = self.source_head(spatial_out)
        src_idx = self._sample_source(source_logits, legal_mask, B, N)
        source_embed = spatial_out[torch.arange(B, device=z_H.device), src_idx]

        direction_logits = self.direction_head(source_embed, spatial_out, ll_out)
        split_logits = self.split_head(source_embed, ll_out)
        pe_dim = self.pass_head.time_pe_dim
        time_pe = _sinusoidal_pe(step_idx + 1, pe_dim, z_H.device)[step_idx].unsqueeze(0).expand(B, pe_dim)
        pass_logits = self.pass_head(ll_out, source_embed, direction_logits, split_logits, time_pe)
        value = self.value_head(hl_out)

        rl_out = self._sample_actions(
            source_logits, direction_logits, split_logits, pass_logits,
            legal_mask, src_idx,
        )

        return {
            "source_logits": source_logits,
            "direction_logits": direction_logits,
            "split_logits": split_logits,
            "pass_logits": pass_logits,
            "value": value,
            "q_halt": q_halt,
            "hidden": hidden,
            **rl_out,
        }

    def _step_st(self, grid, globals_vec, legal_mask, wm_embed, st_cache):
        cfg = self.cfg
        B, d = grid.shape[0], cfg.d_model
        N = cfg.grid_h * cfg.grid_w

        spatial = self.spatial_encoder(grid)
        spatial = spatial + self.global_proj(globals_vec).unsqueeze(-1).unsqueeze(-1)

        tokens = spatial.reshape(B, d, N).permute(0, 2, 1)
        tokens = torch.cat([tokens, self.hl_token.expand(B, -1, -1),
                            self.ll_token.expand(B, -1, -1)], dim=1)
        tokens = tokens + self._build_spatial_pe()

        t_cur = 0 if st_cache is None else st_cache[0].shape[1]
        t_pe = _sinusoidal_pe(t_cur + 1, d, tokens.device)[t_cur]
        tokens = tokens + t_pe.unsqueeze(0).unsqueeze(0)

        if wm_embed is not None:
            tokens[:, N:N + 1] += wm_embed.unsqueeze(1)

        # Stage 1: spatiotemporal with KV cache
        tokens = self.pre_norm(tokens).unsqueeze(1)
        new_caches: List[torch.Tensor] = []
        for i, layer in enumerate(self.st_layers):
            prev = st_cache[i] if st_cache is not None else None
            tokens, cache_i = layer.forward_step(tokens, prev)
            new_caches.append(cache_i)
        tokens = self.mid_norm(tokens.squeeze(1))

        # Stage 2: TRM with ACT
        z_H, q_halt, carry = self.trm(tokens)
        for _ in range(cfg.halt_max_steps - 1):
            if (q_halt > 0).all():
                break
            z_H, q_halt, carry = self.trm(tokens, carry)
        z_H = self.post_norm(z_H)

        spatial_out = z_H[:, :N]
        hl_out = z_H[:, N]
        ll_out = z_H[:, N + 1]

        source_logits = self.source_head(spatial_out)  # raw logits, no mask
        src_idx = self._sample_source(source_logits, legal_mask, B, N)
        source_embed = spatial_out[torch.arange(B, device=z_H.device), src_idx]

        direction_logits = self.direction_head(source_embed, spatial_out, ll_out)
        split_logits = self.split_head(source_embed, ll_out)
        pe_dim = self.pass_head.time_pe_dim
        time_pe = _sinusoidal_pe(t_cur + 1, pe_dim, z_H.device)[t_cur].unsqueeze(0).expand(B, pe_dim)
        pass_logits = self.pass_head(ll_out, source_embed, direction_logits, split_logits, time_pe)
        value = self.value_head(hl_out)

        # Sample direction/split and compute log-probs for RL rollout
        rl_out = self._sample_actions(
            source_logits, direction_logits, split_logits, pass_logits,
            legal_mask, src_idx,
        )

        return {
            "source_logits": source_logits,
            "direction_logits": direction_logits,
            "split_logits": split_logits,
            "pass_logits": pass_logits,
            "value": value,
            "q_halt": q_halt,
            "st_cache": new_caches,
            **rl_out,
        }
