"""Generals.io game adapter — observation encoder and action decoder.

This module contains exactly the game-specific logic that used to live
inside ``GeneralsNetwork``: the conv grid encoder with masked BatchNorm,
the 2-D positional embedding, the next-observation world-model head, and
the autoregressive Source → Direction / Split / Pass action decoder.

The numeric behavior is a faithful port — rollout sampling, PPO log-prob
replay and IL teacher forcing all match the original implementation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.encoder import SpatialEncoder
from model.components.heads import DirectionHead, PassHead, SourceHead, SplitHead
from model.components.transformer import _sinusoidal_pe
from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    ObservationEncoder,
    masked_sample,
    safe_masked_logits,
)


class GeneralsEncoder(ObservationEncoder):
    """Grid observation → 625 spatial tokens.

    obs keys:
        grid_seq    [B, T, C, H, W]  (11 one-hot state channels + 1 army)
        globals_seq [B, T, G]        (myArmy, myLand, opArmy, opLand)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_tokens = cfg.grid_h * cfg.grid_w

        # Optional derived linear army channels (cfg.derive_army_channels):
        # the conv stack widens by 3; grid_channels stays 12 because it also
        # defines the RAW obs format (next-obs head target, env, IL data).
        in_channels = cfg.grid_channels + (3 if cfg.derive_army_channels else 0)
        self.spatial_encoder = SpatialEncoder(cfg, in_channels=in_channels)
        self.global_proj = nn.Sequential(
            nn.Linear(cfg.global_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # 2-D spatial PE (learnable row + column embeddings, summed)
        self.row_embed = nn.Parameter(
            torch.randn(1, cfg.grid_h, 1, cfg.d_model) * 0.02,
        )
        self.col_embed = nn.Parameter(
            torch.randn(1, 1, cfg.grid_w, cfg.d_model) * 0.02,
        )

        # Optional self-supervised next-observation head: predicts the next
        # frame's state one-hot + army channel per spatial token from the
        # Stage-1 belief — the training pressure that forces the temporal
        # state to actually retain information.
        self.next_obs_head: Optional[nn.Linear] = None
        if cfg.predict_next_obs:
            self.next_obs_head = nn.Linear(cfg.d_model, cfg.grid_channels)

    # ------------------------------------------------------------------

    def _augment(self, grid: torch.Tensor) -> torch.Tensor:
        """Append 3 derived LINEAR army channels: own / enemy / neutral-city
        army as army/50 (clipped at 10 ~ 500 army).

        grid: [..., C, H, W] with the standard 12-channel layout. Channel
        11 stores log1p(army)/10, so army = expm1(10*ch11), inverted in
        fp32 BEFORE any autocast cast — bf16 would destroy exactly the
        precision this feature exists to expose.
        """
        if not self.cfg.derive_army_channels:
            return grid
        army = torch.expm1(grid[..., 11, :, :].float() * 10.0)
        lin = (army / 50.0).clamp(max=10.0)
        own = grid[..., 5, :, :] + grid[..., 6, :, :] + grid[..., 7, :, :]
        enemy = grid[..., 8, :, :] + grid[..., 9, :, :] + grid[..., 10, :, :]
        neutral_city = grid[..., 3, :, :]
        extra = torch.stack(
            [lin * own.float(), lin * enemy.float(),
             lin * neutral_city.float()], dim=-3)
        return torch.cat([grid, extra.to(grid.dtype)], dim=-3)

    def _encode_frames(self, grid_seq, frame_mask):
        """Run the spatial encoder, excluding padded frames from BatchNorm
        statistics when ``frame_mask`` ([B, T] bool, True = real frame) is
        given.  Padded frames get zero features; they are masked in the loss
        and cannot influence real frames because collate pads only at the
        sequence TAIL and both temporal mixers are causal (attention masks
        the future; the recurrence only carries state forward)."""
        B, T, C, H, W = grid_seq.shape
        BT = B * T
        flat = self._augment(grid_seq.reshape(BT, C, H, W))
        if frame_mask is None or bool(frame_mask.all()):
            return self.spatial_encoder(flat).reshape(B, T, -1, H, W)
        fm = frame_mask.reshape(BT)
        enc = self.spatial_encoder(flat[fm])
        feat = enc.new_zeros(BT, enc.shape[1], H, W)
        feat[fm] = enc
        return feat.reshape(B, T, -1, H, W)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        grid_seq = obs["grid_seq"]
        globals_seq = obs["globals_seq"]
        B, T, C, H, W = grid_seq.shape
        d = self.cfg.d_model

        spatial = self._encode_frames(grid_seq, frame_mask)
        spatial = spatial + self.global_proj(globals_seq).unsqueeze(-1).unsqueeze(-1)
        return spatial.reshape(B, T, d, H * W).permute(0, 1, 3, 2)

    def encode_step(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid = obs["grid_seq"]              # [B, C, H, W]
        globals_vec = obs["globals_seq"]    # [B, G]
        B, d = grid.shape[0], self.cfg.d_model

        spatial = self.spatial_encoder(self._augment(grid))
        spatial = spatial + self.global_proj(globals_vec).unsqueeze(-1).unsqueeze(-1)
        return spatial.reshape(B, d, self.n_tokens).permute(0, 2, 1)

    def positional_embedding(self) -> torch.Tensor:
        return (self.row_embed + self.col_embed).reshape(1, -1, self.cfg.d_model)

    # -- world model -----------------------------------------------------

    @property
    def supports_next_obs(self) -> bool:
        return True

    def next_obs_outputs(
        self, belief_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        assert self.next_obs_head is not None
        no = self.next_obs_head(belief_tokens)               # [B, T, N, C]
        return {
            "next_state_logits": no[..., : self.cfg.n_grid_types],
            "next_army": no[..., self.cfg.n_grid_types],
        }


class GeneralsDecoder(ActionDecoder):
    """Autoregressive move decoder: Source → Direction / Split → Pass.

    legal keys:   source    [B, T, 1, H, W] bool — owned tiles with army > 1
                  direction [B, T, 4, H, W] bool (optional) — per-cell
                  direction validity (in-bounds, not a mountain). When given,
                  the returned direction logits are hard-masked (-1e9) at the
                  selected source, so illegal directions are never sampled
                  and carry no mass in log-prob/entropy/KL computations.
    labels keys:  source_y / source_x [B, T]  — teacher-forcing source
                  (−1 → sample from the model's own source distribution)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.source_head = SourceHead(cfg.d_model)
        self.direction_head = DirectionHead(cfg.d_model, cfg.grid_h, cfg.grid_w)
        self.split_head = SplitHead(cfg.d_model, cfg.n_split)
        self.pass_head = PassHead(cfg.d_model, cfg.n_directions, cfg.n_split)

    # ------------------------------------------------------------------
    # Source sampling (no grad)
    # ------------------------------------------------------------------

    def _sample_source(self, source_logits, legal_mask, BT, N):
        """Sample source index from model's own predictions (no grad).

        ``legal_mask`` (flattened-able to [BT, N]) restricts sampling when
        given; without it the model may select any position (RL mode).
        Delegates to the shared :func:`masked_sample` guard (all-masked-row
        fallback + NaN-from-bf16 uniform fallback).
        """
        mask = legal_mask.reshape(BT, N) if legal_mask is not None else None
        if self.greedy:
            with torch.no_grad():
                return safe_masked_logits(source_logits, mask).argmax(-1)
        return masked_sample(source_logits, mask)

    # ------------------------------------------------------------------
    # Decode: tokens → logits
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
        N = self.cfg.grid_h * self.cfg.grid_w
        W = self.cfg.grid_w
        device = tokens.device
        legal_mask = legal.get("source") if legal else None

        source_logits = self.source_head(tokens)  # raw logits, no mask

        source_y = labels.get("source_y") if labels else None
        source_x = labels.get("source_x") if labels else None
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

        source_embed = tokens[torch.arange(BT, device=device), src_idx]

        direction_logits = self.direction_head(
            source_embed, tokens, ll, src_idx,
        )  # [BT, n_dir]
        split_logits = self.split_head(source_embed, ll)  # [BT, n_split]

        # Time PE for PassHead: absolute frame indices t0 .. t0+T-1
        pe_dim = self.pass_head.time_pe_dim
        time_pe = _sinusoidal_pe(t0 + T, pe_dim, device)[t0: t0 + T]  # [T, pe]
        time_pe = time_pe.unsqueeze(0).expand(B, T, pe_dim).reshape(BT, pe_dim)

        # PassHead reads the RAW direction logits (masking first would feed
        # it -1e9 saturation); the mask is applied to what the policy over
        # directions actually sees, below.
        pass_logits = self.pass_head(
            ll, source_embed, direction_logits, split_logits, time_pe,
        )  # [BT]

        # Hard direction mask at the selected source. -1e9 (finite) instead
        # of -inf keeps p*log_p and its backward NaN-free without special
        # casing. Rows with no valid direction (a cell fully enclosed by
        # mountains/borders) fall back to the raw logits.
        dir_legal = legal.get("direction") if legal else None
        if dir_legal is not None:
            dm = dir_legal.reshape(BT, cfg.n_directions, N).gather(
                2, src_idx.view(BT, 1, 1).expand(BT, cfg.n_directions, 1),
            ).squeeze(-1)                                     # [BT, 4]
            masked_dir = direction_logits.masked_fill(~dm, -1e9)
            dead = ~dm.any(-1)
            if dead.any():
                masked_dir = torch.where(
                    dead.unsqueeze(-1), direction_logits, masked_dir)
            direction_logits = masked_dir

        return {
            "source_logits": source_logits.reshape(B, T, N),
            "direction_logits": direction_logits.reshape(B, T, cfg.n_directions),
            "split_logits": split_logits.reshape(B, T, cfg.n_split),
            "pass_logits": pass_logits.reshape(B, T),
            "src_idx": src_idx.reshape(B, T),
        }

    # ------------------------------------------------------------------
    # Rollout sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample direction/split/pass and compute log-probs for all heads.

        Source has already been sampled during :meth:`decode`
        (``pred["src_idx"]``); direction and split are sampled here.  With
        ``legal`` the source log-prob uses the masked distribution; without
        it (RL mode) the full unmasked distribution.
        """
        source_logits = pred["source_logits"]
        src_idx = pred["src_idx"]
        direction_logits = pred["direction_logits"]
        split_logits = pred["split_logits"]
        pass_logits = pred["pass_logits"]
        legal_mask = legal.get("source") if legal else None

        B = source_logits.shape[0]
        N = source_logits.shape[-1]
        device = source_logits.device
        arange_B = torch.arange(B, device=device)

        with torch.no_grad():
            # --- pass decision (Bernoulli; argmax when self.greedy) ---
            if self.greedy:
                did_pass = torch.sigmoid(pass_logits) > 0.5
            else:
                did_pass = torch.bernoulli(torch.sigmoid(pass_logits)).bool()
            pass_lp = torch.where(
                did_pass,
                F.logsigmoid(pass_logits),
                F.logsigmoid(-pass_logits),
            )

            # --- source log-prob ---
            # safe_masked_logits is EXACTLY the distribution masked_sample
            # drew the source from in decode() (same all-masked one-hot-0
            # fallback AND the same NaN-row uniform fallback) — computing
            # the bookkeeping from anything else lets the recorded log-prob
            # diverge from the acted distribution when logits go non-finite.
            lm = legal_mask.reshape(B, N) if legal_mask is not None else None
            safe = safe_masked_logits(source_logits, lm)
            source_lp = F.log_softmax(safe, dim=-1)[arange_B, src_idx]

            # --- direction ---
            if self.greedy:
                dir_idx = direction_logits.argmax(-1)
            else:
                dir_probs = F.softmax(direction_logits, dim=-1)
                dir_idx = torch.multinomial(dir_probs, 1).squeeze(-1)
            dir_lp = F.log_softmax(direction_logits, dim=-1)[arange_B, dir_idx]

            # --- split ---
            if self.greedy:
                spl_idx = split_logits.argmax(-1)
            else:
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

    # ------------------------------------------------------------------
    # PPO log-prob replay
    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        source_logits = pred["source_logits"]          # [B, T, N]
        B, T, N = source_logits.shape
        legal_mask = legal.get("source") if legal else None

        # --- apply legal mask to source logits ---
        # -1e9, NOT -inf: this tensor sits on the GRADIENT path (entropy
        # bonus + policy loss). With -inf, p·log p is 0·(−inf) — the forward
        # is cleaned by nan_to_num but mul's backward still produces NaN and
        # log_softmax's backward spreads it to the unmasked positions (see
        # categorical_entropy in games/base.py). exp(-1e9) == 0 exactly in
        # fp32, so log-probs match the -inf-masked rollout bookkeeping
        # bit-for-bit. All-masked rows fall back to one-hot index 0,
        # mirroring safe_masked_logits / sample().
        if legal_mask is not None:
            lm = legal_mask.reshape(B, T, N)
            sl_masked = source_logits.masked_fill(~lm, -1e9)
            bad_rows = ~lm.any(dim=-1)                           # [B, T]
            if bad_rows.any():
                sl_masked = sl_masked.clone()
                sl_masked[bad_rows, 0] = 0.0
        else:
            sl_masked = source_logits
        # NaN-row fallback mirroring safe_masked_logits: the rollout sampled
        # such rows UNIFORMLY, so the replay must reproduce that (all-zero
        # logits, detached — no gradient through a numerically broken row)
        # instead of propagating NaN into ratio/losses where it would also
        # blind the target_kl guard (NaN > x is False).
        nan_rows = sl_masked.isnan().any(dim=-1)                 # [B, T]
        if nan_rows.any():
            sl_masked = torch.where(
                nan_rows.unsqueeze(-1), torch.zeros_like(sl_masked),
                sl_masked)

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

        # Per-head mean entropy (for logging, fully detached)
        with torch.no_grad():
            mask_f = (~passed).float()
            n_moves = mask_f.sum().clamp(min=1)
            head_entropy = {
                "ent_pass": pass_ent.detach().mean(),
                "ent_src": (src_ent.detach() * mask_f).sum() / n_moves,
                "ent_dir": (dir_ent.detach() * mask_f).sum() / n_moves,
                "ent_spl": (spl_ent.detach() * mask_f).sum() / n_moves,
            }

        return total_lp, total_ent, head_entropy
