"""Generals.io game adapter — observation encoder and action decoder.

This module contains exactly the game-specific logic that used to live
inside ``GeneralsNetwork``: the conv grid encoder with masked BatchNorm,
the 2-D positional embedding, the next-observation world-model head, and
the autoregressive Source → Direction / Split / Pass action decoder.

The numeric behavior is a faithful port — rollout sampling, PPO log-prob
replay and IL teacher forcing all match the original implementation.

The module also holds the generals pieces of the generic IL contract used
by the multi-game trainers (training/train_il_game.py, train_multi_game.py):
:func:`generals_il_loss` (both decoders' ``il_loss``), the teacher-forcing
labels derived from the stored targets (:func:`source_labels`), the legality
dict read from the observation (:func:`legal_from_obs`) and the
``turn_globals`` features for recorded games (:func:`append_turn_globals`).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.encoder import SpatialEncoder
from model.components.heads import DirectionHead, PassHead, SourceHead, SplitHead
from model.components.transformer import sinusoidal_pe_rows
from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    ObservationEncoder,
    categorical_sample,
    masked_sample,
    safe_masked_logits,
)
from model.losses import ImitationLoss


#: globals cfg.turn_globals appends to the 4 recorded ones (global_features 4 → 7)
TURN_GLOBAL_FEATURES = 3


def append_turn_globals(globals_seq: torch.Tensor) -> torch.Tensor:
    """[B, T, G] → [B, T, G + 3]: the env's ``turn_globals`` features
    ((turn % 50) / 50, turn % 2, turn / 500) for a recorded sequence.

    IL records carry only the 4 base globals and frame t IS turn t — they
    are only ever truncated at the tail (see data_processing/games/generals.py).
    """
    B, T, _ = globals_seq.shape
    t = torch.arange(T, device=globals_seq.device, dtype=torch.float32)
    t = t.view(1, T, 1).expand(B, -1, -1)
    extra = torch.cat([(t % 50) / 50.0, t % 2, t / 500.0], dim=-1)
    return torch.cat([globals_seq, extra.to(globals_seq.dtype)], dim=-1)


def source_labels(tgt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Teacher-forcing labels from the stored targets: the ground-truth
    source cell on move frames (count > 0), −1 elsewhere, where the decoder
    samples its own source — training/train_il.py's ``use_gt`` rule."""
    move = tgt["count"] > 0
    sy, sx = tgt["src_y"][..., 0], tgt["src_x"][..., 0]
    return {"source_y": torch.where(move, sy, torch.full_like(sy, -1)),
            "source_x": torch.where(move, sx, torch.full_like(sx, -1))}


def legal_from_obs(obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Decoder ``legal`` dict from a recorded observation: the source mask,
    plus the per-cell direction mask when the data carries one."""
    legal = {"source": obs["legal_source_mask_seq"].bool()}
    if "direction_mask_seq" in obs:
        legal["direction"] = obs["direction_mask_seq"].bool()
    return legal


#: il_loss statistics taken over from ImitationLoss (besides loss / answer_*);
#: its value / q_halt terms are the generic trainer's (batch_loss)
_IL_STAT_KEYS = ("source_loss", "direction_loss", "split_loss", "pass_loss",
                 "src_acc", "dir_acc", "full_acc", "complete_acc")


def generals_il_loss(
    cfg: ModelConfig,
    pred: Dict[str, torch.Tensor],
    tgt: Dict[str, torch.Tensor],
    seq_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """``ActionDecoder.il_loss`` for generals, delegated to
    :class:`model.losses.ImitationLoss` — the training/train_il.py objective.

    ``loss`` is its policy part: focal source CE + conditional direction /
    split CE + pass BCE (default weights 3.0 / 1.5 / 0.3 / 1.0, focal γ 1.5)
    on the ``seq_mask`` frames.  training.train_il_game.batch_loss adds the
    value term (MSE, or HL-Gauss CE for an HL-Gauss head; weight
    --value_weight, default 0.2 = ImitationLoss's) and, for the TRM core
    only, the q_halt BCE against ``answer_correct`` on ``answer_mask``
    (weight Q_HALT_WEIGHT 0.5 = ImitationLoss's; train_il.py also zeroes it
    for the other cores) — so the summed objective is exactly
    ImitationLoss's.  Needs tgt count / src_y / src_x / dir_id / ratio_id
    and pred's factorized logits (+ ``src_idx``), which the joint decoder
    also provides as views.
    """
    out = ImitationLoss(grid_w=cfg.grid_w).compute(pred, tgt, frame_mask=seq_mask)
    res = {"loss": out["policy_loss"]}
    res.update({k: out[k] for k in _IL_STAT_KEYS})
    # whole move (or pass) exactly right → exact_acc and the TRM q_halt target
    res["answer_correct"] = out["answer_correct"]
    res["answer_mask"] = out["answer_mask"]
    return res


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

        # Time PE for PassHead: absolute frame indices t0 .. t0+T-1 (t0 an
        # int, or a device tensor for a CUDA-graph-captured step)
        pe_dim = self.pass_head.time_pe_dim
        time_pe = sinusoidal_pe_rows(t0, T, pe_dim, device)  # [T, pe]
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
        # mountains/borders) fall back to the raw logits (branch-free: no
        # host sync on the inference path).
        dir_legal = legal.get("direction") if legal else None
        if dir_legal is not None:
            dm = dir_legal.reshape(BT, cfg.n_directions, N).gather(
                2, src_idx.view(BT, 1, 1).expand(BT, cfg.n_directions, 1),
            ).squeeze(-1)                                     # [BT, 4]
            dead = ~dm.any(-1)
            direction_logits = torch.where(
                dead.unsqueeze(-1), direction_logits,
                direction_logits.masked_fill(~dm, -1e9))

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
                dir_idx = categorical_sample(dir_probs)
            dir_lp = F.log_softmax(direction_logits, dim=-1)[arange_B, dir_idx]

            # --- split ---
            if self.greedy:
                spl_idx = split_logits.argmax(-1)
            else:
                spl_probs = F.softmax(split_logits, dim=-1)
                spl_idx = categorical_sample(spl_probs)
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
                # Column-slice write: `sl_masked[bad_rows, 0]` is valid
                # advanced indexing only for a 1-D mask; on [B, T, N] the
                # 2-D mask mis-aligns and raises IndexError.
                sl_masked = sl_masked.clone()
                sl_masked[..., 0] = sl_masked[..., 0].masked_fill(
                    bad_rows, 0.0)
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

    # ------------------------------------------------------------------
    # Imitation loss (generic trainers)
    # ------------------------------------------------------------------

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """See :func:`generals_il_loss` (ImitationLoss, frame-mask aware)."""
        return generals_il_loss(self.cfg, pred, tgt, seq_mask)


class JointGeneralsDecoder(ActionDecoder):
    """AverageJoe-style flat joint-action decoder.

    ONE categorical over ``2*4*N + 1`` outcomes: for every cell, four
    full-army move directions and four half-army (split) directions,
    plus a single global pass read from the HL token. Every concrete
    move competes with every other move in the same softmax, so credit
    lands on the exact arrow played. The factorized Source→Direction
    hierarchy instead scores sources by a direction-agnostic marginal —
    a wall-locked frontier cell can look attractive until the direction
    head is left with nothing but bad options (the observed
    beeline-into-mountains pathology).

    Joint index layout: ``idx = (split * 4 + direction) * N + cell``;
    ``idx == 8N`` is pass.

    Contract compatibility: ``pred`` additionally carries the factorized
    VIEWS (``source/direction/split/pass_logits``) derived as exact
    marginals of the joint distribution, so the magnet KL, the opening
    anchor, halt self-consistency labels and invalidity telemetry
    consume them unchanged.
    """

    #: (drow, dcol) per direction — MUST match dir_id label semantics
    #: (0=down 1=right 2=up 3=left), same as DirectionHead._OFFSETS
    _OFFSETS = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # Neighbor-aware arrow scorer, mirroring the factorized
        # DirectionHead's inputs (source embed + the tile the move would
        # ENTER + decision token). A per-cell head sees only the source
        # token, where "good source" lives but "which way" does not —
        # distilling through it left kl_dir on a ~0.8 plateau, the same
        # uniform-direction failure the old cross-attention head had.
        # One scorer shared across the four directions; the concat-MLP is
        # split into three added projections so the cell/context terms
        # are computed once for all directions.
        self.edge = nn.Parameter(torch.zeros(d))
        self.arrow_cell = nn.Linear(d, d)
        self.arrow_nbr = nn.Linear(d, d, bias=False)
        self.arrow_ctx = nn.Linear(d, d, bias=False)
        self.arrow_out = nn.Linear(d, 2)  # full-move / half-move logits
        self.pass_proj = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))

    # ------------------------------------------------------------------

    def _arrow_logits(self, tokens, ll, BT):
        """[BT, 2D, N] arrow logits; channel = split * D + dir."""
        cfg = self.cfg
        H, W, d = cfg.grid_h, cfg.grid_w, cfg.d_model
        D = cfg.n_directions
        N = H * W
        tok2d = tokens.reshape(BT, H, W, d)
        base = self.arrow_cell(tok2d) + self.arrow_ctx(ll).reshape(
            BT, 1, 1, d)                                    # [BT, H, W, d]
        edge = self.edge.to(tok2d.dtype)
        chans = [None] * (2 * D)
        for di, (drow, dcol) in enumerate(self._OFFSETS):
            nbr = torch.roll(tok2d, shifts=(-drow, -dcol), dims=(1, 2))
            # rolled-in wrap rows/cols are off-board → edge embedding
            if drow == 1:
                nbr = torch.cat([nbr[:, :-1], edge.expand(BT, 1, W, d)], 1)
            elif drow == -1:
                nbr = torch.cat([edge.expand(BT, 1, W, d), nbr[:, 1:]], 1)
            elif dcol == 1:
                nbr = torch.cat([nbr[:, :, :-1], edge.expand(BT, H, 1, d)], 2)
            else:
                nbr = torch.cat([edge.expand(BT, H, 1, d), nbr[:, :, 1:]], 2)
            two = self.arrow_out(F.relu(base + self.arrow_nbr(nbr)))
            chans[0 * D + di] = two[..., 0].reshape(BT, N)  # full move
            chans[1 * D + di] = two[..., 1].reshape(BT, N)  # half move
        return torch.stack(chans, dim=1)                    # [BT, 2D, N]

    def _joint_and_views(self, tokens, hl, ll, BT, legal):
        cfg = self.cfg
        D = cfg.n_directions
        N = cfg.grid_h * cfg.grid_w
        device = tokens.device

        arr = self._arrow_logits(tokens, ll, BT)      # [BT, 2D, N]
        pass_logit = self.pass_proj(hl).squeeze(-1)   # [BT]
        joint_raw = torch.cat(
            [arr.reshape(BT, 2 * D * N), pass_logit.unsqueeze(-1)], dim=-1)

        legal_mask = legal.get("source") if legal else None
        dir_legal = legal.get("direction") if legal else None
        if legal_mask is not None:
            src_ok = legal_mask.reshape(BT, 1, N)
            if dir_legal is not None:
                dir_ok = dir_legal.reshape(BT, D, N)
            else:
                dir_ok = torch.ones(BT, D, N, dtype=torch.bool,
                                    device=device)
            arrow_ok = (src_ok & dir_ok).repeat(1, 2, 1)   # [BT, 2D, N]
            ok = torch.cat(
                [arrow_ok.reshape(BT, 2 * D * N),
                 torch.ones(BT, 1, dtype=torch.bool, device=device)],
                dim=-1)
            joint = joint_raw.masked_fill(~ok, -1e9)
        else:
            joint = joint_raw

        # Pass is always legal, so no all-masked fallback is ever needed;
        # NaN rows (bf16 overflow etc.) fall back to uniform, mirroring
        # safe_masked_logits semantics (branch-free: no host sync).
        nan_rows = joint.isnan().any(dim=-1, keepdim=True)
        joint = joint.masked_fill(nan_rows, 0.0)
        return joint, arr, pass_logit

    def _views(self, arr, pass_logit, src_idx, BT, legal, move_lse):
        """Factorized marginals of the joint distribution (view layer)."""
        cfg = self.cfg
        D = cfg.n_directions
        N = arr.shape[-1]
        device = arr.device

        source_logits = torch.logsumexp(arr, dim=1)            # [BT, N]
        sel = arr.gather(
            2, src_idx.view(BT, 1, 1).expand(BT, 2 * D, 1),
        ).squeeze(-1).reshape(BT, 2, D)                        # [BT, 2, D]

        dir_legal = legal.get("direction") if legal else None
        if dir_legal is not None:
            dm = dir_legal.reshape(BT, D, N).gather(
                2, src_idx.view(BT, 1, 1).expand(BT, D, 1)).squeeze(-1)
            dead = ~dm.any(-1)
            sel = torch.where(dead.view(BT, 1, 1), sel,
                              sel.masked_fill(~dm.unsqueeze(1), -1e9))

        direction_logits = torch.logsumexp(sel, dim=1)         # [BT, D]
        split_logits = torch.logsumexp(sel, dim=2)             # [BT, 2]
        # Bernoulli pass-vs-move logit of the joint policy over LEGAL
        # moves (the masked joint's move mass — the raw arrow mass counts
        # thousands of illegal arrows, inflating the distilled pass logit
        # by ~log(8N/legal) ≈ 5 nats in the opening). Clamped: rows with
        # no legal move have move_lse ≈ -1e9 and the pass decision is
        # forced anyway.
        pass_view = (pass_logit - move_lse).clamp(-30.0, 30.0)
        return source_logits, direction_logits, split_logits, pass_view

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _draw_joint(self, joint: torch.Tensor, move_lse: torch.Tensor) -> torch.Tensor:
        """[BT] joint action index drawn from the (masked) joint policy;
        ``2DN`` = pass."""
        cfg = self.cfg
        pass_idx = 2 * cfg.n_directions * cfg.grid_h * cfg.grid_w
        if self.greedy:
            # Hierarchical greedy. Pass is a single ATOM of the joint but
            # carries the whole "don't move" intent: raw argmax would play
            # pass whenever the move mass is spread over many arrows
            # (p_pass > p_best_arrow even at p_pass ≈ 1%) — a systematic
            # hesitation bias under greedy eval. Decide pass-vs-move by
            # MASS, then take the best arrow.
            best_arrow = joint[..., :-1].argmax(-1)
            return torch.where(joint[..., -1] > move_lse,
                               torch.full_like(best_arrow, pass_idx), best_arrow)
        return categorical_sample(F.softmax(joint.float(), dim=-1))

    def _joint_source(self, j_idx: torch.Tensor) -> torch.Tensor:
        """Source cell of joint indices (cell 0 for pass)."""
        N = self.cfg.grid_h * self.cfg.grid_w
        return torch.where(j_idx >= 2 * self.cfg.n_directions * N,
                           torch.zeros_like(j_idx), j_idx % N)

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
        D = cfg.n_directions
        N = cfg.grid_h * cfg.grid_w
        W = cfg.grid_w

        joint, arr, pass_logit = self._joint_and_views(
            tokens, hl, ll, BT, legal)
        # masked move mass — the pass view's denominator and the greedy
        # pass-vs-move decision are both statements about LEGAL moves
        move_lse = torch.logsumexp(joint[..., :-1], dim=-1)    # [BT]

        source_y = labels.get("source_y") if labels else None
        source_x = labels.get("source_x") if labels else None
        j_idx = None
        if source_y is not None and source_x is not None:
            # Replay / teacher-forcing path: the action tuple is re-encoded
            # by log_prob_entropy; here only the source index is needed to
            # anchor the factorized views.
            sy = source_y.reshape(BT)
            sx = source_x.reshape(BT)
            src_idx = (sy * W + sx).long().clamp(0, N - 1)
            # Unlabelled frames (sy < 0: teacher forcing dropped the label,
            # or a pass / dropped-move frame) anchor the views on the
            # policy's OWN draw, exactly as without labels — like
            # GeneralsDecoder.  Clamping −1 to cell 0 would supervise the
            # conditional direction / split views only on frames whose
            # ground-truth source happens to be cell 0.
            needs_sample = sy < 0
            if needs_sample.any():
                drawn = self._draw_joint(joint, move_lse)
                src_idx = torch.where(needs_sample, self._joint_source(drawn), src_idx)
        else:
            # Rollout: draw the arrow from the joint categorical.
            j_idx = self._draw_joint(joint, move_lse)
            src_idx = self._joint_source(j_idx)

        src_logits, dir_logits, spl_logits, pass_view = self._views(
            arr, pass_logit, src_idx, BT, legal, move_lse)

        pred = {
            "joint_logits": joint.reshape(B, T, 2 * D * N + 1),
            "source_logits": src_logits.reshape(B, T, N),
            "direction_logits": dir_logits.reshape(B, T, D),
            "split_logits": spl_logits.reshape(B, T, 2),
            "pass_logits": pass_view.reshape(B, T),
            "src_idx": src_idx.reshape(B, T),
        }
        if j_idx is not None:
            pred["joint_idx"] = j_idx.reshape(B, T)
        return pred

    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        D = cfg.n_directions
        joint = pred["joint_logits"]           # [B, 8N+1] (T squeezed)
        j_idx = pred["joint_idx"]              # [B]
        B = joint.shape[0]
        N = cfg.grid_h * cfg.grid_w

        with torch.no_grad():
            lp = F.log_softmax(joint, dim=-1)
            total_lp = lp.gather(-1, j_idx.unsqueeze(-1)).squeeze(-1)

            did_pass = j_idx >= 2 * D * N
            ch = torch.where(did_pass, torch.zeros_like(j_idx),
                             j_idx // N)
            cell = torch.where(did_pass, torch.zeros_like(j_idx),
                               j_idx % N)
            dir_idx = ch % D
            spl_idx = ch // D

        return {
            "action_source": cell,
            "action_direction": dir_idx,
            "action_split": spl_idx,
            "action_pass": did_pass,
            "action_log_prob": total_lp,
        }

    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        D = cfg.n_directions
        joint = pred["joint_logits"]           # [B, T, 8N+1], masked
        B, T, A = joint.shape
        N = cfg.grid_h * cfg.grid_w

        passed = actions["passed"]             # [B, T] bool
        move_idx = ((actions["split"] * D + actions["direction"]) * N
                    + actions["source_idx"]).long()
        j_idx = torch.where(passed,
                            torch.full_like(move_idx, 2 * D * N),
                            move_idx)

        log_p = F.log_softmax(joint, dim=-1)
        total_lp = log_p.gather(-1, j_idx.unsqueeze(-1)).squeeze(-1)
        probs = torch.exp(log_p)
        total_ent = -(probs * log_p).nan_to_num(0.0).sum(-1)

        with torch.no_grad():
            # Per-head telemetry from the factorized views
            sl = F.log_softmax(pred["source_logits"], dim=-1)
            src_ent = -(torch.exp(sl) * sl).nan_to_num(0.0).sum(-1)
            dl = F.log_softmax(pred["direction_logits"], dim=-1)
            dir_ent = -(torch.exp(dl) * dl).sum(-1)
            spl = F.log_softmax(pred["split_logits"], dim=-1)
            spl_ent = -(torch.exp(spl) * spl).sum(-1)
            p_pass = torch.sigmoid(pred["pass_logits"])
            pass_ent = -(
                p_pass * F.logsigmoid(pred["pass_logits"])
                + (1 - p_pass) * F.logsigmoid(-pred["pass_logits"]))
            mask_f = (~passed).float()
            n_moves = mask_f.sum().clamp(min=1)
            head_entropy = {
                "ent_pass": pass_ent.detach().mean(),
                "ent_src": (src_ent.detach() * mask_f).sum() / n_moves,
                "ent_dir": (dir_ent.detach() * mask_f).sum() / n_moves,
                "ent_spl": (spl_ent.detach() * mask_f).sum() / n_moves,
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """ImitationLoss on the factorized views (:func:`generals_il_loss`) —
        what training/train_il.py optimises for a joint_decoder config."""
        return generals_il_loss(self.cfg, pred, tgt, seq_mask)
