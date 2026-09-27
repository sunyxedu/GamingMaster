"""ARC-AGI-3 adapter of the tier-3 network (env/vec_arc3.py is its environment).

Frame (the env's obs): ``grid`` [B, T, 64, 64] uint8 palette indices (0..15), ``delta``
[B, T, 64, 64] bool (cells changed since the previous frame), ``globals`` [B, T, 22]
(available actions, level progress, the previous action and click, level-up / game-over
flags; see env/vec_arc3.py).  No game identity is given: the network has to work out
what an unseen game does from what its actions change (the held-out games).

Encoder: the palette one-hot (16) + the delta plane at full resolution through the
conv stack (conv → ResBlock → conv → ResBlock, :func:`conv_stack`), then a
``arc_patch`` × ``arc_patch`` strided patch projection to d_model: (64 / patch)² tokens
(256 at patch 4) with learned row / column embeddings; the globals MLP is added to
every token.

Decoder (hierarchical, one policy for rollout, replay and IL):

    type   [8]    RESET, ACTION1..ACTION7 from the high-level decision token, masked
                  by the frame's legal types
    patch  [P²]   the clicked patch (pointer over the patch tokens, low-level token as
                  the query) — part of the action only when type = ACTION6
    sub    [p²]   the cell inside the committed patch (MLP of that patch's token and
                  the low-level token): cell (x, y) = (px·p + sx, py·p + sy)

    log π(a) = log π(type) + [type = ACTION6] · (log π(patch) + log π(sub | patch))

``decode`` commits the patch (teacher-forced from ``labels["click_idx"]``, else
sampled) so the sub head conditions on it — the IL / RL replay teacher-forces the
stored click, as FromToDecoder does its from-square.

legal keys (bool, or {0, 1} floats from shards):
    type    [B, T, 8]       legal action types
    click   [B, T, 4096]    legal cells (the env: all of them when ACTION6 is legal)
    patch   [B, T, P²]      compact: legal patches        (RL replay / IL shards)
    sub_of  [B, T, p²]      compact: legal cells of the LABELLED patch
labels:  click_idx [B, T] long (−1 = sample)
actions: type_idx / click_idx [B, T] long (y · 64 + x)
IL targets (tgt): type_idx, click_idx (+ the masks legal_type / legal_patch /
legal_sub_of as ``--il_legal_mask`` inputs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.encoder import conv_stack, encode_frames, encoder_norm_of, uses_batch_stats
from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    MLPHead,
    ObservationEncoder,
    PointerHead,
    masked_log_prob,
    masked_sample,
)
from model.games.board import _categorical_head, _log_prob_and_entropy, _pick_index

GRID = 64
N_COLOURS = 16
N_TYPES = 8
CLICK = 6


@dataclass
class Arc3Config(ModelConfig):
    """ARC-AGI-3 dimensions on top of the shared core config."""

    arc_grid: int = GRID
    arc_patch: int = 4
    arc_colours: int = N_COLOURS
    arc_global_features: int = 22
    encoder_channels: int = 64

    @property
    def arc_patches(self) -> int:
        """Patches per side (64 / patch)."""
        return self.arc_grid // self.arc_patch


def patch_of(click: torch.Tensor, grid: int, patch: int) -> torch.Tensor:
    """Cell index y·grid + x → patch index (y // p)·(grid // p) + x // p."""
    y, x = click // grid, click % grid
    return (y // patch) * (grid // patch) + x // patch


def sub_of(click: torch.Tensor, grid: int, patch: int) -> torch.Tensor:
    """Cell index → its cell inside the patch, (y % p)·p + x % p."""
    y, x = click // grid, click % grid
    return (y % patch) * patch + x % patch


def click_of(patch_idx: torch.Tensor, sub: torch.Tensor, grid: int, patch: int) -> torch.Tensor:
    n = grid // patch
    y = (patch_idx // n) * patch + sub // patch
    x = (patch_idx % n) * patch + sub % patch
    return y * grid + x


def _bool(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if m is None else (m if m.dtype == torch.bool else m > 0.5)


class Arc3Encoder(ObservationEncoder):
    """Palette one-hot + delta → conv stack → patch tokens (see the module docstring)."""

    def __init__(self, cfg: Arc3Config):
        super().__init__()
        self.cfg = cfg
        n = cfg.arc_patches
        if n * cfg.arc_patch != cfg.arc_grid:
            raise ValueError(f"arc_patch {cfg.arc_patch} must divide arc_grid {cfg.arc_grid}")
        self.n_tokens = n * n
        self.conv = conv_stack(cfg.arc_colours + 1, cfg.encoder_channels, cfg.encoder_channels,
                               encoder_norm_of(cfg))
        self.batch_stats = uses_batch_stats(cfg)
        self.patchify = nn.Conv2d(cfg.encoder_channels, cfg.d_model, cfg.arc_patch,
                                  stride=cfg.arc_patch)
        self.global_proj = nn.Sequential(
            nn.Linear(cfg.arc_global_features, cfg.d_model), nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model))
        self.row_embed = nn.Parameter(torch.randn(1, n, 1, cfg.d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, n, cfg.d_model) * 0.02)

    def forward(self, obs: Dict[str, torch.Tensor],
                frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        grid = obs["grid"]
        B, T, H, W = grid.shape
        cfg = self.cfg
        onehot = F.one_hot(grid.long().clamp(0, cfg.arc_colours - 1), cfg.arc_colours)
        planes = torch.cat([onehot, obs["delta"].unsqueeze(-1).to(onehot.dtype)], dim=-1)
        planes = planes.permute(0, 1, 4, 2, 3).reshape(B * T, cfg.arc_colours + 1, H, W)
        planes = planes.to(self.patchify.weight.dtype)
        feat = encode_frames(self.conv, planes, B, T, frame_mask, self.batch_stats)
        feat = self.patchify(feat.reshape(B * T, *feat.shape[2:]))           # [BT, d, n, n]
        d = cfg.d_model
        n = cfg.arc_patches
        tokens = feat.reshape(B, T, d, n * n).permute(0, 1, 3, 2)
        glb = self.global_proj(obs["globals"].to(tokens.dtype))
        return tokens + glb.unsqueeze(2)

    def positional_embedding(self) -> torch.Tensor:
        n, d = self.cfg.arc_patches, self.cfg.d_model
        return (self.row_embed + self.col_embed).reshape(1, n * n, d)


class Arc3Decoder(ActionDecoder):
    """type → (ACTION6) patch → cell (see the module docstring)."""

    def __init__(self, cfg: Arc3Config):
        super().__init__()
        self.cfg = cfg
        self.n_patches = cfg.arc_patches ** 2
        self.n_sub = cfg.arc_patch ** 2
        self.type_head = MLPHead(cfg.d_model, N_TYPES)
        self.patch_head = PointerHead(cfg.d_model, n_queries=1)
        self.sub_head = MLPHead(2 * cfg.d_model, self.n_sub)

    # -- masks ------------------------------------------------------------

    def _patch_mask(self, legal, BT: int) -> Optional[torch.Tensor]:
        if not legal:
            return None
        if "patch" in legal:
            return _bool(legal["patch"]).reshape(BT, self.n_patches)
        if "click" in legal:
            c = self.cfg
            click = _bool(legal["click"]).reshape(BT, c.arc_patches, c.arc_patch,
                                                  c.arc_patches, c.arc_patch)
            return click.any(dim=(2, 4)).reshape(BT, self.n_patches)
        return None

    def _sub_mask(self, legal, patch_idx: torch.Tensor, lbl_patch: Optional[torch.Tensor],
                  BT: int) -> Optional[torch.Tensor]:
        """Cells of the committed patch: the env's click table at it, else the
        compact ``sub_of`` (valid where the committed patch is the label)."""
        if not legal:
            return None
        c = self.cfg
        if "click" in legal:
            click = _bool(legal["click"]).reshape(BT, c.arc_patches, c.arc_patch,
                                                  c.arc_patches, c.arc_patch)
            click = click.permute(0, 1, 3, 2, 4).reshape(BT, self.n_patches, self.n_sub)
            return click[torch.arange(BT, device=patch_idx.device), patch_idx]
        if "sub_of" in legal:
            m = _bool(legal["sub_of"]).reshape(BT, self.n_sub)
            if lbl_patch is None:
                return None
            applies = (lbl_patch.reshape(BT) >= 0) & (patch_idx == lbl_patch.reshape(BT))
            return torch.where(applies.unsqueeze(-1), m, torch.ones_like(m))
        return None

    # -- decode -------------------------------------------------------------

    def decode(self, tokens, hl, ll, B, T, t0=0, legal=None, labels=None):
        BT = B * T
        P, Q = self.n_patches, self.n_sub
        c = self.cfg
        arange = torch.arange(BT, device=tokens.device)
        type_logits = self.type_head(hl)                                  # [BT, 8]
        patch_logits = self.patch_head(tokens, ll)                        # [BT, P]
        type_mask = _bool(legal.get("type")).reshape(BT, N_TYPES) \
            if legal and legal.get("type") is not None else None
        patch_mask = self._patch_mask(legal, BT)
        lbl_patch = None
        if labels and labels.get("click_idx") is not None:
            lbl = labels["click_idx"].reshape(BT).long()
            lbl_patch = torch.where(lbl >= 0, patch_of(lbl.clamp(min=0), c.arc_grid, c.arc_patch),
                                    torch.full_like(lbl, -1))
        if self.greedy and lbl_patch is None:
            safe = patch_logits.detach().float()
            if patch_mask is not None:
                safe = safe.masked_fill(~patch_mask, float("-inf"))
            patch_idx = safe.argmax(-1)
        else:
            patch_idx = _pick_index(patch_logits, lbl_patch, patch_mask, BT, P)
        patch_tok = tokens[arange, patch_idx]
        sub_logits = self.sub_head(torch.cat([patch_tok, ll], dim=-1))    # [BT, Q]
        sub_mask = self._sub_mask(legal, patch_idx, lbl_patch, BT)
        out = {"type_logits": type_logits.reshape(B, T, N_TYPES),
               "patch_logits": patch_logits.reshape(B, T, P),
               "sub_logits": sub_logits.reshape(B, T, Q),
               "patch_idx": patch_idx.reshape(B, T)}
        for key, m, k in (("type_mask", type_mask, N_TYPES), ("patch_mask", patch_mask, P),
                          ("sub_mask", sub_mask, Q)):
            if m is not None:
                out[key] = m.reshape(B, T, k)
        return out

    # -- rollout ---------------------------------------------------------------

    def sample(self, pred, legal=None):
        c = self.cfg
        with torch.no_grad():
            tm, pm, sm = pred.get("type_mask"), pred.get("patch_mask"), pred.get("sub_mask")
            tl = pred["type_logits"].float()
            if self.greedy:
                safe = tl if tm is None else tl.masked_fill(~tm, float("-inf"))
                type_idx = safe.argmax(-1)
                sl = pred["sub_logits"].float()
                sub = (sl if sm is None else sl.masked_fill(~sm, float("-inf"))).argmax(-1)
            else:
                type_idx = masked_sample(tl, tm)
                sub = masked_sample(pred["sub_logits"], sm)
            patch_idx = pred["patch_idx"]
            lp = masked_log_prob(tl, type_idx, tm)
            click_lp = (masked_log_prob(pred["patch_logits"].float(), patch_idx, pm)
                        + masked_log_prob(pred["sub_logits"].float(), sub, sm))
            is_click = type_idx == CLICK
            lp = lp + torch.where(is_click, click_lp, torch.zeros_like(click_lp))
            click = click_of(patch_idx, sub, c.arc_grid, c.arc_patch)
        out = {"action_type": type_idx, "action_click": click, "action_log_prob": lp}
        for key, m in (("legal_type", tm), ("legal_patch", pm), ("legal_sub", sm)):
            if m is not None:
                out[key] = m
        return out

    # -- replay ------------------------------------------------------------------

    def log_prob_entropy(self, pred, actions, legal=None):
        c = self.cfg
        tm, pm, sm = pred.get("type_mask"), pred.get("patch_mask"), pred.get("sub_mask")
        type_idx = actions["type_idx"].long()
        click = actions["click_idx"].long().clamp(0, c.arc_grid * c.arc_grid - 1)
        t_lp, t_ent = _log_prob_and_entropy(pred["type_logits"], type_idx, tm)
        p_lp, p_ent = _log_prob_and_entropy(
            pred["patch_logits"], patch_of(click, c.arc_grid, c.arc_patch), pm)
        s_lp, s_ent = _log_prob_and_entropy(
            pred["sub_logits"], sub_of(click, c.arc_grid, c.arc_patch), sm)
        is_click = type_idx == CLICK
        lp = t_lp + torch.where(is_click, p_lp + s_lp, torch.zeros_like(p_lp))
        # entropy of the hierarchical policy: H(type) + π(ACTION6) · H(click)
        tl = pred["type_logits"].float()
        if tm is not None:
            tl = tl.masked_fill(~tm, float("-inf"))
        p_click = torch.softmax(tl, -1)[..., CLICK].nan_to_num(0.0)
        ent = t_ent + p_click * (p_ent + s_ent)
        head = {"ent_type": t_ent.detach().mean(), "ent_patch": p_ent.detach().mean(),
                "ent_sub": s_ent.detach().mean()}
        return lp, ent, head

    # -- imitation ------------------------------------------------------------------

    def _il_parts(self, pred, tgt, seq_mask):
        c = self.cfg
        valid = seq_mask.float()
        n_valid = valid.sum().clamp(min=1.0)
        tgt_type = tgt["type_idx"].long().clamp(0, N_TYPES - 1)
        click = tgt["click_idx"].long().clamp(0, c.arc_grid * c.arc_grid - 1)
        tgt_patch = patch_of(click, c.arc_grid, c.arc_patch)
        tgt_sub = sub_of(click, c.arc_grid, c.arc_patch)
        terms, ok, illegal = {}, {}, {}
        terms["type"], ok["type"], illegal["type"] = _categorical_head(
            pred["type_logits"], tgt_type, pred.get("type_mask"), valid, n_valid)
        clicks = valid * (tgt_type == CLICK).float()
        n_clicks = clicks.sum().clamp(min=1.0)
        terms["patch"], ok["patch"], illegal["patch"] = _categorical_head(
            pred["patch_logits"], tgt_patch, pred.get("patch_mask"), clicks, n_clicks)
        cond = clicks * (pred["patch_idx"] == tgt_patch).float()
        n_cond = cond.sum().clamp(min=1.0)
        terms["sub"], ok["sub"], illegal["sub"] = _categorical_head(
            pred["sub_logits"], tgt_sub, pred.get("sub_mask"), cond, n_cond)
        return terms, ok, illegal, tgt_type

    def il_head_terms(self, pred, tgt, seq_mask):
        """Heads ``type`` (every frame), ``patch`` (ACTION6 frames), ``sub`` (ACTION6
        frames whose committed patch is the target's; teacher forcing makes it all).
        Soft targets: ``type_probs`` [8], ``patch_probs`` [P²] = p(patch | click),
        ``sub_probs`` [p²] = p(cell | the target patch)."""
        return self._il_parts(pred, tgt, seq_mask)[0]

    def il_loss(self, pred, tgt, seq_mask):
        terms, ok, illegal, tgt_type = self._il_parts(pred, tgt, seq_mask)
        head = {k: (t["ce"] * t["weight"]).sum() / t["norm"] for k, t in terms.items()}
        loss = head["type"] + head["patch"] + head["sub"]
        with torch.no_grad():
            def acc(k):
                return (ok[k].float() * terms[k]["weight"]).sum() / terms[k]["norm"]

            is_click = tgt_type == CLICK
            answer = ok["type"] & (~is_click | (ok["patch"] & ok["sub"]))
            out = {f"{k}_loss": v.detach() for k, v in head.items()}
            out.update({f"{k}_acc": acc(k) for k in terms})
            valid = seq_mask.bool()
            bad = [m for m in illegal.values() if m is not None]
            if bad:
                any_bad = torch.stack(bad).any(0) & valid
                out["illegal_target_frac"] = any_bad.float().sum() / valid.float().sum().clamp(min=1)
        return {"loss": loss, **out, "answer_correct": answer, "answer_mask": seq_mask.bool()}


def arc3_replay_legal(sample_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The compact legal dict that replays a rollout exactly: the masks
    :meth:`Arc3Decoder.sample` used, as ``type`` / ``patch`` / ``sub_of`` (replay
    teacher-forces the stored click, so ``sub_of`` applies at the stored patch)."""
    out = {}
    for k, v in (("legal_type", "type"), ("legal_patch", "patch"), ("legal_sub", "sub_of")):
        if k in sample_out:
            out[v] = sample_out[k]
    return out


def build_arc3_network(**overrides) -> nn.Module:
    from model.game_network import GameNetwork

    cfg = Arc3Config(**overrides)
    return GameNetwork(cfg, Arc3Encoder(cfg), Arc3Decoder(cfg))
