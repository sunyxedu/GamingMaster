"""StarCraft II adapter of the tier-3 network: gmsc2's frames and window decisions
(data_processing/rs/gmsc2/src/lib.rs module docs; ``gmdata sc2 convert`` shards, the
env/vec_sc2.py simulator).  The game is seen once per 8-loop window; each decision on a
window's frame is one raw unit command, or "done" (end the window).

obs (shard names without ``obs.``), [B, T, ...]:
    units          [U, F] uint8   0-2 alliance / display / cloak (categorical), 3 flags (bits),
                                  4.. quantised features (0..255 = 0..1)
    unit_xy        [U, 2] uint16  position / map size × 65535
    unit_type      [U] uint16     vocabulary ids (0 empty, 1 unknown)
    unit_order     [U] uint16     the first order's action id
    unit_buffs     [U, 2] uint16
    unit_mask      [U] bool       a unit in the slot
    unit_own       [U] bool       the player's unit (the possible actors)
    unit_commanded [U] {0, 1}     given a command earlier in this window
    map_bits       [9, 64, 64]    visibility, explored, creep, pathable, placeable, resource,
                                  height thermometer
    globals        [G] float32    economy, supply, time, races, map size, window commands
    upgrades       [NU] {0, 1}
    last_action    [] long        the window's previous command (0 none)
    window_start   [] bool        the window's first decision

Encoder: U unit tokens (embeddings of type, order, buffs, the categorical columns + a
projection of the flags, the continuous features, Fourier features of the position and the
commanded bit; a learned "empty" vector for an empty slot; a learned per-slot embedding —
slots are stable while a unit lives, so the per-slot recurrence follows one unit), the map
(the bit planes + own / enemy / neutral unit counts scattered from the positions → conv
stack → ``sc2_map_patch``² patches: 256 tokens at patch 4) and one global token.

Decoder (hierarchical; one policy for rollout, replay and IL):

    type    [A]      0 = done, else a general ability (index 1, unknown, never)
    kind    [3]      no target / a point / a unit — masked by the ability's target kinds
                     (the vocabulary's AbilityData.target)
    actors  [U]      an independent Bernoulli per own unit (the command's units)
    unit    [U]      the target unit: pointer over the unit tokens
    coarse  [P²]     the target point's map token, then
    fine    [L²/P²]  its cell inside the token: point = LOC × LOC over the map; the
                     add-on ``addon_fine`` adds per-cell scores from what the token blurs
                     (units at sub-token positions, the token's raw map pixels, the
                     offset from the committed actors)
    queued  Bernoulli

The query of the argument heads is the low-level token + the committed type's embedding;
the target heads add the pooled tokens of the committed actors.  ``decode`` commits the
type, the actors and the coarse cell (teacher-forced from ``labels`` action_type / actors /
target_point, else sampled), so

    log π(a) = log π(type) + [type ≥ 2] · (log π(kind) + Σ_own log π(actor_u) + log π(queued)
               + [kind = point] · (log π(coarse) + log π(fine | coarse))
               + [kind = unit] · log π(unit))

IL targets (tgt): action_type, actors [U], target_kind, target_unit, target_point, queued,
value.  The actor head's IL loss is the mean BCE over the frame's own units.
"""

from __future__ import annotations

import json
import math
import os
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
    bernoulli_log_prob_entropy,
    masked_log_prob,
    masked_sample,
)
from model.games.board import _categorical_head, _log_prob_and_entropy, _pick_index

# gmsc2's layout (data_processing/rs/gmsc2/src/lib.rs)
U = 320
F_UNIT = 27
N_CAT = 3
FLAGS_COL = 3
N_FLAGS = 6
CONT_COL = 4
MAP = 64
MAP_BITS = 9
G = 21
LOC = 256
BUFFS = 2
DONE = 0
UNKNOWN = 1
N_KINDS = 3
KIND_POINT = 1
KIND_UNIT = 2
N_FOURIER = 6
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_VOCAB = os.path.join(ROOT, "data_processing", "games", "sc2", "sc2_vocab_55958.json")
#: AbilityData.Target → (none, point, unit) accepted
TARGET_KINDS = {1: (True, False, False), 2: (False, True, False), 3: (False, False, True),
                4: (False, True, True), 5: (True, True, False)}


@dataclass
class Sc2Config(ModelConfig):
    """StarCraft II dimensions on top of the shared core config (the shard meta)."""

    sc2_units: int = U
    sc2_unit_features: int = F_UNIT
    sc2_map_bits: int = MAP_BITS
    sc2_globals: int = G
    sc2_loc: int = LOC
    sc2_unit_vocab: int = 948
    sc2_action_vocab: int = 1773
    sc2_buff_vocab: int = 156
    sc2_upgrades: int = 142
    sc2_map_patch: int = 4
    #: the vocabulary file (the ability target kinds); "" = the repo's 3.16.1 one
    sc2_vocab: str = ""
    encoder_channels: int = 64

    @property
    def sc2_map_tokens(self) -> int:
        return (MAP // self.sc2_map_patch) ** 2


def _bool(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if m is None else (m if m.dtype == torch.bool else m > 0.5)


def _embedding(n: int, d: int) -> nn.Embedding:
    e = nn.Embedding(n, d)
    nn.init.trunc_normal_(e.weight, std=0.02)      # the core's init skips embeddings
    return e


def target_kind_table(n_actions: int, vocab_path: str = "") -> torch.Tensor:
    """[A, 3] bool: the target kinds each action takes (done / unknown: none; an action
    outside the vocabulary file: any)."""
    tab = torch.ones(n_actions, N_KINDS, dtype=torch.bool)
    tab[DONE] = torch.tensor([True, False, False])
    tab[UNKNOWN] = torch.tensor([True, False, False])
    path = vocab_path or DEFAULT_VOCAB
    if os.path.exists(path):
        with open(path) as f:
            actions = json.load(f)["actions"]
        for i, (_, _, target) in enumerate(actions):
            if i + 2 < n_actions:
                tab[i + 2] = torch.tensor(TARGET_KINDS.get(int(target), (True, True, True)))
    return tab


def coarse_of(point: torch.Tensor, cfg: Sc2Config) -> torch.Tensor:
    """LOC-grid index y·LOC + x → the map token holding it."""
    n = MAP // cfg.sc2_map_patch
    cell = cfg.sc2_loc // n
    y, x = point // cfg.sc2_loc, point % cfg.sc2_loc
    return (y // cell) * n + x // cell


def fine_of(point: torch.Tensor, cfg: Sc2Config) -> torch.Tensor:
    n = MAP // cfg.sc2_map_patch
    cell = cfg.sc2_loc // n
    y, x = point // cfg.sc2_loc, point % cfg.sc2_loc
    return (y % cell) * cell + x % cell


def point_of(coarse: torch.Tensor, fine: torch.Tensor, cfg: Sc2Config) -> torch.Tensor:
    n = MAP // cfg.sc2_map_patch
    cell = cfg.sc2_loc // n
    y = (coarse // n) * cell + fine // cell
    x = (coarse % n) * cell + fine % cell
    return y * cfg.sc2_loc + x


class Sc2Encoder(ObservationEncoder):
    """Unit tokens ‖ map tokens ‖ the global token (see the module docstring)."""

    def __init__(self, cfg: Sc2Config):
        super().__init__()
        if cfg.sc2_unit_features != F_UNIT or cfg.sc2_map_bits != MAP_BITS or cfg.sc2_globals != G:
            raise ValueError("sc2 frame layout differs from gmsc2's (units / map bits / globals)")
        if MAP % cfg.sc2_map_patch or cfg.sc2_loc % (MAP // cfg.sc2_map_patch):
            raise ValueError(f"sc2_map_patch {cfg.sc2_map_patch} must divide {MAP} and LOC / tokens")
        self.cfg = cfg
        d = cfg.d_model
        self.n_units = cfg.sc2_units
        self.n_map = cfg.sc2_map_tokens
        self.n_tokens = self.n_units + self.n_map + 1
        self.type_embed = _embedding(cfg.sc2_unit_vocab, d)
        self.order_embed = _embedding(cfg.sc2_action_vocab, d)
        self.buff_embed = _embedding(cfg.sc2_buff_vocab, d)
        self.cat_embed = _embedding(N_CAT * 4, d)
        n_num = (F_UNIT - CONT_COL) + N_FLAGS + 1 + 2 + 2 * 2 * N_FOURIER
        self.num_proj = nn.Linear(n_num, d)
        self.unit_mlp = nn.Sequential(nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.empty = nn.Parameter(torch.randn(d) * 0.02)
        self.slot_embed = nn.Parameter(torch.randn(1, self.n_units, d) * 0.02)
        self.register_buffer("freqs", (2.0 ** torch.arange(N_FOURIER)) * math.pi, persistent=False)
        self.register_buffer("flag_bits", 2 ** torch.arange(N_FLAGS), persistent=False)
        c = cfg.encoder_channels
        self.conv = conv_stack(MAP_BITS + 3, c, c, encoder_norm_of(cfg))
        self.batch_stats = uses_batch_stats(cfg)
        self.patchify = nn.Conv2d(c, d, cfg.sc2_map_patch, stride=cfg.sc2_map_patch)
        n = MAP // cfg.sc2_map_patch
        self.row_embed = nn.Parameter(torch.randn(1, n, 1, d) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, n, d) * 0.02)
        self.global_proj = nn.Sequential(nn.Linear(G + cfg.sc2_upgrades, d), nn.ReLU(), nn.Linear(d, d))
        self.last_action_embed = _embedding(cfg.sc2_action_vocab, d)
        self.window_start_embed = _embedding(2, d)
        self.global_pos = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def decode_context(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """The own units (the possible actors), the occupied slots (the possible targets)
        and, for the fine head's add-on, the unit positions and the map bits."""
        return {"own": obs["unit_own"].bool(), "present": obs["unit_mask"].bool(),
                "unit_xy": obs["unit_xy"], "map_bits": obs["map_bits"]}

    def _units(self, obs: Dict[str, torch.Tensor], dtype) -> torch.Tensor:
        u = obs["units"]
        B, T, n, _ = u.shape
        cat = u[..., :N_CAT].long().clamp(0, 3) + torch.arange(N_CAT, device=u.device) * 4
        flags = ((u[..., FLAGS_COL].long().unsqueeze(-1) & self.flag_bits) > 0).to(dtype)
        cont = u[..., CONT_COL:].to(dtype) / 255.0
        xy = obs["unit_xy"].to(torch.float32) / 65535.0
        ang = xy.unsqueeze(-1) * self.freqs                                   # [B, T, U, 2, K]
        four = torch.cat([ang.sin(), ang.cos()], dim=-1).reshape(B, T, n, -1).to(dtype)
        commanded = obs["unit_commanded"].to(dtype).unsqueeze(-1)
        num = torch.cat([cont, flags, commanded, xy.to(dtype), four], dim=-1)
        x = (self.type_embed(obs["unit_type"].long()) + self.order_embed(obs["unit_order"].long())
             + self.buff_embed(obs["unit_buffs"].long()).sum(-2) + self.cat_embed(cat).sum(-2)
             + self.num_proj(num))
        x = self.unit_mlp(x)
        present = obs["unit_mask"].bool().unsqueeze(-1)
        return torch.where(present, x, self.empty.to(x.dtype).expand_as(x))

    def _density(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, T, 3, 64, 64]: own / enemy / neutral units per cell (/ 4, at most 1)."""
        xy = obs["unit_xy"].long()
        cx = (xy[..., 0] * MAP // 65536).clamp(0, MAP - 1)
        cy = (xy[..., 1] * MAP // 65536).clamp(0, MAP - 1)
        idx = cy * MAP + cx                                                   # [B, T, U]
        alliance = obs["units"][..., 0].long()
        present = obs["unit_mask"].bool()
        groups = torch.stack([(alliance <= 1), (alliance == 3), (alliance == 2)], dim=2) & present.unsqueeze(2)
        B, T = idx.shape[:2]
        out = torch.zeros(B, T, 3, MAP * MAP, device=idx.device, dtype=torch.float32)
        out.scatter_add_(-1, idx.unsqueeze(2).expand(-1, -1, 3, -1), groups.float())
        return (out / 4.0).clamp(max=1.0).reshape(B, T, 3, MAP, MAP)

    def forward(self, obs: Dict[str, torch.Tensor], frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.cfg
        dtype = self.num_proj.weight.dtype
        units = self._units(obs, dtype)                                        # [B, T, U, d]
        B, T = units.shape[:2]
        planes = torch.cat([obs["map_bits"].to(torch.float32), self._density(obs)], dim=2)
        planes = planes.reshape(B * T, MAP_BITS + 3, MAP, MAP).to(self.patchify.weight.dtype)
        feat = encode_frames(self.conv, planes, B, T, frame_mask, self.batch_stats)
        feat = self.patchify(feat.reshape(B * T, *feat.shape[2:]))           # [BT, d, n, n]
        d = cfg.d_model
        map_tokens = feat.reshape(B, T, d, self.n_map).permute(0, 1, 3, 2)
        glb = torch.cat([obs["globals"].to(dtype), obs["upgrades"].to(dtype)], dim=-1)
        g = (self.global_proj(glb) + self.last_action_embed(obs["last_action"].long().clamp(min=0))
             + self.window_start_embed(obs["window_start"].long()))
        return torch.cat([units, map_tokens.to(units.dtype), g.unsqueeze(2).to(units.dtype)], dim=2)

    def positional_embedding(self) -> torch.Tensor:
        n, d = MAP // self.cfg.sc2_map_patch, self.cfg.d_model
        mp = (self.row_embed + self.col_embed).reshape(1, n * n, d)
        return torch.cat([self.slot_embed, mp, self.global_pos], dim=1)


#: offsets from the actors' centroid: in fine cells / FINE_OFFSET_SCALE, clipped to
#: [-1, 1], + sin / cos at N_FINE_FREQS frequencies (a one-cell step stays visible)
FINE_OFFSET_SCALE = 8.0
N_FINE_FREQS = 4
#: per fine cell: own / other units and committed actors in it, the 9 map bits of its
#: pixel, its offset from the actors' centroid (x, y + their Fourier features) and
#: "has actors"
N_FINE_FEATS = 3 + MAP_BITS + 2 + 4 * N_FINE_FREQS + 1


def fine_cell_features(ctx: Dict[str, torch.Tensor], coarse_idx: torch.Tensor, actors: torch.Tensor,
                       cfg: Sc2Config) -> torch.Tensor:
    """[BT, L²/P², N_FINE_FEATS] features of every fine cell of the committed coarse
    token: what lies inside it at sub-token resolution, which the map token (a patch
    embedding) blurs — unit positions at 1/4 map pixel and the token's raw map pixels."""
    n = MAP // cfg.sc2_map_patch
    cell = cfg.sc2_loc // n                                  # fine cells per token side
    BT = coarse_idx.shape[0]
    n_units = cfg.sc2_units
    dev = coarse_idx.device
    xy = ctx["unit_xy"].reshape(BT, n_units, 2).float() / 65535.0 * cfg.sc2_loc   # LOC grid
    oy = (coarse_idx // n * cell).float().unsqueeze(1)
    ox = (coarse_idx % n * cell).float().unsqueeze(1)
    rx, ry = xy[..., 0] - ox, xy[..., 1] - oy
    inside = (rx >= 0) & (rx < cell) & (ry >= 0) & (ry < cell)
    present = ctx["present"].reshape(BT, n_units)
    own = ctx["own"].reshape(BT, n_units)
    idx = (ry.clamp(0, cell - 1).long() * cell + rx.clamp(0, cell - 1).long())       # [BT, U]
    groups = torch.stack([own & present, ~own & present, actors & present], dim=-1) & inside.unsqueeze(-1)
    counts = torch.zeros(BT, cell * cell, 3, device=dev, dtype=torch.float32)
    counts.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, 3), groups.float())
    counts = (counts / 2.0).clamp(max=1.0)
    # the map pixel under each fine cell
    f = torch.arange(cell * cell, device=dev)
    per_px = cell // cfg.sc2_map_patch                       # fine cells per map pixel side
    py = (coarse_idx // n * cfg.sc2_map_patch).unsqueeze(1) + (f // cell // per_px).unsqueeze(0)
    px = (coarse_idx % n * cfg.sc2_map_patch).unsqueeze(1) + (f % cell // per_px).unsqueeze(0)
    bits = ctx["map_bits"].reshape(BT, MAP_BITS, MAP * MAP).float()
    pix = (py * MAP + px).unsqueeze(1).expand(-1, MAP_BITS, -1)
    mbits = bits.gather(2, pix).transpose(1, 2)              # [BT, cells, 9]
    # offset of each fine cell's centre from the committed actors' centroid
    w = (actors & present).float()
    has = w.sum(-1, keepdim=True) > 0
    cen = (xy * w.unsqueeze(-1)).sum(1) / w.sum(-1, keepdim=True).clamp(min=1.0)       # [BT, 2]
    cx = ox + (f % cell).float().unsqueeze(0) + 0.5
    cy = oy + (f // cell).float().unsqueeze(0) + 0.5
    off = (torch.stack([cx - cen[:, :1], cy - cen[:, 1:]], dim=-1) / FINE_OFFSET_SCALE).clamp(-1.0, 1.0)
    freqs = (2.0 ** torch.arange(N_FINE_FREQS, device=dev)) * (math.pi / 2)
    ang = (off.unsqueeze(-1) * freqs).flatten(-2)                            # [BT, cells, 2K]
    geo = torch.cat([off, ang.sin(), ang.cos()], dim=-1) * has.unsqueeze(-1)
    return torch.cat([counts, mbits, geo, has.float().unsqueeze(1).expand(-1, cell * cell, 1)], dim=-1)


class FineContext(nn.Module):
    """Add-on of the fine head (``addon_fine``): a per-fine-cell score from
    :func:`fine_cell_features`, a learned cell embedding and the argument query,
    ADDED to the fine logits.  Its output layer starts at zero, so a network trained
    without it keeps its function when the add-on joins (model.multi_game.ADDON_PREFIX)."""

    def __init__(self, d: int, n_fine: int, hidden: int = 64):
        super().__init__()
        self.cell_proj = nn.Linear(N_FINE_FEATS, hidden)
        self.pos = nn.Parameter(torch.randn(n_fine, hidden) * 0.02)
        self.query_proj = nn.Linear(d, hidden)
        self.out = nn.Linear(hidden, 1)
        self.reset_addon()

    def reset_addon(self) -> None:
        """Zero output (also after GameNetwork's network-wide re-init)."""
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        h = self.cell_proj(feats.to(self.cell_proj.weight.dtype)) + self.pos.to(self.cell_proj.weight.dtype)
        h = F.gelu(h + self.query_proj(q.to(self.query_proj.weight.dtype)).unsqueeze(1))
        return self.out(h).squeeze(-1)


class Sc2Decoder(ActionDecoder):
    """type → kind, actors, queued, target unit / point (see the module docstring)."""

    def __init__(self, cfg: Sc2Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        A = cfg.sc2_action_vocab
        self.n_map = cfg.sc2_map_tokens
        n = MAP // cfg.sc2_map_patch
        self.n_fine = (cfg.sc2_loc // n) ** 2
        self.type_head = MLPHead(d, A)
        self.type_embed = _embedding(A, d)
        self.kind_head = MLPHead(2 * d, N_KINDS)
        self.actor_head = MLPHead(2 * d, 1)
        self.actor_pool = nn.Linear(d, d)
        self.unit_head = PointerHead(d, n_queries=1)
        self.coarse_head = PointerHead(d, n_queries=1)
        self.fine_head = MLPHead(2 * d, self.n_fine)
        self.addon_fine = FineContext(d, self.n_fine)
        self.queued_head = MLPHead(2 * d, 1)
        type_mask = torch.ones(A, dtype=torch.bool)
        type_mask[UNKNOWN] = False
        self.register_buffer("type_mask", type_mask, persistent=False)
        self.register_buffer("kinds", target_kind_table(A, cfg.sc2_vocab), persistent=False)

    # -- decode -------------------------------------------------------------

    def decode(self, tokens, hl, ll, B, T, t0=0, legal=None, labels=None, ctx=None):
        if ctx is None:
            raise ValueError("Sc2Decoder needs the encoder's decode context (own / present units)")
        cfg = self.cfg
        BT = B * T
        n_units = cfg.sc2_units
        unit_tok = tokens[:, :n_units]
        map_tok = tokens[:, n_units:n_units + self.n_map]
        arange = torch.arange(BT, device=tokens.device)
        own = ctx["own"].reshape(BT, n_units)
        present = ctx["present"].reshape(BT, n_units)
        A = cfg.sc2_action_vocab
        type_logits = self.type_head(hl)                                       # [BT, A]
        tmask = self.type_mask.expand(BT, A)
        lbl_type = labels.get("action_type") if labels else None
        if self.greedy and lbl_type is None:
            type_idx = type_logits.detach().float().masked_fill(~tmask, float("-inf")).argmax(-1)
        else:
            type_idx = _pick_index(type_logits, lbl_type, tmask, BT, A)
        temb = self.type_embed(type_idx).to(ll.dtype)
        q = ll + temb
        arg_in = torch.cat([ll, temb], dim=-1)
        kind_logits = self.kind_head(arg_in)                                   # [BT, 3]
        kind_mask = self.kinds[type_idx]
        actor_logits = self.actor_head(torch.cat([unit_tok, q.unsqueeze(1).expand(-1, n_units, -1)], dim=-1)).squeeze(-1)
        # the committed actors: labelled frames keep theirs, the others sample (own units)
        with torch.no_grad():
            if self.greedy:
                sampled = actor_logits.detach() > 0
            else:
                sampled = torch.bernoulli(torch.sigmoid(actor_logits.detach().float())).bool()
            actors = sampled & own
            lbl_actors = labels.get("actors") if labels else None
            if lbl_actors is not None:
                la = lbl_actors.reshape(BT, n_units).float()
                given = (la >= 0).all(-1, keepdim=True)
                actors = torch.where(given, (la > 0.5) & own, actors)
        w = actors.to(unit_tok.dtype).unsqueeze(-1)
        pooled = (unit_tok * w).sum(1) / w.sum(1).clamp(min=1.0)
        q2 = q + self.actor_pool(pooled)
        unit_logits = self.unit_head(unit_tok, q2)                             # [BT, U]
        coarse_logits = self.coarse_head(map_tok, q2)                          # [BT, M]
        lbl_point = labels.get("target_point") if labels else None
        lbl_coarse = None
        if lbl_point is not None:
            lp_ = lbl_point.reshape(BT).long()
            lbl_coarse = torch.where(lp_ >= 0, coarse_of(lp_.clamp(min=0), cfg), torch.full_like(lp_, -1))
        if self.greedy and lbl_coarse is None:
            coarse_idx = coarse_logits.detach().float().argmax(-1)
        else:
            coarse_idx = _pick_index(coarse_logits, lbl_coarse, None, BT, self.n_map)
        fine_logits = self.fine_head(torch.cat([map_tok[arange, coarse_idx], q2], dim=-1))
        feats = fine_cell_features(ctx, coarse_idx, actors, cfg)
        fine_logits = fine_logits + self.addon_fine(feats, q2).to(fine_logits.dtype)
        queued_logits = self.queued_head(arg_in).squeeze(-1)
        r = lambda x, *s: x.reshape(B, T, *s)
        return {"type_logits": r(type_logits, A), "type_mask": r(tmask, A), "type_idx": r(type_idx),
                "kind_logits": r(kind_logits, N_KINDS), "kind_mask": r(kind_mask, N_KINDS),
                "actor_logits": r(actor_logits, n_units), "actors_used": r(actors, n_units), "own": r(own, n_units),
                "unit_logits": r(unit_logits, n_units), "unit_mask": r(present, n_units),
                "coarse_logits": r(coarse_logits, self.n_map), "coarse_idx": r(coarse_idx),
                "fine_logits": r(fine_logits, self.n_fine), "queued_logits": r(queued_logits)}

    # -- log-prob pieces ------------------------------------------------------------

    def _arg_lp(self, pred, kind, actors, queued, unit, point):
        """(log π of the arguments, their entropy) at the committed type; [...]"""
        cfg = self.cfg
        k_lp, k_ent = _log_prob_and_entropy(pred["kind_logits"], kind, pred["kind_mask"])
        own = pred["own"]
        a_lp, a_ent = bernoulli_log_prob_entropy(pred["actor_logits"], actors & own)
        ownf = own.float()
        a_lp = (a_lp * ownf).sum(-1)
        a_ent = (a_ent * ownf).sum(-1) / ownf.sum(-1).clamp(min=1.0)
        q_lp, q_ent = bernoulli_log_prob_entropy(pred["queued_logits"], queued)
        u_lp, u_ent = _log_prob_and_entropy(pred["unit_logits"], unit.clamp(0, cfg.sc2_units - 1), pred["unit_mask"])
        pt = point.clamp(0, cfg.sc2_loc * cfg.sc2_loc - 1)
        c_lp, c_ent = _log_prob_and_entropy(pred["coarse_logits"], coarse_of(pt, cfg), None)
        f_lp, f_ent = _log_prob_and_entropy(pred["fine_logits"], fine_of(pt, cfg), None)
        is_pt, is_unit = kind == KIND_POINT, kind == KIND_UNIT
        z = torch.zeros_like(k_lp)
        lp = k_lp + a_lp + q_lp + torch.where(is_pt, c_lp + f_lp, z) + torch.where(is_unit, u_lp, z)
        ent = k_ent + a_ent + q_ent + torch.where(is_pt, c_ent + f_ent, z) + torch.where(is_unit, u_ent, z)
        return lp, ent, {"ent_kind": k_ent, "ent_actor": a_ent, "ent_queued": q_ent, "ent_unit": u_ent,
                         "ent_coarse": c_ent, "ent_fine": f_ent}

    # -- rollout ---------------------------------------------------------------

    def sample(self, pred, legal=None):
        cfg = self.cfg
        with torch.no_grad():
            type_idx = pred["type_idx"]
            lp = masked_log_prob(pred["type_logits"].float(), type_idx, pred["type_mask"])
            if self.greedy:
                kind = pred["kind_logits"].float().masked_fill(~pred["kind_mask"], float("-inf")).argmax(-1)
                queued = pred["queued_logits"] > 0
                unit = pred["unit_logits"].float().masked_fill(~pred["unit_mask"], float("-inf")).argmax(-1)
                fine = pred["fine_logits"].float().argmax(-1)
            else:
                kind = masked_sample(pred["kind_logits"], pred["kind_mask"])
                queued = torch.bernoulli(torch.sigmoid(pred["queued_logits"].float())).bool()
                unit = masked_sample(pred["unit_logits"], pred["unit_mask"])
                fine = masked_sample(pred["fine_logits"], None)
            point = point_of(pred["coarse_idx"], fine, cfg)
            actors = pred["actors_used"]
            a_lp, _, _ = self._arg_lp(pred, kind, actors, queued, unit, point)
            cmd = type_idx >= 2
            lp = lp + torch.where(cmd, a_lp, torch.zeros_like(a_lp))
            no = torch.full_like(type_idx, -1)
            out = {"action_type": type_idx, "action_actors": actors & cmd.unsqueeze(-1),
                   "action_target_kind": torch.where(cmd, kind, torch.zeros_like(kind)),
                   "action_target_unit": torch.where(cmd & (kind == KIND_UNIT), unit, no),
                   "action_target_point": torch.where(cmd & (kind == KIND_POINT), point, no),
                   "action_queued": queued & cmd, "action_log_prob": lp}
        return out

    # -- replay ------------------------------------------------------------------

    def log_prob_entropy(self, pred, actions, legal=None):
        type_idx = actions["action_type"].long()
        t_lp, t_ent = _log_prob_and_entropy(pred["type_logits"], type_idx.clamp(min=0), pred["type_mask"])
        kind = actions["target_kind"].long().clamp(0, N_KINDS - 1)
        a_lp, a_ent, heads = self._arg_lp(pred, kind, _bool(actions["actors"]), _bool(actions["queued"]),
                                          actions["target_unit"].long(), actions["target_point"].long())
        cmd = type_idx >= 2
        z = torch.zeros_like(t_lp)
        lp = t_lp + torch.where(cmd, a_lp, z)
        ent = t_ent + torch.where(cmd, a_ent, z)
        head = {"ent_type": t_ent.detach().mean(), **{k: v.detach().mean() for k, v in heads.items()}}
        return lp, ent, head

    # -- imitation ------------------------------------------------------------------

    def _il_parts(self, pred, tgt, seq_mask):
        cfg = self.cfg
        valid = seq_mask.float()
        n_valid = valid.sum().clamp(min=1.0)
        A = cfg.sc2_action_vocab
        t_type = tgt["action_type"].long().clamp(0, A - 1)
        terms, ok, illegal = {}, {}, {}
        terms["type"], ok["type"], illegal["type"] = _categorical_head(
            pred["type_logits"], t_type, pred["type_mask"], valid, n_valid)
        cmd = valid * (t_type >= 2).float() * (pred["type_idx"] == t_type).float()
        n_cmd = cmd.sum().clamp(min=1.0)
        t_kind = tgt["target_kind"].long().clamp(0, N_KINDS - 1)
        terms["kind"], ok["kind"], illegal["kind"] = _categorical_head(
            pred["kind_logits"], t_kind, pred["kind_mask"], cmd, n_cmd)
        own = pred["own"].float()
        t_actors = _bool(tgt["actors"])
        a_ce = F.binary_cross_entropy_with_logits(pred["actor_logits"].float(), t_actors.float(),
                                                  reduction="none").nan_to_num(0.0)
        a_w = (cmd.unsqueeze(-1) * own / own.sum(-1, keepdim=True).clamp(min=1.0)).detach()
        terms["actors"] = {"kind": "bernoulli", "logits": pred["actor_logits"], "ce": a_ce, "weight": a_w,
                           "norm": a_w.sum().clamp(min=1.0)}
        with torch.no_grad():
            a_ok = ((pred["actor_logits"] > 0) == t_actors) | (own < 0.5)
            ok["actors"] = a_ok.all(-1)
        t_q = _bool(tgt["queued"])
        q_ce = F.binary_cross_entropy_with_logits(pred["queued_logits"].float(), t_q.float(),
                                                  reduction="none").nan_to_num(0.0)
        terms["queued"] = {"kind": "bernoulli", "logits": pred["queued_logits"], "ce": q_ce, "weight": cmd,
                           "norm": n_cmd}
        ok["queued"] = (pred["queued_logits"] > 0) == t_q
        t_unit = tgt["target_unit"].long()
        u_w = cmd * (t_kind == KIND_UNIT).float() * (t_unit >= 0).float()
        terms["unit"], ok["unit"], illegal["unit"] = _categorical_head(
            pred["unit_logits"], t_unit.clamp(0, cfg.sc2_units - 1), pred["unit_mask"], u_w, u_w.sum().clamp(min=1.0))
        t_point = tgt["target_point"].long()
        p_w = cmd * (t_kind == KIND_POINT).float() * (t_point >= 0).float()
        pt = t_point.clamp(0, cfg.sc2_loc * cfg.sc2_loc - 1)
        t_coarse = coarse_of(pt, cfg)
        terms["coarse"], ok["coarse"], illegal["coarse"] = _categorical_head(
            pred["coarse_logits"], t_coarse, None, p_w, p_w.sum().clamp(min=1.0))
        f_w = p_w * (pred["coarse_idx"] == t_coarse).float()
        terms["fine"], ok["fine"], illegal["fine"] = _categorical_head(
            pred["fine_logits"], fine_of(pt, cfg), None, f_w, f_w.sum().clamp(min=1.0))
        return terms, ok, illegal, t_type, t_kind

    def il_head_terms(self, pred, tgt, seq_mask):
        """Heads ``type`` (every frame); ``kind`` / ``actors`` / ``queued`` (command
        frames); ``unit`` (unit-target commands), ``coarse`` / ``fine`` (point-target
        commands; ``fine`` where the committed coarse cell is the target's)."""
        return self._il_parts(pred, tgt, seq_mask)[0]

    def il_loss(self, pred, tgt, seq_mask):
        terms, ok, illegal, t_type, t_kind = self._il_parts(pred, tgt, seq_mask)
        head = {k: (t["ce"] * t["weight"]).sum() / t["norm"] for k, t in terms.items()}
        loss = sum(head.values())
        with torch.no_grad():
            def acc(k):
                w = terms[k]["weight"]
                if w.dim() > ok[k].dim():
                    w = (w > 0).any(-1).float()
                return (ok[k].float() * w).sum() / w.sum().clamp(min=1.0)

            out = {f"{k}_loss": v.detach() for k, v in head.items()}
            out.update({f"{k}_acc": acc(k) for k in terms})
            cmd = t_type >= 2
            answer = ok["type"] & (~cmd | (ok["kind"] & ok["actors"]
                                           & ((t_kind != KIND_UNIT) | ok["unit"])
                                           & ((t_kind != KIND_POINT) | (ok["coarse"] & ok["fine"]))))
            valid = seq_mask.bool()
            out["command_frac"] = (cmd & valid).float().sum() / valid.float().sum().clamp(min=1.0)
            bad = [m for m in illegal.values() if m is not None]
            if bad:
                any_bad = torch.stack(bad).any(0) & valid
                out["illegal_target_frac"] = any_bad.float().sum() / valid.float().sum().clamp(min=1)
        return {"loss": loss, **out, "answer_correct": answer, "answer_mask": seq_mask.bool()}


def build_sc2_network(**overrides) -> nn.Module:
    from model.game_network import GameNetwork

    cfg = Sc2Config(**overrides)
    return GameNetwork(cfg, Sc2Encoder(cfg), Sc2Decoder(cfg))
