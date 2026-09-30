"""Honor of Kings (王者荣耀) 1v1 adapter of the tier-3 network (env/vec_hok.py is its
environment: Tencent's Honor of Kings AI Open Research Environment, hok_env ``hok1v1``).

Frame: ``features`` [B, T, 705] float32 — the research environment's per-player feature
vector, in the order its reference model splits it (hok_env aiarena/1v1/common/config.py
``DimConfig``): own hero 235 | enemy hero 235 | main-hero 14 | own soldiers 4 × 18 | enemy
soldiers 4 × 18 | own towers 2 × 18 | enemy towers 2 × 18 | global 5.  (The gamecore
20260420 emits 705; the Hokoff datasets' 725 carry 20 more global columns that are constant
over them — 1 then zeros — and are dropped by the converter.)

Encoder: one token per entity — the two heroes, the main-hero vector, 8 soldiers, 4 towers
and the global vector (16 tokens).  A per-kind MLP (shared by the own and the enemy
entities of a kind) projects each slice to d_model; the learned slot embedding also says
own / enemy.

Decoder (the environment's factored action; one policy for rollout, replay and IL):

    button     [12]  None1, None2, Move, Attack, Skill1, Skill2, Skill3, HealSkill,
                     ChosenSkill, Recall, Skill4, EquipSkill — from the high-level token
    move_x/z   [16]  movement direction bins
    skill_x/z  [16]  skill direction bins
    target     [8]   none, enemy hero, own hero, enemy soldiers 1-4, enemy tower: a pointer
                     over those entity tokens (the reference model's target order; the
                     tower slot pools the two enemy towers, "none" is a learned vector)

The argument heads read the low-level token and the committed button's embedding.
Which heads a button uses is the frame's ``sub`` table (the environment's
``sub_action_mask``; columns = button, move_x, move_z, skill_x, skill_z, target):

    log π(a) = log π(button) + Σ_{h ≥ 1} sub[button, h] · log π(h | button)

``decode`` commits the button (teacher-forced from ``labels["button_idx"]``, else sampled)
so the argument heads condition on it; the replay teacher-forces the stored button.

legal keys (bool, or {0, 1} floats from shards):
    button [12], move_x [16], move_z [16], skill_x [16], skill_z [16]
    target     [12, 8]  legal targets per button (the environment's table)
    target_of  [8]      compact: the committed (labelled) button's row (IL shards, replay)
    sub        [12, 6]  the heads each button uses
    sub_of     [6]      compact: the committed button's row
labels:  button_idx [B, T] long (−1 = sample)
actions: button_idx, move_x_idx, move_z_idx, skill_x_idx, skill_z_idx, target_idx [B, T]
IL targets (tgt): the six indices (−1 where the button does not use the head) + the
``legal_*`` masks as ``--il_legal_mask`` inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

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

#: (name, start, per-entity width, count) of the 705-d feature vector, in order
SEGMENTS = (
    ("hero", 0, 235, 1), ("hero", 235, 235, 1), ("main", 470, 14, 1),
    ("soldier", 484, 18, 4), ("soldier", 556, 18, 4),
    ("organ", 628, 18, 2), ("organ", 664, 18, 2), ("global", 700, 5, 1),
)
N_FEATURES = 705
N_TOKENS = 16
BUTTONS = 12
DIR_BINS = 16
N_TARGETS = 8
#: argument heads (column h of the sub table; column 0 is the button itself)
ARG_HEADS = ("move_x", "move_z", "skill_x", "skill_z", "target")
ARG_SIZES = (DIR_BINS, DIR_BINS, DIR_BINS, DIR_BINS, N_TARGETS)
ACTION_KEYS = ("button_idx",) + tuple(f"{h}_idx" for h in ARG_HEADS)
#: token of each target slot (None = the learned "none" vector, a tuple = mean of tokens)
_OWN_HERO, _ENEMY_HERO = 0, 1
_ENEMY_SOLDIERS = (7, 8, 9, 10)
_ENEMY_TOWERS = (13, 14)
#: Fog of war.  The gamecore's per-player feature vector keeps enemy-hero columns live while
#: the enemy is out of sight (docs problems log 119: counterfactual gamecore runs, a labelled
#: self-play run and the Hokoff shards each showed a different set, 30+ columns in all).
#: Column 251 is an exact visibility flag (1e-4 ⇔ hidden; 40 % of the Hokoff frames); while
#: it says hidden, the whole enemy-hero slice but the flag is zeroed — in the encoder, so IL
#: shards, replay and the RL environment are masked alike.
HIDDEN_FLAG_DIM = 251
HIDDEN_FLAG_VALUE = 1e-4
ENEMY_HERO = slice(235, 470)


def mask_hidden_enemy(x: torch.Tensor) -> torch.Tensor:
    """``x`` [..., 705] with the enemy hero's columns (but the flag) zeroed where it is hidden."""
    hidden = (x[..., HIDDEN_FLAG_DIM] - HIDDEN_FLAG_VALUE).abs() < 1e-6
    if not bool(hidden.any()):
        return x
    keep = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    keep[ENEMY_HERO] = 0
    keep[HIDDEN_FLAG_DIM] = 1
    return torch.where(hidden.unsqueeze(-1), x * keep, x)


@dataclass
class HokConfig(ModelConfig):
    """Honor of Kings 1v1 dimensions on top of the shared core config."""

    hok_features: int = N_FEATURES


def _bool(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if m is None else (m if m.dtype == torch.bool else m > 0.5)


class HokEncoder(ObservationEncoder):
    """Entity slices of the feature vector → 16 tokens (see the module docstring)."""

    def __init__(self, cfg: HokConfig):
        super().__init__()
        if cfg.hok_features != N_FEATURES:
            raise ValueError(f"hok_features must be {N_FEATURES} (the hok1v1 feature vector)")
        self.cfg = cfg
        self.n_tokens = N_TOKENS
        d = cfg.d_model
        widths = {"hero": 235, "main": 14, "soldier": 18, "organ": 18, "global": 5}
        self.proj = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(w, d), nn.ReLU(), nn.Linear(d, d)) for k, w in widths.items()})
        self.slot_embed = nn.Parameter(torch.randn(1, N_TOKENS, d) * 0.02)

    def forward(self, obs: Dict[str, torch.Tensor],
                frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = obs["features"]
        B, T, F = x.shape
        if F != N_FEATURES:
            raise ValueError(f"hok features have {F} columns, expected {N_FEATURES}")
        x = mask_hidden_enemy(x.float()).to(self.slot_embed.dtype)
        tokens = []
        for kind, start, width, count in SEGMENTS:
            seg = x[..., start:start + width * count].reshape(B, T, count, width)
            tokens.append(self.proj[kind](seg))
        return torch.cat(tokens, dim=2)

    def positional_embedding(self) -> torch.Tensor:
        return self.slot_embed


class HokDecoder(ActionDecoder):
    """button → its argument heads (see the module docstring)."""

    def __init__(self, cfg: HokConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.button_head = MLPHead(d, BUTTONS)
        self.button_embed = nn.Embedding(BUTTONS, d)
        self.dir_heads = nn.ModuleDict({h: MLPHead(2 * d, n) for h, n in zip(ARG_HEADS[:4], ARG_SIZES[:4])})
        self.none_target = nn.Parameter(torch.randn(d) * 0.02)
        self.target_head = PointerHead(d, n_queries=1)

    # -- masks ------------------------------------------------------------------

    @staticmethod
    def _row(legal, full: str, compact: str, button: torch.Tensor, BT: int,
             width: int) -> Optional[torch.Tensor]:
        """The committed button's row of a per-button table: the full [12, width] table
        at ``button``, else the compact row (valid for the labelled button)."""
        if not legal:
            return None
        if legal.get(full) is not None:
            tab = _bool(legal[full]).reshape(BT, BUTTONS, width)
            return tab[torch.arange(BT, device=button.device), button]
        if legal.get(compact) is not None:
            return _bool(legal[compact]).reshape(BT, width)
        return None

    def _targets(self, tokens: torch.Tensor) -> torch.Tensor:
        """[BT, 8, d] the target slots' vectors (the module docstring's order)."""
        BT, _, d = tokens.shape
        slots = [self.none_target.to(tokens.dtype).expand(BT, d), tokens[:, _ENEMY_HERO],
                 tokens[:, _OWN_HERO]]
        slots += [tokens[:, i] for i in _ENEMY_SOLDIERS]
        slots.append(tokens[:, list(_ENEMY_TOWERS)].mean(1))
        return torch.stack(slots, dim=1)

    # -- decode ------------------------------------------------------------------

    def decode(self, tokens, hl, ll, B, T, t0=0, legal=None, labels=None):
        BT = B * T
        button_logits = self.button_head(hl)                                   # [BT, 12]
        bm = _bool(legal.get("button")).reshape(BT, BUTTONS) \
            if legal and legal.get("button") is not None else None
        label = labels.get("button_idx") if labels else None
        if self.greedy and label is None:
            safe = button_logits.detach().float()
            if bm is not None:
                safe = safe.masked_fill(~bm, float("-inf"))
            button = safe.argmax(-1)
        else:
            button = _pick_index(button_logits, label, bm, BT, BUTTONS)
        emb = self.button_embed(button).to(ll.dtype)
        query = ll + emb
        arg_in = torch.cat([ll, emb], dim=-1)
        out = {"button_logits": button_logits.reshape(B, T, BUTTONS),
               "button_idx": button.reshape(B, T)}
        if bm is not None:
            out["button_mask"] = bm.reshape(B, T, BUTTONS)
        for h, n in zip(ARG_HEADS[:4], ARG_SIZES[:4]):
            out[f"{h}_logits"] = self.dir_heads[h](arg_in).reshape(B, T, n)
            if legal and legal.get(h) is not None:
                out[f"{h}_mask"] = _bool(legal[h]).reshape(B, T, n)
        target_logits = self.target_head(self._targets(tokens), query)       # [BT, 8]
        out["target_logits"] = target_logits.reshape(B, T, N_TARGETS)
        tm = self._row(legal, "target", "target_of", button, BT, N_TARGETS)
        if tm is not None:
            out["target_mask"] = tm.reshape(B, T, N_TARGETS)
        sub = self._row(legal, "sub", "sub_of", button, BT, 1 + len(ARG_HEADS))
        if sub is not None:
            out["sub_mask"] = sub.reshape(B, T, 1 + len(ARG_HEADS))
        return out

    def _uses(self, pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """[..., 5] bool: the argument heads the committed button uses (all of them
        when the frame carries no sub table)."""
        sub = pred.get("sub_mask")
        if sub is None:
            lead = pred["button_idx"].shape
            return torch.ones(*lead, len(ARG_HEADS), dtype=torch.bool,
                              device=pred["button_idx"].device)
        return sub[..., 1:]

    # -- rollout ------------------------------------------------------------------

    def sample(self, pred, legal=None):
        with torch.no_grad():
            button = pred["button_idx"]
            lp = masked_log_prob(pred["button_logits"].float(), button, pred.get("button_mask"))
            uses = self._uses(pred)
            out = {"action_button": button}
            for j, h in enumerate(ARG_HEADS):
                logits = pred[f"{h}_logits"].float()
                m = pred.get(f"{h}_mask")
                if self.greedy:
                    idx = (logits if m is None else logits.masked_fill(~m, float("-inf"))).argmax(-1)
                else:
                    idx = masked_sample(logits, m)
                h_lp = masked_log_prob(logits, idx, m)
                lp = lp + torch.where(uses[..., j], h_lp, torch.zeros_like(h_lp))
                out[f"action_{h}"] = idx
            out["action_log_prob"] = lp
            for key in ("button_mask", "move_x_mask", "move_z_mask", "skill_x_mask",
                        "skill_z_mask", "target_mask", "sub_mask"):
                if key in pred:
                    out[f"legal_{key[:-5]}"] = pred[key]
        return out

    # -- replay ------------------------------------------------------------------

    def log_prob_entropy(self, pred, actions, legal=None):
        button = actions["button_idx"].long().clamp(0, BUTTONS - 1)
        lp, ent = _log_prob_and_entropy(pred["button_logits"], button, pred.get("button_mask"))
        head = {"ent_button": ent.detach().mean()}
        uses = self._uses(pred)
        # the argument heads condition on the committed button (= the stored one in the
        # replay): their entropy at it is a one-sample estimate of E_b[H(args | b)]
        for j, (h, n) in enumerate(zip(ARG_HEADS, ARG_SIZES)):
            idx = actions[f"{h}_idx"].long().clamp(0, n - 1)
            h_lp, h_ent = _log_prob_and_entropy(pred[f"{h}_logits"], idx, pred.get(f"{h}_mask"))
            use = uses[..., j]
            lp = lp + torch.where(use, h_lp, torch.zeros_like(h_lp))
            ent = ent + torch.where(use, h_ent, torch.zeros_like(h_ent))
            head[f"ent_{h}"] = h_ent.detach().mean()
        return lp, ent, head

    # -- imitation ------------------------------------------------------------------

    def _il_parts(self, pred, tgt, seq_mask):
        valid = seq_mask.float()
        terms, ok, illegal = {}, {}, {}
        t_button = tgt["button_idx"].long()
        has_button = valid * (t_button >= 0).float()
        terms["button"], ok["button"], illegal["button"] = _categorical_head(
            pred["button_logits"], t_button.clamp(0, BUTTONS - 1), pred.get("button_mask"),
            has_button, has_button.sum().clamp(min=1.0))
        # the argument heads condition on the committed button: supervised where it is
        # the labelled one (teacher forcing makes it all) and the label uses the head
        cond = has_button * (pred["button_idx"] == t_button.clamp(0, BUTTONS - 1)).float()
        uses = self._uses(pred)
        for j, (h, n) in enumerate(zip(ARG_HEADS, ARG_SIZES)):
            t = tgt[f"{h}_idx"].long()
            w = cond * uses[..., j].float() * (t >= 0).float()
            terms[h], ok[h], illegal[h] = _categorical_head(
                pred[f"{h}_logits"], t.clamp(0, n - 1), pred.get(f"{h}_mask"), w,
                w.sum().clamp(min=1.0))
        return terms, ok, illegal, uses

    def il_head_terms(self, pred, tgt, seq_mask):
        """Heads ``button`` (every labelled frame) and ``move_x`` / ``move_z`` /
        ``skill_x`` / ``skill_z`` / ``target`` (frames whose labelled button uses the
        head).  Soft targets: ``<head>_probs``."""
        return self._il_parts(pred, tgt, seq_mask)[0]

    def il_loss(self, pred, tgt, seq_mask):
        terms, ok, illegal, uses = self._il_parts(pred, tgt, seq_mask)
        head = {k: (t["ce"] * t["weight"]).sum() / t["norm"] for k, t in terms.items()}
        loss = sum(head.values())
        with torch.no_grad():
            out = {f"{k}_loss": v.detach() for k, v in head.items()}
            out.update({f"{k}_acc": (ok[k].float() * terms[k]["weight"]).sum() / terms[k]["norm"]
                        for k in terms})
            answer = ok["button"].clone()
            for j, h in enumerate(ARG_HEADS):
                used = uses[..., j] & (tgt[f"{h}_idx"].long() >= 0)
                answer &= ~used | ok[h]
            valid = seq_mask.bool() & (tgt["button_idx"].long() >= 0)
            bad = [m for m in illegal.values() if m is not None]
            if bad:
                any_bad = torch.stack(bad).any(0) & valid
                out["illegal_target_frac"] = any_bad.float().sum() / valid.float().sum().clamp(min=1)
        return {"loss": loss, **out, "answer_correct": answer, "answer_mask": valid}


def hok_replay_legal(sample_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The compact legal dict that replays a rollout exactly: the masks
    :meth:`HokDecoder.sample` used (the per-button tables at the sampled button become
    ``target_of`` / ``sub_of``; the replay teacher-forces that button)."""
    out = {}
    for k in ("button", "move_x", "move_z", "skill_x", "skill_z"):
        if f"legal_{k}" in sample_out:
            out[k] = sample_out[f"legal_{k}"]
    if "legal_target" in sample_out:
        out["target_of"] = sample_out["legal_target"]
    if "legal_sub" in sample_out:
        out["sub_of"] = sample_out["legal_sub"]
    return out


def build_hok_network(**overrides) -> nn.Module:
    from model.game_network import GameNetwork

    cfg = HokConfig(**overrides)
    return GameNetwork(cfg, HokEncoder(cfg), HokDecoder(cfg))
