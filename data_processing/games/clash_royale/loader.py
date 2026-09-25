"""Clash Royale decision frames (``cr-frames.v1``, written by the Rust
``crdata`` converter) → the multi-game shard format consumed by
``data_processing.games.dataset.GameShardDataset`` and the
``model/games/clash_royale.py`` adapter.

    python -m data_processing.games.clash_royale.loader \
        --frames data/cr/frames --out data/cr/shards [--arena-channels 16]

Observation built here (the replay carries plays, not units, so the arena
planes are *deployment* planes — where each side committed elixir, and how
long ago — rather than unit positions; see README):

    arena_seq   [T, C, 32, 18]  C planes, own perspective (row 0 = own back line)
        0  own recent deployments, weight exp(-age / tau)
        1  opponent recent deployments, weight exp(-age / tau)
        2  own recent deployments, elixir cost / 10 at the cell
        3  opponent recent deployments, elixir cost / 10
        4  own half (rows 0..15)            5  enemy half (rows 16..31)
        6  river band (rows 15..16)         7  own elixir / 10 (broadcast)
        8  opponent elixir / 10             9  match time / 300 s
        10 double-elixir flag              11 triple-elixir flag
        12 overtime flag                   13.. zeros (reserved)
    hand_ids    [T, 4] long   catalog ids (0 empty, N+1 unknown)
    hand_feats  [T, 4, 8]     cost/10, troop, building, spell, evolved, hero, affordable, P(card in hand)
    hand_mask   [T, 4] bool   always True (the hand is always full after the first deal)
    globals_seq [T, 6]        own elixir/10, opp elixir/10, time/300, phase/2, overtime, next-card cost/10

Targets: ``waited`` [T] bool, ``card_slot`` [T] long (−1 on waited frames),
(the hand cycle is only partially observable from plays: ``hand_feats[..., 7]``
is the fraction of cycle states consistent with the replay in which the slot's
card really is in hand),
``place_idx`` [T] long (row * 18 + col, −1 on waited frames), ``value`` [T]
float (+1 win / −1 loss / 0 draw from this player's perspective).
Legal masks (``legal_card`` [T, 4], ``legal_place`` [T, 576]) are stored in
``obs`` so the IL trainer can pass them through: a card is legal when
affordable; placement is the own half for troops/buildings and the whole
arena when the hand holds a spell (placement legality also depends on tower
state, which the replay does not carry).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from data_processing.games.dataset import save_shard

GRID_H, GRID_W = 32, 18
CONVERTER_VERSION = 1

FRAME_ARRAYS = [
    "tick", "time_s", "phase", "overtime", "elixir", "opp_elixir", "hand_ids", "hand_form", "hand_kind",
    "hand_cost", "hand_prob", "next_card", "recent_side", "recent_card", "recent_x", "recent_y", "recent_age",
    "waited", "card_slot", "place_x", "place_y", "ability", "ability_card",
]


def load_frames_shard(path: str) -> Dict:
    """One ``shard-XXXXXX/`` directory → {"index": ..., <array name>: np.ndarray}."""
    with open(os.path.join(path, "index.json")) as fh:
        index = json.load(fh)
    if index.get("format") != "cr-frames.v1":
        raise ValueError(f"{path}: unexpected format {index.get('format')!r}")
    out = {"index": index}
    for name in FRAME_ARRAYS:
        out[name] = np.load(os.path.join(path, name + ".npy"))
    return out


def catalog_costs(catalog_path: Optional[str] = None) -> np.ndarray:
    """cost lookup by catalog id (index 0 = empty, last = unknown)."""
    path = catalog_path or os.path.join(os.path.dirname(__file__), "catalog", "cards.json")
    with open(path) as fh:
        cat = json.load(fh)
    n = len(cat["cards"])
    costs = np.zeros(n + 2, dtype=np.float32)
    for c in cat["cards"]:
        costs[c["id"]] = float(c["cost"] or 0.0)
    return costs


def _deploy_planes(shard: Dict, s: int, e: int, costs: np.ndarray, tau_ticks: float) -> np.ndarray:
    """[T, 4, H, W]: (own heat, opp heat, own cost, opp cost) from the recent-play window."""
    T = e - s
    side = shard["recent_side"][s:e]            # [T, K]
    card = shard["recent_card"][s:e]
    x = shard["recent_x"][s:e].astype(np.int64)
    y = shard["recent_y"][s:e].astype(np.int64)
    age = shard["recent_age"][s:e].astype(np.float32)
    planes = np.zeros((T, 4, GRID_H, GRID_W), dtype=np.float32)
    valid = (side != 255) & (x >= 0) & (y >= 0)
    t_idx, k_idx = np.nonzero(valid)
    if len(t_idx) == 0:
        return planes
    own = side[t_idx, k_idx] == 0
    w = np.exp(-age[t_idx, k_idx] / tau_ticks)
    c = costs[np.clip(card[t_idx, k_idx], 0, len(costs) - 1)] / 10.0
    yy, xx = y[t_idx, k_idx], x[t_idx, k_idx]
    # newest play wins on collisions (window is newest-first, so iterate reversed)
    order = np.argsort(-k_idx, kind="stable")
    for i in order:
        p = 0 if own[i] else 1
        planes[t_idx[i], p, yy[i], xx[i]] = max(planes[t_idx[i], p, yy[i], xx[i]], w[i])
        planes[t_idx[i], p + 2, yy[i], xx[i]] = c[i]
    return planes


def frames_to_sample(shard: Dict, meta: Dict, costs: np.ndarray, *, arena_channels: int = 16,
                     card_feature_dim: int = 8, tau_ticks: float = 60.0, max_len: Optional[int] = None) -> Dict:
    s, e = meta["start"], meta["start"] + meta["length"]
    if max_len is not None:
        e = min(e, s + max_len)
    T = e - s
    if T <= 0:
        raise ValueError("empty sample")
    elixir = shard["elixir"][s:e]
    opp_elixir = shard["opp_elixir"][s:e]
    time_s = shard["time_s"][s:e]
    phase = shard["phase"][s:e].astype(np.float32)
    overtime = shard["overtime"][s:e].astype(np.float32)

    arena = np.zeros((T, arena_channels, GRID_H, GRID_W), dtype=np.float32)
    arena[:, :4] = _deploy_planes(shard, s, e, costs, tau_ticks)
    arena[:, 4, :16, :] = 1.0
    arena[:, 5, 16:, :] = 1.0
    arena[:, 6, 15:17, :] = 1.0
    arena[:, 7] = (elixir / 10.0)[:, None, None]
    arena[:, 8] = (opp_elixir / 10.0)[:, None, None]
    arena[:, 9] = (time_s / 300.0)[:, None, None]
    arena[:, 10] = (phase >= 1)[:, None, None]
    arena[:, 11] = (phase >= 2)[:, None, None]
    arena[:, 12] = overtime[:, None, None]

    hand_ids = shard["hand_ids"][s:e].astype(np.int64)
    kind = shard["hand_kind"][s:e]
    form = shard["hand_form"][s:e]
    cost = shard["hand_cost"][s:e]
    feats = np.zeros((T, 4, card_feature_dim), dtype=np.float32)
    feats[..., 0] = cost / 10.0
    feats[..., 1] = kind == 0
    feats[..., 2] = kind == 1
    feats[..., 3] = kind == 2
    feats[..., 4] = (form >= 1) & (form < 200)
    feats[..., 5] = form == 200
    affordable = cost <= elixir[:, None] + 1e-6
    feats[..., 6] = affordable
    feats[..., 7] = shard["hand_prob"][s:e]
    next_cost = costs[np.clip(shard["next_card"][s:e], 0, len(costs) - 1)]
    globals_seq = np.stack([elixir / 10.0, opp_elixir / 10.0, time_s / 300.0, phase / 2.0, overtime,
                            next_cost / 10.0], axis=1).astype(np.float32)

    waited = shard["waited"][s:e].astype(bool)
    card_slot = shard["card_slot"][s:e].astype(np.int64)
    px = shard["place_x"][s:e].astype(np.int64)
    py = shard["place_y"][s:e].astype(np.int64)
    place_idx = np.where(waited, -1, py * GRID_W + px)

    legal_card = affordable
    has_spell = (kind == 2).any(axis=1)
    legal_place = np.zeros((T, GRID_H * GRID_W), dtype=bool)
    legal_place[:, : 16 * GRID_W] = True
    legal_place[has_spell] = True

    value = np.full((T,), float(meta["result"]), dtype=np.float32)
    obs = {
        "arena_seq": torch.from_numpy(arena),
        "hand_ids": torch.from_numpy(hand_ids),
        "hand_feats": torch.from_numpy(feats),
        "hand_mask": torch.ones((T, 4), dtype=torch.bool),
        "globals_seq": torch.from_numpy(globals_seq),
        "legal_card": torch.from_numpy(legal_card),
        "legal_place": torch.from_numpy(legal_place),
    }
    tgt = {
        "waited": torch.from_numpy(waited),
        "card_slot": torch.from_numpy(card_slot),
        "place_idx": torch.from_numpy(place_idx),
        "value": torch.from_numpy(value),
    }
    sample_meta = {k: meta[k] for k in ("tag", "side", "result", "own_crowns", "opp_crowns", "battle_type",
                                        "game_mode", "deck_ids", "tower_key", "hand_consistent_states",
                                        "elixir_violations", "n_plays")}
    return {"obs": obs, "tgt": tgt, "meta": sample_meta}


def convert_dir(frames_dir: str, out_dir: str, *, arena_channels: int = 16, card_feature_dim: int = 8,
                tau_ticks: float = 60.0, max_len: Optional[int] = None, catalog_path: Optional[str] = None,
                samples_per_shard: Optional[int] = None) -> List[str]:
    costs = catalog_costs(catalog_path)
    shards = sorted(glob.glob(os.path.join(frames_dir, "shard-*")))
    if not shards:
        raise FileNotFoundError(f"no shard-* directories under {frames_dir}")
    written: List[str] = []
    n_cards = len(costs) - 1
    for si, sd in enumerate(shards):
        fs = load_frames_shard(sd)
        idx = fs["index"]
        meta = {"arena_h": GRID_H, "arena_w": GRID_W, "arena_channels": arena_channels, "hand_size": 4,
                "n_cards": n_cards + 1, "card_feature_dim": card_feature_dim, "cr_global_features": 6,
                "step_ticks": idx["step_ticks"], "recent": idx["recent"], "catalog_version": idx["catalog_version"],
                "tau_ticks": tau_ticks}
        samples = [frames_to_sample(fs, m, costs, arena_channels=arena_channels, card_feature_dim=card_feature_dim,
                                    tau_ticks=tau_ticks, max_len=max_len) for m in idx["samples"]]
        per = samples_per_shard or len(samples)
        for j in range(0, len(samples), per):
            path = os.path.join(out_dir, f"cr-{si:06d}-{j // per:03d}.pt")
            save_shard(path, "clash_royale", samples[j:j + per], converter_version=CONVERTER_VERSION, meta=meta)
            written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True, help="directory of cr-frames.v1 shard-* dirs")
    ap.add_argument("--out", required=True, help="output directory for .pt shards")
    ap.add_argument("--arena-channels", type=int, default=16)
    ap.add_argument("--card-feature-dim", type=int, default=8)
    ap.add_argument("--tau-ticks", type=float, default=60.0, help="deployment-plane decay constant (ticks)")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--samples-per-shard", type=int, default=None)
    a = ap.parse_args()
    paths = convert_dir(a.frames, a.out, arena_channels=a.arena_channels, card_feature_dim=a.card_feature_dim,
                        tau_ticks=a.tau_ticks, max_len=a.max_len, samples_per_shard=a.samples_per_shard)
    print(f"wrote {len(paths)} shard(s) to {a.out}")


if __name__ == "__main__":
    main()
