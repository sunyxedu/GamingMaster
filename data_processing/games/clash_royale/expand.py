"""Load-time expansion of Clash Royale gm-shards into the adapter's observation.

The Rust converter (`crdata convert`) stores compact per-frame fields (`raw.*`);
materialising the dense arena planes on disk would cost ~36 KB per frame, so they
are built here when a sample is loaded (`GameShardDataset` calls
:func:`expand` for game ``clash_royale``).

Observation (own perspective, row 0 = own back line):

    arena_seq   [T, C, 32, 18]  0 own deployments exp(-age/tau)   1 opponent deployments
                                2 own deployment cost/10          3 opponent deployment cost/10
                                4 own half   5 enemy half   6 river   7 own elixir/10
                                8 opp elixir/10   9 time/300 s   10 double   11 triple
                                12 overtime   13.. reserved (0)
    hand_ids    [T, 4] long     hand_feats [T, 4, 8]  cost/10, troop, building, spell,
                                evolved, hero, affordable, P(card really in hand)
    hand_mask   [T, 4] bool     globals_seq [T, 6]  elixir/10, opp/10, time/300, phase/2,
                                overtime, next-card cost/10
    legal_card  [T, 4] bool     affordable
    legal_place [T, 576] bool   own half; whole arena when the hand holds a spell

Deployment planes stand in for unit positions, which the replays do not carry.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

GRID_H, GRID_W = 32, 18
OBS_KEYS = ("arena_seq", "hand_ids", "hand_feats", "hand_mask", "globals_seq", "legal_card", "legal_place")


def _np(v) -> np.ndarray:
    return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def expand(raw: Dict, shard_meta: Dict) -> Dict[str, torch.Tensor]:
    C = int(shard_meta.get("arena_channels", 16))
    F = int(shard_meta.get("card_feature_dim", 8))
    tau = float(shard_meta.get("tau_ticks", 60.0))
    elixir = _np(raw["elixir"]).astype(np.float32)
    opp_elixir = _np(raw["opp_elixir"]).astype(np.float32)
    time_s = _np(raw["time_s"]).astype(np.float32)
    phase = _np(raw["phase"]).astype(np.float32)
    overtime = _np(raw["overtime"]).astype(np.float32)
    T = elixir.shape[0]

    arena = np.zeros((T, C, GRID_H, GRID_W), dtype=np.float32)
    side = _np(raw["recent_side"])
    x = _np(raw["recent_x"]).astype(np.int64)
    y = _np(raw["recent_y"]).astype(np.int64)
    age = _np(raw["recent_age"]).astype(np.float32)
    cost = _np(raw["recent_cost"]).astype(np.float32) / 10.0
    t_idx = np.arange(T)
    # oldest first, so the newest play at a cell writes its cost plane last
    for k in range(side.shape[1] - 1, -1, -1):
        ok = (side[:, k] != 255) & (x[:, k] >= 0) & (y[:, k] >= 0)
        if not ok.any():
            continue
        tt, yy, xx = t_idx[ok], y[ok, k], x[ok, k]
        plane = np.where(side[ok, k] == 0, 0, 1)
        w = np.exp(-age[ok, k] / tau).astype(np.float32)
        arena[tt, plane, yy, xx] = np.maximum(arena[tt, plane, yy, xx], w)
        arena[tt, plane + 2, yy, xx] = cost[ok, k]
    arena[:, 4, :16, :] = 1.0
    arena[:, 5, 16:, :] = 1.0
    arena[:, 6, 15:17, :] = 1.0
    arena[:, 7] = (elixir / 10.0)[:, None, None]
    arena[:, 8] = (opp_elixir / 10.0)[:, None, None]
    arena[:, 9] = (time_s / 300.0)[:, None, None]
    arena[:, 10] = (phase >= 1)[:, None, None]
    arena[:, 11] = (phase >= 2)[:, None, None]
    arena[:, 12] = overtime[:, None, None]

    kind = _np(raw["hand_kind"])
    form = _np(raw["hand_form"])
    hcost = _np(raw["hand_cost"]).astype(np.float32)
    feats = np.zeros((T, 4, F), dtype=np.float32)
    feats[..., 0] = hcost / 10.0
    feats[..., 1] = kind == 0
    feats[..., 2] = kind == 1
    feats[..., 3] = kind == 2
    feats[..., 4] = (form >= 1) & (form < 200)
    feats[..., 5] = form == 200
    affordable = hcost <= elixir[:, None] + 1e-6
    feats[..., 6] = affordable
    feats[..., 7] = _np(raw["hand_prob"])
    globals_seq = np.stack([elixir / 10.0, opp_elixir / 10.0, time_s / 300.0, phase / 2.0, overtime,
                            _np(raw["next_cost"]).astype(np.float32) / 10.0], axis=1).astype(np.float32)
    legal_place = np.zeros((T, GRID_H * GRID_W), dtype=bool)
    legal_place[:, : 16 * GRID_W] = True
    legal_place[(kind == 2).any(axis=1)] = True
    return {
        "arena_seq": torch.from_numpy(arena),
        "hand_ids": torch.from_numpy(_np(raw["hand_ids"]).astype(np.int64)),
        "hand_feats": torch.from_numpy(feats),
        "hand_mask": torch.ones((T, 4), dtype=torch.bool),
        "globals_seq": torch.from_numpy(globals_seq),
        "legal_card": torch.from_numpy(affordable),
        "legal_place": torch.from_numpy(legal_place),
    }
