"""Tiny synthetic generals gio records for tests.

Same JSON layout as the Rust converter (data_processing/rs/gmdata/src/generals/gio.rs):
12 grid planes (11 one-hot tile states + the float32 log1p(army)/10 channel),
4 log-scaled globals, the own-tiles-with-army>1 source mask, one move slot
of max_actions_per_tick, count 1 / 0 / -1, a value in [-1, 1], tick features
and the all-zero enemy_heatmap / next_visibility placeholders.  Cells of the
last column are off-map (all-zero one-hot), as on maps smaller than the grid.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List

import torch

OWN = (5, 6, 7)           # own tile / city / general (legal sources with army > 1)


def _f32(x: float) -> float:
    return float(torch.tensor(x, dtype=torch.float32))


def synthetic_gio(T: int, H: int = 5, W: int = 5, seed: int = 0, K: int = 4) -> Dict:
    rng = random.Random(seed)
    rec: Dict[str, List] = {k: [] for k in (
        "grid_seq", "globals_seq", "legal_source_mask_seq", "count", "src_y", "src_x",
        "dir_id", "ratio_id", "value", "tick_seq", "tick_bucket_seq", "enemy_heatmap",
        "next_visibility")}
    for t in range(T):
        planes = [[[0.0] * W for _ in range(H)] for _ in range(12)]
        legal = [[[False] * W for _ in range(H)]]
        for i in range(H):
            for j in range(W - 1):                     # column W-1: off-map
                s = rng.choice([0, 0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10])
                planes[s][i][j] = 1.0
                army = rng.randint(0, 80) if s in (3,) + OWN + (8, 9, 10) else 0
                if army > 0:
                    planes[11][i][j] = _f32(math.log1p(army) / 10.0)
                legal[0][i][j] = s in OWN and army > 1
        cells = [(i, j) for i in range(H) for j in range(W) if legal[0][i][j]]
        u = rng.random()
        sy, sx, d, r = [0] * K, [0] * K, [0] * K, [0] * K
        if cells and u < 0.75:
            count = 1
            sy[0], sx[0] = rng.choice(cells)
            d[0], r[0] = rng.randint(0, 3), rng.randint(0, 1)
        else:
            count = -1 if u > 0.95 else 0
        w = (t / max(T - 1, 1)) ** 2
        rec["grid_seq"].append(planes)
        rec["globals_seq"].append([math.log1p(rng.randint(1, 300)) / 10.0 for _ in range(4)])
        rec["legal_source_mask_seq"].append(legal)
        rec["count"].append(count)
        rec["src_y"].append(sy)
        rec["src_x"].append(sx)
        rec["dir_id"].append(d)
        rec["ratio_id"].append(r)
        rec["value"].append(w * 1.0 + (1 - w) * math.tanh(rng.uniform(-2, 2)))
        rec["tick_seq"].append(t)
        rec["tick_bucket_seq"].append(((t % 50) + 1) // 2)
        rec["enemy_heatmap"].append([[[0.0] * W for _ in range(H)]])
        rec["next_visibility"].append([[[0.0] * W for _ in range(H)]])
    rec["max_actions_per_tick"] = K
    return rec


def write_gio_dir(root: str, lengths: List[int], H: int = 5, W: int = 5,
                  seed: int = 0, prefix: str = "g") -> List[str]:
    """One record per entry of ``lengths`` → ``root/<prefix><i>.json``."""
    os.makedirs(root, exist_ok=True)
    paths = []
    for i, T in enumerate(lengths):
        p = os.path.join(root, f"{prefix}{i:03d}.json")
        with open(p, "w") as f:
            json.dump(synthetic_gio(T, H, W, seed=seed * 1000 + i), f)
        paths.append(p)
    return paths
