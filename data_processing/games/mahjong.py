"""Load-time expansion of riichi mahjong gm-shards (``gmdata mahjong convert``).

From converter_version 3 on, the tile planes are stored as ``raw.tiles_q``
[T, 34, 16] uint8 — every tile feature is a count / 4 or a 0/1 flag, stored
exactly as quarters (value × 4) — a quarter of the float32 ``tiles_seq``'s
size (the full Tenhou Houou corpus: ~1.1 TB instead of ~3.3 TB).  They are
expanded here when a sample is read (``GameShardDataset`` calls :func:`expand`
for game ``mahjong``), bit-identical to the float32 planes; layout in
model/games/mahjong.py.  Older shards store ``obs.tiles_seq`` itself.
"""
from __future__ import annotations

from typing import Dict

import torch

OBS_KEYS = ("tiles_seq",)


def expand(raw: Dict, shard_meta: Dict) -> Dict[str, torch.Tensor]:
    q = raw.get("tiles_q")
    if q is None:                    # converter_version <= 2: obs.tiles_seq on disk
        return {}
    q = q if isinstance(q, torch.Tensor) else torch.as_tensor(q)
    return {"tiles_seq": q.to(torch.float32) / 4.0}
