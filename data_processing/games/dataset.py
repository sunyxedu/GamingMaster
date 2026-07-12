"""Shared shard format + PyTorch dataset for multi-game imitation learning.

Shard format (one ``.pt`` file, ``torch.save``):

    {
      "format_version": 1,
      "game": "chess",                     # registry name
      "samples": [                          # one entry per game/episode
        {
          "obs": {key: Tensor [T, ...]},    # keys match the game's
                                            #   ObservationEncoder obs dict
          "tgt": {key: Tensor [T, ...]},    # keys match the game's
                                            #   ActionDecoder.il_loss targets;
                                            #   "value" [T] float is consumed
                                            #   by the shared value head loss
          "meta": {...},                    # free-form (source file, players…)
        },
        ...
      ],
    }

Design notes:
  - One sample = one full game from one player's perspective (like the
    generals pipeline). Converters decide perspective/augmentation.
  - Shards hold many short games so file count stays sane at Lichess scale;
    converters should target ~256–1024 samples per shard.
  - Collate pads every sequence at the TAIL to the batch max length and
    returns ``lengths`` + ``frame_mask`` — tail padding is what the core's
    masked-BatchNorm / causal-mixer guarantees assume.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

FORMAT_VERSION = 1


def save_shard(path: str, game: str, samples: List[Dict]) -> None:
    """Write one shard; converters' single entry point for output."""
    for s in samples:
        T = next(iter(s["obs"].values())).shape[0]
        for k, v in {**s["obs"], **s["tgt"]}.items():
            if v.shape[0] != T:
                raise ValueError(
                    f"sample tensor {k!r} has T={v.shape[0]}, expected {T}"
                )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"format_version": FORMAT_VERSION, "game": game, "samples": samples},
        path,
    )


class GameShardDataset(Dataset):
    """Index over every sample in every ``*.pt`` shard under ``data_dir``.

    Shards are loaded lazily and the most recent one is cached.  The
    cache only helps when consecutive accesses hit the same shard:
    plain ``shuffle=True`` yields i.i.d. indices and reloads a full
    shard for almost every sample.  Use :class:`ShardLocalitySampler`
    (shuffles shard order, then sample order within each shard) to get
    shuffling with cache hits.
    """

    def __init__(self, data_dir: str, game: Optional[str] = None,
                 max_len: Optional[int] = None):
        self.paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.pt"),
                                      recursive=True))
        if not self.paths:
            raise FileNotFoundError(f"no .pt shards under {data_dir}")
        self.game = game
        self.max_len = max_len

        # Build the (shard_idx, sample_idx) index once up front.
        self._index: List[Tuple[int, int]] = []
        for si, p in enumerate(self.paths):
            shard = torch.load(p, map_location="cpu", weights_only=False)
            self._check(shard, p)
            self._index.extend((si, j) for j in range(len(shard["samples"])))
        self._cache_idx: int = -1
        self._cache: Optional[Dict] = None

    def _check(self, shard: Dict, path: str) -> None:
        if shard.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format_version {shard.get('format_version')!r}, "
                f"expected {FORMAT_VERSION}"
            )
        if self.game is not None and shard["game"] != self.game:
            raise ValueError(
                f"{path}: shard is for game {shard['game']!r}, expected {self.game!r}"
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        si, j = self._index[idx]
        if si != self._cache_idx:
            self._cache = torch.load(
                self.paths[si], map_location="cpu", weights_only=False,
            )
            self._cache_idx = si
        sample = self._cache["samples"][j]
        if self.max_len is not None:
            T = next(iter(sample["obs"].values())).shape[0]
            if T > self.max_len:
                # Random crop keeps late-game frames reachable across epochs.
                t0 = int(torch.randint(0, T - self.max_len + 1, ()).item())
                sample = {
                    "obs": {k: v[t0: t0 + self.max_len] for k, v in sample["obs"].items()},
                    "tgt": {k: v[t0: t0 + self.max_len] for k, v in sample["tgt"].items()},
                    "meta": sample.get("meta", {}),
                }
        return sample


class ShardLocalitySampler(torch.utils.data.Sampler):
    """Shuffled sampling that keeps shard locality.

    Visits shards in a fresh random order each epoch and shuffles the
    sample order WITHIN each shard, so :class:`GameShardDataset`'s
    one-shard cache gets a hit on nearly every access instead of
    reloading a shard per sample (i.i.d. ``shuffle=True`` behavior).
    Randomness comes from the global torch RNG (seed it for
    reproducibility).
    """

    def __init__(self, dataset: GameShardDataset):
        self.dataset = dataset
        # global index ranges per shard (dataset._index is built in
        # shard-major order, so each shard's samples are contiguous)
        self._by_shard: Dict[int, List[int]] = {}
        for gi, (si, _) in enumerate(dataset._index):
            self._by_shard.setdefault(si, []).append(gi)

    def __iter__(self):
        shard_order = torch.randperm(len(self.dataset.paths)).tolist()
        for si in shard_order:
            idxs = self._by_shard.get(si, [])
            if not idxs:
                continue
            for j in torch.randperm(len(idxs)).tolist():
                yield idxs[j]

    def __len__(self) -> int:
        return len(self.dataset)


def collate_game_sequences(batch: List[Dict]) -> Dict:
    """Tail-pad every obs/tgt tensor to the batch max length.

    Returns:
        {
          "obs": {k: [B, T_max, ...]},
          "tgt": {k: [B, T_max, ...]},   # padded with zeros
          "meta": {"lengths": [B] ints, "frame_mask": [B, T_max] bool},
        }
    """
    lengths = [next(iter(s["obs"].values())).shape[0] for s in batch]
    T_max = max(lengths)
    B = len(batch)

    def _pad_stack(key_group: str, key: str) -> torch.Tensor:
        ref = batch[0][key_group][key]
        out = torch.zeros(B, T_max, *ref.shape[1:], dtype=ref.dtype)
        for i, s in enumerate(batch):
            v = s[key_group][key]
            out[i, : v.shape[0]] = v
        return out

    obs = {k: _pad_stack("obs", k) for k in batch[0]["obs"]}
    tgt = {k: _pad_stack("tgt", k) for k in batch[0]["tgt"]}
    frame_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, L in enumerate(lengths):
        frame_mask[i, :L] = True
    return {
        "obs": obs,
        "tgt": tgt,
        "meta": {"lengths": lengths, "frame_mask": frame_mask},
    }
