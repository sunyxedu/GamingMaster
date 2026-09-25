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
GM_SHARD_FORMAT = "gm-shard.v1"   # directories written by the Rust converters (gmshard crate)


class GmShard:
    """One ``gm-shard.v1`` directory (``index.json`` + ``obs.*.npy`` / ``tgt.*.npy``).

    Arrays are memory-mapped; a sample is a frame-range slice of every tensor, so
    nothing is materialised until ``sample(j)`` is called.  Presents the same
    ``{"format_version", "game", "converter_version", "meta", "samples"}``
    surface as a ``.pt`` shard dict (``samples`` is lazy)."""

    def __init__(self, path: str):
        import json
        with open(os.path.join(path, "index.json")) as fh:
            self.index = json.load(fh)
        if self.index.get("format") != GM_SHARD_FORMAT:
            raise ValueError(f"{path}: not a {GM_SHARD_FORMAT} shard")
        self.path = path
        self._arrays: Optional[Dict] = None

    @property
    def game(self) -> str:
        return self.index["game"]

    def arrays(self) -> Dict:
        if self._arrays is None:
            import numpy as np
            self._arrays = {name: np.load(os.path.join(self.path, info["file"]), mmap_mode="r")
                            for name, info in self.index["tensors"].items()}
        return self._arrays

    def __len__(self) -> int:
        return len(self.index["samples"])

    def sample(self, j: int) -> Dict:
        import numpy as np
        info = self.index["samples"][j]
        a, b = info["start"], info["start"] + info["length"]
        out: Dict = {"obs": {}, "tgt": {}, "raw": {}, "meta": info.get("meta") or {}}
        for name, arr in self.arrays().items():
            group, key = name.split(".", 1)
            tinfo = self.index["tensors"][name]
            dtype = tinfo["dtype"]
            if tinfo.get("storage") == "packbits":
                per = int(np.prod(tinfo["frame_shape"])) if tinfo["frame_shape"] else 1
                bits = np.unpackbits(np.asarray(arr[a:b]), axis=1, count=per)
                v = torch.from_numpy(bits.reshape((b - a, *tinfo["frame_shape"])).astype(np.float32))
                out[group][key] = v
                continue
            v = torch.from_numpy(np.array(arr[a:b]))  # copy: mmap slices are read-only
            if dtype == "bool":
                v = v.bool()
            out[group][key] = v
        raw = out.pop("raw")
        exp = _expander(self.game)
        if exp is not None:
            out["obs"].update(exp[0](raw, self.index.get("meta") or {}))
        elif raw:
            out["raw"] = raw
        return out

    def as_shard_dict(self) -> Dict:
        return {"format_version": FORMAT_VERSION, "game": self.game,
                "converter_version": self.index.get("converter_version", 1),
                "meta": self.index.get("meta") or {},
                "samples": [self.sample(j) for j in range(len(self))]}


def _expander(game: str):
    """Load-time expanders: games whose gm-shards store compact ``raw.*`` fields
    and build their dense observation when a sample is read."""
    if game == "clash_royale":
        from data_processing.games.clash_royale import expand
        return expand.expand, frozenset(expand.OBS_KEYS)
    return None


def find_gm_shards(data_dir: str) -> List[str]:
    return sorted(os.path.dirname(p) for p in
                  glob.glob(os.path.join(data_dir, "**", "index.json"), recursive=True)
                  if _is_gm_shard(os.path.dirname(p)))


def _is_gm_shard(path: str) -> bool:
    import json
    try:
        with open(os.path.join(path, "index.json")) as fh:
            return json.load(fh).get("format") == GM_SHARD_FORMAT
    except (OSError, ValueError):
        return False


def load_gm_samples(data_dir: str) -> List[Dict]:
    """Every sample of every gm-shard under ``data_dir`` (tests / small data)."""
    out: List[Dict] = []
    for d in find_gm_shards(data_dir):
        sh = GmShard(d)
        out.extend(sh.sample(j) for j in range(len(sh)))
    return out


def save_shard(path: str, game: str, samples: List[Dict],
               converter_version: int = 1,
               meta: Optional[Dict] = None) -> None:
    """Write one shard; converters' single entry point for output.

    ``converter_version`` marks semantic revisions of a converter's
    output (bump it when re-converted shards are incompatible with older
    ones — e.g. the chess rank-mirror fix); ``meta`` carries shard-level
    facts the model config must match (board_size, move_buckets, …).
    GameShardDataset enforces both uniform across a data_dir, so stale
    and fresh generations can never be mixed silently.
    """
    for s in samples:
        T = next(iter(s["obs"].values())).shape[0]
        for k, v in {**s["obs"], **s["tgt"]}.items():
            if v.shape[0] != T:
                raise ValueError(
                    f"sample tensor {k!r} has T={v.shape[0]}, expected {T}"
                )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"format_version": FORMAT_VERSION, "game": game,
         "converter_version": converter_version, "meta": dict(meta or {}),
         "samples": samples},
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
        # .pt shards (torch.save) and/or gm-shard.v1 directories (Rust converters)
        self.paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.pt"),
                                      recursive=True)) + find_gm_shards(data_dir)
        if not self.paths:
            raise FileNotFoundError(f"no .pt shards or gm-shard directories under {data_dir}")
        self.game = game
        self.max_len = max_len

        # Build the (shard_idx, sample_idx) index once up front.
        self._index: List[Tuple[int, int]] = []
        self.converter_version: Optional[int] = None
        self.shard_meta: Dict = {}
        self._ref_path: Optional[str] = None
        self._obs_keys: Optional[frozenset] = None
        self._tgt_keys: Optional[frozenset] = None
        for si, p in enumerate(self.paths):
            shard = self._open(p, header_only=True)
            self._check(shard, p)
            self._index.extend((si, j) for j in range(shard["n_samples"]))
        self._cache_idx: int = -1
        self._cache: Optional[Dict] = None

    @staticmethod
    def _open(path: str, header_only: bool = False) -> Dict:
        """Shard dict for a .pt file or a gm-shard directory. ``header_only``
        returns keys/meta/counts without materialising gm-shard samples."""
        if os.path.isdir(path):
            sh = GmShard(path)
            head = {"format_version": FORMAT_VERSION, "game": sh.game,
                    "converter_version": sh.index.get("converter_version", 1),
                    "meta": sh.index.get("meta") or {}, "n_samples": len(sh), "_gm": sh}
            if header_only:
                names = list(sh.index["tensors"])
                obs_keys = frozenset(n.split(".", 1)[1] for n in names if n.startswith("obs."))
                exp = _expander(sh.game)
                if exp is not None:
                    obs_keys = obs_keys | exp[1]
                head["_keys"] = (obs_keys, frozenset(n.split(".", 1)[1] for n in names if n.startswith("tgt.")))
            return head
        shard = torch.load(path, map_location="cpu", weights_only=False)
        shard["n_samples"] = len(shard["samples"])
        return shard

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
        # Converter generation + shard meta + sample schema must be
        # uniform across the directory: shards from different converter
        # revisions (e.g. chess pre/post rank-mirror) look identical at
        # the tensor level, so this is the only place mixing is caught.
        cv = shard.get("converter_version", 1)
        meta = shard.get("meta", {})
        if "_keys" in shard:
            obs_keys, tgt_keys = shard["_keys"] if shard["n_samples"] else (None, None)
        else:
            samples = shard["samples"]
            obs_keys = frozenset(samples[0]["obs"]) if samples else None
            tgt_keys = frozenset(samples[0]["tgt"]) if samples else None
        if self.converter_version is None:
            self.converter_version = cv
            self.shard_meta = meta
            self._ref_path = path
            self._obs_keys, self._tgt_keys = obs_keys, tgt_keys
            return
        if cv != self.converter_version:
            raise ValueError(
                f"{path}: converter_version {cv} but {self._ref_path} has "
                f"{self.converter_version} — the directory mixes shard "
                f"generations; re-convert everything with the current converter"
            )
        if meta != self.shard_meta:
            raise ValueError(
                f"{path}: shard meta {meta!r} but {self._ref_path} has "
                f"{self.shard_meta!r} — shards were converted with different "
                f"settings; re-convert to one configuration"
            )
        for kind, ks, ref in (("obs", obs_keys, self._obs_keys),
                              ("tgt", tgt_keys, self._tgt_keys)):
            if ks is not None and ref is not None and ks != ref:
                raise ValueError(
                    f"{path}: sample {kind} keys {sorted(ks)} but "
                    f"{self._ref_path} has {sorted(ref)} — the directory "
                    f"mixes converter outputs; re-convert everything"
                )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        si, j = self._index[idx]
        if si != self._cache_idx:
            self._cache = self._open(self.paths[si])
            self._cache_idx = si
        sample = (self._cache["_gm"].sample(j) if "_gm" in self._cache
                  else self._cache["samples"][j])
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

    # Backstop for datasets not built via GameShardDataset: keys are
    # taken from batch[0], so a schema mismatch would otherwise KeyError
    # or silently drop the extra keys for the whole batch.
    obs_keys, tgt_keys = set(batch[0]["obs"]), set(batch[0]["tgt"])
    for s in batch[1:]:
        if set(s["obs"]) != obs_keys or set(s["tgt"]) != tgt_keys:
            raise ValueError(
                "collate: samples disagree on obs/tgt keys "
                f"({sorted(obs_keys)}/{sorted(tgt_keys)} vs "
                f"{sorted(s['obs'])}/{sorted(s['tgt'])}) — mixed shard "
                "generations in one directory? Re-convert everything"
            )

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
