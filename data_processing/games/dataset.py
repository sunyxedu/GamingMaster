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

``tgt["value"]`` contract (all converters; checked by
tests/test_rust_converters_parity.py): the FINAL outcome of the game (match
for lux) from the perspective of the player acting in that frame, in
[-VALUE_TARGET_BOUND, VALUE_TARGET_BOUND] = [-1, 1]:

    chess, shogi, go     +1 win / 0 draw or no result / −1 loss (mover)
    othello              sign(final own − opponent discs) for the mover
    gomoku               ±1 for the mover (undecided kept games: last mover wins)
    lux                  ±1 / 0 per match for the agent's team (wins, then points)
    mahjong              final placement 1st..4th → +1, +1/3, −1/3, −1 (seat)
    pokemon              ±1 for the recorded side
    clash_royale         +1 victory / 0 draw / −1 defeat for the side (crdata)
    generals             NOT a pure outcome: the gio record's blend
                         w_t·outcome + (1 − w_t)·tanh(army/city ratio − 1),
                         w_t = (t / (T − 1))², outcome ±1 for the recorded
                         player (the training/train_il.py target) — in [-1, 1]

The board-game samples (chess, shogi, go, othello, gomoku) hold both players'
moves, so the sign alternates with the mover; the other games' samples hold
one player's view, so their value is constant over the sample (except the
generals blend above).
training/train_il_game.py rejects targets outside an HL-Gauss value head's
support (check_value_support).

Optional engine / legality targets (all [T, ...] like every tgt tensor; absent
= the plain hard-label objective; consumed generically by
training/train_il_game.py ``batch_loss``, contract in
``model.games.base.ActionDecoder.il_head_terms``):

    <h>_probs              soft policy target of decoder head h (the head names
                           of the decoder's il_head_terms): categorical
                           [T, ..., N] teacher distribution (engine multipv /
                           MCTS visit counts; rows of zeros = no teacher),
                           bernoulli [T, ...] = P(class 1)
    <h>_probs_idx          sparse categorical alternative: long [T, ..., K]
    <h>_probs_val          class ids (-1 = pad) + float [T, ..., K] probabilities
    <h>_probs_mask         optional bool [T, ...] teacher presence (default:
                           categorical rows with positive mass, bernoulli all)
    engine_value           float [T] in [-1, 1]: the mover-POV engine expected
                           score 2·(W + D/2) − 1 (blended into the value target,
                           --engine_value_weight)
    engine_value_mask      optional bool [T]: frames with an engine evaluation
    value_mask             optional bool [T]: frames whose outcome ``value`` is valid
    legal_<key>            IL legality mask (bool, or packbits-stored {0, 1}
                           float) → the decoder's ``legal["<key>"]`` under the
                           trainers' --il_legal_mask; never a soft target
    il_weight              optional float [T]: per-frame weight of the decoder's
                           imitation terms (mahjong, converter v3: 0 on a recorded
                           pass whose choice the log does not show — a call it
                           could make was pre-empted); the value target is kept

Collate pads all of them with zeros (False), i.e. "no teacher / no engine
value / nothing legal" on padded frames, which the frame mask excludes anyway.

Design notes:
  - One sample = one full game from one player's perspective (like the
    generals pipeline). Converters decide perspective/augmentation.
  - Shards hold many short games so file count stays sane at Lichess scale;
    converters should target ~256–1024 samples per shard.
  - Collate pads every sequence at the TAIL to the batch max length and
    returns ``lengths`` + ``frame_mask`` — tail padding is what the core's
    masked-BatchNorm / causal-mixer guarantees assume.
  - ``max_len`` crops a longer sample to a random window (``crop="random"``)
    or to its first frames (``crop="prefix"``, for adapters that read the
    absolute frame index, e.g. generals).
  - generals also trains from its gio JSON record directories:
    :func:`open_game_dataset` (see data_processing/games/generals.py).
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

FORMAT_VERSION = 1
GM_SHARD_FORMAT = "gm-shard.v1"   # directories written by the Rust converters (gmshard crate)
VALUE_TARGET_BOUND = 1.0          # |tgt["value"]| ≤ this for every converter (see above)


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
    and build their dense observation when a sample is read.  Returns
    ``(expand(raw, shard_meta), obs_keys(raw column names))`` or None."""
    if game == "clash_royale":
        from data_processing.games.clash_royale import expand
        return expand.expand, expand.obs_keys
    if game == "generals":
        from data_processing.games import generals
        return generals.expand, lambda raw_names: frozenset(generals.OBS_KEYS)
    if game == "mahjong":
        from data_processing.games import mahjong
        # converter_version >= 3 stores raw.tiles_q; older shards carry obs.tiles_seq
        return mahjong.expand, (lambda raw_names: frozenset(mahjong.OBS_KEYS)
                                if "tiles_q" in raw_names else frozenset())
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

    ``max_len`` crops longer samples: a random window (``crop="random"``,
    re-drawn per access) or the first ``max_len`` frames (``"prefix"``).
    """

    CROPS = ("random", "prefix")

    def __init__(self, data_dir: str, game: Optional[str] = None,
                 max_len: Optional[int] = None, crop: str = "random"):
        if crop not in self.CROPS:
            raise ValueError(f"crop must be one of {self.CROPS}, got {crop!r}")
        self.crop = crop
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
                    raw_names = [n.split(".", 1)[1] for n in names if n.startswith("raw.")]
                    obs_keys = obs_keys | exp[1](raw_names)
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

    def total_frames(self) -> int:
        """Frames the dataset supervises (each sample capped at max_len).
        Cheap for gm-shard directories (index only); .pt shards are loaded."""
        total = 0
        for p in self.paths:
            if os.path.isdir(p):
                lengths = [int(s["length"]) for s in GmShard(p).index["samples"]]
            else:
                lengths = [next(iter(s["obs"].values())).shape[0]
                           for s in torch.load(p, map_location="cpu", weights_only=False)["samples"]]
            if self.max_len is not None:
                lengths = [min(L, self.max_len) for L in lengths]
            total += sum(lengths)
        return total

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
                t0 = (0 if self.crop == "prefix"
                      else int(torch.randint(0, T - self.max_len + 1, ()).item()))
                sample = {
                    "obs": {k: v[t0: t0 + self.max_len] for k, v in sample["obs"].items()},
                    "tgt": {k: v[t0: t0 + self.max_len] for k, v in sample["tgt"].items()},
                    "meta": sample.get("meta", {}),
                }
        return sample


def has_shards(data_dir: str) -> bool:
    """Whether ``data_dir`` holds any .pt shard or gm-shard directory."""
    return bool(glob.glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)
                or find_gm_shards(data_dir))


def open_game_dataset(data_dir: str, game: str, max_len: Optional[int] = None,
                      crop: str = "random") -> Dataset:
    """The training dataset of ``game`` under ``data_dir``.

    A :class:`GameShardDataset` over the .pt shards / gm-shard directories —
    or, for generals when there are none, a
    :class:`~data_processing.games.generals.GeneralsGioDataset` over the gio
    JSON records (``crop`` must then be ``"prefix"``: gio records are only
    ever truncated at the tail).  Both expose ``total_frames()``,
    ``shard_meta`` and ``converter_version``."""
    if game == "generals" and os.path.isdir(data_dir) and not has_shards(data_dir):
        from data_processing.games.generals import GeneralsGioDataset
        if max_len is not None and crop != "prefix":
            raise ValueError("generals gio records keep their first max_len frames: "
                             "crop must be 'prefix'")
        return GeneralsGioDataset(data_dir, max_len=max_len)
    return GameShardDataset(data_dir, game=game, max_len=max_len, crop=crop)


def training_sampler(dataset: Dataset) -> torch.utils.data.Sampler:
    """Shuffling sampler for training: :class:`ShardLocalitySampler` for a
    GameShardDataset (keeps its one-shard cache hot), a plain random
    permutation for per-record datasets (GeneralsGioDataset).  Both draw
    from the global torch RNG."""
    if isinstance(dataset, GameShardDataset):
        return ShardLocalitySampler(dataset)
    return torch.utils.data.RandomSampler(dataset)


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
