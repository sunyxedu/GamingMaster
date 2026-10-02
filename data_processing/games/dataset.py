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
    lux                  ±1 / 0 of the frame's match for the agent's team (wins, then
                         points); a sample is the whole game, its 5 matches in a row
    mahjong              final placement 1st..4th → +1, +1/3, −1/3, −1 (seat)
    pokemon              ±1 for the recorded side
    clash_royale         +1 victory / 0 draw / −1 defeat for the side (crdata)
    generals             NOT a pure outcome: the gio record's blend
                         w_t·outcome + (1 − w_t)·tanh(army/city ratio − 1),
                         w_t = (t / (T − 1))², outcome ±1 for the recorded
                         player (the training/train_il.py target) — in [-1, 1]

The board-game records (chess, shogi, go, othello, gomoku, ...) hold both
players' moves, so the sign alternates with the mover; GameShardDataset serves
them as per-seat streams (:data:`SEAT_SPLIT_GAMES`), one player's frames each,
like the other games' samples: their value is constant over the sample (except
the generals blend above).
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

import bisect
import glob
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

FORMAT_VERSION = 1
GM_SHARD_FORMAT = "gm-shard.v1"   # directories written by the Rust converters (gmshard crate)
VALUE_TARGET_BOUND = 1.0          # |tgt["value"]| ≤ this for every converter (see above)


#: Per-shard sidecars next to ``index.json`` (scripts/build_shard_index.py): the index without
#: its per-sample list and every sample's frame count.  A dataset's start-up only needs these —
#: the full index (hundreds of bytes per sample: start, length, record meta) is parsed when a
#: shard is first read.  Used only while at least as new as index.json.
HEAD_SIDECAR = "index.head.json"
LENGTHS_SIDECAR = "index.lengths.npy"


def shard_header(path: str) -> Tuple[Dict, np.ndarray]:
    """``(index without "samples" + "n_samples", lengths int32 [n])`` of a gm-shard
    directory: from its sidecars when they are fresh, else from index.json."""
    import json
    idx = os.path.join(path, "index.json")
    hp, lp = os.path.join(path, HEAD_SIDECAR), os.path.join(path, LENGTHS_SIDECAR)
    try:
        t = os.stat(idx).st_mtime
        if os.stat(hp).st_mtime >= t and os.stat(lp).st_mtime >= t:
            with open(hp) as fh:
                head = json.load(fh)
            lengths = np.load(lp)
            if head.get("format") == GM_SHARD_FORMAT and len(lengths) == head.get("n_samples"):
                return head, lengths
    except (OSError, ValueError):
        pass
    with open(idx) as fh:
        index = json.load(fh)
    samples = index.pop("samples")
    index["n_samples"] = len(samples)
    return index, np.asarray([int(x["length"]) for x in samples], dtype=np.int32)


def write_shard_sidecars(path: str) -> int:
    """Write (atomically) the sidecars of one gm-shard directory; returns its sample count."""
    import json
    head, lengths = shard_header(path)
    tmp = f".tmp-{os.getpid()}"
    with open(os.path.join(path, HEAD_SIDECAR + tmp), "w") as fh:
        json.dump(head, fh)
    with open(os.path.join(path, LENGTHS_SIDECAR + tmp), "wb") as fh:
        np.save(fh, lengths.astype(np.int32))
    os.replace(os.path.join(path, LENGTHS_SIDECAR + tmp), os.path.join(path, LENGTHS_SIDECAR))
    os.replace(os.path.join(path, HEAD_SIDECAR + tmp), os.path.join(path, HEAD_SIDECAR))
    return int(head["n_samples"])


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


#: index.json's head up to its top-level "format": only plain scalar keys before it (the Rust
#: writers put "format" first or after "converter_version")
_FORMAT_HEAD = None


def _is_gm_shard(path: str) -> bool:
    """Whether ``path/index.json`` is a gm-shard index: its "format" from the file's head
    (a full index can be megabytes), else from a fresh head sidecar, else the whole file."""
    import json
    import re
    global _FORMAT_HEAD
    if _FORMAT_HEAD is None:
        _FORMAT_HEAD = re.compile(r'\s*\{(?:\s*"\w+"\s*:\s*(?:-?\d+|"[^"\\]*")\s*,)*'
                                  r'\s*"format"\s*:\s*"([^"\\]*)"')
    try:
        with open(os.path.join(path, "index.json")) as fh:
            m = _FORMAT_HEAD.match(fh.read(512))
            if m:
                return m.group(1) == GM_SHARD_FORMAT
            hp = os.path.join(path, HEAD_SIDECAR)
            if os.path.exists(hp) and os.stat(hp).st_mtime >= os.fstat(fh.fileno()).st_mtime:
                with open(hp) as h:
                    return json.load(h).get("format") == GM_SHARD_FORMAT
            fh.seek(0)
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


#: Two-player perfect-information games whose records hold BOTH movers' frames, each
#: from its mover's view, with strictly alternating movers (chess, chess variants,
#: shogi, xiangqi and gomoku alternate every ply; othello's converter inserts the forced
#: pass frames; go-kgs: 6400 of 6400 records checked).  :class:`GameShardDataset` serves
#: each such record as TWO samples, one per mover: frames 0, 2, 4, … and 1, 3, 5, ….
#: The network's Stage-1 memory runs per token slot across frames, and a record that
#: switches the view every frame (ranks mirrored or the board rotated for the second
#: player, own / opponent planes swapped) would make a slot's memory mix two squares or
#: two colours.  A per-seat stream keeps one view throughout; the opponent's replies
#: show as the change between consecutive frames, as they do for an agent that plays
#: one side.  Shards whose meta says ``layout: per_seat`` are served as they are.
SEAT_SPLIT_GAMES = frozenset({"chess", "chess_variants", "go", "gomoku", "othello",
                              "shogi", "xiangqi"})


def seat_stream_lengths(lengths: np.ndarray) -> np.ndarray:
    """Record lengths → the lengths of their two seat streams, interleaved
    (first mover, second mover per record).  A one-frame record's second stream
    repeats its frame (no empty samples)."""
    lengths = np.asarray(lengths, dtype=np.int32)
    out = np.empty(2 * len(lengths), dtype=np.int32)
    out[0::2] = (lengths + 1) // 2
    out[1::2] = np.maximum(lengths // 2, 1)
    return out


def seat_stream(sample: Dict, seat: int) -> Dict:
    """The frames of ``seat`` (0 = the record's first mover) of an alternating
    record: every second frame from frame ``seat``; the sample meta gains
    ``seat_stream``."""
    T = next(iter(sample["obs"].values())).shape[0]
    idx = torch.arange(seat, T, 2) if T > seat else torch.zeros(1, dtype=torch.long)
    return {"obs": {k: v.index_select(0, idx) for k, v in sample["obs"].items()},
            "tgt": {k: v.index_select(0, idx) for k, v in sample["tgt"].items()},
            "meta": {**(sample.get("meta") or {}), "seat_stream": seat}}


class GameShardDataset(Dataset):
    """Index over every sample in every ``*.pt`` shard under ``data_dir``.

    Shards are loaded lazily and the most recent one is cached.  The
    cache only helps when consecutive accesses hit the same shard:
    plain ``shuffle=True`` yields i.i.d. indices and reloads a full
    shard for almost every sample.  Use :class:`ShardLocalitySampler`
    (shuffles shard order, then sample order within each shard) to get
    shuffling with cache hits.

    ``max_len`` crops longer samples to a window of ``max_len`` frames,
    re-drawn per access: ``crop="random"`` starts it uniformly in
    [0, T − L], which draws the first and last frames of a long sample only
    1/min(L, T − L + 1) as often as its middle frames (a 250-frame go game at
    L = 64: its first 16 moves 7× less often); ``"balanced"`` draws the start
    uniformly in [−(L − 1), T − 1] and clips it to a full window inside the
    sample, so every frame of a sample with T ≥ 2L is drawn at least as often
    as a middle frame (at most ~2× in the first / last L frames) and shorter
    ones lose at most ~2× at the ends; ``"prefix"`` keeps the first
    ``max_len`` frames.

    ``seat_split`` (default: the game is in :data:`SEAT_SPLIT_GAMES` and its shards
    are not ``layout: per_seat``) serves every record as its two seat streams
    (:func:`seat_stream`): sample ``2r`` / ``2r + 1`` = record ``r``'s first /
    second mover, cropped like any sample.
    """

    CROPS = ("random", "balanced", "prefix")

    def __init__(self, data_dir: str, game: Optional[str] = None,
                 max_len: Optional[int] = None, crop: str = "random",
                 seat_split: Optional[bool] = None):
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

        # Samples are numbered shard-major: shard si holds the global indices
        # [_offsets[si], _offsets[si + 1]).  O(#shards) memory — a per-sample
        # index would cost ~150 B per sample in every process (the trainer's
        # ranks and each DataLoader worker), gigabytes for 1e7-sample corpora.
        self._offsets: List[int] = [0]
        # per shard: (samples, frames capped at max_len, raw frames) — frame_stats()
        self._frame_sums: List[Tuple[int, int, int]] = []
        # every sample's frame count (int32, shard-major: 4 B per sample — numpy, so
        # forked DataLoader workers share it): the whole-record batch sampler
        parts: List[np.ndarray] = []
        self.converter_version: Optional[int] = None
        self.shard_meta: Dict = {}
        self._ref_path: Optional[str] = None
        self._obs_keys: Optional[frozenset] = None
        self._tgt_keys: Optional[frozenset] = None
        self.seat_split = False
        for si, p in enumerate(self.paths):
            shard = self._open(p, header_only=True)
            self._check(shard, p)
            if si == 0:        # one decision per dataset (_check: one meta, one game)
                self.seat_split = (seat_split if seat_split is not None else
                                   shard["game"] in SEAT_SPLIT_GAMES
                                   and (shard.get("meta") or {}).get("layout") != "per_seat")
            lengths = np.asarray(shard["_lengths"] if "_lengths" in shard
                                 else [next(iter(x["obs"].values())).shape[0]
                                       for x in shard["samples"]], dtype=np.int32)
            if self.seat_split:
                lengths = seat_stream_lengths(lengths)
            self._offsets.append(self._offsets[-1] + len(lengths))
            capped = (int(np.minimum(lengths, max_len).sum()) if max_len is not None
                      else int(lengths.sum()))
            self._frame_sums.append((len(lengths), capped, int(lengths.sum())))
            parts.append(lengths)
        self._lengths = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
        self._cache_idx: int = -1
        self._cache: Optional[Dict] = None

    @staticmethod
    def _open(path: str, header_only: bool = False) -> Dict:
        """Shard dict for a .pt file or a gm-shard directory. ``header_only``
        returns keys/meta/counts without materialising gm-shard samples."""
        if os.path.isdir(path):
            if header_only:         # the sidecars when fresh: no per-sample index parsed
                index, lengths = shard_header(path)
                game = index["game"]
                head = {"format_version": FORMAT_VERSION, "game": game,
                        "converter_version": index.get("converter_version", 1),
                        "meta": index.get("meta") or {}, "n_samples": int(index["n_samples"])}
                names = list(index["tensors"])
                obs_keys = frozenset(n.split(".", 1)[1] for n in names if n.startswith("obs."))
                exp = _expander(game)
                if exp is not None:
                    raw_names = [n.split(".", 1)[1] for n in names if n.startswith("raw.")]
                    obs_keys = obs_keys | exp[1](raw_names)
                head["_keys"] = (obs_keys, frozenset(n.split(".", 1)[1] for n in names if n.startswith("tgt.")))
                head["_lengths"] = lengths
                return head
            sh = GmShard(path)
            return {"format_version": FORMAT_VERSION, "game": sh.game,
                    "converter_version": sh.index.get("converter_version", 1),
                    "meta": sh.index.get("meta") or {}, "n_samples": len(sh), "_gm": sh}
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
        if schema_meta(meta) != schema_meta(self.shard_meta):
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
        return self._offsets[-1]

    def shard_range(self, si: int) -> range:
        """The global sample indices of shard ``si`` (contiguous)."""
        return range(self._offsets[si], self._offsets[si + 1])

    def locate(self, idx: int) -> Tuple[int, int]:
        """Global sample index → (shard index, sample index within the shard)."""
        if not 0 <= idx < len(self):
            raise IndexError(f"sample {idx} out of range for {len(self)} samples")
        si = bisect.bisect_right(self._offsets, idx) - 1
        return si, idx - self._offsets[si]

    def frame_stats(self) -> Dict[str, int]:
        """``samples``, ``frames`` (each sample capped at max_len: what the
        dataset supervises per draw at most) and ``raw_frames`` (uncapped) —
        summed while the index was built (no second pass over the shards)."""
        n, capped, raw = (sum(x) for x in zip(*self._frame_sums)) if self._frame_sums else (0, 0, 0)
        return {"samples": n, "frames": capped, "raw_frames": raw}

    def total_frames(self) -> int:
        """Frames the dataset supervises (each sample capped at max_len)."""
        return self.frame_stats()["frames"]

    def sample_lengths(self) -> np.ndarray:
        """Every sample's frame count as served (capped at max_len), shard-major."""
        if self.max_len is None:
            return self._lengths
        return np.minimum(self._lengths, self.max_len)

    def __getitem__(self, idx: int) -> Dict:
        si, j = self.locate(idx)
        if si != self._cache_idx:
            self._cache = self._open(self.paths[si])
            self._cache_idx = si
        rec = j // 2 if self.seat_split else j
        sample = (self._cache["_gm"].sample(rec) if "_gm" in self._cache
                  else self._cache["samples"][rec])
        if self.seat_split:
            sample = seat_stream(sample, j % 2)
        if self.max_len is not None:
            T = next(iter(sample["obs"].values())).shape[0]
            if T > self.max_len:
                L = self.max_len
                if self.crop == "prefix":
                    t0 = 0
                elif self.crop == "balanced":
                    # edge-balanced start (class docstring): clip a start drawn with
                    # L − 1 frames of slack on both sides to a full window
                    u = int(torch.randint(-(L - 1), T, ()).item())
                    t0 = min(max(u, 0), T - L)
                else:
                    t0 = int(torch.randint(0, T - L + 1, ()).item())
                sample = {
                    "obs": {k: v[t0: t0 + self.max_len] for k, v in sample["obs"].items()},
                    "tgt": {k: v[t0: t0 + self.max_len] for k, v in sample["tgt"].items()},
                    "meta": sample.get("meta", {}),
                }
        return sample


#: ``meta["teacher"]`` fields (gmdata label) that record HOW a teacher was run, not
#: what its columns mean: shards labelled with another search budget, frame
#: stride, engine build or options share the tensors and their semantics (the
#: heads / K, the policy and its temperature, the value definition stay checked).
TEACHER_PROVENANCE = ("engine", "engine_options", "every_nth_frame", "limits", "max_samples")


def schema_meta(meta: Dict) -> Dict:
    """The part of a shard's meta every shard of one dataset must share: all of
    it except the teacher's provenance (:data:`TEACHER_PROVENANCE`)."""
    teacher = meta.get("teacher")
    if not isinstance(teacher, dict):
        return meta
    return {**meta, "teacher": {k: v for k, v in teacher.items()
                                if k not in TEACHER_PROVENANCE}}


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

    def __iter__(self):
        shard_order = torch.randperm(len(self.dataset.paths)).tolist()
        for si in shard_order:
            idxs = self.dataset.shard_range(si)     # contiguous (shard-major numbering)
            if not idxs:
                continue
            for j in torch.randperm(len(idxs)).tolist():
                yield idxs[j]

    def __len__(self) -> int:
        return len(self.dataset)


def whole_record_batches(lengths: np.ndarray, frames: int, max_batch: int,
                         order: np.ndarray) -> List[List[int]]:
    """Cut the records ``order`` (indices into ``lengths``, already shuffled) into
    batches of similar length: sorted by length (stable, so equal lengths keep the
    shuffled order), then greedily filled while records x the batch's longest length
    stays within ``frames`` (and at most ``max_batch`` records).  A record longer than
    ``frames`` gets a batch of its own."""
    srt = order[np.argsort(lengths[order], kind="stable")]
    out: List[List[int]] = []
    cur: List[int] = []
    for i in srt.tolist():
        L = int(lengths[i])                        # ascending: the batch's longest so far
        if cur and ((len(cur) + 1) * L > frames or len(cur) >= max_batch):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


class WholeRecordBatchSampler(torch.utils.data.Sampler):
    """Batches of WHOLE records (each from its first frame, up to the dataset's
    ``max_len`` with a prefix crop), sized by a frame budget: every trained frame
    carries its record's whole history in the recurrent state, as in play, and a
    batch's padded size (records x its longest record) stays within ``frames`` — short
    records many to a batch, long ones few.  Shard-local like
    :class:`ShardLocalitySampler` (the one-shard cache stays hot): shards in a random
    order, each shard's records shuffled and cut into similar-length batches
    (:func:`whole_record_batches`, little padding), a shard's batches in a random order.
    Randomness: the global torch RNG (per epoch)."""

    def __init__(self, dataset: "GameShardDataset", frames: int, max_batch: int = 1024):
        if frames <= 0:
            raise ValueError(f"frame budget must be positive, got {frames}")
        if dataset.max_len is not None and dataset.crop != "prefix":
            raise ValueError("whole-record batches need the full records or a prefix crop "
                             f"(crop={dataset.crop!r} with max_len={dataset.max_len} would cut "
                             "random windows)")
        self.dataset = dataset
        self.frames = int(frames)
        self.max_batch = int(max_batch)
        self.lengths = dataset.sample_lengths()
        self._n_batches: Optional[int] = None

    def _shard_batches(self, si: int) -> List[List[int]]:
        r = self.dataset.shard_range(si)
        if not len(r):
            return []
        order = np.asarray(r, dtype=np.int64)[torch.randperm(len(r)).numpy()]
        batches = whole_record_batches(self.lengths, self.frames, self.max_batch, order)
        return [batches[b] for b in torch.randperm(len(batches)).tolist()]

    def __iter__(self) -> Iterator[List[int]]:
        for si in torch.randperm(len(self.dataset.paths)).tolist():
            yield from self._shard_batches(si)

    def __len__(self) -> int:
        """Batches per epoch (the bucketing of one epoch; within a few of any other)."""
        if self._n_batches is None:
            n = 0
            for si in range(len(self.dataset.paths)):
                r = self.dataset.shard_range(si)
                n += len(whole_record_batches(self.lengths, self.frames, self.max_batch,
                                              np.asarray(r, dtype=np.int64)))
            self._n_batches = n
        return self._n_batches

    def records_per_batch(self) -> float:
        """Mean records per batch over an epoch (the mixture's per-draw sample count)."""
        return len(self.dataset) / max(len(self), 1)


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
