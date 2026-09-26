"""generals.io training data in the multi-game sample format.

generals records are the per-game ``gio`` JSON files of the Rust pipeline
(``gmdata generals pipeline|ladder``), decoded by
``data_processing.dataset_seq.SeqReplayDataset`` for training/train_il.py.
The multi-game trainers (training/train_multi_game.py, train_il_game.py)
consume GameShardDataset samples instead —
``{"obs": {k: [T, ...]}, "tgt": {k: [T, ...]}, "meta": {...}}`` — and two
sources produce them, tensor-for-tensor identical:

  gio directories   :class:`GeneralsGioDataset`: SeqReplayDataset's
                    decoding of every ``*.json`` under the root (no
                    conversion step, but ~0.5 ms of JSON parsing per frame
                    whatever ``max_len`` is — give the DataLoader workers —
                    and a first-run scan that reads every record for its
                    length).  Small corpora and quick experiments.
  gm-shard dirs     ``gmdata generals shard --input <gio dir> --out <dir>``
                    (Rust): the same tensors bit-packed and memory-mapped,
                    ~10x smaller than the JSON and ~200x faster to load,
                    read by GameShardDataset with the load-time
                    :func:`expand` below; records are shuffled across shards
                    (``--seed``).  The format for real corpora
                    (tests/test_rust_generals_parity.py pins it to
                    SeqReplayDataset).

Sample schema (both sources):

  obs.grid_seq               float32 [T, 12, H, W]  11 one-hot tile states + log1p(army)/10
  obs.globals_seq            float32 [T, G]         log1p(own/opp army, land)/10 (G = 4)
  obs.legal_source_mask_seq  bool    [T, 1, H, W]   own tiles with army > 1
  tgt.count                  int64   [T]            1 move, 0 pass, -1 move dropped by the server
  tgt.src_y / src_x / dir_id / ratio_id  int64 [T, K]  the move in slot 0 (K = max_actions_per_tick)
  tgt.value                  float32 [T]            outcome / army-ratio heuristic blend in [-1, 1]
  meta.file                  record file name

Left out: the gio placeholders ``enemy_heatmap`` / ``next_visibility`` (all
zero) and the tick features (equal to the frame index here).

Frames keep their absolute time: the pass head reads the frame index
(``GameNetwork.forward`` decodes at t0 = 0), ``turn_globals`` are derived
from it and the fog-of-war memory needs the game from its first frame.  So
``max_len`` keeps the FIRST ``max_len`` frames (SeqReplayDataset's
``max_sequence_length``, what train_il.py's ``--max_seq_len`` does) — never
a random crop.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data_processing.dataset_seq import SeqReplayDataset

#: obs keys :func:`expand` builds from a gm-shard's ``raw.*`` tensors
OBS_KEYS = ("grid_seq", "legal_source_mask_seq")
#: the sample's obs / tgt keys (both sources)
SAMPLE_OBS_KEYS = ("grid_seq", "globals_seq", "legal_source_mask_seq")
SAMPLE_TGT_KEYS = ("count", "src_y", "src_x", "dir_id", "ratio_id", "value")
#: per-directory cache of record lengths (deliberately not ``*.json``:
#: SeqReplayDataset / train_il.py list every ``*.json`` of a directory)
LENGTH_CACHE = ".gm_gio_lengths"
#: records above which :func:`gio_dataset_advice` recommends gm-shards
LARGE_GIO_DATASET = 1000
_CHUNK = 1 << 20


def expand(raw: Dict[str, torch.Tensor], meta: Dict) -> Dict[str, torch.Tensor]:
    """gm-shard ``raw.*`` → obs (GameShardDataset's load-time expander):
    ``grid_onehot`` [T, 11, H, W] + ``army`` [T, H, W] → ``grid_seq``,
    ``legal_source_mask`` (bits) → bool ``legal_source_mask_seq``."""
    grid = torch.cat([raw["grid_onehot"], raw["army"].unsqueeze(1)], dim=1)
    return {"grid_seq": grid, "legal_source_mask_seq": raw["legal_source_mask"].bool()}


def sample_from_seq_item(item: Dict) -> Dict:
    """One ``SeqReplayDataset`` item ([1, T, ...] tensors) → one sample."""
    return {"obs": {k: item["obs"][k][0] for k in SAMPLE_OBS_KEYS},
            "tgt": {k: item["tgt"][k][0] for k in SAMPLE_TGT_KEYS},
            "meta": {"file": item["meta"]["file"]}}


def find_gio_files(root: str) -> List[str]:
    """Every ``*.json`` under ``root`` (recursive; e.g. the pipeline's
    ``training/<player>/<id>.json``), ordered by path components — the order
    ``gmdata generals shard`` writes them in."""
    out = []
    for d, _dirs, files in os.walk(root):
        out.extend(os.path.join(d, f) for f in files if f.endswith(".json"))
    return sorted(out, key=lambda p: os.path.relpath(p, root).split(os.sep))


def _rfind_in_file(f, size: int, needle: bytes) -> int:
    """Offset of the last ``needle`` in the open binary file ``f`` (reading
    backwards in chunks), -1 if absent."""
    pos, carry = size, b""
    while pos > 0:
        step = min(_CHUNK, pos)
        pos -= step
        f.seek(pos)
        blk = f.read(step) + carry
        i = blk.rfind(needle)
        if i >= 0:
            return pos + i
        carry = blk[:len(needle) - 1]
    return -1


def record_length(path: str) -> int:
    """Frames of one gio record without decoding its tensors: the length of
    its ``count`` array (one entry per frame), else of ``grid_seq``.
    0 for an unreadable record (SeqReplayDataset substitutes another one).

    The converter writes ``count`` after the grids (~87% into the file), so
    the key is searched from the END: only the tail (the per-frame targets
    and placeholders, ~13%) is read, not the whole record."""
    try:
        with open(path, "rb") as f:
            size = f.seek(0, os.SEEK_END)
            o = _rfind_in_file(f, size, b'"count"')
            if o >= 0:
                f.seek(o)
                head = b""
                while b"]" not in head:
                    blk = f.read(_CHUNK)
                    if not blk:
                        raise ValueError("unterminated count array")
                    head += blk
                a = head.index(b"[")
                return len(json.loads(head[a:head.index(b"]", a) + 1]))
            f.seek(0)
            return len(json.loads(f.read())["grid_seq"])
    except (OSError, ValueError, KeyError, UnicodeDecodeError):
        return 0


def gio_dataset_advice(n_records: int, num_workers: int) -> Optional[str]:
    """A warning for training on gio JSON records at a scale or with a
    DataLoader setup where that is slow (None when fine)."""
    tips = []
    if n_records >= LARGE_GIO_DATASET:
        tips.append(f"{n_records} gio JSON records: every draw parses a whole record "
                    f"(~0.1 s per 150 frames) and the first run reads every record for "
                    f"its length — convert them once with `gmdata generals shard --input "
                    f"<gio dir> --out <shard dir>` (~10x smaller, ~200x faster to load) "
                    f"and train on the shard directory")
    if num_workers <= 0:
        tips.append("gio records decode on the training process (--num_workers 0): "
                    "give the DataLoader workers, or use `gmdata generals shard` shards")
    return "; ".join(tips) or None


def _user_cache_path(root: str) -> str:
    """Fallback length-cache location for a read-only data directory:
    ``$GM_CACHE_DIR`` (default ``$XDG_CACHE_HOME/gamingmaster``, i.e.
    ``~/.cache/gamingmaster``) / gio_lengths / sha1(realpath(root))."""
    base = os.environ.get("GM_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
        "gamingmaster")
    key = hashlib.sha1(os.path.realpath(root).encode()).hexdigest()
    return os.path.join(base, "gio_lengths", key)


class _GioRecords(SeqReplayDataset):
    """SeqReplayDataset over an explicit file list.  :meth:`sample` is its
    ``__getitem__`` decoding (the one train_il.py trains on, prefix
    truncation and absent-field defaults included) producing only the
    sample's tensors: the lists are cut to the first ``max_seq_len`` frames
    BEFORE tensor construction and the placeholders / tick features are
    skipped — tensor-for-tensor identical (tests/test_generals_multigame.py),
    about a third cheaper; the JSON parse itself remains."""

    def __init__(self, paths: List[str], max_sequence_length: Optional[int] = None):
        self.root = None
        self.max_seq_len = max_sequence_length
        self.paths = list(paths)

    def sample(self, idx: int) -> Dict:
        try:
            with open(self.paths[idx], "r") as f:
                obj = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # SeqReplayDataset's policy: a corrupt record is replaced by a random one
            return self.sample(torch.randint(len(self), (1,)).item())
        grid_rows = obj["grid_seq"]
        T = len(grid_rows)
        if self.max_seq_len and T > self.max_seq_len:
            T = self.max_seq_len
        grid = torch.tensor(grid_rows[:T], dtype=torch.float32)
        _, _, H, W = grid.shape
        K = obj.get("max_actions_per_tick", 4)

        def rows(key, default, dtype):
            v = obj.get(key)
            return torch.tensor(v[:T] if v is not None else default(), dtype=dtype)

        def slots():
            return [[0] * K for _ in range(T)]

        return {
            "obs": {
                "grid_seq": grid,
                "globals_seq": rows("globals_seq", lambda: [[0.0] * 4 for _ in range(T)],
                                    torch.float32),
                "legal_source_mask_seq": rows(
                    "legal_source_mask_seq",
                    lambda: [[[[True] * W for _ in range(H)]] for _ in range(T)], torch.bool),
            },
            "tgt": {
                "count": rows("count", lambda: [0] * T, torch.long),
                "src_y": rows("src_y", slots, torch.long),
                "src_x": rows("src_x", slots, torch.long),
                "dir_id": rows("dir_id", slots, torch.long),
                "ratio_id": rows("ratio_id", slots, torch.long),
                "value": rows("value", lambda: [0.0] * T, torch.float32),
            },
            "meta": {"file": os.path.basename(self.paths[idx])},
        }


class GeneralsGioDataset(Dataset):
    """GameShardDataset-format samples straight from a directory of gio
    records (searched recursively), one sample per record.

    ``max_len`` keeps the first ``max_len`` frames of a record (see the
    module docstring).  :attr:`shard_meta` (grid size, channels, globals,
    actions per tick; from the first record) plays the role of a gm-shard's
    index meta for META_ARCH_KEYS; :meth:`total_frames` reads every
    record's length once and caches it in ``<root>/.gm_gio_lengths`` (keyed
    by path, size and mtime), or under the user cache directory when the
    data directory is read-only (:func:`_user_cache_path`).  Data-parallel
    trainers split that scan over the ranks (:meth:`scan_lengths` +
    :meth:`adopt_lengths`).
    """

    #: sample format generation, compared like a shard's converter_version
    converter_version = 1

    def __init__(self, root: str, max_len: Optional[int] = None):
        self.root = root
        self.max_len = max_len
        self.paths = find_gio_files(root)
        if not self.paths:
            raise FileNotFoundError(f"no gio records (*.json) under {root}")
        self._records = _GioRecords(self.paths, max_sequence_length=max_len)
        self._lengths: Optional[List[int]] = None
        self._meta: Optional[Dict] = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        return self._records.sample(idx)

    @property
    def shard_meta(self) -> Dict:
        if self._meta is None:
            s = self[0]
            _, C, H, W = s["obs"]["grid_seq"].shape
            self._meta = {"grid_h": H, "grid_w": W, "grid_channels": C,
                          "global_features": s["obs"]["globals_seq"].shape[-1],
                          "max_actions_per_tick": s["tgt"]["src_y"].shape[-1]}
        return self._meta

    def _cache_paths(self) -> List[str]:
        return [os.path.join(self.root, LENGTH_CACHE), _user_cache_path(self.root)]

    def _load_cache(self) -> Dict[str, list]:
        for path in self._cache_paths():
            try:
                with open(path) as f:
                    cache = json.load(f)
                if isinstance(cache, dict):
                    return cache
            except (OSError, ValueError):
                continue
        return {}

    def scan_lengths(self, rank: int = 0, world: int = 1) -> Dict[int, list]:
        """``{idx: [size, mtime_ns, frames]}`` for records ``rank::world``:
        cache hits, else read from the record (:func:`record_length`)."""
        cache = self._load_cache()
        out = {}
        for i in range(rank, len(self.paths), world):
            p = self.paths[i]
            st = os.stat(p)
            key = [st.st_size, st.st_mtime_ns]
            hit = cache.get(os.path.relpath(p, self.root))
            if isinstance(hit, list) and len(hit) == 3 and hit[:2] == key:
                out[i] = key + [int(hit[2])]
            else:
                out[i] = key + [record_length(p)]
        return out

    def adopt_lengths(self, entries: Dict[int, list], write_cache: bool = True) -> None:
        """Take every record's ``[size, mtime_ns, frames]`` (the union of
        :meth:`scan_lengths` over all ranks) and, with ``write_cache``,
        store them when they differ from the cache on disk: atomically,
        through a uniquely named temporary file (ranks of several nodes may
        share the filesystem), in the data directory or — read-only — the
        user cache directory."""
        missing = [i for i in range(len(self.paths)) if i not in entries]
        if missing:
            raise ValueError(f"lengths of {len(missing)} records missing (e.g. {missing[:3]})")
        self._lengths = [int(entries[i][2]) for i in range(len(self.paths))]
        if not write_cache:
            return
        cache = {os.path.relpath(p, self.root): list(entries[i])
                 for i, p in enumerate(self.paths)}
        if cache == self._load_cache():
            return
        for path in self._cache_paths():
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path),
                                           prefix=os.path.basename(path) + ".", suffix=".tmp")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(cache, f)
                    os.replace(tmp, path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                return
            except OSError:
                continue        # read-only location: try the next one

    def lengths(self) -> List[int]:
        """Full (untruncated) frame count of every record."""
        if self._lengths is None:
            self.adopt_lengths(self.scan_lengths())
        return self._lengths

    def total_frames(self) -> int:
        """Frames the dataset supervises (each record capped at max_len)."""
        return sum(min(n, self.max_len) if self.max_len is not None else n
                   for n in self.lengths())
