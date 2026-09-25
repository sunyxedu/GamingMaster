"""generals.io training data in the multi-game sample format.

generals records are the per-game ``gio`` JSON files of the Rust pipeline
(``gmdata generals pipeline|ladder``), decoded by
``data_processing.dataset_seq.SeqReplayDataset`` for training/train_il.py.
The multi-game trainers (training/train_multi_game.py, train_il_game.py)
consume GameShardDataset samples instead —
``{"obs": {k: [T, ...]}, "tgt": {k: [T, ...]}, "meta": {...}}`` — and two
sources produce them, tensor-for-tensor identical:

  gio directories   :class:`GeneralsGioDataset`: SeqReplayDataset's own
                    decoding of every ``*.json`` under the root (no
                    conversion step; parsing costs ~0.7 ms per frame, so give
                    the DataLoader workers).
  gm-shard dirs     ``gmdata generals shard --input <gio dir> --out <dir>``
                    (Rust): the same tensors bit-packed and memory-mapped,
                    ~10x smaller than the JSON, read by GameShardDataset with
                    the load-time :func:`expand` below.  Recommended for
                    large corpora (tests/test_rust_generals_parity.py pins
                    it to SeqReplayDataset).

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

import json
import os
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


def record_length(path: str) -> int:
    """Frames of one gio record without decoding its tensors: the length of
    its ``count`` array (one entry per frame), else of ``grid_seq``.
    0 for an unreadable record (SeqReplayDataset substitutes another one)."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        i = data.find(b'"count"')
        if i >= 0:
            a = data.index(b"[", i)
            return len(json.loads(data[a:data.index(b"]", a) + 1]))
        return len(json.loads(data)["grid_seq"])
    except (OSError, ValueError, KeyError, UnicodeDecodeError):
        return 0


class _GioRecords(SeqReplayDataset):
    """SeqReplayDataset over an explicit file list.  ``__getitem__`` — the
    exact decoding train_il.py trains on, prefix truncation included — is
    inherited unchanged."""

    def __init__(self, paths: List[str], max_sequence_length: Optional[int] = None):
        self.root = None
        self.max_seq_len = max_sequence_length
        self.paths = list(paths)


class GeneralsGioDataset(Dataset):
    """GameShardDataset-format samples straight from a directory of gio
    records (searched recursively), one sample per record.

    ``max_len`` keeps the first ``max_len`` frames of a record (see the
    module docstring).  :attr:`shard_meta` (grid size, channels, globals,
    actions per tick; from the first record) plays the role of a gm-shard's
    index meta for META_ARCH_KEYS; :meth:`total_frames` reads every
    record's length once and caches it in ``<root>/.gm_gio_lengths``
    (keyed by path, size and mtime) when the directory is writable.
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
        return sample_from_seq_item(self._records[idx])

    @property
    def shard_meta(self) -> Dict:
        if self._meta is None:
            s = self[0]
            _, C, H, W = s["obs"]["grid_seq"].shape
            self._meta = {"grid_h": H, "grid_w": W, "grid_channels": C,
                          "global_features": s["obs"]["globals_seq"].shape[-1],
                          "max_actions_per_tick": s["tgt"]["src_y"].shape[-1]}
        return self._meta

    def lengths(self) -> List[int]:
        """Full (untruncated) frame count of every record."""
        if self._lengths is not None:
            return self._lengths
        cache_path = os.path.join(self.root, LENGTH_CACHE)
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
        out, dirty = [], False
        for p in self.paths:
            rel = os.path.relpath(p, self.root)
            st = os.stat(p)
            key = [st.st_size, st.st_mtime_ns]
            hit = cache.get(rel)
            if isinstance(hit, list) and hit[:2] == key:
                out.append(int(hit[2]))
                continue
            n = record_length(p)
            cache[rel] = key + [n]
            out.append(n)
            dirty = True
        if dirty:
            try:
                tmp = f"{cache_path}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    json.dump(cache, f)
                os.replace(tmp, cache_path)
            except OSError:
                pass            # read-only data directory: recompute next time
        self._lengths = out
        return out

    def total_frames(self) -> int:
        """Frames the dataset supervises (each record capped at max_len)."""
        return sum(min(n, self.max_len) if self.max_len is not None else n
                   for n in self.lengths())
