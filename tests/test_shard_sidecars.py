"""Start-up sidecars of gm-shards (index.head.json + index.lengths.npy): the dataset built
from them equals the one built from index.json, stale sidecars are ignored, and the format
check reads only the head of index.json."""
import json
import os
import time

import numpy as np

from data_processing.games.dataset import (HEAD_SIDECAR, LENGTHS_SIDECAR, GameShardDataset,
                                           _is_gm_shard, shard_header, write_shard_sidecars)


def _shard(d, lengths):
    os.makedirs(d)
    n = int(sum(lengths))
    np.save(os.path.join(d, "obs.x.npy"), np.arange(n, dtype=np.float32).reshape(n, 1))
    np.save(os.path.join(d, "tgt.action.npy"), np.zeros(n, dtype=np.int64))
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    index = {"format": "gm-shard.v1", "game": "toy", "converter_version": 1, "meta": {"k": 1},
             "n_frames": n,
             "tensors": {"obs.x": {"file": "obs.x.npy", "dtype": "float32", "frame_shape": [1]},
                         "tgt.action": {"file": "tgt.action.npy", "dtype": "int64", "frame_shape": []}},
             "samples": [{"start": s, "length": L, "meta": {"i": i}}
                         for i, (s, L) in enumerate(zip(starts, lengths))]}
    with open(os.path.join(d, "index.json"), "w") as fh:
        json.dump(index, fh)


def test_sidecars_give_the_same_dataset(tmp_path):
    _shard(str(tmp_path / "a"), [3, 1, 4])
    _shard(str(tmp_path / "b" / "c"), [2, 5])
    plain = GameShardDataset(str(tmp_path), game="toy")
    for d in (tmp_path / "a", tmp_path / "b" / "c"):
        assert write_shard_sidecars(str(d)) in (3, 2)
        assert (d / HEAD_SIDECAR).exists() and (d / LENGTHS_SIDECAR).exists()
    fast = GameShardDataset(str(tmp_path), game="toy")
    assert len(fast) == len(plain) == 5
    assert fast.sample_lengths().tolist() == plain.sample_lengths().tolist() == [3, 1, 4, 2, 5]
    assert fast.frame_stats() == plain.frame_stats()
    for i in range(5):
        assert fast[i]["obs"]["x"].tolist() == plain[i]["obs"]["x"].tolist()


def test_a_stale_sidecar_is_ignored(tmp_path):
    d = str(tmp_path / "a")
    _shard(d, [3, 1])
    write_shard_sidecars(d)
    old = time.time() - 100
    for f in (HEAD_SIDECAR, LENGTHS_SIDECAR):
        os.utime(os.path.join(d, f), (old, old))
    np.save(os.path.join(d, LENGTHS_SIDECAR), np.array([9, 9], dtype=np.int32))   # wrong ...
    os.utime(os.path.join(d, LENGTHS_SIDECAR), (old, old))                       # ... and stale
    head, lengths = shard_header(d)
    assert lengths.tolist() == [3, 1] and head["n_samples"] == 2 and "samples" not in head


def test_format_check_reads_the_head(tmp_path):
    d = str(tmp_path / "a")
    _shard(d, [2])
    assert _is_gm_shard(d)
    with open(os.path.join(d, "index.json"), "w") as fh:     # format not first: full parse
        json.dump({"game": "toy", "format": "gm-shard.v1"}, fh)
    assert _is_gm_shard(d)
    with open(os.path.join(d, "index.json"), "w") as fh:
        fh.write('{"format": "other"}')
    assert not _is_gm_shard(d)
