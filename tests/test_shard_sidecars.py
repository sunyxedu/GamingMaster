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
    with open(os.path.join(d, "index.json"), "w") as fh:     # after scalars: still the head
        fh.write('{"converter_version":2,"format":"gm-shard.v1","game":"toy"}')
    assert _is_gm_shard(d)
    with open(os.path.join(d, "index.json"), "w") as fh:     # after a nested value: full parse
        json.dump({"meta": {"format": "x"}, "format": "gm-shard.v1"}, fh)
    assert _is_gm_shard(d)
    with open(os.path.join(d, "index.json"), "w") as fh:
        fh.write('{"format": "other"}')
    assert not _is_gm_shard(d)


def test_parallel_walk_finds_what_glob_finds(tmp_path, monkeypatch):
    """walk_data_root / find_gm_shards / GameShardDataset (directories listed and shard headers
    read on SHARD_IO_THREADS threads): the same shards, in the same order, as the one-at-a-time
    glob walk — symlinked shard directories followed (the frozen w* roots), hidden entries
    skipped, a root that is itself a shard included."""
    import glob

    from data_processing.games import dataset as D

    root = tmp_path / "root"
    for i, lengths in enumerate(([3, 1], [2], [4, 4, 1], [5])):
        _shard(str(tmp_path / "store" / f"s{i}"), lengths)
    _shard(str(root / "a" / "x"), [2, 2])
    _shard(str(root / "b"), [1])
    _shard(str(root / ".hidden" / "h"), [7])                    # glob's ** skips hidden entries
    (root / "frozen").mkdir()
    for i in (2, 0, 3):                                         # symlinks, names out of order
        os.symlink(tmp_path / "store" / f"s{i}", root / "frozen" / f"l{2 - i + 5}")
    (root / "notes.txt").write_text("not a shard")
    (root / "a" / "stray.json").write_text("{}")
    write_shard_sidecars(str(root / "b"))
    want_idx = sorted(glob.glob(os.path.join(str(root), "**", "index.json"), recursive=True))
    want_pt = sorted(glob.glob(os.path.join(str(root), "**", "*.pt"), recursive=True))
    for threads in (1, 16):
        monkeypatch.setattr(D, "SHARD_IO_THREADS", threads)
        pts, idx = D.walk_data_root(str(root))
        assert (pts, idx) == (want_pt, want_idx) and len(idx) == 5
        assert D.find_gm_shards(str(root)) == sorted(os.path.dirname(p) for p in want_idx)
        ds = D.GameShardDataset(str(root), game="toy")
        assert ds.sample_lengths().tolist() == [2, 2, 1, 5, 4, 4, 1, 3, 1]
        assert [ds[i]["obs"]["x"].tolist() for i in (0, 3, 8)] == [
            [[0.0], [1.0]], [[0.0], [1.0], [2.0], [3.0], [4.0]], [[3.0]]]
    one = tmp_path / "one"
    _shard(str(one), [2])                                        # the root itself is a shard
    assert D.find_gm_shards(str(one)) == [str(one)]
    assert D.walk_data_root(str(tmp_path / "missing")) == ([], [])
