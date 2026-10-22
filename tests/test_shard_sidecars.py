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


def test_data_manifest_opens_the_same_dataset_and_goes_stale(tmp_path, monkeypatch):
    """write_data_manifest / load_data_manifest: the dataset built from the manifest equals the
    walked one (paths, lengths, frame stats, samples; mixed per-shard metas kept), without
    reading any shard; adding or removing a shard makes it stale (the walk again, the new shard
    included); GM_DATA_MANIFEST=0 ignores it; a root with .pt shards gets none."""
    import json

    from data_processing.games import dataset as D

    root = tmp_path / "root"
    _shard(str(root / "a" / "x"), [2, 3])
    _shard(str(root / "a" / "y"), [1])
    _shard(str(root / "b"), [4, 1, 1])
    (root / "frozen").mkdir()
    _shard(str(tmp_path / "store" / "s0"), [5])
    os.symlink(tmp_path / "store" / "s0", root / "frozen" / "l0")
    for d, eng in (("a/x", "x"), ("a/y", "x"), ("b", "y"), ("frozen/l0", "x")):   # provenance-only
        idx = json.loads((root / d / "index.json").read_text())                    # meta differences
        idx["meta"] = {"k": 1, "teacher": {"kind": "soft", "engine": eng}}
        (root / d / "index.json").write_text(json.dumps(idx))
    walked = GameShardDataset(str(root), game="toy")
    assert not walked.from_manifest
    out = D.write_data_manifest(str(root))
    assert out == str(root / D.MANIFEST)
    # from the manifest: no shard is opened
    opened = []
    real_open = D.GameShardDataset._open
    monkeypatch.setattr(D.GameShardDataset, "_open", staticmethod(
        lambda p, header_only=False: opened.append(p) or real_open(p, header_only)))
    fast = GameShardDataset(str(root), game="toy")
    assert fast.from_manifest and opened == []
    assert fast.paths == walked.paths
    assert fast.sample_lengths().tolist() == walked.sample_lengths().tolist() == [2, 3, 1, 4, 1, 1, 5]
    assert fast.frame_stats() == walked.frame_stats()
    assert fast.shard_meta == walked.shard_meta
    import numpy as _np
    z = _np.load(out)
    assert len(z["headers"]) == 2                                     # both metas recorded
    monkeypatch.setattr(D.GameShardDataset, "_open", staticmethod(real_open))
    for i in range(7):
        assert fast[i]["obs"]["x"].tolist() == walked[i]["obs"]["x"].tolist()
    # disabled
    monkeypatch.setenv("GM_DATA_MANIFEST", "0")
    assert not GameShardDataset(str(root), game="toy").from_manifest
    monkeypatch.delenv("GM_DATA_MANIFEST")
    # a new shard (anywhere in the tree) makes it stale and is found by the walk
    _shard(str(root / "a" / "z"), [7])
    idx = json.loads((root / "a" / "z" / "index.json").read_text())
    idx["meta"] = {"k": 1, "teacher": {"kind": "soft", "engine": "x"}}
    (root / "a" / "z" / "index.json").write_text(json.dumps(idx))
    again = GameShardDataset(str(root), game="toy")
    assert not again.from_manifest and again.sample_lengths().tolist() == [2, 3, 1, 7, 4, 1, 1, 5]
    D.write_data_manifest(str(root))
    assert GameShardDataset(str(root), game="toy").from_manifest
    import shutil
    shutil.rmtree(root / "a" / "y")                                   # a removed shard: stale
    assert D.load_data_manifest(str(root)) is None
    # .pt shards: no manifest
    pt_root = tmp_path / "pt"
    pt_root.mkdir()
    (pt_root / "s.pt").write_bytes(b"")
    _shard(str(pt_root / "g"), [1])
    assert D.write_data_manifest(str(pt_root)) is None

