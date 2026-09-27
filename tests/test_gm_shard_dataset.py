"""GameShardDataset reads gm-shard.v1 directories (written by the Rust converters)."""
import json
import os

import numpy as np
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, collate_game_sequences, load_gm_samples


def _write(root, name, game="toy", cv=2, meta=None, lengths=(3, 5), keys=("obs.x", "tgt.y", "tgt.b")):
    d = os.path.join(root, name)
    os.makedirs(d)
    n = sum(lengths)
    arrays = {"obs.x": np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2),
              "tgt.y": np.arange(n, dtype=np.int64), "tgt.b": (np.arange(n) % 2).astype(np.uint8)}
    tensors = {}
    for k in keys:
        np.save(os.path.join(d, k + ".npy"), arrays[k])
        tensors[k] = {"file": k + ".npy", "dtype": {"obs.x": "float32", "tgt.y": "int64", "tgt.b": "bool"}[k],
                      "frame_shape": list(arrays[k].shape[1:])}
    starts = np.cumsum((0,) + tuple(lengths[:-1]))
    json.dump({"format": "gm-shard.v1", "game": game, "converter_version": cv, "meta": meta or {"k": 1},
               "n_frames": n, "tensors": tensors,
               "samples": [{"start": int(s), "length": int(l), "meta": {"i": i}} for i, (s, l) in enumerate(zip(starts, lengths))]},
              open(os.path.join(d, "index.json"), "w"))


def test_reads_gm_shards(tmp_path):
    _write(str(tmp_path), "toy_00000")
    _write(str(tmp_path), "toy_00001", lengths=(2,))
    ds = GameShardDataset(str(tmp_path), game="toy")
    assert len(ds) == 3 and ds.converter_version == 2 and ds.shard_meta == {"k": 1}
    s = ds[1]
    assert s["obs"]["x"].shape == (5, 2, 2) and s["tgt"]["y"].tolist() == [3, 4, 5, 6, 7]
    assert s["tgt"]["b"].dtype == torch.bool and s["meta"] == {"i": 1}
    batch = collate_game_sequences([ds[0], ds[1], ds[2]])
    assert batch["obs"]["x"].shape == (3, 5, 2, 2)
    assert len(load_gm_samples(str(tmp_path))) == 3


def test_rejects_mixed_generations(tmp_path):
    _write(str(tmp_path), "toy_00000")
    _write(str(tmp_path), "toy_00001", cv=3)
    with pytest.raises(ValueError, match="converter_version"):
        GameShardDataset(str(tmp_path))
    _write(str(tmp_path / "b"), "toy_00000")
    _write(str(tmp_path / "b"), "toy_00001", keys=("obs.x", "tgt.y"))
    with pytest.raises(ValueError, match="keys"):
        GameShardDataset(str(tmp_path / "b"))


def test_teacher_provenance_may_differ_across_shards(tmp_path):
    """Shards labelled with another search budget / frame stride / engine build
    are one dataset; another teacher temperature or head layout is refused."""
    import pytest

    from data_processing.games.dataset import GameShardDataset, save_shard, schema_meta
    from tests.test_core_moe import _board_samples

    def teacher(**kw):
        base = {"engine": "E 1.0", "limits": {"nodes": "1000000"}, "every_nth_frame": 1,
                "policy": {"kind": "softmax", "tau": 0.025}, "heads": {"from": {"k": 8}}}
        return {"global_features": 4, "teacher": {**base, **kw}}

    save_shard(str(tmp_path / "a" / "a.pt"), "othello", _board_samples("othello", 2, 3),
               meta=teacher())
    save_shard(str(tmp_path / "a" / "b.pt"), "othello", _board_samples("othello", 2, 3),
               meta=teacher(engine="E 1.1", limits={"nodes": "250000"}, every_nth_frame=4))
    ds = GameShardDataset(str(tmp_path / "a"), game="othello")
    assert len(ds) == 4 and ds.shard_meta["teacher"]["every_nth_frame"] == 1
    assert schema_meta(teacher()) == schema_meta(teacher(every_nth_frame=4))
    for i, bad in enumerate((teacher(policy={"kind": "softmax", "tau": 0.05}),
                             teacher(heads={"from": {"k": 16}}))):
        d = tmp_path / f"bad{i}"
        save_shard(str(d / "a.pt"), "othello", _board_samples("othello", 2, 3), meta=teacher())
        save_shard(str(d / "b.pt"), "othello", _board_samples("othello", 2, 3), meta=bad)
        with pytest.raises(ValueError, match="different settings"):
            GameShardDataset(str(d), game="othello")
