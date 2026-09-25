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
