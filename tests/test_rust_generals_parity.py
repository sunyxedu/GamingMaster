"""The Rust generals pipeline (`gmdata generals`) must reproduce the Python
reference bit-for-bit: .gior decoding, the fog-of-war simulator and the gio
training record — and `gmdata generals shard` must hold exactly the tensors
SeqReplayDataset decodes from those records. Uses one real 1v1 replay
committed as a fixture; skipped when cargo is unavailable."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "generals", "TyvMRVMys.gior")


def _cmp(a, b, path="", out=None):
    out = [] if out is None else out
    if len(out) > 5:
        return out
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            _cmp(a.get(k), b.get(k), path + "." + k, out)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            out.append((path, "len", len(a), b if not isinstance(b, list) else len(b)))
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > 1e-9 * max(1.0, abs(a)):
            out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))
    return out


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    build = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                           capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"cargo build failed: {build.stderr[-500:]}")
    return os.path.join(WS, "target", "release", "gmdata")


def test_decode_matches_python(gmdata, tmp_path):
    from data_processing.download.deserializer import deserialize_replay
    py = deserialize_replay(open(FIXTURE, "rb").read())
    rs = json.loads(subprocess.run([gmdata, "generals", "decode", FIXTURE], capture_output=True,
                                   text=True, check=True).stdout)
    assert _cmp(py, rs) == []
    json.dump(py, open(tmp_path / "replay.json", "w"))


def test_simulate_and_convert_match_python(gmdata, tmp_path):
    from data_processing.download.deserializer import deserialize_replay
    from data_processing.parsers.generals_json import convert_simulator_to_gio
    from data_processing.processing.simulator import ReplaySimulator
    replay = deserialize_replay(open(FIXTURE, "rb").read())
    rp = tmp_path / "replay.json"
    json.dump(replay, open(rp, "w"))
    for player in replay["usernames"]:
        sim = ReplaySimulator(replay, player).simulate()
        gio = convert_simulator_to_gio(sim)
        subprocess.run([gmdata, "generals", "simulate", "--replay", rp, "--player", player,
                        "--out", tmp_path / "sim.json"], check=True)
        subprocess.run([gmdata, "generals", "convert", "--replay", rp, "--player", player,
                        "--out", tmp_path / "gio.json"], check=True)
        assert _cmp(sim, json.load(open(tmp_path / "sim.json"))) == []
        assert _cmp(gio, json.load(open(tmp_path / "gio.json"))) == []
        assert len(gio["count"]) == len(sim["first"]) - 1 > 50


def test_rust_rejects_unsupported_replays(gmdata, tmp_path):
    from data_processing.download.deserializer import deserialize_replay
    replay = deserialize_replay(open(FIXTURE, "rb").read())
    replay["modifiers"] = [11]
    rp = tmp_path / "mod.json"
    json.dump(replay, open(rp, "w"))
    res = subprocess.run([gmdata, "generals", "convert", "--replay", rp, "--player", replay["usernames"][0],
                          "--out", tmp_path / "x.json"], capture_output=True, text=True)
    assert res.returncode != 0 and "unsupported modifiers" in res.stderr


def _assert_same_samples(a, b):
    assert a["meta"]["file"] == b["meta"]["file"]
    for grp in ("obs", "tgt"):
        assert a[grp].keys() == b[grp].keys(), grp
        for k in a[grp]:
            x, y = a[grp][k], b[grp][k]
            assert x.dtype == y.dtype and x.shape == y.shape, (k, x.dtype, y.dtype)
            assert torch.equal(x, y), k


def test_shard_matches_seq_replay_dataset(gmdata, tmp_path):
    """`gmdata generals shard` gm-shards hold exactly the tensors SeqReplayDataset
    decodes from the gio records (the multi-game trainers' two generals sources)."""
    from data_processing.dataset_seq import SeqReplayDataset
    from data_processing.download.deserializer import deserialize_replay
    from data_processing.games.dataset import GameShardDataset, open_game_dataset
    from data_processing.games.generals import GeneralsGioDataset

    replay = deserialize_replay(open(FIXTURE, "rb").read())
    rp = tmp_path / "replay.json"
    json.dump(replay, open(rp, "w"))
    gio_dir = tmp_path / "gio"
    gio_dir.mkdir()
    for i, player in enumerate(replay["usernames"]):
        subprocess.run([gmdata, "generals", "convert", "--replay", rp, "--player", player,
                        "--out", gio_dir / f"r{i}.json"], check=True)
    res = subprocess.run([gmdata, "generals", "shard", "--input", gio_dir, "--out",
                          tmp_path / "shards", "--shard-size", "1"],
                         check=True, capture_output=True, text=True)
    summary = json.loads(res.stdout)
    assert summary["converted"] == 2 and summary["failed"] == 0 and summary["shards"] == 2

    shards = open_game_dataset(str(tmp_path / "shards"), "generals")
    gio = open_game_dataset(str(gio_dir), "generals")
    assert isinstance(shards, GameShardDataset) and isinstance(gio, GeneralsGioDataset)
    assert len(shards) == len(gio) == 2
    assert shards.shard_meta == gio.shard_meta == {
        "grid_h": 25, "grid_w": 25, "grid_channels": 12, "global_features": 4,
        "max_actions_per_tick": 4}
    assert shards.total_frames() == gio.total_frames() == summary["frames"]
    seq = SeqReplayDataset(str(gio_dir))
    for i in range(2):
        a = shards[i]
        _assert_same_samples(a, gio[i])
        item = seq[i]                           # the train_il.py decoding itself
        for grp, keys in (("obs", ("grid_seq", "globals_seq", "legal_source_mask_seq")),
                          ("tgt", ("count", "src_y", "src_x", "dir_id", "ratio_id", "value"))):
            for k in keys:
                assert torch.equal(a[grp][k], item[grp][k][0]), k
    # prefix truncation agrees too
    s8 = GameShardDataset(str(tmp_path / "shards"), game="generals", max_len=8, crop="prefix")
    g8 = GeneralsGioDataset(str(gio_dir), max_len=8)
    for i in range(2):
        _assert_same_samples(s8[i], g8[i])
    assert s8.total_frames() == g8.total_frames() == 16
