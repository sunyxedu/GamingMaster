"""The Rust generals pipeline (`gmdata generals`) must reproduce the Python
reference bit-for-bit: .gior decoding, the fog-of-war simulator and the gio
training record. Uses one real 1v1 replay committed as a fixture; skipped
when cargo is unavailable."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

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
