"""Rust converters (`gmdata <game> convert`) must emit exactly the tensors the
Python reference converters emit, on committed real-data fixtures."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess

import pytest
import torch

from data_processing.games.dataset import GameShardDataset, load_gm_samples

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
FIX = os.path.join(ROOT, "tests", "fixtures")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS, capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return os.path.join(WS, "target", "release", "gmdata")


def _py_samples(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.pt"))):
        out += torch.load(p, weights_only=False)["samples"]
    return out


def assert_same(py_dir, rs_dir, meta_keys=()):
    a, b = _py_samples(py_dir), load_gm_samples(rs_dir)
    assert len(a) == len(b) > 0
    for i, (x, y) in enumerate(zip(a, b)):
        for g in ("obs", "tgt"):
            assert set(x[g]) == set(y[g])
            for k in x[g]:
                assert x[g][k].dtype == y[g][k].dtype, (i, g, k)
                assert torch.equal(x[g][k], y[g][k]), (i, g, k)
        for k in meta_keys:
            assert x["meta"].get(k) == y["meta"].get(k), (i, k)
    return len(a)


def test_othello(gmdata, tmp_path):
    from data_processing.games.othello.convert import convert
    convert(os.path.join(FIX, "othello"), str(tmp_path / "py"))
    subprocess.run([gmdata, "othello", "convert", "--input", os.path.join(FIX, "othello"), "--out", tmp_path / "rs"],
                   check=True, capture_output=True)
    n = assert_same(tmp_path / "py", tmp_path / "rs", ("game_index", "replayed_black", "n_pass_frames"))
    assert n > 300
    assert len(GameShardDataset(str(tmp_path / "rs"), game="othello")) == n


@pytest.mark.parametrize("rule,keep", [("freestyle", False), ("standard", True)])
def test_gomoku(gmdata, tmp_path, rule, keep):
    from data_processing.games.gomoku.convert import convert
    convert(os.path.join(FIX, "gomoku"), str(tmp_path / "py"), board_size=20, rule=rule, keep_undecided=keep)
    cmd = [gmdata, "gomoku", "convert", "--input", os.path.join(FIX, "gomoku"), "--out", tmp_path / "rs",
           "--board-size", "20", "--rule", rule] + (["--keep-undecided"] if keep else [])
    subprocess.run(cmd, check=True, capture_output=True)
    assert_same(tmp_path / "py", tmp_path / "rs", ("source", "plies", "winner"))
    assert GameShardDataset(str(tmp_path / "rs")).shard_meta == {"board_size": 20}


@pytest.mark.parametrize("flags", [(), ("handicap", "no_result")])
def test_go(gmdata, tmp_path, flags):
    from data_processing.games.go import convert as goconvert
    args = ["--input", os.path.join(FIX, "go"), "--output_dir", str(tmp_path / "py")]
    args += [f"--allow_{f}" for f in flags]
    goconvert.main(args)
    cmd = [gmdata, "go", "convert", "--input", os.path.join(FIX, "go"), "--out", tmp_path / "rs"]
    cmd += [f"--allow-{f.replace('_', '-')}" for f in flags]
    subprocess.run(cmd, check=True, capture_output=True)
    assert_same(tmp_path / "py", tmp_path / "rs", ("result", "komi", "handicap", "n_moves", "players"))


@pytest.mark.parametrize("flags", [(), ("--min-elo", "1500", "--skip-bullet")])
def test_chess(gmdata, tmp_path, flags):
    pytest.importorskip("chess")
    from data_processing.games.chess.convert import convert
    pgn = os.path.join(FIX, "chess", "lichess_2013-01_head.pgn")
    kw = {"min_elo": 1500, "skip_bullet": True} if flags else {}
    convert(pgn, str(tmp_path / "py"), **kw)
    subprocess.run([gmdata, "chess", "convert", "--input", pgn, "--out", tmp_path / "rs", *flags],
                   check=True, capture_output=True)
    assert_same(tmp_path / "py", tmp_path / "rs", ("white", "black", "result"))
    assert GameShardDataset(str(tmp_path / "rs")).converter_version == 2
