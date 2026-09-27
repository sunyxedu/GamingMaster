"""The Rust converters (`gmdata <game> convert`) must reproduce, bit for bit, the
tensors of the retired Python reference converters on the committed real-data
fixtures. The reference outputs are pinned as per-tensor SHA-256 digests in
tests/fixtures/golden/ (see tests/golden_digests.py)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

from data_processing.games.dataset import VALUE_TARGET_BOUND, GameShardDataset, load_gm_samples
from model.config import ModelConfig

sys.path.insert(0, os.path.dirname(__file__))
from golden_digests import sample_digests  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
FIX = os.path.join(ROOT, "tests", "fixtures")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")

PGN = os.path.join(FIX, "chess", "lichess_2013-01_head.pgn")

# golden name → (gmdata args, meta keys compared)
CASES = {
    "othello": (["othello", "convert", "--input", f"{FIX}/othello"], ("game_index", "replayed_black", "n_pass_frames")),
    "gomoku_freestyle": (["gomoku", "convert", "--input", f"{FIX}/gomoku", "--board-size", "20"], ("source", "plies", "winner")),
    "gomoku_standard_keep": (["gomoku", "convert", "--input", f"{FIX}/gomoku", "--board-size", "20", "--rule", "standard",
                              "--keep-undecided"], ("source", "plies", "winner")),
    "go": (["go", "convert", "--input", f"{FIX}/go"], ("result", "komi", "handicap", "n_moves", "players")),
    "go_handicap_noresult": (["go", "convert", "--input", f"{FIX}/go", "--allow-handicap", "--allow-no-result"],
                             ("result", "komi", "handicap", "n_moves", "players")),
    # hand-written 9x9 / 13x13 records (tests/fixtures/go_sim: captures, ko fights, a
    # snapback, fixed / free handicap, passes, suicide and ko-retake rejections, a
    # double-ko cycle), pinned by the converter before its rules moved into gmrules
    "go_sim9": (["go", "convert", "--input", f"{FIX}/go_sim", "--board-size", "9", "--min-moves", "1",
                 "--allow-handicap", "--allow-no-result"], ("result", "komi", "handicap", "n_moves", "players")),
    "go_sim13": (["go", "convert", "--input", f"{FIX}/go_sim", "--board-size", "13", "--min-moves", "1",
                  "--allow-handicap", "--allow-no-result"], ("result", "komi", "handicap", "n_moves", "players")),
    "chess": (["chess", "convert", "--input", PGN], ("white", "black", "elo", "result")),
    "chess_elo1500_nobullet": (["chess", "convert", "--input", PGN, "--min-elo", "1500", "--skip-bullet"],
                               ("white", "black", "elo", "result")),
    "shogi": (["shogi", "convert", "--input", f"{FIX}/shogi"], ("players", "result", "plies")),
    "shogi_noresult": (["shogi", "convert", "--input", f"{FIX}/shogi", "--allow-no-result"], ("players", "result", "plies")),
    "lux": (["lux", "convert", "--input", f"{FIX}/lux"], ("agent", "team", "match_index", "match_value")),
    "mahjong": (["mahjong", "convert", "--input", f"{FIX}/mahjong"], ("name", "seat", "kyokus", "placement", "final_score")),
    "pokemon": (["pokemon", "convert", "--input", f"{FIX}/pokemon"], ("replay_id", "side", "player", "rating", "won")),
}

# tgt.value contract (data_processing/games/dataset.py): the outcome levels each
# converter may emit, and whose perspective a sample holds — "mover": both
# players' frames, so only |value| is constant (the sign follows the mover);
# "player": one player's view, so the value is constant over the sample.
VALUE_CONTRACT = {
    "othello": ((-1.0, 0.0, 1.0), "mover"),
    "gomoku": ((-1.0, 1.0), "mover"),
    "go": ((-1.0, 0.0, 1.0), "mover"),
    "chess": ((-1.0, 0.0, 1.0), "mover"),
    "shogi": ((-1.0, 0.0, 1.0), "mover"),
    "lux": ((-1.0, 0.0, 1.0), "player"),
    "mahjong": ((-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0), "player"),
    "pokemon": ((-1.0, 1.0), "player"),
}


def check_value_contract(case, samples):
    levels, view = VALUE_CONTRACT[case.split("_")[0]]
    assert VALUE_TARGET_BOUND < ModelConfig().hl_gauss_vmax     # inside the HL-Gauss support
    lv = torch.tensor(levels)
    for i, s in enumerate(samples):
        v = s["tgt"]["value"].float()
        assert v.ndim == 1 and v.numel() > 0, (case, i)
        assert torch.isfinite(v).all() and v.abs().max() <= VALUE_TARGET_BOUND, (case, i)
        assert ((v[:, None] - lv).abs().min(-1).values < 1e-6).all(), (case, i, v.unique())
        ref = v.abs() if view == "mover" else v
        assert torch.all(ref == ref[0]), (case, i, v)


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS, capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return os.path.join(WS, "target", "release", "gmdata")


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_reference(gmdata, tmp_path, case):
    args, meta_keys = CASES[case]
    out = tmp_path / "rs"
    subprocess.run([gmdata, *args, "--out", str(out)], check=True, capture_output=True)
    golden = json.load(open(os.path.join(FIX, "golden", f"{case}.json")))
    samples = load_gm_samples(str(out))
    check_value_contract(case, samples)
    got = sample_digests(samples, meta_keys)
    assert len(got) == len(golden["samples"]) > 0
    for i, (g, w) in enumerate(zip(got, golden["samples"])):
        assert g == w, f"{case}: sample {i} differs from the reference"
    ds = GameShardDataset(str(out))
    assert ds.converter_version == golden["converter_version"]
    assert ds.shard_meta == golden["shard_meta"]
