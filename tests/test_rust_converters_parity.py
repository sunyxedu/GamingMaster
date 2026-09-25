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

from data_processing.games.dataset import GameShardDataset, load_gm_samples

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
    "chess": (["chess", "convert", "--input", PGN], ("white", "black", "elo", "result")),
    "chess_elo1500_nobullet": (["chess", "convert", "--input", PGN, "--min-elo", "1500", "--skip-bullet"],
                               ("white", "black", "elo", "result")),
    "shogi": (["shogi", "convert", "--input", f"{FIX}/shogi"], ("players", "result", "plies")),
    "shogi_noresult": (["shogi", "convert", "--input", f"{FIX}/shogi", "--allow-no-result"], ("players", "result", "plies")),
    "lux": (["lux", "convert", "--input", f"{FIX}/lux"], ("agent", "team", "match_index", "match_value")),
    "mahjong": (["mahjong", "convert", "--input", f"{FIX}/mahjong"], ("name", "seat", "kyokus", "placement", "final_score")),
    "pokemon": (["pokemon", "convert", "--input", f"{FIX}/pokemon"], ("replay_id", "side", "player", "rating", "won")),
}


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
    got = sample_digests(load_gm_samples(str(out)), meta_keys)
    assert len(got) == len(golden["samples"]) > 0
    for i, (g, w) in enumerate(zip(got, golden["samples"])):
        assert g == w, f"{case}: sample {i} differs from the reference"
    ds = GameShardDataset(str(out))
    assert ds.converter_version == golden["converter_version"]
    assert ds.shard_meta == golden["shard_meta"]
