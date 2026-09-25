"""Clash Royale data pipeline: Rust converter (crdata) → cr-frames.v1 → .pt shards.

The Rust half is exercised through its release binary (built on demand with
cargo; skipped when cargo is unavailable). The Python half is tested on a
synthetic frames shard so it runs everywhere.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from data_processing.games.clash_royale import loader
from data_processing.games.dataset import GameShardDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATE = os.path.join(ROOT, "data_processing", "games", "clash_royale", "crdata")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")


def _synthetic_frames_shard(root: str, T: int = 6, K: int = 4) -> str:
    d = os.path.join(root, "shard-000000")
    os.makedirs(d)
    N = T
    arrays = {
        "tick": np.arange(0, 10 * N, 10, dtype=np.int32),
        "time_s": (np.arange(N) * 0.5).astype(np.float32),
        "phase": np.zeros(N, np.uint8), "overtime": np.zeros(N, np.uint8),
        "elixir": np.array([5, 5.2, 1.4, 1.6, 1.8, 2.0], np.float32),
        "opp_elixir": np.full(N, 6.0, np.float32),
        "hand_ids": np.tile(np.array([[22, 3, 7, 40]], np.uint16), (N, 1)),
        "hand_form": np.zeros((N, 4), np.uint8),
        "hand_kind": np.tile(np.array([[0, 0, 2, 1]], np.uint8), (N, 1)),
        "hand_cost": np.tile(np.array([[4, 5, 4, 3]], np.float32), (N, 1)),
        "hand_prob": np.tile(np.array([[1.0, 0.5, 0.5, 0.5]], np.float32), (N, 1)),
        "next_card": np.full(N, 9, np.uint16),
        "recent_side": np.full((N, K), 255, np.uint8),
        "recent_card": np.zeros((N, K), np.uint16),
        "recent_x": np.full((N, K), -1, np.int8), "recent_y": np.full((N, K), -1, np.int8),
        "recent_age": np.full((N, K), -1, np.int32),
        "waited": np.array([1, 0, 1, 1, 1, 1], np.uint8),
        "card_slot": np.array([-1, 0, -1, -1, -1, -1], np.int8),
        "place_x": np.array([-1, 9, -1, -1, -1, -1], np.int8),
        "place_y": np.array([-1, 14, -1, -1, -1, -1], np.int8),
        "ability": np.zeros(N, np.uint8), "ability_card": np.zeros(N, np.uint16),
    }
    arrays["hand_ids"][1:, 0] = 9          # the played slot is refilled
    arrays["recent_side"][2:, 0] = 0
    arrays["recent_card"][2:, 0] = 22
    arrays["recent_x"][2:, 0] = 9
    arrays["recent_y"][2:, 0] = 14
    arrays["recent_age"][2:, 0] = np.arange(10, 10 * (N - 1), 10)
    for k, v in arrays.items():
        np.save(os.path.join(d, k + ".npy"), v)
    index = {"format": "cr-frames.v1", "catalog_version": "cr-card-catalog.v1", "step_ticks": 10, "recent": K,
             "hand": 4, "grid": [18, 32], "n_frames": N,
             "samples": [{"tag": "t", "side": "team", "start": 0, "length": N, "result": 1, "own_crowns": 1,
                          "opp_crowns": 0, "battle_type": "pathOfLegend", "game_mode": "Ranked",
                          "deck_ids": [22, 3, 7, 40, 9, 1, 2, 5], "tower_key": "tower-princess",
                          "hand_consistent_states": 3, "elixir_violations": 0, "n_plays": 1}]}
    with open(os.path.join(d, "index.json"), "w") as fh:
        json.dump(index, fh)
    return root


def test_loader_builds_adapter_obs(tmp_path):
    frames = _synthetic_frames_shard(str(tmp_path / "frames"))
    out = str(tmp_path / "shards")
    paths = loader.convert_dir(frames, out)
    assert len(paths) == 1
    ds = GameShardDataset(out, game="clash_royale")
    assert len(ds) == 1
    s = ds[0]
    obs, tgt = s["obs"], s["tgt"]
    assert obs["arena_seq"].shape == (6, 16, 32, 18)
    assert obs["hand_ids"].shape == (6, 4) and obs["hand_ids"].dtype == torch.int64
    assert obs["hand_feats"].shape == (6, 4, 8) and obs["globals_seq"].shape == (6, 6)
    assert bool(obs["hand_mask"].all())
    assert float(obs["hand_feats"][0, 0, 7]) == 1.0 and float(obs["hand_feats"][0, 1, 7]) == 0.5
    # label frame 1: hog (slot 0) at (9, 14) -> place_idx = 14*18 + 9
    assert not bool(tgt["waited"][1]) and int(tgt["card_slot"][1]) == 0
    assert int(tgt["place_idx"][1]) == 14 * 18 + 9 and int(tgt["place_idx"][0]) == -1
    assert torch.equal(tgt["value"], torch.ones(6))
    # deployment plane: own heat at the played cell, decaying with age
    heat = obs["arena_seq"][:, 0, 14, 9]
    assert heat[0] == 0 and heat[2] > heat[3] > 0
    assert obs["arena_seq"][0, 4, :16].all() and not obs["arena_seq"][0, 4, 16:].any()
    # affordability: at frame 2 elixir 1.4 -> nothing affordable; at frame 0 (5.0) cost-4 cards are
    assert not obs["legal_card"][2].any() and bool(obs["legal_card"][0][0])
    # spell in hand -> whole arena legal
    assert bool(obs["legal_place"][0].all())
    assert ds.shard_meta["n_cards"] == 124 and ds.shard_meta["step_ticks"] == 10


@pytest.mark.skipif(not os.path.exists(CARGO), reason="cargo not installed")
def test_rust_converter_end_to_end(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    def payload(plays):
        deck = lambda keys: [{"card_key": k, "level": 15, "name": k} for k in keys]
        team = ["hog-rider", "ice-spirit", "ice-golem", "skeletons-ev1", "cannon", "musketeer", "fireball", "the-log"]
        opp = ["giant", "witch", "knight", "archers", "zap", "minions", "fireball", "tornado"]
        events = [{"kind": "play_card", "side": s, "card_key": k, "card_play_number": 1, "replay_tick_20hz": t,
                   "time_seconds": t / 20, "form_at_play": "unknown", "deck_card_key_candidates": [k],
                   "ability_source_candidates": [],
                   "coordinates": {"grid_cell_floor": {"x": x, "y": y}, "inside_18x32_arena": True}, "source_index": i}
                  for i, (s, k, t, x, y) in enumerate(plays)]
        return json.dumps({
            "battle": {"battle_type": "pathOfLegend", "game_mode": "Ranked", "result": "victory",
                       "team": {"crowns": 1, "players": [{"deck": deck(team), "tower_card": {"card_key": "tower-princess", "level": 15, "name": "TP"}}]},
                       "opponent": {"crowns": 0, "players": [{"deck": deck(opp), "tower_card": {"card_key": "tower-princess", "level": 15, "name": "TP"}}]}},
            "events": events, "replay": {"duration": {"timeline_seconds": 30.0, "display_seconds": 30}},
            "schema_version": "royaleapi-battle-replay.v1", "source": {"site": "RoyaleAPI"}})

    raw = tmp_path / "raw" / "replays"
    raw.mkdir(parents=True)
    rows = [("a" * 8, payload([("team", "hog-rider", 40, 9, 14), ("opponent", "giant", 65, 8, 20), ("team", "ice-spirit", 120, 3, 10)])),
            ("b" * 8, payload([("opponent", "knight", 30, 10, 25)]))]
    table = pa.table({"schema_version": ["royaleapi-battle-replay.v1"] * 2, "replay_tag": [r[0] for r in rows],
                      "battle_type": ["pathOfLegend"] * 2, "game_mode": ["Ranked"] * 2, "result": ["victory"] * 2,
                      "team_crowns": [1, 1], "opponent_crowns": [0, 0], "event_count": [3, 1], "warning_count": [0, 0],
                      "payload_json": pa.array([r[1] for r in rows], pa.large_string()), "requested_player_tag": ["", ""]})
    pq.write_table(table, raw / "part-000000.parquet")

    build = subprocess.run([CARGO, "build", "--release", "-q"], cwd=CRATE, capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"cargo build failed: {build.stderr[-500:]}")
    binary = os.path.join(CRATE, "target", "release", "crdata")
    frames = tmp_path / "frames"
    res = subprocess.run([binary, "convert", "--input", str(tmp_path / "raw"), "--out", str(frames), "--recent", "4"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    summary = json.load(open(frames / "convert_summary.json"))
    assert summary["rows"] == 2 and summary["samples"] == 4 and summary["parse_errors"] == 0
    fs = loader.load_frames_shard(str(frames / "shard-000000"))
    assert fs["index"]["n_frames"] == 4 * 61                      # 30 s / 0.5 s + 1 per side
    first = fs["index"]["samples"][0]
    assert first["side"] == "team" and first["n_plays"] == 2 and first["result"] == 1
    # frame 4 (tick 40) labels the hog play in the team's own frame
    assert fs["waited"][4] == 0 and (fs["place_x"][4], fs["place_y"][4]) == (9, 14)
    # elixir before the play: start + 2.0 s of regen
    assert abs(fs["elixir"][4] - (6.0 + 2.0 / 2.8)) < 1e-4
    assert fs["hand_prob"].shape == (4 * 61, 4) and fs["hand_prob"][4].max() == 1.0
    # opponent-perspective sample sees its own knight play mirrored into rows < 16
    opp = fs["index"]["samples"][1]
    o = opp["start"]
    assert opp["side"] == "opponent" and opp["result"] == -1
    f = o + 6
    assert fs["waited"][f] == 0 and (fs["place_x"][f], fs["place_y"][f]) == (17 - 8, 31 - 20)
    # and the whole thing loads into .pt shards for the adapter
    paths = loader.convert_dir(str(frames), str(tmp_path / "shards"))
    ds = GameShardDataset(str(tmp_path / "shards"), game="clash_royale")
    assert len(ds) == 4 and ds[0]["obs"]["arena_seq"].shape[1:] == (16, 32, 18)
