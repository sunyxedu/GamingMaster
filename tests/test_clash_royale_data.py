"""Clash Royale: Rust converter (crdata) → gm-shard (compact raw.* + targets) →
GameShardDataset expands the adapter observation at load time."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from data_processing.games.clash_royale.expand import expand
from data_processing.games.dataset import VALUE_TARGET_BOUND, GameShardDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")


def _raw(T=6, K=4):
    r = {
        "time_s": np.arange(T, dtype=np.float32) * 0.5, "phase": np.zeros(T, np.uint8), "overtime": np.zeros(T, np.uint8),
        "elixir": np.array([5, 5.2, 1.4, 1.6, 1.8, 2.0], np.float32), "opp_elixir": np.full(T, 6.0, np.float32),
        "hand_ids": np.tile(np.array([[22, 3, 7, 40]], np.uint16), (T, 1)), "hand_form": np.zeros((T, 4), np.uint8),
        "hand_kind": np.tile(np.array([[0, 0, 2, 1]], np.uint8), (T, 1)),
        "hand_cost": np.tile(np.array([[4, 5, 4, 3]], np.float32), (T, 1)),
        "hand_prob": np.tile(np.array([[1.0, 0.5, 0.5, 0.5]], np.float32), (T, 1)), "next_cost": np.full(T, 2.0, np.float32),
        "recent_side": np.full((T, K), 255, np.uint8), "recent_cost": np.zeros((T, K), np.float32),
        "recent_x": np.full((T, K), -1, np.int8), "recent_y": np.full((T, K), -1, np.int8),
        "recent_age": np.full((T, K), -1, np.int32),
    }
    r["recent_side"][2:, 0] = 0
    r["recent_cost"][2:, 0] = 4.0
    r["recent_x"][2:, 0] = 9
    r["recent_y"][2:, 0] = 14
    r["recent_age"][2:, 0] = np.arange(10, 10 * (T - 1), 10)
    return {k: torch.from_numpy(v) for k, v in r.items()}


def test_expand_builds_adapter_obs():
    obs = expand(_raw(), {"arena_channels": 16, "card_feature_dim": 8, "tau_ticks": 60.0})
    assert obs["arena_seq"].shape == (6, 16, 32, 18) and obs["hand_feats"].shape == (6, 4, 8)
    heat = obs["arena_seq"][:, 0, 14, 9]
    assert heat[0] == 0 and heat[2] > heat[3] > 0
    assert float(obs["arena_seq"][2, 2, 14, 9]) == pytest.approx(0.4)
    assert obs["arena_seq"][0, 4, :16].all() and not obs["arena_seq"][0, 4, 16:].any()
    assert not obs["legal_card"][2].any() and bool(obs["legal_card"][0][0])
    assert bool(obs["legal_place"][0].all())                      # spell in hand
    assert float(obs["hand_feats"][0, 1, 7]) == 0.5
    assert obs["hand_ids"].dtype == torch.int64


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
        tower = {"card_key": "tower-princess", "level": 15, "name": "TP"}
        return json.dumps({
            "battle": {"battle_type": "pathOfLegend", "game_mode": "Ranked", "result": "victory",
                       "team": {"crowns": 1, "players": [{"deck": deck(team), "tower_card": tower}]},
                       "opponent": {"crowns": 0, "players": [{"deck": deck(opp), "tower_card": tower}]}},
            "events": events, "replay": {"duration": {"timeline_seconds": 30.0, "display_seconds": 30}},
            "schema_version": "royaleapi-battle-replay.v1", "source": {"site": "RoyaleAPI"}})

    raw = tmp_path / "raw" / "replays"
    raw.mkdir(parents=True)
    rows = [("a" * 8, payload([("team", "hog-rider", 40, 9, 14), ("opponent", "giant", 65, 8, 20), ("team", "ice-spirit", 120, 3, 10)])),
            ("b" * 8, payload([("opponent", "knight", 30, 10, 25)]))]
    table = pa.table({"replay_tag": [r[0] for r in rows], "game_mode": ["Ranked"] * 2,
                      "payload_json": pa.array([r[1] for r in rows], pa.large_string())})
    pq.write_table(table, raw / "part-000000.parquet")
    build = subprocess.run([CARGO, "build", "--release", "-q", "-p", "crdata"], cwd=WS, capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"cargo build failed: {build.stderr[-500:]}")
    binary = os.path.join(WS, "target", "release", "crdata")
    out = tmp_path / "shards"
    res = subprocess.run([binary, "convert", "--input", str(tmp_path / "raw"), "--out", str(out), "--recent", "4"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    ds = GameShardDataset(str(out), game="clash_royale")
    assert len(ds) == 4 and ds.converter_version == 2 and ds.shard_meta["n_cards"] == 124
    team = ds[0]
    assert team["meta"]["side"] == "team" and team["meta"]["result"] == 1
    obs, tgt = team["obs"], team["tgt"]
    assert obs["arena_seq"].shape == (61, 16, 32, 18)            # 30 s / 0.5 s + 1
    # frame 4 (tick 40) labels the hog play at (9, 14) from the team's own perspective
    assert not bool(tgt["waited"][4]) and int(tgt["place_idx"][4]) == 14 * 18 + 9
    slot = int(tgt["card_slot"][4])
    assert float(obs["hand_feats"][4, slot, 7]) == 1.0           # certainly in hand
    assert abs(float(obs["globals_seq"][4, 0]) * 10 - (6.0 + 2.0 / 2.8)) < 1e-4
    assert float(obs["arena_seq"][5, 0, 14, 9]) > 0              # own deployment heat after the play
    opp = ds[1]
    assert opp["meta"]["side"] == "opponent" and torch.all(opp["tgt"]["value"] == -1)
    # tgt.value contract: the battle result for the sample's side, constant
    # over the sample, inside [-1, 1] (and so inside the HL-Gauss support)
    for i in range(len(ds)):
        s = ds[i]
        v = s["tgt"]["value"]
        assert torch.all(v == s["meta"]["result"]) and v.abs().max() <= VALUE_TARGET_BOUND
    f = 6                                                        # its giant at tick 65 on (8, 20), mirrored
    assert not bool(opp["tgt"]["waited"][f]) and int(opp["tgt"]["place_idx"][f]) == (31 - 20) * 18 + (17 - 8)
