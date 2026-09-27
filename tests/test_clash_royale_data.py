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
        "recent_age": np.full((T, K), -1, np.int32), "recent_card": np.zeros((T, K), np.uint16),
        "tick": np.arange(T, dtype=np.int32) * 10,
    }
    r["recent_side"][2:, 0] = 0
    r["recent_card"][2:, 0] = 22
    r["recent_cost"][2:, 0] = 4.0
    r["recent_x"][2:, 0] = 9
    r["recent_y"][2:, 0] = 14
    r["recent_age"][2:, 0] = np.arange(10, 10 * (T - 1), 10)
    return {k: torch.from_numpy(v) for k, v in r.items()}


def test_expand_builds_legacy_hand_obs():
    """converter_version <= 2 shards (raw.hand_*) still expand to the legacy hand
    observation (legacy cr_card_tokens="hand" checkpoints)."""
    obs = expand(_raw(), {"arena_channels": 16, "card_feature_dim": 8, "tau_ticks": 60.0})
    assert set(obs) == {"arena_seq", "hand_ids", "hand_feats", "hand_mask", "globals_seq", "legal_card",
                        "legal_place"}
    assert obs["arena_seq"].shape == (6, 16, 32, 18) and obs["hand_feats"].shape == (6, 4, 8)
    heat = obs["arena_seq"][:, 0, 14, 9]
    assert heat[0] == 0 and heat[2] > heat[3] > 0
    assert float(obs["arena_seq"][2, 2, 14, 9]) == pytest.approx(0.4)
    assert obs["arena_seq"][0, 4, :16].all() and not obs["arena_seq"][0, 4, 16:].any()
    assert not obs["legal_card"][2].any() and bool(obs["legal_card"][0][0])
    assert bool(obs["legal_place"][0].all())                      # spell in hand
    assert float(obs["hand_feats"][0, 1, 7]) == 0.5
    assert obs["hand_ids"].dtype == torch.int64


def _deck_raw(T=6, K=4):
    r = {k: v for k, v in _raw(T, K).items() if not k.startswith("hand_") and k != "next_cost"}
    tile = lambda row, dt: torch.from_numpy(np.tile(np.array(row, dt), (T, 1)))   # noqa: E731
    r.update({
        "deck_ids": tile([3, 7, 22, 26, 40, 51, 60, 88], np.uint16),
        "deck_form": tile([0] * 8, np.uint8),
        "deck_kind": tile([0, 2, 0, 1, 0, 0, 1, 2], np.uint8),
        "deck_cost": tile([5, 4, 2, 3, 2, 6, 5, 2], np.float32),
        # the deal prior, then card 22 (index 2) played between frames 1 and 2
        "deck_in_hand": torch.tensor([[0.5] * 8] * 2 + [[4 / 7, 4 / 7, 0.0] + [4 / 7] * 5] * (T - 2)),
        "deck_next": torch.tensor([[0.125] * 8] * 2 + [[1 / 7, 1 / 7, 0.0] + [1 / 7] * 5] * (T - 2)),
    })
    return r


def test_expand_builds_deck_obs():
    """converter_version 3: 8 deck tokens with the causal hand belief; legal card =
    P(in hand) > 0 and affordable at the frame."""
    raw = _deck_raw()
    obs = expand(raw, {"arena_channels": 16, "card_feature_dim": 9, "tau_ticks": 60.0})
    assert set(obs) == {"arena_seq", "deck_ids", "deck_feats", "deck_mask", "globals_seq", "legal_card",
                        "legal_place", "recent_ids", "recent_feats", "recent_mask", "recent_cell",
                        "opp_ids", "opp_feats", "opp_mask"}
    assert obs["deck_feats"].shape == (6, 8, 9) and obs["deck_ids"].dtype == torch.int64
    # the own play of card 22 (tick 10, cell (9, 14)) from frame 2 on: the match's play 0,
    # ring slot 0; no opponent play yet
    assert not obs["recent_mask"][:2].any() and obs["recent_mask"][2:, 0].all()
    assert obs["recent_mask"][2:].sum() == 4
    assert (obs["recent_ids"][2:, 0] == 22).all() and (obs["recent_cell"][2:, 0] == 14 * 18 + 9).all()
    f = obs["recent_feats"][2:, 0]
    assert (f[:, :3] == torch.tensor([1.0, 0.0, 0.0])).all() and torch.allclose(f[:, 5], torch.tensor(0.4))
    assert torch.allclose(f[:, 3], torch.exp(-torch.tensor([10.0, 20.0, 30.0, 40.0]) / 60.0))
    assert not obs["opp_mask"].any() and (obs["opp_ids"] == 0).all()
    assert obs["deck_mask"].all()
    assert torch.equal(obs["deck_feats"][:, :, 7], raw["deck_in_hand"])
    assert torch.equal(obs["deck_feats"][:, :, 8], raw["deck_next"])
    # frame 0: elixir 5 → the 6-cost card (index 5) is unaffordable; every card may be in hand
    assert obs["legal_card"][0].tolist() == [True, True, True, True, True, False, True, True]
    # frame 5: elixir 2.0 affords the 2-cost cards, but card 22 (index 2) is known to be
    # out of the hand
    assert obs["legal_card"][5].tolist() == [False, False, False, False, True, False, False, True]
    assert bool(obs["legal_place"][5].all())                      # a playable spell (index 7)
    assert not obs["legal_place"][2, 16 * 18:].any()              # elixir 1.4: nothing playable
    assert float(obs["globals_seq"][0, 5]) == pytest.approx(29 / 8 / 10)   # E[next cost] / 10
    with pytest.raises(ValueError, match="card_feature_dim"):
        expand(raw, {"arena_channels": 16, "card_feature_dim": 8, "tau_ticks": 60.0})


@pytest.mark.skipif(not os.path.exists(CARGO), reason="cargo not installed")
@pytest.mark.parametrize("fmt", ["jsonl", "parquet"])
def test_rust_converter_end_to_end(tmp_path, fmt):
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
    if fmt == "parquet":
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        table = pa.table({"replay_tag": [r[0] for r in rows], "game_mode": ["Ranked"] * 2,
                          "payload_json": pa.array([r[1] for r in rows], pa.large_string())})
        pq.write_table(table, raw / "part-000000.parquet")
    else:                                     # JSON lines, the test fixture's format
        with open(raw / "part-000000.jsonl", "w") as fh:
            for tag, js in rows:
                fh.write(json.dumps({"tag": tag, "payload": json.loads(js)}) + "\n")
    build = subprocess.run([CARGO, "build", "--release", "-q", "-p", "crdata"], cwd=WS, capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"cargo build failed: {build.stderr[-500:]}")
    binary = os.path.join(WS, "target", "release", "crdata")
    out = tmp_path / "shards"
    res = subprocess.run([binary, "convert", "--input", str(tmp_path / "raw"), "--out", str(out), "--recent", "4"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    ds = GameShardDataset(str(out), game="clash_royale")
    assert len(ds) == 4 and ds.converter_version == 3 and ds.shard_meta["n_cards"] == 124
    meta = ds.shard_meta
    assert (meta["card_tokens"], meta["deck_size"], meta["card_feature_dim"]) == ("deck", 8, 9)
    team = ds[0]
    assert team["meta"]["side"] == "team" and team["meta"]["result"] == 1
    obs, tgt = team["obs"], team["tgt"]
    assert obs["arena_seq"].shape == (61, 16, 32, 18)            # 30 s / 0.5 s + 1
    assert obs["deck_ids"].shape == (61, 8) and bool((obs["deck_ids"][0].diff() > 0).all())  # by id
    # frame 4 (tick 40) labels the hog play at (9, 14) from the team's own perspective;
    # the label is the hog's deck token
    assert not bool(tgt["waited"][4]) and int(tgt["place_idx"][4]) == 14 * 18 + 9
    slot = int(tgt["card_slot"][4])
    # nothing was played before the decision: the causal belief is the deal prior (the
    # converter_version 2 hand marked the coming hog "certainly in hand")
    assert torch.all(obs["deck_feats"][4, :, 7] == 0.5) and torch.all(obs["deck_feats"][4, :, 8] == 0.125)
    assert bool(obs["legal_card"][4, slot])
    # once played, the hog is out of the hand; the ice spirit played at tick 120 had 4/7
    assert float(obs["deck_feats"][5, slot, 7]) == 0.0 and not bool(obs["legal_card"][5, slot])
    spirit = int(tgt["card_slot"][12])
    assert spirit not in (-1, slot) and float(obs["deck_feats"][12, spirit, 7]) == pytest.approx(4 / 7)
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
