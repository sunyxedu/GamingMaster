"""Lux AI S3 pipeline tests: Kaggle replay JSON → frames → shards → network.

  1. converter constants stay in sync with LuxConfig defaults
  2. plane construction ([x][y] → [y][x] transpose, masking, one-hots)
  3. action extraction (obs[t] pairs with steps[t+1].action) + clamping
  4. match splitting at match_steps == 0 + per-match value targets
  5. trainer LEGAL_FROM_OBS hook → pred["unit_present"]; absent slots
     contribute exactly 0 to il_loss
  6. integration on a REAL championship episode (skipped if the sample
     file is absent): convert → shard → dataset → collate → network
     forward + il_loss + backward, all finite
"""

import json
import os

import numpy as np
import pytest
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
)
from data_processing.games.lux import convert as luxconvert
from data_processing.games.lux.convert import (
    MAP_CHANNELS,
    MAP_H,
    MAP_W,
    MAX_UNITS,
    N_UNIT_ACTIONS,
    UNIT_FEATURE_DIM,
    build_frame,
    episode_to_samples,
)

# Real Kaggle episode for schema verification — the only test that can
# falsify the converter's [x][y]-transposition and obs[t]↔steps[t+1]
# alignment assumptions against ground truth (the synthetic fixture below
# is built FROM those assumptions, so it cannot).  The asserts are pinned
# to episode 71903094; download it from the Kaggle Lux S3 episode dump
# and opt in via:
#     LUX_REAL_EPISODE=/path/to/episode-71903094-replay.json pytest ...
REAL_EPISODE = os.environ.get("LUX_REAL_EPISODE", "")

SMALL = dict(
    d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
    trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
    halt_max_steps=2, dropout=0.0, encoder_channels=16,
)


# ---------------------------------------------------------------------------
# Synthetic mini-episode: 7 steps = 2 matches of 3 frames + trailing frame,
# 2 present units on team 0, 1 on team 1.
# ---------------------------------------------------------------------------

def _obs_dict(steps, match_steps, team_points, team_wins):
    """Kaggle-shaped obs dict. Map arrays are [x][y] (transposed on load)."""
    pos = [[[-1, -1]] * MAX_UNITS for _ in range(2)]
    energy = [[-1] * MAX_UNITS for _ in range(2)]
    mask = [[False] * MAX_UNITS for _ in range(2)]
    # team 0: units in slots 0 and 2; team 1: one unit in slot 5.
    pos[0][0], energy[0][0], mask[0][0] = [2, 5], 100, True
    pos[0][2], energy[0][2], mask[0][2] = [7, 1], 200, True
    pos[1][5], energy[1][5], mask[1][5] = [20, 3], 400, True

    tile = [[0] * MAP_H for _ in range(MAP_W)]      # [x][y]
    tile[3][1] = 2                                   # asteroid at x=3, y=1
    tile[4][2] = 1                                   # nebula   at x=4, y=2
    tile[10][10] = -1                                # unseen
    sensor = [[True] * MAP_H for _ in range(MAP_W)]
    sensor[10][10] = False
    menergy = [[0] * MAP_H for _ in range(MAP_W)]
    menergy[6][9] = 10                               # x=6, y=9
    menergy[10][10] = -1                             # unseen → masked out
    relic = [[12, 8]] + [[-1, -1]] * 5
    relic_mask = [True] + [False] * 5

    return {
        "units": {"position": pos, "energy": energy},
        "units_mask": mask,
        "sensor_mask": sensor,
        "map_features": {"energy": menergy, "tile_type": tile},
        "relic_nodes": relic,
        "relic_nodes_mask": relic_mask,
        "team_points": team_points,
        "team_wins": team_wins,
        "steps": steps,
        "match_steps": match_steps,
    }


def _mini_episode():
    """7 steps: match 0 = steps 0..2, match 1 = steps 3..5, trailing step 6.
    Team 0 wins match 0 (team_wins → [1,0]) and loses match 1 (→ [1,1]).
    Action at steps[t] is the response to obs[t−1]: unit 0's action_type
    is t (mod 6) so the pairing offset is directly observable."""
    match_steps = [0, 1, 2, 0, 1, 2, 0]
    wins = [[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 1]]
    points = [[0, 0], [3, 1], [6, 2], [0, 0], [1, 4], [2, 8], [0, 0]]
    steps = []
    for t in range(7):
        o = _obs_dict(t, match_steps[t], points[t], wins[t])
        entries = []
        for agent in range(2):
            oa = dict(o)
            oa["team_points"] = points[t]
            action = None if t == 0 else [
                [t % 6, 0, 0] if u == 0 else
                ([9, 2, -1] if u == 2 else [0, 0, 0])   # slot 2: clamp check
                for u in range(MAX_UNITS)
            ]
            entries.append({
                "action": action if action is not None else {},
                "info": {},
                "observation": {"obs": json.dumps(oa),
                                "player": f"player_{agent}"},
                "reward": 0,
                "status": "ACTIVE",
            })
        steps.append(entries)
    return {
        "id": 42,
        "configuration": {"env_cfg": {
            "map_height": MAP_H, "map_width": MAP_W,
            "match_count_per_episode": 2, "max_steps_in_match": 2,
            "max_units": MAX_UNITS,
        }},
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# 1. constants ↔ LuxConfig
# ---------------------------------------------------------------------------

def test_constants_match_lux_config():
    from model.games.lux import LuxConfig
    cfg = LuxConfig()
    assert (MAP_H, MAP_W) == (cfg.map_h, cfg.map_w)
    assert MAP_CHANNELS == cfg.map_channels
    assert MAX_UNITS == cfg.max_units
    assert UNIT_FEATURE_DIM == cfg.unit_feature_dim
    assert luxconvert.GLOBAL_FEATURES == cfg.lux_global_features
    assert N_UNIT_ACTIONS == cfg.n_unit_actions


# ---------------------------------------------------------------------------
# 2. plane construction
# ---------------------------------------------------------------------------

def test_plane_construction():
    o = _obs_dict(steps=105, match_steps=4, team_points=[50, 25],
                  team_wins=[1, 0])
    planes, units, mask, globs = build_frame(o, team=0)
    assert planes.shape == (MAP_CHANNELS, MAP_H, MAP_W)
    assert planes.dtype == np.float32

    # Tile one-hots, transposed [x][y] → plane[c, y, x].
    assert planes[2, 1, 3] == 1.0 and planes[0, 1, 3] == 0.0   # asteroid
    assert planes[1, 2, 4] == 1.0                              # nebula
    # Unseen tile (−1): all three tile planes zero, sensor plane zero.
    assert planes[0, 10, 10] == planes[1, 10, 10] == planes[2, 10, 10] == 0.0
    assert planes[4, 10, 10] == 0.0 and planes[4, 0, 0] == 1.0
    # Energy normalised and sensor-masked.
    assert planes[3, 9, 6] == pytest.approx(10 / 20)
    assert planes[3, 10, 10] == 0.0                            # −1 masked to 0
    # Relic one-hot.
    assert planes[5, 8, 12] == 1.0 and planes[5].sum() == 1.0
    # Own units: count + energy sum at (x=2,y=5) and (x=7,y=1).
    assert planes[6, 5, 2] == pytest.approx(1 / 4)
    assert planes[6, 1, 7] == pytest.approx(1 / 4)
    assert planes[6].sum() == pytest.approx(2 / 4)
    assert planes[8, 5, 2] == pytest.approx(100 / 400)
    # Opp unit at (x=20,y=3).
    assert planes[7, 3, 20] == pytest.approx(1 / 4)
    assert planes[9, 3, 20] == pytest.approx(400 / 400)
    # Progress + ones planes are constant.
    assert np.allclose(planes[10], 4 / 100)
    assert np.all(planes[11] == 1.0)

    # Unit slots.
    assert mask.tolist() == [True, False, True] + [False] * 13
    assert units.shape == (MAX_UNITS, UNIT_FEATURE_DIM)
    np.testing.assert_allclose(
        units[0, :4], [2 / 24, 5 / 24, 100 / 400, 1.0])
    assert units[0, 4:].sum() == 0                    # padding dims stay zero
    assert units[1].sum() == 0                        # absent slot all-zero

    # Globals (own/opp swap under team=1).
    np.testing.assert_allclose(
        globs, [50 / 500, 25 / 500, 1 / 5, 0 / 5, 4 / 101, 105 / 506])
    _, _, mask1, globs1 = build_frame(o, team=1)
    np.testing.assert_allclose(globs1[:4], [25 / 500, 50 / 500, 0, 1 / 5])
    assert mask1.tolist() == [False] * 5 + [True] + [False] * 10


# ---------------------------------------------------------------------------
# 3 + 4. action extraction, match splitting, values
# ---------------------------------------------------------------------------

def test_action_extraction_and_match_splitting():
    samples, hist, counters = episode_to_samples(_mini_episode(), "mini")
    # 2 matches × 2 agents; trailing frame (step 6) has no following action.
    assert len(samples) == 4
    assert counters["matches"] == 4 and counters["empty_matches"] == 2
    for s in samples:
        assert s["obs"]["map_seq"].shape == (3, MAP_CHANNELS, MAP_H, MAP_W)
        assert s["obs"]["units_seq"].shape == (3, MAX_UNITS, UNIT_FEATURE_DIM)
        assert s["obs"]["unit_mask"].dtype == torch.bool
        assert s["tgt"]["unit_actions"].shape == (3, MAX_UNITS)
        assert s["tgt"]["unit_actions"].dtype == torch.long
        assert torch.equal(s["obs"]["unit_mask"], s["tgt"]["unit_mask"])

    a0m0 = samples[0]
    assert (a0m0["meta"]["agent"], a0m0["meta"]["match_index"]) == (0, 0)
    assert a0m0["meta"]["step_range"] == (0, 3)
    # obs[t] pairs with the action stored at steps[t+1] → unit 0 gets t+1.
    assert a0m0["tgt"]["unit_actions"][:, 0].tolist() == [1, 2, 3]
    # Recorded 9 clamps into [0, 5].
    assert a0m0["tgt"]["unit_actions"][:, 2].tolist() == [5, 5, 5]

    a0m1 = samples[1]
    assert a0m1["meta"]["step_range"] == (3, 6)
    assert a0m1["tgt"]["unit_actions"][:, 0].tolist() == [4, 5, 0]  # t=4,5,6

    # Values: team 0 wins match 0, team 1 wins match 1 (team_wins deltas).
    assert samples[0]["tgt"]["value"].tolist() == [1.0] * 3   # agent 0, m0
    assert samples[1]["tgt"]["value"].tolist() == [-1.0] * 3  # agent 0, m1
    assert samples[2]["tgt"]["value"].tolist() == [-1.0] * 3  # agent 1, m0
    assert samples[3]["tgt"]["value"].tolist() == [1.0] * 3   # agent 1, m1

    # Histogram counts present units only: agent 0 has 2 units × 6 frames,
    # agent 1 has 1 unit × 6 frames.
    assert sum(hist.values()) == 2 * 6 + 1 * 6


def test_value_fallback_on_truncated_replay():
    ep = _mini_episode()
    ep["steps"] = ep["steps"][:6]     # drop the trailing boundary frame
    samples, _, _ = episode_to_samples(ep, "mini")
    # Match 1 now has 2 usable frames and no wins delta → points fallback:
    # last frame points [2, 8] → agent 0 loses, agent 1 wins.
    a0m1 = [s for s in samples
            if s["meta"]["agent"] == 0 and s["meta"]["match_index"] == 1][0]
    assert a0m1["tgt"]["value"].shape == (2,)
    assert a0m1["tgt"]["value"].tolist() == [-1.0, -1.0]


# ---------------------------------------------------------------------------
# 5. trainer hook + presence masking through the real network
# ---------------------------------------------------------------------------

def test_legal_hook_and_absent_slots_contribute_zero():
    from model.games import build_game_network
    from train_il_game import LEGAL_FROM_OBS, run_batch

    samples, _, _ = episode_to_samples(_mini_episode(), "mini")
    batch = collate_game_sequences(samples[:2])
    torch.manual_seed(0)
    net = build_game_network("lux", **SMALL)
    net.eval()

    legal = LEGAL_FROM_OBS["lux"](batch["obs"])
    assert legal["unit"].dtype == torch.bool
    with torch.no_grad():
        pred = net(batch["obs"], legal=legal,
                   frame_mask=batch["meta"]["frame_mask"])
    # decode() echoes the mask as pred["unit_present"].
    assert torch.equal(pred["unit_present"], batch["obs"]["unit_mask"].bool())

    # Absent slots contribute exactly 0: perturbing their targets (and
    # their logits) cannot change the IL loss.
    fm = batch["meta"]["frame_mask"]
    loss1 = net.action_decoder.il_loss(pred, batch["tgt"], fm)["loss"]
    tgt2 = {k: v.clone() for k, v in batch["tgt"].items()}
    absent = ~batch["obs"]["unit_mask"].bool()
    tgt2["unit_actions"][absent] = 3
    pred2 = dict(pred)
    pred2["unit_action_logits"] = pred["unit_action_logits"].clone()
    pred2["unit_action_logits"][absent] += 100.0
    loss2 = net.action_decoder.il_loss(pred2, tgt2, fm)["loss"]
    assert torch.allclose(loss1, loss2)

    # And the full trainer path (hook applied inside run_batch) trains.
    net.train()
    loss, stats = run_batch(net, "lux", batch, torch.device("cpu"),
                            value_weight=0.2, tf_prob=1.0, act_steps=1,
                            bf16=False)
    assert torch.isfinite(loss)
    assert "unit_action_loss" in stats and "value_loss" in stats
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# 6. integration: real championship episode
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(REAL_EPISODE),
                    reason="real Lux episode sample not on this machine")
def test_real_episode_end_to_end(tmp_path):
    from model.games import build_game_network

    out_dir = tmp_path / "shards"
    stats = luxconvert.main([
        "--input", REAL_EPISODE, "--output_dir", str(out_dir),
        "--shard_size", "64",
    ])
    # 506 steps = 5 matches × 101 frames + trailing frame; × 2 agents.
    assert stats["episodes"] == 1
    assert stats["matches"] == 10
    assert stats["samples"] == 10
    assert stats["frames"] == 1010
    assert stats["shards"] == 1

    hist = stats["action_hist"]
    moves = sum(hist.get(a, 0) for a in (1, 2, 3, 4))
    assert sum(hist.values()) > 0
    assert moves > hist.get(5, 0) > 0          # mostly moves, some saps
    assert set(hist) <= set(range(N_UNIT_ACTIONS))

    ds = GameShardDataset(str(out_dir), game="lux", max_len=12)
    assert len(ds) == 10
    s = ds[0]
    assert s["obs"]["map_seq"].shape == (12, MAP_CHANNELS, MAP_H, MAP_W)
    # Every present unit's action is a valid type.
    present = s["obs"]["unit_mask"].bool()
    acts = s["tgt"]["unit_actions"][present]
    assert ((acts >= 0) & (acts < N_UNIT_ACTIONS)).all()
    # Player 0 swept this episode 5–0: values are ±1 by agent.
    full = torch.load(sorted(out_dir.glob("*.pt"))[0], weights_only=False)
    for sample in full["samples"]:
        want = 1.0 if sample["meta"]["agent"] == 0 else -1.0
        assert (sample["tgt"]["value"] == want).all()
        assert sample["tgt"]["value"].shape == (101,)

    batch = collate_game_sequences([ds[0], ds[7]])
    torch.manual_seed(0)
    net = build_game_network("lux", **SMALL)
    fm = batch["meta"]["frame_mask"]
    pred = net(batch["obs"], legal={"unit": batch["obs"]["unit_mask"].bool()},
               frame_mask=fm)
    B, T = fm.shape
    assert pred["unit_action_logits"].shape == (B, T, MAX_UNITS, N_UNIT_ACTIONS)
    assert torch.equal(pred["unit_present"], batch["obs"]["unit_mask"].bool())

    losses = net.action_decoder.il_loss(pred, batch["tgt"], fm)
    v_loss = torch.nn.functional.mse_loss(
        pred["value"].float(), batch["tgt"]["value"].float())
    loss = losses["loss"] + 0.2 * v_loss
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
