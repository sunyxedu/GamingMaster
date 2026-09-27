"""Riichi mahjong simulator (env/vec_mahjong.py over gmsim's MahjongSim, Rust crate
data_processing/rs/gmmahjong): the env contract, determinism, and IL → RL end to end.

That a simulator frame is the converter's frame of the same position is pinned in Rust
(``gmdata`` test ``mahjong::sim_parity``: every fixture Houou game replayed through the
simulator, frame for frame); here: the rows, masks and rewards the RL trainer relies on,
the frame conventions shared with the IL loader (tile planes = quarters / 4), and a tiny
train_multi_game → train_rl_joint run on the fixture shards with μ = π_old.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest
import torch

from tests.test_rust_converters_parity import FIX, gmdata  # noqa: F401  (fixture)

gmsim = pytest.importorskip("gmsim")
if not hasattr(gmsim._native, "MahjongSim"):
    pytest.skip("gmsim was built without MahjongSim (uv sync --all-extras --reinstall-package gmsim)",
                allow_module_level=True)

from env.vec_mahjong import MahjongVecEnv  # noqa: E402

PLACEMENT = torch.tensor([1.0, 1 / 3, -1 / 3, -1.0])
USES_TILE = torch.tensor([True, True, True, True, True, False, False, False])


def _play(env, gen, max_steps=20_000):
    obs = env.observe()
    trace, steps = [], 0
    while not bool(env.finished.all()):
        act = obs.active
        # every active row: a legal type, a legal tile of it
        assert bool(obs.legal["action_type"][act].any(-1).all())
        a = env.random_actions(obs, gen)
        t = a["action_type"]
        assert bool(obs.legal["action_type"][act, t[act]].all())
        tiles = obs.legal["tile_cond"][torch.arange(t.shape[0]), t]
        assert bool(tiles[act, a["tile_idx"][act]].all())
        trace.append((obs.obs["tiles_seq"][act].sum().item(), t[act].tolist()))
        obs, rew, done, info = env.step(a)
        for e in torch.nonzero(done).flatten().tolist():
            assert torch.allclose(rew[e].sort(descending=True).values, PLACEMENT)
            assert int(info["episode_len"][e]) > 20 and bool(info["terminated"][e])
        steps += 1
        assert steps < max_steps
    return trace


def test_env_rows_masks_rewards_and_determinism():
    env = MahjongVecEnv(3, seed=5)
    obs = env.observe()
    assert env.obs_schema() == {"tiles_seq": ((34, 37), torch.float32), "players_seq": ((4, 12), torch.float32),
                                "table_seq": ((12,), torch.float32)}
    assert obs.obs["tiles_seq"].shape == (12, 34, 37) and obs.legal["tile_cond"].shape == (12, 8, 34)
    # the dealer (seat 0 of each game) acts first: its own-turn frame (table column 9)
    assert obs.active.tolist() == [True, False, False, False] * 3
    assert bool((obs.obs["table_seq"][obs.active, 9] == 1).all())
    assert bool(obs.new_episode.all()) and bool((obs.step_idx == 0).all())
    # tile planes are the converter's quarters / 4 (data_processing/games/mahjong.py)
    q = obs.obs["tiles_seq"] * 4
    assert torch.equal(q, q.round()) and float(q.max()) <= 255
    assert torch.equal(obs.stream_seats, torch.eye(4, dtype=torch.bool).repeat(3, 1))
    trace = _play(env, torch.Generator().manual_seed(0))
    assert env.failures == 0
    # the same seeds and actions: the same games
    env2 = MahjongVecEnv(3, seed=5)
    assert _play(env2, torch.Generator().manual_seed(0)) == trace
    # the returns are the final points, conserved (leftover riichi sticks go to the top)
    assert all(sum(env.scores(e)) == 100_000 for e in range(3))


def test_autoreset_and_threads_give_the_same_games():
    a = MahjongVecEnv(4, seed=11, autoreset=True, threads=1)
    b = MahjongVecEnv(4, seed=11, autoreset=True, threads=3)
    ga, gb = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    oa, ob = a.observe(), b.observe()
    ends = 0
    for _ in range(3000):
        aa, ab = a.random_actions(oa, ga), b.random_actions(ob, gb)
        oa, ra, da, _ = a.step(aa)
        ob, rb, db, _ = b.step(ab)
        assert torch.equal(ra, rb) and torch.equal(da, db)
        assert torch.equal(oa.obs["tiles_seq"], ob.obs["tiles_seq"])
        ends += int(da.sum())
    assert ends >= 4 and a.failures == 0
    assert not bool(a.finished.any())                    # autoreset: always a game running


def test_il_then_rl_on_the_simulator(gmdata, tmp_path):  # noqa: F811
    from training import train_multi_game, train_rl_joint

    subprocess.run([gmdata, "mahjong", "convert", "--input", f"{FIX}/mahjong", "--out", str(tmp_path / "il")],
                   check=True, capture_output=True)
    core = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
            "core_type=transformer", "core_layers=1", "moe_experts=4", "temporal_mixer=recurrent"]
    train_multi_game.main(["--data", f"mahjong={tmp_path / 'il'}", "--steps", "3", "--batch_size", "2",
                           "--warmup_steps", "1", "--eval_every", "0", "--device", "cpu", "--il_legal_mask",
                           "--save_dir", str(tmp_path / "ck"), "--override", *core])
    res = train_rl_joint.main([
        "--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "mahjong", "--tokens_per_iter", "300",
        "--envs", "mahjong=2", "--epochs", "1", "--minibatches", "2", "--snapshot_dir", "memory",
        "--device", "cpu", "--seed", "1", "--iterations", "1", "--save_dir", str(tmp_path / "rl")])
    row = res["history"][-1]["games"]["mahjong"]
    assert row["tokens"] >= 1
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    assert np.isfinite(row["entropy"]) and np.isfinite(row["value_loss"])
    json.dumps(res["history"][-1])
