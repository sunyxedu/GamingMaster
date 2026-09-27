"""env/vec_hok.py against a stand-in worker (tests/fixtures/hok_fake_worker.py: the same
protocol as env/hok_worker.py, whose SDK and licensed gamecore are cluster-only): the
VecObs layout (two seat rows per env, the legal tables of model/games/hok.py), rewards
(shaping per step, ±outcome to the winner / loser when a crystal falls), truncation,
autoreset, and a rollout of the hok network through the env."""

from __future__ import annotations

import os
import sys

import pytest
import torch

from env.vec_hok import HokVecEnv
from model.games import build_game_network

FAKE = [sys.executable, os.path.join(os.path.dirname(__file__), "fixtures", "hok_fake_worker.py")]


@pytest.fixture
def env():
    e = HokVecEnv(3, worker_cmd=FAKE, seed=0, shaping=1.0, outcome=1.0, port_base=40000)
    yield e
    e.close()


def test_layout_and_legal_tables(env):
    obs = env.reset()
    assert obs.n_rows == 6 and obs.obs["features"].shape == (6, 725)
    assert obs.legal["button"].shape == (6, 12) and obs.legal["target"].shape == (6, 12, 8)
    assert obs.legal["sub"].shape == (6, 12, 6) and obs.legal["sub"][..., 0].all()
    assert torch.equal(obs.to_play, torch.tensor([0, 1, 0, 1, 0, 1]))
    assert obs.active.all() and obs.new_episode.all() and not obs.forced.any()
    assert torch.equal(obs.stream_seats.long().argmax(-1), obs.to_play)
    assert env.obs_schema() == {"features": ((725,), torch.float32)}


def test_rewards_outcome_and_finish(env):
    obs = env.reset()
    total = torch.zeros(3, 2)
    for _ in range(20):
        obs, rew, done, info = env.step(env.random_actions(obs))
        total += rew
        if env.finished.all():
            break
    assert env.finished.all() and not obs.active.any()
    # port 40000 (even): camp 1 = seat 0 wins; 40002 even too; 40004 even
    outcome = total - 0.01 * torch.tensor([[5 + 40000 % 3], [5 + 40002 % 3], [5 + 40004 % 3]]).float()
    torch.testing.assert_close(outcome, torch.tensor([[1.0, -1.0]] * 3), rtol=0, atol=1e-5)


def test_truncation_and_autoreset():
    e = HokVecEnv(2, worker_cmd=FAKE, seed=1, max_steps=2, port_base=41000, autoreset=True)
    try:
        obs = e.reset()
        obs, rew, done, info = e.step(e.random_actions(obs))
        assert not done.any()
        obs, rew, done, info = e.step(e.random_actions(obs))
        assert done.all() and info["truncated"].all() and not info["terminated"].any()
        assert obs.active.all() and obs.new_episode.all() and (obs.step_idx == 0).all()
    finally:
        e.close()


def test_network_rollout_through_the_env(env):
    torch.manual_seed(0)
    net = build_game_network("hok", d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
                             core_type="transformer", core_layers=1,
                             temporal_mixer="recurrent").eval()
    obs = env.reset()
    cache = None
    for t in range(4):
        with torch.no_grad():
            out = net.step(obs.obs, legal=obs.legal, st_cache=cache, step_idx=t)
        cache = out["st_cache"]
        b = out["action_button"]
        assert bool(obs.legal["button"][torch.arange(6), b].all())
        obs, rew, done, info = env.step(out)
        assert torch.equal(info["applied"]["button_idx"], b)
