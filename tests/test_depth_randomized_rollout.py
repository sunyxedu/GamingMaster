"""Tests for depth-randomized rollout (TRM deep-supervision analog):
per-env pondering depths in the vectorized collector, per-buffer depth
replay in the PPO update."""

import math

import pytest
import torch

from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.losses import RLLoss
from model.network import GeneralsNetwork
from training.train_rl import collect_selfplay_episodes_vec, ppo_update

TINY = dict(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1, trm_L_cycles=2,
            trm_block_layers=1)


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig(hl_gauss_value=True, hl_gauss_bins=32,
                       temporal_mixer="recurrent", **TINY)


@pytest.fixture(scope="module")
def net(cfg):
    torch.manual_seed(0)
    return GeneralsNetwork(cfg)


def _envs(turn_limits):
    return [GeneralsEnv(grid_h=25, grid_w=25, max_turns=t)
            for t in turn_limits]


def test_mixed_depths_collect(net):
    envs = _envs([8, 8, 8])
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=8, gamma=1.0, gae_lambda=0.9,
        record=False, act_steps=[1, 2, 1], legal_mask=True,
    )
    assert len(bufs) == 6
    # buffers carry their env's depth (interleaved seat0/seat1 per env)
    assert [b.act_steps for b in bufs] == [1, 1, 2, 2, 1, 1]
    for buf in bufs:
        assert 0 < len(buf) <= 8
        assert buf.dones[-1]
        assert all(math.isfinite(lp) for lp in buf.log_probs)
        assert torch.isfinite(buf.advantages).all()


def test_mixed_depths_early_finish_compaction(net):
    # different turn limits inside the SAME depth group and across groups:
    # per-group cache compaction must survive both
    envs = _envs([4, 10, 4, 10])
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=10, gamma=1.0, gae_lambda=0.9,
        record=False, act_steps=[1, 1, 2, 2], legal_mask=True,
    )
    assert len(bufs) == 8
    lens = [len(b) for b in bufs[0::2]]
    assert lens[0] <= 6 and lens[2] <= 6
    assert lens[1] > lens[0] and lens[3] > lens[2]
    for buf in bufs:
        assert buf.dones[-1]


def test_depth_list_length_mismatch_raises(net):
    envs = _envs([6, 6])
    with pytest.raises(ValueError):
        collect_selfplay_episodes_vec(
            net, envs, torch.device("cpu"),
            max_steps=6, act_steps=[1, 2, 4], legal_mask=True,
        )


def test_ppo_update_replays_per_buffer_depth(net, cfg):
    envs = _envs([6, 6])
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=6, gamma=1.0, gae_lambda=0.9,
        record=False, act_steps=[1, 2], legal_mask=True,
    )
    crit = RLLoss(hl_gauss=True, hl_gauss_vmax=cfg.hl_gauss_vmax,
                  hl_gauss_sigma=cfg.hl_gauss_sigma)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    metrics = ppo_update(
        net, None, bufs, crit, opt, torch.device("cpu"), cfg,
        kl_weight=0.0, invalidity_weight=0.0, legal_masked=True,
        act_steps=1,
    )
    assert math.isfinite(metrics["policy_loss"])
    assert math.isfinite(metrics["approx_kl"])
    # depth-1 buffers must replay to (near-)zero ratio drift: their
    # log-probs recompute exactly; sanity ceiling on approx_kl
    assert metrics["approx_kl"] < 0.5
