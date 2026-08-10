"""Tests for adaptive ACT (per-row q_halt-gated pondering) and the
halt-head self-consistency calibration loss."""

import math

import pytest
import torch

from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.losses import RLLoss
from model.network import GeneralsNetwork
from train_rl import (collect_selfplay_episode, collect_selfplay_episodes_vec,
                      ppo_update)

TINY = dict(d_model=64, n_heads=4, trm_H_cycles=1, trm_L_cycles=2,
            trm_block_layers=1)


def _make_net(halt_max=4):
    torch.manual_seed(0)
    cfg = ModelConfig(hl_gauss_value=True, hl_gauss_bins=32,
                      temporal_mixer="recurrent", halt_max_steps=halt_max,
                      **TINY)
    return GeneralsNetwork(cfg), cfg


def _obs(cfg, b=3):
    torch.manual_seed(1)
    return (torch.randn(b, 12, 25, 25), torch.randn(b, cfg.global_features),
            torch.ones(b, 1, 25, 25, dtype=torch.bool),
            torch.ones(b, 25, 25, 4, dtype=torch.bool))


def test_adaptive_runs_to_max_when_q_never_halts():
    net, cfg = _make_net(halt_max=3)
    # fresh q_head: zero weight, bias -5 → never halts → full depth
    grid, glb, lm, dm = _obs(cfg)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=None)
    assert torch.equal(out["act_depth"],
                       torch.full((3,), 3, dtype=torch.long))


def test_adaptive_halts_immediately_when_q_positive():
    net, cfg = _make_net(halt_max=4)
    torch.nn.init.constant_(net.trm.q_head.bias, 5.0)
    grid, glb, lm, dm = _obs(cfg)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=None)
    assert torch.equal(out["act_depth"],
                       torch.full((3,), 1, dtype=torch.long))


def test_fixed_depth_reports_constant_act_depth():
    net, cfg = _make_net()
    grid, glb, lm, dm = _obs(cfg)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=2)
    assert torch.equal(out["act_depth"],
                       torch.full((3,), 2, dtype=torch.long))


def test_adaptive_per_row_divergence_and_bounds():
    net, cfg = _make_net(halt_max=4)
    # nonzero q weights + zero bias → input-dependent halting
    torch.manual_seed(7)
    torch.nn.init.normal_(net.trm.q_head.weight, std=1.0)
    torch.nn.init.zeros_(net.trm.q_head.bias)
    grid, glb, lm, dm = _obs(cfg, b=6)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=None)
    d = out["act_depth"]
    assert d.min() >= 1 and d.max() <= 4
    assert torch.isfinite(out["source_logits"]).all()


def test_collect_adaptive_requires_no_buffers():
    net, _ = _make_net()
    env = GeneralsEnv(grid_h=25, grid_w=25, max_turns=6)
    with pytest.raises(ValueError):
        collect_selfplay_episode(
            net, net, env, torch.device("cpu"), max_steps=6,
            act_steps={0: None, 1: 1}, legal_mask=True,
            collect_buffers=True)


def test_collect_adaptive_reports_depth_telemetry():
    net, _ = _make_net(halt_max=3)
    env = GeneralsEnv(grid_h=25, grid_w=25, max_turns=6)
    _, _, rw = collect_selfplay_episode(
        net, net, env, torch.device("cpu"), max_steps=6,
        act_steps={0: None, 1: 2}, legal_mask=True,
        collect_buffers=False)
    assert rw["d0_n"] > 0
    mean_d = rw["d0_sum"] / rw["d0_n"]
    assert 1.0 <= mean_d <= 3.0


def test_vec_collector_rejects_adaptive():
    net, _ = _make_net()
    envs = [GeneralsEnv(grid_h=25, grid_w=25, max_turns=6) for _ in range(2)]
    with pytest.raises(ValueError):
        collect_selfplay_episodes_vec(
            net, envs, torch.device("cpu"), max_steps=6,
            act_steps=[None, 1], legal_mask=True)


def test_halt_consistency_loss_runs():
    net, cfg = _make_net()
    envs = [GeneralsEnv(grid_h=25, grid_w=25, max_turns=6) for _ in range(2)]
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"), max_steps=6,
        gamma=1.0, gae_lambda=0.9, act_steps=[1, 2], legal_mask=True)
    crit = RLLoss(hl_gauss=True, hl_gauss_vmax=cfg.hl_gauss_vmax,
                  hl_gauss_sigma=cfg.hl_gauss_sigma)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    metrics = ppo_update(
        net, None, bufs, crit, opt, torch.device("cpu"), cfg,
        kl_weight=0.0, invalidity_weight=0.0, legal_masked=True,
        act_steps=1, halt_weight=0.1, halt_ref_depth=3, halt_frac=1.0,
    )
    assert math.isfinite(metrics["halt_loss"])
    assert metrics["halt_loss"] > 0
    assert 0.0 <= metrics["halt_acc"] <= 1.0
