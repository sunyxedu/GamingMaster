"""Inference depth (reference TRM: always the full halt_max_steps) and the
RL halt-head self-consistency calibration loss."""

import math

import torch

from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.losses import RLLoss
from model.network import GeneralsNetwork
from training.train_rl import collect_selfplay_episode, collect_selfplay_episodes_vec, ppo_update

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


def test_default_inference_runs_full_depth_even_when_q_halts():
    """Reference TRM: q_halt only shortens TRAINING; inference always runs
    all halt_max_steps supervision steps, whatever q_halt says."""
    net, cfg = _make_net(halt_max=4)
    net.eval()                                           # no dropout noise
    torch.nn.init.constant_(net.trm.q_head.bias, 5.0)   # "halt now" everywhere
    grid, glb, lm, dm = _obs(cfg)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=None)
        ref = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=4)
    assert torch.equal(out["act_depth"], torch.full((3,), 4, dtype=torch.long))
    assert torch.allclose(out["source_logits"], ref["source_logits"])


def test_fixed_depth_reports_constant_act_depth():
    net, cfg = _make_net()
    grid, glb, lm, dm = _obs(cfg)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=2)
    assert torch.equal(out["act_depth"],
                       torch.full((3,), 2, dtype=torch.long))


def test_collect_none_depth_resolves_to_full_depth_and_is_replayable():
    net, _ = _make_net(halt_max=3)
    env = GeneralsEnv(grid_h=25, grid_w=25, max_turns=6)
    bufs, _, _ = collect_selfplay_episode(
        net, net, env, torch.device("cpu"), max_steps=6,
        act_steps={0: None, 1: 1}, legal_mask=True,
        collect_buffers=True, collect_both=True)
    assert [b.act_steps for b in bufs] == [3, 1]


def test_vec_collector_none_depth_resolves_to_full_depth():
    net, _ = _make_net(halt_max=3)
    envs = [GeneralsEnv(grid_h=25, grid_w=25, max_turns=6) for _ in range(2)]
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"), max_steps=6,
        act_steps=[None, 1], legal_mask=True)
    assert sorted({b.act_steps for b in bufs}) == [1, 3]


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
