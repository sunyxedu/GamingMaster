"""Tests for the AverageJoe-style joint-action decoder
(cfg.joint_decoder=True): one categorical over 8N+1 arrows, factorized
views as exact marginals, full rollout/replay parity."""

import math

import pytest
import torch

from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.losses import RLLoss
from model.network import GeneralsNetwork
from train_rl import collect_selfplay_episodes_vec, ppo_update

TINY = dict(d_model=64, n_heads=4, trm_H_cycles=1, trm_L_cycles=2,
            trm_block_layers=1)


def _make(joint=True):
    torch.manual_seed(0)
    cfg = ModelConfig(hl_gauss_value=True, hl_gauss_bins=32,
                      temporal_mixer="recurrent", joint_decoder=joint,
                      **TINY)
    net = GeneralsNetwork(cfg)
    net.eval()  # rollout/replay parity needs frozen BN stats + no dropout
    return net, cfg


def _obs(cfg, b=3):
    torch.manual_seed(1)
    # dir_mask layout matches the env: [4, H, W] (direction-major)
    return (torch.randn(b, 12, 25, 25), torch.randn(b, cfg.global_features),
            torch.rand(b, 1, 25, 25) > 0.7,
            torch.rand(b, 4, 25, 25) > 0.2)


def test_joint_step_shapes_and_legality():
    net, cfg = _make()
    grid, glb, lm, dm = _obs(cfg)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=1)
    N = 625
    assert out["joint_logits"].shape == (3, 8 * N + 1)
    assert out["source_logits"].shape == (3, N)
    assert out["direction_logits"].shape == (3, 4)
    assert out["pass_logits"].shape == (3,)
    # sampled actions legal: source only on legal cells (unless pass)
    for b in range(3):
        if not bool(out["action_pass"][b]):
            src = int(out["action_source"][b])
            assert bool(lm.reshape(3, -1)[b, src])
            d = int(out["action_direction"][b])
            assert bool(dm.reshape(3, 4, 625)[b, d, src])
    assert torch.isfinite(out["action_log_prob"]).all()


def test_joint_replay_matches_rollout_log_prob():
    net, cfg = _make()
    grid, glb, lm, dm = _obs(cfg, b=4)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=1)
        # replay the same single frame through forward()
        pred = net(grid.unsqueeze(1), glb.unsqueeze(1),
                   legal_mask=lm.unsqueeze(1),
                   source_y=(out["action_source"] // 25).unsqueeze(1),
                   source_x=(out["action_source"] % 25).unsqueeze(1),
                   act_steps=1,
                   direction_mask=dm.unsqueeze(1))
        actions = {
            "source_idx": out["action_source"].unsqueeze(1),
            "direction": out["action_direction"].unsqueeze(1),
            "split": out["action_split"].unsqueeze(1),
            "passed": out["action_pass"].unsqueeze(1),
        }
        lp, ent, heads = net.compute_log_prob(
            pred, actions, legal={"source": lm.unsqueeze(1)})
    # step() and forward() share weights and masks → identical log-probs
    assert torch.allclose(lp.squeeze(1), out["action_log_prob"], atol=1e-4)
    assert torch.isfinite(ent).all()
    assert set(heads) == {"ent_pass", "ent_src", "ent_dir", "ent_spl"}


def test_joint_views_are_marginals():
    net, cfg = _make()
    grid, glb, lm, dm = _obs(cfg)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=1)
    # source view must equal logsumexp over the 8 arrow channels
    N = 625
    joint = out["joint_logits"][:, :8 * N].reshape(3, 8, N)
    # joint is MASKED; source_logits view is built from raw arrows —
    # so only check consistency where all arrows are unmasked
    finite = (joint > -1e8).all(dim=1)
    lse = torch.logsumexp(joint, dim=1)
    diff = (lse - out["source_logits"]).abs()
    assert diff[finite].max() < 1e-3


def test_joint_end_to_end_collect_and_update():
    net, cfg = _make()
    envs = [GeneralsEnv(grid_h=25, grid_w=25, max_turns=8)
            for _ in range(2)]
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=8, gamma=1.0, gae_lambda=0.9,
        act_steps=[1, 2], legal_mask=True,
    )
    assert len(bufs) == 4
    for buf in bufs:
        assert buf.dones[-1]
        assert all(math.isfinite(lp) for lp in buf.log_probs)
    crit = RLLoss(hl_gauss=True, hl_gauss_vmax=cfg.hl_gauss_vmax,
                  hl_gauss_sigma=cfg.hl_gauss_sigma)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    metrics = ppo_update(
        net, None, bufs, crit, opt, torch.device("cpu"), cfg,
        kl_weight=0.0, invalidity_weight=0.0, legal_masked=True,
        act_steps=1,
    )
    assert math.isfinite(metrics["policy_loss"])
    # zero-lr replay of fresh rollouts → ratios ≈ 1, tiny approx_kl
    assert metrics["approx_kl"] < 0.05


def test_joint_with_magnet_anchor():
    net, cfg = _make()
    envs = [GeneralsEnv(grid_h=25, grid_w=25, max_turns=6)
            for _ in range(2)]
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=6, gamma=1.0, gae_lambda=0.9,
        act_steps=1, legal_mask=True,
    )
    crit = RLLoss(hl_gauss=True, hl_gauss_vmax=cfg.hl_gauss_vmax,
                  hl_gauss_sigma=cfg.hl_gauss_sigma)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    metrics = ppo_update(
        net, None, bufs, crit, opt, torch.device("cpu"), cfg,
        kl_weight=0.01, invalidity_weight=0.0, legal_masked=True,
        magnet=True, act_steps=1,
    )
    # magnet KL consumes the factorized views of the joint policy
    assert math.isfinite(metrics["kl_reg"])
    assert metrics["kl_reg"] > 0
