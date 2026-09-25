"""Tests for memory vision (per-player sighted-mountain persistence) and
unmasked-direction training (--dir_mask 0)."""

import math

import pytest
import torch

from env.generals_env import GeneralsEnv, _MOUNTAIN
from model.config import ModelConfig
from model.losses import RLLoss
from model.network import GeneralsNetwork
from training.train_rl import collect_selfplay_episodes_vec, ppo_update

TINY = dict(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1, trm_L_cycles=2,
            trm_block_layers=1)


def _far_mountain(env, min_dist=4):
    """A mountain far from BOTH generals (guaranteed out of vision)."""
    for idx in env.mountains:
        r, c = divmod(idx, env.grid_w)
        ok = True
        for g in env.generals:
            gr, gc = divmod(g, env.grid_w)
            if abs(r - gr) <= min_dist and abs(c - gc) <= min_dist:
                ok = False
                break
        if ok:
            return idx
    return None


def test_remembered_mountain_stays_encoded():
    env = GeneralsEnv(grid_h=25, grid_w=25, max_turns=50, n_mountains=30)
    env.reset(seed=5)
    idx = _far_mountain(env)
    assert idx is not None, "map has no mountain far from both generals"
    r, c = divmod(idx, env.grid_w)

    # out of vision and unremembered → generic fog obstacle (state 1)
    obs = env._get_observation(0)
    assert obs["grid"][1, r, c] == 1.0
    assert obs["grid"][4, r, c] == 0.0

    # inject memory: the SAME cell now encodes as a mountain (state 4)
    env.seen_mountains[0].add(idx)
    obs = env._get_observation(0)
    assert obs["grid"][4, r, c] == 1.0
    assert obs["grid"][1, r, c] == 0.0


def test_memory_is_per_player():
    env = GeneralsEnv(grid_h=25, grid_w=25, max_turns=50, n_mountains=30)
    env.reset(seed=5)
    idx = _far_mountain(env)
    assert idx is not None
    r, c = divmod(idx, env.grid_w)
    env.seen_mountains[0].add(idx)
    # player 1 has NOT seen it: still a fog obstacle in THEIR observation
    obs1 = env._get_observation(1)
    assert obs1["grid"][4, r, c] == 0.0
    assert obs1["grid"][1, r, c] == 1.0


def test_visible_mountain_enters_memory():
    env = GeneralsEnv(grid_h=25, grid_w=25, max_turns=50, n_mountains=0)
    env.reset(seed=3)
    # plant a mountain right next to player 0's general → in vision
    g = env.generals[0]
    gr, gc = divmod(g, env.grid_w)
    tr, tc = gr, gc + 1
    if tc >= env.real_w:
        tr, tc = gr, gc - 1
    env.map[tr][tc].terrain = _MOUNTAIN
    env._get_observation(0)
    assert tr * env.grid_w + tc in env.seen_mountains[0]
    # player 1 never saw it
    assert tr * env.grid_w + tc not in env.seen_mountains[1]


def test_reset_clears_memory():
    env = GeneralsEnv(grid_h=25, grid_w=25, max_turns=50, n_mountains=30)
    env.reset(seed=5)
    env.seen_mountains[0].add(env.mountains[0])
    env.reset(seed=6)
    assert env.seen_mountains[0] == set()


def test_memory_vision_toggle():
    env = GeneralsEnv(grid_h=25, grid_w=25, max_turns=50, n_mountains=30)
    env.reset(seed=5)
    idx = _far_mountain(env)
    assert idx is not None
    r, c = divmod(idx, env.grid_w)
    env.seen_mountains[0].add(idx)
    env.memory_vision = False
    obs = env._get_observation(0)
    assert obs["grid"][4, r, c] == 0.0  # pure instantaneous fog again


def test_unmasked_directions_collect_and_update():
    torch.manual_seed(0)
    cfg = ModelConfig(hl_gauss_value=True, hl_gauss_bins=32,
                      temporal_mixer="recurrent", **TINY)
    net = GeneralsNetwork(cfg)
    envs = [GeneralsEnv(grid_h=25, grid_w=25, max_turns=8) for _ in range(2)]
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=8, gamma=1.0, gae_lambda=0.9,
        act_steps=[1, 2], legal_mask=True, mask_directions=False,
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
        kl_weight=0.0, invalidity_weight=0.1, legal_masked=True,
        mask_directions=False, act_steps=1,
    )
    assert math.isfinite(metrics["policy_loss"])
    # direction-validity aux supervision must be ACTIVE when dirs unmasked
    assert metrics["inv_loss"] > 0
