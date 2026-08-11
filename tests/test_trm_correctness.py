"""TRM correctness audit (user directive: TRM must be CORRECT and able
to predict when to halt).

1. Adaptive per-row halting must be computation-equivalent: a row that
   halts at depth k inside a shrinking batch must produce the same
   outputs as running that row alone at fixed act_steps=k.
2. Depth-k rollout and PPO replay must agree (ratio ≈ 1 at zero lr) —
   the cache-incremental step() path and the sequence forward() path
   run the same recursion.
3. Recursion is not a no-op: deeper pondering changes the outputs.
4. The halt-consistency loss works on the joint decoder's views.
"""

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


def _make(halt_max=4, joint=False):
    torch.manual_seed(0)
    cfg = ModelConfig(hl_gauss_value=True, hl_gauss_bins=32,
                      temporal_mixer="recurrent", halt_max_steps=halt_max,
                      joint_decoder=joint, **TINY)
    net = GeneralsNetwork(cfg)
    net.eval()
    return net, cfg


def _obs(cfg, b):
    torch.manual_seed(3)
    return (torch.randn(b, 12, 25, 25), torch.randn(b, cfg.global_features),
            torch.ones(b, 1, 25, 25, dtype=torch.bool),
            torch.ones(b, 4, 25, 25, dtype=torch.bool))


def test_adaptive_row_matches_fixed_depth():
    net, cfg = _make(halt_max=4)
    # input-dependent halting so rows diverge and the batch shrinks.
    # A random q_head gives large positive low-variance logits — measure
    # the depth-1 spread and bias to its median so the batch splits.
    torch.manual_seed(11)
    torch.nn.init.normal_(net.trm.q_head.weight, std=1.5)
    torch.nn.init.zeros_(net.trm.q_head.bias)
    b = 6
    grid, glb, lm, dm = _obs(cfg, b)
    with torch.no_grad():
        q1 = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                      act_steps=1)["q_halt"]
    torch.nn.init.constant_(net.trm.q_head.bias, -float(q1.median()))
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=None)
    depths = out["act_depth"].tolist()
    assert len(set(depths)) > 1, "need divergent depths to test shrinking"
    for r in range(b):
        with torch.no_grad():
            ref = net.step(grid[r:r + 1], glb[r:r + 1],
                           legal_mask=lm[r:r + 1],
                           direction_mask=dm[r:r + 1],
                           act_steps=depths[r])
        assert torch.allclose(out["source_logits"][r],
                              ref["source_logits"][0], atol=1e-4), (
            f"row {r} (depth {depths[r]}): adaptive != fixed")
        assert torch.allclose(out["value"][r], ref["value"][0], atol=1e-4)


@pytest.mark.parametrize("depth", [2, 4])
def test_depth_rollout_replay_parity(depth):
    net, cfg = _make()
    envs = [GeneralsEnv(grid_h=25, grid_w=25, max_turns=8)
            for _ in range(2)]
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=8, gamma=1.0, gae_lambda=0.9,
        act_steps=[depth, depth], legal_mask=True,
    )
    crit = RLLoss(hl_gauss=True, hl_gauss_vmax=cfg.hl_gauss_vmax,
                  hl_gauss_sigma=cfg.hl_gauss_sigma)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    metrics = ppo_update(
        net, None, bufs, crit, opt, torch.device("cpu"), cfg,
        kl_weight=0.0, invalidity_weight=0.0, legal_masked=True,
        act_steps=1,  # per-buffer act_steps must override this
    )
    # zero-lr replay of the just-collected rollout: ratios must be ~1
    assert metrics["approx_kl"] < 1e-3, (
        f"depth-{depth} replay diverges from rollout "
        f"(approx_kl={metrics['approx_kl']:.5f})")


def test_deeper_pondering_changes_outputs():
    net, cfg = _make()
    grid, glb, lm, dm = _obs(cfg, 2)
    with torch.no_grad():
        o1 = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                      act_steps=1)
        o4 = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                      act_steps=4)
    diff = (o1["source_logits"] - o4["source_logits"]).abs().max()
    assert diff > 1e-3, "recursion is a no-op — deeper thinking changed nothing"


def test_halt_loss_on_joint_decoder():
    net, cfg = _make(joint=True)
    envs = [GeneralsEnv(grid_h=25, grid_w=25, max_turns=6)
            for _ in range(2)]
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=6, gamma=1.0, gae_lambda=0.9,
        act_steps=[1, 2], legal_mask=True,
    )
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
