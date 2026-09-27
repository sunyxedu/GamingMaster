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
from training.train_rl import collect_selfplay_episodes_vec, ppo_update

TINY = dict(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1, trm_L_cycles=2,
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


def test_none_depth_equals_fixed_full_depth():
    """act_steps=None is the fixed full depth (cfg.halt_max_steps), per row
    and batched alike — no data-dependent early exit at inference."""
    net, cfg = _make(halt_max=4)
    torch.manual_seed(11)
    torch.nn.init.normal_(net.trm.q_head.weight, std=1.5)
    torch.nn.init.zeros_(net.trm.q_head.bias)
    b = 6
    grid, glb, lm, dm = _obs(cfg, b)
    with torch.no_grad():
        out = net.step(grid, glb, legal_mask=lm, direction_mask=dm,
                       act_steps=None)
    assert out["act_depth"].tolist() == [4] * b
    for r in range(b):
        with torch.no_grad():
            ref = net.step(grid[r:r + 1], glb[r:r + 1],
                           legal_mask=lm[r:r + 1],
                           direction_mask=dm[r:r + 1],
                           act_steps=4)
        assert torch.allclose(out["source_logits"][r],
                              ref["source_logits"][0], atol=1e-4)
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


def test_wsd_schedule_warmup_stable_decay():
    """Warmup-Stable-Decay: the reference warmup, a constant rate until the
    last decay_frac of the steps, then a decay to min_ratio; nothing before
    the decay depends on total_steps (an extended run is the same until its
    own decay)."""
    import math

    from model.optim import wsd_lr_lambda

    f = wsd_lr_lambda(10, 100, 0.2, 0.1)
    assert f(0) == 0.0 and f(5) == 0.5 and f(10) == 1.0
    assert f(79) == 1.0 and f(80) == 1.0                     # decay starts at step 80
    assert math.isclose(f(90), 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert math.isclose(f(100), 0.1) and math.isclose(f(150), 0.1)
    lin = wsd_lr_lambda(10, 100, 0.2, 0.0, shape="linear")
    assert math.isclose(lin(90), 0.5) and lin(100) == 0.0
    longer = wsd_lr_lambda(10, 1000, 0.1, 0.1)
    assert all(f(s) == longer(s) for s in range(0, 80))     # same until the decay
    with pytest.raises(ValueError):
        wsd_lr_lambda(10, 100, 0.0, 0.1)


def test_wsd_takes_over_on_resume():
    """A resumed run's LambdaLR follows the NEW run's lambda (the state dict
    carries the step and base rates, not the function): a constant run
    resumed with --lr_schedule wsd and a larger --steps anneals at its end."""
    import torch

    from model.optim import trm_lr_lambda, wsd_lr_lambda

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    s1 = torch.optim.lr_scheduler.LambdaLR(opt, trm_lr_lambda(2, 10, 1.0))
    for _ in range(10):
        opt.step()
        s1.step()
    assert s1.get_last_lr()[0] == 1.0
    opt2 = torch.optim.SGD([p], lr=1.0)
    s2 = torch.optim.lr_scheduler.LambdaLR(opt2, wsd_lr_lambda(2, 20, 0.5, 0.0, "linear"))
    s2.load_state_dict(s1.state_dict())
    lrs = []
    for _ in range(10):
        opt2.step()
        s2.step()
        lrs.append(s2.get_last_lr()[0])
    assert lrs[0] == 0.9 and lrs[-1] == 0.0 and lrs == sorted(lrs, reverse=True)
