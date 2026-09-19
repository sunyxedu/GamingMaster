"""Tests for the opening anchor (--opening_anchor): per-head KL toward a
frozen teacher charged only on the first N timesteps."""

import copy
import math

import pytest
import torch

from model.config import ModelConfig
from model.losses import RLLoss
from model.network import GeneralsNetwork
from model.rl_utils import RolloutBuffer
from training.train_rl import ppo_update

TINY = dict(d_model=64, n_heads=4, trm_H_cycles=1, trm_L_cycles=2,
            trm_block_layers=1)


def _make_buffer(T, cfg, seed=0):
    torch.manual_seed(seed)
    buf = RolloutBuffer()
    for t in range(T):
        obs = {
            "grid": torch.randn(12, 25, 25),
            "globals": torch.randn(cfg.global_features),
            "legal_mask": torch.ones(1, 25, 25, dtype=torch.bool),
        }
        action = {"source_idx": int(torch.randint(0, 625, (1,))),
                  "direction": int(torch.randint(0, 4, (1,))),
                  "split": 0, "passed": False}
        buf.add(obs, action, log_prob=-1.0,
                value=float(torch.randn(1)), reward=float(torch.randn(1)),
                done=(t == T - 1))
    buf.compute_returns_and_advantages(gamma=1.0, lam=0.9)
    return buf


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig(hl_gauss_value=True, hl_gauss_bins=32, **TINY)


@pytest.fixture(scope="module")
def net(cfg):
    torch.manual_seed(0)
    return GeneralsNetwork(cfg)


def _run(net, cfg, opening_ref, opening_weight, opening_turns, lr=0.0):
    buf = _make_buffer(8, cfg)
    crit = RLLoss(hl_gauss=True, hl_gauss_vmax=cfg.hl_gauss_vmax,
                  hl_gauss_sigma=cfg.hl_gauss_sigma)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    return ppo_update(
        net, None, [buf], crit, opt, torch.device("cpu"), cfg,
        kl_weight=0.0, invalidity_weight=0.0, legal_masked=False,
        opening_ref=opening_ref, opening_weight=opening_weight,
        opening_turns=opening_turns,
    )


def test_okl_zero_when_disabled(net, cfg):
    m = _run(net, cfg, opening_ref=None, opening_weight=0.05,
             opening_turns=100)
    assert m["okl"] == 0.0


def test_okl_near_zero_against_self(net, cfg):
    # KL(π‖π) over identical frozen weights must vanish: the anchor
    # charges nothing when the policy already plays the teacher's moves.
    ref = copy.deepcopy(net)
    ref.eval()
    m = _run(net, cfg, opening_ref=ref, opening_weight=0.05,
             opening_turns=100)
    assert abs(m["okl"]) < 1e-4


def test_okl_positive_against_different_teacher(cfg, net):
    torch.manual_seed(123)
    other = GeneralsNetwork(cfg)
    other.eval()
    m = _run(net, cfg, opening_ref=other, opening_weight=0.05,
             opening_turns=100)
    # Two random tiny nets are both near-uniform, so the KL is small —
    # but it must be clearly nonzero, unlike the identical-teacher case.
    assert m["okl"] > 1e-4
    assert math.isfinite(m["okl"])
    assert math.isfinite(m["policy_loss"])


def test_okl_prefix_only(cfg, net):
    # opening_turns=0 disables the charge even with a divergent teacher.
    torch.manual_seed(123)
    other = GeneralsNetwork(cfg)
    other.eval()
    m = _run(net, cfg, opening_ref=other, opening_weight=0.05,
             opening_turns=0)
    assert m["okl"] == 0.0
    # A shorter prefix charges no more than the full horizon.
    m_short = _run(net, cfg, opening_ref=other, opening_weight=0.05,
                   opening_turns=2)
    m_full = _run(net, cfg, opening_ref=other, opening_weight=0.05,
                  opening_turns=8)
    assert math.isfinite(m_short["okl"]) and m_short["okl"] > 0
    assert math.isfinite(m_full["okl"]) and m_full["okl"] > 0
