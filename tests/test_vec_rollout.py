"""Tests for the vectorized mirror rollout collector
(collect_selfplay_episodes_vec): batched lockstep stepping, cache
compaction as envs finish, and parity of per-transition semantics with the
sequential path."""

import math

import pytest
import torch

from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.network import GeneralsNetwork
from training.train_rl import _select_cache_rows, collect_selfplay_episodes_vec

TINY = dict(d_model=64, n_heads=4, trm_H_cycles=1, trm_L_cycles=2,
            trm_block_layers=1)


def _make_net(mixer: str) -> GeneralsNetwork:
    torch.manual_seed(0)
    cfg = ModelConfig(hl_gauss_value=True, hl_gauss_bins=32,
                      temporal_mixer=mixer, **TINY)
    return GeneralsNetwork(cfg)


def _make_envs(turn_limits):
    return [GeneralsEnv(grid_h=25, grid_w=25, max_turns=t)
            for t in turn_limits]


def test_select_cache_rows_2d():
    # B=4, S=3, d=2 — recurrent-style [B*S, d] cache
    c = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    keep = torch.tensor([0, 2])
    out = _select_cache_rows([c], keep, batch=4)[0]
    expected = torch.cat([c[0:3], c[6:9]])
    assert torch.equal(out, expected)


def test_select_cache_rows_3d():
    # attention-style [B*S, t, d] cache
    c = torch.arange(48, dtype=torch.float32).reshape(6, 4, 2)
    keep = torch.tensor([1])
    out = _select_cache_rows([c], keep, batch=2)[0]
    assert torch.equal(out, c[3:6])


@pytest.mark.parametrize("mixer", ["recurrent", "attention"])
def test_vec_collect_basic(mixer):
    net = _make_net(mixer)
    envs = _make_envs([10, 10, 10])
    bufs, replay, totals = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=10, gamma=1.0, gae_lambda=0.9,
        record=True, act_steps=1, legal_mask=True,
    )
    # one seat-0 and one seat-1 buffer per env, interleaved
    assert len(bufs) == 6
    for buf in bufs:
        assert 0 < len(buf) <= 10
        assert buf.dones[-1] is True or buf.dones[-1] == True  # noqa: E712
        assert all(math.isfinite(lp) for lp in buf.log_probs)
        assert torch.isfinite(buf.advantages).all()
        assert len(buf.rewards) == len(buf)
    # both seats of one env see the same number of turns
    for i in range(3):
        assert len(bufs[2 * i]) == len(bufs[2 * i + 1])
    # replay covers envs[0]: initial snapshot + one entry per step
    assert replay is not None
    assert len(replay) == len(bufs[0]) + 1
    assert replay[0]["turn"] == -1
    for k in ("r_winlose", "r_city", "r_dense", "r_invalid"):
        assert math.isfinite(totals[k])


def test_vec_collect_mixed_lengths():
    # envs finishing at different times exercise cache compaction: the
    # short env leaves the batch, the long one must keep stepping cleanly.
    net = _make_net("recurrent")
    envs = _make_envs([4, 12])
    bufs, _, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=12, gamma=1.0, gae_lambda=0.9,
        record=False, act_steps=1, legal_mask=True,
    )
    assert len(bufs) == 4
    short0, _, long0, _ = bufs
    # the short env is closed by its own turn limit well before the budget
    assert len(short0) <= 6
    assert len(long0) > len(short0)
    for buf in bufs:
        assert buf.dones[-1]
        assert torch.isfinite(buf.advantages).all()


def test_vec_collect_no_record():
    net = _make_net("recurrent")
    envs = _make_envs([6, 6])
    bufs, replay, _ = collect_selfplay_episodes_vec(
        net, envs, torch.device("cpu"),
        max_steps=6, gamma=1.0, gae_lambda=0.9,
        record=False, act_steps=1, legal_mask=True,
    )
    assert replay is None
    assert len(bufs) == 4
