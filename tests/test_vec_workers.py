"""Tests for the env worker pool (env/vec_workers.py) and its integration
with the vectorized collector: parallel stepping must be observationally
identical to in-process stepping."""

import math

import pytest
import torch

from env.generals_env import GeneralsEnv
from env.vec_workers import EnvWorkerPool
from model.config import ModelConfig
from model.network import GeneralsNetwork
from training.train_rl import collect_selfplay_episodes_vec

TINY = dict(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1, trm_L_cycles=2,
            trm_block_layers=1)

ENV_KWARGS = dict(grid_h=25, grid_w=25, max_turns=8)
ENV_ATTRS = dict(breakdown_pid=0)


@pytest.fixture(scope="module")
def pool():
    p = EnvWorkerPool(n_envs=4, n_workers=2,
                      env_kwargs=ENV_KWARGS, env_attrs=ENV_ATTRS)
    yield p
    p.close()


def test_pool_reset_and_step(pool):
    obs = pool.reset_all()
    # slots 1..3 are remote; slot 0 is the caller's
    assert sorted(obs.keys()) == [1, 2, 3]
    for s, pair in obs.items():
        for pid in (0, 1):
            assert pair[pid]["grid"].shape == (12, 25, 25)
            assert "capture_mask" in pair[pid]  # full obs, nothing dropped

    results = pool.step({s: (None, None, 0, 0) for s in (1, 2, 3)})
    assert sorted(results.keys()) == [1, 2, 3]
    for s, (pair, rewards, done, info, vd) in results.items():
        assert isinstance(rewards, tuple) and len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)
        assert "turn" in info and "illegal" in info
        assert len(vd) == 2 and len(vd[0]) == 4


def test_pool_truncation_scores(pool):
    pool.reset_all()
    ts = pool.truncation_scores(2)
    assert len(ts) == 2
    assert all(math.isfinite(t) for t in ts)


def test_pipe_payload_never_carries_torch_tensors():
    # Torch tensors cross the pipe via fd-sharing and exhaust the fd
    # budget at production scale ("received 0 items of ancdata") — the
    # worker must convert to numpy before send, the parent back to torch.
    import numpy as np
    from env.vec_workers import _obs_to_numpy, _obs_to_torch

    env = GeneralsEnv(**ENV_KWARGS)
    pair = env.reset()
    wire = _obs_to_numpy(pair)
    for pid in (0, 1):
        for k, v in wire[pid].items():
            assert not isinstance(v, torch.Tensor), (pid, k)
            assert isinstance(v, np.ndarray), (pid, k)
    back = _obs_to_torch(wire)
    for pid in (0, 1):
        for k, v in back[pid].items():
            assert isinstance(v, torch.Tensor), (pid, k)
            assert torch.equal(v, pair[pid][k]), (pid, k)


def test_vec_collect_with_pool():
    torch.manual_seed(0)
    cfg = ModelConfig(hl_gauss_value=True, hl_gauss_bins=32,
                      temporal_mixer="recurrent", **TINY)
    net = GeneralsNetwork(cfg)
    env0 = GeneralsEnv(**ENV_KWARGS)
    p = EnvWorkerPool(n_envs=3, n_workers=2,
                      env_kwargs=ENV_KWARGS, env_attrs=ENV_ATTRS)
    try:
        bufs, replay, totals = collect_selfplay_episodes_vec(
            net, [env0], torch.device("cpu"),
            max_steps=8, gamma=1.0, gae_lambda=0.9,
            record=True, act_steps=1, legal_mask=True,
            worker_pool=p,
        )
    finally:
        p.close()
    assert len(bufs) == 6  # 3 envs x 2 seats
    for buf in bufs:
        assert 0 < len(buf) <= 8
        assert buf.dones[-1]
        assert all(math.isfinite(lp) for lp in buf.log_probs)
        assert torch.isfinite(buf.advantages).all()
    # replay follows the LOCAL slot-0 env
    assert replay is not None
    assert len(replay) == len(bufs[0]) + 1
    for k in ("r_winlose", "r_invalid"):
        assert math.isfinite(totals[k])
