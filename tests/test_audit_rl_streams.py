"""Seat privacy of the joint-RL rollout path (training/train_rl_joint.py,
env/league.py): a network acting for one seat must only ever be influenced by
that seat's own stream.

In a simultaneous fog-of-war game (generals) every (env, seat) is its own
row / temporal stream, and the learner steps ALL its rows in one batched
call — in mirror self-play both seats of an env, and in ``--actor fixed``
mode every row of the env (the opponent's rows are computed and discarded).
So the per-row outputs AND the per-stream Stage-1 state must be independent
of the other rows: perturbing the other seat's observations (its private
fog view) over a whole episode leaves this seat's logits, sampled actions,
values and cache bit-identical — for the StreamStepper path (any temporal
mixer) and for the fixed-batch ``initial_st_cache`` path (recurrent mixer).
The routing itself (which rows a policy is handed) is covered by
tests/test_vec_game.py::test_league_seat_streams_in_generals.
"""
from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from env.league import StreamStepper
from env.vec_generals import GeneralsVecEnv
from model.config import ModelConfig
from model.network import GeneralsNetwork

G = 10
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=2, encoder_channels=8, dropout=0.1,
            core_type="recurrent_depth", rd_prelude_layers=1, rd_recurrent_layers=1,
            rd_coda_layers=1, rd_mean_iters=2, rd_backprop_iters=1, grid_h=G, grid_w=G)
T = 6


def _frames(seed: int) -> List:
    env = GeneralsVecEnv(2, grid_h=G, grid_w=G, n_mountains=4, n_cities=2, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    obs = env.reset()
    out = []
    for _ in range(T):
        out.append(obs)
        obs, *_ = env.step(env.random_actions(obs, generator=gen))
    return out


def _mix(a, b, rows):
    """VecObs dicts of ``a`` with ``rows`` taken from ``b``."""
    def m(x, y):
        x = x.clone()
        x[rows] = y[rows]
        return x
    return ({k: m(v, b.obs[k]) for k, v in a.obs.items()},
            {k: m(v, b.legal[k]) for k, v in a.legal.items()})


def _net(mixer: str) -> GeneralsNetwork:
    torch.manual_seed(0)
    return GeneralsNetwork(ModelConfig(**TINY, temporal_mixer=mixer)).eval()


def _row_outputs(out: Dict, rows) -> Dict[str, torch.Tensor]:
    return {k: v[rows] for k, v in out.items()
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 4}


@pytest.fixture(scope="module")
def episodes():
    return _frames(0), _frames(1)


@pytest.mark.parametrize("mixer", ["recurrent", "attention"])
def test_stream_stepper_seat_streams_are_private(episodes, mixer):
    own, other = episodes
    seat0, seat1 = torch.tensor([0, 2]), torch.tensor([1, 3])
    assert (own[0].to_play[seat0] == 0).all() and (own[0].to_play[seat1] == 1).all()
    assert not torch.equal(own[3].obs["grid_seq"][seat1], other[3].obs["grid_seq"][seat1])
    net = _net(mixer)
    runs = []
    for perturb in (False, True):
        st = StreamStepper(net)
        outs = []
        for t in range(T):
            o, lg = ((own[t].obs, own[t].legal) if not perturb
                     else _mix(own[t], other[t], seat1))
            torch.manual_seed(100 + t)
            outs.append(_row_outputs(st.step(o, lg, [0, 1, 2, 3], [t] * 4), seat0))
        runs.append(outs)
    for t, (a, b) in enumerate(zip(*runs)):
        assert set(a) == set(b) and "value" in a
        for k in a:
            assert torch.equal(a[k], b[k]), f"{mixer} frame {t}: seat 0 {k} depends on seat 1"


def test_fixed_batch_actor_rows_are_private(episodes):
    """``--actor fixed``: one fixed-shape batch over every row of the env
    (``initial_st_cache`` + tensor step index), as LearnerActor._fixed_step."""
    own, other = episodes
    seat0, seat1 = torch.tensor([0, 2]), torch.tensor([1, 3])
    net = _net("recurrent")
    runs = []
    for perturb in (False, True):
        cache = net.initial_st_cache(4)
        outs = []
        for t in range(T):
            o, lg = ((own[t].obs, own[t].legal) if not perturb
                     else _mix(own[t], other[t], seat1))
            torch.manual_seed(200 + t)
            with torch.no_grad():
                out = net.step(o, legal=lg, st_cache=cache, step_idx=torch.tensor([t]))
            cache = out.pop("st_cache")
            S = cache[0].shape[0] // 4
            rows = _row_outputs(out, seat0)
            rows["cache"] = torch.stack([c.view(4, S, -1)[seat0] for c in cache])
            outs.append(rows)
        runs.append(outs)
    for t, (a, b) in enumerate(zip(*runs)):
        for k in a:
            assert torch.equal(a[k], b[k]), f"frame {t}: seat 0 {k} depends on seat 1"
