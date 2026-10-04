"""League(balanced_seats=True): an evaluation's learner seats split evenly over the envs and
alternate with the seat offset (the wave), instead of random draws."""
import torch

from env.league import League, RandomPolicy
from env.vec_game import make_vec_env


def test_balanced_seats_split_evenly_and_alternate():
    env = make_vec_env("othello", 5, seed=1)
    rnd = RandomPolicy(env, seed=0)
    counts = torch.zeros(env.n_seats)
    for wave in range(2):
        league = League(env, {"opp": rnd}, mirror_prob=0.0, balanced_seats=True, seat_offset=wave)
        obs = env.reset(seed=wave)
        league.update(obs)
        seats = league.learner_seats.float().argmax(1)
        assert seats.tolist() == [(i + wave) % env.n_seats for i in range(env.num_envs)]
        counts += torch.bincount(seats, minlength=env.n_seats).float()
        league.release_all()
    assert counts.tolist() == [5.0, 5.0]        # 5 envs x 2 waves: 3+2 then 2+3


def test_network_policy_bf16_acts_under_autocast():
    """NetworkPolicy(bf16=True) steps its network under bf16 autocast, as the learner's actor
    does with --bf16 (the frozen --init opponent and the pool snapshots acted in fp32: a
    different player than the learner it is compared with)."""
    from env.league import NetworkPolicy
    from model.multi_game import build_multi_game
    from tests.test_core_moe import CORE

    env = make_vec_env("othello", 2, seed=1)
    net = build_multi_game(["othello"], CORE)["othello"]
    seen = []
    step = net.step

    def spy(*args, **kw):
        seen.append(torch.is_autocast_enabled("cpu"))
        return step(*args, **kw)

    net.step = spy
    obs = env.reset(seed=0)
    rows = obs.active.nonzero().squeeze(1)
    for bf16 in (False, True):
        pol = NetworkPolicy(net, device=torch.device("cpu"), bf16=bf16)
        out = pol(obs, rows)
        assert out
    assert seen == [False, True]
