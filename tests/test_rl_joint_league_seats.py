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
