"""``GmSimVecEnv.random_actions`` answers for the rows of the VecObs it is given: a
``VecObs.select`` subset (what the league hands a policy — ``RandomPolicy`` as an
opponent or learner) gets one action per selected row, not one per env row."""

from __future__ import annotations

import torch

from env.league import League, RandomPolicy, play_episodes
from env.vec_gmsim import GmSimVecEnv


def test_random_actions_follow_the_obs_rows():
    env = GmSimVecEnv("kuhn_poker", 4, seed=0, players=3)
    obs = env.observe()
    full = env.random_actions(obs, generator=torch.Generator().manual_seed(1))["action_idx"]
    assert full.shape == (12,)
    rows = obs.active.nonzero().squeeze(1)
    sub = env.random_actions(obs.select(rows), generator=torch.Generator().manual_seed(1))
    assert torch.equal(sub["action_idx"], full[rows])


def test_random_policy_as_league_opponent_and_learner():
    for game, kw in (("kuhn_poker", {"players": 3}), ("goofspiel", {"players": 2})):
        env = GmSimVecEnv(game, 6, seed=0, **kw)
        league = League(env, {"rand": RandomPolicy(env, seed=1)}, mirror_prob=0.0, seed=2)
        res = play_episodes(env, league, RandomPolicy(env, seed=3))
        assert bool(env.finished.all()) and not bool(league.is_mirror().any())
        assert not torch.isnan(res["score"]).any()
