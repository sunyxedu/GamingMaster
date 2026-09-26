"""Team games in the gmrules framework (gmrules ``Game::teams``): the env exposes
the teams, the league gives a policy whole teams in non-mirror episodes, and
``learner_score`` scores the winning team.  The Rust side (team winner, teammates'
outcomes checked by ``check_game``) is tested in gmrules; bridge
(tests/test_bridge_*.py) is the first team game."""

from __future__ import annotations

import gmsim
import torch

from env.league import League, RandomPolicy
from env.vec_gmsim import GmSimVecEnv


def test_games_without_teams():
    assert gmsim.game_info("kuhn_poker", {"players": 3})["teams"] is None
    env = GmSimVecEnv("kuhn_poker", 2, players=3)
    assert env.teams is None


def test_league_assigns_whole_teams():
    env = GmSimVecEnv("kuhn_poker", 8, seed=0, players=4)
    env.teams = (0, 1, 0, 1)            # pretend partnerships (the league only reads env.teams)
    league = League(env, {"rand": RandomPolicy(env, seed=1)}, mirror_prob=0.0, seed=2)
    league.update(env.reset(seed=3))
    seats = league.learner_seats
    assert not bool(league.is_mirror().any())
    for row in seats.tolist():
        assert row in ([True, False, True, False], [False, True, False, True])
    # both teams drawn over the envs; the score names the winning team's lowest seat
    assert {tuple(r) for r in seats.tolist()} == {(True, False, True, False),
                                                  (False, True, False, True)}
    ns = seats[:, 0]
    score = league.learner_score(torch.zeros(8, dtype=torch.long))       # N/S won
    assert torch.equal(score, ns.double())
    score = league.learner_score(torch.ones(8, dtype=torch.long))        # E/W won
    assert torch.equal(score, (~ns).double())
    pinned = League(env, {"rand": RandomPolicy(env)}, mirror_prob=0.0, learner_seat=3)
    pinned.update(env.reset(seed=4))
    assert bool((pinned.learner_seats == torch.tensor([False, True, False, True])).all())
