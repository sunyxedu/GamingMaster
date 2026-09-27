"""League.learner_score with the episodes' final payoffs: SHARED wins (teammates of a
side decided in play — Dou Dizhu's two peasants — or co-winners) count as wins for
the learner seats that share the top payoff and as losses for the others; without
payoffs (or with a single winner / a full tie) the result is unchanged."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from env.league import League


def _league(learner_seat):
    env = SimpleNamespace(n_seats=3, num_envs=4)
    league = League(env, {"opp": lambda obs, rows: {}}, mirror_prob=0.0)
    league.learner_seats[:] = False
    for i, s in enumerate(learner_seat):
        if s is None:
            league.learner_seats[i] = True          # a mirror episode
        else:
            league.learner_seats[i, s] = True
    return league


def test_shared_top_payoff_is_a_win_for_the_team():
    # env 0: the peasants (seats 1, 2) beat landlord 0; env 1: landlord 0 wins;
    # env 2: nobody bid / truncated (all zero); env 3: a mirror episode
    winner = torch.tensor([-1, 0, -1, -1])
    returns = torch.tensor([[-4.0, 2.0, 2.0], [6.0, -3.0, -3.0], [0.0, 0.0, 0.0],
                            [-2.0, 1.0, 1.0]], dtype=torch.float64)
    for seat, want in ((0, [0.0, 1.0, 0.5]), (1, [1.0, 0.0, 0.5]), (2, [1.0, 0.0, 0.5])):
        league = _league([seat, seat, seat, None])
        got = league.learner_score(winner, returns=returns)
        assert got[:3].tolist() == want, seat
        assert torch.isnan(got[3])
        # without the payoffs a team win is a draw (the historical behaviour)
        assert league.learner_score(winner)[:3].tolist() == [0.5, want[1], 0.5]


def test_single_winners_and_ties_are_unchanged():
    league = _league([0, 1, 2, 0])
    winner = torch.tensor([0, 0, -1, 2])
    returns = torch.tensor([[1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.5, 0.5, 0.5],
                            [-1.0, 0.0, 1.0]], dtype=torch.float64)
    assert league.learner_score(winner, returns=returns).tolist() == \
        league.learner_score(winner).tolist() == [1.0, 0.0, 0.5, 0.0]
