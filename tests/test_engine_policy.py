"""env/engine_policy.py: Stockfish / Pikafish / YaneuraOu as league opponents on the gmsim
chess / xiangqi / shogi envs — every engine move maps to one of our legal actions, whole
games end, and the engine beats a random player."""

import os
import shutil

import pytest
import torch

from env.league import League, RandomPolicy
from env.vec_game import make_vec_env

ENGINES = {g: os.path.expanduser(f"~/gm-engines/bin/{e}")
           for g, e in (("chess", "stockfish"), ("xiangqi", "pikafish"), ("shogi", "yaneuraou"))}


def _have(path):
    return os.path.exists(path) or shutil.which(path)


@pytest.mark.parametrize("game", sorted(ENGINES))
def test_engine_beats_random_through_the_league(game):
    if not _have(ENGINES[game]):
        pytest.skip(f"no {ENGINES[game]}")
    from env.engine_policy import EnginePolicy

    env = make_vec_env(game, 2, seed=3)
    pol = EnginePolicy(env, ENGINES[game], go="go nodes 300")
    rnd = RandomPolicy(env, seed=1)
    try:
        league = League(env, {"engine": pol}, mirror_prob=0.0, seed=5)
        obs = env.reset(seed=11)
        league.update(obs)
        winner = torch.full((env.num_envs,), -1, dtype=torch.long)
        ended = torch.zeros(env.num_envs, dtype=torch.bool)
        for _ in range(2000):
            if not bool(obs.active.any()):
                break
            rows = league.learner_rows(obs).nonzero().squeeze(1)
            la = {}
            if rows.numel():
                from env.league import scatter_rows
                la = scatter_rows(env.replay_actions(rnd(obs, rows)), rows, obs.n_rows)
            obs, rew, done, info = env.step(league.act(obs, la))
            winner = torch.where(done & ~ended, info["winner"].long(), winner)
            ended |= done
            league.update(obs)
        assert bool(ended.all()), "games did not end"
        assert pol.moves > 10
        score = league.learner_score(winner)
        assert float(score.mean()) <= 0.25, (game, score)     # the random learner loses
    finally:
        league.release_all()
        pol.close()
