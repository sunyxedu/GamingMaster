"""gmsim episodes that start from recorded trajectory prefixes (``Sim.reset_from`` /
``GmSimVecEnv.reset_from``): the env's row of the next decision holds exactly the
replay's (= the converter's) frame of that decision, with the stream position as
``step_idx`` — what golden tests of the RL env on real records need (a chance game's
env cannot otherwise be steered into a recorded deal)."""

from __future__ import annotations

import gmsim
import numpy as np
import pytest

from env.vec_gmsim import GmSimVecEnv


@pytest.mark.parametrize("game,config", [("kuhn_poker", {"players": 3}),
                                         ("pig", {"players": 2, "winscore": 10,
                                                  "diceoutcomes": 6, "horizon": 30})])
def test_reset_from_prefixes_reproduces_replay_frames(game, config):
    for seed in range(5):
        ev = gmsim.random_trajectory(game, config, seed)
        r = gmsim.replay(game, config, ev)
        n = r["n"]
        env = GmSimVecEnv(game, n, config=config)
        prefixes = [ev[:int(r["event"][d])] for d in range(n)]
        obs = env.reset_from(prefixes, seed=seed)
        streams = {s: idx for s, idx in r["streams"]}
        for d in range(n):
            seat = int(r["seat"][d])
            row = d * env.rows_per_env + (seat if env.rows_per_env > 1 else 0)
            assert bool(obs.active[row]) and bool(obs.new_episode[row])
            key = -1 if env.interleaved else seat
            assert int(obs.step_idx[row]) == streams[key].index(d)
            for k, v in r["obs"].items():
                assert np.array_equal(obs.obs[k][row].numpy(), v[d]), (game, d, k)
            assert np.array_equal(env.trajectory(d), prefixes[d])
        env.step(env.random_actions(obs))
    with pytest.raises(ValueError):
        env.reset_from([[[-1, 0], [-1, 0]]] * env.num_envs)      # a card dealt twice
