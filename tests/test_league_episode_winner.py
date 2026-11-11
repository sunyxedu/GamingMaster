"""An episode's result is the game's: an env whose episode is a game of several matches
(Lux: ``info["winner"]`` = the match a step ended, ``info["game_winner"]`` = the game)
is scored by the game's winner — by ``play_episodes`` and by the RL trainer's evaluation —
and an env without ``game_winner`` by ``info["winner"]`` as before."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from env.league import League, RandomPolicy, episode_winner, play_episodes
from env.vec_game import StepResult, VecEnvBase, VecObs

#: per env: the winners of its three matches — the last match's winner is not the game's
MATCHES = [[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]]
GAME = [0, 1, 1, 0]


class MatchesEnv(VecEnvBase):
    """Two seats move together; a game is three one-step matches with fixed winners."""

    game = "matches"
    n_seats = 2
    rows_per_env = 2
    ACTION_KEYS = {"action_idx": "action_idx"}

    def __init__(self, n_envs: int = 4, with_game_winner: bool = True):
        super().__init__(n_envs, False, 0)
        self.with_game_winner = with_game_winner
        self._t = torch.zeros(n_envs, dtype=torch.long)
        self._new = torch.ones(n_envs, dtype=torch.bool)

    @property
    def finished(self) -> torch.Tensor:
        return self._t >= 3

    def obs_schema(self):
        return {"x": ((1,), torch.float32)}

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids=None) -> VecObs:
        self._t[:] = 0
        self._new[:] = True
        return self.observe()

    def observe(self) -> VecObs:
        R = self.num_rows
        env = self._row_env()
        live = ~self.finished[env]
        seats = torch.arange(R) % 2
        return VecObs(obs={"x": torch.zeros(R, 1)},
                      legal={"action": torch.ones(R, 2, dtype=torch.bool)},
                      to_play=seats, env_id=env, stream_id=torch.arange(R),
                      step_idx=self._t[env].clone(), new_episode=self._new[env].clone(),
                      active=live, forced=torch.zeros(R, dtype=torch.bool),
                      stream_seats=torch.nn.functional.one_hot(seats, 2).bool())

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        return {"action_idx": torch.zeros(obs.n_rows, dtype=torch.long)}

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        n = self.num_envs
        live = ~self.finished
        t = self._t.clone()
        winner = torch.full((n,), -1, dtype=torch.long)
        for e in range(n):
            if live[e]:
                winner[e] = MATCHES[e % 4][int(t[e])]
        self._t = torch.where(live, self._t + 1, self._t)
        done = live & (self._t >= 3)
        self._new[:] = False
        rew = torch.zeros(n, 2)
        rew[live, winner[live]] = 1.0
        rew[live, 1 - winner[live]] = -1.0
        info: Dict[str, Any] = {
            "winner": winner,
            "episode_len": torch.where(done, self._t, torch.zeros_like(self._t)),
            "terminated": done.clone(), "truncated": torch.zeros(n, dtype=torch.bool)}
        if self.with_game_winner:
            game = torch.tensor([GAME[e % 4] for e in range(n)])
            info["game_winner"] = torch.where(done, game, torch.full((n,), -1))
        return self.observe(), rew, done, info


def test_episode_winner_prefers_the_games_winner():
    w = torch.tensor([0, 1])
    assert torch.equal(episode_winner({"winner": w}), w)
    both = {"winner": w, "game_winner": torch.tensor([1, 1])}
    assert torch.equal(episode_winner(both), torch.tensor([1, 1]))


def test_play_episodes_scores_the_game_not_its_last_match():
    env = MatchesEnv(4)
    league = League(env, {"rand": RandomPolicy(env, seed=1)}, mirror_prob=0.0, seed=0)
    res = play_episodes(env, league, RandomPolicy(env, seed=2))
    assert res["winner"].tolist() == GAME                       # not [1, 0, 1, 0], the last matches
    seats = res["learner_seats"]
    want = [1.0 if bool(seats[e, GAME[e]]) else 0.0 for e in range(4)]
    assert res["score"].tolist() == want
    assert (res["episode_len"] == 3).all()


def test_an_env_without_game_winner_is_scored_by_its_winner():
    env = MatchesEnv(4, with_game_winner=False)
    league = League(env, {"rand": RandomPolicy(env, seed=1)}, mirror_prob=0.0, seed=0)
    res = play_episodes(env, league, RandomPolicy(env, seed=2))
    assert res["winner"].tolist() == [m[-1] for m in MATCHES]
