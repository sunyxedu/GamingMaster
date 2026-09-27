"""gmrules simulators through gmsim and env/vec_gmsim.py (the VecGameEnv of
every gmrules game), on the candidate-action games: rows / streams / seats for per-seat alternating
(kuhn_poker, 2-4 players) and simultaneous (goofspiel) games, the RL frames equal
to the replay (= converter) frames of the envs' own trajectories, determinism,
errors, and the league / policy plumbing with per-seat streams."""

from __future__ import annotations

import gmsim
import numpy as np
import pytest
import torch

from env.league import League, NetworkPolicy, RandomPolicy, play_episodes
from env.vec_game import VecGameEnv, has_simulator, make_vec_env
from env.vec_gmsim import CandidateVecEnv
from model.games import build_game_network

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1, temporal_mixer="recurrent")


def _run_random(env: CandidateVecEnv, seed: int = 0, max_steps: int = 500):
    """Play every env to the end with random legal actions; returns the frames each
    row saw (in order) and the step infos."""
    gen = torch.Generator().manual_seed(seed)
    obs = env.observe()
    seen = {r: [] for r in range(obs.n_rows)}
    infos, rewards = [], torch.zeros(env.num_envs, env.n_seats)
    steps = 0
    while bool(obs.active.any()):
        for r in obs.active.nonzero().squeeze(1).tolist():
            assert int(obs.step_idx[r]) == len(seen[r])
            seen[r].append({k: v[r].clone() for k, v in obs.obs.items()})
        acts = env.random_actions(obs, generator=gen)
        obs, rew, done, info = env.step(acts)
        rewards += rew
        infos.append(info)
        steps += 1
        assert steps < max_steps
    return seen, infos, rewards


def test_registry_and_protocol():
    assert {"kuhn_poker", "goofspiel"} <= set(gmsim.games())
    assert has_simulator("kuhn_poker") and has_simulator("goofspiel")
    env = make_vec_env("kuhn_poker", 3, seed=1, players=3)
    assert isinstance(env, CandidateVecEnv) and isinstance(env, VecGameEnv)
    assert (env.n_seats, env.rows_per_env, env.num_rows, env.interleaved, env.lockstep) \
        == (3, 3, 9, False, False)
    schema = env.obs_schema()
    obs = env.observe()
    for k, (shape, dtype) in schema.items():
        assert obs.obs[k].shape == (9, *shape) and obs.obs[k].dtype == dtype, k
    assert set(obs.obs) == set(schema)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_kuhn_per_seat_rows(players):
    env = CandidateVecEnv("kuhn_poker", 4, seed=3, players=players)
    obs = env.reset(seed=11)
    R, S = 4 * players, players
    seat = torch.arange(R) % S
    # stream = the seat's own view; only seat 0 acts first; every row flagged new
    assert torch.equal(obs.stream_seats, torch.nn.functional.one_hot(seat, S).bool())
    assert torch.equal(obs.to_play, seat)
    assert torch.equal(obs.active, seat == 0)
    assert bool(obs.new_episode.all()) and bool((obs.step_idx == 0).all())
    assert torch.equal(obs.legal["action"], obs.obs["action_mask"])
    assert not bool(obs.forced.any())
    # inactive rows carry no frame
    assert not bool(obs.obs["entity_mask"][~obs.active].any())
    seen, infos, rewards = _run_random(env, seed=players)
    assert bool(env.finished.all())
    done = torch.stack([i["terminated"] for i in infos]).any(0)
    assert bool(done.all())
    # zero-sum chips: outcomes sum to 0 per env; exactly one winner
    assert torch.allclose(rewards.sum(-1), torch.zeros(4), atol=1e-6)
    winners = torch.stack([i["winner"] for i in infos]).max(0).values
    assert bool((winners >= 0).all())
    # every seat acted at least once, and no new_episode after the reset step
    assert all(len(seen[r]) >= 1 for r in range(R))


def test_frames_equal_the_replay_of_the_trajectory():
    """RL frames = the converter's frames: every row's stream is the replay stream of
    its env's logged trajectory (same encoder, gmrules encode_frame)."""
    for game, cfg in (("kuhn_poker", {"players": 3}), ("goofspiel", {"num_cards": 5, "players": 3}),
                      ("pig", {"players": 3, "winscore": 8, "diceoutcomes": 3, "horizon": 40})):
        env = CandidateVecEnv(game, 5, seed=0, **cfg)
        env.reset(seed=5)
        seen, _, rewards = _run_random(env, seed=1)
        for e in range(5):
            r = gmsim.replay(game, cfg, env.trajectory(e))
            assert r["terminal"]
            assert torch.allclose(torch.tensor(r["outcome"]), rewards[e])
            for seat, idx in r["streams"]:
                row = e * env.rows_per_env + max(seat, 0)       # seat -1: interleaved
                assert len(seen[row]) == len(idx)
                for k in seen[row][0]:
                    ours = torch.stack([f[k] for f in seen[row]])
                    theirs = torch.from_numpy(np.ascontiguousarray(r["obs"][k][idx]))
                    assert ours.dtype == theirs.dtype, (game, k)
                    assert torch.equal(ours, theirs), (game, e, seat, k)


def test_goofspiel_simultaneous_rows():
    env = CandidateVecEnv("goofspiel", 3, seed=0, num_cards=4, players=3)
    obs = env.observe()
    assert env.layout == "per_seat" and env.rows_per_env == 3
    assert bool(obs.active.all())                 # every seat bids every turn
    seen, infos, _ = _run_random(env, seed=2)
    # 3 decision turns (the 4th is automatic), every seat each turn: lockstep in practice
    assert all(len(v) == 3 for v in seen.values())
    assert all(int(i["episode_len"].max()) in (0, 3) for i in infos)


def test_pig_interleaved_stream():
    """Perfect information: ONE stream per game holding every decision, each from the
    mover's view (stream_seats all True), rows in lockstep; forced frames (stop is
    the only candidate) apply their single candidate whatever is sent."""
    env = CandidateVecEnv("pig", 4, seed=0, players=3, winscore=8, diceoutcomes=3, horizon=40)
    obs = env.observe()
    assert (env.layout, env.rows_per_env, env.interleaved, env.lockstep) \
        == ("interleaved", 1, True, True)
    assert bool(obs.stream_seats.all()) and bool(obs.active.all())
    assert obs.to_play.tolist() == [0] * 4
    gen = torch.Generator().manual_seed(0)
    movers, forced_seen = {e: [] for e in range(4)}, 0
    while bool(obs.active.any()):
        for e in obs.active.nonzero().squeeze(1).tolist():
            movers[e].append(int(obs.to_play[e]))
        # lockstep: every live row is at the same frame index
        assert len(set(obs.step_idx[obs.active].tolist())) == 1
        acts = env.random_actions(obs, generator=gen)["action_idx"]
        if bool(obs.forced.any()):
            forced_seen += 1
            # garbage on forced rows (the env applies their single candidate)
            acts = torch.where(obs.forced, torch.full_like(acts, 1), acts)
        obs, _, _, info = env.step({"action_idx": acts})
    for e in range(4):
        r = gmsim.replay("pig", env.config, env.trajectory(e))
        (seat, idx), = r["streams"]
        assert seat == -1 and r["seat"].tolist() == movers[e]
    assert len({m for ms in movers.values() for m in ms}) == 3 and forced_seen > 0


def test_determinism_and_seeds():
    def traj(seed):
        env = CandidateVecEnv("goofspiel", 4, seed=0, players=2)
        env.reset(seed=seed)
        _run_random(env, seed=9)
        return [env.trajectory(e).tolist() for e in range(4)]

    assert traj(3) == traj(3)
    assert traj(3) != traj(4)
    env = CandidateVecEnv("kuhn_poker", 2, seed=0)
    env.reset(seed=7)
    a = [env.trajectory(e)[:2].tolist() for e in range(2)]
    env.reset(seed=7, env_ids=[1])              # episode seeds seed + k over the reset envs
    assert env.trajectory(1)[:2].tolist() == a[0]
    obs = env.observe()
    assert obs.new_episode.tolist() == [False, False, True, True]
    env.reset(n=3)
    assert env.num_rows == 6


def test_illegal_action_raises_and_config_errors():
    env = CandidateVecEnv("kuhn_poker", 2, seed=0)
    bad = torch.zeros(4, dtype=torch.long)
    bad[0] = 5
    with pytest.raises(ValueError, match="not a legal candidate"):
        env.step({"action_idx": bad})
    env.step({"action_idx": torch.zeros(4, dtype=torch.long)})     # state unchanged by the error
    with pytest.raises(ValueError, match="players must be 2..=4"):
        CandidateVecEnv("kuhn_poker", 1, players=7)
    with pytest.raises(ValueError, match="unknown field"):
        CandidateVecEnv("kuhn_poker", 1, cards=3)
    with pytest.raises(ValueError, match="unknown gmrules game"):
        gmsim.Sim("nope")
    with pytest.raises(ValueError, match="impossible"):
        gmsim.replay("kuhn_poker", None, [[-1, 0], [-1, 0]])
    with pytest.raises(ValueError, match="illegal"):
        gmsim.replay("kuhn_poker", None, [[-1, 0], [-1, 1], [0, 2]])


def test_state_api_matches_replay():
    st = gmsim.State("kuhn_poker", {"players": 2})
    assert st.turn() == ("chance", []) and len(st.chance_outcomes()) == 3
    for ev in ([-1, 2], [-1, 0], [0, 0], [1, 1]):
        st.apply(ev)
    frame = st.frame(0)
    r = gmsim.replay("kuhn_poker", {"players": 2}, st.history())
    assert st.turn() == ("act", [0])
    assert r["n"] == 2 and frame["legal_ids"] == [0, 1]
    c = st.clone()
    c.apply_action(0, 1)
    assert c.turn() == ("terminal", []) and st.turn() == ("act", [0])
    assert c.returns() == [2.0, -2.0] and c.outcome() == [1.0, -1.0]
    with pytest.raises(ValueError):
        st.apply_action(1, 0)


def _tiny_net(game):
    torch.manual_seed(0)
    return build_game_network(game, **TINY).eval()


def test_network_policy_keeps_per_seat_streams_between_turns():
    """play_episodes with a network on per-seat streams: a row idle on the other
    seats' turns keeps its Stage-1 state (released only when its episode ends), and
    later seats start their stream at step_idx 0."""
    env = CandidateVecEnv("kuhn_poker", 6, seed=0, players=3)
    net = _tiny_net("kuhn_poker")
    learner = NetworkPolicy(net)
    league = League(env, {"random": RandomPolicy(env, seed=1)}, mirror_prob=0.5, seed=3)
    res = play_episodes(env, league, learner)
    assert bool((res["winner"] >= 0).all())
    assert learner.stepper.live_streams() == []
    assert learner.stepper.n_calls > 0
    # goofspiel: every seat acts every turn (the lockstep-like simultaneous case)
    env = CandidateVecEnv("goofspiel", 4, seed=0, players=2)
    res = play_episodes(env, League(env), NetworkPolicy(_tiny_net("goofspiel")))
    assert res["episode_len"].tolist() == [4, 4, 4, 4]


def test_env_state_dict_roundtrip():
    env = CandidateVecEnv("kuhn_poker", 2, seed=5)
    sd = env.state_dict()
    a = env.random_actions(env.observe())
    env.load_state_dict(sd)
    b = env.random_actions(env.observe())
    assert torch.equal(a["action_idx"], b["action_idx"])
