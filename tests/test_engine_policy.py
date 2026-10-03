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
# the small b10 net's raw policy when installed (~0.06 s a move on one CPU thread; the b18 net
# at 4 visits took minutes per game on a CPU-only machine), else the b18 net
_KG_SMALL = os.path.expanduser("~/gm-engines/nets/katago/kata1-b10c128-s1141046784-d204142634.txt.gz")
_KG_NET = _KG_SMALL if os.path.exists(_KG_SMALL) else os.path.expanduser(
    "~/gm-engines/nets/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")
_KG_CFG = os.environ.get("GM_KATAGO_GTP_CFG", "/opt/homebrew/share/katago/configs/gtp_example.cfg")
ENGINES["go"] = (f"{os.path.expanduser('~/gm-engines/bin/katago')} gtp -model {_KG_NET} -config {_KG_CFG} "
                 f"-override-config maxVisits={1 if _KG_NET == _KG_SMALL else 4},logDir=,"
                 f"logAllGTPCommunication=false,numSearchThreads=1")


def _have(cmd):
    path = cmd.split()[0]
    files = [w for w in cmd.split() if w.endswith((".gz", ".cfg"))]
    return (os.path.exists(path) or shutil.which(path)) and all(os.path.exists(f) for f in files)


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


def test_egaroucid_beats_random_on_the_othello_env():
    from training.arena_othello import default_engine
    if not _have(default_engine()):
        pytest.skip("no egaroucid")
    from env.engine_policy import OthelloEnginePolicy
    from env.league import scatter_rows

    env = make_vec_env("othello", 4, seed=3)
    pol = OthelloEnginePolicy(env, level=1)
    rnd = RandomPolicy(env, seed=1)
    league = League(env, {"engine": pol}, mirror_prob=0.0, seed=5)
    try:
        obs = env.reset(seed=11)
        league.update(obs)
        winner = torch.full((env.num_envs,), -1, dtype=torch.long)
        ended = torch.zeros(env.num_envs, dtype=torch.bool)
        for _ in range(300):
            if not bool(obs.active.any()):
                break
            rows = league.learner_rows(obs).nonzero().squeeze(1)
            la = scatter_rows(env.replay_actions(rnd(obs, rows)), rows, obs.n_rows) if rows.numel() else {}
            obs, rew, done, info = env.step(league.act(obs, la))
            winner = torch.where(done & ~ended, info["winner"].long(), winner)
            ended |= done
            league.update(obs)
        assert bool(ended.all()) and pol.moves > 20
        assert float(league.learner_score(winner).mean()) <= 0.25
    finally:
        league.release_all()
        pol.close()


@pytest.mark.skipif(not shutil.which("uv"), reason="needs uv (the gnubg wheel)")
@pytest.mark.parametrize("match", [1, 3])
def test_gnubg_beats_random_on_the_backgammon_env(match):
    """match=3: a game inside a match ends with gnubg bearing off its last checker while the
    state already shows the next game (the 2026-10-03 bench lost its backgammon referee to
    'reachable by none of our legal moves' there)."""
    from env.engine_policy import BackgammonEnginePolicy
    from env.league import scatter_rows

    env = make_vec_env("backgammon", 4, seed=3, match_length=match)
    try:
        pol = BackgammonEnginePolicy(env, plies=0)
    except Exception as e:                      # noqa: BLE001 — no gnubg wheel offline
        pytest.skip(f"gnubg bridge: {e}")
    rnd = RandomPolicy(env, seed=1)
    league = League(env, {"engine": pol}, mirror_prob=0.0, seed=5)
    try:
        obs = env.reset(seed=11)
        league.update(obs)
        winner = torch.full((env.num_envs,), -1, dtype=torch.long)
        ended = torch.zeros(env.num_envs, dtype=torch.bool)
        for _ in range(2000 * match):
            if not bool(obs.active.any()):
                break
            rows = league.learner_rows(obs).nonzero().squeeze(1)
            la = scatter_rows(env.replay_actions(rnd(obs, rows)), rows, obs.n_rows) if rows.numel() else {}
            obs, rew, done, info = env.step(league.act(obs, la))
            winner = torch.where(done & ~ended, info["winner"].long(), winner)
            ended |= done
            league.update(obs)
        assert bool(ended.all()) and pol.moves > 20
        assert float(league.learner_score(winner).mean()) <= 0.25
    finally:
        league.release_all()
        pol.close()


def test_league_merges_opponent_actions_of_another_dtype():
    """The network samples shogi's Bernoulli heads (dropped / promoted) as Bool, an engine
    policy answers Long: League.act merges both (the 2026-10-03 bench disabled the shogi
    referee on 'Index put requires the source and destination dtypes match')."""
    from env.league import scatter_rows

    env = make_vec_env("shogi", 2, seed=3)
    rnd = RandomPolicy(env, seed=1)

    class LongStub:
        acts_only = True

        def reset_streams(self, streams):
            pass

        def release_streams(self, streams):
            pass

        def __call__(self, obs, rows):
            return {k: v.long() for k, v in rnd(obs, rows).items()}

    league = League(env, {"stub": LongStub()}, mirror_prob=0.0, seed=5)
    obs = env.reset(seed=11)
    league.update(obs)
    for _ in range(6):
        rows = league.learner_rows(obs).nonzero().squeeze(1)
        la = {}
        if rows.numel():
            la = {k: (v.bool() if k in ("dropped", "promoted") else v)
                  for k, v in env.replay_actions(rnd(obs, rows)).items()}
            la = scatter_rows(la, rows, obs.n_rows)
        obs, _, _, _ = env.step(league.act(obs, la))
        league.update(obs)
    league.release_all()


def test_noisy_referee_mixes_engine_and_random_moves():
    """NoisyReferee: eps=1 never asks the engine; eps=0.5 plays whole games through the
    League, the engine answering only its rows."""
    from training.arena_othello import default_engine
    if not _have(default_engine()):
        pytest.skip("no egaroucid")
    from env.engine_policy import NoisyReferee, OthelloEnginePolicy
    from env.league import scatter_rows

    for eps in (1.0, 0.5):
        env = make_vec_env("othello", 4, seed=3)
        base = OthelloEnginePolicy(env, level=1)
        pol = NoisyReferee(base, env, eps, seed=2)
        rnd = RandomPolicy(env, seed=1)
        league = League(env, {"engine": pol}, mirror_prob=0.0, seed=5)
        try:
            obs = env.reset(seed=11)
            league.update(obs)
            ended = torch.zeros(env.num_envs, dtype=torch.bool)
            for _ in range(300):
                if not bool(obs.active.any()):
                    break
                rows = league.learner_rows(obs).nonzero().squeeze(1)
                la = scatter_rows(env.replay_actions(rnd(obs, rows)), rows, obs.n_rows) \
                    if rows.numel() else {}
                obs, _, done, _ = env.step(league.act(obs, la))
                ended |= done
                league.update(obs)
            assert bool(ended.all())
            assert (pol.moves == 0) if eps == 1.0 else (pol.moves > 0)
        finally:
            league.release_all()
            pol.close()


def test_engine_pool_answers_each_row_with_its_own_position():
    """workers=3: the rows of a call are split over three engine processes (threads); every
    answer is a legal move of ITS row's position (rows are in different positions), so no
    answer lands on another row.  (Moves can differ from one engine's: Stockfish keeps its
    hash table between searches.)"""
    if not _have(ENGINES["chess"]):
        pytest.skip(f"no {ENGINES['chess']}")
    from env.engine_policy import EnginePolicy, action_keys, replay_state
    from env.league import scatter_rows

    env = make_vec_env("chess", 6, seed=3)
    obs = env.reset(seed=11)
    rnd = RandomPolicy(env, seed=1)
    for _ in range(5):                          # a few random plies: different positions
        rows = obs.active.nonzero().squeeze(1)
        obs, _, _, _ = env.step(scatter_rows(env.replay_actions(rnd(obs, rows)), rows, obs.n_rows))
    rows = obs.active.nonzero().squeeze(1)
    many = EnginePolicy(env, ENGINES["chess"], go="go nodes 200", options={"Threads": "1"}, workers=3)
    try:
        b = many(obs, rows)
        assert len(many.engines) == 3 and many.moves == len(rows)
        keys = tuple(env.action_keys)
        seen = set()
        for i, r in enumerate(rows.tolist()):
            e, seat = int(obs.env_id[r]), int(obs.to_play[r])
            state, _ = replay_state("chess", env.trajectory(e), "")
            legal = {tuple(action_keys(state, seat, a, keys)[k] for k in keys)
                     for a in state.legal_actions(seat)}
            got = tuple(int(b[k][i]) for k in keys)
            assert got in legal, (i, got)
            seen.add(tuple(sorted(legal)))
        assert len(seen) > 1                    # the rows really were in different positions
    finally:
        many.close()


@pytest.mark.skipif(not shutil.which("uv"), reason="needs uv (the gnubg wheel)")
def test_gnubg_ends_a_game_inside_a_match():
    """gnubg plays both seats of 5-point matches (random players end games by dropping
    doubles): wherever the player to move can bear off its last checker with a choice of
    moves, gnubg's move is found — the state shows the NEXT game's opening position
    afterwards, never gnubg's empty board (the 2026-10-03 bench lost its referee there)."""
    import numpy as np

    from env.engine_policy import BackgammonEnginePolicy, bg_parse, replay_state
    from env.league import scatter_rows

    env = make_vec_env("backgammon", 6, seed=5, match_length=5)
    try:
        pol = BackgammonEnginePolicy(env, plies=0)
    except Exception as e:                      # noqa: BLE001 — no gnubg wheel offline
        pytest.skip(f"gnubg bridge: {e}")
    found = 0
    try:
        obs = env.reset(seed=7)
        for _ in range(4000):
            if found >= 2 or not bool(obs.active.any()):
                break
            rows = obs.active.nonzero().squeeze(1)
            for r in rows.tolist():
                e, seat = int(obs.env_id[r]), int(obs.to_play[r])
                state, _ = replay_state("backgammon", np.asarray(env.trajectory(e)).reshape(-1, 2),
                                        pol.cfg_json)
                legal = state.legal_actions(seat)
                p0 = bg_parse(str(state))
                if len(legal) < 2 or max(p0["score"]) >= 4 or str(state).startswith(("phase=Cube", "phase=Take")):
                    continue
                for a in legal:
                    nxt = state.clone()
                    nxt.apply_action(seat, int(a))
                    if bg_parse(str(nxt))["score"] != p0["score"]:
                        found += 1                  # pol(...) below must answer this row
                        break
            # pol answers every row (both seats): an unreachable target raises here
            obs, _, _, _ = env.step(scatter_rows(env.replay_actions(pol(obs, rows)), rows, obs.n_rows))
        assert found >= 1, "no game-ending bear-off with a choice was reached"
    finally:
        pol.close()
