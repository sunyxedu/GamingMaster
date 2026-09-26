"""env/vec_game.py (VecGameEnv protocol + registry), the othello / generals /
gomoku adapters (env/vec_othello.py, env/vec_generals.py, env/vec_gomoku.py)
and the league interface (env/league.py).  Frame-by-frame equality with the
IL data produced by the Rust converters lives in test_vec_game_golden.py."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from data_processing.games.othello import rules
from env.generals_env import _MOUNTAIN
from env.league import (
    CallablePolicy,
    League,
    NetworkPolicy,
    PoolPolicy,
    RandomPolicy,
    SnapshotPool,
    StreamStepper,
    play_episodes,
    scatter_rows,
)
from env.vec_game import (
    VEC_ENV_REGISTRY,
    VecGameEnv,
    VecObs,
    has_simulator,
    make_vec_env,
    register_vec_env,
)
from env.vec_generals import GeneralsVecEnv, full_state_grids, replay_tiebreak_by_step
from env.vec_gomoku import GomokuVecEnv
from env.vec_othello import OthelloVecEnv
from model.games import build_game_network

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, encoder_channels=16,
            dropout=0.0, rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1,
            rd_mean_iters=2, rd_backprop_iters=1)


def _check_schema(env, obs: VecObs):
    R = env.num_rows
    schema = env.obs_schema()
    assert set(obs.obs) == set(schema)
    for k, (shape, dtype) in schema.items():
        assert obs.obs[k].shape == (R, *shape) and obs.obs[k].dtype == dtype, k
    for f in ("to_play", "env_id", "stream_id", "step_idx"):
        assert getattr(obs, f).shape == (R,) and getattr(obs, f).dtype == torch.long, f
    for f in ("new_episode", "active", "forced"):
        assert getattr(obs, f).shape == (R,) and getattr(obs, f).dtype == torch.bool, f
    assert obs.stream_seats.shape == (R, env.n_seats)
    assert torch.equal(obs.env_id, torch.arange(R) // env.rows_per_env)
    assert torch.equal(obs.stream_id, torch.arange(R))
    # every row's own seat acts on its stream
    assert obs.stream_seats.gather(1, obs.to_play.view(-1, 1)).all()


# ---------------------------------------------------------------------------
# protocol / registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["othello", "generals", "gomoku"])
def test_registry_and_protocol(game):
    env = make_vec_env(game, 3)
    assert isinstance(env, VecGameEnv) and env.game == game and has_simulator(game)
    assert env.num_envs == 3 and env.num_rows == 3 * env.rows_per_env
    obs = env.observe()
    _check_schema(env, obs)
    assert obs.new_episode.all() and obs.active.all() and (obs.step_idx == 0).all()
    assert not env.finished.any()
    # resize
    obs = env.reset(5)
    assert env.num_envs == 5 and obs.n_rows == 5 * env.rows_per_env
    _check_schema(env, obs)
    # partial reset only flags the reset envs
    env.step(env.random_actions(obs, generator=torch.Generator().manual_seed(0)))
    obs = env.reset(env_ids=[2])
    assert obs.new_episode.tolist() == [e == 2 for e in obs.env_id.tolist()]
    assert (obs.step_idx[obs.env_id == 2] == 0).all()
    assert (obs.step_idx[obs.env_id != 2] == 1).all()
    with pytest.raises(KeyError):
        make_vec_env("chess")
    assert not has_simulator("chess")
    with pytest.raises(ValueError):
        register_vec_env(game, lambda n: env)
    sub = obs.select(torch.tensor([0, obs.n_rows - 1]))
    assert sub.n_rows == 2 and torch.equal(sub.env_id, obs.env_id[[0, -1]])


def test_registry_lists_the_simulator_games():
    assert set(VEC_ENV_REGISTRY) >= {"generals", "othello", "gomoku"}


def test_replay_actions_normalisation():
    env = OthelloVecEnv(2)
    a = env.replay_actions({"action_place": torch.tensor([19, 26]),
                            "action_pass": torch.tensor([False, True]),
                            "action_log_prob": torch.zeros(2), "value": torch.zeros(2)})
    assert set(a) == {"place_idx", "passed"} and a["place_idx"].tolist() == [19, 26]
    assert env.replay_actions({"place_idx": torch.tensor([1, 2])})["place_idx"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="twice"):
        env.replay_actions({"action_place": torch.tensor([1, 2]),
                            "place_idx": torch.tensor([1, 3])})
    with pytest.raises(KeyError, match="place_idx"):
        env.step({"passed": torch.zeros(2, dtype=torch.bool)})
    g = GeneralsVecEnv(1)
    ga = g.replay_actions({"action_source": torch.tensor([3, 4]),
                           "action_direction": torch.tensor([0, 1]),
                           "action_split": torch.tensor([1, 0]),
                           "action_pass": torch.tensor([True, False])})
    assert set(ga) == {"source_idx", "direction", "split", "passed"}


def test_env_rng_state_roundtrip():
    env = GeneralsVecEnv(2, seed=3)
    st = env.state_dict()
    a = env.reset().obs["grid_seq"]
    env.load_state_dict(st)
    b = env.reset().obs["grid_seq"]
    assert torch.equal(a, b)
    assert not torch.equal(a, env.reset().obs["grid_seq"])


@pytest.mark.parametrize("game", ["othello", "generals", "gomoku"])
def test_random_actions_without_generator_follow_the_env_seed(game):
    """random_actions(generator=None) draws from the env's own torch RNG:
    seeded by the env seed, restored by load_state_dict, and independent of
    torch's global RNG."""
    noise = iter(range(10**6, 10**6 + 1000, 7))

    def draws(env, k=5):
        obs = env.observe() if game == "generals" else env.reset()
        out = []
        for _ in range(k):
            a = env.random_actions(obs)
            key = "source_idx" if game == "generals" else "place_idx"
            out.append(a[key].clone())
            torch.manual_seed(next(noise))          # a different global RNG every call
            obs, *_ = env.step(a)
        return torch.stack(out)

    kw = {"max_turns": 40} if game == "generals" else {}
    a = draws(make_vec_env(game, 4, seed=5, **kw))
    b = draws(make_vec_env(game, 4, seed=5, **kw))
    assert torch.equal(a, b)
    env = make_vec_env(game, 4, seed=5, **kw)
    st = env.state_dict()
    first = draws(env)
    env.load_state_dict(st)
    if game != "generals":                     # one start position
        assert torch.equal(draws(env), first)
    else:       # the restored episode RNG deals the fresh env's second episodes
        env.reset()
        fresh = make_vec_env(game, 4, seed=5, **kw)
        fresh.reset()
        assert torch.equal(draws(env), draws(fresh))
    if game != "generals":                     # reset(seed) reseeds it too
        e = make_vec_env(game, 4, seed=1)
        e.reset(seed=5)
        assert torch.equal(draws(e), a)


# ---------------------------------------------------------------------------
# othello
# ---------------------------------------------------------------------------

def test_othello_matches_reference_rules_on_random_games():
    """Batched bitboard play == rules.py step by step: planes (own/opp/empty
    from the mover), legal masks, forced pass frames, termination, rewards."""
    G = 24
    env = OthelloVecEnv(G)
    gen = torch.Generator().manual_seed(1)
    boards = [rules.initial_board() for _ in range(G)]
    movers = [rules.BLACK] * G
    obs = env.reset()
    n_pass = 0
    finished = [False] * G
    while bool(obs.active.any()):
        acts = env.random_actions(obs, generator=gen)
        for i in range(G):
            assert bool(obs.active[i]) == (not finished[i])
            if finished[i]:
                continue
            m = movers[i]
            assert int(obs.to_play[i]) == m - 1
            assert torch.equal(obs.obs["board_seq"][i], rules.board_planes(boards[i], m))
            ref = torch.zeros(64, dtype=torch.bool)
            for r, c in rules.legal_moves(boards[i], m):
                ref[r * 8 + c] = True
            assert torch.equal(obs.legal["place"][i], ref)
            assert bool(obs.forced[i]) == (not ref.any()) == bool(obs.legal["pass"][i])
        obs, rew, done, info = env.step(acts)
        for i in range(G):
            if finished[i]:
                assert not info["acted"][i] and not done[i]
                continue
            assert info["acted"][i]
            if info["applied"]["passed"][i]:
                n_pass += 1
                assert int(info["applied"]["place_idx"][i]) == 0
            else:
                sq = int(acts["place_idx"][i])
                assert int(info["applied"]["place_idx"][i]) == sq
                rules.apply_move(boards[i], movers[i], sq // 8, sq % 8)
            movers[i] = rules.opponent(movers[i])
            assert bool(done[i]) == rules.is_game_over(boards[i])
            if done[i]:
                finished[i] = True
                b, w = rules.disc_counts(boards[i])
                sign = float((b > w) - (b < w))
                assert rew[i].tolist() == [sign, -sign]
                assert int(info["winner"][i]) == (0 if b > w else 1 if w > b else -1)
                assert info["score"][i].tolist() == [b, w]
                assert int(info["episode_len"][i]) == int(obs.step_idx[i])
            else:
                assert rew[i].tolist() == [0.0, 0.0]
    assert all(finished) and env.finished.all() and n_pass > 0


def test_othello_stream_semantics_and_forced_passes():
    env = OthelloVecEnv(8, seed=0)
    gen = torch.Generator().manual_seed(5)
    obs = env.reset()
    prev_to_play = obs.to_play.clone()
    saw_forced = False
    while bool(obs.active.any()):
        active = obs.active.clone()
        forced = obs.forced.clone()
        # garbage placements on forced rows are ignored (the rules pass)
        acts = env.random_actions(obs, generator=gen)
        acts["place_idx"][forced] = 63
        step_before = obs.step_idx.clone()
        obs, _, done, info = env.step(acts)
        saw_forced |= bool(forced.any())
        assert torch.equal(info["applied"]["passed"], forced)
        live = active & ~done
        # one frame per step on the interleaved stream, movers alternate
        # (a pass frame hands the SAME board to the opponent)
        assert torch.equal(obs.step_idx[live], step_before[live] + 1)
        assert torch.equal(obs.to_play[live], 1 - prev_to_play[live])
        assert obs.stream_seats.all() and not obs.new_episode.any()
        prev_to_play = obs.to_play.clone()
    assert saw_forced


def test_othello_illegal_placement_raises():
    env = OthelloVecEnv(2)
    obs = env.reset()
    assert not obs.legal["place"][0, 0]
    with pytest.raises(ValueError, match="illegal"):
        env.step({"place_idx": torch.tensor([0, 19])})


def test_othello_autoreset():
    env = OthelloVecEnv(4, autoreset=True, seed=0)
    gen = torch.Generator().manual_seed(2)
    obs = env.reset()
    ended = 0
    for _ in range(200):
        obs, rew, done, info = env.step(env.random_actions(obs, generator=gen))
        assert obs.active.all()
        if done.any():
            ended += int(done.sum())
            assert torch.equal(obs.new_episode, done)
            assert (obs.step_idx[done] == 0).all() and (obs.to_play[done] == 0).all()
            assert (info["episode_len"][done] > 0).all()
            assert torch.equal(rew[done, 0], -rew[done, 1])     # zero-sum outcome
            assert (rew[~done] == 0).all()
    assert ended >= 4 and not env.finished.any()


# ---------------------------------------------------------------------------
# gomoku
# ---------------------------------------------------------------------------

def _gomoku_has_five(own: np.ndarray, exact: bool) -> bool:
    """Brute force over the WHOLE board (not incremental): does ``own``
    [H, W] bool hold five in a row — exactly five (a run of 5 not extended
    at either end) with ``exact``?"""
    H, W = own.shape
    o = np.pad(own, 6)

    def shift(k, dr, dc):                     # own at (r + k·dr, c + k·dc), False off-board
        return o[6 + k * dr:6 + k * dr + H, 6 + k * dc:6 + k * dc + W]

    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        win = shift(0, dr, dc) & shift(1, dr, dc) & shift(2, dr, dc) & shift(3, dr, dc) \
            & shift(4, dr, dc)
        if exact:
            win = win & ~shift(-1, dr, dc) & ~shift(5, dr, dc)
        if win.any():
            return True
    return False


def _play_gomoku(env, moves):
    """Feed env 0 the (row, col) moves (the other envs: their first empty
    point); returns the last step's outputs."""
    out = None
    for r, c in moves:
        acts = env.observe().legal["place"].to(torch.uint8).argmax(-1)
        acts[0] = r * env.board_w + c
        out = env.step({"place_idx": acts})
    return out


@pytest.mark.parametrize("board_size,rule", [(20, "freestyle"), ((7, 9), "freestyle"),
                                             ((7, 9), "standard"), (4, "freestyle")])
def test_gomoku_matches_bruteforce_rules_on_random_games(board_size, rule):
    """Batched incremental play == a whole-board brute-force scan step by
    step: planes (mover / opponent / empty), legal = empty points, pass never
    legal, termination (five / full board), rewards, winner, episode length."""
    G = 24
    env = GomokuVecEnv(G, board_size=board_size, rule=rule, seed=0)
    H, W = env.board_h, env.board_w
    gen = torch.Generator().manual_seed(3)
    boards = [np.zeros((H, W), dtype=np.int8) for _ in range(G)]
    plies = [0] * G
    finished = [False] * G
    obs = env.reset()
    _check_schema(env, obs)
    assert env.obs_schema() == {"board_seq": ((3, H, W), torch.float32)}
    outcomes = set()
    while bool(obs.active.any()):
        acts = env.random_actions(obs, generator=gen)
        for i in range(G):
            assert bool(obs.active[i]) == (not finished[i])
            if finished[i]:
                assert not obs.legal["place"][i].any()
                continue
            m = plies[i] % 2
            b = boards[i]
            want = torch.from_numpy(np.stack([b == m + 1, b == 2 - m, b == 0]).astype(np.float32))
            assert int(obs.to_play[i]) == m and int(obs.step_idx[i]) == plies[i]
            assert torch.equal(obs.obs["board_seq"][i], want)
            assert torch.equal(obs.legal["place"][i], torch.from_numpy(b.reshape(-1) == 0))
        assert not obs.legal["pass"].any() and not obs.forced.any()
        assert obs.stream_seats.all()
        obs, rew, done, info = env.step(acts)
        for i in range(G):
            if finished[i]:
                assert not info["acted"][i] and not done[i] and (rew[i] == 0).all()
                continue
            sq = int(acts["place_idx"][i])
            assert int(info["applied"]["place_idx"][i]) == sq
            m = plies[i] % 2
            b = boards[i]
            assert b.flat[sq] == 0
            b.flat[sq] = m + 1
            plies[i] += 1
            five = _gomoku_has_five(b == m + 1, exact=rule == "standard")
            full = bool((b != 0).all())
            assert bool(done[i]) == (five or full), (i, plies[i])
            assert bool(info["terminated"][i]) == (five or full)
            assert not info["truncated"][i]
            if five:
                assert rew[i].tolist() == ([1.0, -1.0] if m == 0 else [-1.0, 1.0])
                assert int(info["winner"][i]) == m
                outcomes.add(m)
            else:
                assert rew[i].tolist() == [0.0, 0.0] and int(info["winner"][i]) == -1
                if full:
                    outcomes.add(-1)
            if done[i]:
                finished[i] = True
                assert int(info["episode_len"][i]) == plies[i]
    assert all(finished) and env.finished.all()
    if H * W <= 16:
        assert outcomes == {-1}                  # no five fits: every game is a draw
    else:
        assert {0, 1} <= outcomes


def test_gomoku_overline_wins_freestyle_only():
    """Six in a row (a gap filled between three and two) wins freestyle, not
    standard; standard then wins with exactly five."""
    overline = [(5, 3), (12, 0), (5, 4), (12, 2), (5, 5), (12, 4), (5, 7), (12, 6), (5, 8),
                (12, 8), (5, 6)]
    free = GomokuVecEnv(1, rule="freestyle")
    _, rew, done, info = _play_gomoku(free, overline)
    assert done.all() and rew[0].tolist() == [1.0, -1.0] and int(info["winner"][0]) == 0
    std = GomokuVecEnv(1, rule="standard")
    _, rew, done, _ = _play_gomoku(std, overline)
    assert not done.any() and (rew == 0).all()
    five = [(14, 14), (8, 3), (14, 16), (8, 4), (14, 18), (8, 5), (16, 14), (8, 6), (16, 16),
            (8, 7)]                              # white stones first: it is white's turn
    _, rew, done, info = _play_gomoku(std, five[:-1])
    assert not done.any()
    _, rew, done, info = _play_gomoku(std, five[-1:])
    assert done.all() and rew[0].tolist() == [1.0, -1.0] and int(info["winner"][0]) == 0
    assert int(info["episode_len"][0]) == len(overline) + len(five)


def test_gomoku_draw_truncation_and_max_moves():
    # a full board without five: a (terminated) draw
    env = GomokuVecEnv(3, board_size=4, seed=0)
    gen = torch.Generator().manual_seed(0)
    obs = env.reset()
    for t in range(16):
        assert obs.active.all()
        obs, rew, done, info = env.step(env.random_actions(obs, generator=gen))
        assert bool(done.all()) == (t == 15)
    assert info["terminated"].all() and not info["truncated"].any()
    assert (rew == 0).all() and (info["winner"] == -1).all() and (info["episode_len"] == 16).all()
    assert not obs.active.any() and not obs.legal["place"].any()
    # max_moves: a soft terminal scored as a draw ...
    env = GomokuVecEnv(2, max_moves=10, seed=0)
    obs = env.reset()
    for t in range(10):
        obs, rew, done, info = env.step(env.random_actions(obs, generator=gen))
        assert bool(done.all()) == (t == 9)
    assert info["truncated"].all() and not info["terminated"].any()
    assert (rew == 0).all() and (info["winner"] == -1).all() and (info["episode_len"] == 10).all()
    assert env.finished.all()
    # ... unless the last allowed move wins
    env = GomokuVecEnv(1, max_moves=9)
    fastest = [(0, 15), (5, 5), (0, 16), (6, 6), (0, 17), (7, 7), (0, 18), (8, 8), (0, 19)]
    _, rew, done, info = _play_gomoku(env, fastest)
    assert done.all() and info["terminated"].all() and not info["truncated"].any()
    assert rew[0].tolist() == [1.0, -1.0]
    with pytest.raises(ValueError):
        GomokuVecEnv(1, max_moves=0)
    with pytest.raises(ValueError):
        GomokuVecEnv(1, rule="renju")


def test_gomoku_no_wraparound_across_row_ends():
    """Five consecutive point indices across a row end (or a diagonal that
    would wrap) are no line."""
    env = GomokuVecEnv(1, board_size=20)
    # black 3,17 3,18 3,19 4,0 4,1 = indices 77..81; white far apart
    moves = [(3, 17), (10, 10), (3, 18), (12, 5), (3, 19), (15, 15), (4, 0), (8, 3), (4, 1)]
    _, _, done, _ = _play_gomoku(env, moves)
    assert not done.any()
    # diagonal wrap (index step 21): 0,18 1,19 3,0 4,1 5,2
    env = GomokuVecEnv(1, board_size=20)
    moves = [(0, 18), (15, 7), (1, 19), (16, 9), (3, 0), (17, 11), (4, 1), (18, 13), (5, 2)]
    _, _, done, _ = _play_gomoku(env, moves)
    assert not done.any()


def test_gomoku_illegal_placement_raises_and_pass_is_ignored():
    env = GomokuVecEnv(2, board_size=9)
    obs = env.reset()
    assert obs.legal["place"].all() and not obs.legal["pass"].any()
    obs, *_ = env.step({"action_place": torch.tensor([40, 0]),
                        "action_pass": torch.tensor([True, True])})     # pass flag ignored
    assert not obs.legal["place"][0, 40] and not obs.legal["place"][1, 0]
    assert obs.obs["board_seq"][0, 1, 4, 4] == 1                        # opponent stone now
    with pytest.raises(ValueError, match="illegal"):
        env.step({"place_idx": torch.tensor([40, 1])})                  # occupied
    with pytest.raises(ValueError, match="illegal"):
        env.step({"place_idx": torch.tensor([81, 1])})                  # off the board
    with pytest.raises(KeyError, match="place_idx"):
        env.step({"passed": torch.zeros(2, dtype=torch.bool)})
    # a finished env ignores whatever is sent on its row
    env = GomokuVecEnv(2, max_moves=1)
    env.step({"place_idx": torch.tensor([0, 0])})
    _, rew, done, info = env.step({"place_idx": torch.tensor([0, 0])})
    assert not info["acted"].any() and not done.any() and (rew == 0).all()


def test_gomoku_rectangular_board_and_rng_state():
    env = GomokuVecEnv(2, board_size=(6, 11), seed=4)
    obs = env.observe()
    assert obs.obs["board_seq"].shape == (2, 3, 6, 11) and obs.legal["place"].shape == (2, 66)
    # a vertical five needs 5 of the 6 rows, a horizontal one fits anywhere
    _, rew, done, _ = _play_gomoku(env, [(1, 10), (0, 0), (2, 10), (0, 2), (3, 10), (0, 4),
                                         (4, 10), (0, 6), (5, 10)])
    assert bool(done[0]) and not done[1] and rew[0].tolist() == [1.0, -1.0]
    st = env.state_dict()
    a = env.random_actions(env.reset())["place_idx"]
    env.load_state_dict(st)
    assert torch.equal(env.random_actions(env.reset())["place_idx"], a)


def test_gomoku_random_actions_uniform_over_legal_points():
    env = GomokuVecEnv(4, board_size=5, max_moves=2, seed=0)
    obs = env.reset()
    obs, *_ = env.step({"place_idx": torch.tensor([0, 1, 2, 3])})
    gen = torch.Generator().manual_seed(1)
    counts = torch.zeros(4, 25)
    for _ in range(2000):
        a = env.random_actions(obs, generator=gen)["place_idx"]
        assert obs.legal["place"][torch.arange(4), a].all()
        counts[torch.arange(4), a] += 1
    legal = obs.legal["place"]
    assert (counts[~legal] == 0).all()
    p = counts[legal] / 2000
    assert ((p - 1 / 24).abs() < 0.02).all()                  # 24 legal points per row
    obs, *_ = env.step(env.random_actions(obs, generator=gen))
    assert not obs.active.any()                               # max_moves=2: all done
    a = env.random_actions(obs, generator=gen)["place_idx"]   # inactive rows: in range
    assert ((a >= 0) & (a < 25)).all()


def test_gomoku_autoreset_and_partial_reset():
    env = GomokuVecEnv(3, board_size=6, autoreset=True, seed=0)
    gen = torch.Generator().manual_seed(2)
    obs = env.reset()
    ended = 0
    for _ in range(300):
        obs, rew, done, info = env.step(env.random_actions(obs, generator=gen))
        assert obs.active.all()
        if done.any():
            ended += int(done.sum())
            assert torch.equal(obs.new_episode, done)
            assert (obs.step_idx[done] == 0).all() and (obs.to_play[done] == 0).all()
            assert (obs.obs["board_seq"][done][:, 2] == 1).all()          # empty board
            assert torch.equal(rew[done, 0], -rew[done, 1])
            assert (rew[~done] == 0).all()
    assert ended >= 3 and not env.finished.any()
    obs = env.reset(env_ids=[1])
    assert obs.new_episode.tolist() == [False, True, False]
    assert int(obs.step_idx[1]) == 0 and (obs.obs["board_seq"][1, 2] == 1).all()


def test_gomoku_network_sample_steps_the_env():
    """The gomoku decoder has no pass head: its sample() output steps the env
    directly (action_pass is always False), via a NetworkPolicy."""
    torch.manual_seed(0)
    net = build_game_network("gomoku", **TINY, board_h=9, board_w=9).eval()
    assert net.action_decoder.pass_head is None
    env = GomokuVecEnv(3, board_size=9, max_moves=30, seed=0)
    league = League(env, {"net": NetworkPolicy(net)}, mirror_prob=0.0, seed=0)
    res = play_episodes(env, league, NetworkPolicy(net))
    assert env.finished.all() and (res["episode_len"] > 0).all()
    assert (res["episode_len"] <= 30).all()


# ---------------------------------------------------------------------------
# generals
# ---------------------------------------------------------------------------

def _pass_all(R):
    zero = torch.zeros(R, dtype=torch.long)
    return {"source_idx": zero, "direction": zero, "split": zero,
            "passed": torch.ones(R, dtype=torch.bool)}


def test_generals_rows_legal_and_turn_globals():
    env = GeneralsVecEnv(2, turn_globals=True, seed=1)
    obs = env.observe()
    _check_schema(env, obs)
    assert obs.to_play.tolist() == [0, 1, 0, 1]
    assert obs.stream_seats.tolist() == [[True, False], [False, True]] * 2
    assert set(obs.legal) == {"source", "direction"}
    assert torch.equal(obs.legal["source"], obs.obs["legal_source_mask_seq"])
    assert obs.legal["direction"].shape == (4, 4, 25, 25)
    # turn 0: every army is 1 — nothing can move, the rows are forced passes
    assert obs.forced.all()
    for t in range(1, 60):
        obs, rew, done, info = env.step(_pass_all(4))
        assert (obs.step_idx == t).all() and (info["count"] == 0).all()
        g = obs.obs["globals_seq"]
        assert g.shape == (4, 7)
        want = torch.tensor([(t % 50) / 50.0, float(t % 2), t / 500.0], dtype=torch.float32)
        assert torch.equal(g[:, 4:], want.expand(4, 3))
    assert not obs.forced.any()          # generals grew: sources are movable
    plain = GeneralsVecEnv(1)
    assert plain.obs_schema()["globals_seq"][0] == (4,)
    assert "direction" not in GeneralsVecEnv(1, direction_mask=False).observe().legal


def test_generals_forced_rows_pass_whatever_is_sent():
    env = GeneralsVecEnv(1, seed=0)
    obs = env.observe()
    assert obs.forced.all()
    src = int(env.envs[0].generals[0])
    acts = {"source_idx": torch.tensor([src, 0]), "direction": torch.tensor([0, 0]),
            "split": torch.tensor([0, 0]), "passed": torch.tensor([False, False])}
    _, _, _, info = env.step(acts)
    assert info["count"].tolist() == [0, 0] and not info["illegal"].any()
    assert info["applied"]["passed"].all()


def test_generals_move_counts_executed_dropped_and_illegal():
    """info["count"] follows the IL target semantics: 1 executed, 0 pass,
    -1 sent but dropped (here: off the board / into a mountain)."""
    env = GeneralsVecEnv(1, seed=4, direction_mask=False)
    # general of seat 0 in the top-left corner, a mountain right of it
    m = {"map_w": 6, "map_h": 5, "generals": [0, 29], "cities": [], "city_armies": [],
         "mountains": [1]}
    obs = env.reset(maps=[m])
    for _ in range(12):                                   # let the generals grow
        obs, *_ = env.step(_pass_all(2))
    e = env.envs[0]
    assert e.map[0][0].army > 2 and not obs.forced.any()
    down, right, up, left = 0, 1, 2, 3
    acts = {"source_idx": torch.tensor([0, e.generals[1]]), "direction": torch.tensor([down, 0]),
            "split": torch.tensor([1, 0]), "passed": torch.tensor([False, True])}
    obs, _, _, info = env.step(acts)
    assert info["count"].tolist() == [1, 0] and not info["illegal"].any()
    assert info["applied"]["source_idx"][0] == 0 and info["applied"]["split"][0] == 1
    assert info["applied"]["passed"].tolist() == [False, True]
    assert e.map[1][0].occupied == 0                      # captured the tile below
    for d in (up, left, right):                           # off-board ×2, mountain
        acts["direction"][0] = d
        _, rew, _, info = env.step(acts)
        assert info["count"][0] == -1 and info["illegal"][0] and not info["applied"]["passed"][0]
        assert rew[0, 0] < 0                              # the env's invalid-move charge


def test_generals_global_rng_untouched_and_seeded_episodes():
    random.seed(123)
    state = random.getstate()
    env = GeneralsVecEnv(3, seed=7)
    env.reset()
    assert random.getstate() == state
    a = GeneralsVecEnv(2, seed=0).reset(seed=11).obs["grid_seq"]
    b = GeneralsVecEnv(2, seed=99).reset(seed=11).obs["grid_seq"]
    assert torch.equal(a, b)
    assert not torch.equal(a[0], a[2])                    # seed + k per env


def test_generals_sparse_reward_truncation_and_autoreset():
    env = GeneralsVecEnv(2, max_turns=30, sparse_reward=True, autoreset=True, seed=0)
    assert all(e.city_reward_scale == 0.0 and e.dense_reward_scale == 0.0 for e in env.envs)
    gen = torch.Generator().manual_seed(0)
    obs = env.observe()
    for t in range(1, 31):
        obs, rew, done, info = env.step(env.random_actions(obs, generator=gen))
        if t < 30:
            assert not done.any()
            # sparse: nothing but the -0.05 invalid-move charge before the end
            assert ((rew == 0) | (rew == -0.05)).all()
    assert done.all() and info["truncated"].all() and not info["terminated"].any()
    assert (info["episode_len"] == 30).all() and (info["winner"] == -1).all()
    # soft terminal: ±0.35 × material heuristic
    assert (rew.abs() <= 0.35 + 0.05 + 1e-6).all()
    assert obs.new_episode.all() and (obs.step_idx == 0).all() and obs.active.all()


def _full_state_grids_per_tile(env) -> torch.Tensor:
    """The original per-tile privileged view (oracle for full_state_grids)."""
    import math

    import numpy as np
    H, W = env.grid_h, env.grid_w
    st = np.full((2, H, W), -1, dtype=np.int64)
    army = np.zeros((H, W), dtype=np.float32)
    for r in range(env.real_h):
        for c in range(env.real_w):
            t = env.map[r][c]
            for pid in (0, 1):
                st[pid, r, c] = env._tile_to_state(t, 1 if t.occupied == pid else 2, pid)
            if t.terrain != _MOUNTAIN:
                army[r, c] = math.log1p(t.army) / 10.0
    onehot = (st[:, None] == np.arange(11).reshape(1, -1, 1, 1)).astype(np.float32)
    return torch.from_numpy(np.concatenate([onehot, np.broadcast_to(army, (2, 1, H, W))], 1))


def test_generals_privileged_view_matches_per_tile_reference():
    env = GeneralsVecEnv(4, privileged=True, seed=5, max_turns=400, autoreset=True)
    gen = torch.Generator().manual_seed(3)
    obs = env.observe()
    kinds = set()
    for t in range(400):
        if t % 20 == 0:
            ref = torch.cat([_full_state_grids_per_tile(e) for e in env.envs])
            assert torch.equal(obs.privileged["grid_full"], ref), t
            kinds |= set(torch.nonzero(ref[:, :11].sum((2, 3))).flatten().tolist()[1::2])
        obs, *_ = env.step(env.random_actions(obs, generator=gen))
    assert {2, 3, 4, 5, 7, 8, 10} <= kinds    # neutral land + cities, mountains, own, enemy


def test_generals_memory_vision_and_privileged_view():
    env = GeneralsVecEnv(1, privileged=True, seed=2)
    assert not env.envs[0].memory_vision
    assert GeneralsVecEnv(1, memory_vision=True).envs[0].memory_vision
    gen = torch.Generator().manual_seed(1)
    obs = env.observe()
    for _ in range(40):
        obs, *_ = env.step(env.random_actions(obs, generator=gen))
    full = obs.privileged["grid_full"]
    assert full.shape == (2, 12, 25, 25)
    assert torch.equal(full, full_state_grids(env.envs[0]))
    assert torch.equal(full, _full_state_grids_per_tile(env.envs[0]))
    # the fog view agrees with the full view wherever the seat sees
    for p in (0, 1):
        fog, fv = obs.obs["grid_seq"][p], full[p]
        seen = fog[2:11].sum(0) > 0
        assert torch.equal(fog[:, seen], fv[:, seen])
    # seat 1's full view is seat 0's with own / enemy swapped
    swap = [0, 1, 2, 3, 4, 8, 9, 10, 5, 6, 7, 11]
    assert torch.equal(full[1], full[0][swap])
    # capture-feasibility labels ride along as aux labels
    assert obs.aux["capture_label"].shape == (2, 4, 25, 25)


def test_generals_defaults_follow_the_stage2_recipe():
    env = GeneralsVecEnv(1, trunc_penalty=0.15)
    e = env.envs[0]
    assert e.city_reward_scale == 0.0 and e.dense_reward_scale == 0.0     # sparse ±1
    assert e.max_turns == 2048 and e.trunc_penalty == 0.15
    dense = GeneralsVecEnv(1, sparse_reward=False).envs[0]
    assert dense.city_reward_scale == 1.0 and dense.dense_reward_scale == 1.0


def _head_on_swap_env():
    """Both generals adjacent with equal armies: seat 0 moves right onto
    seat 1's general, seat 1 left onto seat 0's — a full priority tie."""
    env = GeneralsVecEnv(1, seed=0)
    m = {"map_w": 6, "map_h": 5, "generals": [0, 1], "cities": [], "city_armies": [],
         "mountains": []}
    env.reset(maps=[m])
    for _ in range(12):
        env.step(_pass_all(2))
    e = env.envs[0]
    assert e.map[0][0].army == e.map[0][1].army > 2
    right, left = 1, 3
    acts = {"source_idx": torch.tensor([0, 1]), "direction": torch.tensor([right, left]),
            "split": torch.tensor([0, 0]), "passed": torch.tensor([False, False])}
    return env, acts


def test_generals_tiebreak_seat_for_replays():
    """Full priority ties go to seat 0 by default; ``tiebreak_first`` (the
    replay's move order) flips which seat is executed and which dropped."""
    for tb, want in ((None, [1, -1]), ([0], [1, -1]), ([1], [-1, 1])):
        env, acts = _head_on_swap_env()
        a = env.envs[0].map[0][0].army
        _, _, done, info = env.step(acts, tiebreak_first=tb)
        assert info["count"].tolist() == want and not done.any()
        e = env.envs[0]
        assert e.tiebreak_first == 0                         # restored after the step
        # the board is the same either way: both generals end at army 1 + growth
        assert e.map[0][0].occupied == 0 and e.map[0][1].occupied == 1
        assert {e.map[0][0].army, e.map[0][1].army} <= {1, 2} and a > 2
    env, acts = _head_on_swap_env()
    with pytest.raises(ValueError, match="tiebreak_first"):
        env.step(acts, tiebreak_first=[2])
    replay = {"mapWidth": 6, "moves": [
        {"turn": 3, "index": 1, "start": 1, "end": 0, "is50": False},
        {"turn": 3, "index": 0, "start": 0, "end": 1, "is50": False},
        {"turn": 5, "index": 0, "start": 0, "end": 1, "is50": False},
        {"turn": 6, "index": 0, "start": 0, "end": 1, "is50": False},
        {"turn": 6, "index": 1, "start": 1, "end": 0, "is50": False}]}
    assert replay_tiebreak_by_step(replay) == {4: 1, 7: 0}


def test_generals_pinned_map_keeps_seats():
    env = GeneralsVecEnv(1, seed=0)
    m = {"map_w": 6, "map_h": 5, "generals": [0, 29], "cities": [14], "city_armies": [45],
         "mountains": [7, 8]}
    for _ in range(3):
        env.reset(maps=[m])
        e = env.envs[0]
        assert e.generals == [0, 4 * 25 + 5] and (e.real_h, e.real_w) == (5, 6)
        assert e.map[1][1].terrain == _MOUNTAIN


# ---------------------------------------------------------------------------
# league
# ---------------------------------------------------------------------------

def test_league_mirror_by_default():
    env = OthelloVecEnv(3)
    league = League(env)
    obs = env.reset()
    assert league.update(obs) == [0, 1, 2]
    assert league.is_mirror().all()
    assert torch.equal(league.learner_rows(obs), obs.active)
    assert torch.equal(league.learner_acting(obs), obs.active)
    assert league.learner_score(torch.tensor([0, 1, -1])).isnan().all()


def test_league_interleaved_stream_routes_both_movers():
    """othello vs an opponent: the learner processes EVERY frame of the
    stream (its temporal state needs the opponent's frames too) but acts —
    and takes the loss — only on its own seat's frames."""
    env = OthelloVecEnv(4, seed=0)
    gen_l, gen_o = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
    calls = []

    def opp(o):
        a = env.random_actions(o, generator=gen_o)
        calls.append((o.stream_id.clone(), a["place_idx"].clone()))
        return a

    league = League(env, {"x": CallablePolicy(opp)}, mirror_prob=0.0, learner_seat=0)
    obs = env.reset()
    league.update(obs)
    assert not league.is_mirror().any() and league.learner_seats.tolist() == [[True, False]] * 4
    assert league.opponent_of == ["x"] * 4
    n_steps = 0
    while bool(obs.active.any()):
        assert torch.equal(league.learner_rows(obs), obs.active)
        acting = league.learner_acting(obs)
        assert torch.equal(acting, obs.active & (obs.to_play == 0))
        part, opp_acting = league.opponent_rows(obs, "x")
        assert torch.equal(part, obs.active)
        assert torch.equal(opp_acting, obs.active & (obs.to_play == 1))
        la = env.random_actions(obs, generator=gen_l)
        calls.clear()
        merged = league.act(obs, la)
        # the opponent processed every active frame of the stream ...
        assert len(calls) == 1 and torch.equal(calls[0][0], part.nonzero().squeeze(1))
        opp_place = torch.zeros(4, dtype=torch.long)
        opp_place[calls[0][0]] = calls[0][1]
        # ... but its actions are used only where it is to play
        assert torch.equal(merged["place_idx"][acting], la["place_idx"][acting])
        assert torch.equal(merged["place_idx"][opp_acting], opp_place[opp_acting])
        obs, *_ = env.step(merged)
        league.update(obs)
        n_steps += 1
    assert n_steps >= 60


def test_league_seat_streams_in_generals():
    env = GeneralsVecEnv(2, seed=0)
    league = League(env, {"rand": RandomPolicy(env)}, mirror_prob=0.0, learner_seat=1)
    obs = env.reset()
    league.update(obs)
    # the learner plays seat 1: only its own fog-of-war stream
    assert league.learner_rows(obs).tolist() == [False, True, False, True]
    assert league.learner_acting(obs).tolist() == [False, True, False, True]
    part, acting = league.opponent_rows(obs, "rand")
    assert part.tolist() == acting.tolist() == [True, False, True, False]
    acts = league.act(obs, scatter_rows(env.random_actions(obs.select(torch.tensor([1, 3]))),
                                        torch.tensor([1, 3]), 4))
    assert set(acts) == {"source_idx", "direction", "split", "passed"}
    env.step(acts)


def test_league_seat_randomisation_and_state():
    env = OthelloVecEnv(64)
    league = League(env, {"a": RandomPolicy(env), "b": RandomPolicy(env, seed=1)},
                    opponent_weights={"a": 1.0, "b": 3.0}, mirror_prob=0.25, seed=3)
    obs = env.reset()
    league.update(obs)
    mirror = league.is_mirror()
    assert 0 < int(mirror.sum()) < 64
    seats = league.learner_seats[~mirror].float().argmax(1)
    assert set(seats.tolist()) == {0, 1}
    names = [n for n in league.opponent_of if n is not None]
    assert names.count("b") > names.count("a") > 0
    st = league.state_dict()
    league.update(env.reset())
    a = (league.learner_seats.clone(), list(league.opponent_of))
    league.load_state_dict(st)
    league.update(env.reset())
    assert torch.equal(a[0], league.learner_seats) and a[1] == league.opponent_of


@pytest.mark.parametrize("game", ["othello", "generals", "gomoku"])
def test_play_episodes_random_vs_random(game):
    kw = {"max_turns": 60} if game == "generals" else {}
    env = make_vec_env(game, 6, seed=0, **kw)
    league = League(env, {"rand": RandomPolicy(env, seed=1)}, mirror_prob=0.0, seed=0)
    res = play_episodes(env, league, RandomPolicy(env, seed=2))
    assert env.finished.all()
    assert set(res["score"].tolist()) <= {0.0, 0.5, 1.0}
    assert (res["episode_len"] > 0).all()
    if game == "othello":
        assert (res["winner"] >= -1).all() and (res["episode_len"] >= 60).all()


# ---------------------------------------------------------------------------
# network policies
# ---------------------------------------------------------------------------

def _othello_net(mixer, seed=0):
    torch.manual_seed(seed)
    return build_game_network("othello", **TINY, temporal_mixer=mixer).eval()


@pytest.mark.parametrize("mixer", ["attention", "recurrent"])
def test_stream_stepper_regrouping_equals_per_stream_stepping(mixer):
    """Streams stepped in changing subsets / orders (block gathers) give the
    outputs of stepping each stream alone with its own cache."""
    net = _othello_net(mixer)
    env = OthelloVecEnv(4, seed=0)
    gen = torch.Generator().manual_seed(0)
    frames = []
    obs = env.reset()
    for _ in range(6):
        frames.append(obs)
        obs, *_ = env.step(env.random_actions(obs, generator=gen))

    ref = {s: [] for s in range(4)}
    for s in range(4):
        cache = None
        for t, o in enumerate(frames):
            with torch.no_grad():
                out = net.step({"board_seq": o.obs["board_seq"][s:s + 1]},
                               legal={"place": o.legal["place"][s:s + 1]},
                               st_cache=cache, step_idx=t)
            cache = out["st_cache"]
            ref[s].append(out["place_logits"][0])

    st = StreamStepper(net)
    # t=0, t=1 lockstep (the second call reuses the batched cache as is);
    # at t=2 streams 3, 1 step alone, then 2, 0 (gathers out of one block);
    # t=3 merges the two blocks, t=4 permutes the order
    orders = [[0, 1, 2, 3], [0, 1, 2, 3], [3, 1], [2, 0], [0, 1, 2, 3], [1, 3, 0, 2]]
    seen = {s: 0 for s in range(4)}
    for order in orders:
        ts = [seen[s] for s in order]
        board = torch.stack([frames[seen[s]].obs["board_seq"][s] for s in order])
        legal = torch.stack([frames[seen[s]].legal["place"][s] for s in order])
        out = st.step({"board_seq": board}, {"place": legal}, order, ts)
        for k, s in enumerate(order):
            assert torch.allclose(out["place_logits"][k], ref[s][seen[s]], atol=1e-5), (order, s)
            seen[s] += 1
    assert all(st.frames_seen(s) == 5 for s in range(4))
    # mixed frame indices in one call: grouped per index
    out = st.step({"board_seq": frames[5].obs["board_seq"][[0, 1]]}, None, [0, 1], [5, 5])
    assert out["place_logits"].shape == (2, 64)
    with pytest.raises(ValueError, match="frames seen"):
        st.step({"board_seq": frames[0].obs["board_seq"][:1]}, None, [0], [0])
    st.reset([0])
    st.step({"board_seq": frames[0].obs["board_seq"][:1]}, None, [0], [0])
    assert st.frames_seen(0) == 1


def _count_frames(stepper):
    """Wrap ``stepper.step`` to record the last frame fed per stream."""
    fed = {}
    step = stepper.step

    def counting(obs, legal, streams, step_idx):
        for s, t in zip(streams, step_idx):
            fed[int(s)] = int(t) + 1
        return step(obs, legal, streams, step_idx)

    stepper.step = counting
    return fed


def test_network_opponent_sees_every_frame_of_its_stream():
    env = OthelloVecEnv(3, seed=0)
    net = _othello_net("recurrent")
    opp = NetworkPolicy(net)
    fed = _count_frames(opp.stepper)
    league = League(env, {"net": opp}, mirror_prob=0.0, learner_seat=1)
    res = play_episodes(env, league, RandomPolicy(env, seed=0))
    # the interleaved stream was fed to the opponent at every frame, both movers
    for s in range(3):
        assert fed[s] == int(res["episode_len"][s])
    assert set(res["score"].tolist()) <= {0.0, 0.5, 1.0}
    assert res["opponent"] == ["net"] * 3 and res["opponent_snapshot"] == [None] * 3
    assert not net.training
    # lockstep: one batched call per env step; finished streams were released
    assert opp.stepper.n_calls == int(res["steps"]) and opp.stepper.n_rows == sum(fed.values())
    assert opp.stepper.live_streams() == [] and opp.stepper.cache_numel() == 0


def _own_state_numel(st: StreamStepper) -> int:
    """Elements the live streams' own states need (their rows of the blocks)."""
    own = 0
    for s in st.live_streams():
        blk = st._blocks[st._where[s][0]]
        own += sum(c.numel() // len(blk.streams) for c in blk.cache)
    return own


@pytest.mark.parametrize("mixer", ["attention", "recurrent"])
def test_stream_stepper_memory_stays_with_live_streams(mixer):
    """Streams that stop being fed (envs ending at different turns) must not
    pin the batched blocks they were last stepped in: the held cache stays
    within 2x the live streams' own state, and release frees it."""
    net = _othello_net(mixer)
    N = 16
    obs = OthelloVecEnv(N, seed=0).reset()
    st = StreamStepper(net)
    live = list(range(N))
    for t in range(N - 1):
        st.step({"board_seq": obs.obs["board_seq"][live]}, None, live, [t] * len(live))
        live = live[:-1]                        # one env ends after every step
    assert len(st.live_streams()) == N
    assert st.frames_seen(0) == N - 1 and st.frames_seen(N - 1) == 1
    assert st.cache_numel() <= 2 * _own_state_numel(st)
    assert st.n_calls == N - 1
    st.reset(range(N))
    assert st.cache_numel() == 0 and st.live_streams() == [] and not st._blocks


def test_stream_stepper_compaction_keeps_outputs_exact():
    """Compacted blocks give the same outputs as stepping each stream alone."""
    net = _othello_net("attention")
    env = OthelloVecEnv(6, seed=1)
    gen = torch.Generator().manual_seed(0)
    frames = []
    obs = env.reset()
    for _ in range(5):
        frames.append(obs)
        obs, *_ = env.step(env.random_actions(obs, generator=gen))
    st, solo = StreamStepper(net), StreamStepper(net)
    # t0: all six together; then stream groups shrink so the first block is
    # compacted (rows 0-3 leave, 4-5 stay behind), then released streams
    plan = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3], [0, 1, 2, 3], [4, 5], [4, 5]]
    seen = {s: 0 for s in range(6)}
    for k, order in enumerate(plan):
        if k == 3:
            st.reset([0, 1])                          # two envs end
        board = torch.stack([frames[seen[s]].obs["board_seq"][s] for s in order])
        out = st.step({"board_seq": board}, None, order, [seen[s] for s in order])
        for j, s in enumerate(order):
            o = solo.step({"board_seq": board[j:j + 1]}, None, [s], [seen[s]])
            assert torch.allclose(out["place_logits"][j], o["place_logits"][0], atol=1e-5)
            seen[s] += 1
    assert set(st.live_streams()) == {2, 3, 4, 5}
    assert st.cache_numel() <= 2 * _own_state_numel(st)


def test_network_policy_generals_and_greedy_restores_state():
    torch.manual_seed(0)
    net = build_game_network("generals", **TINY).train()
    env = GeneralsVecEnv(2, seed=0, max_turns=8)
    pol = NetworkPolicy(net, greedy=True, output_keys=("value",))
    league = League(env, {"net": pol}, mirror_prob=0.0, learner_seat=0)
    res = play_episodes(env, league, RandomPolicy(env))
    assert (res["episode_len"] == 8).all()
    assert net.training and net.action_decoder.greedy is False
    obs = env.reset()
    out = pol(obs, torch.tensor([0, 3]))
    assert {"action_source", "action_direction", "action_split", "action_pass", "value"} <= set(out)
    assert out["action_source"].shape == (2,)


def test_snapshot_pool_and_pool_policy():
    pool = SnapshotPool(max_size=2, seed=0)
    nets = [_othello_net("recurrent", seed=s) for s in range(3)]
    ids = [pool.add(n.state_dict(), {"iter": k}) for k, n in enumerate(nets)]
    assert pool.ids() == ids[1:] and len(pool) == 2 and pool.meta(ids[2]) == {"iter": 2}
    st = pool.state_dict()
    pool2 = SnapshotPool()
    pool2.load_state_dict(st)
    assert pool2.ids() == pool.ids()
    assert [pool.sample() for _ in range(8)] == [pool2.sample() for _ in range(8)]
    pool.weights = lambda ids_, meta: [0.0, 1.0]
    assert {pool.sample() for _ in range(5)} == {ids[2]}
    pool.weights = None

    built = []

    def load_net(state):
        n = build_game_network("othello", **TINY, temporal_mixer="recurrent")
        n.load_state_dict(state)
        built.append(n)
        return n.eval()

    env = OthelloVecEnv(6, seed=0)
    pp = PoolPolicy(pool, load_net, max_nets=2)
    league = League(env, {"pool": pp}, mirror_prob=0.0)
    res = play_episodes(env, league, RandomPolicy(env))
    assert set(res["score"].tolist()) <= {0.0, 0.5, 1.0}
    sids = set(res["opponent_snapshot"])
    assert sids <= set(pool.ids()) and 1 <= len(built) <= 2
    # every stream was released when its env ended
    assert all(pp.snapshot_of(s) is None for s in range(6))
    assert all(p.stepper.cache_numel() == 0 for p in pp._nets.values())
    with pytest.raises(RuntimeError, match="empty"):
        PoolPolicy(SnapshotPool(), load_net).reset_streams([0])


def _pool_of(n, **kw):
    pool = SnapshotPool(max_size=16, seed=0, **kw)
    for s in range(n):
        pool.add(_othello_net("recurrent", seed=s).state_dict(), {"iter": s})
    return pool


def _loader(built):
    def load_net(state):
        n = build_game_network("othello", **TINY, temporal_mixer="recurrent")
        n.load_state_dict(state)
        built.append(n)
        return n.eval()
    return load_net


def test_pool_policy_caps_networks_and_calls_per_iteration():
    """8 snapshots, max_nets=2: each play_episodes iteration draws an active
    set of 2, so at most 2 networks are loaded and each env step makes at
    most 2 batched opponent calls; the next iteration redraws the set."""
    pool = _pool_of(8)
    built = []
    pp = PoolPolicy(pool, _loader(built), max_nets=2)
    env = OthelloVecEnv(32, seed=0)
    league = League(env, {"pool": pp}, mirror_prob=0.0, seed=1)
    seen_sets = []
    for _ in range(3):
        res = play_episodes(env, league, RandomPolicy(env))
        used = set(res["opponent_snapshot"])
        assert used <= set(pp.active) and len(pp.active) == 2
        assert len(pp.loaded()) <= 2
        calls = sum(p.stepper.n_calls for p in pp._nets.values())
        assert calls <= 2 * int(res["steps"])
        seen_sets.append(tuple(sorted(pp.active)))
        for p in pp._nets.values():
            p.stepper.n_calls = 0
        assert pp._sid == {}
    assert len(set(seen_sets)) > 1                 # redrawn per iteration
    assert len(built) <= 2 * 3
    # the pool fits the active set: episodes follow the pool's weights (PFSP hook)
    small = _pool_of(3)
    small.weights = lambda ids, meta: [float(m["iter"] == 1) for m in meta]
    pp = PoolPolicy(small, _loader([]), max_nets=4)
    res = play_episodes(env, League(env, {"pool": pp}, mirror_prob=0.0), RandomPolicy(env))
    assert sorted(pp.active) == small.ids() and set(res["opponent_snapshot"]) == {1}


def test_league_releases_reassigned_opponent_streams():
    """An env handed from a network opponent to mirror / another opponent
    releases the old opponent's streams (autoreset)."""
    env = OthelloVecEnv(8, autoreset=True, seed=0)
    net = NetworkPolicy(_othello_net("recurrent"))
    league = League(env, {"net": net, "rand": RandomPolicy(env)}, mirror_prob=0.3, seed=0)
    gen = torch.Generator().manual_seed(0)
    obs = env.reset()
    league.update(obs)
    for _ in range(300):
        obs, _, done, info = env.step(league.act(obs, env.random_actions(obs, generator=gen)))
        league.update(obs)
        held = {s for i, ss in league._held.items() if league.opponent_of[i] == "net" for s in ss}
        assert set(net.stepper.live_streams()) <= held
    assert net.stepper.cache_numel() <= 2 * _own_state_numel(net.stepper)


def test_league_autoreset_keeps_finished_episode_assignment():
    """With autoreset, update() reassigns an env on the step it ended; the
    finished episode's assignment stays readable (ended / prev_learner_seats)."""
    env = OthelloVecEnv(32, autoreset=True, seed=0)
    league = League(env, {"a": RandomPolicy(env), "b": RandomPolicy(env, seed=1)},
                    mirror_prob=0.2, seed=0)
    gen = torch.Generator().manual_seed(0)
    obs = env.reset()
    league.update(obs)
    n_ended = n_changed = 0
    for _ in range(140):
        before_seats = league.learner_seats.clone()
        before_opp = list(league.opponent_of)
        obs, _, done, info = env.step(league.act(obs, env.random_actions(obs, generator=gen)))
        want = league.learner_score(info["winner"])          # before update
        league.update(obs)
        got = league.learner_score(info["winner"], league.prev_learner_seats)
        assert torch.equal(league.prev_learner_seats, before_seats)
        d = done.nonzero().squeeze(1).tolist()
        assert set(league.ended) == set(d)
        for i in d:
            n_ended += 1
            rec = league.ended[i]
            assert torch.equal(rec["learner_seats"], before_seats[i])
            assert rec["opponent"] == before_opp[i]
            n_changed += int(not torch.equal(before_seats[i], league.learner_seats[i]))
            assert torch.equal(want[i], got[i]) or (want[i].isnan() and got[i].isnan())
    assert n_ended >= 32 and n_changed > 0


def test_league_policy_mask_drops_forced_frames():
    env = OthelloVecEnv(64, seed=0)
    league = League(env)
    gen = torch.Generator().manual_seed(0)
    obs = env.reset()
    league.update(obs)
    n_forced = 0
    while bool(obs.active.any()):
        mask = league.learner_policy_mask(obs)
        assert torch.equal(mask, league.learner_acting(obs) & ~obs.forced)
        n_forced += int((league.learner_acting(obs) & obs.forced).sum())
        obs, *_ = env.step(env.random_actions(obs, generator=gen))
        league.update(obs)
    assert n_forced > 0


def test_snapshot_pool_file_backed(tmp_path):
    """store_dir: one file per snapshot, state_dict keeps names + meta only,
    eviction deletes the file, a resume may point store_dir elsewhere."""
    nets = [_othello_net("recurrent", seed=s) for s in range(3)]
    d = tmp_path / "snaps"
    pool = SnapshotPool(max_size=2, store_dir=str(d))
    ids = [pool.add(n.state_dict(), {"iter": k}) for k, n in enumerate(nets)]
    assert sorted(os.listdir(d)) == ["snapshot_000001.pt", "snapshot_000002.pt"]
    st = pool.state_dict()
    assert all(isinstance(e, str) for _, e, _ in st["snapshots"])
    got = pool.get(ids[2])
    ref = nets[2].state_dict()
    assert set(got) == set(ref) and all(torch.equal(got[k], ref[k]) for k in ref)
    moved = tmp_path / "moved"
    os.rename(d, moved)
    pool2 = SnapshotPool(store_dir=str(moved))
    pool2.load_state_dict(st)
    assert pool2.ids() == ids[1:] and pool2.meta(ids[1]) == {"iter": 1}
    assert torch.equal(pool2.get(ids[1])[next(iter(ref))], nets[1].state_dict()[next(iter(ref))])
    os.remove(moved / "snapshot_000001.pt")
    pool3 = SnapshotPool(store_dir=str(moved))
    with pytest.warns(UserWarning, match="missing"):
        pool3.load_state_dict(st)
    assert pool3.ids() == [ids[2]]


def test_snapshot_pool_deferred_delete(tmp_path):
    """defer_delete: evicted files stay on disk (a checkpoint written before
    the eviction still restores the full pool) until flush_deletes."""
    d = tmp_path / "snaps"
    pool = SnapshotPool(max_size=1, store_dir=str(d), defer_delete=True)
    pool.add({"w": torch.zeros(2)})
    st = pool.state_dict()                        # "checkpoint" before the eviction
    pool.add({"w": torch.ones(2)})
    assert pool.ids() == [1] and pool.pending_deletes == ["snapshot_000000.pt"]
    assert sorted(os.listdir(d)) == ["snapshot_000000.pt", "snapshot_000001.pt"]
    resumed = SnapshotPool(store_dir=str(d))
    resumed.load_state_dict(st)                   # no "missing" warning: nothing dropped
    assert resumed.ids() == [0] and torch.equal(resumed.get(0)["w"], torch.zeros(2))
    pool.flush_deletes()
    assert os.listdir(d) == ["snapshot_000001.pt"] and pool.pending_deletes == []


class _MultiHeadNet(torch.nn.Module):
    """A stub step(): per-unit actions, a replay mask and logits under
    non-``action_*`` keys (Lux / mahjong / chess decoders)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def step(self, obs, legal=None, st_cache=None, step_idx=0, act_steps=None):
        B = obs["board_seq"].shape[0]
        return {"unit_actions": torch.arange(B * 3).view(B, 3),
                "legal_tile": torch.ones(B, 5).bool(),
                "tile_logits": torch.zeros(B, 5), "tera_logit": torch.zeros(B),
                "action_log_prob": torch.zeros(B), "value": torch.ones(B),
                "q_halt": torch.zeros(()), "st_cache": [torch.zeros(B, 2)]}


def test_network_policy_keeps_every_decoder_output_but_logits():
    env = OthelloVecEnv(3, seed=0)
    obs = env.reset()
    out = NetworkPolicy(_MultiHeadNet())(obs, torch.tensor([0, 2]))
    assert set(out) == {"unit_actions", "legal_tile", "action_log_prob", "value"}
    assert out["unit_actions"].shape == (2, 3)
    out = NetworkPolicy(_MultiHeadNet(), output_keys=("tile_logits",))(obs, torch.tensor([1]))
    assert "tile_logits" in out and "tera_logit" not in out


def test_replay_actions_keep_per_row_shapes():
    from env.vec_game import VecEnvBase

    class _Units(VecEnvBase):
        game = "units"
        ACTION_KEYS = {"unit_actions": "unit_actions"}

    e = _Units(2, autoreset=False, seed=0)
    a = e.replay_actions({"unit_actions": torch.arange(6).view(2, 3), "value": torch.zeros(2)})
    assert a["unit_actions"].shape == (2, 3)
    e._require(a, ("unit_actions",))
    full = scatter_rows({"unit_actions": torch.ones(1, 3, dtype=torch.long)}, torch.tensor([1]), 2)
    assert full["unit_actions"].tolist() == [[0, 0, 0], [1, 1, 1]]


def test_scatter_rows():
    full = scatter_rows({"a": torch.tensor([5, 6])}, torch.tensor([3, 0]), 4)
    assert full["a"].tolist() == [6, 0, 0, 5]
