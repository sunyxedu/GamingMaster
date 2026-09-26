"""env/vec_game.py (VecGameEnv protocol + registry), the othello / generals
adapters (env/vec_othello.py, env/vec_generals.py) and the league interface
(env/league.py).  Frame-by-frame equality with the IL data produced by the
Rust converters lives in test_vec_game_golden.py."""

from __future__ import annotations

import random

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
from env.vec_generals import GeneralsVecEnv, full_state_grids
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

@pytest.mark.parametrize("game", ["othello", "generals"])
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
    assert set(VEC_ENV_REGISTRY) >= {"generals", "othello"}


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


@pytest.mark.parametrize("game", ["othello", "generals"])
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


def test_network_opponent_sees_every_frame_of_its_stream():
    env = OthelloVecEnv(3, seed=0)
    net = _othello_net("recurrent")
    opp = NetworkPolicy(net)
    league = League(env, {"net": opp}, mirror_prob=0.0, learner_seat=1)
    res = play_episodes(env, league, RandomPolicy(env, seed=0))
    # the interleaved stream was fed to the opponent at every frame, both movers
    for s in range(3):
        assert opp.stepper.frames_seen(s) == int(res["episode_len"][s])
    assert set(res["score"].tolist()) <= {0.0, 0.5, 1.0}
    assert not net.training


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
    sids = {pp.snapshot_of(s) for s in range(6)}
    assert sids <= set(pool.ids()) and 1 <= len(built) <= 2
    with pytest.raises(RuntimeError, match="empty"):
        PoolPolicy(SnapshotPool(), load_net).reset_streams([0])


def test_scatter_rows():
    full = scatter_rows({"a": torch.tensor([5, 6])}, torch.tensor([3, 0]), 4)
    assert full["a"].tolist() == [6, 0, 0, 5]
