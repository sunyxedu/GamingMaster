"""env/per_seat.py: the board games' simulators served as per-seat streams — the layout
IL trains their records in (``dataset.SEAT_SPLIT_GAMES``).

- ``make_vec_env`` serves the SEAT_SPLIT_GAMES per seat by default, the other games as
  their simulators emit them; ``per_seat=False`` keeps the interleaved rows;
- the frames are the interleaved env's, bit for bit: at every step the active per-seat
  row holds the interleaved row's frame (obs, legal, forced), ``step_idx`` counts the
  seat's own frames, rewards / done / per-env info are unchanged, the per-row info is
  spread over the rows;
- a row's frames in order ARE the IL seat stream of the game (``dataset.seat_stream``
  of the interleaved record);
- episodes: autoreset restarts the streams, ``reset(env_ids=...)`` only its envs,
  ``reset_from`` starts every seat at its prefix's decisions, the state round-trips,
  attribute writes reach the wrapped env.
"""

import numpy as np
import pytest
import torch

from data_processing.games.dataset import SEAT_SPLIT_GAMES, seat_stream
from env.per_seat import PerSeatStreams
from env.vec_game import il_streams_per_seat, make_vec_env

BOARD_GAMES = ["othello", "gomoku", "chess", "chess_variants", "go", "shogi", "xiangqi"]
#: small boards / short games where the game has the option
KW = {"gomoku": {"board_size": 9}, "go": {"board_size": 9}}


def _kw(game):
    return dict(KW.get(game, {}))


def _spread(acts, rows, R):
    """Per-env actions [N, ...] → per-seat rows [R, ...] (the mover's row)."""
    out = {}
    for k, v in acts.items():
        full = v.new_zeros((R,) + tuple(v.shape[1:]))
        full[rows] = v
        out[k] = full
    return out


def test_default_layout_follows_the_il_streams():
    assert sorted(SEAT_SPLIT_GAMES) == sorted(BOARD_GAMES)
    for g in BOARD_GAMES:
        assert il_streams_per_seat(g)
        env = make_vec_env(g, 2, seed=0, **_kw(g))
        assert isinstance(env, PerSeatStreams)
        assert not env.interleaved and not env.lockstep
        assert env.rows_per_env == env.n_seats == 2 and env.num_rows == 4 and env.num_envs == 2
        assert env.game == g
        raw = make_vec_env(g, 2, seed=0, per_seat=False, **_kw(g))
        assert raw.interleaved and raw.rows_per_env == 1 and not isinstance(raw, PerSeatStreams)
    # backgammon's records are not split (its movers do not strictly alternate): interleaved
    # in IL and in the simulator alike
    assert not il_streams_per_seat("backgammon")
    assert make_vec_env("backgammon", 1).interleaved
    # hidden-information games are per seat in the simulator already: returned as they are
    hunl = make_vec_env("hunl", 1, per_seat=True)
    assert not isinstance(hunl, PerSeatStreams) and not hunl.interleaved
    with pytest.raises(ValueError, match="interleaved"):
        PerSeatStreams(hunl)


@pytest.mark.parametrize("game", BOARD_GAMES)
def test_rows_hold_the_interleaved_frames_and_the_il_seat_streams(game):
    N, S = 3, 2
    a = make_vec_env(game, N, seed=5, **_kw(game))
    b = make_vec_env(game, N, seed=5, per_seat=False, **_kw(game))
    oa, ob = a.reset(seed=9), b.reset(seed=9)
    gen = torch.Generator().manual_seed(0)
    seen = torch.zeros(N, S, dtype=torch.long)
    record = [[] for _ in range(N)]                   # the interleaved record of every env
    rows_seen = [[] for _ in range(N * S)]            # the frames of every per-seat row
    first_mover = [None] * N
    assert oa.new_episode.all() and oa.n_rows == N * S
    steps = 0
    while bool(ob.active.any()) and steps < 160:
        act = ob.active.bool()
        mover = ob.to_play.long()
        rows = torch.arange(N) * S + mover
        # exactly the movers' rows are active, each holding the interleaved frame
        assert torch.equal(oa.active.nonzero().squeeze(1), rows[act])
        assert torch.equal(oa.to_play, torch.arange(N * S) % S)
        assert torch.equal(oa.env_id, torch.arange(N * S) // S)
        assert torch.equal(oa.stream_seats, torch.eye(S, dtype=torch.bool).repeat(N, 1))
        assert set(oa.obs) == set(ob.obs) and set(oa.legal) == set(ob.legal)
        for k in ob.obs:
            assert oa.obs[k].dtype == ob.obs[k].dtype
            assert torch.equal(oa.obs[k][rows[act]], ob.obs[k][act]), (game, k)
        for k in ob.legal:
            assert torch.equal(oa.legal[k][rows[act]], ob.legal[k][act]), (game, k)
        assert torch.equal(oa.forced[rows[act]], ob.forced[act])
        assert not oa.forced[~oa.active].any()
        # step_idx = the seat's own frame count
        assert torch.equal(oa.step_idx[rows[act]], seen[act, mover[act]])
        for e in act.nonzero().squeeze(1).tolist():
            if first_mover[e] is None:
                first_mover[e] = int(mover[e])
            frame = {k: v[e].clone() for k, v in ob.obs.items()}
            record[e].append(frame)
            rows_seen[int(rows[e])].append(frame)
        acts = b.random_actions(ob, generator=gen)
        seen[act, mover[act]] += 1
        ob, rb, db, ib = b.step(acts)
        oa, ra, da, ia = a.step(_spread(acts, rows, N * S))
        assert torch.equal(ra, rb) and torch.equal(da, db)
        for k in ("terminated", "truncated", "winner", "episode_len"):
            assert torch.equal(ia[k], ib[k]), k
        # per-row info on the per-seat rows
        assert ia["acted"].shape[0] == N * S
        assert torch.equal(ia["acted"].nonzero().squeeze(1), rows[act])
        for k, v in ib["applied"].items():
            assert torch.equal(ia["applied"][k][rows[act]], v[act]), k
        steps += 1
    assert steps > 4
    # a row's frames, in order, are the IL seat stream of the interleaved record
    for e in range(N):
        T = len(record[e])
        sample = {"obs": {k: torch.stack([f[k] for f in record[e]]) for k in record[e][0]},
                  "tgt": {"value": torch.zeros(T)}, "meta": {}}
        for k in range(S):                            # k = 0: the record's first mover
            if T <= k:
                continue
            seat = (first_mover[e] + k) % S
            stream = seat_stream(sample, k)["obs"]
            mine = rows_seen[e * S + seat]
            assert len(mine) == next(iter(stream.values())).shape[0]
            for key, v in stream.items():
                assert torch.equal(torch.stack([f[key] for f in mine]), v), (game, key, k)


def test_autoreset_and_partial_reset_restart_the_streams():
    env = make_vec_env("othello", 3, seed=1, autoreset=True)
    obs = env.reset(seed=2)
    gen = torch.Generator().manual_seed(1)
    ended = 0
    for _ in range(140):
        acts = env.random_actions(obs, generator=gen)
        was = obs.step_idx.clone()
        obs, rew, done, info = env.step(acts)
        for e in done.nonzero().squeeze(1).tolist():
            ended += 1
            # the env restarted on the step: both streams flagged, counts back at 0
            assert obs.new_episode[2 * e: 2 * e + 2].all()
            assert obs.step_idx[2 * e: 2 * e + 2].tolist() == [0, 0]
            assert obs.active[2 * e] and not obs.active[2 * e + 1]      # black moves first
        live = ~done[obs.env_id]
        assert (obs.step_idx[live] >= was[live]).all()
    assert ended >= 3
    # reset of one env leaves the other envs' streams where they are
    env = make_vec_env("gomoku", 3, seed=1, board_size=9)
    obs = env.reset(seed=3)
    for _ in range(5):
        obs, *_ = env.step(env.random_actions(obs, generator=gen))
    before = obs.step_idx.clone()
    obs = env.reset(env_ids=[1])
    assert obs.step_idx[2:4].tolist() == [0, 0] and obs.new_episode[2:4].all()
    assert torch.equal(obs.step_idx[[0, 1, 4, 5]], before[[0, 1, 4, 5]])
    assert not obs.new_episode[[0, 1, 4, 5]].any()
    # resizing reset
    obs = env.reset(5, seed=4)
    assert obs.n_rows == 10 and env.num_envs == 5 and not obs.step_idx.any()


def test_random_actions_of_a_row_subset_and_state_roundtrip():
    env = make_vec_env("chess", 4, seed=3)
    obs = env.reset(seed=4)
    gen = torch.Generator().manual_seed(2)
    for _ in range(7):
        obs, *_ = env.step(env.random_actions(obs, generator=gen))
    rows = obs.active.nonzero().squeeze(1)[[2, 0]]
    sub = env.random_actions(obs.select(rows), generator=torch.Generator().manual_seed(8))
    full = env.random_actions(obs, generator=torch.Generator().manual_seed(8))
    for k in sub:
        assert sub[k].shape[0] == 2 and torch.equal(sub[k], full[k][rows])
    state = env.state_dict()
    assert "per_seat_frames" in state
    twin = make_vec_env("chess", 4, seed=99)
    twin.reset(seed=4)
    g2 = torch.Generator().manual_seed(2)
    o2 = twin.observe()
    for _ in range(7):
        o2, *_ = twin.step(twin.random_actions(o2, generator=g2))
    twin.load_state_dict(state)
    assert torch.equal(twin.observe().step_idx, obs.step_idx)
    # a state written without the wrapper (an older checkpoint) loads too
    older = {k: v for k, v in state.items() if k != "per_seat_frames"}
    twin.load_state_dict(older)
    # attribute writes reach the wrapped env (the trainer's duplicate deals)
    env.deal_group = 2
    assert env.env.deal_group == 2 and env.deal_group == 2


def test_reset_from_starts_every_seat_at_its_prefix_decisions():
    raw = make_vec_env("chess", 1, seed=3, per_seat=False)
    obs = raw.reset(seed=6)
    gen = torch.Generator().manual_seed(5)
    for _ in range(9):
        obs, *_ = raw.step(raw.random_actions(obs, generator=gen))
    prefix = np.asarray(raw.trajectory(0)).reshape(-1, 2)
    env = make_vec_env("chess", 2, seed=3)
    env.reset(seed=1)
    obs = env.reset_from([prefix], env_ids=[1])
    assert obs.step_idx[2:4].tolist() == [5, 4]             # white decided 5 times, black 4
    assert obs.step_idx[0:2].tolist() == [0, 0]
    assert obs.active[2:4].tolist() == [False, True]        # black to move after 9 plies
    for k, v in raw.observe().obs.items():
        assert torch.equal(obs.obs[k][3], v[0]), k
