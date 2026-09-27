"""env/vec_lux.py — Lux AI Season 3 over the official env (luxai-s3, JAX):
VecGameEnv protocol, episode = match with game continuation, rewards, legal
masks against the rules, determinism / env-param randomisation, state_dict,
the exact normalisation tables, the LuxDecoder's sample() / log_prob_entropy
consistency on env frames (sap offsets included), the trainer's LuxSpec and
a tiny end-to-end train_rl_joint run.  Frame-by-frame equality with Kaggle
replays and the Rust converter lives in test_vec_lux_golden.py."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("luxai_s3")

from data_processing.games.dataset import save_shard  # noqa: E402
from env.league import League, RandomPolicy, StreamStepper, play_episodes  # noqa: E402
from env.vec_game import VecGameEnv, has_simulator, make_vec_env  # noqa: E402
from env.vec_lux import (  # noqa: E402
    DIVISORS,
    MOVE_DELTAS,
    SAP_WINDOW,
    VIEW_ACTION,
    LuxVecEnv,
    div_tables_np,
    param_defaults,
    sap_class,
    sap_delta,
)
from model.games import build_game_network  # noqa: E402
from model.games.lux import lux_sap_delta, lux_sap_legal, lux_sap_offset  # noqa: E402

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, encoder_channels=8,
            dropout=0.0, rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1,
            rd_mean_iters=2, rd_backprop_iters=1)
OBS_KEYS = ("map_seq", "units_seq", "unit_mask", "globals_seq")
SHORT = {"max_steps_in_match": 10}          # 11-frame matches for the quick tests


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _check_schema(env, obs):
    R = env.num_rows
    for k, (shape, dtype) in env.obs_schema().items():
        assert obs.obs[k].shape == (R, *shape) and obs.obs[k].dtype == dtype, k
    assert obs.legal["unit"].shape == (R, 16) and obs.legal["unit"].dtype == torch.bool
    assert obs.legal["unit_action"].shape == (R, 16, 6)
    assert obs.legal["sap"].shape == (R, 16, (2 * SAP_WINDOW + 1) ** 2)
    for f in ("to_play", "env_id", "stream_id", "step_idx"):
        assert getattr(obs, f).shape == (R,) and getattr(obs, f).dtype == torch.long, f
    for f in ("new_episode", "active", "forced"):
        assert getattr(obs, f).shape == (R,) and getattr(obs, f).dtype == torch.bool, f
    assert torch.equal(obs.env_id, torch.arange(R) // 2)
    assert torch.equal(obs.to_play, torch.arange(R) % 2)
    assert obs.stream_seats.gather(1, obs.to_play.view(-1, 1)).all()
    assert obs.legal["unit_action"][..., 0].all()                   # noop always legal


# ---------------------------------------------------------------------------
# helpers shared with the model
# ---------------------------------------------------------------------------

def test_division_tables_are_lux_rs_normalisations():
    """float32(float64(n) / d) for every int16 n: lux.rs's `(x as f64 / d) as f32`."""
    t = div_tables_np()
    n = np.arange(-(1 << 15), 1 << 15, dtype=np.int64)
    for i, d in enumerate(DIVISORS):
        want = np.array([np.float32(float(v) / d) for v in n[::97]], dtype=np.float32)
        assert np.array_equal(t[i, ::97], want), d
    assert t[DIVISORS.index(20), (1 << 15) + 7] == np.float32(7 / 20)


def test_sap_helpers_agree_with_the_decoder():
    o = torch.arange((2 * SAP_WINDOW + 1) ** 2)
    dx, dy = sap_delta(o)
    assert torch.equal(torch.stack(lux_sap_delta(o, SAP_WINDOW)), torch.stack([dx, dy]))
    assert torch.equal(sap_class(dx, dy), o)
    assert torch.equal(lux_sap_offset(dx, dy, SAP_WINDOW), o)
    assert sap_class(2, -1) == 99 and sap_class(0, 0) == 112


def test_param_defaults_are_typed():
    p = param_defaults()
    assert p["unit_move_cost"].dtype == np.int16 and p["fog_of_war"].dtype == np.bool_
    assert p["energy_node_drift_speed"].dtype == np.float32


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------

def test_registry_protocol_and_resets():
    env = make_vec_env("lux", 3, env_params=SHORT)
    assert isinstance(env, VecGameEnv) and has_simulator("lux") and env.game == "lux"
    assert env.num_rows == 6 and env.rows_per_env == 2 and not env.interleaved
    obs = env.observe()
    _check_schema(env, obs)
    assert obs.new_episode.all() and obs.active.all() and (obs.step_idx == 0).all()
    assert obs.forced.all()                    # frame 0: every unit is removed first
    assert not env.finished.any()
    obs, rew, done, info = env.step(env.random_actions(obs, _gen()))
    assert (obs.step_idx == 1).all() and not obs.new_episode.any() and not done.any()
    assert (rew == 0).all() and info["acted"].all()
    obs = env.reset(env_ids=[1])               # mid-match: env 1 starts a fresh game
    assert obs.new_episode.tolist() == [e == 1 for e in obs.env_id.tolist()]
    assert (obs.step_idx[obs.env_id == 1] == 0).all()
    assert (obs.step_idx[obs.env_id != 1] == 1).all()
    obs = env.reset(5)
    assert env.num_envs == 5 and obs.n_rows == 10 and (obs.step_idx == 0).all()
    _check_schema(env, obs)
    with pytest.raises(KeyError, match="unit_actions"):
        env.step({"sap_offsets": torch.zeros(10, 16, dtype=torch.long)})
    with pytest.raises(ValueError, match="fix array shapes"):
        LuxVecEnv(1, env_params={"map_width": 16})
    with pytest.raises(KeyError, match="unknown"):
        LuxVecEnv(1, env_params={"bogus": 1})


def test_games_are_episodes_and_matches_are_rewarded():
    """One game = one episode (5 matches of match_steps 0..10 here, step_idx = the
    env's steps 0..54): the map, relics and point tiles carry over, so the stream does
    too; each match's winner gets +1, the loser −1 on the step that ends it, and the
    next frame is the next match's frame 0; the 5th match ends the episode, and a
    reset deals a fresh game."""
    env = LuxVecEnv(2, env_params=SHORT, seed=3)
    obs = env.reset(seed=11, new_game=True)
    L = SHORT["max_steps_in_match"] + 1
    g = _gen(1)
    wins = torch.zeros(2, 2, dtype=torch.long)
    for m in range(5):
        assert (env.matches_played == m).all()
        for t in range(L):
            assert (obs.step_idx == m * L + t).all() and obs.active.all()
            assert bool(obs.new_episode.all()) == (m == 0 and t == 0)
            assert not obs.new_episode.any() or (m, t) == (0, 0)
            assert torch.allclose(obs.obs["globals_seq"][:, 4] * 101, torch.tensor(float(t)))
            obs, rew, done, info = env.step(env.random_actions(obs, g))
            if t < L - 1:
                assert not done.any() and (rew == 0).all() and not info["match_ended"].any()
                assert (info["winner"] == -1).all()
        assert info["match_ended"].all() and (info["match_index"] == m).all()
        w = info["winner"]
        assert ((w == 0) | (w == 1)).all()
        assert torch.equal(rew, torch.where(torch.arange(2) == w.view(-1, 1), 1.0, -1.0))
        wins[torch.arange(2), w] += 1
        assert torch.equal(info["team_wins"], wins)
        last = m == 4
        assert done.tolist() == [last] * 2 and info["terminated"].tolist() == [last] * 2
        assert not info["truncated"].any() and info["game_over"].tolist() == [last] * 2
        assert (info["episode_len"] == (5 * L if last else 0)).all()
        assert bool(obs.active.any()) != last and bool(env.finished.all()) == last
        if not last:     # the next match's frame 0, the wins so far in the globals
            wg = obs.obs["globals_seq"][:, 2:4] * 5                  # own / opp wins
            assert torch.equal(wg[0::2].round().long(), wins.float().long())
    assert (info["game_winner"] == wins.argmax(1)).all() and env.game_over.all()
    obs = env.reset()
    assert (env.matches_played == 0).all() and not env.game_over.any()
    assert obs.new_episode.all() and (obs.step_idx == 0).all()
    assert (obs.obs["globals_seq"][:, 2:6] == 0).all()               # wins, steps: fresh


def test_team_1_sees_the_reflected_world():
    """Team 1's frame is team 0's convention on the world reflected by T(x, y) =
    (23 − y, 23 − x) (lux.rs): its units sit at T of their positions, the tiles
    under them are T's; its decoder actions are turned back into the world (up <->
    right, down <-> left, saps (dx, dy) -> (−dy, −dx)) and ``applied`` keeps the view."""
    env = LuxVecEnv(1, seed=21)
    obs = env.observe()
    g = _gen(9)
    for _ in range(12):                                   # units on the map, some energy
        obs, *_ = env.step(env.random_actions(obs, g))
    raw = env.raw_obs()
    pos = raw[1]["units"]["position"][0, 1]                # team 1, world (x, y)
    live = raw[1]["units_mask"][0, 1]
    assert live.any()
    u = obs.obs["units_seq"][1]
    for k in np.flatnonzero(live):
        x, y = int(pos[k][0]), int(pos[k][1])
        assert torch.allclose(u[k, :2], torch.tensor([(23 - y) / 24, (23 - x) / 24]))
    tiles = raw[1]["map_features"]["tile_type"][0]         # [x][y], -1 unseen
    asteroid = obs.obs["map_seq"][1, 2]                     # [y][x] view
    for x in range(24):
        for y in range(24):
            assert float(asteroid[23 - x, 23 - y]) == float(tiles[x, y] == 2)
    # team 0's frame is the world itself
    pos0 = raw[0]["units"]["position"][0, 0]
    for k in np.flatnonzero(raw[0]["units_mask"][0, 0]):
        assert torch.allclose(obs.obs["units_seq"][0, k, :2],
                              torch.tensor([pos0[k][0] / 24, pos0[k][1] / 24]).float())
    # actions: the view's "up" of a team 1 unit moves it right in the world
    ua = torch.zeros(2, 16, dtype=torch.long)
    k = int(np.flatnonzero(live)[0])
    ua[1, k] = 1
    env_act, applied = env._env_actions({"unit_actions": ua})
    assert env_act[0, 1, k].tolist() == [2, 0, 0] and applied["unit_actions"][1, k] == 1
    so = torch.full((2, 16), sap_class(0, 0))
    ua[1, k], so[1, k] = 5, sap_class(2, -1)
    env_act, applied = env._env_actions({"unit_actions": ua, "sap_offsets": so})
    assert env_act[0, 1, k].tolist() == [5, 1, -2] and applied["sap_offsets"][1, k] == sap_class(2, -1)
    raw_act = np.zeros((1, 2, 16, 3), dtype=np.int64)
    raw_act[0, 1, k] = [5, 1, -2]
    raw_act[0, 1, (k + 1) % 16] = [3, 0, 0]
    _, applied = env._raw_actions(raw_act)
    assert applied["sap_offsets"][1, k] == sap_class(2, -1)
    assert applied["unit_actions"][1, (k + 1) % 16] == 4                  # down -> left


def test_game_reward_and_autoreset():
    env = LuxVecEnv(2, env_params=SHORT, game_reward=0.5, autoreset=True, seed=0)
    obs = env.observe()
    g = _gen(2)
    L = SHORT["max_steps_in_match"] + 1
    seen_over = False
    for k in range(5 * L):
        obs, rew, done, info = env.step(env.random_actions(obs, g))
        assert obs.active.all()
        if done.any():
            assert done.all() and obs.new_episode.all() and (obs.step_idx == 0).all()
            if info["game_over"].any():
                seen_over = True
                gw = info["game_winner"]
                w = info["winner"]
                exp = torch.where(torch.arange(2) == w.view(-1, 1), 1.0, -1.0) + \
                    torch.where(torch.arange(2) == gw.view(-1, 1), 0.5, -0.5)
                assert torch.equal(rew, exp)
                assert (env.matches_played == 0).all()                # autoreset: new game
    assert seen_over


def test_seeded_games_are_deterministic_and_params_randomised():
    a = LuxVecEnv(8, env_params=SHORT, seed=0)
    oa = a.reset(seed=123, new_game=True)
    b = LuxVecEnv(8, env_params=SHORT, seed=99)
    ob = b.reset(seed=123, new_game=True)
    for k in OBS_KEYS:
        assert torch.equal(oa.obs[k], ob.obs[k])
    for k in a._params:
        assert np.array_equal(a._params[k], b._params[k])
    # the competition's randomisation: different games draw different params
    assert len(set(a._params["unit_sap_cost"].tolist())) > 1
    assert set(a._params["unit_sap_range"].tolist()) <= set(range(3, 8))
    g = oa.obs["globals_seq"]
    assert torch.allclose(g[:, 7] * 50, torch.from_numpy(
        np.repeat(a._params["unit_sap_cost"], 2)).float())
    assert (g[:, 10] == 1).all()                                     # env params known
    # pinned params (constructor and per-reset)
    c = LuxVecEnv(2, env_params={**SHORT, "unit_sap_range": 5}, seed=0)
    assert (c._params["unit_sap_range"] == 5).all()
    c.reset(new_game=True, params=[{"unit_move_cost": 4}, None])
    assert c._params["unit_move_cost"][0] == 4 and (c._params["unit_sap_range"] == 5).all()


def test_state_dict_continues_exactly():
    env = LuxVecEnv(2, env_params=SHORT, seed=4)
    g = _gen(3)
    obs = env.observe()
    for _ in range(4):
        obs, *_ = env.step(env.random_actions(obs, g))
    st = env.state_dict()
    acts = []
    frames = []
    for _ in range(9):
        a = env.random_actions(obs, g)
        acts.append(a)
        obs, rew, done, info = env.step(a)
        frames.append((obs.obs["map_seq"].clone(), rew.clone(), done.clone()))
        if done.all():
            obs = env.reset()
    fresh = LuxVecEnv(3, env_params=SHORT, seed=77)          # resized by the load
    fresh.load_state_dict(st)
    assert fresh.num_envs == 2
    assert fresh._rng.bit_generator.state == st["rng"]
    assert torch.equal(fresh._torch_rng.get_state(), st["torch_rng"])
    obs = fresh.observe()
    for a, (m, r, d) in zip(acts, frames):
        obs, rew, done, info = fresh.step(a)
        assert torch.equal(obs.obs["map_seq"], m) and torch.equal(rew, r) and torch.equal(done, d)
        if done.all():
            obs = fresh.reset()


# ---------------------------------------------------------------------------
# legal masks vs the rules
# ---------------------------------------------------------------------------

def test_legal_masks_follow_the_rules():
    """Legal moves move a surviving unit by its delta and cost unit_move_cost;
    masked moves leave it in place; saps are legal exactly within the sap
    range on the map (lux_sap_legal), and need energy >= unit_sap_cost."""
    env = LuxVecEnv(4, seed=5)
    g = _gen(4)
    obs = env.observe()
    checked_legal = checked_masked = 0
    for step in range(60):
        raw = env.raw_obs()
        a = env.random_actions(obs, g)
        if step % 3 == 0:                           # probe masked moves too
            a["unit_actions"] = torch.randint(0, 5, (8, 16), generator=g)
        legal = obs.legal
        new, rew, done, info = env.step(a)
        raw2 = env.raw_obs()
        for r in range(8):
            i, p = r // 2, r % 2
            for k in range(16):
                if not legal["unit"][r, k]:
                    continue
                act = int(a["unit_actions"][r, k])
                if not 1 <= act <= 4 or not raw2[p]["units_mask"][i, p, k]:
                    continue
                pos0 = raw[p]["units"]["position"][i, p, k]
                pos1 = raw2[p]["units"]["position"][i, p, k]
                if legal["unit_action"][r, k, act]:
                    # team 1 acts in its reflected view: the world move is T's
                    dx, dy = MOVE_DELTAS[(VIEW_ACTION[act] if p == 1 else act) - 1]
                    assert pos1.tolist() == [pos0[0] + dx, pos0[1] + dy], (step, r, k, act)
                    checked_legal += 1
                else:
                    assert pos1.tolist() == pos0.tolist(), (step, r, k, act)
                    checked_masked += 1
        # sap legality = lux_sap_legal over the gated units, energy-gated action
        xy = torch.from_numpy(np.stack([raw[p]["units"]["position"][:, p] for p in (0, 1)], 1)
                              ).reshape(8, 16, 2)
        # team 1's saps are legal in its view: its units at T(x, y) = (23 - y, 23 - x)
        view = torch.stack([23 - xy[..., 1], 23 - xy[..., 0]], -1)
        xy = torch.where((torch.arange(8) % 2 == 1).view(8, 1, 1) & (xy >= 0).all(-1, keepdim=True),
                         view, xy)
        rng = torch.from_numpy(np.repeat(env._params["unit_sap_range"].astype(np.int64), 2))
        want = lux_sap_legal(xy, legal["unit"], rng, SAP_WINDOW, 24, 24)
        assert torch.equal(legal["sap"], want)
        en = torch.from_numpy(np.stack([raw[p]["units"]["energy"][:, p] for p in (0, 1)], 1)
                              ).reshape(8, 16)
        cost = torch.from_numpy(np.repeat(env._params["unit_sap_cost"].astype(np.int64), 2))
        assert torch.equal(legal["unit_action"][..., 5],
                           legal["unit"] & (en >= cost.view(-1, 1)) & want.any(-1))
        obs = new
        if done.all():
            obs = env.reset()
    assert checked_legal > 50 and checked_masked > 10


def test_legal_masks_are_exactly_the_actions_with_an_effect():
    """Brute force against the official rules: from mid-match states, every
    probe (one unit's move, or its sap at a sampled legal / masked target,
    every other unit noop) is stepped next to the all-noop baseline from the
    same state and RNG key.  A legal action always changes the next state, a
    masked one never does — except a move off the map, which the env charges
    energy for without moving (masked as useless)."""
    M = 96                                          # copies per batched probe step
    src = LuxVecEnv(2, seed=11)
    probe = LuxVecEnv(M, seed=0)
    g = _gen(1)
    rng = np.random.default_rng(0)
    obs = src.observe()
    n = {"legal": 0, "masked": 0, "offmap": 0}
    for step in range(45):
        obs, *_ = src.step(src.random_actions(obs, g))
        if step not in (20, 44):                    # mid-match (101-frame matches)
            continue
        st = src.state_dict()
        lg = obs.legal
        for i in range(2):
            probes = []                             # (seat, unit, action, sap class, legal)
            for p in (0, 1):
                r = 2 * i + p
                for u in np.flatnonzero(lg["unit"][r].numpy()):
                    for a in (1, 2, 3, 4):
                        probes.append((p, u, a, sap_class(0, 0), bool(lg["unit_action"][r, u, a])))
                    ok = lg["sap"][r, u].numpy()
                    for c in rng.choice(np.flatnonzero(ok), 2, replace=False):
                        probes.append((p, u, 5, int(c), bool(lg["unit_action"][r, u, 5])))
                    for c in rng.choice(np.flatnonzero(~ok), 2, replace=False):
                        probes.append((p, u, 5, int(c), False))
            assert probes
            for k0 in range(0, len(probes), M - 1):
                chunk = probes[k0:k0 + M - 1]
                games = {
                    key: (np.repeat(np.asarray(v)[i:i + 1], M, 0) if key in ("keys", "wins")
                          else v) for key, v in st["games"].items()}
                games["state"] = [np.repeat(x[i:i + 1], M, 0) for x in st["games"]["state"]]
                games["params"] = {k: np.repeat(v[i:i + 1], M, 0)
                                   for k, v in st["games"]["params"].items()}
                for key, val in (("started", True), ("live", True), ("game_over", False),
                                 ("new", False)):
                    games[key] = np.full(M, val)
                probe.load_state_dict({**probe.state_dict(), "games": games})
                ua = torch.zeros(2 * M, 16, dtype=torch.long)
                so = torch.full((2 * M, 16), sap_class(0, 0), dtype=torch.long)
                for j, (p, u, a, c, _) in enumerate(chunk):   # copy 0: the baseline
                    ua[2 * (j + 1) + p, u], so[2 * (j + 1) + p, u] = a, c
                probe.step({"unit_actions": ua, "sap_offsets": so})
                after = probe.state_dict()["games"]["state"]
                for j, (p, u, a, c, legal) in enumerate(chunk):
                    effect = any(not np.array_equal(x[j + 1], x[0]) for x in after)
                    if a <= 4 and not legal:
                        x, y = (obs.obs["units_seq"][2 * i + p, u, :2] * 24).round().long().tolist()
                        dx, dy = MOVE_DELTAS[a - 1]
                        if not (0 <= x + dx < 24 and 0 <= y + dy < 24):
                            n["offmap"] += 1
                            continue
                    assert effect == legal, (step, i, p, u, a, c, legal)
                    n["legal" if legal else "masked"] += 1
    assert n["legal"] > 100 and n["masked"] > 100 and n["offmap"] > 0, n


def test_env_sap_legal_is_the_il_rule_on_the_gated_units():
    """The IL batches' sap legality (model.games.lux.lux_legal_from_obs: unit
    cells from units_seq, the sap range from globals_seq) is the env's
    legal["sap"] restricted to the units whose action has an effect."""
    from model.games.lux import lux_legal_from_obs

    env = LuxVecEnv(3, seed=8)
    obs = env.observe()
    g = _gen(8)
    n = 0
    for _ in range(25):
        obs, *_ = env.step(env.random_actions(obs, g))
        il = lux_legal_from_obs(obs.obs)
        gate = obs.legal["unit"]
        assert not (gate & ~il["unit"]).any()                   # gate ⊆ occupied slots
        assert torch.equal(il["sap"] & gate.unsqueeze(-1), obs.legal["sap"])
        n += int(gate.sum())
    assert n > 0


def test_action_mask_off_and_legacy_window():
    env = LuxVecEnv(2, action_mask=False, sap_window=0, seed=0)
    obs = env.observe()
    for _ in range(6):
        obs, *_ = env.step(env.random_actions(obs, _gen()))
    assert "sap" not in obs.legal
    la = obs.legal["unit_action"]
    assert torch.equal(la[..., 1:], obs.legal["unit"].unsqueeze(-1).expand(-1, -1, 5))
    # a legacy (no sap head) decoder's actions: saps target the own cell
    obs, *_ = env.step({"unit_actions": torch.full((4, 16), 5, dtype=torch.long)})
    assert "sap_offsets" not in env.step({"unit_actions": torch.zeros(4, 16, dtype=torch.long)}
                                         )[3]["applied"]


def test_play_episodes_random_vs_random():
    env = LuxVecEnv(3, env_params=SHORT, seed=0)
    league = League(env, {"rand": RandomPolicy(env, seed=1)}, mirror_prob=0.0, seed=0)
    res = play_episodes(env, league, RandomPolicy(env, seed=2))
    assert env.finished.all()
    assert set(res["score"].tolist()) <= {0.0, 0.5, 1.0}
    assert (res["episode_len"] == 5 * (SHORT["max_steps_in_match"] + 1)).all()   # a game


# ---------------------------------------------------------------------------
# the LuxDecoder on env frames
# ---------------------------------------------------------------------------

def _lux_net(seed=0, **kw):
    torch.manual_seed(seed)
    return build_game_network("lux", **TINY, lux_global_features=11,
                              lux_sap_range=SAP_WINDOW, **kw).eval()


def test_decoder_sample_matches_log_prob_entropy_on_env_frames():
    """Stream-stepped sample() (legal from the env) vs the full-sequence
    forward's log_prob_entropy of the same actions: equal log-probs (sap
    offsets included), legal actions, and the env accepts them."""
    net = _lux_net()
    env = LuxVecEnv(2, env_params=SHORT, seed=6)
    obs = env.observe()
    st = StreamStepper(net)
    R = obs.n_rows
    seq = {"obs": [], "legal": [], "act": [], "lp": []}
    n_sap = 0
    for t in range(5 * (SHORT["max_steps_in_match"] + 1)):          # the whole game
        with torch.no_grad():
            out = st.step(obs.obs, obs.legal, list(range(R)), obs.step_idx.tolist())
        ua, so = out["unit_actions"], out["sap_offsets"]
        gate = obs.legal["unit"]
        assert obs.legal["unit_action"].gather(-1, ua.unsqueeze(-1)).squeeze(-1)[gate].all()
        is_sap = (ua == 5) & gate
        assert obs.legal["sap"].gather(-1, so.unsqueeze(-1)).squeeze(-1)[is_sap].all()
        n_sap += int(is_sap.sum())
        seq["obs"].append(obs.obs)
        seq["legal"].append(obs.legal)
        seq["act"].append({"unit_actions": ua, "sap_offsets": so})
        seq["lp"].append(out["action_log_prob"])
        obs, rew, done, info = env.step(out)
        assert torch.equal(info["applied"]["unit_actions"], ua)
        assert torch.equal(info["applied"]["sap_offsets"], so)
    assert done.all()

    def stack(dicts):
        return {k: torch.stack([d[k] for d in dicts], 1) for k in dicts[0]}

    o, lg, act = stack(seq["obs"]), stack(seq["legal"]), stack(seq["act"])
    with torch.no_grad():
        pred = net(o, legal=lg)
        lp, ent, heads = net.action_decoder.log_prob_entropy(pred, act, lg)
    mu = torch.stack(seq["lp"], 1)
    assert torch.allclose(lp, mu, atol=1e-4), (lp - mu).abs().max()
    assert torch.isfinite(ent).all() and (ent >= 0).all()
    assert n_sap > 0 and "ent_sap" in heads
    # frames without a controllable unit (every frame 0) carry log-prob 0
    assert (lp[:, 0] == 0).all() and (mu[:, 0] == 0).all()


def test_lux_spec():
    from training.rl_joint import head_terms_kl
    from training.train_rl_joint import GAME_SPECS, LuxSpec, game_spec

    spec = game_spec("lux")
    assert isinstance(spec, LuxSpec) and GAME_SPECS["lux"] is spec and spec.full_kl
    net = _lux_net()
    assert spec.env_kwargs(net.cfg) == {"sap_window": SAP_WINDOW}
    env = LuxVecEnv(1, env_params=SHORT, **spec.env_kwargs(net.cfg))
    spec.check_env(env, net)
    with pytest.raises(SystemExit, match="lux_global_features"):
        spec.check_env(env, build_game_network("lux", **TINY))       # legacy v1 net
    with pytest.raises(SystemExit, match="sap window"):
        spec.check_env(LuxVecEnv(1, env_params=SHORT, sap_window=0), net)
    # full-head KL on env frames: 0 against itself, > 0 against another net
    obs = env.observe()
    frames = []
    for _ in range(4):
        frames.append(obs)
        obs, *_ = env.step(env.random_actions(obs, _gen(5)))
    o = {k: torch.stack([f.obs[k] for f in frames], 1) for k in OBS_KEYS}
    lg = {k: torch.stack([f.legal[k] for f in frames], 1) for k in frames[0].legal}
    act = env.random_actions(frames[-1], _gen(6))
    act = {k: v.unsqueeze(1).expand(-1, 4, -1) for k, v in act.items()}
    mask = torch.ones(2, 4, dtype=torch.bool)
    tgt = spec.kl_targets(act, torch.zeros(2, 4, dtype=torch.bool), net.cfg)
    assert torch.equal(tgt["sap_offset"] >= 0, act["unit_actions"] == 5)
    with torch.no_grad():
        p1 = net(o, legal=lg)
        p2 = _lux_net(seed=1)(o, legal=lg)
        t1 = spec.kl_terms(net, p1, tgt, mask, lg)
        t2 = spec.kl_terms(net, p2, tgt, mask, lg)
    assert torch.equal(t1["unit_action"]["mask"], lg["unit_action"])
    assert float(head_terms_kl(t1, t1)["kl"]) == pytest.approx(0.0, abs=1e-6)
    kl = float(head_terms_kl(t1, t2)["kl"])
    assert math.isfinite(kl) and kl > 0


# ---------------------------------------------------------------------------
# a tiny end-to-end train_rl_joint run
# ---------------------------------------------------------------------------

def _rollout_samples(n_envs: int, seed: int):
    """Random-policy matches of the env as converter-v2 IL samples."""
    env = LuxVecEnv(n_envs, env_params=SHORT, seed=seed)
    obs = env.reset(seed=seed, new_game=True)
    g = _gen(seed)
    R = env.num_rows
    rec = [{"obs": {k: [] for k in OBS_KEYS}, "ua": [], "so": []} for _ in range(R)]
    while bool(obs.active.any()):
        a = env.random_actions(obs, g)
        for r in range(R):
            for k in OBS_KEYS:
                rec[r]["obs"][k].append(obs.obs[k][r])
        obs, rew, done, info = env.step(a)
        ap = info["applied"]
        for r in range(R):
            rec[r]["ua"].append(ap["unit_actions"][r])
            rec[r]["so"].append(torch.where(ap["unit_actions"][r] == 5, ap["sap_offsets"][r], -1))
    out = []
    for r in range(R):
        o = {k: torch.stack(v) for k, v in rec[r]["obs"].items()}
        T = o["map_seq"].shape[0]
        out.append({"obs": o, "tgt": {"unit_actions": torch.stack(rec[r]["ua"]),
                                      "sap_offset": torch.stack(rec[r]["so"]),
                                      "unit_mask": o["unit_mask"].clone(),
                                      "value": torch.full((T,), float(rew[r // 2, r % 2]))},
                    "meta": {"src": f"rollout{seed}", "row": r}})
    return out


def _lux_stage1(root, recurrent: bool = False):
    """Converter-v2 shaped lux shards + a tiny Stage-1 checkpoint."""
    from tests.test_core_moe import CORE
    from training import train_multi_game

    data = root / "lux"
    save_shard(str(data / "a.pt"), "lux", _rollout_samples(2, 0) + _rollout_samples(2, 1),
               converter_version=2, meta={"global_features": 11, "sap_range": SAP_WINDOW})
    extra = {"temporal_mixer": "recurrent"} if recurrent else {}
    core = [f"{k}={v}" for k, v in {**CORE, "encoder_channels": 8, **extra}.items()]
    torch.manual_seed(0)
    train_multi_game.main(["--data", f"lux={data}", "--steps", "1", "--batch_size", "2",
                           "--game_max_len", "lux=6", "--warmup_steps", "1",
                           "--save_dir", str(root / "ck"), "--device", "cpu",
                           "--override", *core])
    return data, root / "ck" / "multi_final.pt"


def _lux_rl_args(data, init, save_dir, *extra):
    return ["--init", str(init), "--rl", "lux",
            "--data", f"lux={data}", "--replay_max_len", "lux=6", "--replay_batch_size", "2",
            "--tokens_per_iter", "40", "--envs", "lux=2",
            "--env_kw", "lux:env_params={'max_steps_in_match': 10}",
            "--epochs", "1", "--minibatches", "2", "--seed", "3",
            "--snapshot_dir", "memory", "--device", "cpu", "--save_dir", str(save_dir), *extra]


@pytest.mark.parametrize("actor", ["stream", "fixed"])
def test_train_rl_joint_on_lux(tmp_path, actor):
    """Stage 1 (train_multi_game on converter-v2 shaped lux data) → 2 Stage-2
    iterations of --rl lux: finite metrics, an exact replay (π/π_old = 1),
    the actor's μ = π_old, the full-head KL anchor.  The fixed-batch actor
    (recurrent temporal mixer) checks the lockstep contract: every row's
    step_idx is the wave's frame count (a wave = one game)."""
    from training import train_rl_joint

    data, init = _lux_stage1(tmp_path, recurrent=actor == "fixed")
    torch.manual_seed(0)
    out = train_rl_joint.main(_lux_rl_args(data, init, tmp_path / "rl", "--actor", actor,
                                           "--iterations", "2"))
    hist = out["history"]
    assert [r["iteration"] for r in hist] == [1, 2]
    for rec in hist:
        row = rec["games"]["lux"]
        for k in ("entropy", "kl", "value_loss", "canary_log_old_mu", "canary_log_ratio",
                  "il_loss", "core_gnorm_rl"):
            assert math.isfinite(row[k]), (k, row[k])
        assert row["tokens"] >= 1 and row["episodes"] >= 2
        assert row["ep_len"] == 5 * (SHORT["max_steps_in_match"] + 1)      # a game
        assert row["canary_log_ratio"] < 1e-5 and row["canary_log_old_mu"] < 1e-4
        assert row["kl"] >= 0.0
    assert out["trainer"].rl["lux"].spec.full_kl


def test_train_rl_joint_lux_resume_continues_the_games(tmp_path):
    """The env state (its RNG streams and games) is part of the RL state: a run
    resumed from its iteration-1 checkpoint deals the same games and reproduces
    iteration 2 bit for bit.  A wave plays whole games here (one episode = one
    game of 5 matches)."""
    from training import train_rl_joint

    data, init = _lux_stage1(tmp_path)
    base = ("--iterations", "2", "--save_every", "1", "--mirror_prob", "lux=0.5",
            "--snapshot_every", "1")
    torch.manual_seed(0)
    full = train_rl_joint.main(_lux_rl_args(data, init, tmp_path / "full", *base))
    torch.manual_seed(0)
    res = train_rl_joint.main(_lux_rl_args(data, init, tmp_path / "res", *base, "--resume",
                                           str(tmp_path / "full" / "rl_iter1.pt")))
    played = full["trainer"].rl["lux"].env.matches_played      # a wave plays whole games
    assert (played == 5).all() and full["trainer"].rl["lux"].env.game_over.all()
    assert torch.equal(res["trainer"].rl["lux"].env.matches_played, played)
    a = torch.load(tmp_path / "full" / "rl_iter2.pt", weights_only=False)
    b = torch.load(tmp_path / "res" / "rl_iter2.pt", weights_only=False)
    for k, v in a["train_model_state"].items():
        assert torch.equal(v, b["train_model_state"][k]), k
    ea, eb = a["rl_state"]["envs"][0]["lux"], b["rl_state"]["envs"][0]["lux"]
    assert ea["num_envs"] == eb["num_envs"] == 2
    for x, y in zip(ea["state"]["games"]["state"], eb["state"]["games"]["state"]):
        assert np.array_equal(x, y)
