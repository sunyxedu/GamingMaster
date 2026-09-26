"""Information audit of the Lux AI Season 3 pipeline: every network input must
be information the acting team really has at that decision point.

Inputs audited: ``gmdata lux convert`` (data_processing/rs/gmdata/src/lux.rs,
the IL shards), ``env/vec_lux.py`` (the RL rows: obs + the decoder's legal
masks) and ``model/games/lux.py`` + GameNetwork (legal masks derived from the
obs, the sap decode context, Stage-1 temporal state).  Targets (tgt.value,
actions) may use hindsight and are not inputs.

What the Lux S3 agent really has (luxai_s3 0.2.1, the rules the Kaggle
competition ran):

  * its own observation ``LuxAIS3Env.get_obs(state)["player_<team>"]``: own
    units always, enemy units / tile types / tile energies / relic nodes only
    where its current ``sensor_mask`` is set (everything else is -1 / masked);
    team points, team wins, steps, match steps of BOTH teams (public);
  * ``configuration.env_cfg`` = the parameters ``LuxAIS3GymEnv.reset`` keeps
    for the agent (:data:`AGENT_ENV_CFG`): map size, match counts, max units,
    unit move cost, sap cost, sap range, sensor range.  Hidden: nebula vision /
    energy reduction, sap dropoff, energy void factor, drift speeds, energy
    nodes, relic point tiles (``relic_nodes_map_weights``), the seed.

Two leak classes, each tested mathematically:

  (1) FUTURE (causality): the converter's frames 0..t of a replay truncated
      right after decision t (and scrubbed of end-of-game facts) are
      bit-identical to the full conversion's; the network's outputs at
      frames <= t ignore every later frame.
  (2) PRIVACY: agent p's converted frames are invariant to everything in the
      replay but p's own observation stream and the agent-visible env_cfg
      (the opponent's observations and actions, the full-state ``info.replay``,
      hidden params, results, the seed), and to the values hidden under p's
      own masks; the env's rows (obs AND legal masks) for seat p are
      invariant to every part of the official state p cannot see.
      The recorded observations themselves are fog-of-war views
      (data invariants on the fixture and, if present, raw replays).

Data: tests/fixtures/lux_kaggle/episode-71896508.json.gz (a real Kaggle
episode, always); with raw replays present (``LUX_RAW_DIR``, default
~/gm-data/v3/raw/lux) ``LUX_AUDIT_N`` (default 2) of them as well.  Converter
tests need cargo, simulator tests the ``lux`` extra; both skip cleanly.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import subprocess
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from data_processing.games.dataset import load_gm_samples
from env.vec_lux import ENV_GLOBALS, load_kaggle_replay
from model.game_network import GameNetwork
from model.games.lux import LuxConfig, LuxDecoder, LuxEncoder, lux_legal_from_obs
from tests.test_rust_converters_parity import gmdata  # noqa: F401  (fixture)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE_FIXTURE = os.path.join(ROOT, "tests", "fixtures", "lux_kaggle", "episode-71896508.json.gz")
RAW_DIR = os.environ.get("LUX_RAW_DIR", os.path.expanduser("~/gm-data/v3/raw/lux"))
OBS_KEYS = ("map_seq", "units_seq", "unit_mask", "globals_seq")
ACT_KEYS = ("unit_actions", "sap_offset", "unit_mask")
H = W = 24
U = 16

#: the env parameters the competition hands to the agent: luxai_s3.wrappers
#: LuxAIS3GymEnv.reset keeps exactly these in info["params"] (Kaggle:
#: configuration.env_cfg); test_env_cfg_given_to_agents_is_pinned re-reads them
AGENT_ENV_CFG = frozenset({
    "max_units", "match_count_per_episode", "max_steps_in_match", "map_height", "map_width",
    "num_teams", "unit_move_cost", "unit_sap_cost", "unit_sap_range", "unit_sensor_range"})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _raw_files():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "episode-*-replay.json")))
    n = int(os.environ.get("LUX_AUDIT_N", "2"))
    if not files or n <= 0:
        return []
    return files[:: max(1, len(files) // n)][:n]


RAW_PARAMS = _raw_files() or [None]


def _raw_id(p):
    return os.path.basename(p) if p else "absent"


def _load_raw(path):
    if path is None:
        pytest.skip(f"no raw Lux replays under {RAW_DIR} (set LUX_RAW_DIR)")
    return load_kaggle_replay(path)


@pytest.fixture(scope="module")
def kaggle_ep():
    return load_kaggle_replay(KAGGLE_FIXTURE)


def _obs(entry) -> dict:
    raw = entry["observation"]["obs"]
    return json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)


def _with_obs(entry, o) -> dict:
    """A copy of a step entry with its observation's obs replaced (same encoding)."""
    e = dict(entry)
    e["observation"] = dict(entry["observation"])
    e["observation"]["obs"] = json.dumps(o) if isinstance(entry["observation"]["obs"], str) else o
    return e


def _team(ep, agent: int) -> int:
    return int(ep["steps"][0][agent]["observation"].get("player", f"player_{agent}")[-1])


def _convert(gmdata_bin: str, eps: Dict[str, dict], tmp_path) -> Dict[Tuple[str, int, int], dict]:
    """Convert every episode of ``eps`` ({name: replay}) in ONE gmdata run →
    {(name, agent, match_index): sample}."""
    d = tmp_path / "conv"
    (d / "in").mkdir(parents=True)
    for name, ep in eps.items():
        with open(d / "in" / f"{name}.json", "w") as fh:
            json.dump(ep, fh)
    subprocess.run([gmdata_bin, "lux", "convert", "--input", str(d / "in"),
                    "--out", str(d / "out")], check=True, capture_output=True)
    out = {}
    for s in load_gm_samples(str(d / "out")):
        m = s["meta"]
        key = (m["source"][: -len(".json")], m["agent"], m["match_index"])
        assert key not in out, key
        out[key] = s
    return out


def _of(samples, name):
    return {k[1:]: v for k, v in samples.items() if k[0] == name}


def _same_obs(a: dict, b: dict) -> bool:
    return all(a["obs"][k].dtype == b["obs"][k].dtype and torch.equal(a["obs"][k], b["obs"][k])
               for k in OBS_KEYS)


def _assert_agent_identical(base: dict, got: dict, agent: int, what: str) -> None:
    mine = {k: v for k, v in base.items() if k[0] == agent}
    theirs = {k: v for k, v in got.items() if k[0] == agent}
    assert mine.keys() == theirs.keys() and mine, (what, sorted(mine), sorted(theirs))
    for key in mine:
        a, b = mine[key], theirs[key]
        assert a["meta"]["step_range"] == b["meta"]["step_range"], (what, key)
        for k in OBS_KEYS:
            if not torch.equal(a["obs"][k], b["obs"][k]):
                t = next(t for t in range(a["obs"][k].shape[0])
                         if not torch.equal(a["obs"][k][t], b["obs"][k][t]))
                raise AssertionError(f"{what}: agent {agent} match {key[1]}: obs.{k} changed "
                                     f"at frame {t}")
        for k in ACT_KEYS:
            assert torch.equal(a["tgt"][k], b["tgt"][k]), (what, key, k)


# ---------------------------------------------------------------------------
# (0) the recorded observations are each agent's own fog-of-war view
# ---------------------------------------------------------------------------

def _check_fog_invariants(ep, what: str) -> int:
    ec = ep["configuration"].get("env_cfg") or {}
    assert set(ec) <= AGENT_ENV_CFG, (what, "configuration.env_cfg holds hidden params",
                                       sorted(set(ec) - AGENT_ENV_CFG))
    frames = 0
    for t, st in enumerate(ep["steps"]):
        for a in range(2):
            if st[a].get("observation", {}).get("obs") is None:
                continue
            o, team = _obs(st[a]), _team(ep, a)
            opp = 1 - team
            vis = np.asarray(o["sensor_mask"], bool)                        # [x][y]
            tt = np.asarray(o["map_features"]["tile_type"])
            en = np.asarray(o["map_features"]["energy"])
            where = f"{what} step {t} agent {a}"
            # tile types are masked exactly outside vision (the converter's
            # tile planes rely on it: they are not gated by the sensor mask)
            assert np.array_equal(tt == -1, ~vis), where
            assert (en[~vis] == -1).all(), where
            um = np.asarray(o["units_mask"], bool)
            pos = np.asarray(o["units"]["position"])
            ue = np.asarray(o["units"]["energy"])
            for k in range(um.shape[1]):
                if um[opp, k]:
                    x, y = pos[opp, k]
                    assert vis[x, y], (where, "enemy unit outside vision", k)
                else:
                    assert (pos[opp, k] == -1).all() and ue[opp, k] == -1, (where, k)
                if not um[team, k]:
                    assert (pos[team, k] == -1).all() and ue[team, k] == -1, (where, k)
            for r, m in zip(np.asarray(o["relic_nodes"]), np.asarray(o["relic_nodes_mask"], bool)):
                if m:
                    assert vis[r[0], r[1]], (where, "relic node outside vision")
                else:
                    assert (r == -1).all(), where
            frames += 1
    return frames


def test_fixture_obs_are_fog_of_war_views(kaggle_ep):
    assert _check_fog_invariants(kaggle_ep, "fixture") == 2 * len(kaggle_ep["steps"])


@pytest.mark.parametrize("path", RAW_PARAMS, ids=_raw_id)
def test_raw_replay_obs_are_fog_of_war_views(path):
    ep = _load_raw(path)
    assert _check_fog_invariants(ep, os.path.basename(path)) > 0


# ---------------------------------------------------------------------------
# (1) FUTURE: the converter is causal
# ---------------------------------------------------------------------------

CUTS = (0, 1, 57, 99, 100, 101, 102, 250, 303, 404, 503, 504)


def _truncate_after_decision(ep, t: int) -> dict:
    """The record as it stands right after decision t: observations 0..t and
    the actions answering them (steps[1..t+1]) — step t+1 keeps ONLY its
    actions, not the observation they produced — and none of the end-of-game
    facts (rewards, statuses, the Kaggle top-level info)."""
    cut = {k: v for k, v in ep.items() if k not in ("rewards", "statuses", "info")}
    last = [{"action": e.get("action"),
             "observation": {k: v for k, v in e["observation"].items() if k == "player"}}
            for e in ep["steps"][t + 1]]
    cut["steps"] = ep["steps"][: t + 1] + [last]
    return cut


def test_converter_frames_are_causal(gmdata, tmp_path, kaggle_ep):  # noqa: F811
    """Truncate the replay right after decision t (:func:`_truncate_after_decision`):
    the conversion then has exactly frames 0..t per agent, and each is
    bit-identical — obs and action targets — to the full conversion's.  Cuts
    straddle the match boundaries (frame 100 / 101 of the 5 matches).  (Also
    pins the obs t ↔ steps[t+1] action alignment: a frame built from a later
    observation would lose its last frame here.)"""
    ep = kaggle_ep
    eps = {"full": ep}
    for t in CUTS:
        eps[f"cut{t:03d}"] = _truncate_after_decision(ep, t)
    got = _convert(gmdata, eps, tmp_path)
    full = _of(got, "full")
    assert sum(s["obs"]["map_seq"].shape[0] for s in full.values()) == 2 * (len(ep["steps"]) - 1)
    for t in CUTS:
        cut = _of(got, f"cut{t:03d}")
        assert sum(s["obs"]["map_seq"].shape[0] for s in cut.values()) == 2 * (t + 1), t
        for key, s in cut.items():
            f = full[key]
            L = s["obs"]["map_seq"].shape[0]
            assert s["meta"]["step_range"] == [f["meta"]["step_range"][0],
                                               f["meta"]["step_range"][0] + L], (t, key)
            for k in OBS_KEYS:
                assert torch.equal(s["obs"][k], f["obs"][k][:L]), (t, key, k)
            for k in ACT_KEYS:
                assert torch.equal(s["tgt"][k], f["tgt"][k][:L]), (t, key, k)


# ---------------------------------------------------------------------------
# (2) PRIVACY: the converter reads only the agent's own observation stream
# ---------------------------------------------------------------------------

def _garbage_info(rng) -> dict:
    """A per-step info in the Kaggle replay layout whose content is all
    different (full-state observations, the FULL env params incl. hidden)."""
    return {"replay": {"metadata": {"seed": int(rng.integers(1 << 30))},
                       "params": {"unit_move_cost": 5, "unit_sap_cost": 49, "unit_sap_range": 3,
                                  "unit_sensor_range": 1, "nebula_tile_vision_reduction": 7,
                                  "nebula_tile_energy_reduction": 25,
                                  "unit_sap_dropoff_factor": 1.0,
                                  "unit_energy_void_factor": 0.375},
                       "observations": [{"energy_nodes": [[1, 2]], "relic_nodes": [[3, 4]]}],
                       "actions": []}}


def _scramble_all_but_own(ep, p: int, seed: int) -> dict:
    """``ep`` with everything agent p's decision does not see replaced: the
    other agent's observations (another step's, so they still parse) and
    actions, every per-step info (``info.replay``: full state + hidden
    params), the per-agent rewards / statuses, the top-level results / info
    and the seed.  Kept: p's ``observation`` and ``action`` entries, the
    agent-visible ``configuration.env_cfg``."""
    rng = np.random.default_rng(seed)
    q = 1 - p
    T = len(ep["steps"])
    steps = []
    for t, st in enumerate(ep["steps"]):
        own = {k: v for k, v in st[p].items() if k in ("observation", "action")}
        own.update(info=_garbage_info(rng), reward=-7, status="ERROR")
        other = {k: v for k, v in st[q].items() if k != "info"}
        src = ep["steps"][(t + 37) % T][q]
        other["observation"] = dict(src["observation"])
        other["action"] = rng.integers(-3, 8, size=(U, 3)).tolist() if t else None
        other.update(info=_garbage_info(rng), reward=9, status="DONE")
        pair = [None, None]
        pair[p], pair[q] = own, other
        steps.append(pair)
    out = {k: v for k, v in ep.items() if k not in ("steps", "rewards", "statuses", "info")}
    out["configuration"] = {**ep["configuration"], "seed": int(rng.integers(1 << 30)),
                            "env_cfg": ep["configuration"].get("env_cfg")}
    out["rewards"] = [-1, 99]
    out["statuses"] = ["ERROR", "INVALID"]
    out["info"] = {"TeamNames": ["x", "y"]}
    out["steps"] = steps
    return out


def _privacy_other_stream(gmdata_bin, tmp_path, ep, what):
    eps = {"base": ep}
    for p in (0, 1):
        eps[f"hide{p}"] = _scramble_all_but_own(ep, p, seed=11 + p)
    got = _convert(gmdata_bin, eps, tmp_path)
    base = _of(got, "base")
    for p in (0, 1):
        pert = _of(got, f"hide{p}")
        _assert_agent_identical(base, pert, p, f"{what}: all but agent {p}'s stream scrambled")
        # non-vacuous: the other agent's frames did change
        q = 1 - p
        assert any(not _same_obs(base[k], pert[k]) for k in base if k[0] == q and k in pert) \
            or {k for k in base if k[0] == q} != {k for k in pert if k[0] == q}, (what, p)


def test_converter_reads_only_own_stream_fixture(gmdata, tmp_path, kaggle_ep):  # noqa: F811
    _privacy_other_stream(gmdata, tmp_path, kaggle_ep, "fixture")


@pytest.mark.parametrize("path", RAW_PARAMS[:1], ids=_raw_id)
def test_converter_reads_only_own_stream_raw(gmdata, tmp_path, path):  # noqa: F811
    """The raw Kaggle replay carries the full game state for the viewer in
    agent 0's per-step ``info.replay`` (energy nodes, relic configs, every
    unit, the hidden params) — the converter must not read any of it."""
    ep = _load_raw(path)
    assert "replay" in (ep["steps"][1][0].get("info") or {})
    _privacy_other_stream(gmdata, tmp_path, ep, os.path.basename(path))


def _perturb_masked(o: dict, team: int, rng) -> dict:
    """Agent obs with every value its own masks hide rewritten: enemy / own
    slots with ``units_mask`` False get an on-map position and an energy,
    tile energies outside ``sensor_mask`` random values, relic nodes with
    ``relic_nodes_mask`` False on-map coordinates.  (Tile types outside
    vision stay -1: the official obs format, pinned by the fog invariants.)"""
    o = copy.deepcopy(o)
    um = o["units_mask"]
    for side in (0, 1):
        for k in range(len(um[side])):
            if not um[side][k]:
                o["units"]["position"][side][k] = [int(rng.integers(W)), int(rng.integers(H))]
                o["units"]["energy"][side][k] = int(rng.integers(0, 401))
    vis = o["sensor_mask"]
    en = o["map_features"]["energy"]
    for x in range(W):
        for y in range(H):
            if not vis[x][y]:
                en[x][y] = int(rng.integers(-20, 21))
    for i, m in enumerate(o["relic_nodes_mask"]):
        if not m:
            o["relic_nodes"][i] = [int(rng.integers(W)), int(rng.integers(H))]
    return o


@pytest.mark.parametrize("p", [0, 1])
def test_converter_ignores_values_under_the_obs_masks(gmdata, tmp_path, kaggle_ep, p):  # noqa: F811
    """Agent p's frames are invariant to every value its own observation marks
    as unseen (masked unit slots, tile energies outside vision, undiscovered
    relic nodes).  Positive control: unmasking one hidden enemy slot changes
    that frame — the perturbed fields are read when they are visible."""
    ep, team = kaggle_ep, _team(kaggle_ep, p)
    rng = np.random.default_rng(100 + p)
    pert = dict(ep)
    pert["steps"] = [[_with_obs(st[a], _perturb_masked(_obs(st[a]), team, rng)) if a == p else st[a]
                      for a in range(2)] for st in ep["steps"]]
    ctrl = dict(pert)
    t_ctrl = 150                                            # match 1, frame 49
    o = _obs(pert["steps"][t_ctrl][p])
    k = next(k for k in range(U) if not o["units_mask"][1 - team][k])
    o["units_mask"][1 - team][k] = True
    ctrl["steps"] = list(pert["steps"])
    ctrl["steps"][t_ctrl] = [_with_obs(e, o) if a == p else e
                             for a, e in enumerate(pert["steps"][t_ctrl])]
    got = _convert(gmdata, {"base": ep, "masked": pert, "ctrl": ctrl}, tmp_path)
    base = _of(got, "base")
    _assert_agent_identical(base, _of(got, "masked"), p, "masked values rewritten")
    c = _of(got, "ctrl")[(p, 1)]["obs"]["map_seq"]
    assert not torch.equal(c[t_ctrl - 101], base[(p, 1)]["obs"]["map_seq"][t_ctrl - 101])


def test_converter_env_params_are_agent_visible(gmdata, tmp_path, kaggle_ep):  # noqa: F811
    """The env-parameter globals come from keys the agent is given
    (configuration.env_cfg) and nowhere else: a hidden param in a per-step
    info never reaches them, and every recorded key is agent-visible."""
    got = _convert(gmdata, {"base": kaggle_ep}, tmp_path)
    for s in got.values():
        keys = set(s["meta"]["env_cfg"]) - {"known"}
        assert keys and keys <= AGENT_ENV_CFG, sorted(keys - AGENT_ENV_CFG)
        assert s["meta"]["env_cfg"]["known"] is True
        ec = kaggle_ep["configuration"]["env_cfg"]
        assert all(s["meta"]["env_cfg"][k] == ec[k] for k in keys)
    assert {k for k, _ in ENV_GLOBALS} <= AGENT_ENV_CFG


# ---------------------------------------------------------------------------
# simulator (env/vec_lux.py): rows for seat p ignore the state p cannot see
# ---------------------------------------------------------------------------

CHECKPOINTS = (60, 150, 260, 400)
ROW_KEYS_OBS = OBS_KEYS
ROW_KEYS_LEGAL = ("unit", "unit_action", "sap")


@pytest.fixture(scope="module")
def sim(kaggle_ep):
    """The Kaggle fixture's game in LuxVecEnv(1) (recorded actions), with the
    env state_dict at each CHECKPOINTS step."""
    pytest.importorskip("luxai_s3")
    from env.vec_lux import LuxVecEnv, kaggle_env_actions, kaggle_seed

    env = LuxVecEnv(1, seed=0)
    env.reset(seed=kaggle_seed(kaggle_ep), new_game=True)
    acts = kaggle_env_actions(kaggle_ep)
    snaps = {}
    for t in range(max(CHECKPOINTS) + 1):
        if t in CHECKPOINTS:
            snaps[t] = copy.deepcopy(env.state_dict())
        _, _, done, info = env.step({}, env_actions=acts[t])
        if bool(done[0]) and not bool(info["game_over"][0]):
            env.reset()
    probe = LuxVecEnv(1, seed=0)                  # shares the compiled kernels
    return {"env": env, "probe": probe, "snaps": snaps}


def _rows(probe, sd):
    probe.load_state_dict(sd)
    o = probe.observe()
    return ({k: v.clone() for k, v in o.obs.items()}, {k: v.clone() for k, v in o.legal.items()})


def _state_of(probe, sd):
    from env.vec_lux import _lux
    jax, jnp = _lux().jax, _lux().jnp
    treedef = jax.tree_util.tree_structure(probe._state)
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in sd["games"]["state"]])


def _with_state(sd, state):
    from env.vec_lux import _lux
    jax = _lux().jax
    out = copy.deepcopy(sd)
    out["games"]["state"] = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(state)]
    return out


def _hide_perturb(state, seat: int, rng):
    """A copy of the (N = 1) official EnvState that differs from ``state``
    only in what team ``seat`` cannot see: enemy units outside its sensor
    mask moved / re-energised / spawned / removed, tile types and energies
    outside vision, relic nodes it does not see (moved, (un)spawned), relic
    configs + point tiles, energy nodes, the enemy's sensor mask and both
    vision power maps.  Kept: its own units, visible enemy units, visible
    tiles and relics, points, wins, steps, its sensor mask."""
    from env.vec_lux import _lux
    jnp = _lux().jnp
    q = 1 - seat
    A = lambda x: np.asarray(x).copy()  # noqa: E731
    vis = A(state.sensor_mask)[0, seat].astype(bool)                     # [x][y]
    inv = np.argwhere(~vis)
    assert len(inv) > 0, "the seat sees the whole map"

    def cell():
        return inv[rng.integers(len(inv))]

    def seen(x, y):
        return 0 <= x < W and 0 <= y < H and bool(vis[x, y])

    pos, en, um = A(state.units.position), A(state.units.energy), A(state.units_mask)
    n_hidden = 0
    for k in range(U):
        x, y = pos[0, q, k]
        if um[0, q, k] and seen(x, y):
            continue
        n_hidden += int(um[0, q, k])
        pos[0, q, k] = cell()
        en[0, q, k, 0] = rng.integers(0, 401)
        um[0, q, k] = rng.random() < 0.5
    tt, me = A(state.map_features.tile_type), A(state.map_features.energy)
    tt[0][~vis] = rng.integers(0, 3, size=int((~vis).sum()))
    me[0][~vis] = rng.integers(-20, 21, size=int((~vis).sum()))
    rn, rm = A(state.relic_nodes), A(state.relic_nodes_mask)
    for i in range(rn.shape[1]):
        if rm[0, i] and seen(*rn[0, i]):
            continue
        rn[0, i] = cell()
        rm[0, i] = rng.random() < 0.5
    sm = A(state.sensor_mask)
    sm[0, q] = rng.random(sm[0, q].shape) < 0.5
    vp = A(state.vision_power_map)

    def rand_like(x):
        x = A(x)
        if x.dtype == bool:
            return rng.random(x.shape) < 0.5
        if np.issubdtype(x.dtype, np.integer):
            return rng.integers(0, 7, size=x.shape).astype(x.dtype)
        return rng.standard_normal(x.shape).astype(x.dtype)

    def cast(new, old):
        return jnp.asarray(np.asarray(new).astype(np.asarray(old).dtype))

    new = state.replace(
        units=state.units.replace(position=cast(pos, state.units.position),
                                  energy=cast(en, state.units.energy)),
        units_mask=cast(um, state.units_mask),
        map_features=state.map_features.replace(tile_type=cast(tt, state.map_features.tile_type),
                                                energy=cast(me, state.map_features.energy)),
        relic_nodes=cast(rn, state.relic_nodes),
        relic_nodes_mask=cast(rm, state.relic_nodes_mask),
        relic_node_configs=cast(rand_like(state.relic_node_configs), state.relic_node_configs),
        relic_nodes_map_weights=cast(rand_like(state.relic_nodes_map_weights),
                                     state.relic_nodes_map_weights),
        relic_spawn_schedule=cast(rand_like(state.relic_spawn_schedule),
                                  state.relic_spawn_schedule),
        energy_nodes=cast(rand_like(state.energy_nodes), state.energy_nodes),
        energy_node_fns=cast(rand_like(state.energy_node_fns), state.energy_node_fns),
        energy_nodes_mask=cast(rand_like(state.energy_nodes_mask), state.energy_nodes_mask),
        sensor_mask=cast(sm, state.sensor_mask),
        vision_power_map=cast(rand_like(vp), state.vision_power_map),
    )
    return new, n_hidden


def _assert_row(a, b, row: int, what: str) -> None:
    (oa, la), (ob, lb) = a, b
    for k in ROW_KEYS_OBS:
        assert torch.equal(oa[k][row], ob[k][row]), (what, "obs", k)
    for k in ROW_KEYS_LEGAL:
        if k in la:
            assert torch.equal(la[k][row], lb[k][row]), (what, "legal", k)


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize("t", CHECKPOINTS)
def test_env_rows_invariant_to_state_hidden_from_the_seat(sim, t, seat):
    """Seat p's row — the obs AND the legal masks the policy samples under —
    is bit-identical after rewriting every part of the official game state p
    cannot see (several random draws); the other seat's row changes (the
    perturbation is not vacuous)."""
    probe, sd = sim["probe"], sim["snaps"][t]
    base = _rows(probe, sd)
    state = _state_of(probe, sd)
    rng = np.random.default_rng(1000 * t + seat)
    for draw in range(3):
        pert_state, _ = _hide_perturb(state, seat, rng)
        pert = _rows(probe, _with_state(sd, pert_state))
        _assert_row(base, pert, seat, f"step {t} seat {seat} draw {draw}")
        assert not torch.equal(base[0]["map_seq"][1 - seat], pert[0]["map_seq"][1 - seat])


def test_env_checkpoints_have_hidden_enemy_units(sim):
    """The invariance test above is not vacuous for units: at the checkpoints
    there are live enemy units outside the seat's vision (it rewrites them;
    at other checkpoints a seat sees every enemy unit and only ghost spawns
    are exercised)."""
    probe = sim["probe"]
    n = 0
    for t, sd in sim["snaps"].items():
        state = _state_of(probe, sd)
        for seat in (0, 1):
            n += _hide_perturb(state, seat, np.random.default_rng(0))[1]
    assert n >= 10, n


@pytest.mark.parametrize("t", CHECKPOINTS[1:3])
def test_env_rows_see_visible_changes(sim, t):
    """Positive controls for the invariance test: a visible tile's energy and
    a visible enemy unit's energy DO reach the seat's row."""
    from env.vec_lux import _lux
    jnp = _lux().jnp
    probe, sd = sim["probe"], sim["snaps"][t]
    base = _rows(probe, sd)
    state = _state_of(probe, sd)
    for seat in (0, 1):
        vis = np.asarray(state.sensor_mask)[0, seat].astype(bool)
        x, y = np.argwhere(vis)[0]
        me = np.asarray(state.map_features.energy).copy()
        me[0, x, y] = (int(me[0, x, y]) + 7) % 20
        s2 = state.replace(map_features=state.map_features.replace(
            energy=jnp.asarray(me.astype(np.asarray(state.map_features.energy).dtype))))
        assert not torch.equal(base[0]["map_seq"][seat],
                               _rows(probe, _with_state(sd, s2))[0]["map_seq"][seat])
        pos = np.asarray(state.units.position)
        um = np.asarray(state.units_mask)
        seen = [k for k in range(U) if um[0, 1 - seat, k] and vis[tuple(pos[0, 1 - seat, k])]]
        if seen:
            en = np.asarray(state.units.energy).copy()
            # (the energy planes clamp a negative energy to 0: go positive)
            en[0, 1 - seat, seen[0], 0] = max(int(en[0, 1 - seat, seen[0], 0]), 0) + 57
            s3 = state.replace(units=state.units.replace(
                energy=jnp.asarray(en.astype(np.asarray(state.units.energy).dtype))))
            assert not torch.equal(base[0]["map_seq"][seat],
                                   _rows(probe, _with_state(sd, s3))[0]["map_seq"][seat])


def test_env_rows_invariant_to_hidden_env_params(sim):
    """The env params the agent is NOT given (nebula vision / energy
    reduction, sap dropoff, void factor, drift speeds) never reach a row;
    changing a given one (unit_sap_range) does."""
    from env.vec_lux import _lux
    ranges = _lux().ranges
    probe = sim["probe"]
    for t, sd in sim["snaps"].items():
        base = _rows(probe, sd)
        pert = copy.deepcopy(sd)
        hidden = sorted(set(ranges) - AGENT_ENV_CFG)
        assert hidden
        for k in hidden:
            cur = pert["games"]["params"][k]
            alt = next(v for v in ranges[k] if not np.isclose(v, float(cur[0])))
            pert["games"]["params"][k] = np.asarray([alt], dtype=cur.dtype)
        got = _rows(probe, pert)
        for seat in (0, 1):
            _assert_row(base, got, seat, f"step {t} hidden params seat {seat}")
        vis_ctrl = copy.deepcopy(sd)
        cur = vis_ctrl["games"]["params"]["unit_sap_range"]
        vis_ctrl["games"]["params"]["unit_sap_range"] = np.asarray([3 if cur[0] != 3 else 7],
                                                                   dtype=cur.dtype)
        assert not torch.equal(base[0]["globals_seq"], _rows(probe, vis_ctrl)[0]["globals_seq"])


def test_env_cfg_given_to_agents_is_pinned():
    """AGENT_ENV_CFG is exactly what the official wrapper hands the agent, and
    the obs / legal builders run on those keys alone (a hidden param read by
    either would raise KeyError here)."""
    pytest.importorskip("luxai_s3")
    from luxai_s3.wrappers import LuxAIS3GymEnv

    from env.vec_lux import build_legal, build_obs, div_tables, env_globals_of, json_obs_arrays

    _, info = LuxAIS3GymEnv(numpy_output=True).reset(seed=0)
    kept = info["params"]
    assert set(kept) == AGENT_ENV_CFG
    ep = load_kaggle_replay(KAGGLE_FIXTURE)
    frames = [_obs(ep["steps"][t][0]) for t in (0, 150, 300)]
    a = json_obs_arrays(frames)
    eg = env_globals_of(div_tables(), dict(kept), batch=len(frames))
    build_obs(div_tables(), a, 0, eg)
    build_legal(a, 0, dict(kept), 7, True)


# ---------------------------------------------------------------------------
# model: the network's frame-t outputs ignore later frames
# ---------------------------------------------------------------------------

def _lux_net(mixer: str, core: str, head: str) -> GameNetwork:
    cfg = LuxConfig(
        d_model=32, n_heads=4, d_ff=64, dropout=0.0, n_st_layers=2, temporal_mixer=mixer,
        core_type=core, trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1, halt_max_steps=2,
        rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1, rd_mean_iters=2,
        rd_backprop_iters=1, core_layers=1, encoder_channels=16,
        map_h=8, map_w=8, map_channels=12, max_units=6, unit_feature_dim=16,
        lux_global_features=11, lux_sap_range=7, lux_sap_head=head)
    return GameNetwork(cfg, LuxEncoder(cfg), LuxDecoder(cfg)).eval()


def _frames(cfg, B: int, T: int, g: torch.Generator) -> dict:
    """Random converter-v2-shaped frames on cfg's map (units on integer cells,
    env globals known, sap range 3..7)."""
    um = torch.rand(B, T, cfg.max_units, generator=g) < 0.7
    units = torch.zeros(B, T, cfg.max_units, cfg.unit_feature_dim)
    units[..., 0] = torch.randint(0, cfg.map_w, (B, T, cfg.max_units), generator=g) / cfg.map_w
    units[..., 1] = torch.randint(0, cfg.map_h, (B, T, cfg.max_units), generator=g) / cfg.map_h
    units[..., 2] = torch.rand(B, T, cfg.max_units, generator=g)
    units[..., 3] = 1.0
    units = units * um.unsqueeze(-1)
    glob_ = torch.rand(B, T, cfg.lux_global_features, generator=g)
    glob_[..., 8] = torch.randint(3, 8, (B, T), generator=g) / 8.0
    glob_[..., 10] = 1.0
    return {"map_seq": torch.rand(B, T, cfg.map_channels, cfg.map_h, cfg.map_w, generator=g),
            "units_seq": units, "unit_mask": um, "globals_seq": glob_}


OUT_KEYS = ("unit_action_logits", "unit_present", "sap_logits", "sap_mask", "value", "q_halt")


@pytest.mark.parametrize("head", ["pointer", "mlp"])
@pytest.mark.parametrize("mixer,core", [("attention", "trm"), ("recurrent", "recurrent_depth")])
def test_network_frame_outputs_ignore_later_frames(mixer, core, head):
    """In inference mode, every decoder input and output at frame t — the IL
    legal masks derived from the obs (units + the episode's sap range), the
    sap decode context, the unit / sap logits, the value — is unchanged when
    frames t+1.. are replaced, and equals the forward over the prefix alone
    (so neither later frames nor the sequence length reach frame t)."""
    torch.manual_seed(0)
    net = _lux_net(mixer, core, head)
    cfg = net.cfg
    g = torch.Generator().manual_seed(1)
    B, T = 2, 9
    obs = _frames(cfg, B, T, g)
    with torch.no_grad():
        legal = lux_legal_from_obs(obs)
        out = net(obs, legal=legal)
        for t in (0, 4, 7):
            obs2 = {k: v.clone() for k, v in obs.items()}
            other = _frames(cfg, B, T, g)
            for k in obs2:
                obs2[k][:, t + 1:] = other[k][:, t + 1:]
            legal2 = lux_legal_from_obs(obs2)
            for k in legal:
                assert torch.equal(legal[k][:, : t + 1], legal2[k][:, : t + 1]), (t, k)
            ctx = net.obs_encoder.decode_context(obs)["unit_cell"]
            ctx2 = net.obs_encoder.decode_context(obs2)["unit_cell"]
            assert torch.equal(ctx[:, : t + 1], ctx2[:, : t + 1])
            out2 = net(obs2, legal=legal2)
            pre = {k: v[:, : t + 1] for k, v in obs.items()}
            out3 = net(pre, legal=lux_legal_from_obs(pre))
            for k in OUT_KEYS:
                a = out[k][:, : t + 1]
                for o, name in ((out2, "future replaced"), (out3, "prefix only")):
                    b = o[k][:, : t + 1]
                    if a.dtype == torch.bool:
                        assert torch.equal(a, b), (t, k, name)
                    else:
                        assert torch.allclose(a, b, atol=1e-5, rtol=0), \
                            (t, k, name, float((a - b).abs().max()))
            assert not torch.allclose(out["unit_action_logits"][:, t + 1:],
                                      out2["unit_action_logits"][:, t + 1:])
