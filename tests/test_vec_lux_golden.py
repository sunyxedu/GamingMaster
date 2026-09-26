"""Golden tests: Lux AI Season 3 (env/vec_lux.py, over the official luxai-s3
env) against Kaggle replays and ``gmdata lux convert`` (the IL data).

A replay's game is played in :class:`LuxVecEnv` from the replay's seed (the
competition's ``LuxAIS3GymEnv.reset(seed)``: env params and map), feeding both
agents' recorded actions:

  * the official observations reproduce the replay's recorded ones exactly,
    every step, both agents (the env params, and the final match wins too) —
    the competition ran JAX with ``jax_threefry_partitionable`` off, which
    the adapter reproduces (``kaggle_rng``);
  * the adapter's rows equal the converter's shards frame by frame, bit for
    bit: every (agent, match) sample (obs, the recorded actions as IL targets,
    the match value = the env's match reward), frame t = ``match_steps`` t;
  * the obs builder on the replay's own recorded observations equals the
    shards too (the path for replays the env cannot replay: the synthetic
    converter fixture with its 2-step matches and unknown env_cfg);
  * the RL action path: the recorded actions sent in the decoder's format
    (unit actions + sap target classes) reproduce the replay as well, from a
    middle slot of a batched env.

Always: tests/fixtures/lux_kaggle/episode-71896508.json.gz — a real Kaggle
episode (sap-heavy, stacked units) stripped to what the env and the converter
read (per-step info removed except step 0's replay params) and gzipped.  With
the raw replays present (``LUX_RAW_DIR``, default ~/gm-data/v3/raw/lux):
``LUX_GOLDEN_N`` (default 2) more of them.  The shard comparisons need cargo.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import shutil
import subprocess
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

pytest.importorskip("luxai_s3")

from data_processing.games.dataset import load_gm_samples  # noqa: E402
from env.vec_lux import (  # noqa: E402
    LuxVecEnv,
    build_obs,
    div_tables,
    json_obs_arrays,
    kaggle_env_actions,
    kaggle_env_globals,
    kaggle_obs,
    kaggle_seed,
    load_kaggle_replay,
    sap_class,
)
from tests.test_rust_converters_parity import gmdata  # noqa: E402,F401  (fixture)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures")
KAGGLE_FIXTURE = os.path.join(FIX, "lux_kaggle", "episode-71896508.json.gz")
RAW_DIR = os.environ.get("LUX_RAW_DIR", os.path.expanduser("~/gm-data/v3/raw/lux"))
OBS_KEYS = ("map_seq", "units_seq", "unit_mask", "globals_seq")


def _raw_equal(ours, ref, i: int, path: str = "") -> list:
    """Paths where env ``i`` of ``LuxVecEnv.raw_obs()`` differs from a recorded JSON obs."""
    if isinstance(ref, dict):
        return [p for k in ref for p in _raw_equal(ours[k], ref[k], i, f"{path}/{k}")]
    a, b = np.asarray(ours)[i], np.asarray(ref)
    return [] if a.shape == b.shape and np.array_equal(a, b) else [path]


def replay_in_env(ep: Dict, check_raw: bool = True) -> Tuple[Dict, LuxVecEnv]:
    """Play the replay's game in LuxVecEnv(1): {(agent, match): IL-shaped
    sample} of the adapter's frames, the applied actions (IL target
    convention) and the match rewards; asserts the official observations equal
    the recorded ones at every step when ``check_raw``."""
    env = LuxVecEnv(1, seed=0)
    obs = env.reset(seed=kaggle_seed(ep), new_game=True)
    params = (ep["steps"][0][0].get("info") or {}).get("replay", {}).get("params")
    if params:
        for k, v in params.items():
            if k in env._params:
                assert np.isclose(float(env._params[k][0]), float(v)), (k, env._params[k][0], v)
    acts = kaggle_env_actions(ep)
    n_steps = len(ep["steps"])
    out: Dict[Tuple[int, int], Dict] = {}
    match = 0
    for t in range(n_steps):
        if check_raw:
            raw = env.raw_obs()
            for p in (0, 1):
                bad = _raw_equal(raw[p], kaggle_obs(ep, t, p), 0)
                assert not bad, f"step {t} agent {p}: official obs differs from the replay at {bad}"
        if t == n_steps - 1:
            break
        assert bool(obs.active.all()) and int(obs.step_idx[0]) == t - 101 * match
        assert bool(obs.new_episode[0]) == (int(obs.step_idx[0]) == 0)
        for p in (0, 1):
            s = out.setdefault((p, match), {"obs": {k: [] for k in OBS_KEYS},
                                            "tgt": {"unit_actions": [], "sap_offset": []}})
            for k in OBS_KEYS:
                s["obs"][k].append(obs.obs[k][p])
        obs, rew, done, info = env.step({}, env_actions=acts[t])
        for p in (0, 1):
            s = out[(p, match)]
            s["tgt"]["unit_actions"].append(info["applied"]["unit_actions"][p])
            s["tgt"]["sap_offset"].append(info["applied"]["sap_offsets"][p])
        if bool(done[0]):
            for p in (0, 1):
                s = out[(p, match)]
                s["obs"] = {k: torch.stack(v) for k, v in s["obs"].items()}
                s["tgt"] = {k: torch.stack(v) for k, v in s["tgt"].items()}
                s["value"] = float(rew[0, p])
            assert int(info["match_index"][0]) == match
            match += 1
            if not bool(info["game_over"][0]):
                obs = env.reset()
    assert env.game_over.all() and match == 5
    wins = np.asarray(env.raw_obs()[0]["team_wins"][0]).tolist()
    assert wins == list(ep["rewards"]), (wins, ep["rewards"])
    return out, env


def _convert(gmdata_bin: str, ep_json: Dict, tmp_path, name: str):  # noqa: F811
    d = tmp_path / name
    (d / "in").mkdir(parents=True)
    with open(d / "in" / f"{name}.json", "w") as fh:
        json.dump(ep_json, fh)
    subprocess.run([gmdata_bin, "lux", "convert", "--input", str(d / "in"),
                    "--out", str(d / "out")], check=True, capture_output=True)
    return load_gm_samples(str(d / "out"))


def _assert_matches_shards(ours: Dict, il, what: str) -> None:
    assert len(il) == len(ours) == 10, (what, len(il), len(ours))
    for s in il:
        key = (s["meta"]["agent"], s["meta"]["match_index"])
        a, b0 = s["meta"]["step_range"]
        mine = ours[key]
        assert b0 - a == mine["obs"]["map_seq"].shape[0] == 101, (what, key)
        for k in OBS_KEYS:
            x, y = mine["obs"][k], s["obs"][k]
            assert x.dtype == y.dtype and x.shape == y.shape, (what, key, k, x.shape, y.shape)
            if not torch.equal(x, y):
                t = next(t for t in range(x.shape[0]) if not torch.equal(x[t], y[t]))
                raise AssertionError(f"{what} {key}: {k} differs first at frame {t}")
        for k in ("unit_actions", "sap_offset"):
            assert torch.equal(mine["tgt"][k], s["tgt"][k]), (what, key, k)
        assert torch.equal(s["tgt"]["unit_mask"], s["obs"]["unit_mask"])
        assert (s["tgt"]["value"] == mine["value"]).all(), (what, key, mine["value"])


@pytest.fixture(scope="module")
def kaggle_fixture():
    ep = load_kaggle_replay(KAGGLE_FIXTURE)
    ours, _ = replay_in_env(ep)
    return ep, ours


def test_fixture_replays_exactly_in_the_official_env(kaggle_fixture):
    """replay_in_env asserted the raw observations every step; the adapter's
    frames are also the builder applied to the recorded observations."""
    ep, ours = kaggle_fixture
    tables = div_tables()
    for (p, m), s in ours.items():
        frames = [kaggle_obs(ep, t, p) for t in range(101 * m, 101 * m + 101)]
        ref = build_obs(tables, json_obs_arrays(frames), p, kaggle_env_globals(ep, p)[None])
        for k in OBS_KEYS:
            assert np.array_equal(s["obs"][k].numpy(), np.asarray(ref[k])), (p, m, k)
    saps = sum(int((s["tgt"]["unit_actions"] == 5).sum()) for s in ours.values())
    assert saps > 1000
    # stacks of >= 3 units with different energies: the f32 per-cell energy sums
    # are order-sensitive (lux.rs adds in slot order)
    e = torch.cat([s["obs"]["map_seq"][:, 8:10] for s in ours.values()])
    c = torch.cat([s["obs"]["map_seq"][:, 6:8] for s in ours.values()])
    assert bool(((c >= 0.75) & (e > 0)).any())


def test_decoder_actions_replay_the_fixture_in_a_batched_env():
    """The RL action path: the recorded actions in the DECODER's format
    (``unit_actions`` + sap target classes, as ``sample()`` emits them, not
    raw env actions) reproduce the replay's official observations every step
    and its result — with the game in the middle slot of a batched env whose
    other games play random legal actions (so sap dx/dy, the class layout,
    the move directions and the row ↔ (env, seat) layout are all pinned)."""
    ep = load_kaggle_replay(KAGGLE_FIXTURE)
    acts = kaggle_env_actions(ep)
    N, slot = 5, 2
    env = LuxVecEnv(N, seed=0)
    obs = env.reset(seed=kaggle_seed(ep) - slot, new_game=True)     # env k: seed + k
    g = torch.Generator().manual_seed(0)
    rows = slice(2 * slot, 2 * slot + 2)
    n_steps = len(ep["steps"])
    for t in range(n_steps):
        raw = env.raw_obs()
        for p in (0, 1):
            bad = _raw_equal(raw[p], kaggle_obs(ep, t, p), slot)
            assert not bad, f"step {t} agent {p}: differs at {bad}"
        if t == n_steps - 1:
            break
        ua = torch.from_numpy(acts[t][..., 0].astype(np.int64))
        assert ((ua >= 0) & (ua <= 5)).all()
        so = torch.from_numpy(np.where(ua.numpy() == 5, sap_class(acts[t][..., 1], acts[t][..., 2]),
                                       sap_class(0, 0)).astype(np.int64))
        a = {k: v.clone() for k, v in env.random_actions(obs, g).items()}
        a["unit_actions"][rows], a["sap_offsets"][rows] = ua, so
        obs, rew, done, info = env.step(a)
        if bool(done[slot]) and not bool(info["game_over"][slot]):
            obs = env.reset()
    assert np.asarray(env.raw_obs()[0]["team_wins"][slot]).tolist() == list(ep["rewards"])


def test_fixture_matches_gmdata_shards(kaggle_fixture, gmdata, tmp_path):  # noqa: F811
    ep, ours = kaggle_fixture
    _assert_matches_shards(ours, _convert(gmdata, ep, tmp_path, "fixture"), "fixture")


def test_builder_on_recorded_obs_matches_gmdata(gmdata, tmp_path):  # noqa: F811
    """The fallback path on the synthetic converter fixture (2-step matches,
    env_cfg without the unit keys → default env globals, known = 0): the obs
    builder on each sample's recorded frames equals the shard."""
    d = tmp_path / "synthetic"
    subprocess.run([gmdata, "lux", "convert", "--input", os.path.join(FIX, "lux"),
                    "--out", str(d)], check=True, capture_output=True)
    il = load_gm_samples(str(d))
    assert len(il) == 8
    tables = div_tables()
    by_source = {}
    for s in il:
        src = s["meta"]["source"]
        if src not in by_source:
            with open(os.path.join(FIX, "lux", src)) as fh:
                by_source[src] = json.load(fh)
        ep = by_source[src]
        agent, team = s["meta"]["agent"], s["meta"]["team"]
        a, b = s["meta"]["step_range"]
        frames = [kaggle_obs(ep, t, agent) for t in range(a, b)]
        ref = build_obs(tables, json_obs_arrays(frames), team,
                        np.repeat(kaggle_env_globals(ep, agent)[None], b - a, 0))
        for k in OBS_KEYS:
            assert np.array_equal(np.asarray(ref[k]), s["obs"][k].numpy()), (src, agent, k)
        assert (s["obs"]["globals_seq"][:, 10] == 0).all()        # env_cfg unknown


def _raw_files():
    files = sorted(f for f in glob.glob(os.path.join(RAW_DIR, "episode-*-replay.json")))
    n = int(os.environ.get("LUX_GOLDEN_N", "2"))
    if not files or n <= 0:
        return []
    step = max(1, len(files) // n)
    return files[::step][:n]


@pytest.mark.parametrize("path", _raw_files() or [None],
                         ids=lambda p: os.path.basename(p) if p else "absent")
def test_raw_replays_match_gmdata_shards(path, gmdata, tmp_path):  # noqa: F811
    if path is None:
        pytest.skip(f"no raw Lux replays under {RAW_DIR} (set LUX_RAW_DIR)")
    d = tmp_path / "raw"
    (d / "in").mkdir(parents=True)
    shutil.copy(path, d / "in" / os.path.basename(path))
    subprocess.run([gmdata, "lux", "convert", "--input", str(d / "in"), "--out", str(d / "out")],
                   check=True, capture_output=True)
    ep = load_kaggle_replay(path)
    ours, _ = replay_in_env(ep)
    _assert_matches_shards(ours, load_gm_samples(str(d / "out")), os.path.basename(path))


def test_fixture_is_small_and_gzipped():
    assert os.path.getsize(KAGGLE_FIXTURE) < 1_000_000
    with gzip.open(KAGGLE_FIXTURE, "rt") as fh:
        ep = json.load(fh)
    assert len(ep["steps"]) == 506 and ep["configuration"]["seed"] == kaggle_seed(ep)
