"""Lux AI Season 3 as a :class:`~env.vec_game.VecGameEnv` over the OFFICIAL
environment (``luxai-s3``, Lux-AI-Challenge/Lux-Design-S3, JAX; Apache-2.0;
``uv sync --extra lux``).  The rules are never reimplemented: ``N`` games are
one ``jax.vmap`` of ``LuxAIS3Env.reset_env`` / ``step`` (jit-compiled once per
``N``), on the CPU.

Episodes, streams, rows
-----------------------
A Lux game is 5 matches of 101 frames (``match_steps`` 0..100) on one map;
``gmdata lux convert`` (data_processing/rs/gmdata/src/lux.rs, the IL data)
writes ONE sample per (game, agent, match), starting at ``match_steps == 0``.
So here one EPISODE = one MATCH, and a row's stream is exactly one IL sample:

  * simultaneous moves: ``rows_per_env = 2``, row ``2 * env + seat`` = team
    ``seat``'s own fog-of-war view (``player_<seat>``), one frame per env step;
  * frame t of a match is the observation at ``match_steps == t``;
    ``step_idx = match_steps`` (0..100) and ``new_episode`` marks frame 0, so
    the network's temporal state restarts exactly where an IL sample starts;
  * the step that plays frame 100 ends the match: ``done`` (terminated), the
    rows go inactive.  ``reset`` then CONTINUES the game: the next match's
    frame 0 is the observation that step returned (the env's own match
    transition; the previous match's units are still shown and are removed by
    the next step, as in the IL data).  Only a finished game (5 matches), a
    match abandoned mid-way, ``new_game=True`` or a resize starts a fresh game
    (seed, map, env params).  Lockstep rollouts therefore play one match per
    wave and a fresh game every 5th wave.  (The trainer does not checkpoint
    envs: after ``--resume`` the games restart at match 1; ``state_dict``
    holds the game state for callers that want an exact continuation.)

Observation (the gm-shard ``obs`` schema of converter v2, bit-identical to
``gmdata lux convert`` on the same game: tests/test_vec_lux_golden.py):

  map_seq     float32 [12, 24, 24]  [y][x] planes: tile empty/nebula/asteroid,
                                    energy/20 (visible), sensor mask, relic,
                                    own/opp unit count/4, own/opp energy/400
                                    (per cell, f32 sums in unit-slot order),
                                    match_steps/100, ones
  units_seq   float32 [16, 16]      own unit slot k: [x/24, y/24, energy/400, 1, 0...]
  unit_mask   bool    [16]          own unit present
  globals_seq float32 [11]          points/500 x2, wins/5 x2, match_steps/101,
                                    steps/506, unit_move_cost/5, unit_sap_cost/50,
                                    unit_sap_range/8, unit_sensor_range/8, 1
                                    (env_cfg known)

The env's arrays are indexed [x][y] (as the replay JSON); the planes are [y][x].

Legal (the LuxDecoder's keys):

  unit         bool [16]      units whose action has an effect: present, energy
                              >= 0 (a negative-energy unit is removed before
                              it acts) and not a ``match_steps == 0`` frame (the
                              env removes every unit before acting on it)
  unit_action  bool [16, 6]   0 noop always; with ``action_mask`` (default) the
                              moves 1..4 (up, right, down, left) need an
                              on-map target that is not a visible asteroid
                              and energy >= unit_move_cost; sap (5) needs
                              energy >= unit_sap_cost and a legal target;
                              without it every action of a ``unit`` is legal
  sap          bool [16, O]   sap target classes o = (dy+R)(2R+1) + (dx+R),
                              R = ``sap_window`` (7, the IL schema), legal when
                              |dx|, |dy| <= unit_sap_range and on the map
                              (``model.games.lux.lux_sap_legal``'s rule)

A row with no ``unit`` (every frame 0, all units dead) is ``forced``: nothing
it sends has an effect.

Actions: the LuxDecoder's ``sample()`` output = its replay format
(``unit_actions`` [R, 16] 0..5, ``sap_offsets`` [R, 16] classes; missing
offsets sap the own cell) → env action ``[a, dx, dy]`` per unit.  Replays pass
raw env actions (``step(..., env_actions=[N, 2, 16, 3])``); ``info["applied"]``
then holds them in the IL target convention (action clamped to 0..5, offset −1
unless a sap inside the window).

Rewards per seat: +1 / −1 to the winner / loser of the match on the step that
ends it (the env always decides a winner: points, then unit energy, then a
coin flip) — the IL value target of the match; plus ±``game_reward`` (default
0) for the game result on the game's last step.  ``info``: ``winner`` (match),
``game_over``, ``game_winner``, ``team_wins``, ``match_index``.

New games draw the env parameters exactly as the competition's
``LuxAIS3GymEnv.reset(seed)`` (``luxai_s3.params.env_params_ranges``, the same
key-split sequence) and step with the same key sequence under the
competition's JAX RNG (``jax_threefry_partitionable=False``), so a game from a
Kaggle replay's seed reproduces the replay exactly when both agents' recorded
actions are fed (tests/test_vec_lux_golden.py).
"""

from __future__ import annotations

import contextlib
import dataclasses
import gzip
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

H = W = 24
U = 16                      # unit slots
F = 16                      # unit features
C = 12                      # map planes
G = 11                      # globals
N_UNIT_ACTIONS = 6
SAP_ACTION = 5
SAP_WINDOW = 7              # the IL schema's sap target window (gmdata lux convert SAP_R)
ASTEROID = 2
#: actions 1..4 (the env's directions table): up, right, down, left as (dx, dy)
MOVE_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
#: env parameters fixing array shapes (LuxAIS3Env.fixed_env_params): not overridable
STRUCTURAL_PARAMS = ("map_width", "map_height", "num_teams", "max_units", "max_energy_nodes",
                     "max_relic_nodes", "relic_config_size", "map_type")
#: (env param, normaliser) of the env-parameter globals (lux.rs ENV_KEYS)
ENV_GLOBALS = (("unit_move_cost", 5.0), ("unit_sap_cost", 50.0), ("unit_sap_range", 8.0),
               ("unit_sensor_range", 8.0))
#: lux.rs defaults of the env-parameter globals when a replay lacks them
ENV_GLOBAL_DEFAULTS = {"unit_move_cost": 2.0, "unit_sap_cost": 10.0, "unit_sap_range": 4.0,
                       "unit_sensor_range": 2.0}

#: every normalisation divisor of the obs builder (lux.rs: f64 divide, round to f32)
DIVISORS = (20, 24, 400, 500, 5, 101, 100, 506, 50, 8)
_I16 = 1 << 15

_LUX: Optional[SimpleNamespace] = None
_KERNELS: Dict[Tuple[int, bool], "_Kernels"] = {}
_TABLES: Optional[Any] = None


def _lux() -> SimpleNamespace:
    """The official env (and JAX), imported on first use."""
    global _LUX
    if _LUX is None:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")      # simulate on the host
        try:
            import jax
            import jax.numpy as jnp
            from luxai_s3.env import LuxAIS3Env
            from luxai_s3.params import EnvParams, env_params_ranges
        except ImportError as e:  # pragma: no cover - depends on the install
            raise ImportError("env.vec_lux needs the official Lux AI Season 3 environment "
                              "(luxai-s3 + JAX): run `uv sync --extra lux`") from e
        _LUX = SimpleNamespace(jax=jax, jnp=jnp, Env=LuxAIS3Env, EnvParams=EnvParams,
                               ranges=dict(env_params_ranges))
    return _LUX


def available() -> bool:
    """Whether the official Lux S3 env can be imported."""
    try:
        _lux()
    except ImportError:
        return False
    return True


@contextlib.contextmanager
def kaggle_rng():
    """JAX's RNG as the competition ran it (``jax_threefry_partitionable``
    off: its default before JAX 0.5).  Every Lux kernel runs inside it."""
    jax = _lux().jax
    cm = getattr(jax, "threefry_partitionable", None)
    if cm is not None:
        with cm(False):
            yield
        return
    prev = jax.config.jax_threefry_partitionable      # pragma: no cover - old JAX
    jax.config.update("jax_threefry_partitionable", False)
    try:
        yield
    finally:
        jax.config.update("jax_threefry_partitionable", prev)


def sap_class(dx, dy, window: int = SAP_WINDOW):
    """Env sap deltas → target class (dy+R)(2R+1) + (dx+R) (``model.games.lux.lux_sap_offset``)."""
    return (dy + window) * (2 * window + 1) + (dx + window)


def sap_delta(cls, window: int = SAP_WINDOW):
    """Target class → env sap deltas (dx, dy) (``model.games.lux.lux_sap_delta``)."""
    n = 2 * window + 1
    return cls % n - window, cls // n - window


def div_tables_np() -> np.ndarray:
    """[len(DIVISORS), 65536] float32: ``float32(float64(n) / d)`` for every
    int16 n — lux.rs's normalisation, exactly (XLA's CPU f32 division is not
    correctly rounded, so the builders never divide: they look up)."""
    n = np.arange(-_I16, _I16, dtype=np.float64)
    return (n[None, :] / np.asarray(DIVISORS, dtype=np.float64)[:, None]).astype(np.float32)


def div_tables():
    """:func:`div_tables_np` as a (cached) jax array."""
    global _TABLES
    if _TABLES is None:
        _TABLES = _lux().jnp.asarray(div_tables_np())
    return _TABLES


def exact_div(tables, n, d: int):
    """``float32(n / d)`` rounded like lux.rs (f64 division), for integer
    ``n`` (clipped to int16 — every Lux quantity fits: energies, positions,
    steps, points of one match)."""
    jnp = _lux().jnp
    idx = jnp.clip(jnp.asarray(n).astype(jnp.int32), -_I16, _I16 - 1) + _I16
    return jnp.take(tables[DIVISORS.index(d)], idx)


def param_defaults() -> Dict[str, np.ndarray]:
    """EnvParams defaults as typed numpy scalars: ints int16 (so the env's
    int16 state arithmetic keeps its dtypes, as with the competition's Python
    scalars), floats float32, flags bool."""
    lux = _lux()
    out = {}
    for f in dataclasses.fields(lux.EnvParams):
        v = f.default
        out[f.name] = (np.bool_(v) if isinstance(v, bool) else
                       np.float32(v) if isinstance(v, float) else np.int16(v))
    return out


# ---------------------------------------------------------------------------
# Observation / legal builders (pure jnp, batched over a leading dim B; one
# team's own view).  Inputs: a dict of the env's per-player observation
# arrays (EnvObs field names; map arrays [x][y]).
# ---------------------------------------------------------------------------

def env_obs_arrays(o) -> Dict[str, Any]:
    """A (batched) ``luxai_s3.state.EnvObs`` → the builders' input dict."""
    return {"tile_type": o.map_features.tile_type, "energy": o.map_features.energy,
            "sensor_mask": o.sensor_mask, "relic_nodes": o.relic_nodes,
            "relic_nodes_mask": o.relic_nodes_mask, "unit_position": o.units.position,
            "unit_energy": o.units.energy, "units_mask": o.units_mask,
            "team_points": o.team_points, "team_wins": o.team_wins, "steps": o.steps,
            "match_steps": o.match_steps}


def json_obs_arrays(frames: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Replay JSON observations (``json.loads(steps[t][p]["observation"]["obs"])``)
    → the builders' input dict, batched over the frames."""
    def arr(get, dtype):
        return np.asarray([get(f) for f in frames], dtype=dtype)
    return {
        "tile_type": arr(lambda f: f["map_features"]["tile_type"], np.int32),
        "energy": arr(lambda f: f["map_features"]["energy"], np.int32),
        "sensor_mask": arr(lambda f: f["sensor_mask"], bool),
        "relic_nodes": arr(lambda f: f["relic_nodes"], np.int32).reshape(len(frames), -1, 2),
        "relic_nodes_mask": arr(lambda f: f["relic_nodes_mask"], bool),
        "unit_position": arr(lambda f: f["units"]["position"], np.int32),
        "unit_energy": arr(lambda f: f["units"]["energy"], np.int32),
        "units_mask": arr(lambda f: f["units_mask"], bool),
        "team_points": arr(lambda f: f["team_points"], np.int32),
        "team_wins": arr(lambda f: f["team_wins"], np.int32),
        "steps": arr(lambda f: f["steps"], np.int32),
        "match_steps": arr(lambda f: f["match_steps"], np.int32),
    }


def _param(params, name: str):
    """A field of an ``EnvParams`` (or of a {name: value} mapping)."""
    return params[name] if isinstance(params, Mapping) else getattr(params, name)


def _col(x, B: int):
    """A per-env parameter (scalar or [B]) as a [B] array."""
    jnp = _lux().jnp
    x = jnp.asarray(x)
    return jnp.broadcast_to(x.reshape(-1) if x.ndim else x, (B,))


def env_globals_of(tables, params, known: float = 1.0, batch: Optional[int] = None):
    """[B, 5] float32 env-parameter globals from (batched) integer env params
    (``EnvParams`` or a mapping; scalars need ``batch``)."""
    jnp = _lux().jnp
    cols = [exact_div(tables, _param(params, k), int(norm)) for k, norm in ENV_GLOBALS]
    if batch is not None:
        cols = [_col(c, batch) for c in cols]
    return jnp.stack(cols + [jnp.full_like(cols[0], known)], axis=-1)


def build_obs(tables, a: Mapping[str, Any], team: int, env_globals) -> Dict[str, Any]:
    """One team's gm-shard frame (lux.rs ``build_frame``), batched.

    Normalisations are :func:`exact_div` lookups (lux.rs's f64 division
    rounded to f32), and the per-cell energy sums add in unit-slot order
    like lux.rs's sequential f32 ``+=``."""
    jnp = _lux().jnp
    f32 = jnp.float32
    own, opp = int(team), 1 - int(team)

    def div(n, d):
        return exact_div(tables, n, d)

    tt = jnp.swapaxes(jnp.asarray(a["tile_type"]), -1, -2)                 # [B, H, W]
    en = jnp.swapaxes(jnp.asarray(a["energy"]), -1, -2)
    vis = jnp.swapaxes(jnp.asarray(a["sensor_mask"]), -1, -2).astype(bool)
    B = tt.shape[0]
    ms = jnp.asarray(a["match_steps"])
    ys = jnp.arange(H).reshape(1, 1, H, 1)
    xs = jnp.arange(W).reshape(1, 1, 1, W)

    rn = jnp.asarray(a["relic_nodes"])
    rx, ry = rn[..., 0], rn[..., 1]                                       # [B, N]
    rv = (jnp.asarray(a["relic_nodes_mask"]).astype(bool)
          & (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H))
    relic = (rv[..., None, None] & (rx[..., None, None] == xs)
             & (ry[..., None, None] == ys)).any(1)

    pos = jnp.asarray(a["unit_position"])                                 # [B, 2, U, 2]
    ue = jnp.asarray(a["unit_energy"])                                    # [B, 2, U]
    um = jnp.asarray(a["units_mask"]).astype(bool)                        # [B, 2, U]
    zero = jnp.zeros((B, H, W), f32)

    def side(s):
        px, py = pos[:, s, :, 0], pos[:, s, :, 1]
        ok = um[:, s] & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        at = ok[..., None, None] & (px[..., None, None] == xs) & (py[..., None, None] == ys)
        e = div(jnp.maximum(ue[:, s], 0), 400)                            # [B, U]
        cnt, eng = zero, zero
        for k in range(U):                   # slot order: lux.rs's sequential f32 +=
            cnt = cnt + jnp.where(at[:, k], f32(0.25), f32(0.0))
            eng = eng + jnp.where(at[:, k], e[:, k, None, None], f32(0.0))
        return cnt, eng

    c_own, e_own = side(own)
    c_opp, e_opp = side(opp)
    planes = [(tt == 0).astype(f32), (tt == 1).astype(f32), (tt == 2).astype(f32),
              jnp.where(vis, div(en, 20), f32(0.0)), vis.astype(f32), relic.astype(f32),
              c_own, c_opp, e_own, e_opp,
              jnp.broadcast_to(div(ms, 100)[:, None, None], (B, H, W)),
              jnp.ones((B, H, W), f32)]
    map_seq = jnp.stack(planes, axis=1)

    m = um[:, own]
    feat = jnp.stack([div(pos[:, own, :, 0], W), div(pos[:, own, :, 1], H),
                      div(ue[:, own], 400), jnp.ones((B, U), f32)], axis=-1)
    feat = jnp.where(m[..., None], feat, f32(0.0))
    units_seq = jnp.concatenate([feat, jnp.zeros((B, U, F - 4), f32)], axis=-1)

    pts = jnp.asarray(a["team_points"])
    wins = jnp.asarray(a["team_wins"])
    g6 = jnp.stack([div(pts[:, own], 500), div(pts[:, opp], 500), div(wins[:, own], 5),
                    div(wins[:, opp], 5), div(ms, 101), div(jnp.asarray(a["steps"]), 506)],
                   axis=-1)
    eg = jnp.broadcast_to(jnp.asarray(env_globals, f32).reshape(-1, len(ENV_GLOBALS) + 1),
                          (B, len(ENV_GLOBALS) + 1))                  # [5], [1, 5] or [B, 5]
    globals_seq = jnp.concatenate([g6, eg], axis=-1)
    return {"map_seq": map_seq, "units_seq": units_seq, "unit_mask": m,
            "globals_seq": globals_seq}


def build_legal(a: Mapping[str, Any], team: int, params: Mapping[str, Any], window: int,
                action_mask: bool) -> Dict[str, Any]:
    """One team's legal dict (see the module docstring), batched."""
    jnp = _lux().jnp
    t = int(team)
    pos = jnp.asarray(a["unit_position"])[:, t]                           # [B, U, 2]
    px, py = pos[..., 0].astype(jnp.int32), pos[..., 1].astype(jnp.int32)
    e = jnp.asarray(a["unit_energy"])[:, t].astype(jnp.int32)
    ms = jnp.asarray(a["match_steps"])
    gate = jnp.asarray(a["units_mask"])[:, t].astype(bool) & (e >= 0) & (ms != 0)[:, None]
    B = gate.shape[0]
    move_cost = _col(_param(params, "unit_move_cost"), B).astype(jnp.int32).reshape(B, 1)
    sap_cost = _col(_param(params, "unit_sap_cost"), B).astype(jnp.int32).reshape(B, 1)
    sap_range = _col(_param(params, "unit_sap_range"), B).astype(jnp.int32).reshape(B, 1, 1)
    tiles = jnp.asarray(a["tile_type"]).reshape(B, W * H)                 # [x][y] flat
    moves = []
    for dx, dy in MOVE_DELTAS:
        nx, ny = px + dx, py + dy
        inb = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        flat = jnp.clip(nx, 0, W - 1) * H + jnp.clip(ny, 0, H - 1)
        tile = jnp.take_along_axis(tiles, flat, axis=1)
        ok = gate & inb & (tile != ASTEROID) & (e >= move_cost)
        moves.append(ok if action_mask else gate)
    out: Dict[str, Any] = {"unit": gate}
    if window > 0:
        n = 2 * window + 1
        o = jnp.arange(n * n)
        dxs, dys = o % n - window, o // n - window
        tx, ty = px[..., None] + dxs, py[..., None] + dys
        sap = (gate[..., None] & (jnp.abs(dxs) <= sap_range) & (jnp.abs(dys) <= sap_range)
               & (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H))
        out["sap"] = sap
        can_target = sap.any(-1)
    else:
        can_target = gate                                                 # the own cell
    sap_ok = gate & can_target & ((e >= sap_cost) if action_mask else True)
    out["unit_action"] = jnp.stack([jnp.ones_like(gate)] + moves + [sap_ok], axis=-1)
    return out


# ---------------------------------------------------------------------------
# Batched kernels over the official env
# ---------------------------------------------------------------------------

class _Kernels:
    """jit(vmap(...)) of the official reset / step and the row builders, for
    one (sap window, action mask).  Keys are raw uint32 [N, 2] key data."""

    def __init__(self, window: int, action_mask: bool):
        lux = _lux()
        jax, jnp = lux.jax, lux.jnp
        self.env = env = lux.Env(auto_reset=False)
        self.window, self.action_mask = window, action_mask
        names = list(lux.ranges)
        self.param_names = names
        wrap, data = jax.random.wrap_key_data, jax.random.key_data

        def sample(seed):
            # LuxAIS3GymEnv.reset(seed): key(seed) → reset key, then one split per
            # randomised parameter in env_params_ranges order
            key = jax.random.key(seed)
            key, reset_key = jax.random.split(key)
            vals = []
            for k in names:
                key, sub = jax.random.split(key)
                vals.append(jax.random.choice(sub, jnp.array(lux.ranges[k])))
            return data(key), data(reset_key), tuple(vals)

        def rows(tables, obs, params):
            out = []
            eg = env_globals_of(tables, params)
            for t in (0, 1):
                a = env_obs_arrays(obs[f"player_{t}"])
                r = build_obs(tables, a, t, eg)
                lg = build_legal(a, t, params, window, action_mask)
                r.update({f"legal_{k}": v for k, v in lg.items()})
                out.append(r)
            r = {k: jnp.stack([out[0][k], out[1][k]], axis=1) for k in out[0]}
            o0 = obs["player_0"]
            r.update(ms=o0.match_steps, steps=o0.steps, wins=o0.team_wins)
            return strong(r)

        def strong(tree):
            # strip weak types (EnvState's Python-int defaults): one jit signature
            return jax.tree_util.tree_map(lambda x: jnp.asarray(x).astype(jnp.asarray(x).dtype),
                                          tree)

        def gen(reset_kd, params):
            obs, state = env.reset_env(wrap(reset_kd), params)
            return state, obs

        def gen_rows(tables, reset_kd, params):
            state, obs = jax.vmap(gen)(reset_kd, params)
            return strong(state), rows(tables, obs, params)

        def step_one(kd, state, act, params):
            key, sk = jax.random.split(wrap(kd))
            obs, st2, _, _, _, _ = env.step(sk, state, {"player_0": act[0], "player_1": act[1]},
                                            params)
            return data(key), st2, obs

        def step_rows(tables, kd, state, act, params):
            kd2, st2, obs = jax.vmap(step_one)(kd, state, act, params)
            # keep the reset dtypes (int32 action arithmetic promotes a few leaves)
            st2 = jax.tree_util.tree_map(lambda n, o: n.astype(o.dtype), st2, state)
            return kd2, st2, rows(tables, obs, params)

        def sel(live, new, old):
            def f(n, o):
                return jnp.where(live.reshape((-1,) + (1,) * (n.ndim - 1)), n, o)
            return jax.tree_util.tree_map(f, new, old)

        def step_masked(tables, kd, state, act, params, live, old_rows):
            kd2, st2, r = step_rows(tables, kd, state, act, params)
            return sel(live, kd2, kd), sel(live, st2, state), sel(live, r, old_rows)

        def env_obs(state, params):
            return jax.vmap(lambda s, p: env.get_obs(s, p))(state, params)

        def obs_rows(tables, state, params):
            return rows(tables, env_obs(state, params), params)

        self.sample = jax.jit(jax.vmap(sample))
        self.gen_rows = jax.jit(gen_rows)
        self.step = jax.jit(step_rows)
        self.step_masked = jax.jit(step_masked)
        self.select = jax.jit(sel)
        self.obs_rows = jax.jit(obs_rows)
        self.env_obs = jax.jit(env_obs)


def _kernels(window: int, action_mask: bool) -> _Kernels:
    key = (int(window), bool(action_mask))
    if key not in _KERNELS:
        _KERNELS[key] = _Kernels(*key)
    return _KERNELS[key]


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class LuxVecEnv(VecEnvBase):
    """N Lux AI S3 games (see the module docstring).

    ``sap_window`` R: the decoder's sap target window (``lux_sap_range``; 7 in
    the IL schema; 0 = legacy decoder without a sap head: saps hit the own
    cell).  ``action_mask``: mask moves / saps that the rules would drop or
    that only burn energy.  ``game_reward``: extra ±reward for the game
    result.  ``env_params``: env parameters pinned for every new game (after
    the competition's random draw), e.g. ``{"unit_sap_range": 5}``.
    """

    game = "lux"
    n_seats = 2
    rows_per_env = 2
    interleaved = False
    ACTION_KEYS = {"unit_actions": "unit_actions", "sap_offsets": "sap_offsets"}

    def __init__(self, n_envs: int = 1, *, sap_window: int = SAP_WINDOW,
                 action_mask: bool = True, game_reward: float = 0.0,
                 env_params: Optional[Mapping[str, Any]] = None,
                 autoreset: bool = False, seed: Optional[int] = 0):
        super().__init__(n_envs, autoreset, seed)
        if sap_window < 0:
            raise ValueError(f"sap_window must be >= 0, got {sap_window}")
        self.sap_window = int(sap_window)
        self.n_sap = (2 * self.sap_window + 1) ** 2 if self.sap_window > 0 else 0
        self.action_mask = bool(action_mask)
        self.game_reward = float(game_reward)
        self._defaults = param_defaults()
        self.env_params = self._check_params(env_params or {})
        self._k = _kernels(self.sap_window, self.action_mask)
        self._alloc(self._n)
        self.reset()

    # ------------------------------------------------------------------

    def _check_params(self, p: Mapping[str, Any]) -> Dict[str, Any]:
        unknown = set(p) - set(self._defaults)
        if unknown:
            raise KeyError(f"unknown Lux env params {sorted(unknown)}")
        fixed = [k for k in p if k in STRUCTURAL_PARAMS and p[k] != self._defaults[k]]
        if fixed:
            raise ValueError(f"Lux env params {fixed} fix array shapes and cannot change")
        return dict(p)

    def _alloc(self, n: int) -> None:
        self._n = n
        self._keys = None                 # jax uint32 [N, 2]
        self._state = None                # batched EnvState
        self._rows = None                 # jax row arrays [N, 2, ...]
        self._params = {k: np.repeat(v[None], n) for k, v in self._defaults.items()}
        self._jp = None
        self._started = np.zeros(n, dtype=bool)
        self._live = np.zeros(n, dtype=bool)          # current match in progress
        self._game_over = np.zeros(n, dtype=bool)
        self._new = np.zeros(n, dtype=bool)
        self._wins = np.zeros((n, 2), dtype=np.int64)       # wins as of the last step
        self._wins_now = np.zeros((n, 2), dtype=np.int64)   # wins of the current frame
        self._ms = np.zeros(n, dtype=np.int64)
        self._steps = np.zeros(n, dtype=np.int64)
        self._frame: Dict[str, torch.Tensor] = {}

    def _jparams(self, params: Optional[Mapping[str, np.ndarray]] = None):
        """The (batched) params as an ``EnvParams`` of jax arrays (cached for
        the current games)."""
        lux = _lux()
        if params is not None:
            return lux.EnvParams(**{k: lux.jnp.asarray(v) for k, v in params.items()})
        if self._jp is None:
            self._jp = lux.EnvParams(**{k: lux.jnp.asarray(v) for k, v in self._params.items()})
        return self._jp

    def _start_games(self, fresh: np.ndarray, seeds: np.ndarray,
                     overrides: List[Optional[Mapping[str, Any]]]) -> None:
        """New games for the envs in ``fresh`` (bool [N]) from ``seeds``."""
        k, jnp = self._k, _lux().jnp
        with kaggle_rng():
            kd, reset_kd, vals = k.sample(jnp.asarray(seeds.astype(np.uint32)))
        params = {n: v.copy() for n, v in self._params.items()}
        for name, v in zip(k.param_names, vals):
            v = np.asarray(v)
            params[name] = np.where(fresh, v.astype(self._defaults[name].dtype), params[name])
        for name, v in self.env_params.items():
            params[name] = np.where(fresh, np.asarray(v, dtype=self._defaults[name].dtype),
                                    params[name])
        for i, ov in enumerate(overrides):
            if ov:
                for name, v in self._check_params(ov).items():
                    params[name][i] = np.asarray(v, dtype=self._defaults[name].dtype)
        with kaggle_rng():
            state, rows = k.gen_rows(div_tables(), reset_kd, self._jparams(params))
            if self._state is None or fresh.all():
                self._keys, self._state, self._rows = kd, state, rows
            else:
                live = jnp.asarray(fresh)
                self._keys = k.select(live, kd, self._keys)
                self._state = k.select(live, state, self._state)
                self._rows = k.select(live, rows, self._rows)
        self._params = params
        self._jp = None
        self._started[fresh] = True
        self._game_over[fresh] = False
        self._wins[fresh] = 0

    def _sync_host(self) -> None:
        """Host copies of the frame tensors and bookkeeping scalars."""
        r = self._rows
        R = self.num_rows
        out = {}
        for k in ("map_seq", "units_seq", "unit_mask", "globals_seq", "legal_unit",
                  "legal_unit_action", "legal_sap"):
            if k in r:
                a = np.array(r[k])
                out[k] = torch.from_numpy(a.reshape((R,) + a.shape[2:]))
        self._frame = out
        self._ms = np.asarray(r["ms"]).astype(np.int64)
        self._steps = np.asarray(r["steps"]).astype(np.int64)
        self._wins_now = np.asarray(r["wins"]).astype(np.int64)

    @property
    def finished(self) -> torch.Tensor:
        return torch.from_numpy(~self._live)

    @property
    def game_over(self) -> torch.Tensor:
        """[N] bool: the env's game (5 matches) is over."""
        return torch.from_numpy(self._game_over.copy())

    @property
    def matches_played(self) -> torch.Tensor:
        """[N] long: matches decided so far in each env's game (= the index of
        the current match while it runs)."""
        return torch.from_numpy(self._wins.sum(1))

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None, new_game: bool = False,
              params: Optional[Sequence[Optional[Mapping[str, Any]]]] = None) -> VecObs:
        """New episodes (matches) for all envs (``n`` resizes) or ``env_ids``.

        An env whose match ended continues its game with the next match; a
        fresh game starts for finished games, envs in the middle of a match,
        ``new_game=True``, a resize, or an entry of ``params`` (env params for
        the k-th reset env's new game, e.g. a replay's).  The k-th reset env's
        new game has seed ``seed + k`` (else a draw from the env RNG) — the
        competition's seed: its env params and map come from it."""
        if n is not None and n != self._n:
            if env_ids is not None:
                raise ValueError("resize (n) and env_ids are exclusive")
            self._alloc(int(n))
        ids = self._env_ids(env_ids)
        if params is not None and len(params) != len(ids):
            raise ValueError(f"{len(params)} params for {len(ids)} envs")
        seeds = self._episode_seeds(len(ids), seed)
        fresh = np.zeros(self._n, dtype=bool)
        seed_arr = np.zeros(self._n, dtype=np.int64)
        overrides: List[Optional[Mapping[str, Any]]] = [None] * self._n
        for k, i in enumerate(ids):
            pk = params[k] if params is not None else None
            if (new_game or pk is not None or not self._started[i] or self._game_over[i]
                    or self._live[i]):
                fresh[i] = True
                seed_arr[i] = int(seeds[k]) % 2**32
                overrides[i] = pk
        if fresh.any():
            self._start_games(fresh, seed_arr, overrides)
        self._new[:] = False
        self._new[ids] = True
        self._live[ids] = True
        self._sync_host()
        return self.observe()

    def obs_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {"map_seq": ((C, H, W), torch.float32), "units_seq": ((U, F), torch.float32),
                "unit_mask": ((U,), torch.bool), "globals_seq": ((G,), torch.float32)}

    # ------------------------------------------------------------------

    def observe(self) -> VecObs:
        R = self.num_rows
        f = self._frame
        active = torch.from_numpy(np.repeat(self._live, 2))
        legal = {"unit": f["legal_unit"].clone(), "unit_action": f["legal_unit_action"].clone()}
        if "legal_sap" in f:
            legal["sap"] = f["legal_sap"].clone()
        seat = torch.arange(R, dtype=torch.long) % 2
        return VecObs(
            obs={k: f[k].clone() for k in ("map_seq", "units_seq", "unit_mask", "globals_seq")},
            legal=legal,
            to_play=seat,
            env_id=self._row_env(),
            stream_id=torch.arange(R, dtype=torch.long),
            step_idx=torch.from_numpy(np.repeat(self._ms, 2)),
            new_episode=torch.from_numpy(np.repeat(self._new, 2)) & active,
            active=active,
            forced=active & ~legal["unit"].any(-1),
            stream_seats=torch.nn.functional.one_hot(seat, 2).bool(),
        )

    def _env_actions(self, actions: Mapping[str, torch.Tensor]
                     ) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        """Decoder actions → env actions [N, 2, U, 3] int32 + the applied (replay) format."""
        acts = self.replay_actions(actions)
        self._require(acts, ("unit_actions",))
        ua = acts["unit_actions"].detach().long().cpu().reshape(self.num_rows, U)
        a = ua.clamp(0, N_UNIT_ACTIONS - 1)
        applied = {"unit_actions": a.clone()}
        dx = torch.zeros_like(a)
        dy = torch.zeros_like(a)
        if self.sap_window > 0:
            if "sap_offsets" in acts:
                so = acts["sap_offsets"].detach().long().cpu().reshape(self.num_rows, U)
                so = so.clamp(0, self.n_sap - 1)
            else:
                so = torch.full_like(a, sap_class(0, 0, self.sap_window))
            applied["sap_offsets"] = so.clone()
            sx, sy = sap_delta(so, self.sap_window)
            is_sap = a == SAP_ACTION
            dx, dy = torch.where(is_sap, sx, dx), torch.where(is_sap, sy, dy)
        env_act = torch.stack([a, dx, dy], dim=-1).numpy().astype(np.int32)
        return env_act.reshape(self._n, 2, U, 3), applied

    def _raw_actions(self, env_actions) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        """Raw env actions (replays) → [N, 2, U, 3] + applied in the IL target convention."""
        raw = np.asarray(torch.as_tensor(env_actions).cpu(), dtype=np.int64)
        raw = raw.reshape(self._n, 2, U, 3)
        a = np.clip(raw[..., 0], 0, N_UNIT_ACTIONS - 1)
        applied = {"unit_actions": torch.from_numpy(a.reshape(self.num_rows, U).copy())}
        if self.sap_window > 0:
            dx, dy = raw[..., 1], raw[..., 2]
            w = self.sap_window
            ok = (a == SAP_ACTION) & (np.abs(dx) <= w) & (np.abs(dy) <= w)
            so = np.where(ok, sap_class(dx, dy, self.sap_window), -1)
            applied["sap_offsets"] = torch.from_numpy(so.reshape(self.num_rows, U).copy())
        return raw.astype(np.int32), applied

    def step(self, actions: Mapping[str, torch.Tensor], *,
             env_actions: Optional[Any] = None) -> StepResult:
        """One env step of every live env.  ``env_actions`` ([N, 2, 16, 3] or
        [R, 16, 3] raw Lux actions ``[a, dx, dy]``) replaces ``actions`` —
        the replay path (``info["applied"]`` in the IL target convention)."""
        n = self._n
        if env_actions is not None:
            env_act, applied = self._raw_actions(env_actions)
        else:
            env_act, applied = self._env_actions(actions)
        live = self._live.copy()
        k, jnp = self._k, _lux().jnp
        with kaggle_rng():
            if live.all():
                self._keys, self._state, self._rows = k.step(
                    div_tables(), self._keys, self._state, jnp.asarray(env_act), self._jparams())
            elif live.any():
                self._keys, self._state, self._rows = k.step_masked(
                    div_tables(), self._keys, self._state, jnp.asarray(env_act), self._jparams(),
                    jnp.asarray(live), self._rows)
        prev_wins = self._wins.copy()
        prev_ms = self._ms.copy()
        self._sync_host()
        wins = self._wins_now
        rewards = np.zeros((n, 2), dtype=np.float32)
        ended = live & (self._ms == 0)
        dw = wins - prev_wins
        winner = np.where(ended, np.argmax(dw, axis=1), -1)
        e_idx = np.flatnonzero(ended)
        rewards[e_idx, winner[e_idx]] = 1.0
        rewards[e_idx, 1 - winner[e_idx]] = -1.0
        game_len = ((self._params["max_steps_in_match"].astype(np.int64) + 1)
                    * self._params["match_count_per_episode"].astype(np.int64))
        over = ended & (self._steps >= game_len)
        game_winner = np.where(over, np.argmax(wins, axis=1), -1)
        if self.game_reward and over.any():
            o_idx = np.flatnonzero(over)
            rewards[o_idx, game_winner[o_idx]] += self.game_reward
            rewards[o_idx, 1 - game_winner[o_idx]] -= self.game_reward
        match_index = np.where(ended, prev_wins.sum(1), -1)
        self._wins = np.where(live[:, None], wins, self._wins)
        self._live[ended] = False
        self._game_over |= over
        self._new[:] = False

        acted = torch.from_numpy(np.repeat(live, 2))
        applied = {key: torch.where(acted.view(-1, 1), v, torch.zeros_like(v))
                   for key, v in applied.items()}
        info: Dict[str, Any] = {
            "terminated": torch.from_numpy(ended.copy()),
            "truncated": torch.zeros(n, dtype=torch.bool),
            "winner": torch.from_numpy(winner.astype(np.int64)),
            "episode_len": torch.from_numpy(np.where(ended, prev_ms + 1, 0).astype(np.int64)),
            "acted": acted,
            "applied": applied,
            "game_over": torch.from_numpy(over.copy()),
            "game_winner": torch.from_numpy(game_winner.astype(np.int64)),
            "team_wins": torch.from_numpy(wins.copy()),
            "match_index": torch.from_numpy(match_index.astype(np.int64)),
        }
        if self.autoreset and e_idx.size:
            self.reset(env_ids=e_idx.tolist())
        return self.observe(), torch.from_numpy(rewards), torch.from_numpy(ended), info

    # ------------------------------------------------------------------

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """A uniformly random legal action per unit (and a uniformly random
        legal sap target; the own cell when none): the argmax of uniform
        noise over the legal entries."""
        g = self._generator(generator)

        def pick(legal: torch.Tensor, fallback: int) -> torch.Tensor:
            noise = torch.rand(legal.shape, generator=g)
            noise = torch.where(legal, noise, torch.full_like(noise, -1.0))
            idx = noise.argmax(-1)
            return torch.where(legal.any(-1), idx, torch.full_like(idx, fallback))

        out = {"unit_actions": pick(obs.legal["unit_action"], 0)}
        if "sap" in obs.legal:
            out["sap_offsets"] = pick(obs.legal["sap"], sap_class(0, 0, self.sap_window))
        return out

    def raw_obs(self) -> List[Dict[str, Any]]:
        """Each seat's current official observation (``EnvObs``) as nested
        numpy arrays [N, ...] in the replay JSON layout (``units.position``,
        ``map_features.tile_type`` [x][y], ...) — for replay checks and
        scripted opponents."""
        with kaggle_rng():
            obs = self._k.env_obs(self._state, self._jparams())
        out = []
        for p in (0, 1):
            o = obs[f"player_{p}"]
            out.append({
                "units": {"position": np.asarray(o.units.position),
                          "energy": np.asarray(o.units.energy)},
                "units_mask": np.asarray(o.units_mask), "sensor_mask": np.asarray(o.sensor_mask),
                "map_features": {"energy": np.asarray(o.map_features.energy),
                                 "tile_type": np.asarray(o.map_features.tile_type)},
                "relic_nodes": np.asarray(o.relic_nodes),
                "relic_nodes_mask": np.asarray(o.relic_nodes_mask),
                "team_points": np.asarray(o.team_points), "team_wins": np.asarray(o.team_wins),
                "steps": np.asarray(o.steps), "match_steps": np.asarray(o.match_steps)})
        return out

    # -- checkpointing ---------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """The env RNGs plus every game's state (an exact continuation of
        unfinished games; the trainer itself restarts games on resume)."""
        jax = _lux().jax
        st = super().state_dict()
        st["games"] = {
            "keys": np.asarray(self._keys).copy(),
            "state": [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(self._state)],
            "params": {k: v.copy() for k, v in self._params.items()},
            **{k: getattr(self, "_" + k).copy()
               for k in ("started", "live", "game_over", "new", "wins")},
        }
        return st

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        g = state.get("games")
        if g is not None:
            self._load_games(g)
        super().load_state_dict(state)          # last: the resize below draws no RNG after it

    def _load_games(self, g: Mapping[str, Any]) -> None:
        lux = _lux()
        jax, jnp = lux.jax, lux.jnp
        n = int(g["keys"].shape[0])
        if n != self._n:                        # the batched structures for n envs
            self._alloc(n)
            self.reset(seed=0)
        treedef = jax.tree_util.tree_structure(self._state)
        self._state = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in g["state"]])
        self._keys = jnp.asarray(g["keys"])
        self._params = {k: np.asarray(v).copy() for k, v in g["params"].items()}
        self._jp = None
        for k in ("started", "live", "game_over", "new", "wins"):
            setattr(self, "_" + k, np.asarray(g[k]).copy())
        with kaggle_rng():
            self._rows = self._k.obs_rows(div_tables(), self._state, self._jparams())
        self._sync_host()


# ---------------------------------------------------------------------------
# Kaggle replays
# ---------------------------------------------------------------------------

def load_kaggle_replay(path: str) -> Dict[str, Any]:
    """A Kaggle episode replay (``.json`` or ``.json.gz``)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def kaggle_seed(ep: Mapping[str, Any]) -> int:
    """The episode's env seed (``configuration.seed``: LuxAIS3GymEnv.reset(seed))."""
    return int(ep["configuration"]["seed"])


def kaggle_env_actions(ep: Mapping[str, Any]) -> np.ndarray:
    """[T, 2, 16, 3] int32 raw actions of env steps 1..T (``steps[t][p]["action"]``:
    the interpreter's applied action; zeros where missing)."""
    steps = ep["steps"]
    out = np.zeros((len(steps) - 1, 2, U, 3), dtype=np.int32)
    for t in range(1, len(steps)):
        for p in range(2):
            a = steps[t][p].get("action")
            if isinstance(a, list) and a:
                out[t - 1, p] = np.asarray(a, dtype=np.int64).reshape(U, 3)
    return out


def kaggle_obs(ep: Mapping[str, Any], t: int, p: int) -> Optional[Dict[str, Any]]:
    """Agent ``p``'s parsed observation of step ``t`` (None if absent)."""
    raw = ep["steps"][t][p].get("observation", {}).get("obs")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw if isinstance(raw, dict) else None


def kaggle_env_globals(ep: Mapping[str, Any], agent: int) -> np.ndarray:
    """[5] float32 env-parameter globals as lux.rs reads them: each key from
    ``steps[0][agent].info.env_cfg``, else ``configuration.env_cfg``, else the
    Lux S3 default; known = 1 only when all four came from one source."""
    sources = [((ep["steps"][0][agent].get("info") or {}).get("env_cfg") or {}),
               ((ep.get("configuration") or {}).get("env_cfg") or {})]
    vals, src = [], []
    for key, norm in ENV_GLOBALS:
        v, s = ENV_GLOBAL_DEFAULTS[key], None
        for si, d in enumerate(sources):
            if isinstance(d.get(key), (int, float)) and not isinstance(d.get(key), bool):
                v, s = float(d[key]), si
                break
        vals.append(np.float32(np.float64(v) / norm))
        src.append(s)
    known = src[0] is not None and all(s == src[0] for s in src)
    return np.asarray(vals + [np.float32(known)], dtype=np.float32)


__all__ = ["LuxVecEnv", "build_obs", "build_legal", "env_obs_arrays", "json_obs_arrays",
           "env_globals_of", "kaggle_rng", "load_kaggle_replay", "kaggle_seed",
           "kaggle_env_actions", "kaggle_obs", "kaggle_env_globals", "sap_class", "sap_delta",
           "available", "param_defaults"]
