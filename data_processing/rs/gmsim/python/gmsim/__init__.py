"""gmsim — the gmrules game engines for Python: batched simulators (the RL
environments behind ``env/vec_gmsim.py``), verified replay of trajectories,
single states for step-by-step play, and the invariance / conformance checks.

All rules and encoders live in Rust (``data_processing/rs/gmrules``) and are the
same code the ``gmdata`` converters use, so a simulator frame IS the IL frame of
the same position.  This module only views the extension's buffers as numpy
arrays.

Conventions:
  * ``config``: a dict (JSON-serialisable), a JSON string, or None (the game's
    default config);
  * ``events``: a trajectory, int64 ``[n, 2]`` pairs ``(seat, id)`` with seat −1
    for a chance outcome (array, list of pairs, or the raw bytes);
  * frames are named tensors in three groups, as the game's spec declares them
    (``spec["obs"]`` / ``["legal"]`` / ``["target"]``: lists of ``{"name",
    "shape", "dtype", "bits"}``, per-frame shapes): ``obs`` = the network input
    (gm-shard ``obs.*``), ``legal`` = the decoder's legality masks, ``target`` =
    the IL labels of the taken action (gm-shard ``tgt.*``); ``spec["action_keys"]``
    (int64 scalars among the targets) are the policy's action — what
    :meth:`Sim.step` takes.  Candidate-action games (``spec["schema"] ==
    "candidate.v1"``): obs ``entities [E, Fe]``, ``entity_mask [E]``,
    (``entity_type [E]``), ``globals [G]``, ``actions [A, Fa]``,
    ``action_mask [A]``; legal ``action [A]``; target / action key ``action_idx``.

Build: ``uv sync`` builds this package with maturin (needs ``cargo`` on PATH);
see data_processing/rs/gmsim/README.md.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from gmsim import _native

__all__ = ["CANDIDATE_OBS_KEYS", "CANDIDATE_SCHEMA", "Sim", "State", "check_game", "config_json",
           "events_array", "game_info", "games", "perturb_hidden", "random_trajectory", "replay",
           "spec_of", "tensors_from_buffers"]

#: schema name of the candidate-action games
CANDIDATE_SCHEMA = "candidate.v1"
#: obs tensors of candidate-action games (``entity_type`` only when the game has
#: entity types)
CANDIDATE_OBS_KEYS = ("entities", "entity_mask", "entity_type", "globals", "actions",
                      "action_mask")

_NP = {"f32": np.float32, "bool": np.uint8, "i32": np.int32, "i64": np.int64}

Events = Union[bytes, np.ndarray, Sequence[Sequence[int]]]


def config_json(config: Union[None, str, Mapping[str, Any]]) -> str:
    """Normalise a config argument to the JSON string the extension takes."""
    if config is None:
        return ""
    if isinstance(config, str):
        return config
    return json.dumps(dict(config), sort_keys=True)


def events_array(events: Events) -> np.ndarray:
    """A trajectory as int64 ``[n, 2]`` ``(seat, id)`` pairs."""
    if isinstance(events, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(events), dtype=np.int64).reshape(-1, 2).copy()
    a = np.asarray(events, dtype=np.int64)
    if a.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"events must be (seat, id) pairs, got shape {a.shape}")
    return a


def _ev_bytes(events: Events) -> bytes:
    return np.ascontiguousarray(events_array(events)).tobytes()


def games() -> List[str]:
    """Names of every registered gmrules game."""
    return list(_native.games())


def game_info(game: str, config=None) -> Dict[str, Any]:
    """``{"name", "version", "config" (normalised JSON), "spec": {...}, "teams"}`` —
    ``teams``: the team id of every seat for team games (bridge partnerships
    ``[0, 1, 0, 1]``: teammates share one outcome, ``winner`` names the winning
    team's lowest seat), else None; spec =
    ``n_seats``, ``layout`` ("per_seat" | "interleaved"), ``hidden_info``,
    ``simultaneous``, ``max_decisions``, ``schema``, the tensor lists ``obs`` /
    ``legal`` / ``target``, ``action_keys``, ``meta`` (the shard-meta schema facts,
    e.g. the candidate frame dims) and ``candidate`` (the candidate dims, or None
    for a custom schema)."""
    return dict(_native.game_info(game, config_json(config)))


def spec_of(game: str, config=None) -> Dict[str, Any]:
    """The game's frame spec (:func:`game_info`'s ``"spec"``)."""
    return game_info(game, config)["spec"]


def _array(buf, t: Dict[str, Any], n: Optional[int]) -> np.ndarray:
    a = np.frombuffer(buf, _NP[t["dtype"]])
    if t["dtype"] == "bool":
        a = a.view(np.bool_)
    shape = tuple(t["shape"])
    return a.reshape(shape if n is None else (n,) + shape)


def tensors_from_buffers(d: Mapping[str, Any], specs: Sequence[Dict[str, Any]],
                         n: Optional[int]) -> Dict[str, np.ndarray]:
    """The extension's ``{name: bytearray}`` → ``{name: array [n, *shape]}`` (writable
    views, no copy; ``n=None``: one frame, ``[*shape]``)."""
    return {t["name"]: _array(d[t["name"]], t, n) for t in specs}


def replay(game: str, config, events: Events) -> Dict[str, Any]:
    """Verified replay of ``events`` (raises ValueError on an illegal / malformed
    event).  Returns per decision (in event order) ``seat``, ``action`` (canonical
    id), ``action_idx`` (index among the legal ids = the candidate index), ``turn``,
    ``event`` (index in the trajectory) as int64 ``[D]``; ``obs`` / ``legal`` /
    ``target`` dicts of ``[D, ...]`` arrays and ``legal_ids`` (per decision); plus
    ``terminal``, ``outcome`` / ``returns`` (per seat, None unless terminal),
    ``turns``, ``streams`` (the IL samples: ``[(seat or -1, [decision indices])]``),
    ``spec`` and ``n``."""
    d = dict(_native.replay(game, config_json(config), _ev_bytes(events)))
    n, spec = d["n"], d["spec"]
    out: Dict[str, Any] = {k: np.frombuffer(d[k], np.int64).copy()
                           for k in ("seat", "action", "action_idx", "turn", "event")}
    for group in ("obs", "legal", "target"):
        out[group] = tensors_from_buffers(d[group], spec[group], n)
    for k in ("legal_ids", "terminal", "outcome", "returns", "turns", "streams", "spec", "n"):
        out[k] = d[k]
    return out


def perturb_hidden(game: str, config, events: Events, seat: int, seed: int
                   ) -> Optional[np.ndarray]:
    """The game's privacy perturbation: a trajectory differing from ``events``
    only in information hidden from ``seat`` (None = nothing to perturb)."""
    b = _native.perturb_hidden(game, config_json(config), _ev_bytes(events), int(seat), int(seed))
    return None if b is None else events_array(b)


def random_trajectory(game: str, config, seed: int, max_events: int = 100_000) -> np.ndarray:
    """A uniformly random playout (chance by its probabilities)."""
    return events_array(_native.random_trajectory(game, config_json(config), int(seed),
                                                  int(max_events)))


def check_game(game: str, config=None, n_games: int = 200, seed: int = 0) -> str:
    """gmrules' conformance battery (random playouts: frame checks, action-key round
    trips, causality, privacy, simulator replays); raises ValueError on a violation."""
    return _native.check_game(game, config_json(config), int(n_games), int(seed))


class Sim:
    """N envs of one game stepped together (gmrules ``VecSim``).

    Rows: per-seat layout — ``n_seats`` rows per env, row ``env * n_seats + seat``,
    active only when that seat is to act; interleaved — one row per env.
    ``threads``: 1 = step on the calling thread, 0 = rayon's global pool, n > 1 =
    a pool of n threads (results never depend on it; the GIL is released while
    stepping).  :meth:`observe` returns fresh numpy arrays every call."""

    def __init__(self, game: str, config=None, n_envs: int = 1, *, autoreset: bool = False,
                 max_turns: Optional[int] = None, threads: int = 1):
        self._s = _native.Sim(game, config_json(config), int(n_envs), bool(autoreset),
                              None if max_turns is None else int(max_turns), int(threads))
        self.game = game
        self.spec: Dict[str, Any] = dict(self._s.spec)
        self.action_keys: Tuple[str, ...] = tuple(self.spec["action_keys"])

    @property
    def n_envs(self) -> int:
        return self._s.n_envs

    @property
    def n_rows(self) -> int:
        return self._s.n_rows

    @property
    def rows_per_env(self) -> int:
        return self._s.rows_per_env

    @property
    def layout(self) -> str:
        return self._s.layout

    def resize(self, n: int) -> None:
        self._s.resize(int(n))

    def reset(self, env_ids: Sequence[int], seeds: Sequence[int]) -> None:
        self._s.reset([int(e) for e in env_ids], [int(s) for s in seeds])

    def observe(self) -> Dict[str, Any]:
        """``obs`` / ``legal`` dicts of ``[R, ...]`` arrays + ``n_legal``, ``to_play``,
        ``step_idx`` (int64 [R]) and ``active``, ``new_episode``, ``forced`` (bool [R])."""
        d = self._s.observe()
        R = self._s.n_rows
        out: Dict[str, Any] = {g: tensors_from_buffers(d[g], self.spec[g], R)
                               for g in ("obs", "legal")}
        for k in ("n_legal", "to_play", "step_idx"):
            out[k] = np.frombuffer(d[k], np.int64)
        for k in ("active", "new_episode", "forced"):
            out[k] = np.frombuffer(d[k], np.uint8).view(np.bool_)
        return out

    def _keys_array(self, actions) -> np.ndarray:
        R, K = self._s.n_rows, len(self.action_keys)
        if isinstance(actions, Mapping):
            missing = [k for k in self.action_keys if k not in actions]
            if missing:
                raise KeyError(f"{self.game}: missing action keys {missing} "
                               f"(expected {list(self.action_keys)})")
            cols = [np.asarray(actions[k], dtype=np.int64).reshape(-1) for k in self.action_keys]
            a = np.stack(cols, axis=1) if cols else np.zeros((R, 0), np.int64)
        else:
            a = np.asarray(actions, dtype=np.int64).reshape(R, -1) if R else \
                np.zeros((0, K), np.int64)
        if a.shape != (R, K):
            raise ValueError(f"{self.game}: actions of shape {a.shape}, expected [{R}, {K}] "
                             f"(action keys {list(self.action_keys)})")
        return np.ascontiguousarray(a)

    def step(self, actions) -> Dict[str, Any]:
        """Apply one action per row: a dict ``{action key: [R]}``, or an int array
        ``[R, K]`` (``[R]`` for single-key games, e.g. the candidate index).
        Returns ``rewards`` / ``returns`` ``[N, S]``, ``done`` / ``terminated`` /
        ``truncated`` bool ``[N]``, ``winner`` / ``episode_len`` int64 ``[N]``,
        ``applied`` ``{action key: int64 [R]}`` (−1 = the row did not act) and
        ``acted`` bool ``[R]``."""
        a = self._keys_array(actions)
        d = self._s.step(a.tobytes())
        N, S, K = self._s.n_envs, self.spec["n_seats"], len(self.action_keys)
        applied = np.frombuffer(d["applied"], np.int64).reshape(-1, K)
        return {
            "rewards": np.frombuffer(d["rewards"], np.float32).reshape(N, S),
            "returns": np.frombuffer(d["returns"], np.float64).reshape(N, S),
            "done": np.frombuffer(d["done"], np.uint8).view(np.bool_),
            "terminated": np.frombuffer(d["terminated"], np.uint8).view(np.bool_),
            "truncated": np.frombuffer(d["truncated"], np.uint8).view(np.bool_),
            "winner": np.frombuffer(d["winner"], np.int64),
            "episode_len": np.frombuffer(d["episode_len"], np.int64),
            "applied": {k: applied[:, i].copy() for i, k in enumerate(self.action_keys)},
            "acted": np.frombuffer(d["acted"], np.uint8).view(np.bool_),
        }

    def random_actions(self, seed: int) -> Dict[str, np.ndarray]:
        """``{action key: int64 [R]}`` of a uniformly random legal action on every
        acting row (zeros elsewhere); a pure function of ``seed`` and the state."""
        a = np.frombuffer(self._s.random_actions(int(seed) & (2**64 - 1)), np.int64)
        a = a.reshape(-1, len(self.action_keys))
        return {k: a[:, i].copy() for i, k in enumerate(self.action_keys)}

    def legal_ids(self, row: int) -> List[int]:
        return list(self._s.legal_ids(int(row)))

    def trajectory(self, env: int) -> np.ndarray:
        return events_array(self._s.trajectory(int(env)))

    def done(self, env: int) -> bool:
        return bool(self._s.done(int(env)))

    def done_all(self) -> np.ndarray:
        """Done flag of every env (bool [N])."""
        return np.frombuffer(self._s.done_all(), np.uint8).view(np.bool_)

    def state_string(self, env: int) -> str:
        return self._s.state_string(int(env))


class State:
    """One game state stepped event by event (differential tests vs reference
    implementations, debugging)."""

    def __init__(self, game: str, config=None, _inner=None):
        self._s = _inner if _inner is not None else _native.State(game, config_json(config))
        self.game = game

    def turn(self) -> Tuple[str, List[int]]:
        """("chance", []) | ("act", [seats]) | ("terminal", [])."""
        kind, seats = self._s.turn()
        return kind, list(seats)

    def chance_outcomes(self) -> List[Tuple[int, float]]:
        return list(self._s.chance_outcomes())

    def legal_actions(self, seat: int) -> List[int]:
        return list(self._s.legal_actions(int(seat)))

    def apply_chance(self, outcome: int) -> None:
        self._s.apply_chance(int(outcome))

    def apply_action(self, seat: int, action: int) -> None:
        self._s.apply_action(int(seat), int(action))

    def apply(self, event) -> None:
        seat, i = int(event[0]), int(event[1])
        if seat < 0:
            self.apply_chance(i)
        else:
            self.apply_action(seat, i)

    def returns(self) -> List[float]:
        return list(self._s.returns())

    def outcome(self) -> List[float]:
        return list(self._s.outcome())

    def frame(self, seat: int) -> Dict[str, Any]:
        """``seat``'s decision frame now: ``obs`` / ``legal`` dicts of per-frame
        arrays and ``legal_ids``."""
        d = self._s.frame(int(seat))
        spec = self.spec()
        out: Dict[str, Any] = {g: tensors_from_buffers(d[g], spec[g], None)
                               for g in ("obs", "legal")}
        out["legal_ids"] = list(d["legal_ids"][0])
        return out

    def target(self, seat: int, action: int) -> Dict[str, np.ndarray]:
        """The target tensors of ``seat`` taking the legal ``action`` now."""
        return tensors_from_buffers(self._s.target(int(seat), int(action)),
                                    self.spec()["target"], None)

    def decode(self, seat: int, keys: Sequence[int]) -> int:
        """The legal action of ``seat`` whose action keys are ``keys``."""
        return int(self._s.decode(int(seat), [int(k) for k in keys]))

    def history(self) -> np.ndarray:
        return events_array(self._s.history())

    def spec(self) -> Dict[str, Any]:
        return dict(self._s.spec())

    def action_string(self, seat: int, action: int) -> str:
        return self._s.action_string(int(seat), int(action))

    def clone(self) -> "State":
        return State(self.game, _inner=self._s.clone())

    def __str__(self) -> str:
        return str(self._s)
