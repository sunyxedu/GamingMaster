"""gmsim — the gmrules game engines for Python: batched simulators (the RL
environments behind ``env/vec_candidate.py``), verified replay of trajectories,
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
  * frames: ``entities [n, E, Fe] f32``, ``entity_mask [n, E] bool``,
    ``entity_type [n, E] int32``, ``globals [n, G] f32``, ``actions [n, A, Fa] f32``,
    ``action_mask [n, A] bool`` — the candidate-action gm-shard schema.

Build: ``uv sync`` builds this package with maturin (needs ``cargo`` on PATH);
see data_processing/rs/gmsim/README.md.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from gmsim import _native

__all__ = ["FRAME_KEYS", "Sim", "State", "check_game", "config_json", "events_array",
           "game_info", "games", "perturb_hidden", "random_trajectory", "replay"]

#: per-frame observation keys (the gm-shard ``obs.*`` tensors of candidate games;
#: ``entity_type`` only when the game has entity types)
FRAME_KEYS = ("entities", "entity_mask", "entity_type", "globals", "actions", "action_mask")

Events = Union[bytes, np.ndarray, Sequence[Sequence[int]]]


def config_json(config: Union[None, str, Dict[str, Any]]) -> str:
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
    """``{"name", "version", "config" (normalised JSON), "spec": {...}}`` — spec =
    the frame shapes (max_entities, entity_features, entity_types, global_features,
    max_actions, action_features), n_seats, layout ("per_seat" | "interleaved"),
    hidden_info, simultaneous, bits_* storage hints, max_decisions."""
    return dict(_native.game_info(game, config_json(config)))


def frames_from_buffers(d: Dict[str, Any], spec: Dict[str, Any], n: int) -> Dict[str, np.ndarray]:
    """The extension's frame bytearrays → numpy arrays (writable views, no copy)."""
    E, Fe = spec["max_entities"], spec["entity_features"]
    A, Fa = spec["max_actions"], spec["action_features"]
    return {
        "entities": np.frombuffer(d["entities"], np.float32).reshape(n, E, Fe),
        "entity_mask": np.frombuffer(d["entity_mask"], np.uint8).reshape(n, E).astype(bool),
        "entity_type": np.frombuffer(d["entity_type"], np.int32).reshape(n, E),
        "globals": np.frombuffer(d["globals"], np.float32).reshape(n, spec["global_features"]),
        "actions": np.frombuffer(d["actions"], np.float32).reshape(n, A, Fa),
        "action_mask": np.frombuffer(d["action_mask"], np.uint8).reshape(n, A).astype(bool),
    }


def replay(game: str, config, events: Events) -> Dict[str, Any]:
    """Verified replay of ``events`` (raises ValueError on an illegal / malformed
    event).  Returns per decision (in event order) ``seat``, ``action`` (canonical
    id), ``action_idx`` (candidate index = the IL target), ``turn``, ``event``
    (index in the trajectory) as int64 ``[D]``, the frames (FRAME_KEYS) and
    ``legal`` (candidate ids per decision); plus ``terminal``, ``outcome`` /
    ``returns`` (per seat, None unless terminal), ``turns``, ``streams`` (the
    IL samples: ``[(seat or -1, [decision indices])]``) and ``spec``."""
    d = dict(_native.replay(game, config_json(config), _ev_bytes(events)))
    n = d["n"]
    out: Dict[str, Any] = {k: np.frombuffer(d[k], np.int64).copy()
                           for k in ("seat", "action", "action_idx", "turn", "event")}
    out.update(frames_from_buffers(d, d["spec"], n))
    for k in ("legal", "terminal", "outcome", "returns", "turns", "streams", "spec", "n"):
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
    """gmrules' conformance battery (random playouts: frame checks, causality,
    privacy, simulator replays); raises ValueError on a violation."""
    return _native.check_game(game, config_json(config), int(n_games), int(seed))


class Sim:
    """N envs of one game stepped together (gmrules ``VecSim``).

    Rows: per-seat layout — ``n_seats`` rows per env, row ``env * n_seats + seat``,
    active only when that seat is to act; interleaved — one row per env.
    :meth:`observe` returns fresh numpy arrays every call."""

    def __init__(self, game: str, config=None, n_envs: int = 1, *, autoreset: bool = False,
                 max_turns: Optional[int] = None):
        self._s = _native.Sim(game, config_json(config), int(n_envs), bool(autoreset),
                              None if max_turns is None else int(max_turns))
        self.game = game
        self.spec: Dict[str, Any] = dict(self._s.spec)

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

    def observe(self) -> Dict[str, np.ndarray]:
        """Frames of every row (FRAME_KEYS) + ``n_legal``, ``to_play``, ``step_idx``
        (int64 [R]) and ``active``, ``new_episode``, ``forced`` (bool [R])."""
        d = self._s.observe()
        R = self._s.n_rows
        out = frames_from_buffers(d, self.spec, R)
        for k in ("n_legal", "to_play", "step_idx"):
            out[k] = np.frombuffer(d[k], np.int64)
        for k in ("active", "new_episode", "forced"):
            out[k] = np.frombuffer(d[k], np.uint8).astype(bool)
        return out

    def step(self, action_idx) -> Dict[str, np.ndarray]:
        """Apply one candidate index per row (``[R]``); returns ``rewards`` /
        ``returns`` ``[N, S]``, ``done`` / ``terminated`` / ``truncated`` bool
        ``[N]``, ``winner`` / ``episode_len`` int64 ``[N]``, ``applied`` int64
        ``[R]`` (−1 = the row did not act)."""
        a = np.ascontiguousarray(np.asarray(action_idx, dtype=np.int64).reshape(-1))
        d = self._s.step(a.tobytes())
        N, S = self._s.n_envs, self.spec["n_seats"]
        return {
            "rewards": np.frombuffer(d["rewards"], np.float32).reshape(N, S),
            "returns": np.frombuffer(d["returns"], np.float64).reshape(N, S),
            "done": np.frombuffer(d["done"], np.uint8).astype(bool),
            "terminated": np.frombuffer(d["terminated"], np.uint8).astype(bool),
            "truncated": np.frombuffer(d["truncated"], np.uint8).astype(bool),
            "winner": np.frombuffer(d["winner"], np.int64),
            "episode_len": np.frombuffer(d["episode_len"], np.int64),
            "applied": np.frombuffer(d["applied"], np.int64),
        }

    def legal_ids(self, row: int) -> List[int]:
        return list(self._s.legal_ids(int(row)))

    def trajectory(self, env: int) -> np.ndarray:
        return events_array(self._s.trajectory(int(env)))

    def done(self, env: int) -> bool:
        return bool(self._s.done(int(env)))

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
        d = self._s.frame(int(seat))
        out: Dict[str, Any] = {k: v[0] for k, v in frames_from_buffers(d, self.spec(), 1).items()}
        out["legal"] = list(d["legal"][0])
        return out

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
