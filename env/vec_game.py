"""Vectorised game environments for Stage-2 RL (mixed multi-game PPO).

A :class:`VecGameEnv` steps ``N`` games of one game at once and speaks the
tensor language of the IL-pretrained :class:`~model.multi_game.MultiGameNetwork`:

  observations  one frame per ROW, keyed and typed exactly like that game's
                gm-shard ``obs`` schema (``obs_schema()``; the per-frame slice of
                an IL sample), so the pretrained encoder sees in-distribution
                inputs.  Pinned frame by frame to the Rust converters' IL data
                by tests/test_vec_game_golden.py.
  legal         the decoder's ``legal`` dict (conditional masks, e.g. generals
                source + per-cell direction), plus ``"pass"`` (bool [R], pass
                is legal) for pass-gated placement games.
  actions       ``step`` takes the decoder's ``sample()`` output (``action_*``
                keys) or its ``log_prob_entropy`` replay format
                (``replay_actions``); ``info["applied"]`` reports the action
                the rules actually applied, in replay format.

Rows, streams, seats
--------------------
Row ``r`` belongs to env ``r // rows_per_env`` (a fixed layout: rows never move,
so step caches stay aligned).  A row is one temporal STREAM of the network —
the sequence of frames whose Stage-1 state carries over:

  * simultaneous games (generals, lux): ``rows_per_env = n_seats``; row
    ``(env, seat)`` is that seat's own fog-of-war view, one frame per turn.
    (Lux: one episode = one match — the IL sample — and ``reset`` continues
    the 5-match game; see env/vec_lux.py.)
  * gmrules games (env/vec_gmsim.py) with hidden information or simultaneous
    turns: ``rows_per_env = n_seats``, row
    ``(env, seat)`` holds only that seat's own decision frames; a row is
    ``active`` only on its seat's turns (alternating: one row per env and step),
    so rows advance at different frame indices (``lockstep = False``).
  * alternating games (othello, gomoku, ``interleaved = True``): ``rows_per_env = 1``;
    the row is ONE interleaved stream holding both movers' frames, each from
    the side-to-move perspective, forced pass frames included — exactly the
    IL sample of a whole game.  ``to_play`` says whose frame it is (the loss
    goes on learner seats only; see env/league.py).

``VecObs.stream_seats`` [R, n_seats] marks the seats acting on a row's stream:
a controller of any of them must run its network on EVERY frame of that stream
(also the frames where it does not act) to keep its temporal state in
distribution.

Rewards are per seat (``[N, n_seats]``, earned on the step); ``done`` marks
episodes that ended on the step (``info["terminated"]`` / ``["truncated"]``;
truncation is a soft terminal the env scores itself — do not bootstrap).
With ``autoreset`` an ended env restarts immediately (its rows come back with
``new_episode``); without it its rows go inactive until ``reset``.

Rollout contract (Stage-2 trainer): step the envs in LOCKSTEP — autoreset off,
reset all envs at the iteration start, play until every env is done, with a
bounded episode length (generals ``max_turns``).  All rows then share one frame
index per step, which is what ``GameNetwork.step`` takes (one ``step_idx`` per
call): env/league.py's :class:`~env.league.StreamStepper` runs them as ONE
batched call per step.  With ``autoreset`` the rows drift apart in frame index
and a stateful policy splits into one call per distinct index (fine for
evaluation / smoke runs, not for throughput).

Devices: the adapters simulate on the CPU and return CPU tensors; actions may
come on any device (``step`` moves what it needs to the host).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, fields
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import torch

#: step() return: (next obs, rewards [N, n_seats], done [N], info)
StepResult = Tuple["VecObs", torch.Tensor, torch.Tensor, Dict[str, Any]]


@dataclass
class VecObs:
    """One decision frame per row (tensors are fresh every step: safe to keep).

    obs           {key: [R, ...]}  the game's gm-shard obs schema, one frame
    legal         {key: [R, ...]}  the decoder's legal dict (+ "pass" [R] bool
                                   for pass-gated placement decoders)
    to_play       [R] long   seat acting on the row's current frame (its
                             perspective: every row is seen by its mover)
    env_id        [R] long   env of the row
    stream_id     [R] long   temporal stream of the row (= row index)
    step_idx      [R] long   frame index of the current frame in its stream
                             (the ``step_idx`` / decoder ``t0`` of GameNetwork.step)
    new_episode   [R] bool   the row's env started a new episode on this step
                             (reset temporal state).  Per-seat gmrules streams
                             flag every row of the env, acting or not; a stream
                             whose first frame comes later starts at step_idx 0
                             — policies reset a stream on new_episode OR step_idx 0
    active        [R] bool   the row has a frame and needs an action (False: its
                             episode is over, no autoreset — or, per-seat streams of
                             alternating gmrules games, another seat is to act)
    forced        [R] bool   the rules leave no choice (othello pass frame,
                             generals with no movable army): the env applies
                             the forced action whatever is sent
    stream_seats  [R, S] bool seats acting on the row's stream
    privileged    optional {key: [R, ...]} critic-only information (full state)
    aux           {key: [R, ...]} per-row auxiliary labels (e.g. generals capture labels)
    """

    obs: Dict[str, torch.Tensor]
    legal: Dict[str, torch.Tensor]
    to_play: torch.Tensor
    env_id: torch.Tensor
    stream_id: torch.Tensor
    step_idx: torch.Tensor
    new_episode: torch.Tensor
    active: torch.Tensor
    forced: torch.Tensor
    stream_seats: torch.Tensor
    privileged: Optional[Dict[str, torch.Tensor]] = None
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.to_play.shape[0])

    def select(self, rows: torch.Tensor) -> "VecObs":
        """The rows ``rows`` (LongTensor or bool mask) as a smaller VecObs."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, dict):
                out[f.name] = {k: t[rows] for k, t in v.items()}
            elif v is None:
                out[f.name] = None
            else:
                out[f.name] = v[rows]
        return VecObs(**out)

    def to(self, device) -> "VecObs":
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, dict):
                out[f.name] = {k: t.to(device) for k, t in v.items()}
            elif v is None:
                out[f.name] = None
            else:
                out[f.name] = v.to(device)
        return VecObs(**out)


@runtime_checkable
class VecGameEnv(Protocol):
    """N games of one game stepped together (see the module docstring)."""

    game: str
    n_seats: int
    rows_per_env: int
    interleaved: bool
    autoreset: bool

    @property
    def num_envs(self) -> int: ...

    @property
    def num_rows(self) -> int: ...

    @property
    def finished(self) -> torch.Tensor:
        """[N] bool: episode over and not restarted (autoreset off)."""

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        """Start new episodes: all ``n`` envs (resizing), or only ``env_ids``."""

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        """Apply one action per row (inactive rows ignored)."""

    def observe(self) -> VecObs:
        """The current frame of every row, without stepping."""

    def obs_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        """Per-frame {obs key: (shape, dtype)} — the gm-shard obs schema."""

    def replay_actions(self, actions: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decoder ``sample()`` output → its ``log_prob_entropy`` action format."""

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """A uniformly random legal action for every row (replay format);
        ``generator=None`` draws from the env's own torch RNG (seeded from the
        env seed, part of ``state_dict``)."""


class VecEnvBase:
    """Shared plumbing: row layout, action-key normalisation, the env RNGs
    (numpy ``_rng`` for episode seeds, torch ``_torch_rng`` for
    ``random_actions`` without a generator)."""

    game: str = ""
    n_seats: int = 2
    rows_per_env: int = 1
    interleaved: bool = False
    #: decoder sample() key → log_prob_entropy (replay) key
    ACTION_KEYS: Dict[str, str] = {}

    def __init__(self, n_envs: int, autoreset: bool, seed: Optional[int]):
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.autoreset = bool(autoreset)
        self._seed_rngs(seed)
        self._n = int(n_envs)
        #: duplicate deals (train_rl_joint --eval_duplicate): with a seeded reset, groups of
        #: this many consecutive envs share their episode seed (see _episode_seeds)
        self.deal_group = 1

    def _seed_rngs(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed)
        # an independent stream (drawing it never shifts the episode seeds)
        ss = np.random.default_rng(None if seed is None else [int(seed), 0x7E5])
        self._torch_rng = torch.Generator().manual_seed(int(ss.integers(0, 2**62)))

    def _generator(self, generator: Optional[torch.Generator]) -> torch.Generator:
        return self._torch_rng if generator is None else generator

    # -- layout ------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._n

    @property
    def num_rows(self) -> int:
        return self._n * self.rows_per_env

    def _row_env(self) -> torch.Tensor:
        return torch.arange(self.num_rows, dtype=torch.long) // self.rows_per_env

    def _env_ids(self, env_ids: Optional[Sequence[int]]) -> np.ndarray:
        if env_ids is None:
            return np.arange(self._n)
        ids = np.asarray(list(env_ids), dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self._n):
            raise IndexError(f"env_ids {ids.tolist()} out of range for {self._n} envs")
        return ids

    def _episode_seeds(self, count: int, seed: Optional[int]) -> list:
        """Per-episode seeds: ``seed + k`` when given, else from the env RNG.  With
        ``deal_group`` G > 1 a given seed makes them ``seed + k // G``: G consecutive envs
        play the same deal (cards, dice, wall, map — everything the episode seed draws)."""
        if seed is not None:
            g = max(1, int(getattr(self, "deal_group", 1)))
            return [int(seed) + k // g for k in range(count)]
        return [int(s) for s in self._rng.integers(0, 2**31 - 1, size=count)]

    # -- actions -------------------------------------------------------------

    def replay_actions(self, actions: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Map ``action_*`` sample keys to the replay keys; replay-format keys
        pass through (a key given both ways must agree).  Per-row shapes and
        devices are kept (``[R]`` scalars, ``[R, U]`` per-unit actions, ...)."""
        out: Dict[str, torch.Tensor] = {}
        replay_keys = set(self.ACTION_KEYS.values())
        for k, v in actions.items():
            if k in self.ACTION_KEYS:
                rk = self.ACTION_KEYS[k]
            elif k in replay_keys:
                rk = k
            else:
                continue            # log-probs, values, ... are not actions
            v = torch.as_tensor(v).detach()
            if v.dim() == 0:
                v = v.reshape(1)
            if rk in out and not (out[rk].shape == v.shape
                                  and torch.equal(out[rk], v.to(out[rk].device))):
                raise ValueError(f"action {rk!r} given twice with different values")
            out[rk] = v
        return out

    def _require(self, acts: Dict[str, torch.Tensor], keys: Sequence[str]) -> None:
        for k in keys:
            if k not in acts:
                raise KeyError(f"{self.game}: missing action {k!r} (give the decoder's sample() "
                               f"output or its replay format {sorted(self.ACTION_KEYS.values())})")
            if acts[k].shape[0] != self.num_rows:
                raise ValueError(f"{self.game}: action {k!r} has {acts[k].shape[0]} rows, "
                                 f"expected {self.num_rows}")

    # -- checkpointing ---------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """The env RNGs (episodes restart at iteration boundaries, so the
        RNGs are the only state a full-state resume needs)."""
        return {"rng": self._rng.bit_generator.state, "torch_rng": self._torch_rng.get_state()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.bit_generator.state = state["rng"]
        if "torch_rng" in state:
            self._torch_rng.set_state(state["torch_rng"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _make_generals(n_envs: int = 1, **kw) -> VecGameEnv:
    from env.vec_generals import GeneralsVecEnv
    return GeneralsVecEnv(n_envs, **kw)


def _make_othello(n_envs: int = 1, **kw) -> VecGameEnv:
    from env.vec_othello import OthelloVecEnv
    return OthelloVecEnv(n_envs, **kw)


def _make_gomoku(n_envs: int = 1, **kw) -> VecGameEnv:
    from env.vec_gomoku import GomokuVecEnv
    return GomokuVecEnv(n_envs, **kw)


def _make_lux(n_envs: int = 1, **kw) -> VecGameEnv:
    """Needs the ``lux`` extra (the official luxai-s3 env, JAX)."""
    from env.vec_lux import LuxVecEnv
    return LuxVecEnv(n_envs, **kw)


def _make_arc3(n_envs: int = 1, **kw) -> VecGameEnv:
    """Needs the ``arc3`` extra (the offline ARC-AGI-3 SDK) and ENVIRONMENTS_DIR."""
    from env.vec_arc3 import Arc3VecEnv
    return Arc3VecEnv(n_envs, **kw)


def _make_mahjong(n_envs: int = 1, **kw) -> VecGameEnv:
    """Riichi mahjong hanchan games (gmsim's MahjongSim: riichienv-core + the converter's
    state)."""
    from env.vec_mahjong import MahjongVecEnv
    return MahjongVecEnv(n_envs, **kw)


def _make_pokemon(n_envs: int = 1, **kw) -> VecGameEnv:
    """Pokémon Showdown battles (Showdown's simulator in Node workers; frames from the
    converter's parser)."""
    from env.vec_pokemon import PokemonVecEnv
    return PokemonVecEnv(n_envs, **kw)


def _make_clash_royale(n_envs: int = 1, **kw) -> VecGameEnv:
    """Clash Royale battles in RoyaleSim (``import royalesim``), frames from crdata's builder."""
    from env.vec_cr import ClashRoyaleVecEnv
    return ClashRoyaleVecEnv(n_envs, **kw)


def _make_sc2(n_envs: int = 1, **kw) -> VecGameEnv:
    """StarCraft II against the built-in AI (gmsim's Sc2Sim: headless SC2 processes of the
    3.16.1 Linux client; frames from gmsc2, the converter's)."""
    from env.vec_sc2 import Sc2VecEnv
    return Sc2VecEnv(n_envs, **kw)


def _make_hok(n_envs: int = 1, **kw) -> VecGameEnv:
    """Needs Tencent's Honor of Kings research environment (HOK_ROOT: the SDK venv, Wine,
    the licensed gamecore)."""
    from env.vec_hok import HokVecEnv
    return HokVecEnv(n_envs, **kw)


#: games with a simulator (the others train on their IL replay stream only)
VEC_ENV_REGISTRY: Dict[str, Callable[..., VecGameEnv]] = {
    "generals": _make_generals,
    "othello": _make_othello,
    "gomoku": _make_gomoku,
    "lux": _make_lux,
    "arc3": _make_arc3,
    "hok": _make_hok,
    "mahjong": _make_mahjong,
    "pokemon": _make_pokemon,
    "sc2": _make_sc2,
    "clash_royale": _make_clash_royale,
}


def _make_gmsim(game: str, n_envs: int = 1, **kw) -> VecGameEnv:
    """A gmrules game (gmsim simulator, env/vec_gmsim.py)."""
    from env.vec_gmsim import GmSimVecEnv
    return GmSimVecEnv(game, n_envs, **kw)


def _register_gmsim_envs() -> None:
    """Every simulated game of model/games/gmrules/ (discovered: no edit here per
    game)."""
    from model.games.gmrules import rules_games

    for name, g in rules_games().items():
        if g.sim:
            if name in VEC_ENV_REGISTRY:
                raise ValueError(f"gmrules game {name!r} clashes with a registered env")
            VEC_ENV_REGISTRY[name] = functools.partial(_make_gmsim, name)


_register_gmsim_envs()


def register_vec_env(game: str, builder: Callable[..., VecGameEnv]) -> None:
    if game in VEC_ENV_REGISTRY:
        raise ValueError(f"vec env {game!r} already registered")
    VEC_ENV_REGISTRY[game] = builder


def has_simulator(game: str) -> bool:
    return game in VEC_ENV_REGISTRY


def make_vec_env(game: str, n_envs: int = 1, **kwargs) -> VecGameEnv:
    """``n_envs`` games of ``game`` (kwargs go to the adapter)."""
    try:
        builder = VEC_ENV_REGISTRY[game]
    except KeyError:
        raise KeyError(f"no simulator for {game!r}; have {sorted(VEC_ENV_REGISTRY)}") from None
    return builder(n_envs, **kwargs)


def __getattr__(name: str):
    """Lazy re-exports of the adapters (importing one never pays for the other)."""
    if name == "GeneralsVecEnv":
        from env.vec_generals import GeneralsVecEnv
        return GeneralsVecEnv
    if name == "OthelloVecEnv":
        from env.vec_othello import OthelloVecEnv
        return OthelloVecEnv
    if name == "GomokuVecEnv":
        from env.vec_gomoku import GomokuVecEnv
        return GomokuVecEnv
    if name == "LuxVecEnv":
        from env.vec_lux import LuxVecEnv
        return LuxVecEnv
    if name in ("GmSimVecEnv", "CandidateVecEnv"):
        from env import vec_gmsim
        return getattr(vec_gmsim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VecObs", "VecGameEnv", "VecEnvBase", "StepResult", "VEC_ENV_REGISTRY",
           "register_vec_env", "has_simulator", "make_vec_env"]
