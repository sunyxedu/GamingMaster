"""generals.io as a :class:`~env.vec_game.VecGameEnv` over :class:`env.generals_env.GeneralsEnv`.

Simultaneous moves: two rows per env, row ``2 * env + seat`` = that seat's own
fog-of-war stream (``stream_seats`` one-hot), one frame per turn.  Frame t is
the state after turn t — what the IL records hold: the replay simulator's
state after the turn's moves and army growth (``gmdata generals simulate``),
frame 0 the initial map, and the frame-t action is executed in the following
env step (the gio target of frame t is the move of replay turn t).

Observation = the gm-shard / gio record schema (data_processing/games/generals.py):

  grid_seq               float32 [12, H, W]  11 one-hot tile states + log1p(army)/10
  globals_seq            float32 [G]         log1p(own/opp army, land)/10 (G = 4);
                                             G = 7 with ``turn_globals`` — the
                                             (turn % 50)/50, turn % 2, turn/500
                                             features the IL trainers append
                                             (model.games.generals.append_turn_globals)
  legal_source_mask_seq  bool    [1, H, W]   own tiles with army > 1

``memory_vision`` (sighted mountains stay encoded as mountains after vision is
lost) is OFF by default: the IL data has no such memory, and the default
observation is bit-identical to it (tests/test_vec_game_golden.py).  A
network's setting is its ``cfg.memory_vision`` (training.train_rl_joint's
GeneralsSpec builds the env from it); the training/train_rl.py runs of
2026-08-11 until that field existed played with it on (see
model.config.legacy_memory_vision).

legal = {"source": [1, H, W], "direction": [4, H, W]} (the direction key only
with ``direction_mask``): the source mask, and the in-bounds, not-a-mountain
directions at the row's own tiles (False elsewhere) — like everything in a
row, a function of that seat's own view (tests/test_audit_generals_leaks.py).
A row with no movable army is ``forced``: the env passes whatever is sent.

Actions: the GeneralsDecoder's sample() output (action_source / _direction /
_split / _pass) or its replay format (source_idx, direction, split, passed).
``info["count"]`` gives the IL target semantics of what happened to each
row's action: 1 executed, 0 pass, −1 sent but dropped by the rules (invalid
destination or a source the opponent's earlier move took).  When both moves
of a tick tie on every priority key, the env executes seat 0's first; the
replay simulator executes the move the replay lists first, so replaying a
recorded game passes ``step(..., tiebreak_first=)`` from
:func:`replay_tiebreak_by_step` to reproduce the IL targets exactly (the
board is the same either way; only which seat is executed / dropped flips).

Rewards are the env's per-seat rewards: by default ``sparse_reward`` (the
Stage-2 recipe, = ``training/train_rl.py --sparse_reward``): ±1 terminal, the
material-scored soft terminal at ``max_turns`` minus ``trunc_penalty``, and
the small invalid-move charge; ``sparse_reward=False`` restores the dense /
city shaping.  ``max_turns`` defaults to 2048 (train_rl.py's rollout length):
lockstep rollouts wait for the longest game.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from env.generals_env import (
    _BLANK,
    _CITY,
    _DIR_DC,
    _DIR_DR,
    _GENERAL,
    _MOUNTAIN,
    GeneralsEnv,
    _log_army,
)
from env.vec_game import StepResult, VecEnvBase, VecObs

N_STATES = 11


def map_from_replay(replay: Mapping[str, Any]) -> Dict[str, Any]:
    """A deserialized replay's map as a GeneralsEnv map-pool entry (player i
    on ``generals[i]``, i.e. seat i = ``usernames[i]``)."""
    if len(replay.get("generals", [])) != 2:
        raise ValueError("only 1v1 replays (two generals) are supported")
    return {
        "replay_id": replay.get("id"),
        "map_w": replay["mapWidth"],
        "map_h": replay["mapHeight"],
        "generals": list(replay["generals"]),
        "cities": list(replay.get("cities", [])),
        "city_armies": list(replay.get("cityArmies", [])),
        "mountains": list(replay.get("mountains", [])),
    }


def direction_of(dr: int, dc: int) -> int:
    """(Δrow, Δcol) of an adjacent move → direction id (0 down, 1 right, 2 up, 3 left)."""
    for d in range(4):
        if (_DIR_DR[d], _DIR_DC[d]) == (dr, dc):
            return d
    raise ValueError(f"not an adjacent move: ({dr}, {dc})")


def replay_actions_by_step(replay: Mapping[str, Any], grid_w: int = 25
                           ) -> Dict[int, Dict[int, Dict[str, int]]]:
    """Recorded moves → ``{env step k: {seat: replay-format action}}``.

    The move of replay turn ``k - 1`` is executed in env step ``k`` (1-based),
    i.e. it is decided on frame ``k - 1`` — the gio target of that frame."""
    w0 = replay["mapWidth"]
    out: Dict[int, Dict[int, Dict[str, int]]] = {}
    for mv in replay["moves"]:
        k, p = mv["turn"] + 1, mv["index"]
        sr, sc = divmod(mv["start"], w0)
        er, ec = divmod(mv["end"], w0)
        if p in out.get(k, {}):
            raise ValueError(f"seat {p} moves twice in replay turn {mv['turn']}")
        out.setdefault(k, {})[p] = {"source_idx": sr * grid_w + sc,
                                    "direction": direction_of(er - sr, ec - sc),
                                    "split": int(bool(mv["is50"])), "passed": False}
    return out


def replay_tiebreak_by_step(replay: Mapping[str, Any]) -> Dict[int, int]:
    """``{env step k: seat}`` for the ticks where both seats move: the seat
    whose move the replay lists first in turn ``k - 1`` — the replay
    simulator's final move-order tiebreak (``GeneralsVecEnv.step``'s
    ``tiebreak_first``)."""
    first: Dict[int, int] = {}
    seats: Dict[int, set] = {}
    for mv in replay["moves"]:
        k = mv["turn"] + 1
        first.setdefault(k, int(mv["index"]))
        seats.setdefault(k, set()).add(int(mv["index"]))
    return {k: p for k, p in first.items() if len(seats[k]) == 2}


def full_state_grids(env: GeneralsEnv,
                     arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                     ) -> torch.Tensor:
    """[2, 12, H, W] fog-free grids (seat 0, seat 1 perspective): every
    in-bounds tile encoded as seen (own tiles as owned, the encoding of
    ``GeneralsEnv._tile_to_state``) with its army — the privileged critic
    input.  Whole-grid numpy (pinned to the per-tile reference by
    tests/test_vec_game.py); ``arrays`` = ``env._state_arrays()`` if the
    caller has them."""
    H, W = env.grid_h, env.grid_w
    terrain, army, occ = arrays if arrays is not None else env._state_arrays()
    inb = np.zeros((H, W), dtype=bool)
    inb[:env.real_h, :env.real_w] = True
    blank, city = terrain == _BLANK, terrain == _CITY
    gen, mtn = terrain == _GENERAL, terrain == _MOUNTAIN
    neutral = occ == -1
    own_state = np.select([blank, city, gen], [5, 6, 7], default=5)
    st = np.empty((2, H, W), dtype=np.int64)
    for pid in (0, 1):
        enemy = ~neutral & (occ != pid)
        seen_state = np.select(
            [blank & neutral, city & neutral, mtn, enemy & blank, enemy & city, enemy & gen],
            [2, 3, 4, 8, 9, 10], default=2)
        st[pid] = np.where(inb, np.where(occ == pid, own_state, seen_state), -1)
    has_army = inb & ~mtn
    army_ch = np.where(has_army, _log_army(np.where(has_army, army, 0)), np.float32(0.0))
    onehot = (st[:, None] == np.arange(N_STATES).reshape(1, -1, 1, 1)).astype(np.float32)
    out = np.concatenate([onehot, np.broadcast_to(army_ch, (2, 1, H, W))], axis=1)
    return torch.from_numpy(np.ascontiguousarray(out))


class GeneralsVecEnv(VecEnvBase):
    """N 1v1 generals games (see the module docstring)."""

    game = "generals"
    n_seats = 2
    rows_per_env = 2
    interleaved = False
    ACTION_KEYS = {"action_source": "source_idx", "action_direction": "direction",
                   "action_split": "split", "action_pass": "passed"}

    def __init__(self, n_envs: int = 1, *, grid_h: int = 25, grid_w: int = 25,
                 max_turns: int = 2048, n_mountains: int = 20, n_cities: int = 5,
                 city_army_range: Tuple[int, int] = (40, 50),
                 turn_globals: bool = False, memory_vision: bool = False,
                 direction_mask: bool = True, privileged: bool = False,
                 sparse_reward: bool = True, trunc_penalty: float = 0.0,
                 env_attrs: Optional[Mapping[str, Any]] = None,
                 map_pool: Optional[List[Dict[str, Any]]] = None,
                 autoreset: bool = False, seed: Optional[int] = 0):
        super().__init__(n_envs, autoreset, seed)
        self.grid_h, self.grid_w = grid_h, grid_w
        self.direction_mask = bool(direction_mask)
        self.privileged = bool(privileged)
        self._env_kwargs = dict(grid_h=grid_h, grid_w=grid_w, max_turns=max_turns,
                                n_mountains=n_mountains, n_cities=n_cities,
                                city_army_range=city_army_range)
        attrs: Dict[str, Any] = {"turn_globals": bool(turn_globals),
                                 "memory_vision": bool(memory_vision), "breakdown_pid": 0,
                                 "trunc_penalty": float(trunc_penalty)}
        if sparse_reward:
            attrs.update(city_reward_scale=0.0, dense_reward_scale=0.0)
        attrs.update(env_attrs or {})
        self._env_attrs = attrs
        self.turn_globals = bool(attrs["turn_globals"])
        self.memory_vision = bool(attrs["memory_vision"])
        self.map_pool = map_pool
        self._alloc(self._n)
        self.reset()

    # ------------------------------------------------------------------

    def _make_env(self) -> GeneralsEnv:
        env = GeneralsEnv(**self._env_kwargs)
        for k, v in self._env_attrs.items():
            setattr(env, k, v)
        env.map_pool = self.map_pool
        return env

    def _alloc(self, n: int) -> None:
        self._n = n
        self.envs: List[GeneralsEnv] = [self._make_env() for _ in range(n)]
        self._obs: List[Optional[Dict[int, Dict[str, torch.Tensor]]]] = [None] * n
        self._priv: List[Optional[torch.Tensor]] = [None] * n
        self.done = np.ones(n, dtype=bool)
        self.new_episode = np.zeros(n, dtype=bool)

    def _start(self, i: int, seed: int, map_data: Optional[Dict[str, Any]]) -> None:
        # GeneralsEnv.reset seeds the global `random` module; keep the caller's
        # global RNG stream untouched (episodes draw from this env's own RNG).
        state = random.getstate()
        try:
            self._obs[i] = self.envs[i].reset(seed=seed, map_data=map_data)
        finally:
            random.setstate(state)
        self._set_privileged(i)
        self.done[i] = False
        self.new_episode[i] = True

    def _set_privileged(self, i: int) -> None:
        """The privileged view of env i's current frame (from the state
        arrays its observation was built from)."""
        if self.privileged:
            e = self.envs[i]
            self._priv[i] = full_state_grids(e, getattr(e, "obs_state_arrays", None))

    @property
    def finished(self) -> torch.Tensor:
        return torch.from_numpy(self.done.copy())

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None,
              maps: Optional[Sequence[Optional[Dict[str, Any]]]] = None) -> VecObs:
        """New episodes for all envs (``n`` resizes) or only ``env_ids``.

        ``maps[k]`` (a map-pool entry, e.g. :func:`map_from_replay`) pins the
        k-th reset env's map with seat i on ``generals[i]``; otherwise a random
        map-pool map (random seat swap) or a random map.  ``seed`` makes the
        k-th env's episode seed ``seed + k``."""
        if n is not None and n != self._n:
            if env_ids is not None:
                raise ValueError("resize (n) and env_ids are exclusive")
            self._alloc(int(n))
        ids = self._env_ids(env_ids)
        if maps is not None and len(maps) != len(ids):
            raise ValueError(f"{len(maps)} maps for {len(ids)} envs")
        seeds = self._episode_seeds(len(ids), seed)
        self.new_episode[:] = False
        for k, i in enumerate(ids):
            self._start(int(i), seeds[k], maps[k] if maps is not None else None)
        return self.observe()

    def obs_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        H, W = self.grid_h, self.grid_w
        return {"grid_seq": ((12, H, W), torch.float32),
                "globals_seq": ((7 if self.turn_globals else 4,), torch.float32),
                "legal_source_mask_seq": ((1, H, W), torch.bool)}

    # ------------------------------------------------------------------

    def observe(self) -> VecObs:
        n, R = self._n, self.num_rows
        per_row = [self._obs[i][p] for i in range(n) for p in (0, 1)]
        grid = torch.stack([o["grid"] for o in per_row])
        glb = torch.stack([o["globals"] for o in per_row])
        src = torch.stack([o["legal_mask"] for o in per_row])
        active = torch.from_numpy(np.repeat(~self.done, 2))
        forced = active & ~src.flatten(1).any(1)
        legal = {"source": src.clone()}
        if self.direction_mask:
            legal["direction"] = torch.stack([o["dir_mask"] for o in per_row])
        seat = torch.arange(R, dtype=torch.long) % 2
        turn = torch.tensor([self.envs[i].turn for i in range(n)], dtype=torch.long)
        privileged = None
        if self.privileged:
            privileged = {"grid_full": torch.cat(self._priv)}
        return VecObs(
            obs={"grid_seq": grid, "globals_seq": glb, "legal_source_mask_seq": src},
            legal=legal,
            to_play=seat,
            env_id=self._row_env(),
            stream_id=torch.arange(R, dtype=torch.long),
            step_idx=turn.repeat_interleave(2),
            new_episode=torch.from_numpy(np.repeat(self.new_episode, 2)) & active,
            active=active,
            forced=forced,
            stream_seats=torch.nn.functional.one_hot(seat, 2).bool(),
            privileged=privileged,
            aux={"capture_label": torch.stack([o["capture_label"] for o in per_row]),
                 "capture_mask": torch.stack([o["capture_mask"] for o in per_row])},
        )

    def step(self, actions: Mapping[str, torch.Tensor], *,
             tiebreak_first: Optional[Sequence[int]] = None) -> StepResult:
        """One tick of every active env.  ``tiebreak_first`` ([N] seats,
        replay only): the seat whose move executes first when both moves tie
        on every priority key (default seat 0; see the module docstring)."""
        n, R = self._n, self.num_rows
        tb = None
        if tiebreak_first is not None:
            tb = [int(x) for x in torch.as_tensor(tiebreak_first).reshape(-1).tolist()]
            if len(tb) != n or any(x not in (0, 1) for x in tb):
                raise ValueError(f"tiebreak_first needs one seat (0/1) per env ({n})")
        acts = self.replay_actions(actions)
        self._require(acts, ("source_idx", "direction", "split", "passed"))
        src = acts["source_idx"].long().tolist()
        dirn = acts["direction"].long().tolist()
        split = acts["split"].long().tolist()
        passed = acts["passed"].bool().tolist()

        rewards = np.zeros((n, 2), dtype=np.float32)
        ended = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        winner = np.full(n, -1, dtype=np.int64)
        episode_len = np.zeros(n, dtype=np.int64)
        acted = np.zeros(R, dtype=bool)
        count = np.zeros(R, dtype=np.int64)
        illegal = np.zeros(R, dtype=bool)
        env_info: List[Optional[Dict[str, Any]]] = [None] * n
        self.new_episode[:] = False
        for i in np.flatnonzero(~self.done):
            env = self.envs[i]
            sent: List[Optional[Dict[str, int]]] = []
            for p in (0, 1):
                r = 2 * i + p
                movable = bool(self._obs[i][p]["legal_mask"].any())
                sent.append(None if passed[r] or not movable else
                            {"source_idx": src[r], "direction": dirn[r], "split": split[r]})
            if tb is not None:
                env.tiebreak_first = tb[i]
            try:
                pair, rew, done, info = env.step(sent[0], sent[1])
            finally:
                env.tiebreak_first = 0
            executed = {mv["player"] for mv in env._move_trace}
            for p in (0, 1):
                r = 2 * i + p
                acted[r] = True
                count[r] = 0 if sent[p] is None else (1 if p in executed else -1)
                illegal[r] = bool(info["illegal"][p])
            rewards[i] = (rew[0], rew[1])
            self._obs[i] = pair
            self._set_privileged(i)
            env_info[i] = info
            if done:
                ended[i] = True
                truncated[i] = bool(info.get("truncated", False))
                winner[i] = -1 if env.winner is None else int(env.winner)
                episode_len[i] = env.turn
                self.done[i] = True

        acted_t = torch.from_numpy(acted)
        zero = torch.zeros(R, dtype=torch.long)
        applied = {
            "source_idx": torch.where(acted_t, torch.tensor(src, dtype=torch.long), zero),
            "direction": torch.where(acted_t, torch.tensor(dirn, dtype=torch.long), zero),
            "split": torch.where(acted_t, torch.tensor(split, dtype=torch.long), zero),
            "passed": acted_t & torch.from_numpy(count == 0),
        }
        info_out: Dict[str, Any] = {
            "terminated": torch.from_numpy(ended & ~truncated),
            "truncated": torch.from_numpy(truncated),
            "winner": torch.from_numpy(winner),
            "episode_len": torch.from_numpy(episode_len),
            "acted": acted_t,
            "applied": applied,
            "count": torch.from_numpy(count),
            "illegal": torch.from_numpy(illegal),
            "env_info": env_info,
        }
        if self.autoreset:
            e_idx = np.flatnonzero(ended)
            seeds = self._episode_seeds(len(e_idx), None)
            for k, i in enumerate(e_idx):
                self._start(int(i), seeds[k], None)
        return self.observe(), torch.from_numpy(rewards), torch.from_numpy(ended), info_out

    # ------------------------------------------------------------------

    def random_actions(self, obs: VecObs, generator: Optional[torch.Generator] = None,
                       pass_prob: float = 0.0) -> Dict[str, torch.Tensor]:
        """Uniform legal source, uniform valid direction there, uniform split;
        forced rows pass, others with probability ``pass_prob``."""
        R = obs.n_rows
        generator = self._generator(generator)
        src_mask = obs.legal["source"].flatten(1)
        w = src_mask.float()
        w[~src_mask.any(1)] = 1.0
        src = torch.multinomial(w, 1, generator=generator).squeeze(1)
        if "direction" in obs.legal:
            dm = obs.legal["direction"].flatten(2)
            dm = dm.gather(2, src.view(R, 1, 1).expand(R, dm.shape[1], 1)).squeeze(2)
        else:
            dm = torch.ones(R, 4, dtype=torch.bool)
        wd = dm.float()
        wd[~dm.any(1)] = 1.0
        dirn = torch.multinomial(wd, 1, generator=generator).squeeze(1)
        split = torch.randint(0, 2, (R,), generator=generator)
        passed = obs.forced | (torch.rand(R, generator=generator) < pass_prob)
        return {"source_idx": src, "direction": dirn, "split": split, "passed": passed}
