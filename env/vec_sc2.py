"""StarCraft II as a :class:`~env.vec_game.VecGameEnv`: N games of one player against the
game's built-in AI, stepped together by gmsim's ``Sc2Sim`` (data_processing/rs/gmsc2 ``env``:
an SC2 process per env on its own thread) — the decisions and frames of gmsc2's module docs,
so an RL frame IS the IL frame (``gmdata sc2 convert``) of the same game state.

A decision is one raw unit command on the window's frame, or "done" (action_type 0): the
command goes to the game at once, the frame stays until the window is done (or holds
K_MAX = 16 commands), then the game advances 8 loops.  Reward 0 until the end: +1 victory,
−1 defeat, 0 a tie or the ``max_loops`` timeout (truncated).

    obs     the shards' obs tensors (model/games/sc2.py): units [U, 27] uint8, unit_xy
            [U, 2], unit_type / unit_order [U], unit_buffs [U, 2] (int32 here: uint16 in
            the shards), unit_mask / unit_own [U] bool, unit_commanded [U], map_bits
            [9, 64, 64], globals [21], upgrades [NU] float32, last_action [] long,
            window_start [] bool
    legal   none: the decoder masks from the obs (own units, occupied slots) and the
            vocabulary's target kinds
    action  action_type, actors [U] bool, target_kind, target_unit, target_point, queued

The game files: ``install`` (default ``$GM_SC2_INSTALL``, else ~/gm-data/sc2/3.16.1/StarCraftII)
with the build of ``vocab`` (default data_processing/games/sc2/sc2_vocab_55958.json); maps
default to the 2017 Season 3 ladder (the high-MMR replays' maps).  Linux only (the
headless client); the Blizzard AI and Machine Learning licence: research use.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VOCAB = os.path.join(ROOT, "data_processing", "games", "sc2", "sc2_vocab_55958.json")
U = 320
F_UNIT = 27
MAP = 64
MAP_BITS = 9
G = 21
LOC = 256
RACES = {"terran": 1, "zerg": 2, "protoss": 3, "random": 4}
DIFFICULTIES = {"very_easy": 1, "easy": 2, "medium": 3, "medium_hard": 4, "hard": 5, "harder": 6,
                "very_hard": 7, "cheat_vision": 8, "cheat_money": 9, "cheat_insane": 10}


def default_install() -> str:
    return os.environ.get("GM_SC2_INSTALL") or os.path.expanduser("~/gm-data/sc2/3.16.1/StarCraftII")


def ladder_maps(install: str, season: str = "Ladder2017Season3") -> list:
    return sorted(glob.glob(os.path.join(install, "Maps", season, "*.SC2Map")))


def _code(v, table) -> int:
    return int(table[v]) if isinstance(v, str) else int(v)


class Sc2VecEnv(VecEnvBase):
    """N StarCraft II games against the built-in AI (see the module docstring)."""

    ACTION_KEYS = {"action_type": "action_type", "action_actors": "actors", "action_target_kind": "target_kind",
                   "action_target_unit": "target_unit", "action_target_point": "target_point",
                   "action_queued": "queued"}
    game = "sc2"

    def __init__(self, n_envs: int = 1, *, autoreset: bool = False, seed: Optional[int] = 0,
                 install: Optional[str] = None, vocab: Optional[str] = None, maps: Optional[Sequence[str]] = None,
                 races: Sequence = ("terran", "zerg", "protoss"), ai_race="random", difficulty="medium",
                 max_loops: int = 22 * 60 * 30, port: Optional[int] = None):
        super().__init__(n_envs, autoreset, seed)
        from gmsim import _native

        install = install or default_install()
        maps = list(maps) if maps else ladder_maps(install)
        if not maps:
            raise RuntimeError(f"sc2: no maps under {install}/Maps (pass maps=)")
        if port is None:
            port = int(os.environ.get("GM_SC2_PORT", 25000 + (os.getpid() % 500) * 40))
        seeds = [int(s) for s in self._episode_seeds(self._n, seed)]
        self._sim = _native.Sc2Sim(self._n, install, vocab or DEFAULT_VOCAB, maps,
                                   [_code(r, RACES) for r in races], _code(ai_race, RACES),
                                   _code(difficulty, DIFFICULTIES), int(max_loops), int(port), seeds, bool(autoreset))
        self.n_upgrades = int(self._sim.n_upgrades)
        self._step_idx = np.zeros(self._n, dtype=np.int64)
        self._new = np.ones(self._n, dtype=bool)
        self._done = np.zeros(self._n, dtype=bool)

    @property
    def failures(self) -> int:
        """Games stopped on an SC2 error (restarted; 0 expected)."""
        return int(self._sim.failures)

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        if n is not None and n != self._n:
            raise ValueError("sc2: the number of games is fixed at construction")
        ids = self._env_ids(env_ids)
        self._sim.reset(ids.tolist())
        self._step_idx[ids] = 0
        self._new[ids] = True
        self._done[ids] = False
        return self.observe()

    def obs_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {"units": ((U, F_UNIT), torch.uint8), "unit_xy": ((U, 2), torch.int32),
                "unit_type": ((U,), torch.int32), "unit_order": ((U,), torch.int32),
                "unit_buffs": ((U, 2), torch.int32), "unit_mask": ((U,), torch.bool), "unit_own": ((U,), torch.bool),
                "unit_commanded": ((U,), torch.float32), "map_bits": ((MAP_BITS, MAP, MAP), torch.float32),
                "globals": ((G,), torch.float32), "upgrades": ((self.n_upgrades,), torch.float32),
                "last_action": ((), torch.long), "window_start": ((), torch.bool)}

    def legal_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {}

    # ------------------------------------------------------------------

    def observe(self) -> VecObs:
        o = self._sim.observe()
        n = self._n

        def view(key, dtype, shape, to=None):
            a = np.frombuffer(o[key], dtype=dtype).reshape((n,) + shape)
            return torch.from_numpy(a.astype(to) if to is not None else a.copy())

        obs = {
            "units": view("units", np.uint8, (U, F_UNIT)),
            "unit_xy": view("unit_xy", np.uint16, (U, 2), np.int32),
            "unit_type": view("unit_type", np.uint16, (U,), np.int32),
            "unit_order": view("unit_order", np.uint16, (U,), np.int32),
            "unit_buffs": view("unit_buffs", np.uint16, (U, 2), np.int32),
            "unit_mask": view("unit_mask", np.uint8, (U,)).bool(),
            "unit_own": view("unit_own", np.uint8, (U,)).bool(),
            "unit_commanded": view("unit_commanded", np.uint8, (U,), np.float32),
            "map_bits": view("map_bits", np.uint8, (MAP_BITS, MAP, MAP), np.float32),
            "globals": view("globals", np.float32, (G,)),
            "upgrades": view("upgrades", np.uint8, (self.n_upgrades,), np.float32),
            "last_action": view("last_action", np.int64, ()),
            "window_start": view("window_start", np.uint8, ()).bool(),
        }
        active = torch.from_numpy(~self._done)
        return VecObs(obs=obs, legal={}, to_play=torch.zeros(n, dtype=torch.long),
                      env_id=torch.arange(n, dtype=torch.long), stream_id=torch.arange(n, dtype=torch.long),
                      step_idx=torch.from_numpy(self._step_idx.copy()), new_episode=torch.from_numpy(self._new.copy()),
                      active=active, forced=torch.zeros(n, dtype=torch.bool),
                      stream_seats=torch.ones(n, 1, dtype=torch.bool))

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        acts = self.replay_actions(actions)
        self._require(acts, ("action_type", "actors", "target_kind", "target_unit", "target_point", "queued"))
        n = self._n

        def i64(k):
            return acts[k].detach().cpu().numpy().astype(np.int64).reshape(n)

        actors = acts["actors"].detach().cpu().numpy().astype(bool).reshape(n, U).astype(np.uint8)
        queued = acts["queued"].detach().cpu().numpy().astype(bool).reshape(n).astype(np.uint8)
        info = self._sim.step(i64("action_type").tobytes(), actors.tobytes(), i64("target_kind").tobytes(),
                              i64("target_unit").tobytes(), i64("target_point").tobytes(), queued.tobytes())

        def arr(key, dtype):
            return np.frombuffer(info[key], dtype=dtype).reshape(n).copy()

        rewards = arr("rewards", np.float32)
        done = arr("done", np.uint8).astype(bool)
        truncated = arr("truncated", np.uint8).astype(bool)
        was_active = ~self._done
        self._step_idx += 1
        self._new[:] = False
        if self.autoreset:
            # a finished game was replaced: the row's next frame starts a new episode
            self._new[done] = True
            self._step_idx[done] = 0
        else:
            self._done |= done
        out: Dict[str, Any] = {
            "terminated": torch.from_numpy(done & ~truncated),
            "truncated": torch.from_numpy(truncated),
            "winner": torch.from_numpy(np.where(done & (rewards > 0), 0, -1).astype(np.int64)),
            "episode_len": torch.from_numpy(arr("episode_len", np.int64) * done),
            "acted": torch.from_numpy(was_active),
            "applied": {k: v for k, v in acts.items()},
            "illegal": torch.zeros(n, dtype=torch.bool),
            "sent": torch.from_numpy(arr("sent", np.uint8).astype(bool)),
            "refused": torch.from_numpy(arr("refused", np.uint8).astype(bool)),
            "game_loop": torch.from_numpy(arr("game_loop", np.int64)),
            "failures": self.failures,
        }
        if truncated.any() and self._sim.last_error:
            out["error"] = self._sim.last_error
        return self.observe(), torch.from_numpy(rewards).reshape(n, 1), torch.from_numpy(done), out

    def random_actions(self, obs: VecObs, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """A random decision per env: done half the time, else Smart (right-click) with a
        random own unit at a random point."""
        gen = self._generator(generator)
        n = self._n
        own = obs.obs["unit_own"]
        cmd = (torch.rand(n, generator=gen) < 0.5) & own.any(-1)
        pick = torch.multinomial(own.float() + 1e-9, 1, generator=gen).squeeze(1)
        actors = torch.zeros(n, U, dtype=torch.bool)
        actors[torch.arange(n), pick] = True
        actors &= cmd.unsqueeze(-1)
        smart = 2                                           # the vocabulary's first action: Smart
        return {"action_type": torch.where(cmd, torch.full((n,), smart), torch.zeros(n, dtype=torch.long)),
                "actors": actors, "target_kind": cmd.long(), "target_unit": torch.full((n,), -1),
                "target_point": torch.where(cmd, torch.randint(0, LOC * LOC, (n,), generator=gen), torch.full((n,), -1)),
                "queued": torch.zeros(n, dtype=torch.bool)}
