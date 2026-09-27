"""Riichi mahjong as a :class:`~env.vec_game.VecGameEnv`: N hanchan games (Tenhou rules)
stepped together by gmsim's ``MahjongSim`` (data_processing/rs/gmmahjong: riichienv-core's
engine deals, rules and scores; every mjai event it logs feeds the converter's per-kyoku
state), so an RL frame IS the IL frame of the same position — gmdata's `mahjong convert`
frame, with its exact legality (the Rust test ``mahjong::sim_parity`` replays Houou games
through the simulator frame for frame).

Rows / streams (the per-seat convention of env/vec_gmsim.py): row ``env * 4 + seat`` is that
seat's own stream, active when the seat owes a decision — its own turn (after a draw or a
call) or a claim window on another seat's discard / kakan in which it may do more than pass
(the seats of one window act on the same step, then the engine resolves the calls by
priority).  ``forced`` = a single legal action (applied whatever is sent).  Rows are not in
lockstep (``lockstep = False``: the RL trainer's ``--actor stream``).

    obs     tiles_seq   [34, 37] f32   the tile planes (model/games/mahjong.py)
            players_seq [4, 12]  f32
            table_seq   [12]     f32
    legal   action_type [8]      bool  dahai riichi chi pon kan hora ryukyoku pass
            tile_cond   [8, 34]  bool  the tile arguments per type (all tiles for the
                                       tile-less types, which the decoder never scores)
    action  action_type [R] long, tile_idx [R] long

Reward: the final placement's value (1st +1, 2nd +1/3, 3rd −1/3, 4th −1 — the IL value
target) paid to every seat on the step the hanchan ends; ``info["returns"]`` holds the
final points.  Episode seeds come from ``reset(seed=...)`` (the engine's walls follow from
them).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

N_SEATS = 4
N_TYPES = 8
N_KINDS = 34
TILE_FEATURES = 37
OBS = {"tiles_seq": (N_KINDS, TILE_FEATURES), "players_seq": (4, 12), "table_seq": (12,)}


class MahjongVecEnv(VecEnvBase):
    """N riichi mahjong hanchan games (see the module docstring)."""

    ACTION_KEYS = {"action_type": "action_type", "tile_idx": "tile_idx"}
    rows_per_env = N_SEATS
    lockstep = False
    n_seats = N_SEATS

    def __init__(self, n_envs: int = 1, *, autoreset: bool = False, seed: Optional[int] = 0,
                 threads: Optional[int] = None):
        super().__init__(n_envs, autoreset, seed)
        from gmsim import _native

        if threads is None:
            threads = int(os.environ.get("GM_SIM_THREADS", "1"))
        self.threads = int(threads)
        self._sim = _native.MahjongSim(self._n, [], bool(autoreset), self.threads)
        self.reset()

    # ------------------------------------------------------------------

    @property
    def finished(self) -> torch.Tensor:
        return torch.tensor([self._sim.done(e) for e in range(self._n)], dtype=torch.bool)

    @property
    def failures(self) -> int:
        """Games stopped on an engine / legality disagreement (a bug: 0 expected)."""
        return int(self._sim.failures)

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        if seed is not None:
            self._seed_rngs(seed)
        if n is not None and n != self._n:
            if env_ids is not None:
                raise ValueError("resize (n) and env_ids are exclusive")
            self._n = int(n)
            self._sim.resize(self._n)
        ids = self._env_ids(env_ids)
        self._sim.reset(ids.tolist(), self._episode_seeds(len(ids), seed))
        return self.observe()

    def obs_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {k: (shape, torch.float32) for k, shape in OBS.items()}

    def legal_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {"action_type": ((N_TYPES,), torch.bool), "tile_cond": ((N_TYPES, N_KINDS), torch.bool)}

    def scores(self, env: int):
        """Env ``env``'s current (final, when done) points per seat."""
        return self._sim.scores(env)

    # ------------------------------------------------------------------

    def observe(self) -> VecObs:
        o = self._sim.observe()
        R = self._sim.n_rows

        def view(key, dtype, shape):
            return torch.from_numpy(np.frombuffer(o[key], dtype=dtype).reshape((R,) + shape).copy())

        obs = {k: view(k, np.float32, shape) for k, shape in OBS.items()}
        legal = {"action_type": view("action_type", np.uint8, (N_TYPES,)).bool(),
                 "tile_cond": view("tile_cond", np.uint8, (N_TYPES, N_KINDS)).bool()}
        rows = torch.arange(R, dtype=torch.long)
        return VecObs(
            obs=obs,
            legal=legal,
            to_play=view("to_play", np.int64, ()),
            env_id=rows // N_SEATS,
            stream_id=rows,
            step_idx=view("step_idx", np.int64, ()),
            new_episode=view("new_episode", np.uint8, ()).bool(),
            active=view("active", np.uint8, ()).bool(),
            forced=view("forced", np.uint8, ()).bool(),
            stream_seats=torch.nn.functional.one_hot(rows % N_SEATS, N_SEATS).bool(),
        )

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        acts = self.replay_actions(actions)
        self._require(acts, ("action_type", "tile_idx"))
        R = self._sim.n_rows
        at = acts["action_type"].detach().cpu().numpy().astype(np.int64).reshape(-1)
        tl = acts["tile_idx"].detach().cpu().numpy().astype(np.int64).reshape(-1)
        if at.shape[0] != R or tl.shape[0] != R:
            raise ValueError(f"mahjong: {at.shape[0]} / {tl.shape[0]} actions for {R} rows")
        info = self._sim.step(at.tobytes(), tl.tobytes())
        n = self._n

        def arr(key, dtype, shape):
            return torch.from_numpy(np.frombuffer(info[key], dtype=dtype).reshape(shape).copy())

        rewards = arr("rewards", np.float32, (n, N_SEATS))
        done = arr("done", np.uint8, (n,)).bool()
        out: Dict[str, Any] = {
            "terminated": arr("terminated", np.uint8, (n,)).bool(),
            "truncated": arr("truncated", np.uint8, (n,)).bool(),
            "winner": arr("winner", np.int64, (n,)),
            "episode_len": arr("episode_len", np.int64, (n,)),
            "acted": arr("acted", np.uint8, (R,)).bool(),
            "applied": {"action_type": arr("applied_type", np.int64, (R,)).clamp(min=0),
                        "tile_idx": arr("applied_tile", np.int64, (R,)).clamp(min=0)},
            "illegal": torch.zeros(n, dtype=torch.bool),
            "returns": arr("returns", np.float64, (n, N_SEATS)),
            "failures": self.failures,
        }
        if out["truncated"].any() and self._sim.last_error:
            out["error"] = self._sim.last_error
        return self.observe(), rewards, done, out

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """A uniformly random legal (type, tile) per row of ``obs`` (inactive rows: 0)."""
        gen = self._generator(generator)
        t_mask = obs.legal["action_type"].float()
        t_mask[t_mask.sum(-1) == 0, 0] = 1.0
        t = torch.multinomial(t_mask, 1, generator=gen).squeeze(1)
        tiles = obs.legal["tile_cond"][torch.arange(t.shape[0]), t].float()
        tiles[tiles.sum(-1) == 0, 0] = 1.0
        k = torch.multinomial(tiles, 1, generator=gen).squeeze(1)
        return {"action_type": t, "tile_idx": k}
