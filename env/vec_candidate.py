"""Candidate-action games (gmrules) as :class:`~env.vec_game.VecGameEnv`: a generic
wrapper of the batched Rust simulator ``gmsim.Sim`` — any game registered in
gmrules (and in model/games/candidates/) is an RL environment with no per-game
Python code.

Frames are gmrules ``encode_frame`` output — the encoder of the gmdata converter —
so an RL frame IS the IL frame of the same position (the obs schema of the
game's gm-shards: ``entities``, ``entity_mask``, ``entity_type`` (when the game
has entity types), ``globals``, ``actions``, ``action_mask``).

Rows / streams / seats (see env/vec_game.py), from the game's FrameSpec layout:

  * per-seat (hidden-information or simultaneous games): ``rows_per_env =
    n_seats``, row ``env * n_seats + seat`` is that seat's OWN stream — its own
    decision frames only, so no stream ever carries another seat's private
    information in its temporal state; ``stream_seats`` = the row's seat.  A row
    is ``active`` only when its seat is to act (alternating games: one row per
    env and step; simultaneous turns: several), and its ``step_idx`` counts its
    own frames — rows are therefore NOT in lockstep (``lockstep = False``: the
    RL trainer's ``--actor stream``, which batches rows by frame index; the
    fixed-shape actor needs lockstep).  ``new_episode`` flags every row of an env
    on the step its episode started, acting or not (a seat that acts later
    starts its stream at step_idx 0);
  * interleaved (perfect-information alternating games): one row per env, every
    decision frame from the mover's view, ``stream_seats`` all True (the board
    games' layout; ``lockstep = True``).

``legal = {"action": action_mask}``; ``forced`` = a single legal candidate (the
env applies it whatever is sent).  Actions: ``action_idx`` [R] (the
CandidateDecoder's ``sample()`` key = its replay key), the index of the chosen
candidate; an index outside the legal candidates of an acting row raises.
Rewards [N, n_seats] are the game's outcome (in [-1, 1], = the IL value target)
paid on the step the episode ends; ``info["returns"]`` holds the raw payoffs.
Chance events come from per-episode seeds (``reset(seed=...)``: seed + k).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import gmsim
from env.vec_game import StepResult, VecEnvBase, VecObs


class CandidateVecEnv(VecEnvBase):
    """N episodes of gmrules game ``game`` (see the module docstring).

    ``config`` / extra keyword arguments: the gmrules game config (JSON-able;
    defaults: the game's ``sim_config`` in model/games/candidates/, then the Rust
    defaults), e.g. ``CandidateVecEnv("kuhn_poker", 8, players=3)``.
    ``max_turns`` truncates episodes (zero reward) before the game's own bound.
    """

    ACTION_KEYS = {"action_idx": "action_idx"}

    def __init__(self, game: str, n_envs: int = 1, *, autoreset: bool = False,
                 seed: Optional[int] = 0, max_turns: Optional[int] = None,
                 config: Optional[Mapping[str, Any]] = None, **config_kw):
        super().__init__(n_envs, autoreset, seed)
        from model.games.candidates import candidate_games

        entry = candidate_games().get(game)
        cfg = dict(entry.sim_config) if entry is not None else {}
        cfg.update(config or {})
        cfg.update(config_kw)
        self.game = game
        self.config = cfg
        self.max_turns = max_turns
        self._sim = gmsim.Sim(game, cfg, self._n, autoreset=bool(autoreset), max_turns=max_turns)
        self.spec: Dict[str, Any] = dict(self._sim.spec)
        self.n_seats = int(self.spec["n_seats"])
        self.layout = self.spec["layout"]
        self.interleaved = self.layout == "interleaved"
        self.rows_per_env = 1 if self.interleaved else self.n_seats
        self.lockstep = self.interleaved
        self._has_types = int(self.spec["entity_types"]) > 0
        self.reset()

    # ------------------------------------------------------------------

    @property
    def finished(self) -> torch.Tensor:
        return torch.tensor([self._sim.done(e) for e in range(self._n)], dtype=torch.bool)

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        """New episodes for all envs (resizing to ``n``) or only ``env_ids``;
        episode seeds ``seed + k`` when given (the env RNGs are reseeded too),
        else drawn from the env RNG."""
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
        s = self.spec
        E, A = s["max_entities"], s["max_actions"]
        out = {"entities": ((E, s["entity_features"]), torch.float32),
               "entity_mask": ((E,), torch.bool)}
        if self._has_types:
            out["entity_type"] = ((E,), torch.int32)
        out.update({"globals": ((s["global_features"],), torch.float32),
                    "actions": ((A, s["action_features"]), torch.float32),
                    "action_mask": ((A,), torch.bool)})
        return out

    # ------------------------------------------------------------------

    def observe(self) -> VecObs:
        o = self._sim.observe()
        keys = [k for k in gmsim.FRAME_KEYS if k != "entity_type" or self._has_types]
        obs = {k: torch.from_numpy(np.ascontiguousarray(o[k])) for k in keys}
        R = self._sim.n_rows
        rows = torch.arange(R, dtype=torch.long)
        if self.interleaved:
            stream_seats = torch.ones(R, self.n_seats, dtype=torch.bool)
        else:
            stream_seats = torch.nn.functional.one_hot(rows % self.n_seats, self.n_seats).bool()
        return VecObs(
            obs=obs,
            legal={"action": obs["action_mask"].clone()},
            to_play=torch.from_numpy(o["to_play"].copy()),
            env_id=rows // self.rows_per_env,
            stream_id=rows,
            step_idx=torch.from_numpy(o["step_idx"].copy()),
            new_episode=torch.from_numpy(o["new_episode"]),
            active=torch.from_numpy(o["active"]),
            forced=torch.from_numpy(o["forced"]),
            stream_seats=stream_seats,
        )

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        acts = self.replay_actions(actions)
        self._require(acts, ("action_idx",))
        a = acts["action_idx"].detach().cpu().numpy().astype(np.int64).reshape(-1)
        info = self._sim.step(a)
        applied = torch.from_numpy(info["applied"].copy())
        out: Dict[str, Any] = {
            "terminated": torch.from_numpy(info["terminated"]),
            "truncated": torch.from_numpy(info["truncated"]),
            "winner": torch.from_numpy(info["winner"].copy()),
            "episode_len": torch.from_numpy(info["episode_len"].copy()),
            "acted": applied >= 0,                       # [R] rows that acted

            "applied": {"action_idx": applied.clamp(min=0)},
            "illegal": torch.zeros(self._n, dtype=torch.bool),
            "returns": torch.from_numpy(info["returns"].copy()),
        }
        return (self.observe(), torch.from_numpy(info["rewards"].copy()),
                torch.from_numpy(info["done"]), out)

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        legal = obs.legal["action"]
        w = legal.float()
        w[~legal.any(-1)] = 1.0                          # inactive rows
        idx = torch.multinomial(w, 1, generator=self._generator(generator)).squeeze(-1)
        return {"action_idx": idx}

    # ------------------------------------------------------------------

    def trajectory(self, env: int) -> np.ndarray:
        """Env ``env``'s current episode as a gmrules trajectory (int64 [n, 2]
        ``(seat, id)``, seat −1 = chance) — the converters' record format."""
        return self._sim.trajectory(env)

    def legal_ids(self, row: int):
        """Canonical action ids of a row's candidates."""
        return self._sim.legal_ids(row)
