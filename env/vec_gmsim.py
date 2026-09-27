"""gmrules games as :class:`~env.vec_game.VecGameEnv`: a generic wrapper of the
batched Rust simulator ``gmsim.Sim`` — any game registered in gmrules (and in
model/games/gmrules/) is an RL environment with no per-game Python code.

Frames are gmrules ``encode_frame`` output — the encoder of the gmdata converter —
so an RL frame IS the IL frame of the same position: ``obs`` holds the game's obs
tensors (its gm-shard ``obs`` schema), ``legal`` its legality tensors (the
decoder's ``legal`` dict).  Candidate-action games: obs ``entities``,
``entity_mask``, (``entity_type``), ``globals``, ``actions``, ``action_mask``;
``legal = {"action": action_mask}``; action ``action_idx`` (the index of the
chosen candidate).  Custom schemas: whatever the Rust spec declares (e.g. a board
game: ``board_seq`` / ``globals_seq``, legal ``from`` / ``to_cond``, actions
``from_idx`` / ``to_idx``).

Rows / streams / seats (see env/vec_game.py), from the game's FrameSpec layout:

  * per-seat (hidden-information or simultaneous games): ``rows_per_env =
    n_seats``, row ``env * n_seats + seat`` is that seat's OWN stream — its own
    decision frames only, so no stream ever carries another seat's private
    information in its temporal state; ``stream_seats`` = the row's seat (team
    games: its team, see below).  A row
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

``forced`` = a single legal action (the env applies it whatever is sent).
Team games (``teams``: the team id of every seat, from the game's rules; e.g.
bridge partnerships): teammates share one outcome and ``info["winner"]`` is
the lowest seat of the winning team (−1: a tie); a per-seat row's
``stream_seats`` is its seat's whole TEAM — a stream may act for a
teammate's hand (bridge: the declarer's stream plays dummy's cards,
``to_play`` = declarer, as in OpenSpiel), so one controller must own the
team (env/league.py assigns whole teams).
Actions: the decoder's ``sample()`` output or its replay keys — ``ACTION_KEYS``
maps the former to the game's action keys (the entry's ``sample_keys``); keys
that are no legal action of an acting row raise.  Rewards [N, n_seats] are the
game's outcome (in [-1, 1], = the IL value target) paid on the step the episode
ends; ``info["returns"]`` holds the raw payoffs.  Chance events come from
per-episode seeds (``reset(seed=...)``: seed + k).  ``threads`` steps the envs on
that many Rust threads (default: ``$GM_SIM_THREADS`` or 1; results never depend
on it).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import gmsim
import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

_TORCH = {"f32": torch.float32, "bool": torch.bool, "i32": torch.int32, "i64": torch.int64}


class GmSimVecEnv(VecEnvBase):
    """N episodes of gmrules game ``game`` (see the module docstring).

    ``config`` / extra keyword arguments: the gmrules game config (JSON-able;
    defaults: the entry's ``sim_config`` in model/games/gmrules/, then the Rust
    defaults), e.g. ``GmSimVecEnv("kuhn_poker", 8, players=3)``.  ``max_turns``
    truncates episodes (zero reward) before the game's own bound.
    ``sample_keys`` (decoder sample() key → action key) defaults to the entry's.
    """

    def __init__(self, game: str, n_envs: int = 1, *, autoreset: bool = False,
                 seed: Optional[int] = 0, max_turns: Optional[int] = None,
                 threads: Optional[int] = None, sample_keys: Optional[Mapping[str, str]] = None,
                 config: Optional[Mapping[str, Any]] = None, **config_kw):
        super().__init__(n_envs, autoreset, seed)
        from model.games.gmrules import CANDIDATE_SAMPLE_KEYS, rules_games

        entry = rules_games().get(game)
        cfg = dict(entry.sim_config) if entry is not None else {}
        cfg.update(config or {})
        cfg.update(config_kw)
        if threads is None:
            threads = int(os.environ.get("GM_SIM_THREADS", "1"))
        self.game = game
        self.config = cfg
        self.max_turns = max_turns
        self.threads = int(threads)
        self._sim = gmsim.Sim(game, cfg, self._n, autoreset=bool(autoreset), max_turns=max_turns,
                              threads=self.threads)
        self.spec: Dict[str, Any] = dict(self._sim.spec)
        self.schema = self.spec["schema"]
        self.is_candidate = self.spec["candidate"] is not None
        self.action_keys: Tuple[str, ...] = tuple(self.spec["action_keys"])
        if sample_keys is None:
            sample_keys = (entry.action_keys_map() if entry is not None
                           else CANDIDATE_SAMPLE_KEYS if self.is_candidate else {})
        # decoder sample() key → action key, plus the action keys themselves
        self.ACTION_KEYS = {**{k: k for k in self.action_keys}, **dict(sample_keys)}
        bad = sorted(set(self.ACTION_KEYS.values()) - set(self.action_keys))
        if bad:
            raise ValueError(f"{game}: sample_keys map to {bad}, not action keys "
                             f"{list(self.action_keys)}")
        self.n_seats = int(self.spec["n_seats"])
        # team games (gmrules Game::teams, e.g. bridge partnerships [0, 1, 0, 1]):
        # teammates share one outcome, info["winner"] is the winning team's lowest
        # seat, and env/league.py gives a policy whole teams
        teams = gmsim.game_info(game, cfg)["teams"]
        self.teams: Optional[Tuple[int, ...]] = tuple(teams) if teams is not None else None
        self.layout = self.spec["layout"]
        self.interleaved = self.layout == "interleaved"
        self.rows_per_env = 1 if self.interleaved else self.n_seats
        self.lockstep = self.interleaved
        self.reset()

    # ------------------------------------------------------------------

    @property
    def finished(self) -> torch.Tensor:
        return torch.from_numpy(self._sim.done_all().copy())

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

    def reset_from(self, prefixes: Sequence[Any], *, env_ids: Optional[Sequence[int]] = None,
                   seed: Optional[int] = None) -> VecObs:
        """New episodes for ``env_ids`` (default: every env) that first play the
        recorded trajectory ``prefixes`` (one per env, :meth:`trajectory` format;
        every event verified), then go on with chance from the episode seeds.  The
        rows' frames are the converter's frames of the records' next decisions
        (golden tests on real records; RL from recorded positions)."""
        if seed is not None:
            self._seed_rngs(seed)
        ids = self._env_ids(env_ids)
        if len(prefixes) != len(ids):
            raise ValueError(f"{len(prefixes)} prefixes for {len(ids)} envs")
        self._sim.reset_from(ids.tolist(), self._episode_seeds(len(ids), seed), list(prefixes))
        return self.observe()

    def obs_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {t["name"]: (tuple(t["shape"]), _TORCH[t["dtype"]]) for t in self.spec["obs"]}

    def legal_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {t["name"]: (tuple(t["shape"]), _TORCH[t["dtype"]]) for t in self.spec["legal"]}

    # ------------------------------------------------------------------

    def observe(self) -> VecObs:
        o = self._sim.observe()
        obs = {k: torch.from_numpy(v) for k, v in o["obs"].items()}
        legal = {k: torch.from_numpy(v) for k, v in o["legal"].items()}
        R = self._sim.n_rows
        rows = torch.arange(R, dtype=torch.long)
        if self.interleaved:
            stream_seats = torch.ones(R, self.n_seats, dtype=torch.bool)
        elif self.teams is not None:        # a stream may act for a teammate's hand
            team = torch.tensor(self.teams)
            stream_seats = team[rows % self.n_seats].unsqueeze(1) == team.unsqueeze(0)
        else:
            stream_seats = torch.nn.functional.one_hot(rows % self.n_seats, self.n_seats).bool()
        return VecObs(
            obs=obs,
            legal=legal,
            to_play=torch.from_numpy(o["to_play"].copy()),
            env_id=rows // self.rows_per_env,
            stream_id=rows,
            step_idx=torch.from_numpy(o["step_idx"].copy()),
            new_episode=torch.from_numpy(o["new_episode"].copy()),
            active=torch.from_numpy(o["active"].copy()),
            forced=torch.from_numpy(o["forced"].copy()),
            stream_seats=stream_seats,
        )

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        acts = self.replay_actions(actions)
        self._require(acts, self.action_keys)
        keys = {k: acts[k].detach().cpu().numpy().astype(np.int64).reshape(-1)
                for k in self.action_keys}
        info = self._sim.step(keys)
        out: Dict[str, Any] = {
            "terminated": torch.from_numpy(info["terminated"].copy()),
            "truncated": torch.from_numpy(info["truncated"].copy()),
            "winner": torch.from_numpy(info["winner"].copy()),
            "episode_len": torch.from_numpy(info["episode_len"].copy()),
            "acted": torch.from_numpy(info["acted"].copy()),        # [R] rows that acted
            "applied": {k: torch.from_numpy(v).clamp(min=0) for k, v in info["applied"].items()},
            "illegal": torch.zeros(self._n, dtype=torch.bool),
            "returns": torch.from_numpy(info["returns"].copy()),
        }
        return (self.observe(), torch.from_numpy(info["rewards"].copy()),
                torch.from_numpy(info["done"].copy()), out)

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """A uniformly random legal action per acting row of ``obs`` (action keys,
        one entry per row of ``obs`` — every row, or the rows of a
        ``VecObs.select`` subset, e.g. a policy's rows in the league; drawn by the
        simulator from a seed of the torch generator)."""
        gen = self._generator(generator)
        seed = int(torch.randint(0, 2**62, (1,), generator=gen))
        rows = obs.stream_id.long()            # stream id = row index (observe())
        return {k: torch.from_numpy(v)[rows] for k, v in self._sim.random_actions(seed).items()}

    # ------------------------------------------------------------------

    def trajectory(self, env: int) -> np.ndarray:
        """Env ``env``'s current episode as a gmrules trajectory (int64 [n, 2]
        ``(seat, id)``, seat −1 = chance) — the converters' record format."""
        return self._sim.trajectory(env)

    def legal_ids(self, row: int):
        """Canonical legal action ids of a row's current frame."""
        return self._sim.legal_ids(row)


#: the candidate-action games' env (the same class: the schema comes from the game)
CandidateVecEnv = GmSimVecEnv
