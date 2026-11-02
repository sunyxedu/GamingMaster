"""ARC-AGI-3 as a :class:`~env.vec_game.VecGameEnv`: N offline games stepped together
— the tier-3 network's environment (the tier-1 Meta-Harness plays the same games through
metaharness/testbeds/arc3.py).

Engine: the official SDK (``arc-agi`` + ``arcengine``, the ``arc3`` extra) in OFFLINE
mode on the cached game files (``environments_dir``, else ``$ENVIRONMENTS_DIR``).  The
public set: cd82 lf52 ls20 sb26 (:data:`DEV_GAMES`) and sp80 tn36 tu93 vc33
(:data:`HELDOUT_GAMES`, kept out of training to measure transfer) — 60 levels.  Every
game starts from its engine seed 0, so a game is deterministic given the actions.

One env = one game played from its first level; the episode ends when every level is
won (WIN) or after ``max_actions`` actions (a truncation; RESETs count, as in the
ARC-AGI-3 score).  Which game an env plays rotates with every reset (env i, reset k:
``games[(i + k) % len(games)]``), so lockstep waves cover the games evenly.

Frame (one row per env = one stream, a single seat)::

    obs     grid     [64, 64] uint8  palette index (0..15) of every cell of the frame's
                                     last layer (the settled state after the action's
                                     animation); x = column, y = row
            delta    [64, 64] bool   cells that differ from the previous frame (all
                                     False on the episode's first frame)
            globals  [22] f32        0-7   the frame's available actions (RESET, ACTION1..7)
                                     8     levels completed / the game's levels
                                     9     min(actions used / max_actions, 1)
                                     10-17 the previous action (one-hot; none on frame 0)
                                     18-19 its click (x / 63, y / 63; 0 unless ACTION6)
                                     20    the previous action completed a level
                                     21    the game is over (GAME_OVER: only RESET left)
    legal   type     [8] bool        RESET and the frame's available ACTION1..7; on a
                                     game-over frame RESET alone (a forced frame)
            click    [4096] bool     every cell when ACTION6 is available, else none
    action  type_idx [R] long        0 = RESET, k = ACTIONk
            click_idx[R] long        y * 64 + x, read for ACTION6 only

Reward (seat 0): 1 / (the game's levels) per level the step completed, so winning a
whole game is worth 1 (the value head's range for every game; the ARC-AGI-3 score
weighs levels too); with ``novelty`` > 0 also ``novelty / sqrt(n)`` for a grid seen for
the n-th time in the episode (count-based exploration on the grid's hash; 0 = off, the
default).
A GAME_OVER frame is shown to the policy with RESET as its only legal action (the
env applies it: ``forced``); the SDK then restarts the current level.

Frontier starts (``frontier`` > 0; 0 = off, the default): Go-Explore's "return, then
explore" for these deterministic games.  The env archives, per game, the shortest action
sequence (from a fresh game) that reached each number of completed levels — its own
episodes' level-ups, plus ``frontier_seed`` (arc3_solve.py's solutions: a JSON file or its
``<out>.partial`` directory; the solved levels' actions joined) — and a new episode starts,
with probability ``frontier``, from an archived level chosen uniformly: the sequence is
replayed out of sight, the episode begins at that level's first frame (``new_episode``, no
delta, the full ``max_actions`` budget) and pays only the levels it completes from there.
A replay that does not reproduce its level count drops the entry (a fresh start instead).
Without it the policy reaches a later level only after re-solving every earlier one within
the same ``max_actions``, so the later levels' rewards never come (problem 146).
The archive is part of :meth:`state_dict` (a resume keeps it).

RESET is always a level reset, as in the ARC-AGI-3 competition mode ("Game Resets are not
allowed and become Level Resets") and in ARC Prize's human baseline recordings (arcengine's
``ONLY_RESET_LEVELS``): by default the engine restarts the whole game on a RESET at a fresh
level (no other action since the level began: right after a level-up, a game over's RESET or
another RESET).  A new episode always starts the game from its first level (the engine's full
reset; a reused SDK wrapper otherwise kept the level an earlier, truncated episode ended in).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

GRID = 64
N_CELLS = GRID * GRID
N_TYPES = 8                       # RESET, ACTION1..ACTION7
CLICK = 6                         # ACTION6 takes (x, y)
N_GLOBALS = 22
DEV_GAMES = ("cd82", "lf52", "ls20", "sb26")
HELDOUT_GAMES = ("sp80", "tn36", "tu93", "vc33")


def _sdk(environments_dir: Optional[str]):
    """Import the offline SDK (quiet), pointed at the cached game files."""
    env_dir = environments_dir or os.environ.get("ENVIRONMENTS_DIR")
    if not env_dir or not os.path.isdir(env_dir):
        raise RuntimeError(
            "ARC-AGI-3 needs the cached game files: pass environments_dir= or set "
            f"ENVIRONMENTS_DIR (got {env_dir!r})")
    os.environ["OPERATION_MODE"] = "offline"
    os.environ["ENVIRONMENTS_DIR"] = env_dir
    # RESET is always a level reset (the competition's rule; arcengine's handle_reset reads
    # this at every RESET): without it a RESET on a fresh level — no other action since the
    # level began — restarts the whole game
    os.environ["ONLY_RESET_LEVELS"] = "true"
    for name in ("arc_agi", "arcengine"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        import arc_agi
        from arcengine import GameAction, GameState
    except ImportError as e:                                # pragma: no cover
        raise RuntimeError("ARC-AGI-3 needs the arc3 extra: uv pip install "
                           "'arc-agi==0.9.9' 'arcengine==0.9.3'") from e
    for name in list(logging.root.manager.loggerDict):        # the SDK's sub-loggers
        if name.startswith(("arc_agi", "arcengine")):
            logging.getLogger(name).setLevel(logging.WARNING)
    return arc_agi, GameAction, GameState


class Arc3VecEnv(VecEnvBase):
    """N ARC-AGI-3 games (see the module docstring)."""

    game = "arc3"
    n_seats = 1
    rows_per_env = 1
    interleaved = False
    ACTION_KEYS = {"action_type": "type_idx", "action_click": "click_idx"}

    def __init__(self, n_envs: int = 1, *, games: Sequence[str] = DEV_GAMES,
                 max_actions: int = 400, novelty: float = 0.0, frontier: float = 0.0,
                 frontier_seed: Optional[str] = None,
                 environments_dir: Optional[str] = None, autoreset: bool = False,
                 seed: Optional[int] = 0):
        if not 0.0 <= float(frontier) <= 1.0:
            raise ValueError(f"frontier is a probability, got {frontier}")
        self.frontier = float(frontier)
        # game -> levels completed -> the shortest (type, click) sequence from a fresh game
        self._archive: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
        super().__init__(n_envs, autoreset, seed)
        if isinstance(games, str):
            # "all": every game cached under the environments dir (a directory per game —
            # the 25 official ones and the community ones of data/arc3/env_all);
            # "a,b,c": a list
            if games == "all":
                root = environments_dir or os.environ.get("ENVIRONMENTS_DIR") or ""
                games = sorted(g for g in os.listdir(root) if os.path.isdir(os.path.join(root, g))) if root else []
            else:
                games = [g for g in games.split(",") if g]
        if not games:
            raise ValueError("arc3: at least one game")
        if int(max_actions) < 1:
            raise ValueError(f"max_actions must be >= 1, got {max_actions}")
        self.games = tuple(str(g) for g in games)
        self.max_actions = int(max_actions)
        self.novelty = float(novelty)
        self._arc, self._Action, self._State = _sdk(environments_dir)
        if frontier_seed:
            self.seed_frontier(frontier_seed)
        self._arcade = self._arc.Arcade()
        self._wrappers: Dict[Tuple[int, str], Any] = {}
        self._resets = np.zeros(n_envs, dtype=np.int64)
        self._alloc(self._n)
        self.reset()

    # ------------------------------------------------------------------

    def _alloc(self, n: int) -> None:
        self._n = n
        self.grid = np.zeros((n, GRID, GRID), dtype=np.uint8)
        self.delta = np.zeros((n, GRID, GRID), dtype=bool)
        self.available = np.zeros((n, N_TYPES), dtype=bool)
        self.levels = np.zeros(n, dtype=np.int64)
        self.win_levels = np.ones(n, dtype=np.int64)
        self.actions_used = np.zeros(n, dtype=np.int64)
        self.last_type = np.full(n, -1, dtype=np.int64)
        self.last_click = np.zeros((n, 2), dtype=np.float32)
        self.just_leveled = np.zeros(n, dtype=bool)
        self.game_over = np.zeros(n, dtype=bool)
        self.done = np.ones(n, dtype=bool)
        self.new_episode = np.zeros(n, dtype=bool)
        self.game_of = [""] * n
        self._seen: list = [dict() for _ in range(n)]
        self._trace: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
        if self._resets.shape[0] != n:
            self._resets = np.zeros(n, dtype=np.int64)

    def _wrapper(self, i: int, game: str):
        key = (i, game)
        w = self._wrappers.get(key)
        if w is None and len(self.games) > 8:
            # a large game pool (games="all"): one live game per env, the last one dropped
            for k in [k for k in self._wrappers if k[0] == i]:
                del self._wrappers[k]
        if w is None:
            w = self._arcade.make(game, seed=0)
            if w is None:
                raise RuntimeError(f"arc3: Arcade.make({game!r}) failed — is it cached "
                                   f"under ENVIRONMENTS_DIR?")
            self._wrappers[key] = w
        return w

    def _load(self, i: int, fd) -> None:
        """Frame data → the env's arrays for env ``i`` (delta vs the stored grid)."""
        g = np.asarray(fd.frame[-1], dtype=np.uint8)
        self.delta[i] = g != self.grid[i]
        self.grid[i] = g
        self.available[i] = False
        self.available[i, 0] = True                       # RESET is always possible
        for a in fd.available_actions:
            a = int(a)
            if 1 <= a < N_TYPES:
                self.available[i, a] = True
        self.levels[i] = int(fd.levels_completed)
        self.win_levels[i] = max(int(fd.win_levels), 1)
        self.game_over[i] = fd.state == self._State.GAME_OVER

    def _grid_key(self, i: int) -> bytes:
        return hashlib.blake2b(self.grid[i].tobytes(), digest_size=16).digest()

    def _start(self, ids: np.ndarray) -> None:
        for i in ids.tolist():
            game = self.games[(i + int(self._resets[i])) % len(self.games)]
            self._resets[i] += 1
            self.game_of[i] = game
            w = self._wrapper(i, game)
            # a new episode is a new game: the engine's full reset (the SDK's reset() is a
            # RESET action — a level reset, which would keep a reused wrapper's level)
            w._game.full_reset()
            fd = w.reset()
            self.grid[i] = 0
            self._load(i, fd)
            self._trace[i] = []
            self._frontier_start(i, game, w)
            self.delta[i] = False
            self.actions_used[i] = 0
            self.last_type[i] = -1
            self.last_click[i] = 0.0
            self.just_leveled[i] = False
            self.done[i] = False
            self.new_episode[i] = True
            self._seen[i] = {self._grid_key(i): 1}

    def _seed_rngs(self, seed: Optional[int]) -> None:
        super()._seed_rngs(seed)
        # the frontier choice: its own stream (it never shifts the base RNGs' draws)
        self._frontier_rng = np.random.default_rng(None if seed is None else [int(seed), 0xF0E])

    def _archive_level(self, game: str, levels: int, trace: Sequence[Tuple[int, int]]) -> bool:
        """Keep ``trace`` as ``game``'s way to ``levels`` completed levels if it is the first
        or a shorter one; True when ``levels`` is beyond every archived level of the game."""
        arch = self._archive.setdefault(game, {})
        new_best = levels > max(arch, default=0)
        old = arch.get(levels)
        if old is None or len(trace) < len(old):
            arch[levels] = [(int(a), int(c)) for a, c in trace]
        return new_best

    def seed_frontier(self, path: str) -> Dict[str, List[int]]:
        """Archive arc3_solve.py's solutions (a JSON file, or its ``<out>.partial`` directory
        of per-game files): for each of this env's games, the solved levels' actions joined
        up to every level they reach.  Returns {game: archived level counts}."""
        if os.path.isdir(path):
            sol = {}
            for name in sorted(os.listdir(path)):
                if name.endswith(".json") and not name.startswith("."):
                    with open(os.path.join(path, name)) as fh:
                        sol[name[:-5]] = json.load(fh)
        else:
            with open(path) as fh:
                sol = json.load(fh)
        out: Dict[str, List[int]] = {}
        for game in self.games:
            prefix: List[Tuple[int, int]] = []
            for k, lv in enumerate((sol.get(game) or {}).get("levels") or []):
                if not lv.get("actions") or int(lv.get("level", k)) != k:
                    break                          # the archive needs every earlier level
                for typ, x, y in lv["actions"]:
                    prefix.append((int(typ), int(y) * GRID + int(x) if int(typ) == CLICK else 0))
                self._archive_level(game, k + 1, prefix)
                out.setdefault(game, []).append(k + 1)
        return out

    def frontier_levels(self) -> Dict[str, int]:
        """{game: the furthest archived level count}."""
        return {g: max(a) for g, a in self._archive.items() if a}

    def _replay(self, w, prefix: Sequence[Tuple[int, int]]):
        fd = None
        for a, c in prefix:
            data = {"x": c % GRID, "y": c // GRID} if a == CLICK else None
            fd = w.step(self._Action.from_id(int(a)), data=data)
            if fd is None:
                return None
        return fd

    def _frontier_start(self, i: int, game: str, w) -> int:
        """With probability ``frontier``: replay an archived level of ``game`` into env ``i``
        (uniform over its archived levels); the level count it starts at (0: a fresh game)."""
        arch = self._archive.get(game)
        if self.frontier <= 0.0 or not arch or self._frontier_rng.random() >= self.frontier:
            return 0
        levels = sorted(arch)
        target = levels[int(self._frontier_rng.integers(len(levels)))]
        prefix = arch[target]
        fd = self._replay(w, prefix)
        if (fd is None or int(fd.levels_completed) != target or fd.state == self._State.WIN
                or fd.state == self._State.GAME_OVER):
            # the replay did not reproduce the archived level: drop it, start the game afresh
            del arch[target]
            w._game.full_reset()
            fd = w.reset()
            self.grid[i] = 0
            self._load(i, fd)
            return 0
        self._load(i, fd)
        self._trace[i] = list(prefix)
        return target

    @property
    def finished(self) -> torch.Tensor:
        return torch.from_numpy(self.done.copy())

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        """Every game starts from its engine seed 0 (deterministic), so ``seed`` only
        reseeds the env RNGs (``random_actions`` without a generator)."""
        if seed is not None:
            self._seed_rngs(seed)
        if n is not None and n != self._n:
            if env_ids is not None:
                raise ValueError("resize (n) and env_ids are exclusive")
            self._alloc(int(n))
        self.new_episode[:] = False
        self._start(self._env_ids(env_ids))
        return self.observe()

    def obs_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {"grid": ((GRID, GRID), torch.uint8), "delta": ((GRID, GRID), torch.bool),
                "globals": ((N_GLOBALS,), torch.float32)}

    # ------------------------------------------------------------------

    def _globals(self) -> np.ndarray:
        n = self._n
        g = np.zeros((n, N_GLOBALS), dtype=np.float32)
        g[:, 0:8] = self.available
        g[:, 8] = self.levels / self.win_levels
        g[:, 9] = np.minimum(self.actions_used / self.max_actions, 1.0)
        has = self.last_type >= 0
        g[np.flatnonzero(has), 10 + self.last_type[has]] = 1.0
        g[:, 18:20] = self.last_click
        g[:, 20] = self.just_leveled
        g[:, 21] = self.game_over
        return g

    def legal_type(self) -> np.ndarray:
        """[N, 8] the legal action types: RESET alone on a game-over frame."""
        legal = self.available.copy()
        legal[self.game_over, 1:] = False
        return legal

    def observe(self) -> VecObs:
        n = self._n
        active = ~self.done
        act = torch.from_numpy(active)
        ltype = torch.from_numpy(self.legal_type()) & act[:, None]
        click = (ltype[:, CLICK:CLICK + 1]).expand(n, N_CELLS).clone()
        forced = torch.from_numpy(self.game_over & active)
        return VecObs(
            obs={"grid": torch.from_numpy(self.grid.copy()),
                 "delta": torch.from_numpy(self.delta.copy()),
                 "globals": torch.from_numpy(self._globals())},
            legal={"type": ltype, "click": click},
            to_play=torch.zeros(n, dtype=torch.long),
            env_id=torch.arange(n, dtype=torch.long),
            stream_id=torch.arange(n, dtype=torch.long),
            step_idx=torch.from_numpy(self.actions_used.copy()),
            new_episode=torch.from_numpy(self.new_episode & active),
            active=act,
            forced=forced,
            stream_seats=torch.ones(n, 1, dtype=torch.bool),
        )

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        n = self._n
        acts = self.replay_actions(actions)
        self._require(acts, ("type_idx",))
        types = acts["type_idx"].reshape(-1).cpu().numpy().astype(np.int64)
        clicks = (acts["click_idx"].reshape(-1).cpu().numpy().astype(np.int64)
                  if "click_idx" in acts else np.zeros(n, dtype=np.int64))
        active = ~self.done
        legal = self.legal_type()
        rewards = np.zeros((n, 1), dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        failed = np.zeros(n, dtype=bool)
        applied_type = np.zeros(n, dtype=np.int64)
        applied_click = np.zeros(n, dtype=np.int64)
        for i in np.flatnonzero(active).tolist():
            a = 0 if self.game_over[i] else int(types[i])       # forced RESET
            if not (0 <= a < N_TYPES and legal[i, a]):
                raise ValueError(f"arc3 env {i} ({self.game_of[i]}): action type {a} is not "
                                 f"legal (legal: {np.flatnonzero(legal[i]).tolist()})")
            data = None
            if a == CLICK:
                c = int(clicks[i])
                if not 0 <= c < N_CELLS:
                    raise ValueError(f"arc3 env {i}: click_idx {c} out of range")
                data = {"x": c % GRID, "y": c // GRID}
                applied_click[i] = c
            applied_type[i] = a
            prev_levels = int(self.levels[i])
            fd = self._wrappers[(i, self.game_of[i])].step(self._Action.from_id(a), data=data)
            self.actions_used[i] += 1
            self.last_type[i] = a
            self.last_click[i] = ((c % GRID) / (GRID - 1), (c // GRID) / (GRID - 1)) \
                if a == CLICK else (0.0, 0.0)
            if fd is None:                                   # an engine error: stop here
                failed[i] = truncated[i] = True
                continue
            self._load(i, fd)
            gained = int(self.levels[i]) - prev_levels
            self.just_leveled[i] = gained > 0
            rewards[i, 0] += max(gained, 0) / float(self.win_levels[i])
            if self.frontier > 0.0:
                self._trace[i].append((a, int(clicks[i]) if a == CLICK else 0))
                if gained > 0 and fd.state != self._State.WIN:
                    lv = int(self.levels[i])
                    if self._archive_level(self.game_of[i], lv, self._trace[i]):
                        print(f"[arc3] {self.game_of[i]}: level {lv}/{int(self.win_levels[i])} "
                              f"reached ({len(self._trace[i])} actions from a fresh game); "
                              f"{self.frontier:.0%} of its episodes start from an archived "
                              f"level", flush=True)
            if self.novelty > 0:
                k = self._grid_key(i)
                cnt = self._seen[i].get(k, 0) + 1
                self._seen[i][k] = cnt
                rewards[i, 0] += self.novelty / np.sqrt(cnt)
            if fd.state == self._State.WIN:
                terminated[i] = True
            elif self.actions_used[i] >= self.max_actions:
                truncated[i] = True
        self.new_episode[:] = False
        ended = terminated | truncated
        episode_len = np.where(ended, self.actions_used, 0)
        self.done |= ended
        info: Dict[str, Any] = {
            "terminated": torch.from_numpy(terminated),
            "truncated": torch.from_numpy(truncated),
            "winner": torch.from_numpy(np.where(terminated, 0, -1).astype(np.int64)),
            "episode_len": torch.from_numpy(episode_len),
            "acted": torch.from_numpy(active.copy()),
            "applied": {"type_idx": torch.from_numpy(applied_type),
                        "click_idx": torch.from_numpy(applied_click)},
            "illegal": torch.zeros(n, dtype=torch.bool),
            "levels": torch.from_numpy(self.levels.copy()),
            "failed": torch.from_numpy(failed),
        }
        e_idx = np.flatnonzero(ended)
        if self.autoreset and e_idx.size:
            self._start(e_idx)
        return self.observe(), torch.from_numpy(rewards), torch.from_numpy(ended.copy()), info

    # ------------------------------------------------------------------

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """A uniformly random legal action type per row; a click cell uniformly."""
        gen = self._generator(generator)
        legal = obs.legal["type"].float()
        legal[legal.sum(-1) == 0, 0] = 1.0                   # inactive rows: RESET
        t = torch.multinomial(legal, 1, generator=gen).squeeze(1)
        c = torch.randint(0, N_CELLS, (obs.n_rows,), generator=gen)
        return {"type_idx": t, "click_idx": c}

    def state_dict(self) -> Dict[str, Any]:
        """The env RNGs and the frontier archive (episodes restart: every wave starts its
        games afresh)."""
        st = super().state_dict()
        st["frontier_rng"] = self._frontier_rng.bit_generator.state
        st["frontier_archive"] = {
            g: {int(k): np.asarray(p, dtype=np.int64).reshape(-1, 2) for k, p in a.items()}
            for g, a in self._archive.items()}
        return st

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        if "frontier_rng" in state:
            self._frontier_rng.bit_generator.state = state["frontier_rng"]
        for g, a in (state.get("frontier_archive") or {}).items():
            for k, p in a.items():        # merged: the shorter of a seed's and the saved way
                self._archive_level(str(g), int(k),
                                    [(int(t), int(c)) for t, c in np.asarray(p).reshape(-1, 2)])

    def close(self) -> None:
        self._wrappers.clear()
