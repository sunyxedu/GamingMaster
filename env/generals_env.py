"""
Interactive generals.io environment for RL self-play.

Game mechanics are adapted from ``data_processing/processing/simulator.py``
(the verified replay simulator) into a Gym-like interface that supports
two-player simultaneous-move self-play.

Observation format matches the model's ``step()`` input:
    grid       [C, H, W]     12-channel one-hot + army
    globals    [G]            4 log-normalised stats
    legal_mask [1, H, W]     owned tiles with army > 1

Reward design (per step; discounted returns are NOT bounded to [-1, 1] —
the value head is deliberately unbounded):
    Terminal:  win +1.0, loss -1.0; turn-limit truncation scores
               ±0.35 × material heuristic (soft terminal)
    Dense:     0.6 × delta_heuristic × dense_reward_scale (potential-based)
    Events:    enemy city capture +0.16 / loss -0.16 (symmetric),
               neutral city damage pays unit × damage/G0 per hit
               (unit = min(0.25, city_budget_total / n_cities), so the
               whole map's neutral-city reward stays below the terminal
               reward and a full capture sums to exactly `unit`),
               capture precision bonus ≤ 0.3 × unit,
               invalid move -0.05
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ======================================================================
# Direction offsets  (consistent with data_processing convention)
# ======================================================================
# 0=down  1=right  2=up  3=left
_DIR_DR = [1, 0, -1, 0]
_DIR_DC = [0, 1, 0, -1]

# Terrain constants
_BLANK = 0
_CITY = 1
_GENERAL = 2
_MOUNTAIN = 3

_N_STATES = 11
_STATE_IDS = np.arange(_N_STATES).reshape(-1, 1, 1)
# float32(math.log1p(a) / 10) per army value a — the army channel, looked up
# instead of computed per tile (same float64 math, same float32 rounding as
# writing the Python float into a float32 tensor); grown on demand.
_LOG_ARMY = np.zeros(0, dtype=np.float32)


def _log_army(army: np.ndarray) -> np.ndarray:
    global _LOG_ARMY
    top = int(army.max()) if army.size else 0
    if int(army.min(initial=0)) < 0:
        raise ValueError("math domain error: negative army on a visible tile")
    if top >= len(_LOG_ARMY):
        n = max(top + 1, 2 * len(_LOG_ARMY), 1024)
        _LOG_ARMY = np.array([math.log1p(a) / 10.0 for a in range(n)], dtype=np.float32)
    return _LOG_ARMY[army]


def _pad1(a: np.ndarray, fill=0) -> np.ndarray:
    """``a`` with a one-cell border of ``fill`` (np.pad without its overhead)."""
    out = np.full((a.shape[0] + 2, a.shape[1] + 2), fill, dtype=a.dtype)
    out[1:-1, 1:-1] = a
    return out


class _Tile:
    """Mutable tile state."""
    __slots__ = ("terrain", "army", "occupied")

    def __init__(self, terrain: int = 0, army: int = 0, occupied: int = -1):
        self.terrain = terrain
        self.army = army
        self.occupied = occupied


class GeneralsEnv:
    """Two-player simultaneous-move generals.io environment.

    Parameters
    ----------
    grid_h, grid_w : int
        Map dimensions (default 25×25).
    n_mountains, n_cities : int
        Number of random mountains / cities on the map.
    city_army_range : tuple[int, int]
        Uniform range for initial city army.
    """

    def __init__(
        self,
        grid_h: int = 25,
        grid_w: int = 25,
        n_mountains: int = 20,
        n_cities: int = 5,
        city_army_range: Tuple[int, int] = (40, 50),
        max_turns: int = 9999,
        # Legacy param (ignored, kept for backwards compat)
        dense_reward: bool = False,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.N = grid_h * grid_w
        self.n_mountains = n_mountains
        self.n_cities = n_cities
        self.city_army_range = city_army_range
        self.max_turns = max_turns

        # Real map bounds (for replays smaller than 25×25)
        self.real_h = grid_h
        self.real_w = grid_w

        # Per-step move-execution trace (see _execute_single_move)
        self._move_trace: List[Dict[str, int]] = []

        # State (set by reset)
        self.map: List[List[_Tile]] = []
        self.generals: List[int] = []       # flat index per player (in 25-wide grid)
        self.cities: List[int] = []         # flat indices (in 25-wide grid)
        self.mountains: List[int] = []      # flat indices (in 25-wide grid)
        self.player_land: List[List[int]] = [[], []]
        self.turn: int = 0
        self.done: bool = True
        self.winner: Optional[int] = None
        # Per-player sighted-mountain memory (see reset / _get_observation);
        # memory_vision=False restores pure instantaneous fog encoding.
        self.seen_mountains: List[set] = [set(), set()]
        self.memory_vision: bool = True

        # Heuristic tracking for dense reward
        self.prev_heuristic: Dict[int, float] = {0: 0.0, 1: 0.0}

        # Reward scales (can be decayed independently during RL training)
        self.city_reward_scale: float = 1.0
        self.dense_reward_scale: float = 1.0

        # Truncation soft-terminal knobs (settable by the trainer).  A
        # positive trunc_penalty makes hitting the turn limit strictly worse
        # than converting a win — without it, symmetric self-play can drift
        # into an equilibrium where both sides stall to the clock and farm
        # the material score instead of ever finishing the game.
        self.trunc_material_scale: float = 0.35
        self.trunc_penalty: float = 0.0

        # Append production-cycle features to the globals vector (must match
        # the model's cfg.turn_globals / global_features=7): the temporal
        # attention carries no positional encoding, so without these the
        # policy cannot see the 50-turn all-land production cycle or the
        # city/general half-turn tick at all.
        self.turn_globals: bool = False

        # Which player's rewards/events go into info["reward_breakdown"].
        # The learner's seat is randomized per episode during training (the
        # move-priority tiebreak favors P0, and a learner pinned to P0
        # trains against — and is measured with — a rules advantage it
        # won't have on the real server), so the breakdown must follow it.
        self.breakdown_pid: int = 0

        # Neutral-city reward budget for the WHOLE map: the per-city unit is
        # budget / n_cities (capped at 0.25). A flat per-city bonus let
        # city-dense replay maps pay out 2-3× the terminal reward per game,
        # which dominated win/loss and destabilised training.
        self.city_budget_total: float = 0.8
        self.city_reward_unit: float = 0.25
        # Initial garrison per city (set at reset) — denominator for the
        # damage-proportional payout.
        self.city_initial_army: Dict[int, int] = {}

        # Map pool for replay-based maps
        self.map_pool: Optional[List[Dict]] = None

    # ==================================================================
    # Heuristic (city-adjusted army ratio)
    # ==================================================================

    def _heuristic(self, pid: int) -> float:
        """Evaluate position for player `pid`. Returns value in [-1, 1].

        Same formula as the IL value target in
        data_processing/parsers/generals_json.py so the value head sees a
        consistent signal across IL pre-training and RL fine-tuning.
        """
        s = self._player_stats(pid)
        o = self._player_stats(1 - pid)
        if s["land"] == 0:
            return -1.0
        if o["land"] == 0:
            return 1.0
        city_diff = s["cities"] - o["cities"]
        adj_my = s["army"] + max(city_diff, 0) * 50
        adj_op = o["army"] + max(-city_diff, 0) * 50
        ratio = adj_my / max(adj_op, 1)
        return float(math.tanh(ratio - 1.0))

    def truncation_score(self, pid: int) -> float:
        """Soft terminal reward for truncated games: the city-adjusted
        material heuristic scaled by ``trunc_material_scale``, minus a flat
        ``trunc_penalty`` charged to BOTH players.  Defaults (0.35, 0.0)
        keep who-is-ahead mattering when the clock runs out without ever
        outweighing a real win/loss.  With a penalty, the leading side's
        best truncation outcome (scale − penalty) is far below the +1 for
        winning, so leads must be converted rather than sat on."""
        return self.trunc_material_scale * self._heuristic(pid) - self.trunc_penalty

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(
        self, seed: Optional[int] = None, map_data: Optional[Dict] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Reset the environment and return initial observations.

        ``map_data`` (a map-pool entry: map_w / map_h / generals / cities /
        city_armies / mountains) plays exactly that map, player ``i`` on
        ``generals[i]`` — no random seat swap (replaying a recorded game,
        env.vec_generals).  Otherwise, if a map pool is loaded (via
        ``load_map_pool``), a random replay map is used, else random map
        generation.
        """
        if seed is not None:
            random.seed(seed)

        # Per-player structure memory (fog-of-war honest: each player only
        # remembers what THEY have personally seen). Mountains a player has
        # sighted stay encoded as mountains after vision is lost — without
        # this the policy provably cannot remember walls (a sighted ridge
        # decays back to the generic "obstacle in fog" state) and beelines
        # into terrain it has already scouted.
        self.seen_mountains = [set(), set()]

        self.current_map_idx: Optional[int] = None
        if map_data is not None:
            self._generate_map_from_replay(map_data)
        elif self.map_pool:
            self.current_map_idx = random.randint(0, len(self.map_pool) - 1)
            map_data = self.map_pool[self.current_map_idx]
            # Randomly swap general assignments so P0 doesn't always get generals[0]
            if random.random() < 0.5:
                swapped = dict(map_data)
                swapped["generals"] = [map_data["generals"][1], map_data["generals"][0]]
                self._generate_map_from_replay(swapped)
            else:
                self._generate_map_from_replay(map_data)
        else:
            self.real_h = self.grid_h
            self.real_w = self.grid_w
            self._generate_map()

        self.turn = 0
        self.done = False
        self.winner = None

        # Direction-validity grid [4, H, W]: moving from (r, c) in direction
        # d stays in real bounds and does not enter a mountain. Static per
        # map and player-independent — used to hard-mask the direction head
        # so illegal moves are never sampled.
        dir_valid = torch.zeros(4, self.grid_h, self.grid_w, dtype=torch.bool)
        for r in range(self.real_h):
            for c in range(self.real_w):
                if self.map[r][c].terrain == _MOUNTAIN:
                    continue
                for d in range(4):
                    nr, nc = r + _DIR_DR[d], c + _DIR_DC[d]
                    if (0 <= nr < self.real_h and 0 <= nc < self.real_w
                            and self.map[nr][nc].terrain != _MOUNTAIN):
                        dir_valid[d, r, c] = True
        self.dir_valid = dir_valid

        # Neutral-city payout bookkeeping: every city starts neutral, so the
        # per-city unit and initial garrisons are fixed for the whole game.
        self.city_initial_army = {
            idx: max(1, self.map[idx // self.grid_w][idx % self.grid_w].army)
            for idx in self.cities
        }
        self.city_reward_unit = min(
            0.25, self.city_budget_total / max(len(self.cities), 1),
        )

        # Initialize heuristic tracking
        self.prev_heuristic = {
            0: self._heuristic(0),
            1: self._heuristic(1),
        }

        return self._observations()

    # ==================================================================
    # Step (simultaneous)
    # ==================================================================

    def step(
        self,
        action_0: Optional[Dict[str, int]],
        action_1: Optional[Dict[str, int]],
    ) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],   # observations
        Dict[int, float],                      # rewards
        bool,                                  # done
        Dict[str, Any],                        # info
    ]:
        """Process one turn with both players' actions.

        Each action is ``{"source_idx": int, "direction": int, "split": int}``
        or ``None`` to skip.

        Returns ``(obs, rewards, done, info)``.
        """
        assert not self.done, "call reset() before step()"

        rewards: Dict[int, float] = {0: 0.0, 1: 0.0}
        info: Dict[str, Any] = {"turn": self.turn, "illegal": {0: False, 1: False}}
        # Per-component reward tracking (for player 0 only)
        reward_breakdown: Dict[str, float] = {
            "r_invalid": 0.0, "r_city": 0.0,
            "r_dense": 0.0, "r_winlose": 0.0,
            # Player-0 city event counts (not rewards): neutral captures,
            # failed neutral hits, opponent-city steals, own-city losses.
            "c_cap": 0.0, "c_bash": 0.0, "c_steal": 0.0, "c_loss": 0.0,
        }

        # Convert actions → simulator format (detect illegal moves)
        move_0, illegal_0 = self._action_to_move(0, action_0)
        move_1, illegal_1 = self._action_to_move(1, action_1)

        bp = self.breakdown_pid
        if illegal_0:
            rewards[0] -= 0.05
            info["illegal"][0] = True
        if illegal_1:
            rewards[1] -= 0.05
            info["illegal"][1] = True
        if info["illegal"][bp]:
            reward_breakdown["r_invalid"] -= 0.05

        # NOTE: no pass penalty — it was trivially dodged by shuffling army
        # between owned tiles and punished legitimate accumulation; and no
        # failed-city-attack penalty — whittling a 40-50 army city over
        # several hits is normal play, not a mistake.

        # Track cities before move execution (for capture detection)
        cities_before = {
            0: set(idx for idx in self.player_land[0]
                   if self.map[idx // self.grid_w][idx % self.grid_w].terrain == _CITY),
            1: set(idx for idx in self.player_land[1]
                   if self.map[idx // self.grid_w][idx % self.grid_w].terrain == _CITY),
        }
        # Neutral garrisons before moves — for failed-hit (bash) counting;
        # neutral cities never regrow, so any decrease is a hit.
        neutral_before = {
            idx: self.map[idx // self.grid_w][idx % self.grid_w].army
            for idx in self.cities
            if self.map[idx // self.grid_w][idx % self.grid_w].occupied == -1
        }

        # Execute moves (with priority ordering); the trace records each
        # executed move's destination pre-state for exact reward attribution.
        self._move_trace = []
        self._process_moves(move_0, move_1)

        # Increment armies
        self.turn += 1
        self._increment_armies()

        # Detect city captures and losses.
        # Losing a city costs exactly what the opponent gains: without the
        # symmetric penalty, two agents trading a city back and forth farm
        # net-positive reward every cycle (the dense potential telescopes out).
        for pid in [0, 1]:
            opp = 1 - pid
            cities_after = set(idx for idx in self.player_land[pid]
                               if self.map[idx // self.grid_w][idx % self.grid_w].terrain == _CITY)
            new_cities = cities_after - cities_before[pid]
            for city_idx in new_cities:
                if city_idx in cities_before[opp]:
                    # Took opponent's city — fixed reward, mirrored as a loss
                    city_r = 0.16 * self.city_reward_scale
                    rewards[pid] += city_r
                    rewards[opp] -= city_r
                    if pid == bp:
                        reward_breakdown["r_city"] += city_r
                        reward_breakdown["c_steal"] += 1
                    else:
                        reward_breakdown["r_city"] -= city_r
                        reward_breakdown["c_loss"] += 1
                # Neutral captures are handled exactly from the move trace
                # below (per-move pre-state — a post-hoc diff would mis-pay
                # same-tick double attacks and read post-increment armies).

        # Neutral-city damage payout: each garrison point whittled off pays
        # unit × damage/G0, so a full capture sums to exactly `unit` across
        # however many hits it took. Partial hits pay partially — an agent
        # that cannot yet judge garrison numbers still gets a gradient from
        # every attempt instead of all-or-nothing on completion. Neutral
        # garrisons never regrow, so the payout per city is bounded by
        # `unit` and cannot be farmed. Attribution is per executed move
        # via the trace: each attacker is paid for exactly the garrison it
        # destroyed, at the state the move actually saw.
        for mv in self._move_trace:
            city_idx = mv["end"]
            if city_idx not in neutral_before:
                continue
            pid, opp = mv["player"], 1 - mv["player"]
            captured = mv["moved"] > mv["pre_army"]
            if mv["pre_occupied"] == -1:
                damage = mv["pre_army"] if captured else mv["moved"]
                if damage > 0:
                    g0 = max(self.city_initial_army.get(
                        city_idx, neutral_before[city_idx]), 1)
                    whittle_r = (self.city_reward_unit * (damage / g0)
                                 * self.city_reward_scale)
                    rewards[pid] += whittle_r
                    if pid == bp:
                        reward_breakdown["r_city"] += whittle_r
                        if not captured:
                            # Whittled without capturing (failed hit).
                            reward_breakdown["c_bash"] += 1
                if captured:
                    # Precision bonus for minimal overkill — from the
                    # move's own arithmetic (pre-increment, exact).
                    leftover = mv["moved"] - mv["pre_army"]
                    bonus = (0.3 * self.city_reward_unit
                             * math.exp(-0.1 * leftover)
                             * self.city_reward_scale)
                    rewards[pid] += bonus
                    if pid == bp:
                        reward_breakdown["r_city"] += bonus
                        reward_breakdown["c_cap"] += 1
            elif mv["pre_occupied"] == opp and captured:
                # Same-tick recapture: the opponent took this city moments
                # earlier in the SAME step, so the snapshot-based steal
                # branch above cannot see it — apply the symmetric
                # steal/loss event here instead.
                city_r = 0.16 * self.city_reward_scale
                rewards[pid] += city_r
                rewards[opp] -= city_r
                if pid == bp:
                    reward_breakdown["r_city"] += city_r
                    reward_breakdown["c_steal"] += 1
                else:
                    reward_breakdown["r_city"] -= city_r
                    reward_breakdown["c_loss"] += 1

        # Check win
        winner = self._check_winner()
        if winner is not None:
            self._apply_general_capture(winner)
            self.done = True
            self.winner = winner
            rewards[winner] += 1.0
            rewards[1 - winner] += -1.0
            info["winner"] = winner
            if winner == bp:
                reward_breakdown["r_winlose"] += 1.0
            else:
                reward_breakdown["r_winlose"] -= 1.0
        elif self.turn >= self.max_turns:
            # Turn-limit truncation: scored by material as a SOFT terminal
            # (callers should not bootstrap). A silent 0 made long games
            # where city economy compounds worthless in the terminal
            # channel, structurally undervaluing economic play vs rushing.
            self.done = True
            self.winner = None
            info["truncated"] = True
            for pid in [0, 1]:
                r_trunc = self.truncation_score(pid)
                rewards[pid] += r_trunc
                if pid == bp:
                    reward_breakdown["r_winlose"] += r_trunc

        # Dense reward: heuristic delta
        if not self.done:
            for pid in [0, 1]:
                curr_h = self._heuristic(pid)
                dense_r = 0.6 * (curr_h - self.prev_heuristic[pid]) * self.dense_reward_scale
                rewards[pid] += dense_r
                if pid == bp:
                    reward_breakdown["r_dense"] += dense_r
                self.prev_heuristic[pid] = curr_h

        info["reward_breakdown"] = reward_breakdown
        info["turn"] = self.turn  # turn just completed (post-increment)
        obs = self._observations()
        return obs, rewards, self.done, info

    # ==================================================================
    # Observation
    # ==================================================================

    def _state_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(terrain, army, occupied) [H, W] int64 of the real map bounds;
        padding reads as blank / 0 / neutral."""
        H, W = self.grid_h, self.grid_w
        rH, rW = self.real_h, self.real_w
        terrain = np.zeros((H, W), dtype=np.int64)
        army = np.zeros((H, W), dtype=np.int64)
        occ = np.full((H, W), -1, dtype=np.int64)
        for r in range(rH):
            row = self.map[r]
            terrain[r, :rW] = [t.terrain for t in row[:rW]]
            army[r, :rW] = [t.army for t in row[:rW]]
            occ[r, :rW] = [t.occupied for t in row[:rW]]
        return terrain, army, occ

    def _observations(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Both players' observations (the map read once)."""
        arrays = self._state_arrays()
        return {0: self._get_observation(0, arrays), 1: self._get_observation(1, arrays)}

    def _get_observation(
        self, player_id: int,
        arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, torch.Tensor]:
        """One player's observation, whole-grid numpy (``_tile_to_state`` is
        the per-tile reference of the state encoding;
        tests/test_generals_obs_vectorised.py pins this to the original
        per-tile implementation bit for bit, memory updates included)."""
        H, W = self.grid_h, self.grid_w  # 25, 25 (padded)
        rH, rW = self.real_h, self.real_w  # real map bounds
        terrain, army, occ = arrays if arrays is not None else self._state_arrays()
        inb = np.zeros((H, W), dtype=bool)
        inb[:rH, :rW] = True

        # --- vision: 1 owned (player_land), 2 within the 8-neighbourhood of
        # an owned tile and inside the real map, 0 elsewhere ---
        own = np.zeros(H * W, dtype=bool)
        own[self.player_land[player_id]] = True
        own = own.reshape(H, W)
        pad = _pad1(own, False)
        near = np.zeros((H, W), dtype=bool)
        for dr in range(3):
            for dc in range(3):
                near |= pad[dr:dr + H, dc:dc + W]
        visible = near & inb & ~own
        unseen = inb & ~near

        # --- tile states (padding stays -1 → all-zero one-hot, matching how
        # the IL converter pads data); same precedence as _tile_to_state ---
        blank, city = terrain == _BLANK, terrain == _CITY
        gen, mtn = terrain == _GENERAL, terrain == _MOUNTAIN
        neutral = occ == -1
        enemy = ~neutral & (occ != player_id)
        seen_state = np.select(
            [blank & neutral, city & neutral, mtn, enemy & blank, enemy & city, enemy & gen],
            [2, 3, 4, 8, 9, 10], default=2)
        own_state = np.select([blank, city, gen], [5, 6, 7], default=5)
        state = np.where(own & inb, own_state,
                         np.where(visible, seen_state,
                                  np.where(unseen, np.where(blank | gen, 0, 1), -1)))
        memory = (self.seen_mountains[player_id]
                  if getattr(self, "memory_vision", True) else None)
        if memory is not None:
            sighted = (own | visible) & inb & mtn
            if sighted.any():
                memory.update(np.flatnonzero(sighted).tolist())
            if memory:
                # Sighted-before mountain: stays a mountain in this player's
                # observation instead of decaying to the generic fog-obstacle
                # state.
                mem = np.zeros(H * W, dtype=bool)
                mem[[i for i in memory if 0 <= i < H * W]] = True
                state = np.where(unseen & mem.reshape(H, W), 4, state)

        # --- grid: 11 one-hot channels + army (only for seen tiles) ---
        seen_army = (own | visible) & inb & ~mtn
        army_ch = np.where(seen_army, _log_army(np.where(seen_army, army, 0)),
                           np.float32(0.0))
        grid = torch.from_numpy(np.concatenate(
            [(state[None] == _STATE_IDS).astype(np.float32), army_ch[None]]))

        # --- globals ---
        stats = self._player_stats(player_id)
        opp_stats = self._player_stats(1 - player_id)
        glb = [
            math.log1p(stats["army"]) / 10.0,
            math.log1p(stats["land"]) / 10.0,
            math.log1p(opp_stats["army"]) / 10.0,
            math.log1p(opp_stats["land"]) / 10.0,
        ]
        if self.turn_globals:
            # Production clock (see _increment_armies): all land +1 every
            # 50 turns, cities/generals +1 every 2 turns.
            glb += [
                (self.turn % 50) / 50.0,
                float(self.turn % 2),
                self.turn / 500.0,
            ]
        globals_vec = torch.tensor(glb, dtype=torch.float32)

        # --- legal mask (owned tiles with army > 1) ---
        movable = own & (army > 1)
        legal = torch.from_numpy(movable[None].copy())

        # --- capture-feasibility labels (auxiliary supervision) ---
        # For each movable owned tile and direction onto a DEFENDED
        # neighbour (neutral city or enemy tile): would a full move capture
        # it, i.e. (src.army - 1) > dst.army? The single log-compressed army
        # channel makes this threshold nearly invisible in input space;
        # supervising it directly forces the encoder to represent the
        # comparison. Neighbours of owned tiles are always visible, so the
        # labels never leak fogged information.
        inb_p = _pad1(inb, False)
        ter_p = _pad1(terrain)
        occ_p = _pad1(occ, -1)
        army_p = _pad1(army)
        cap_mask_np = np.zeros((4, H, W), dtype=bool)
        cap_label_np = np.zeros((4, H, W), dtype=np.float32)
        for d in range(4):
            sl = (slice(1 + _DIR_DR[d], 1 + _DIR_DR[d] + H),
                  slice(1 + _DIR_DC[d], 1 + _DIR_DC[d] + W))
            t_ter, t_occ = ter_p[sl], occ_p[sl]
            defended = (t_occ == 1 - player_id) | ((t_occ == -1) & (t_ter == _CITY))
            m = movable & inb_p[sl] & (t_ter != _MOUNTAIN) & defended
            cap_mask_np[d] = m
            cap_label_np[d] = m & (army - 1 > army_p[sl])
        cap_mask = torch.from_numpy(cap_mask_np)
        cap_label = torch.from_numpy(cap_label_np)

        return {"grid": grid, "globals": globals_vec, "legal_mask": legal,
                "capture_label": cap_label, "capture_mask": cap_mask,
                "dir_mask": self.dir_valid}

    # ==================================================================
    # Internal: state encoding (same as simulator)
    # ==================================================================

    @staticmethod
    def _tile_to_state(tile: _Tile, vision: int, player_id: int) -> int:
        """Map (tile, vision) → state integer 0–10."""
        if vision == 0:
            if tile.terrain in (_BLANK, _GENERAL):
                return 0   # unseen blank
            return 1       # unseen not-blank

        if vision == 2:  # visible, not owned
            if tile.terrain == _BLANK and tile.occupied == -1:
                return 2   # seen blank
            if tile.terrain == _CITY and tile.occupied == -1:
                return 3   # seen uncaptured city
            if tile.terrain == _MOUNTAIN:
                return 4
            if tile.occupied != -1 and tile.occupied != player_id:
                if tile.terrain == _BLANK:
                    return 8
                if tile.terrain == _CITY:
                    return 9
                if tile.terrain == _GENERAL:
                    return 10
            # Edge case: visible own tile via neighbour
            return 2

        # vision == 1  (owned)
        if tile.terrain == _BLANK:
            return 5
        if tile.terrain == _CITY:
            return 6
        if tile.terrain == _GENERAL:
            return 7
        return 5  # fallback

    # ==================================================================
    # State snapshot (for replay recording)
    # ==================================================================

    def get_state_snapshot(self) -> Dict[str, List[List[int]]]:
        """Return full-information state and army grids for replay recording.

        Returns dict with:
            state: [H][W] int — tile state using same encoding as IL (0-10),
                   full-information view (P0 perspective: own=5-7, enemy=8-10)
            army:  [H][W] int — army counts (0 for empty/mountain/padding)
        """
        H, W = self.grid_h, self.grid_w
        rH, rW = self.real_h, self.real_w
        state = [[-1] * W for _ in range(H)]
        army_grid = [[0] * W for _ in range(H)]

        for r in range(rH):
            for c in range(rW):
                tile = self.map[r][c]
                if tile.terrain == _MOUNTAIN:
                    state[r][c] = 4
                elif tile.occupied == 0:
                    state[r][c] = {_BLANK: 5, _CITY: 6, _GENERAL: 7}[tile.terrain]
                elif tile.occupied == 1:
                    state[r][c] = {_BLANK: 8, _CITY: 9, _GENERAL: 10}[tile.terrain]
                elif tile.terrain == _CITY:
                    state[r][c] = 3  # uncaptured city
                else:
                    state[r][c] = 2  # blank
                army_grid[r][c] = tile.army

        return {"state": state, "army": army_grid}

    # ==================================================================
    # Internal: legal move check
    # ==================================================================

    def _has_legal_moves(self, player_id: int) -> bool:
        """Check if player has any tile with army > 1 (can make a move)."""
        for idx in self.player_land[player_id]:
            r, c = divmod(idx, self.grid_w)
            if self.map[r][c].army > 1:
                return True
        return False

    def get_valid_directions(self, source_idx: int) -> List[bool]:
        """Return [down, right, up, left] validity for a given source tile."""
        src_r, src_c = divmod(source_idx, self.grid_w)
        valid = []
        for d in range(4):
            dst_r = src_r + _DIR_DR[d]
            dst_c = src_c + _DIR_DC[d]
            if (0 <= dst_r < self.real_h and 0 <= dst_c < self.real_w
                    and self.map[dst_r][dst_c].terrain != _MOUNTAIN):
                valid.append(True)
            else:
                valid.append(False)
        return valid

    # ==================================================================
    # Internal: player stats
    # ==================================================================

    def _player_stats(self, player_id: int) -> Dict[str, int]:
        army = 0
        cities = 0
        for idx in self.player_land[player_id]:
            r, c = divmod(idx, self.grid_w)
            army += self.map[r][c].army
            if self.map[r][c].terrain in (_CITY, _GENERAL):
                cities += 1
        return {"land": len(self.player_land[player_id]), "army": army, "cities": cities}

    # ==================================================================
    # Internal: action → move
    # ==================================================================

    def _action_to_move(
        self, player_id: int, action: Optional[Dict[str, int]],
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Convert action dict to internal move format.

        Returns:
            (move_dict or None, was_illegal: bool)
            - action is None (pass): returns (None, False)
            - action is valid: returns (move_dict, False)
            - action is illegal: returns (None, True)
        """
        if action is None:
            return None, False

        src_idx = action["source_idx"]
        direction = action["direction"]
        is_50 = action["split"]

        src_r, src_c = divmod(src_idx, self.grid_w)
        dr, dc = _DIR_DR[direction], _DIR_DC[direction]
        dst_r, dst_c = src_r + dr, src_c + dc

        # Validate bounds (use real map size, not padded 25×25)
        if not (0 <= dst_r < self.real_h and 0 <= dst_c < self.real_w):
            return None, True

        # Validate: can't move into mountains
        if self.map[dst_r][dst_c].terrain == _MOUNTAIN:
            return None, True

        # Validate: must own source with army > 1
        src_tile = self.map[src_r][src_c]
        if src_tile.occupied != player_id or src_tile.army <= 1:
            return None, True

        dst_idx = dst_r * self.grid_w + dst_c
        return {
            "start": src_idx,
            "end": dst_idx,
            "index": player_id,
            "is50": is_50,
        }, False

    # ==================================================================
    # Internal: move execution (from simulator.py, verified)
    # ==================================================================

    def _execute_single_move(self, move: Dict[str, Any]) -> None:
        start_idx = move["start"]
        end_idx = move["end"]
        player_idx = move["index"]
        is_50 = move["is50"]

        sr, sc = divmod(start_idx, self.grid_w)
        er, ec = divmod(end_idx, self.grid_w)
        start_tile = self.map[sr][sc]
        end_tile = self.map[er][ec]

        if start_tile.occupied != player_idx or start_tile.army <= 1:
            return

        if is_50:
            army_to_move = start_tile.army // 2
        else:
            army_to_move = start_tile.army - 1
        army_remaining = start_tile.army - army_to_move

        # Trace the destination's PRE-move state: rewards need exact
        # per-move attribution (who damaged which garrison by how much) —
        # a pooled post-hoc diff mis-pays same-tick double attacks and
        # reads garrisons after _increment_armies.
        self._move_trace.append({
            "player": player_idx, "end": end_idx,
            "pre_occupied": end_tile.occupied, "pre_army": end_tile.army,
            "moved": army_to_move,
        })

        if end_tile.occupied == player_idx:
            end_tile.army += army_to_move
            start_tile.army = army_remaining
        else:
            if army_to_move > end_tile.army:
                # Capture
                if end_tile.occupied != -1:
                    if end_idx in self.player_land[end_tile.occupied]:
                        self.player_land[end_tile.occupied].remove(end_idx)
                self.player_land[player_idx].append(end_idx)
                end_tile.occupied = player_idx
                end_tile.army = army_to_move - end_tile.army
            else:
                end_tile.army -= army_to_move
            start_tile.army = army_remaining

    def _process_moves(
        self,
        move0: Optional[Dict],
        move1: Optional[Dict],
    ) -> None:
        moves = []
        if move0 is not None:
            moves.append((move0, 0))
        if move1 is not None:
            moves.append((move1, 1))

        if len(moves) == 0:
            return
        if len(moves) == 1:
            self._execute_single_move(moves[0][0])
            return

        # Two moves — determine priority (from simulator)
        m0, m1 = moves[0][0], moves[1][0]
        p0 = self._move_priority(m0, m1, 0)
        p1 = self._move_priority(m1, m0, 1)

        if p0 <= p1:
            first, second = m0, m1
        else:
            first, second = m1, m0
        self._execute_single_move(first)
        # Real rules: once a general falls the game is decided — the server
        # drops the loser's queued move. This also makes the winner of a
        # would-be "mutual capture" tick follow move-execution order instead
        # of _check_winner's scan order.
        if self._check_winner() is None:
            self._execute_single_move(second)

    def _move_priority(self, move: Dict, other: Dict, idx: int) -> Tuple:
        # Chase
        chase = 1
        if move["end"] == other["start"] and move["start"] != other["end"]:
            chase = 0
        # Defensive
        er, ec = divmod(move["end"], self.grid_w)
        defense = 0 if self.map[er][ec].occupied == move["index"] else 1
        # General attack
        gen_atk = 1 if move["end"] in self.generals else 0
        sr, sc = divmod(move["start"], self.grid_w)
        army = -self.map[sr][sc].army
        return (chase, defense, gen_atk, army, idx)

    # ==================================================================
    # Internal: army increment
    # ==================================================================

    def _increment_armies(self) -> None:
        if self.turn == 0:
            return
        # Cities + generals every 2 turns
        if self.turn % 2 == 0:
            for city_idx in self.cities:
                r, c = divmod(city_idx, self.grid_w)
                if self.map[r][c].occupied != -1:
                    self.map[r][c].army += 1
            for gen_idx in self.generals:
                r, c = divmod(gen_idx, self.grid_w)
                self.map[r][c].army += 1
        # All land every 50 turns
        if self.turn % 50 == 0:
            for pid in range(2):
                for idx in self.player_land[pid]:
                    r, c = divmod(idx, self.grid_w)
                    self.map[r][c].army += 1

    # ==================================================================
    # Internal: win check
    # ==================================================================

    def _check_winner(self) -> Optional[int]:
        for pid, gen_idx in enumerate(self.generals):
            r, c = divmod(gen_idx, self.grid_w)
            tile = self.map[r][c]
            if tile.occupied != -1 and tile.occupied != pid:
                return tile.occupied
        return None

    def _apply_general_capture(self, winner: int) -> None:
        """generals.io rules: capturing the general transfers the loser's
        whole territory to the winner with armies halved (rounded up) and
        turns the fallen general into a city."""
        loser = 1 - winner
        for idx in list(self.player_land[loser]):
            r, c = divmod(idx, self.grid_w)
            tile = self.map[r][c]
            tile.occupied = winner
            tile.army = (tile.army + 1) // 2
            self.player_land[winner].append(idx)
        self.player_land[loser] = []

        gen_idx = self.generals[loser]
        r, c = divmod(gen_idx, self.grid_w)
        gen_tile = self.map[r][c]
        if gen_tile.terrain == _GENERAL:
            gen_tile.terrain = _CITY
            if gen_idx not in self.cities:
                self.cities.append(gen_idx)

    # ==================================================================
    # Internal: map generation
    # ==================================================================

    def _generate_map(self) -> None:
        # Retry until the generals are mutually reachable — uniform mountain
        # placement can occasionally wall a general in, producing unwinnable
        # episodes of pure shaping noise.
        for _ in range(20):
            self._generate_map_once()
            if self._generals_connected():
                return
        # Practically unreachable fallback: keep the last map without mountains
        for idx in self.mountains:
            r, c = divmod(idx, self.grid_w)
            self.map[r][c] = _Tile()
        self.mountains = []

    def _generals_connected(self) -> bool:
        """BFS over non-mountain tiles between the two generals."""
        from collections import deque
        H, W = self.grid_h, self.grid_w
        start, goal = self.generals
        seen = {start}
        dq = deque([start])
        while dq:
            idx = dq.popleft()
            if idx == goal:
                return True
            r, c = divmod(idx, W)
            for d in range(4):
                nr, nc = r + _DIR_DR[d], c + _DIR_DC[d]
                if 0 <= nr < H and 0 <= nc < W:
                    nidx = nr * W + nc
                    if nidx not in seen and self.map[nr][nc].terrain != _MOUNTAIN:
                        seen.add(nidx)
                        dq.append(nidx)
        return False

    def _generate_map_once(self) -> None:
        H, W = self.grid_h, self.grid_w

        # Fresh map
        self.map = [[_Tile() for _ in range(W)] for _ in range(H)]
        self.player_land = [[], []]

        occupied: set = set()

        # Place generals: opposite quadrants with minimum distance
        for _ in range(200):
            r0 = random.randint(1, H // 3)
            c0 = random.randint(1, W // 3)
            r1 = random.randint(2 * H // 3, H - 2)
            c1 = random.randint(2 * W // 3, W - 2)
            if abs(r0 - r1) + abs(c0 - c1) >= max(H, W) // 2:
                break

        g0 = r0 * W + c0
        g1 = r1 * W + c1
        # 50% side swap so the learner (always player 0) doesn't acquire a
        # top-left positional prior — the replay-map path already swaps.
        if random.random() < 0.5:
            g0, g1 = g1, g0
        self.generals = [g0, g1]
        occupied.update(self.generals)

        for pid, gidx in enumerate(self.generals):
            r, c = divmod(gidx, W)
            self.map[r][c] = _Tile(_GENERAL, 1, pid)
            self.player_land[pid].append(gidx)

        # Place mountains
        self.mountains = []
        attempts = 0
        while len(self.mountains) < self.n_mountains and attempts < 500:
            idx = random.randint(0, H * W - 1)
            if idx not in occupied:
                r, c = divmod(idx, W)
                self.map[r][c] = _Tile(_MOUNTAIN, 0, -1)
                self.mountains.append(idx)
                occupied.add(idx)
            attempts += 1

        # Place cities
        self.cities = []
        attempts = 0
        while len(self.cities) < self.n_cities and attempts < 500:
            idx = random.randint(0, H * W - 1)
            if idx not in occupied:
                r, c = divmod(idx, W)
                army = random.randint(*self.city_army_range)
                self.map[r][c] = _Tile(_CITY, army, -1)
                self.cities.append(idx)
                occupied.add(idx)
            attempts += 1

    # ==================================================================
    # Replay-based map loading
    # ==================================================================

    def load_map_pool(self, replay_dir: str, filtered_file: str) -> int:
        """Load map data from filtered replays.

        Args:
            replay_dir: Directory with raw replay JSON files
            filtered_file: Path to filtered replay list (e.g., filtered/Human.exe.json)

        Returns:
            Number of maps loaded
        """
        import json as _json

        with open(filtered_file) as f:
            filtered = _json.load(f)

        self.map_pool = []
        n_invalid = 0
        for entry in filtered:
            replay_id = entry["id"]
            replay_path = os.path.join(replay_dir, f"{replay_id}.json")
            if not os.path.exists(replay_path):
                continue
            try:
                with open(replay_path) as f:
                    data = _json.load(f)
                # Only keep 1v1 maps (2 generals) that fit in grid
                if len(data.get("generals", [])) != 2:
                    continue
                if data["mapHeight"] > self.grid_h or data["mapWidth"] > self.grid_w:
                    continue
                # A few replays carry special-mode encodings (e.g. negative
                # cityArmies) or out-of-range tile indices; a bad garrison
                # poisons reset() (log1p of a negative army) mid-training,
                # so drop the whole map loudly rather than clamp silently.
                n_tiles = data["mapHeight"] * data["mapWidth"]
                generals = data["generals"]
                if (any(not (0 <= g < n_tiles) for g in generals)
                        or generals[0] == generals[1]
                        or any(not (0 <= c < n_tiles) for c in data.get("cities", []))
                        or any(not (0 <= m < n_tiles) for m in data.get("mountains", []))
                        or any(a < 0 for a in data.get("cityArmies", []))):
                    n_invalid += 1
                    continue
                self.map_pool.append({
                    "replay_id": replay_id,
                    "map_w": data["mapWidth"],
                    "map_h": data["mapHeight"],
                    "generals": data["generals"],
                    "cities": data.get("cities", []),
                    "city_armies": data.get("cityArmies", []),
                    "mountains": data.get("mountains", []),
                })
            except (KeyError, _json.JSONDecodeError):
                continue

        if n_invalid:
            print(f"  [map-pool] dropped {n_invalid} replays with invalid "
                  f"map data (bad tile indices or negative city armies)")
        return len(self.map_pool)

    def _generate_map_from_replay(self, map_data: Dict) -> None:
        """Initialize the map from real replay data.

        Places the original map in the top-left corner of the 25×25 grid,
        matching how the IL converter pads data.
        """
        H, W = self.grid_h, self.grid_w  # 25, 25
        orig_h = map_data["map_h"]
        orig_w = map_data["map_w"]

        # Store real bounds for move validation
        self.real_h = orig_h
        self.real_w = orig_w

        # Fresh 25×25 map (padding = blank tiles, same as IL data)
        self.map = [[_Tile() for _ in range(W)] for _ in range(H)]
        self.player_land = [[], []]

        # Helper: convert flat index from original map width to 25-wide
        def remap(flat_idx: int) -> Tuple[int, int, int]:
            """Returns (row, col, new_flat_idx_in_25_wide)."""
            r = flat_idx // orig_w
            c = flat_idx % orig_w
            return r, c, r * W + c

        # Place generals
        self.generals = []
        for pid, gen_orig in enumerate(map_data["generals"]):
            r, c, new_idx = remap(gen_orig)
            if 0 <= r < H and 0 <= c < W:
                self.map[r][c] = _Tile(_GENERAL, 1, pid)
                self.generals.append(new_idx)
                self.player_land[pid].append(new_idx)

        # Place mountains
        self.mountains = []
        for mtn_orig in map_data["mountains"]:
            r, c, new_idx = remap(mtn_orig)
            if 0 <= r < H and 0 <= c < W:
                self.map[r][c] = _Tile(_MOUNTAIN, 0, -1)
                self.mountains.append(new_idx)

        # Place cities
        self.cities = []
        city_armies = map_data.get("city_armies", [])
        for i, city_orig in enumerate(map_data["cities"]):
            r, c, new_idx = remap(city_orig)
            if 0 <= r < H and 0 <= c < W:
                army = city_armies[i] if i < len(city_armies) else 40
                self.map[r][c] = _Tile(_CITY, army, -1)
                self.cities.append(new_idx)
