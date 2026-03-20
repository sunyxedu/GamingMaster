"""
Interactive generals.io environment for RL self-play.

Game mechanics are adapted from ``data_processing/processing/simulator.py``
(the verified replay simulator) into a Gym-like interface that supports
two-player simultaneous-move self-play.

Observation format matches the model's ``step()`` input:
    grid       [C, H, W]     12-channel one-hot + army
    globals    [G]            4 log-normalised stats
    legal_mask [1, H, W]     owned tiles with army > 1

Reward design (all in [-1, 1] to match ValueHead Tanh):
    Terminal:  win +1.0, loss -1.0
    Dense:     0.02 × delta_heuristic (city-adjusted army ratio)
    Events:    capture city +0.004, illegal move -0.1
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

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
        # Legacy params (ignored, kept for backwards compat)
        max_turns: int = 9999,
        dense_reward: bool = False,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.N = grid_h * grid_w
        self.n_mountains = n_mountains
        self.n_cities = n_cities
        self.city_army_range = city_army_range

        # State (set by reset)
        self.map: List[List[_Tile]] = []
        self.generals: List[int] = []       # flat index per player
        self.cities: List[int] = []
        self.mountains: List[int] = []
        self.player_land: List[List[int]] = [[], []]
        self.turn: int = 0
        self.done: bool = True
        self.winner: Optional[int] = None

        # Heuristic tracking for dense reward
        self.prev_heuristic: Dict[int, float] = {0: 0.0, 1: 0.0}

    # ==================================================================
    # Heuristic (city-adjusted army ratio)
    # ==================================================================

    def _heuristic(self, pid: int) -> float:
        """Evaluate position for player `pid`. Returns value in [-1, 1]."""
        s = self._player_stats(pid)
        o = self._player_stats(1 - pid)
        if s["land"] == 0:
            return -1.0
        if o["land"] == 0:
            return 1.0
        adj_my = s["army"] + s["cities"] * 50
        adj_op = o["army"] + o["cities"] * 50
        ratio = adj_my / max(adj_op, 1)
        return float(math.tanh(ratio - 1.0))

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self, seed: Optional[int] = None) -> Dict[int, Dict[str, torch.Tensor]]:
        """Reset the environment and return initial observations."""
        if seed is not None:
            random.seed(seed)

        self._generate_map()
        self.turn = 0
        self.done = False
        self.winner = None

        # Initialize heuristic tracking
        self.prev_heuristic = {
            0: self._heuristic(0),
            1: self._heuristic(1),
        }

        return {0: self._get_observation(0), 1: self._get_observation(1)}

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

        # Convert actions → simulator format (detect illegal moves)
        move_0, illegal_0 = self._action_to_move(0, action_0)
        move_1, illegal_1 = self._action_to_move(1, action_1)

        if illegal_0:
            rewards[0] -= 0.1
            info["illegal"][0] = True
        if illegal_1:
            rewards[1] -= 0.1
            info["illegal"][1] = True

        # Track cities before move execution (for capture detection)
        cities_before = {
            0: set(idx for idx in self.player_land[0]
                   if self.map[idx // self.grid_w][idx % self.grid_w].terrain == _CITY),
            1: set(idx for idx in self.player_land[1]
                   if self.map[idx // self.grid_w][idx % self.grid_w].terrain == _CITY),
        }

        # Execute moves (with priority ordering)
        self._process_moves(move_0, move_1)

        # Increment armies
        self.turn += 1
        self._increment_armies()

        # Detect city captures
        for pid in [0, 1]:
            cities_after = set(idx for idx in self.player_land[pid]
                               if self.map[idx // self.grid_w][idx % self.grid_w].terrain == _CITY)
            new_cities = cities_after - cities_before[pid]
            if new_cities:
                rewards[pid] += 0.004 * len(new_cities)

        # Check win
        winner = self._check_winner()
        if winner is not None:
            self.done = True
            self.winner = winner
            rewards[winner] += 1.0
            rewards[1 - winner] += -1.0
            info["winner"] = winner

        # Dense reward: heuristic delta
        if not self.done:
            for pid in [0, 1]:
                curr_h = self._heuristic(pid)
                rewards[pid] += 0.02 * (curr_h - self.prev_heuristic[pid])
                self.prev_heuristic[pid] = curr_h

        obs = {0: self._get_observation(0), 1: self._get_observation(1)}
        return obs, rewards, self.done, info

    # ==================================================================
    # Observation
    # ==================================================================

    def _get_observation(self, player_id: int) -> Dict[str, torch.Tensor]:
        H, W = self.grid_h, self.grid_w

        # --- compute vision for this player ---
        vision = [[0] * W for _ in range(H)]
        for idx in self.player_land[player_id]:
            r, c = divmod(idx, W)
            vision[r][c] = 1   # owned
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and vision[nr][nc] != 1:
                        vision[nr][nc] = 2  # visible

        # --- build grid channels ---
        grid = torch.zeros(12, H, W, dtype=torch.float32)
        for r in range(H):
            for c in range(W):
                tile = self.map[r][c]
                v = vision[r][c]
                state = self._tile_to_state(tile, v, player_id)
                if 0 <= state <= 10:
                    grid[state, r, c] = 1.0
                # Army channel (only for visible tiles)
                if v >= 1 and tile.terrain != _MOUNTAIN:
                    grid[11, r, c] = math.log1p(tile.army) / 10.0

        # --- globals ---
        stats = self._player_stats(player_id)
        opp_stats = self._player_stats(1 - player_id)
        globals_vec = torch.tensor([
            math.log1p(stats["army"]) / 10.0,
            math.log1p(stats["land"]) / 10.0,
            math.log1p(opp_stats["army"]) / 10.0,
            math.log1p(opp_stats["land"]) / 10.0,
        ], dtype=torch.float32)

        # --- legal mask (owned tiles with army > 1) ---
        legal = torch.zeros(1, H, W, dtype=torch.bool)
        for idx in self.player_land[player_id]:
            r, c = divmod(idx, W)
            if self.map[r][c].army > 1:
                legal[0, r, c] = True

        return {"grid": grid, "globals": globals_vec, "legal_mask": legal}

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

        # Validate bounds
        if not (0 <= dst_r < self.grid_h and 0 <= dst_c < self.grid_w):
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
            self._execute_single_move(m0)
            self._execute_single_move(m1)
        else:
            self._execute_single_move(m1)
            self._execute_single_move(m0)

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

    # ==================================================================
    # Internal: map generation
    # ==================================================================

    def _generate_map(self) -> None:
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
