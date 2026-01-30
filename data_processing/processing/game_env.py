"""
Game Environment for self-play.

Provides a game environment that can:
- Initialize from MapData (extracted from replays)
- Execute actions and return new states
- Check for terminal conditions
- Generate legal moves
"""

import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .models import Tile, Player, MapData


@dataclass
class GameState:
    """
    Complete game state for self-play.

    All tensors are ready for neural network input.
    """
    # Map state
    map_tiles: List[List[Tile]]  # Raw tile data
    map_width: int
    map_height: int

    # Player info
    players: List[Player]
    current_player: int  # 0 or 1
    turn: int

    # Special positions
    generals: List[int]  # Flat indices
    cities: List[int]
    mountains: List[int]

    # Cached tensors (for efficiency)
    _grid_tensor: Optional[torch.Tensor] = None
    _globals_tensor: Optional[torch.Tensor] = None
    _legal_mask_tensor: Optional[torch.Tensor] = None

    def get_grid_tensor(self) -> torch.Tensor:
        """Get encoded grid tensor [12, 25, 25]."""
        if self._grid_tensor is None:
            self._grid_tensor = self._encode_grid()
        return self._grid_tensor

    def get_globals_tensor(self) -> torch.Tensor:
        """Get global features tensor [4]."""
        if self._globals_tensor is None:
            self._globals_tensor = self._encode_globals()
        return self._globals_tensor

    def get_legal_mask_tensor(self) -> torch.Tensor:
        """Get legal source mask tensor [1, 25, 25]."""
        if self._legal_mask_tensor is None:
            self._legal_mask_tensor = self._compute_legal_mask()
        return self._legal_mask_tensor

    def _encode_grid(self) -> torch.Tensor:
        """
        Encode grid as tensor.

        Channels:
        0-10: One-hot state encoding
        11: Army count (log normalized)
        """
        grid = torch.zeros(12, 25, 25)
        player = self.players[self.current_player]

        for i in range(min(self.map_height, 25)):
            for j in range(min(self.map_width, 25)):
                tile = self.map_tiles[i][j]
                vision = player.vision[i][j]

                # Compute state code
                state_code = self._get_state_code(tile, vision, i, j)
                if 0 <= state_code <= 10:
                    grid[state_code, i, j] = 1.0

                # Army channel
                if vision > 0 and tile.army > 0:
                    grid[11, i, j] = np.log1p(tile.army) / 10.0

        return grid

    def _get_state_code(self, tile: Tile, vision: int, row: int, col: int) -> int:
        """Get state encoding code for a tile."""
        flat_idx = row * self.map_width + col

        if vision == 0:  # Unseen
            if tile.terrain in [0, 2]:
                return 0  # Unseen blank
            else:
                return 1  # Unseen not blank

        elif vision == 2:  # Visible but not owned
            if tile.terrain == 0 and tile.occupied == -1:
                return 2  # Seen blank
            elif tile.terrain == 1 and tile.occupied == -1:
                return 3  # Seen uncaptured city
            elif tile.terrain == 3:
                return 4  # Seen mountain
            elif tile.occupied != -1 and tile.occupied != self.current_player:
                if tile.terrain == 0:
                    return 8  # Opponent land
                elif tile.terrain == 1:
                    return 9  # Opponent city
                elif tile.terrain == 2:
                    return 10  # Opponent general

        elif vision == 1:  # Owned
            if tile.terrain == 0:
                return 5  # Your land
            elif tile.terrain == 1:
                return 6  # Your city
            elif tile.terrain == 2:
                return 7  # Your general

        return 0  # Default

    def _encode_globals(self) -> torch.Tensor:
        """Encode global features [myArmy, myLand, opArmy, opLand]."""
        my_player = self.players[self.current_player]
        op_idx = 1 - self.current_player
        op_player = self.players[op_idx] if len(self.players) > 1 else None

        my_army = my_player.army
        my_land = len(my_player.land)
        op_army = op_player.army if op_player else 0
        op_land = len(op_player.land) if op_player else 0

        return torch.tensor([
            np.log1p(my_army) / 10.0,
            np.log1p(my_land) / 10.0,
            np.log1p(op_army) / 10.0,
            np.log1p(op_land) / 10.0
        ], dtype=torch.float32)

    def _compute_legal_mask(self) -> torch.Tensor:
        """
        Compute legal source positions mask.

        The mask is in 25x25 format, matching the grid tensor.
        tile_idx in player.land is computed as row * map_width + col
        where row and col are in the actual map coordinates.
        """
        mask = torch.zeros(1, 25, 25, dtype=torch.bool)
        player = self.players[self.current_player]

        for tile_idx in player.land:
            # Convert tile_idx back to row/col using actual map width
            row = tile_idx // self.map_width
            col = tile_idx % self.map_width
            if row < 25 and col < 25:
                tile = self.map_tiles[row][col]
                if tile.army > 1:
                    # Mark as legal in the 25x25 grid
                    # Note: tile_idx for 25x25 grid = row * 25 + col
                    mask[0, row, col] = True

        return mask

    def invalidate_cache(self):
        """Invalidate cached tensors (call after state changes)."""
        self._grid_tensor = None
        self._globals_tensor = None
        self._legal_mask_tensor = None


class GameEnvironment:
    """
    Game environment for self-play.

    Supports:
    - Initializing from MapData (extracted from replays)
    - Executing actions
    - Checking terminal conditions
    - Getting state tensors for neural network
    """

    # Direction offsets: down, right, up, left
    DIR_OFFSETS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self):
        self.map_width: int = 0
        self.map_height: int = 0
        self.state: Optional[GameState] = None

    def reset(self, map_data: MapData) -> GameState:
        """
        Initialize game from MapData.

        Args:
            map_data: Map configuration (extracted from replay via MapData.from_replay())

        Returns:
            Initial game state
        """
        self.map_width = map_data.width
        self.map_height = map_data.height

        # Initialize empty map
        map_tiles = [
            [Tile(0, 0, -1) for _ in range(self.map_width)]
            for _ in range(self.map_height)
        ]

        # Place mountains
        for mountain in map_data.mountains:
            row = mountain // self.map_width
            col = mountain % self.map_width
            map_tiles[row][col] = Tile(3, 0, -1)  # terrain=3 (mountain)

        # Place cities
        for i, city in enumerate(map_data.cities):
            row = city // self.map_width
            col = city % self.map_width
            map_tiles[row][col] = Tile(1, map_data.city_armies[i], -1)  # terrain=1 (city)

        # Initialize players (2 players for 1v1)
        num_players = len(map_data.generals)
        players = [
            Player(
                name=f"Player{i}",
                land=[],
                army=0,
                vision=[[0] * self.map_width for _ in range(self.map_height)]
            )
            for i in range(num_players)
        ]

        # Place generals
        for i, general in enumerate(map_data.generals):
            row = general // self.map_width
            col = general % self.map_width
            map_tiles[row][col] = Tile(2, 1, i)  # terrain=2 (general), army=1, owner=i
            players[i].land.append(general)

        # Create state
        self.state = GameState(
            map_tiles=map_tiles,
            map_width=self.map_width,
            map_height=self.map_height,
            players=players,
            current_player=0,
            turn=0,
            generals=list(map_data.generals),
            cities=list(map_data.cities),
            mountains=list(map_data.mountains)
        )

        # Update vision
        self._update_vision()

        return self.state

    def step(
        self,
        action: Tuple[int, int, int],
        other_action: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Execute action and return new state.

        Args:
            action: (source_idx, dir_id, ratio_id) for current player
            other_action: Optional action for opponent (for simultaneous play)

        Returns:
            new_state: Updated game state
            reward: Reward for current player (+1 win, -1 loss, 0 ongoing)
            done: Whether game is over
            info: Additional information
        """
        if self.state is None:
            raise ValueError("Game not initialized. Call reset first.")

        current_player = self.state.current_player

        # Convert actions to move format
        move1 = self._action_to_move(action, current_player)
        move2 = self._action_to_move(other_action, 1 - current_player) if other_action else None

        # Execute moves
        self._process_moves(move1, move2)

        # Increment turn and armies
        self.state.turn += 1
        self._increment_armies()

        # Update vision
        self._update_vision()

        # Switch player (for alternating turns in self-play)
        self.state.current_player = 1 - self.state.current_player

        # Invalidate cached tensors
        self.state.invalidate_cache()

        # Check terminal
        winner = self._check_winner()
        done = winner is not None

        if done:
            reward = 1.0 if winner == current_player else -1.0
        else:
            reward = 0.0

        info = {
            'winner': winner,
            'turn': self.state.turn
        }

        return self.state, reward, done, info

    def _action_to_move(
        self,
        action: Optional[Tuple[int, int, int]],
        player_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Convert action tuple to move dict.

        Note: source_idx is in 25x25 format (row * 25 + col) from neural network.
        We convert it back to actual map coordinates for game execution.
        """
        if action is None:
            return None

        source_idx_25, dir_id, ratio_id = action

        # Convert from 25x25 format to actual row/col
        source_row = source_idx_25 // 25
        source_col = source_idx_25 % 25

        # Convert to actual map index
        source_idx = source_row * self.map_width + source_col

        dy, dx = self.DIR_OFFSETS[dir_id]
        dest_row = source_row + dy
        dest_col = source_col + dx
        dest_idx = dest_row * self.map_width + dest_col

        return {
            'start': source_idx,
            'end': dest_idx,
            'index': player_idx,
            'is50': ratio_id == 1
        }

    def _process_moves(
        self,
        move1: Optional[Dict[str, Any]],
        move2: Optional[Dict[str, Any]]
    ):
        """Process moves with priority ordering."""
        moves = []
        if move1:
            moves.append(move1)
        if move2:
            moves.append(move2)

        # Simple ordering: process in order for now
        # TODO: Implement full priority logic from ReplaySimulator
        for move in moves:
            self._execute_move(move)

    def _execute_move(self, move: Dict[str, Any]):
        """Execute a single move."""
        start_idx = move['start']
        end_idx = move['end']
        player_idx = move['index']
        is_50 = move['is50']

        start_row = start_idx // self.map_width
        start_col = start_idx % self.map_width
        end_row = end_idx // self.map_width
        end_col = end_idx % self.map_width

        # Bounds check
        if not (0 <= end_row < self.map_height and 0 <= end_col < self.map_width):
            return

        start_tile = self.state.map_tiles[start_row][start_col]
        end_tile = self.state.map_tiles[end_row][end_col]

        # Validate
        if start_tile.occupied != player_idx or start_tile.army <= 1:
            return
        if end_tile.terrain == 3:  # Mountain
            return

        # Calculate armies
        if is_50:
            army_to_move = start_tile.army // 2
            army_remaining = start_tile.army - army_to_move
        else:
            army_to_move = start_tile.army - 1
            army_remaining = 1

        # Execute
        if end_tile.occupied == player_idx:
            # Own territory
            end_tile.army += army_to_move
            start_tile.army = army_remaining
        else:
            # Attack
            if army_to_move > end_tile.army:
                # Capture
                if end_tile.occupied != -1:
                    if end_idx in self.state.players[end_tile.occupied].land:
                        self.state.players[end_tile.occupied].land.remove(end_idx)

                self.state.players[player_idx].land.append(end_idx)
                end_tile.occupied = player_idx
                end_tile.army = army_to_move - end_tile.army
            else:
                # Failed attack
                end_tile.army -= army_to_move

            start_tile.army = army_remaining

    def _increment_armies(self):
        """Increment armies based on game rules."""
        turn = self.state.turn
        if turn == 0:
            return

        # Cities and generals every 2 turns
        if turn % 2 == 0:
            for city in self.state.cities:
                row, col = city // self.map_width, city % self.map_width
                tile = self.state.map_tiles[row][col]
                if tile.occupied != -1:
                    tile.army += 1

            for general in self.state.generals:
                row, col = general // self.map_width, general % self.map_width
                self.state.map_tiles[row][col].army += 1

        # All land every 50 turns
        if turn % 50 == 0:
            for player in self.state.players:
                for tile_idx in player.land:
                    row, col = tile_idx // self.map_width, tile_idx % self.map_width
                    self.state.map_tiles[row][col].army += 1

    def _update_vision(self):
        """Update vision for all players."""
        for player in self.state.players:
            player.vision = [[0] * self.map_width for _ in range(self.map_height)]
            player.army = 0

            for tile_idx in player.land:
                row = tile_idx // self.map_width
                col = tile_idx % self.map_width

                player.vision[row][col] = 1
                player.army += self.state.map_tiles[row][col].army

                # Adjacent tiles visible
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.map_height and 0 <= nc < self.map_width:
                            if player.vision[nr][nc] != 1:
                                player.vision[nr][nc] = 2

    def _check_winner(self) -> Optional[int]:
        """Check if game has a winner."""
        for i, general in enumerate(self.state.generals):
            row, col = general // self.map_width, general % self.map_width
            tile = self.state.map_tiles[row][col]
            if tile.occupied != -1 and tile.occupied != i:
                return tile.occupied
        return None

    def get_legal_actions(self) -> List[Tuple[int, int, int]]:
        """
        Get all legal actions for current player.

        Returns:
            List of (source_idx, dir_id, ratio_id) tuples
        """
        if self.state is None:
            return []

        actions = []
        player = self.state.players[self.state.current_player]

        for tile_idx in player.land:
            # tile_idx is row * map_width + col in actual map coordinates
            row = tile_idx // self.map_width
            col = tile_idx % self.map_width
            tile = self.state.map_tiles[row][col]

            if tile.army <= 1:
                continue

            # Convert to 25x25 format for neural network
            source_idx_25 = row * 25 + col

            for dir_id, (dy, dx) in enumerate(self.DIR_OFFSETS):
                dest_row, dest_col = row + dy, col + dx

                # Bounds check
                if not (0 <= dest_row < self.map_height and 0 <= dest_col < self.map_width):
                    continue

                # Mountain check
                if self.state.map_tiles[dest_row][dest_col].terrain == 3:
                    continue

                # Add both ratio options (using 25x25 format source_idx)
                for ratio_id in [0, 1]:
                    actions.append((source_idx_25, dir_id, ratio_id))

        return actions

    def clone(self) -> 'GameEnvironment':
        """Create a deep copy of the environment."""
        new_env = GameEnvironment()
        new_env.map_width = self.map_width
        new_env.map_height = self.map_height
        if self.state is not None:
            new_env.state = GameState(
                map_tiles=[[Tile(t.terrain, t.army, t.occupied) for t in row]
                          for row in self.state.map_tiles],
                map_width=self.state.map_width,
                map_height=self.state.map_height,
                players=[
                    Player(
                        name=p.name,
                        land=list(p.land),
                        army=p.army,
                        vision=[list(row) for row in p.vision]
                    )
                    for p in self.state.players
                ],
                current_player=self.state.current_player,
                turn=self.state.turn,
                generals=list(self.state.generals),
                cities=list(self.state.cities),
                mountains=list(self.state.mountains)
            )
        return new_env

    def get_state_for_nn(self) -> Dict[str, torch.Tensor]:
        """
        Get state tensors ready for neural network.

        Returns:
            Dictionary with grid, globals, legal_mask, tick
        """
        if self.state is None:
            raise ValueError("Game not initialized")

        return {
            'grid': self.state.get_grid_tensor(),
            'globals': self.state.get_globals_tensor(),
            'legal_mask': self.state.get_legal_mask_tensor(),
            'tick': self.state.turn
        }


def is_legal(
    grid: torch.Tensor,
    action: Tuple[int, int, int],
    grid_height: int = 25,
    grid_width: int = 25
) -> bool:
    """
    Check if an action is legal given the current player's grid view.

    This is used for computing illegal action penalty during training.
    The model should learn to predict legal actions through this penalty,
    enabling better generalization without explicit masking.

    Args:
        grid: Grid tensor [12, H, W] or [C, H, W] representing current player's view
              Channel 5: Your land
              Channel 6: Your city
              Channel 7: Your general
              Channel 4: Mountain
              Channel 11: Army count (log normalized)
        action: (source_idx, dir_id, ratio_id) tuple
                source_idx is in 25x25 format (row * 25 + col)
        grid_height: Grid height (default 25)
        grid_width: Grid width (default 25)

    Returns:
        True if action is legal, False otherwise
    """
    source_idx, dir_id, ratio_id = action

    # Convert source_idx to row/col
    source_row = source_idx // grid_width
    source_col = source_idx % grid_width

    # Check bounds for source
    if not (0 <= source_row < grid_height and 0 <= source_col < grid_width):
        return False

    # Check if source is owned by current player (channels 5, 6, 7)
    is_own_land = grid[5, source_row, source_col] > 0.5
    is_own_city = grid[6, source_row, source_col] > 0.5
    is_own_general = grid[7, source_row, source_col] > 0.5

    if not (is_own_land or is_own_city or is_own_general):
        return False

    # Check if source has army > 1 (channel 11 is log normalized army)
    # log1p(army) / 10 > log1p(1) / 10 ≈ 0.0693
    army_value = grid[11, source_row, source_col].item()
    if army_value <= 0.0693:  # log1p(1) / 10
        return False

    # Direction offsets: down, right, up, left
    dir_offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    if dir_id < 0 or dir_id >= 4:
        return False

    dy, dx = dir_offsets[dir_id]
    dest_row = source_row + dy
    dest_col = source_col + dx

    # Check destination bounds
    if not (0 <= dest_row < grid_height and 0 <= dest_col < grid_width):
        return False

    # Check if destination is mountain (channel 4)
    is_mountain = grid[4, dest_row, dest_col] > 0.5
    if is_mountain:
        return False

    # Check ratio_id is valid
    if ratio_id not in [0, 1]:
        return False

    return True


def is_legal_batch(
    grid: torch.Tensor,
    source_idx: torch.Tensor,
    dir_idx: torch.Tensor,
    ratio_idx: torch.Tensor,
    grid_height: int = 25,
    grid_width: int = 25
) -> torch.Tensor:
    """
    Batch version of is_legal for efficient training penalty computation.

    Args:
        grid: Grid tensor [B, 12, H, W] or [B, T, 12, H, W]
        source_idx: Source indices [B] or [B, T]
        dir_idx: Direction indices [B] or [B, T]
        ratio_idx: Ratio indices [B] or [B, T] (currently not checked, always valid if 0 or 1)
        grid_height: Grid height (default 25)
        grid_width: Grid width (default 25)

    Returns:
        Boolean tensor [B] or [B, T] indicating legal actions
    """
    device = grid.device

    # Handle sequence dimension
    if grid.dim() == 5:  # [B, T, C, H, W]
        B, T, C, H, W = grid.shape
        grid = grid.view(B * T, C, H, W)
        source_idx = source_idx.view(-1)
        dir_idx = dir_idx.view(-1)
        has_seq = True
    else:
        B = grid.shape[0]
        T = 1
        has_seq = False

    batch_size = grid.shape[0]

    # Convert source_idx to row/col
    source_row = source_idx // grid_width
    source_col = source_idx % grid_width

    # Check source bounds
    source_in_bounds = (
        (source_row >= 0) & (source_row < grid_height) &
        (source_col >= 0) & (source_col < grid_width)
    )

    # Clamp for safe indexing
    safe_row = source_row.clamp(0, grid_height - 1)
    safe_col = source_col.clamp(0, grid_width - 1)

    # Check if source is owned (channels 5, 6, 7)
    batch_idx = torch.arange(batch_size, device=device)
    is_own_land = grid[batch_idx, 5, safe_row, safe_col] > 0.5
    is_own_city = grid[batch_idx, 6, safe_row, safe_col] > 0.5
    is_own_general = grid[batch_idx, 7, safe_row, safe_col] > 0.5
    source_owned = is_own_land | is_own_city | is_own_general

    # Check army > 1 (log1p(1)/10 ≈ 0.0693)
    army_value = grid[batch_idx, 11, safe_row, safe_col]
    has_army = army_value > 0.0693

    # Direction offsets: down, right, up, left
    dir_offsets = torch.tensor([
        [1, 0],   # down
        [0, 1],   # right
        [-1, 0],  # up
        [0, -1]   # left
    ], device=device, dtype=torch.long)

    # Clamp dir_idx for safe indexing
    safe_dir = dir_idx.clamp(0, 3)
    dy = dir_offsets[safe_dir, 0]
    dx = dir_offsets[safe_dir, 1]

    dest_row = source_row + dy
    dest_col = source_col + dx

    # Check destination bounds
    dest_in_bounds = (
        (dest_row >= 0) & (dest_row < grid_height) &
        (dest_col >= 0) & (dest_col < grid_width)
    )

    # Check mountain (channel 4) for in-bounds destinations
    safe_dest_row = dest_row.clamp(0, grid_height - 1)
    safe_dest_col = dest_col.clamp(0, grid_width - 1)
    is_mountain = grid[batch_idx, 4, safe_dest_row, safe_dest_col] > 0.5

    # Combine all checks
    is_legal = (
        source_in_bounds &
        source_owned &
        has_army &
        (dir_idx >= 0) & (dir_idx < 4) &
        dest_in_bounds &
        ~is_mountain
    )

    if has_seq:
        is_legal = is_legal.view(B, T)

    return is_legal


def load_replay_as_env(replay_path: str) -> GameEnvironment:
    """
    Load a replay file and create environment.

    Args:
        replay_path: Path to replay JSON

    Returns:
        Initialized game environment
    """
    with open(replay_path, 'r') as f:
        replay_data = json.load(f)

    map_data = MapData.from_replay(replay_data)
    env = GameEnvironment()
    env.reset(map_data)
    return env
