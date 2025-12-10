"""
Replay simulator for generals.io.

Simulates the game from each player's perspective, generating state sequences
including fog of war, visible units, and actions taken.
"""

import json
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Tile:
    """Represents a single map tile."""
    terrain: int  # 0: blank, 1: city, 2: general, 3: mountain
    army: int
    occupied: int  # -1: unoccupied, 0-N: player index


@dataclass
class Player:
    """Represents a player state."""
    land: List[int] = field(default_factory=list)  # List of tile indices
    army: int = 0
    vision: List[List[int]] = field(default_factory=list)  # 0: unseen, 1: owned, 2: visible


@dataclass
class Turn:
    """Represents game state at a turn."""
    turn: int = 0
    state: List[List[int]] = field(default_factory=list)  # Map state (terrain + ownership)
    army: List[List[int]] = field(default_factory=list)  # Army counts
    start: int = -1  # Move start position
    end: int = -1  # Move end position
    is50: int = -1  # Whether move is 50%
    myLand: int = 0
    myArmy: int = 0
    opLand: int = 0
    opArmy: int = 0


class ReplaySimulator:
    """Simulates a generals.io replay from a player's perspective."""
    
    def __init__(self, replay_data: Dict[str, Any], player_name: str):
        """
        Initialize the simulator.
        
        Args:
            replay_data: Deserialized replay data
            player_name: Name of the player to simulate from perspective of
        """
        self.data = replay_data
        self.player_name = player_name
        
        # Map dimensions
        self.map_width = replay_data["mapWidth"]
        self.map_height = replay_data["mapHeight"]
        
        # Map features
        self.cities = replay_data["cities"]
        self.city_armies = replay_data["cityArmies"]
        self.generals = replay_data["generals"]
        self.mountains = replay_data["mountains"]
        
        # Game data
        self.moves = replay_data["moves"]
        self.usernames = replay_data["usernames"]
        
        # Find player ID
        try:
            self.player_id = self.usernames.index(player_name)
        except ValueError:
            raise ValueError(f"Player {player_name} not found in replay")
        
        # Initialize game state
        self.map: List[List[Tile]] = []
        self.players: List[Player] = []
        self.turns: List[Turn] = []
        self.move_index = 0
        
    def init_map(self):
        """Initialize the map with starting positions."""
        # Create empty map
        self.map = [
            [Tile(0, 0, -1) for _ in range(self.map_width)]
            for _ in range(self.map_height)
        ]
        
        # Initialize players
        self.players = [
            Player(
                land=[],
                army=0,
                vision=[[0 for _ in range(self.map_width)] for _ in range(self.map_height)]
            )
            for _ in range(len(self.usernames))
        ]
        
        # Place generals
        for i, general in enumerate(self.generals):
            row = general // self.map_width
            col = general % self.map_width
            self.map[row][col] = Tile(2, 1, i)
            self.players[i].land.append(general)
        
        # Place cities
        for i, city in enumerate(self.cities):
            row = city // self.map_width
            col = city % self.map_width
            self.map[row][col] = Tile(1, self.city_armies[i], -1)
        
        # Place mountains
        for mountain in self.mountains:
            row = mountain // self.map_width
            col = mountain % self.map_width
            self.map[row][col] = Tile(3, 0, -1)
    
    def process_move(self, move: Dict[str, Any]):
        """Process a single move."""
        start_idx = move["start"]
        end_idx = move["end"]
        player_idx = move["index"]
        is_50 = move["is50"]
        
        start_row = start_idx // self.map_width
        start_col = start_idx % self.map_width
        end_row = end_idx // self.map_width
        end_col = end_idx % self.map_width
        
        start_tile = self.map[start_row][start_col]
        end_tile = self.map[end_row][end_col]
        
        # Calculate armies to move
        if is_50:
            army_to_move = start_tile.army // 2
            army_remaining = start_tile.army - army_to_move
        else:
            army_to_move = start_tile.army - 1
            army_remaining = 1
        
        # Check if attacking own territory or enemy
        if end_tile.occupied == player_idx:
            # Moving to own territory
            end_tile.army += army_to_move
            start_tile.army = army_remaining
        else:
            # Attacking enemy or neutral
            if army_to_move > end_tile.army:
                # Successful capture
                if end_tile.occupied != -1:
                    # Remove from previous owner
                    self.players[end_tile.occupied].land.remove(end_idx)
                
                # Add to current player
                self.players[player_idx].land.append(end_idx)
                end_tile.occupied = player_idx
                end_tile.army = army_to_move - end_tile.army
            else:
                # Failed attack
                end_tile.army -= army_to_move
            
            start_tile.army = army_remaining
    
    def update_vision(self):
        """Update vision for all players."""
        for player in self.players:
            player.vision = [[0 for _ in range(self.map_width)] for _ in range(self.map_height)]
            player.army = 0
            
            for tile_idx in player.land:
                row = tile_idx // self.map_width
                col = tile_idx % self.map_width
                
                # Mark owned tile
                player.vision[row][col] = 1
                player.army += self.map[row][col].army
                
                # Mark adjacent tiles as visible
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), 
                              (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.map_height and 0 <= nc < self.map_width:
                        if player.vision[nr][nc] != 1:
                            player.vision[nr][nc] = 2
    
    def get_state_encoding(self, player_idx: int) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Get the encoded state from a player's perspective.
        
        State encoding:
        0: Unseen Blank
        1: Unseen not Blank
        2: Seen Blank
        3: Seen Uncaptured City
        4: Seen Mountain
        5: Seen Your Land
        6: Seen Your City
        7: Seen Your General
        8: Seen Opponent's Land
        9: Seen Opponent's City
        10: Seen Opponent's General
        """
        state = [[-1 for _ in range(self.map_width)] for _ in range(self.map_height)]
        army = [[-1 for _ in range(self.map_width)] for _ in range(self.map_height)]
        
        player = self.players[player_idx]
        
        for i in range(self.map_height):
            for j in range(self.map_width):
                tile = self.map[i][j]
                vision = player.vision[i][j]
                
                # Unseen areas
                if vision == 0:
                    if tile.terrain in [0, 2]:  # blank or general (hidden)
                        state[i][j] = 0
                    else:
                        state[i][j] = 1
                
                # Visible but not owned
                elif vision == 2:
                    if tile.terrain == 0 and tile.occupied == -1:
                        state[i][j] = 2
                    elif tile.terrain == 1 and tile.occupied == -1:
                        state[i][j] = 3
                    elif tile.terrain == 3:
                        state[i][j] = 4
                    elif tile.terrain == 0 and tile.occupied != -1 and tile.occupied != player_idx:
                        state[i][j] = 8
                    elif tile.terrain == 1 and tile.occupied != -1 and tile.occupied != player_idx:
                        state[i][j] = 9
                    elif tile.terrain == 2 and tile.occupied != -1 and tile.occupied != player_idx:
                        state[i][j] = 10
                    
                    # Set army count for visible tiles
                    if tile.terrain in [0, 1, 2]:
                        army[i][j] = tile.army
                
                # Owned
                elif vision == 1:
                    if tile.terrain == 0:
                        state[i][j] = 5
                    elif tile.terrain == 1:
                        state[i][j] = 6
                    elif tile.terrain == 2:
                        state[i][j] = 7
                    
                    army[i][j] = tile.army
        
        return state, army
    
    def increment_armies(self, turn: int):
        """Increment armies based on game rules."""
        # Cities and generals increment every 2 turns
        if turn % 2 == 0:
            for city in self.cities:
                row = city // self.map_width
                col = city % self.map_width
                if self.map[row][col].occupied != -1:
                    self.map[row][col].army += 1
            
            for general in self.generals:
                row = general // self.map_width
                col = general % self.map_width
                self.map[row][col].army += 1
        
        # All land increments every 50 turns
        if turn % 50 == 0:
            for player in self.players:
                for tile_idx in player.land:
                    row = tile_idx // self.map_width
                    col = tile_idx % self.map_width
                    self.map[row][col].army += 1
    
    def simulate(self) -> Dict[str, Any]:
        """
        Simulate the entire replay from the player's perspective.
        
        Returns:
            Dictionary containing game state sequences
        """
        self.init_map()
        
        # Storage for output
        first = []
        map_info = []
        armies_num = []
        info_myArmy = []
        info_myLand = []
        info_opArmy = []
        info_opLand = []
        move_start = []
        move_end = []
        move_is_50 = []
        
        # Get total turns
        max_turn = self.moves[-1]["turn"] if self.moves else 0
        
        move_idx = 0
        
        for turn in range(max_turn + 1):
            # Process moves for this turn
            player_move_start = -1
            player_move_end = -1
            player_move_is_50 = -1
            
            while move_idx < len(self.moves) and self.moves[move_idx]["turn"] == turn:
                move = self.moves[move_idx]
                
                # Record the move if it's from our player
                if move["index"] == self.player_id:
                    player_move_start = move["start"]
                    player_move_end = move["end"]
                    player_move_is_50 = move["is50"]
                
                self.process_move(move)
                move_idx += 1
            
            # Update vision and stats
            self.update_vision()
            
            # Get state encoding
            state, army = self.get_state_encoding(self.player_id)
            
            # Pad to 25x25
            padded_state = [[-1 for _ in range(25)] for _ in range(25)]
            padded_army = [[-1 for _ in range(25)] for _ in range(25)]
            
            for i in range(min(self.map_height, 25)):
                for j in range(min(self.map_width, 25)):
                    padded_state[i][j] = state[i][j]
                    padded_army[i][j] = army[i][j]
            
            # Calculate stats
            my_player = self.players[self.player_id]
            opponent_idx = 1 - self.player_id if len(self.players) == 2 else None
            
            if opponent_idx is not None:
                op_player = self.players[opponent_idx]
                op_army = op_player.army
                op_land = len(op_player.land)
            else:
                op_army = 0
                op_land = 0
            
            # Store turn data
            is_first = 1 if (self.player_id == 0 and turn % 2 == 0) or \
                           (self.player_id == 1 and turn % 2 == 1) else 0
            
            first.append(is_first)
            map_info.append(padded_state)
            armies_num.append(padded_army)
            info_myArmy.append(my_player.army)
            info_myLand.append(len(my_player.land))
            info_opArmy.append(op_army)
            info_opLand.append(op_land)
            
            # Store move info
            if player_move_start != -1:
                mx = player_move_start // self.map_width
                my = player_move_start % self.map_width
                move_start.append([mx, my])
                
                mx = player_move_end // self.map_width
                my = player_move_end % self.map_width
                move_end.append([mx, my])
            else:
                move_start.append([-1, -1])
                move_end.append([-1, -1])
            
            move_is_50.append(player_move_is_50)
            
            # Increment armies at end of turn
            self.increment_armies(turn)
        
        return {
            "id": self.data["id"],
            "player": self.player_name,
            "first": first,
            "map": map_info,
            "armies": armies_num,
            "info_myArmy": info_myArmy,
            "info_myLand": info_myLand,
            "info_opArmy": info_opArmy,
            "info_opLand": info_opLand,
            "move_start": move_start,
            "move_end": move_end,
            "move_is_50": move_is_50,
        }


def simulate_replay(
    replay_path: str,
    player_name: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simulate a replay from a player's perspective.
    
    Args:
        replay_path: Path to replay JSON file
        player_name: Name of player to simulate
        output_path: Optional path to save output
        
    Returns:
        Simulated game data
    """
    with open(replay_path, 'r', encoding='utf-8') as f:
        replay_data = json.load(f)
    
    simulator = ReplaySimulator(replay_data, player_name)
    result = simulator.simulate()
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
    
    return result

