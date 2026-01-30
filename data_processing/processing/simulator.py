"""
Replay simulator for generals.io.

Simulates the game from each player's perspective, generating state sequences
including fog of war, visible units, and actions taken.
"""

import json
import math
from typing import List, Dict, Any, Tuple, Optional
from data_processing.processing.models import Tile, Player, Turn


class ReplaySimulator:
    """Simulates a generals.io replay from a player's perspective."""

    # Standard map size for neural network input
    TARGET_SIZE = 25

    def __init__(self, replay_data: Dict[str, Any], player_name: str):
        """
        Initialize the simulator.

        Args:
            replay_data: Deserialized replay data
            player_name: Name of the player to simulate from perspective of
        """
        self.data = replay_data
        self.player_name = player_name

        # Original map dimensions (for index conversion)
        self.orig_width = replay_data["mapWidth"]
        self.orig_height = replay_data["mapHeight"]

        # Padded map dimensions (25x25)
        self.map_width = self.TARGET_SIZE
        self.map_height = self.TARGET_SIZE

        # Convert flat indices from original to padded coordinates
        def convert_idx(flat_idx: int) -> int:
            row = flat_idx // self.orig_width
            col = flat_idx % self.orig_width
            return row * self.TARGET_SIZE + col

        self._convert_idx = convert_idx

        # Map features (converted to 25x25 coordinates)
        self.cities = [convert_idx(c) for c in replay_data["cities"]]
        self.city_armies = replay_data["cityArmies"]
        self.generals = [convert_idx(g) for g in replay_data["generals"]]
        self.mountains = [convert_idx(m) for m in replay_data["mountains"]]
        
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
                name=self.usernames[i],
                land=[],
                army=0,
                vision=[[0 for _ in range(self.map_width)] for _ in range(self.map_height)]
            )
            for i in range(len(self.usernames))
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
    
    def _execute_single_move(self, move: Dict[str, Any]):
        """Execute a single move on the map."""
        # Convert from original coordinates to padded 25x25 coordinates
        start_idx = self._convert_idx(move["start"])
        end_idx = self._convert_idx(move["end"])
        player_idx = move["index"]
        is_50 = move["is50"]

        start_row = start_idx // self.map_width
        start_col = start_idx % self.map_width
        end_row = end_idx // self.map_width
        end_col = end_idx % self.map_width

        start_tile = self.map[start_row][start_col]
        end_tile = self.map[end_row][end_col]

        # Validate move: must own the start tile and have army > 1
        if start_tile.occupied != player_idx or start_tile.army <= 1:
            return

        # Calculate armies to move
        if is_50:
            army_to_move = start_tile.army // 2
            army_remaining = start_tile.army - army_to_move
        else:
            army_to_move = start_tile.army - 1
            army_remaining = 1

        # Check if moving to own territory or attacking
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
                    if end_idx in self.players[end_tile.occupied].land:
                        self.players[end_tile.occupied].land.remove(end_idx)

                # Add to current player
                self.players[player_idx].land.append(end_idx)
                end_tile.occupied = player_idx
                end_tile.army = army_to_move - end_tile.army
            else:
                # Failed attack
                end_tile.army -= army_to_move

            start_tile.army = army_remaining

    def _is_chase_move(self, chaser: Dict[str, Any], target: Dict[str, Any]) -> bool:
        """Check if chaser is chasing target (chaser.end == target.start and not swapping)."""
        # Compare in original coordinates (no need to convert for equality check)
        return chaser["end"] == target["start"] and chaser["start"] != target["end"]

    def _is_defensive_move(self, move: Dict[str, Any]) -> bool:
        """Check if move is defensive (friendly->friendly)."""
        end_idx = self._convert_idx(move["end"])
        end_row = end_idx // self.map_width
        end_col = end_idx % self.map_width
        end_tile = self.map[end_row][end_col]
        return end_tile.occupied == move["index"]

    def _is_attacking_general(self, move: Dict[str, Any]) -> bool:
        """Check if move is attacking a general."""
        end_idx = self._convert_idx(move["end"])
        return end_idx in self.generals

    def check_winner(self) -> Optional[int]:
        """
        Check if the game has a winner.

        A player wins by capturing an opponent's general.
        Returns the winner's player index, or None if game is ongoing.
        """
        for i, general in enumerate(self.generals):
            row = general // self.map_width
            col = general % self.map_width
            tile = self.map[row][col]
            # If someone else owns this general tile, that player wins
            if tile.occupied != -1 and tile.occupied != i:
                return tile.occupied
        return None

    def _get_start_tile_army(self, move: Dict[str, Any]) -> int:
        """Get the army count at the start tile of a move."""
        start_idx = self._convert_idx(move["start"])
        start_row = start_idx // self.map_width
        start_col = start_idx % self.map_width
        return self.map[start_row][start_col].army

    def _get_move_priority(self, move: Dict[str, Any], other_move: Optional[Dict[str, Any]], original_index: int) -> Tuple[int, int, int, int, int]:
        """
        Calculate move priority tuple for sorting.
        Lower values = higher priority (executed first).

        Priority order:
        1. Chase moves (highest priority) - priority 0
        2. Defensive moves - priority 1
        3. Non-general-attacking moves by army size (larger first) - priority 2
        4. General-attacking moves by army size (larger first) - priority 3
        5. Original order as final tiebreaker

        Returns: (chase_priority, defense_priority, attack_priority, army_tiebreak, original_index)
        """
        # Priority 1: Chase moves (chasing the other move)
        chase_priority = 1
        if other_move is not None and self._is_chase_move(move, other_move):
            chase_priority = 0

        # Priority 2: Defensive moves
        defense_priority = 1
        if self._is_defensive_move(move):
            defense_priority = 0

        # Priority 3: Larger tiles first, but general attacks come last
        attacking_general = self._is_attacking_general(move)
        army = self._get_start_tile_army(move)

        if attacking_general:
            # General attacks have lower priority (higher number)
            attack_priority = 1
        else:
            attack_priority = 0

        # Negative army so larger armies sort first
        army_tiebreak = -army

        return chase_priority, defense_priority, attack_priority, army_tiebreak, original_index

    def process_moves(self, move1: Optional[Dict[str, Any]], move2: Optional[Dict[str, Any]]):
        """
        Process moves for a turn with proper priority ordering.

        Priority rules:
        1. Chase moves - highest priority (catch fleeing target in 1 move)
        2. Defensive moves (friendly->friendly) - beat attackers to friendly tiles
        3. Larger tile first (affects general capture tiebreaks), general attacks last
        4. General trades between players = tie (not implemented yet)
        5. Original move order as final tiebreaker
        """
        moves = []
        if move1 is not None:
            moves.append((move1, 0))
        if move2 is not None:
            moves.append((move2, 1))

        if len(moves) == 0:
            return

        if len(moves) == 1:
            self._execute_single_move(moves[0][0])
            return

        # Two moves - determine order based on priority
        m1, idx1 = moves[0]
        m2, idx2 = moves[1]

        # Calculate priorities
        priority1 = self._get_move_priority(m1, m2, idx1)
        priority2 = self._get_move_priority(m2, m1, idx2)

        # Sort by priority (lower = first)
        if priority1 <= priority2:
            self._execute_single_move(m1)
            self._execute_single_move(m2)
        else:
            self._execute_single_move(m2)
            self._execute_single_move(m1)
    
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
        # No increment on turn 0
        if turn == 0:
            return

        # Cities and generals increment every 2 turns (turn 2, 4, 6, ...)
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

        # All land increments every 50 turns (turn 50, 100, 150, ...)
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
        
        for turn in range(0, max_turn + 2):
            # Process moves for this turn
            player_move_start = -1
            player_move_end = -1
            player_move_is_50 = -1

            turn_moves = [None, None]
            cnt = 0
            while move_idx < len(self.moves) and self.moves[move_idx]["turn"] + 1 == turn:
                move = self.moves[move_idx]

                # Record the move if it's from our player
                if move["index"] == self.player_id:
                    player_move_start = move["start"]
                    player_move_end = move["end"]
                    player_move_is_50 = move["is50"]
                if cnt < 2:
                    turn_moves[cnt] = move
                cnt += 1
                move_idx += 1

            self.process_moves(turn_moves[0], turn_moves[1])

            # Increment armies at end of turn
            self.increment_armies(turn)

            # Update vision and stats
            self.update_vision()
            
            # Get state encoding (already 25x25)
            state, army = self.get_state_encoding(self.player_id)

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
            map_info.append(state)
            armies_num.append(army)
            info_myArmy.append(my_player.army)
            info_myLand.append(len(my_player.land))
            info_opArmy.append(op_army)
            info_opLand.append(op_land)
            
            # Store move info (convert to 25x25 coordinates)
            if player_move_start != -1:
                converted_start = self._convert_idx(player_move_start)
                mx = converted_start // self.map_width
                my = converted_start % self.map_width
                move_start.append([mx, my])

                converted_end = self._convert_idx(player_move_end)
                mx = converted_end // self.map_width
                my = converted_end % self.map_width
                move_end.append([mx, my])
            else:
                move_start.append([-1, -1])
                move_end.append([-1, -1])
            
            move_is_50.append(player_move_is_50)

            winner = self.check_winner()
            if winner is not None:
                print(f"Winner: {self.players[winner].name} at turn {turn}")
                break

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

