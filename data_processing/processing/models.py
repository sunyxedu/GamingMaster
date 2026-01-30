"""
Data models for replay simulation.
"""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MapData:
    """
    Map configuration extracted from replay data.

    Can be reused across multiple games for RL training.
    """
    width: int
    height: int
    generals: List[int]  # Flat indices of general positions
    cities: List[int]  # Flat indices of city positions
    city_armies: List[int]  # Initial army count for each city
    mountains: List[int]  # Flat indices of mountain positions

    @classmethod
    def from_replay(cls, replay_data: Dict[str, Any], target_size: int = 25) -> 'MapData':
        """
        Extract MapData from deserialized replay JSON and pad to target_size x target_size.

        Args:
            replay_data: Deserialized replay JSON
            target_size: Target map size (default 25x25)

        Returns:
            MapData with coordinates converted to target_size grid
        """
        orig_width = replay_data["mapWidth"]
        orig_height = replay_data["mapHeight"]

        def convert_idx(flat_idx: int) -> int:
            """Convert flat index from original coords to target_size coords."""
            row = flat_idx // orig_width
            col = flat_idx % orig_width
            return row * target_size + col

        # Convert all flat indices to new coordinate system
        generals = [convert_idx(g) for g in replay_data["generals"]]
        cities = [convert_idx(c) for c in replay_data["cities"]]
        mountains = [convert_idx(m) for m in replay_data["mountains"]]

        return cls(
            width=target_size,
            height=target_size,
            generals=generals,
            cities=cities,
            city_armies=replay_data["cityArmies"],
            mountains=mountains,
        )

    @classmethod
    def create_random(
        cls,
        width: int = 25,
        height: int = 25,
        seed: Optional[int] = None,
        mountain_ratio: float = 0.12,
        city_ratio: float = 0.06,
    ) -> 'MapData':
        """
        Create a random map for testing/training.

        Args:
            width: Map width
            height: Map height
            seed: Random seed for reproducibility
            mountain_ratio: Fraction of tiles that are mountains
            city_ratio: Fraction of tiles that are cities

        Returns:
            MapData with randomly placed features
        """
        if seed is not None:
            random.seed(seed)

        total_tiles = width * height
        all_positions = list(range(total_tiles))

        # Place mountains
        num_mountains = int(total_tiles * mountain_ratio)
        mountains = random.sample(all_positions, num_mountains)

        # Place cities (not on mountains)
        available = [i for i in all_positions if i not in mountains]
        num_cities = int(total_tiles * city_ratio)
        cities = random.sample(available, min(num_cities, len(available)))
        city_armies = [random.randint(40, 50) for _ in cities]

        # Place generals (not on mountains or cities, far apart)
        available = [i for i in all_positions if i not in mountains and i not in cities]
        top_half = [i for i in available if (i // width) < height // 2]
        bottom_half = [i for i in available if (i // width) >= height // 2]

        g1 = random.choice(top_half) if top_half else available[0]
        g2 = random.choice(bottom_half) if bottom_half else available[-1]
        generals = [g1, g2]

        return cls(
            width=width,
            height=height,
            generals=generals,
            cities=cities,
            city_armies=city_armies,
            mountains=mountains,
        )


@dataclass
class Tile:
    """Represents a single map tile."""
    terrain: int  # 0: blank, 1: city, 2: general, 3: mountain
    army: int
    occupied: int  # -1: unoccupied, 0-N: player index


@dataclass
class Player:
    """Represents a player state."""
    name: str
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

