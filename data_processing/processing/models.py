"""
Data models for replay simulation.
"""

from dataclasses import dataclass, field
from typing import List


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

