"""Filter replays based on criteria."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def filter_replays(
    replays: List[Dict[str, Any]],
    min_turns: int = 80,
    game_types: Optional[List[str]] = None,
    max_players: int = 2,
    start_date: Optional[str] = None,
    player_name: Optional[str] = None,
    min_opponent_stars: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter replays based on various criteria.
    
    Args:
        replays: List of replay metadata dictionaries
        min_turns: Minimum number of turns required
        game_types: List of allowed game types (e.g., ['1v1', 'custom'])
        max_players: Maximum number of players
        start_date: Optional start date filter (ISO format)
        player_name: Target player name (for opponent star filtering)
        min_opponent_stars: Minimum stars required for opponents
        
    Returns:
        Filtered list of replays
    """
    if game_types is None:
        game_types = ['1v1', 'custom']

    filtered = []

    for entry in replays:
        # Check game type. 1v1-queue games are always two-player; custom
        # lobbies can hold more, so enforce max_players on them.
        gtype = entry.get("type")
        n_players = len(entry.get("ranking", []))
        is_valid_type = gtype in game_types and (
            gtype == '1v1' or n_players <= max_players
        )
        
        # Check minimum turns
        has_min_turns = entry.get("turns", 0) >= min_turns
        
        # Check start date if specified
        if start_date:
            try:
                started = entry.get("started", 0)
                if isinstance(started, (int, float)):
                    # Unix timestamp in milliseconds
                    entry_date = datetime.utcfromtimestamp(started / 1000)
                else:
                    entry_date = datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ")
                filter_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
                passes_date = entry_date >= filter_date
            except (ValueError, KeyError, OSError):
                passes_date = False
        else:
            passes_date = True
        
        # Check opponent stars if specified
        passes_opponent_stars = True
        if min_opponent_stars is not None and player_name is not None:
            ranking = entry.get("ranking", [])
            # Find opponents (players that are not the target player)
            opponents = [p for p in ranking if p.get("name") != player_name]
            if opponents:
                # Check if any opponent has at least min_opponent_stars
                max_opponent_stars = max((p.get("stars") or 0) for p in opponents)
                passes_opponent_stars = max_opponent_stars >= min_opponent_stars
            else:
                passes_opponent_stars = False
        
        if is_valid_type and has_min_turns and passes_date and passes_opponent_stars:
            filtered.append(entry)
    
    return filtered


def filter_replay_file(
    input_path: str,
    output_path: str,
    min_turns: int = 80,
    game_types: Optional[List[str]] = None,
    player_name: Optional[str] = None,
    min_opponent_stars: Optional[int] = None
) -> int:
    """
    Filter replays from a JSON file and save to another file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        min_turns: Minimum number of turns required
        game_types: List of allowed game types
        player_name: Target player name (for opponent star filtering)
        min_opponent_stars: Minimum stars required for opponents
        
    Returns:
        Number of replays in filtered output
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Input: {len(data)} replays")
    
    filtered_data = filter_replays(
        data, 
        min_turns=min_turns, 
        game_types=game_types,
        player_name=player_name,
        min_opponent_stars=min_opponent_stars
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    
    print(f"Output: {len(filtered_data)} replays")
    
    return len(filtered_data)

