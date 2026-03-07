"""
Parser to convert simulator output to gio training format.

Converts the replay simulation data into the sequence format expected by gio:
- grid_seq: [T, C, H, W] - spatial state
- globals_seq: [T, G] - global features
- legal_source_mask_seq: [T, 1, H, W] - legal source positions
- Action targets: count, src_y, src_x, dir_id, ratio_id
- Value and auxiliary targets
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def compute_direction(start_pos: List[int], end_pos: List[int]) -> int:
    """
    Compute direction ID from start to end position.
    
    Direction mapping:
    0: down (row++)
    1: right (col++)
    2: up (row--)
    3: left (col--)
    
    Args:
        start_pos: [row, col]
        end_pos: [row, col]
        
    Returns:
        Direction ID (0-3)
    """
    if start_pos[0] == -1 or end_pos[0] == -1:
        return 0  # No move
    
    dr = end_pos[0] - start_pos[0]
    dc = end_pos[1] - start_pos[1]
    
    # Map to direction
    if dr > 0:
        return 0  # down
    elif dc > 0:
        return 1  # right
    elif dr < 0:
        return 2  # up
    else:  # dc < 0
        return 3  # left


def encode_state_as_channels(
    state: List[List[int]], max_h: int = 25, max_w: int = 25,
) -> np.ndarray:
    """
    Encode the 11-category state as one-hot channels.
    
    Args:
        state: State matrix with values 0-10
        max_h, max_w: Maximum dimensions
        
    Returns:
        Array of shape [11, max_h, max_w]
    """
    channels = np.zeros((11, max_h, max_w), dtype=np.float32)
    
    for i in range(len(state)):
        for j in range(len(state[0])):
            val = state[i][j]
            if 0 <= val <= 10:
                channels[val, i, j] = 1.0
    
    return channels


def create_legal_source_mask(state: List[List[int]], armies: List[List[int]]) -> np.ndarray:
    """
    Create legal source mask (can only move from owned tiles with army > 1).
    
    Args:
        state: State matrix
        armies: Army count matrix
        
    Returns:
        Binary mask of shape [1, H, W]
    """
    mask = np.zeros((1, 25, 25), dtype=bool)
    
    for i in range(len(state)):
        for j in range(len(state[0])):
            # Can move from owned land (state 5, 6, 7) with army > 1
            if state[i][j] in [5, 6, 7] and armies[i][j] > 1:
                mask[0, i, j] = True
    
    return mask


def convert_simulator_to_gio(
    sim_data: Dict[str, Any],
    max_actions_per_tick: int = 4,
    estimate_value: bool = True
) -> Dict[str, Any]:
    """
    Convert simulator output to gio sequence format.

    IMPORTANT: the simulator records state[t] AFTER executing move[t] and
    applying army growth.  For IL training the correct pairing is
    (observation=state[t], target_action=move[t+1]) because the player
    decides move[t+1] *after* seeing state[t].  This means we iterate
    t in [0, T-2] and look up the action target at t+1.

    Args:
        sim_data: Output from ReplaySimulator
        max_actions_per_tick: Max number of actions per turn (K)
        estimate_value: Whether to estimate value from land/army advantage

    Returns:
        Dictionary in gio format
    """
    T_raw = len(sim_data["first"])
    T = T_raw - 1  # last observation has no future action
    H, W = 25, 25
    K = max_actions_per_tick

    grid_seq = []
    globals_seq = []
    legal_source_mask_seq = []

    count = []
    src_y = []
    src_x = []
    dir_id = []
    ratio_id = []
    value = []

    tick_seq: List[int] = []
    tick_bucket_seq: List[int] = []

    for t in range(T):
        # --- observation at time t (state the player sees) ---------------
        state_channels = encode_state_as_channels(sim_data["map"][t])

        army_channel = np.zeros((1, H, W), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                army_val = sim_data["armies"][t][i][j]
                if army_val > 0:
                    army_channel[0, i, j] = np.log1p(army_val) / 10.0

        grid = np.vstack([state_channels, army_channel])
        grid_seq.append(grid.tolist())

        my_army = sim_data["info_myArmy"][t]
        my_land = sim_data["info_myLand"][t]
        op_army = sim_data["info_opArmy"][t]
        op_land = sim_data["info_opLand"][t]

        globals_seq.append([
            np.log1p(my_army) / 10.0,
            np.log1p(my_land) / 10.0,
            np.log1p(op_army) / 10.0,
            np.log1p(op_land) / 10.0,
        ])

        legal_mask = create_legal_source_mask(sim_data["map"][t], sim_data["armies"][t])
        legal_source_mask_seq.append(legal_mask.tolist())

        # --- action target at time t+1 (decided based on state t) -------
        move_start = sim_data["move_start"][t + 1]
        move_end = sim_data["move_end"][t + 1]
        is_50 = sim_data["move_is_50"][t + 1]

        if move_start[0] >= 0:
            count.append(1)
            sy = [move_start[0]] + [0] * (K - 1)
            sx = [move_start[1]] + [0] * (K - 1)
            direction = compute_direction(move_start, move_end)
            dirs = [direction] + [0] * (K - 1)
            ratio = [1 if is_50 else 0] + [0] * (K - 1)
            src_y.append(sy)
            src_x.append(sx)
            dir_id.append(dirs)
            ratio_id.append(ratio)
        else:
            count.append(0)
            src_y.append([0] * K)
            src_x.append([0] * K)
            dir_id.append([0] * K)
            ratio_id.append([0] * K)

        # --- value (blended: heuristic + game outcome) --------------------
        if estimate_value:
            if op_land == 0:
                heuristic = 1.0
            elif my_land == 0:
                heuristic = -1.0
            else:
                my_cities = sim_data.get("info_myCities", [0] * T_raw)[t]
                op_cities = sim_data.get("info_opCities", [0] * T_raw)[t]
                city_diff = my_cities - op_cities
                adj_my = my_army + max(city_diff, 0) * 50
                adj_op = op_army + max(-city_diff, 0) * 50
                army_ratio = adj_my / max(adj_op, 1)
                heuristic = float(np.tanh(army_ratio - 1.0))

            winner = sim_data.get("winner")
            player = sim_data.get("player")
            if winner is not None and player is not None:
                outcome = 1.0 if winner == player else -1.0
                w = (t / max(T - 1, 1)) ** 2
                val = w * outcome + (1.0 - w) * heuristic
            else:
                val = heuristic
            value.append(float(val))
        else:
            value.append(0.0)

        tick_seq.append(t)
        tick_bucket_seq.append(int(((t % 50) + 1) // 2))

    enemy_heatmap = [[[[0.0] * W for _ in range(H)]] for _ in range(T)]
    next_visibility = [[[[0.0] * W for _ in range(H)]] for _ in range(T)]

    return {
        "grid_seq": grid_seq,
        "globals_seq": globals_seq,
        "legal_source_mask_seq": legal_source_mask_seq,
        "max_actions_per_tick": K,
        "count": count,
        "src_y": src_y,
        "src_x": src_x,
        "dir_id": dir_id,
        "ratio_id": ratio_id,
        "value": value,
        "tick_seq": tick_seq,
        "tick_bucket_seq": tick_bucket_seq,
        "enemy_heatmap": enemy_heatmap,
        "next_visibility": next_visibility,
    }


def parse_replay_to_gio(
    sim_output_path: str,
    gio_output_path: str,
    max_actions_per_tick: int = 4
) -> Dict[str, Any]:
    """
    Parse simulator output file and save in gio format.
    
    Args:
        sim_output_path: Path to simulator output JSON
        gio_output_path: Path to save gio format JSON
        max_actions_per_tick: Max actions per turn
        
    Returns:
        Converted data in gio format
    """
    with open(sim_output_path, 'r') as f:
        sim_data = json.load(f)
    
    gio_data = convert_simulator_to_gio(sim_data, max_actions_per_tick)
    
    with open(gio_output_path, 'w') as f:
        json.dump(gio_data, f)
    
    print(f"Converted {sim_output_path} -> {gio_output_path}")
    
    return gio_data


def batch_parse_directory(
    sim_dir: str,
    gio_dir: str,
    max_actions_per_tick: int = 4
) -> int:
    """
    Parse all simulator outputs in a directory.
    
    Args:
        sim_dir: Directory containing simulator outputs
        gio_dir: Directory to save gio format files
        max_actions_per_tick: Max actions per turn
        
    Returns:
        Number of files processed
    """
    sim_path = Path(sim_dir)
    gio_path = Path(gio_dir)
    gio_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for sim_file in sim_path.glob("*.json"):
        gio_file = gio_path / sim_file.name
        try:
            parse_replay_to_gio(str(sim_file), str(gio_file), max_actions_per_tick)
            count += 1
        except Exception as e:
            print(f"Error processing {sim_file}: {e}")
    
    print(f"Processed {count} files")
    return count
