#!/usr/bin/env python3
"""
Visualize self-play games.

Uses direct policy sampling for action selection.

Usage:
    # Watch a live self-play game
    uv run python visualize_game.py --live

    # Save a game and replay it
    uv run python visualize_game.py --save game.json
    uv run python visualize_game.py --replay game.json

    # Generate GIF
    uv run python visualize_game.py --save game.json --gif game.gif
"""

import argparse
import json
import time
import os
import sys

import torch
import numpy as np

# Terminal colors
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Player colors
    PLAYER0 = '\033[94m'  # Blue
    PLAYER1 = '\033[91m'  # Red

    # Terrain colors
    MOUNTAIN = '\033[90m'  # Gray
    CITY = '\033[93m'      # Yellow
    GENERAL = '\033[95m'   # Magenta
    EMPTY = '\033[37m'     # White
    FOG = '\033[90m'       # Dark gray


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize generals.io games")
    parser.add_argument("--live", action="store_true", help="Watch live self-play")
    parser.add_argument("--save", type=str, help="Save game to JSON file")
    parser.add_argument("--replay", type=str, help="Replay game from JSON file")
    parser.add_argument("--gif", type=str, help="Save as GIF (requires --save)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between frames")
    parser.add_argument("--map_width", type=int, default=25, help="Map width")
    parser.add_argument("--map_height", type=int, default=25, help="Map height")
    parser.add_argument("--max_turns", type=int, default=100, help="Max turns")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load model from checkpoint")
    return parser.parse_args()


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def render_game_state(env, turn: int, last_action=None, player_names=["Player 0", "Player 1"]):
    """Render game state to terminal."""
    state = env.state
    lines = []

    # Header
    lines.append(f"\n{'='*50}")
    lines.append(f"  Turn {turn}  |  {Colors.PLAYER0}{player_names[0]}{Colors.RESET} vs {Colors.PLAYER1}{player_names[1]}{Colors.RESET}")
    lines.append(f"{'='*50}")

    # Stats
    p0 = state.players[0]
    p1 = state.players[1]
    lines.append(f"  {Colors.PLAYER0}[P0] Army: {p0.army:4d}  Land: {len(p0.land):3d}{Colors.RESET}")
    lines.append(f"  {Colors.PLAYER1}[P1] Army: {p1.army:4d}  Land: {len(p1.land):3d}{Colors.RESET}")
    lines.append("")

    # Last action
    if last_action:
        player, action = last_action
        color = Colors.PLAYER0 if player == 0 else Colors.PLAYER1
        if action == "PASS":
            lines.append(f"  {color}Player {player} passed{Colors.RESET}")
        else:
            src, dir_id, ratio = action
            dir_names = ['↓', '→', '↑', '←']
            ratio_str = "ALL" if ratio == 0 else "50%"
            src_row, src_col = src // env.map_width, src % env.map_width
            lines.append(f"  {color}Player {player} moved from ({src_row},{src_col}) {dir_names[dir_id]} [{ratio_str}]{Colors.RESET}")
    else:
        lines.append("  Waiting for first move...")
    lines.append("")

    # Column headers
    col_header = "    " + " ".join(f"{i:2d}" for i in range(env.map_width))
    lines.append(col_header)
    lines.append("    " + "---" * env.map_width)

    # Grid
    for row in range(env.map_height):
        line = f"{row:2d} |"
        for col in range(env.map_width):
            tile = state.map_tiles[row][col]
            cell = format_cell(tile, state.generals, row, col, env.map_width)
            line += cell
        lines.append(line)

    # Legend
    lines.append("")
    lines.append(f"  Legend: {Colors.MOUNTAIN}## Mountain{Colors.RESET}  "
                f"{Colors.CITY}CC City{Colors.RESET}  "
                f"{Colors.GENERAL}GG General{Colors.RESET}  "
                f"{Colors.FOG}?? Fog{Colors.RESET}")
    lines.append(f"          {Colors.PLAYER0}P0 Player 0{Colors.RESET}  "
                f"{Colors.PLAYER1}P1 Player 1{Colors.RESET}")

    return "\n".join(lines)


def format_cell(tile, generals, row, col, map_width):
    """Format a single cell for display."""
    flat_idx = row * map_width + col

    # Check terrain
    if tile.terrain == 3:  # Mountain
        return f"{Colors.MOUNTAIN} ##{Colors.RESET}"

    # Check ownership
    if tile.occupied == -1:
        # Unowned
        if tile.terrain == 1:  # City
            if tile.army > 0:
                return f"{Colors.CITY}{tile.army:3d}{Colors.RESET}"
            return f"{Colors.CITY} CC{Colors.RESET}"
        else:
            return f"{Colors.EMPTY}  .{Colors.RESET}"
    else:
        # Owned by a player
        color = Colors.PLAYER0 if tile.occupied == 0 else Colors.PLAYER1

        # Check if general
        if flat_idx in generals:
            if tile.army > 99:
                return f"{color}{tile.army:3d}{Colors.RESET}"
            return f"{color}G{tile.army:02d}{Colors.RESET}"

        # Regular owned tile
        if tile.army > 99:
            return f"{color}{tile.army:3d}{Colors.RESET}"
        return f"{color}{tile.army:3d}{Colors.RESET}"


def sample_action(model, grid, globals_vec, legal_mask, history_latent, tick_bucket, device):
    """
    Sample action using direct policy sampling.

    Returns:
        action: (source_idx, dir_id, ratio_id) tuple or None if no legal moves
        history_latent: updated history latent
    """
    # Forward pass
    tokens = model.input_embedding(
        grid, globals_vec, history_latent, tick_bucket
    )
    y, z, _, _ = model.backbone(tokens)

    # Get policy output with sampling
    policy_out = model.policy_head(
        y, legal_mask, grid, sample=True
    )

    # Get value (not used for action selection, but useful for debugging)
    value = model.value_head(y + z, globals_vec)

    # Extract action from nested 'action' dict
    action_dict = policy_out['action']
    source_idx = action_dict['source_idx'].item()
    dir_id = action_dict['dir_id'].item()
    ratio_id = action_dict['ratio_id'].item()

    # Update history latent
    new_history_latent = model.update_history_latent(
        history_latent, grid, globals_vec
    )

    return (source_idx, dir_id, ratio_id), new_history_latent


def play_and_record(args):
    """Play a game with direct policy sampling and optionally record it."""
    from model.config import ModelConfig
    from model.model import GeneralsModel
    from data_processing.processing.game_env import GameEnvironment

    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig()
    model = GeneralsModel(model_config).to(device)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)

    model.eval()

    # Create environment
    from data_processing.processing.models import MapData
    map_data = MapData.create_random(width=args.map_width, height=args.map_height)
    env = GameEnvironment()
    env.reset(map_data)

    # Record game
    game_record = {
        "map_width": args.map_width,
        "map_height": args.map_height,
        "frames": []
    }

    # History latents for both players
    history_latents = [
        model.get_initial_latent(1, device),
        model.get_initial_latent(1, device)
    ]

    done = False
    turn = 0
    last_action = None
    info = {}

    print("\nStarting game (direct policy sampling)...")
    time.sleep(1)

    while not done and turn < args.max_turns:
        current_player = env.state.current_player

        # Record frame
        frame = {
            "turn": turn,
            "current_player": current_player,
            "tiles": [],
            "players": [],
            "last_action": last_action
        }

        # Save tile states
        for row in range(env.map_height):
            for col in range(env.map_width):
                tile = env.state.map_tiles[row][col]
                frame["tiles"].append({
                    "row": row, "col": col,
                    "terrain": tile.terrain,
                    "army": tile.army,
                    "occupied": tile.occupied
                })

        # Save player stats
        for i, player in enumerate(env.state.players):
            frame["players"].append({
                "army": player.army,
                "land": len(player.land)
            })

        game_record["frames"].append(frame)

        # Render
        if args.live or not args.save:
            clear_screen()
            print(render_game_state(env, turn, last_action))
            time.sleep(args.delay)
        else:
            # Progress indicator
            if turn % 10 == 0:
                print(f"  Turn {turn}...")

        # Get state tensors
        grid = env.state.get_grid_tensor().unsqueeze(0).to(device)
        globals_vec = env.state.get_globals_tensor().unsqueeze(0).to(device)
        legal_mask = env.state.get_legal_mask_tensor().unsqueeze(0).to(device)
        # FIX: Use same formula as dataset: ((t % 50) + 1) // 2
        tick_bucket = torch.tensor([((env.state.turn % 50) + 1) // 2], device=device)

        # Check if there are any legal moves
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            # No legal moves, pass turn
            last_action = (current_player, "PASS")
            env.state.current_player = 1 - current_player
            env.state.turn += 1
            turn += 1
            continue

        # Sample action using direct policy
        with torch.no_grad():
            action, history_latents[current_player] = sample_action(
                model, grid, globals_vec, legal_mask,
                history_latents[current_player], tick_bucket, device
            )

        # Validate action is legal
        if action not in legal_actions:
            # Fallback to random legal action
            import random
            action = random.choice(legal_actions)

        # Execute move
        last_action = (current_player, action)
        _, _, done, info = env.step(action)

        turn += 1

    # Final frame
    winner = info.get('winner', None)
    game_record["winner"] = winner
    game_record["total_turns"] = turn

    # Final render
    if args.live or not args.save:
        clear_screen()
        print(render_game_state(env, turn, last_action))
        print(f"\n{'='*50}")
        if winner is not None:
            color = Colors.PLAYER0 if winner == 0 else Colors.PLAYER1
            print(f"  {color}Player {winner} WINS!{Colors.RESET}")
        else:
            print(f"  Draw (max turns reached)")
        print(f"{'='*50}\n")

    # Save game
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(game_record, f, indent=2)
        print(f"\nGame saved to {args.save}")

    # Generate GIF
    if args.gif:
        generate_gif(game_record, args.gif)

    return game_record


def replay_game(args):
    """Replay a recorded game."""
    with open(args.replay, 'r') as f:
        game_record = json.load(f)

    print(f"\nReplaying game from {args.replay}")
    print(f"Map: {game_record['map_width']}x{game_record['map_height']}")
    print(f"Turns: {game_record['total_turns']}")
    time.sleep(1)

    for frame in game_record["frames"]:
        clear_screen()
        print(render_frame(frame, game_record))
        time.sleep(args.delay)

    # Final result
    winner = game_record.get("winner")
    print(f"\n{'='*50}")
    if winner is not None:
        color = Colors.PLAYER0 if winner == 0 else Colors.PLAYER1
        print(f"  {color}Player {winner} WINS!{Colors.RESET}")
    else:
        print(f"  Draw")
    print(f"{'='*50}\n")


def render_frame(frame, game_record):
    """Render a recorded frame."""
    lines = []
    map_width = game_record["map_width"]
    map_height = game_record["map_height"]

    # Header
    lines.append(f"\n{'='*50}")
    lines.append(f"  Turn {frame['turn']}  |  Current: Player {frame['current_player']}")
    lines.append(f"{'='*50}")

    # Stats
    for i, p in enumerate(frame["players"]):
        color = Colors.PLAYER0 if i == 0 else Colors.PLAYER1
        lines.append(f"  {color}[P{i}] Army: {p['army']:4d}  Land: {p['land']:3d}{Colors.RESET}")
    lines.append("")

    # Last action
    last_action = frame.get("last_action")
    if last_action:
        player, action = last_action
        color = Colors.PLAYER0 if player == 0 else Colors.PLAYER1
        if action == "PASS":
            lines.append(f"  {color}Player {player} passed{Colors.RESET}")
        else:
            src, dir_id, ratio = action
            dir_names = ['↓', '→', '↑', '←']
            ratio_str = "ALL" if ratio == 0 else "50%"
            src_row, src_col = src // map_width, src % map_width
            lines.append(f"  {color}Player {player}: ({src_row},{src_col}) {dir_names[dir_id]} [{ratio_str}]{Colors.RESET}")
    lines.append("")

    # Grid
    col_header = "    " + " ".join(f"{i:2d}" for i in range(map_width))
    lines.append(col_header)
    lines.append("    " + "---" * map_width)

    # Build tile lookup
    tiles = {}
    for t in frame["tiles"]:
        tiles[(t["row"], t["col"])] = t

    for row in range(map_height):
        line = f"{row:2d} |"
        for col in range(map_width):
            t = tiles[(row, col)]
            cell = format_recorded_cell(t)
            line += cell
        lines.append(line)

    return "\n".join(lines)


def format_recorded_cell(tile):
    """Format a recorded tile for display."""
    terrain = tile["terrain"]
    army = tile["army"]
    occupied = tile["occupied"]

    if terrain == 3:  # Mountain
        return f"{Colors.MOUNTAIN} ##{Colors.RESET}"

    if occupied == -1:
        if terrain == 1:  # City
            if army > 0:
                return f"{Colors.CITY}{army:3d}{Colors.RESET}"
            return f"{Colors.CITY} CC{Colors.RESET}"
        else:
            return f"{Colors.EMPTY}  .{Colors.RESET}"
    else:
        color = Colors.PLAYER0 if occupied == 0 else Colors.PLAYER1
        if army > 99:
            return f"{color}{army:3d}{Colors.RESET}"
        return f"{color}{army:3d}{Colors.RESET}"


def generate_gif(game_record, output_path):
    """Generate animated GIF from game record."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Warning: matplotlib not installed. Skipping GIF generation.")
        print("Install with: uv pip install matplotlib")
        return

    print(f"\nGenerating GIF: {output_path}")

    map_width = game_record["map_width"]
    map_height = game_record["map_height"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color map
    cmap = {
        'empty': '#E0E0E0',
        'mountain': '#505050',
        'city': '#FFD700',
        'player0': '#4A90D9',
        'player1': '#D94A4A',
    }

    def draw_frame(frame_idx):
        ax.clear()
        frame = game_record["frames"][frame_idx]

        # Build tile lookup
        tiles = {}
        for t in frame["tiles"]:
            tiles[(t["row"], t["col"])] = t

        # Draw grid
        for row in range(map_height):
            for col in range(map_width):
                t = tiles[(row, col)]

                # Determine color
                if t["terrain"] == 3:
                    color = cmap['mountain']
                elif t["occupied"] == 0:
                    color = cmap['player0']
                elif t["occupied"] == 1:
                    color = cmap['player1']
                elif t["terrain"] == 1:
                    color = cmap['city']
                else:
                    color = cmap['empty']

                rect = mpatches.Rectangle((col, map_height - row - 1), 1, 1,
                                          facecolor=color, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)

                # Draw army count
                is_mountain = (t["terrain"] == 3)
                if t["army"] > 0 and not is_mountain:
                    text_color = 'white' if t["occupied"] != -1 else 'black'
                    ax.text(col + 0.5, map_height - row - 0.5, str(t["army"]),
                           ha='center', va='center', fontsize=10, color=text_color, fontweight='bold')

        # Title and stats
        p0 = frame["players"][0]
        p1 = frame["players"][1]
        ax.set_title(f"Turn {frame['turn']}\n"
                    f"Blue: {p0['army']} army, {p0['land']} land  |  "
                    f"Red: {p1['army']} army, {p1['land']} land",
                    fontsize=12)

        ax.set_xlim(0, map_width)
        ax.set_ylim(0, map_height)
        ax.set_aspect('equal')
        ax.axis('off')

    # Create animation
    anim = FuncAnimation(fig, draw_frame, frames=len(game_record["frames"]),
                        interval=500, repeat=True)

    # Save
    writer = PillowWriter(fps=2)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"GIF saved to {output_path}")


def main():
    args = parse_args()

    if args.replay:
        replay_game(args)
    else:
        play_and_record(args)


if __name__ == "__main__":
    main()
