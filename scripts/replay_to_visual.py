#!/usr/bin/env python3
"""
End-to-end pipeline: download a generals.io replay → simulate → visualize.

Usage examples:
    # From a replay URL
    python3 scripts/replay_to_visual.py \
        --url "https://generals.io/replays/asS2S5O_y?p=%7EHemlix%7E"

    # From a replay ID (auto-picks first player)
    python3 scripts/replay_to_visual.py --id asS2S5O_y

    # Specify player & output directory, god-view (no fog)
    python3 scripts/replay_to_visual.py --id asS2S5O_y \
        --player "~Hemlix~" --out /tmp/viz --god

    # Render only a slice of turns
    python3 scripts/replay_to_visual.py --id asS2S5O_y \
        --start 100 --end 200 --step 5

    # Use a local replay JSON file
    python3 scripts/replay_to_visual.py --file /path/to/replay.json

The script produces:
    <out>/frames/turn_XXXX.png   – one PNG per rendered turn
    <out>/summary.png            – 3×3 grid of key moments
"""

import argparse
import asyncio
import json
import os
import ssl
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp
import matplotlib

matplotlib.use("Agg")  # headless – must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from data_processing.download.replay_fetcher import ReplayFetcher
from data_processing.processing.simulator import ReplaySimulator

# ── colour scheme (matching the original visualizer) ────────────────────

COLORS = {
    -1: "#1a1a2e",  # Out of bounds / padding
    0:  "#2d2d44",  # Unseen Blank
    1:  "#4a4a6a",  # Unseen Obstacle (mountain / city behind fog)
    2:  "#e8e8e8",  # Seen Blank
    3:  "#ffd700",  # Seen Neutral City
    4:  "#555555",  # Seen Mountain
    5:  "#4CAF50",  # Your Land
    6:  "#2E7D32",  # Your City
    7:  "#1B5E20",  # Your General
    8:  "#f44336",  # Opponent's Land
    9:  "#c62828",  # Opponent's City
    10: "#b71c1c",  # Opponent's General
}

STATE_NAMES = {
    -1: "Padding",
    0:  "Unseen Blank",
    1:  "Unseen Obstacle",
    2:  "Seen Blank",
    3:  "Neutral City",
    4:  "Mountain",
    5:  "Your Land",
    6:  "Your City",
    7:  "Your General",
    8:  "Enemy Land",
    9:  "Enemy City",
    10: "Enemy General",
}

# God-mode uses a simpler, ownership-centric palette
GOD_COLORS = {
    "blank":    "#e8e8e8",
    "mountain": "#555555",
    "city_n":   "#ffd700",
    "p0":       "#4CAF50",
    "p0_city":  "#2E7D32",
    "p0_gen":   "#1B5E20",
    "p1":       "#f44336",
    "p1_city":  "#c62828",
    "p1_gen":   "#b71c1c",
}

DIR_ARROWS = {0: "↓", 1: "→", 2: "↑", 3: "←"}
DIR_DELTA  = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    return int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255


# ── download ────────────────────────────────────────────────────────────

async def download_replay(replay_id: str, cache_dir: str) -> Dict[str, Any]:
    """Download & cache a single replay, return the deserialised dict."""
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, f"{replay_id}.json")
    if os.path.exists(cached):
        with open(cached) as f:
            return json.load(f)

    fetcher = ReplayFetcher(server="na")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ctx)

    async with aiohttp.ClientSession(connector=connector) as session:
        data = await fetcher.fetch_replay(session, replay_id)
    if data is None:
        raise RuntimeError(f"Failed to download replay {replay_id}")

    with open(cached, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ── rendering helpers ───────────────────────────────────────────────────

def render_fog_frame(
    sim_result: Dict[str, Any],
    replay_data: Dict[str, Any],
    turn: int,
    player_name: str,
    ax: plt.Axes,
) -> None:
    """Render one turn from the player's fog-of-war perspective."""
    H = replay_data["mapHeight"]
    W = replay_data["mapWidth"]
    state = sim_result["map"][turn]
    army = sim_result["armies"][turn]
    state = [row[:W] for row in state[:H]]
    army = [row[:W] for row in army[:H]]

    color_grid = np.zeros((H, W, 3))
    for r in range(H):
        for c in range(W):
            color_grid[r, c] = _hex_to_rgb(COLORS.get(state[r][c], "#000000"))
    ax.imshow(color_grid, aspect="equal")

    for r in range(H):
        for c in range(W):
            a = army[r][c]
            if a > 0:
                tc = "white" if state[r][c] in (5, 6, 7, 8, 9, 10) else "black"
                fs = 7 if a < 100 else 5 if a < 1000 else 4
                ax.text(c, r, str(a), ha="center", va="center",
                        fontsize=fs, color=tc, fontweight="bold")

    # draw move arrow
    ms = sim_result["move_start"][turn]
    me = sim_result["move_end"][turn]
    if ms[0] >= 0:
        ax.annotate(
            "", xy=(me[1], me[0]), xytext=(ms[1], ms[0]),
            arrowprops=dict(arrowstyle="->", color="yellow", lw=2),
        )

    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#333333", linewidth=0.3)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.tick_params(labelsize=5)

    # stats overlay
    my_a = sim_result["info_myArmy"][turn]
    my_l = sim_result["info_myLand"][turn]
    op_a = sim_result["info_opArmy"][turn]
    op_l = sim_result["info_opLand"][turn]
    others = [n for n in replay_data["usernames"] if n != player_name]
    opp = others[0] if others else "?"
    stats = (
        f"{'Player':<12} {'Army':>5} {'Land':>5}\n"
        f"{player_name:<12} {my_a:>5} {my_l:>5}\n"
        f"{opp:<12} {op_a:>5} {op_l:>5}"
    )
    ax.text(
        0.99, 0.99, stats, transform=ax.transAxes, fontsize=6,
        fontfamily="monospace", va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="gray"),
    )
    total_turns = len(sim_result["map"])
    ax.set_title(f"Turn {turn}/{total_turns - 1}  ({player_name})", fontsize=8)


def render_god_frame(
    sim: ReplaySimulator,
    replay_data: Dict[str, Any],
    turn: int,
    ax: plt.Axes,
) -> None:
    """Render one turn from a 'god' perspective (no fog, full info)."""
    H = sim.map_height
    W = sim.map_width

    color_grid = np.zeros((H, W, 3))
    for r in range(H):
        for c in range(W):
            tile = sim.map[r][c]
            if tile.terrain == 3:
                color_grid[r, c] = _hex_to_rgb(GOD_COLORS["mountain"])
            elif tile.occupied == -1:
                if tile.terrain == 1:
                    color_grid[r, c] = _hex_to_rgb(GOD_COLORS["city_n"])
                else:
                    color_grid[r, c] = _hex_to_rgb(GOD_COLORS["blank"])
            else:
                pid = tile.occupied
                prefix = f"p{pid}"
                if tile.terrain == 2:
                    key = f"{prefix}_gen"
                    color_grid[r, c] = _hex_to_rgb(
                        GOD_COLORS.get(key, GOD_COLORS[prefix]),
                    )
                elif tile.terrain == 1:
                    key = f"{prefix}_city"
                    color_grid[r, c] = _hex_to_rgb(
                        GOD_COLORS.get(key, GOD_COLORS[prefix]),
                    )
                else:
                    color_grid[r, c] = _hex_to_rgb(GOD_COLORS.get(prefix, "#888888"))

    ax.imshow(color_grid, aspect="equal")

    for r in range(H):
        for c in range(W):
            tile = sim.map[r][c]
            if tile.army > 0 and tile.terrain != 3:
                tc = "white" if tile.occupied != -1 else "black"
                fs = 7 if tile.army < 100 else 5 if tile.army < 1000 else 4
                ax.text(c, r, str(tile.army), ha="center", va="center",
                        fontsize=fs, color=tc, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#333333", linewidth=0.3)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.tick_params(labelsize=5)

    # stats
    names = sim.usernames
    lines = [f"{'Player':<12} {'Army':>6} {'Land':>5}"]
    for i, p in enumerate(sim.players):
        lines.append(f"{names[i]:<12} {p.army:>6} {len(p.land):>5}")
    ax.text(
        0.99, 0.99, "\n".join(lines), transform=ax.transAxes, fontsize=6,
        fontfamily="monospace", va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="gray"),
    )
    ax.set_title(f"Turn {turn}  (God view)", fontsize=8)


# ── god-mode simulator (step-by-step, recording frames) ─────────────────

def simulate_god(replay_data: Dict[str, Any]):
    """
    Run the simulator step-by-step in god-mode and yield
    (turn, simulator_ref) after each tick so we can render.
    """
    sim = ReplaySimulator(replay_data, replay_data["usernames"][0])
    sim.init_map()

    max_turn = sim.moves[-1]["turn"] if sim.moves else 0
    move_idx = 0

    for turn in range(0, max_turn + 2):
        turn_moves = [None, None]
        cnt = 0
        while move_idx < len(sim.moves) and sim.moves[move_idx]["turn"] + 1 == turn:
            move = sim.moves[move_idx]
            if cnt < 2:
                turn_moves[cnt] = move
            cnt += 1
            move_idx += 1

        sim.process_moves(turn_moves[0], turn_moves[1])
        sim.increment_armies(turn)
        sim.update_vision()

        yield turn, sim, turn_moves

        if sim.check_winner() is not None:
            return


# ── main pipeline ───────────────────────────────────────────────────────

def run_pipeline(args):
    # 1. Resolve replay data
    replay_data = None
    replay_id = None

    if args.file:
        with open(args.file) as f:
            replay_data = json.load(f)
        replay_id = replay_data.get("id", Path(args.file).stem)
    else:
        if args.url:
            parsed = urlparse(args.url)
            parts = parsed.path.strip("/").split("/")
            replay_id = parts[-1] if parts else None
            qs = parse_qs(parsed.query)
            if "p" in qs and not args.player:
                args.player = unquote(qs["p"][0])
        elif args.id:
            replay_id = args.id

        if not replay_id:
            print("Error: supply --url, --id, or --file")
            sys.exit(1)

        cache = os.path.join(args.out, ".cache")
        print(f"Downloading replay {replay_id} …")
        replay_data = asyncio.run(download_replay(replay_id, cache))

    print(f"Replay {replay_data.get('id', '?')}")
    print(f"  Map  : {replay_data['mapWidth']}×{replay_data['mapHeight']}")
    print(f"  Players: {replay_data['usernames']}")
    print(f"  Moves: {len(replay_data['moves'])}")

    # pick player
    player = args.player or replay_data["usernames"][0]
    if player not in replay_data["usernames"]:
        print(f"Warning: '{player}' not in usernames {replay_data['usernames']}, "
              f"using {replay_data['usernames'][0]}")
        player = replay_data["usernames"][0]

    # 2. Simulate
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    god = args.god

    if god:
        print("Running god-mode simulation …")
        all_turns: List[Tuple[int, Any]] = []
        for turn, sim_ref, t_moves in simulate_god(replay_data):
            all_turns.append((turn, sim_ref, t_moves))

        total = len(all_turns)
        start = args.start or 0
        end = min(args.end or total, total)
        step = args.step or 1
        render_indices = list(range(start, end, step))
        print(f"Rendering {len(render_indices)} god-view frames "
              f"(turns {start}..{end - 1}, step {step}) …")

        # Re-simulate for real rendering (because sim_ref is mutated in-place)
        frame_i = 0
        rendered = set(render_indices)
        for turn, sim_ref, t_moves in simulate_god(replay_data):
            if turn in rendered:
                fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
                render_god_frame(sim_ref, replay_data, turn, ax)
                path = os.path.join(frames_dir, f"turn_{turn:04d}.png")
                fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
                plt.close(fig)
                frame_i += 1
                if frame_i % 50 == 0:
                    print(f"  {frame_i}/{len(render_indices)} frames")
        print(f"  Saved {frame_i} frames to {frames_dir}/")

    else:
        print(f"Simulating from {player}'s perspective …")
        sim = ReplaySimulator(replay_data, player)
        result = sim.simulate()
        total = len(result["map"])
        start = args.start or 0
        end = min(args.end or total, total)
        step = args.step or 1
        render_indices = list(range(start, end, step))
        print(f"Rendering {len(render_indices)} fog-view frames "
              f"(turns {start}..{end - 1}, step {step}) …")

        for i, turn in enumerate(render_indices):
            fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
            render_fog_frame(result, replay_data, turn, player, ax)
            path = os.path.join(frames_dir, f"turn_{turn:04d}.png")
            fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(render_indices)} frames")
        print(f"  Saved {len(render_indices)} frames to {frames_dir}/")

    # 3. Summary montage (pick ~9 evenly-spaced key turns)
    print("Generating summary montage …")
    if god:
        n_summary = min(9, len(render_indices))
        indices = [render_indices[i * len(render_indices) // n_summary]
                   for i in range(n_summary)]
        fig, axes = plt.subplots(3, 3, figsize=(24, 24), dpi=100)
        axes = axes.flatten()
        frame_idx = 0
        for turn, sim_ref, _ in simulate_god(replay_data):
            if turn in indices:
                render_god_frame(sim_ref, replay_data, turn, axes[frame_idx])
                frame_idx += 1
                if frame_idx >= n_summary:
                    break
        for j in range(frame_idx, 9):
            axes[j].axis("off")
    else:
        total = len(result["map"])
        n_summary = min(9, total)
        if n_summary > 1:
            indices = [i * (total - 1) // (n_summary - 1) for i in range(n_summary)]
        else:
            indices = [0]
        fig, axes = plt.subplots(3, 3, figsize=(24, 24), dpi=100)
        axes = axes.flatten()
        for i, turn in enumerate(indices):
            render_fog_frame(result, replay_data, turn, player, axes[i])
        for j in range(len(indices), 9):
            axes[j].axis("off")

    fig.suptitle(
        f"Replay {replay_id}  |  "
        f"{' vs '.join(replay_data['usernames'])}  |  "
        f"Map {replay_data['mapWidth']}×{replay_data['mapHeight']}",
        fontsize=14, fontweight="bold",
    )
    summary_path = os.path.join(args.out, "summary.png")
    fig.savefig(summary_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Summary saved to {summary_path}")
    print("Done!")


def main():
    p = argparse.ArgumentParser(
        description="Download → Simulate → Visualise a generals.io replay",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--url", help="Full replay URL from generals.io")
    src.add_argument("--id", help="Replay ID (e.g. asS2S5O_y)")
    src.add_argument("--file", help="Path to a local replay JSON")

    p.add_argument("--player", help="Player name (default: first in replay)")
    p.add_argument("--out", default="viz_output", help="Output directory")
    p.add_argument("--god", action="store_true",
                   help="Render god-view (no fog, full map)")
    p.add_argument("--start", type=int, default=None, help="First turn to render")
    p.add_argument("--end", type=int, default=None, help="Last turn (exclusive)")
    p.add_argument("--step", type=int, default=None, help="Render every N-th turn")
    args = p.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
