#!/usr/bin/env python3
"""
Visualize simulated replay data from a player's perspective.

Reads simulated JSON files and renders the game state frame-by-frame
using matplotlib, with optional GIF export.

Usage:
    # Interactive mode (Space=pause, 1=back, 2=forward)
    uv run python scripts/visualize_replay.py data/my_replays/simulated/MeltedToast/0v1x6p-8g.json

    # Export GIF
    uv run python scripts/visualize_replay.py data/my_replays/simulated/MeltedToast/0v1x6p-8g.json --gif out.gif

    # Adjust speed (default 0.5s)
    uv run python scripts/visualize_replay.py data/my_replays/simulated/MeltedToast/0v1x6p-8g.json --interval 0.2
"""

import argparse
import json
import sys
import os

import numpy as np

# Color scheme matching test_simulator_visual.py
COLORS = {
    -1: '#1a1a2e',      # Out of bounds / padding
    0: '#2d2d44',       # Unseen Blank
    1: '#4a4a6a',       # Unseen not Blank
    2: '#e8e8e8',       # Seen Blank
    3: '#ffd700',       # Seen Uncaptured City
    4: '#555555',       # Seen Mountain
    5: '#4CAF50',       # Your Land
    6: '#2E7D32',       # Your City
    7: '#1B5E20',       # Your General
    8: '#f44336',       # Opponent's Land
    9: '#c62828',       # Opponent's City
    10: '#b71c1c',      # Opponent's General
}

STATE_NAMES = {
    -1: 'Out of bounds',
    0: 'Unseen Blank',
    1: 'Unseen Obstacle',
    2: 'Seen Blank',
    3: 'Neutral City',
    4: 'Mountain',
    5: 'Your Land',
    6: 'Your City',
    7: 'Your General',
    8: 'Enemy Land',
    9: 'Enemy City',
    10: 'Enemy General',
}

# Direction arrows for action annotation
DIR_ARROWS = {
    (0, -1): '\u2190',   # left
    (0, 1): '\u2192',    # right
    (-1, 0): '\u2191',   # up
    (1, 0): '\u2193',    # down
}


def hex_to_rgb(hex_color):
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return r, g, b


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize simulated replay data from a player's perspective."
    )
    parser.add_argument("replay_path", help="Path to simulated replay JSON file")
    parser.add_argument("--gif", type=str, default=None, help="Export to GIF at this path")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Auto-advance interval in seconds (default: 0.5)")
    parser.add_argument("--start", type=int, default=0, help="Starting turn index")
    parser.add_argument("--end", type=int, default=None, help="Ending turn index (exclusive)")
    return parser.parse_args()


class ReplayVisualizer:
    def __init__(self, data, interval=0.5, start_turn=0):
        self.data = data
        self.total_turns = len(data['map'])
        self.map_height = len(data['map'][0])
        self.map_width = len(data['map'][0][0])
        self.current_turn = min(start_turn, self.total_turns - 1)
        self.interval = interval

        print(f"Replay: {data['id']}")
        print(f"Player: {data['player']}")
        print(f"Map: {self.map_width}x{self.map_height}")
        print(f"Total turns: {self.total_turns}")
        print(f"Controls: Space=pause/resume, 1=back, 2=forward")

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        self.plt = plt
        self.mpatches = mpatches

        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.paused = False
        self.timer = self.fig.canvas.new_timer(interval=int(interval * 1000))
        self.timer.add_callback(self.advance)
        self.timer.start()

        self.render()
        plt.show()

    def on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            if self.paused:
                self.timer.stop()
            else:
                self.timer.start()
        elif event.key == '1':
            if self.current_turn > 0:
                self.current_turn -= 1
                self.render()
        elif event.key == '2':
            if self.current_turn < self.total_turns - 1:
                self.current_turn += 1
                self.render()

    def advance(self):
        if self.paused:
            return
        self.current_turn += 1
        if self.current_turn >= self.total_turns:
            self.current_turn = self.total_turns - 1
            self.timer.stop()
            print("Replay finished!")
        else:
            self.render()

    def render(self):
        self.ax.clear()
        t = self.current_turn

        state = self.data['map'][t]
        army = self.data['armies'][t]

        # Create color grid
        color_grid = np.zeros((self.map_height, self.map_width, 3))
        for i in range(self.map_height):
            for j in range(self.map_width):
                color_grid[i, j] = hex_to_rgb(COLORS.get(state[i][j], '#000000'))

        self.ax.imshow(color_grid, aspect='equal')

        # Army counts
        for i in range(self.map_height):
            for j in range(self.map_width):
                count = army[i][j]
                tile = state[i][j]
                if count > 0:
                    text_color = 'white' if tile in (5, 6, 7, 8, 9, 10) else 'black'
                    fontsize = 8 if count < 100 else 6
                    self.ax.text(j, i, str(count), ha='center', va='center',
                                fontsize=fontsize, color=text_color, fontweight='bold')

        # Action annotation: arrow from start toward end
        start = self.data['move_start'][t]
        end = self.data['move_end'][t]
        if start != [-1, -1] and end != [-1, -1]:
            sy, sx = start
            ey, ex = end
            dy, dx = ey - sy, ex - sx
            arrow = DIR_ARROWS.get((dy, dx), '?')
            is_50 = self.data['move_is_50'][t]
            label = f"{arrow}{'(50%)' if is_50 == 1 else ''}"
            # Draw arrow on the source tile
            self.ax.annotate(
                '', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5),
            )
            # Small label near source
            self.ax.text(sx, sy - 0.4, label, ha='center', va='center',
                        fontsize=7, color='yellow', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))

        # Grid lines
        self.ax.set_xticks(np.arange(-0.5, self.map_width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.map_height, 1), minor=True)
        self.ax.grid(which='minor', color='#333333', linewidth=0.5)
        self.ax.tick_params(which='minor', size=0)
        self.ax.set_xticks(range(self.map_width))
        self.ax.set_yticks(range(self.map_height))

        # Title
        first = self.data['first'][t]
        first_str = "(1st)" if first == 0 else "(2nd)"
        self.ax.set_title(
            f"Turn {t}/{self.total_turns - 1}  {first_str}",
            fontsize=12, fontweight='bold'
        )

        # Stats overlay
        my_army = self.data['info_myArmy'][t]
        my_land = self.data['info_myLand'][t]
        op_army = self.data['info_opArmy'][t]
        op_land = self.data['info_opLand'][t]

        player_name = self.data['player']
        stats_text = (
            f"{'Player':<12} {'Army':>5} {'Land':>5}\n"
            f"{player_name:<12} {my_army:>5} {my_land:>5}\n"
            f"{'Opponent':<12} {op_army:>5} {op_land:>5}"
        )
        self.ax.text(
            0.99, 0.99, stats_text,
            transform=self.ax.transAxes,
            fontsize=10, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'),
        )

        # Legend
        legend_patches = [
            self.mpatches.Patch(color=COLORS[s], label=STATE_NAMES[s])
            for s in [7, 6, 5, 10, 9, 8, 3, 4, 2, 1, 0]
        ]
        self.ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1),
                       fontsize=8, title='Tile Types')

        self.plt.tight_layout()
        self.fig.canvas.draw()


def generate_gif(data, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation, PillowWriter

    total_turns = len(data['map'])
    map_height = len(data['map'][0])
    map_width = len(data['map'][0][0])
    player_name = data['player']

    print(f"Generating GIF: {output_path}  ({total_turns} frames)")

    fig, ax = plt.subplots(figsize=(12, 12))

    def draw_frame(t):
        ax.clear()
        state = data['map'][t]
        army = data['armies'][t]

        color_grid = np.zeros((map_height, map_width, 3))
        for i in range(map_height):
            for j in range(map_width):
                color_grid[i, j] = hex_to_rgb(COLORS.get(state[i][j], '#000000'))

        ax.imshow(color_grid, aspect='equal')

        for i in range(map_height):
            for j in range(map_width):
                count = army[i][j]
                tile = state[i][j]
                if count > 0:
                    text_color = 'white' if tile in (5, 6, 7, 8, 9, 10) else 'black'
                    fontsize = 8 if count < 100 else 6
                    ax.text(j, i, str(count), ha='center', va='center',
                            fontsize=fontsize, color=text_color, fontweight='bold')

        # Action arrow
        start = data['move_start'][t]
        end = data['move_end'][t]
        if start != [-1, -1] and end != [-1, -1]:
            sy, sx = start
            ey, ex = end
            ax.annotate(
                '', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5),
            )

        ax.set_xticks(np.arange(-0.5, map_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, map_height, 1), minor=True)
        ax.grid(which='minor', color='#333333', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        ax.set_xticks(range(map_width))
        ax.set_yticks(range(map_height))

        first = data['first'][t]
        first_str = "(1st)" if first == 0 else "(2nd)"
        ax.set_title(f"Turn {t}/{total_turns - 1}  {first_str}",
                     fontsize=12, fontweight='bold')

        my_army = data['info_myArmy'][t]
        my_land = data['info_myLand'][t]
        op_army = data['info_opArmy'][t]
        op_land = data['info_opLand'][t]

        stats_text = (
            f"{'Player':<12} {'Army':>5} {'Land':>5}\n"
            f"{player_name:<12} {my_army:>5} {my_land:>5}\n"
            f"{'Opponent':<12} {op_army:>5} {op_land:>5}"
        )
        ax.text(
            0.99, 0.99, stats_text,
            transform=ax.transAxes,
            fontsize=10, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'),
        )

        legend_patches = [
            mpatches.Patch(color=COLORS[s], label=STATE_NAMES[s])
            for s in [7, 6, 5, 10, 9, 8, 3, 4, 2, 1, 0]
        ]
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=8, title='Tile Types')

        plt.tight_layout()

    anim = FuncAnimation(fig, draw_frame, frames=total_turns, interval=500, repeat=True)
    writer = PillowWriter(fps=2)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"GIF saved to {output_path}")


def main():
    args = parse_args()

    with open(args.replay_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Slice turn range if specified
    total = len(data['map'])
    start = max(0, args.start)
    end = min(args.end, total) if args.end is not None else total

    if args.gif and (start > 0 or end < total):
        seq_keys = ['map', 'armies', 'first', 'info_myArmy', 'info_myLand',
                     'info_opArmy', 'info_opLand', 'move_start', 'move_end', 'move_is_50']
        for key in seq_keys:
            data[key] = data[key][start:end]
        print(f"Exporting turns {start}-{end - 1} ({end - start} frames)")

    if args.gif:
        generate_gif(data, args.gif)
    else:
        ReplayVisualizer(data, interval=args.interval, start_turn=args.start)


if __name__ == "__main__":
    main()
