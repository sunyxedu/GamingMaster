#!/usr/bin/env python3
"""
Interactive visual test for the simulator.
Right-click to advance one turn.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from data_processing.processing.simulator import ReplaySimulator

# Color scheme for different tile types
COLORS = {
    -1: '#1a1a2e',      # Out of bounds / padding (dark blue)
    0: '#2d2d44',       # Unseen Blank (dark gray)
    1: '#4a4a6a',       # Unseen not Blank (lighter gray)
    2: '#e8e8e8',       # Seen Blank (light gray)
    3: '#ffd700',       # Seen Uncaptured City (gold)
    4: '#555555',       # Seen Mountain (dark gray)
    5: '#4CAF50',       # Your Land (green)
    6: '#2E7D32',       # Your City (dark green)
    7: '#1B5E20',       # Your General (very dark green)
    8: '#f44336',       # Opponent's Land (red)
    9: '#c62828',       # Opponent's City (dark red)
    10: '#b71c1c',      # Opponent's General (very dark red)
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


class InteractiveVisualizer:
    def __init__(self, replay_path: str, player_name: str, interval: float = 0.5):
        with open(replay_path, 'r', encoding='utf-8') as f:
            self.replay_data = json.load(f)

        self.player_name = player_name
        self.map_height = self.replay_data['mapHeight']
        self.map_width = self.replay_data['mapWidth']
        self.interval = interval

        print(f"Replay: {self.replay_data['id']}")
        print(f"Map: {self.map_width}x{self.map_height}")
        print(f"Players: {self.replay_data['usernames']}")
        print(f"Simulating from {player_name}'s perspective...")

        simulator = ReplaySimulator(self.replay_data, player_name)
        self.result = simulator.simulate()
        self.total_turns = len(self.result['map'])
        self.current_turn = 0

        print(f"Total turns: {self.total_turns}")
        print(f"Auto-advancing every {interval}s.")
        print("Controls: Space=pause/resume, 1=back, 2=forward, Close window to exit.")

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Setup animation timer
        self.paused = False
        self.timer = self.fig.canvas.new_timer(interval=int(interval * 1000))
        self.timer.add_callback(self.advance)
        self.timer.start()

        self.render()
        plt.show()

    def on_key(self, event):
        if event.key == ' ':  # Space - pause/resume
            self.paused = not self.paused
            if self.paused:
                self.timer.stop()
                print("Paused")
            else:
                self.timer.start()
                print("Resumed")
        elif event.key == '1':  # 1 - back one turn
            if self.current_turn > 0:
                self.current_turn -= 1
                self.render()
        elif event.key == '2':  # 2 - forward one turn
            if self.current_turn < self.total_turns - 1:
                self.current_turn += 1
                self.render()

    def advance(self):
        if self.paused:
            return
        self.current_turn += 1
        if self.current_turn >= self.total_turns:
            self.timer.stop()
            print("Replay finished!")
        else:
            self.render()

    def render(self):
        self.ax.clear()

        state = self.result['map'][self.current_turn]
        army = self.result['armies'][self.current_turn]

        # Trim to actual map size
        state = [row[:self.map_width] for row in state[:self.map_height]]
        army = [row[:self.map_width] for row in army[:self.map_height]]

        # Create color map
        color_grid = np.zeros((self.map_height, self.map_width, 3))
        for i in range(self.map_height):
            for j in range(self.map_width):
                hex_color = COLORS.get(state[i][j], '#000000')
                r = int(hex_color[1:3], 16) / 255
                g = int(hex_color[3:5], 16) / 255
                b = int(hex_color[5:7], 16) / 255
                color_grid[i, j] = [r, g, b]

        self.ax.imshow(color_grid, aspect='equal')

        # Add army counts
        for i in range(self.map_height):
            for j in range(self.map_width):
                army_count = army[i][j]
                tile_state = state[i][j]

                if army_count > 0:
                    if tile_state in [5, 6, 7, 8, 9, 10]:
                        text_color = 'white'
                    else:
                        text_color = 'black'

                    fontsize = 8 if army_count < 100 else 6
                    self.ax.text(j, i, str(army_count), ha='center', va='center',
                                fontsize=fontsize, color=text_color, fontweight='bold')

        # Grid
        self.ax.set_xticks(np.arange(-0.5, self.map_width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.map_height, 1), minor=True)
        self.ax.grid(which='minor', color='#333333', linewidth=0.5)
        self.ax.tick_params(which='minor', size=0)

        # Labels
        self.ax.set_xticks(range(self.map_width))
        self.ax.set_yticks(range(self.map_height))

        my_army = self.result['info_myArmy'][self.current_turn]
        my_land = self.result['info_myLand'][self.current_turn]
        op_army = self.result['info_opArmy'][self.current_turn]
        op_land = self.result['info_opLand'][self.current_turn]

        self.ax.set_title(
            f'Turn {self.current_turn}/{self.total_turns-1}',
            fontsize=12, fontweight='bold'
        )

        # Player stats overlay in top-right corner (like generals.io UI)
        opponent_name = [n for n in self.replay_data['usernames'] if n != self.player_name][0]
        stats_text = (
            f"{'Player':<12} {'Army':>5} {'Land':>5}\n"
            f"{self.player_name:<12} {my_army:>5} {my_land:>5}\n"
            f"{opponent_name:<12} {op_army:>5} {op_land:>5}"
        )
        self.ax.text(
            0.99, 0.99, stats_text,
            transform=self.ax.transAxes,
            fontsize=10,
            fontfamily='monospace',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'),
        )

        # Legend
        legend_patches = [
            mpatches.Patch(color=COLORS[s], label=STATE_NAMES[s])
            for s in [7, 6, 5, 10, 9, 8, 3, 4, 2, 1, 0]
        ]
        self.ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1),
                      fontsize=8, title='Tile Types')

        plt.tight_layout()
        self.fig.canvas.draw()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: python tests/test_simulator_visual.py <replay.json>")
    replay_path = sys.argv[1]

    with open(replay_path, 'r') as f:
        data = json.load(f)

    player_name = data['usernames'][0]
    InteractiveVisualizer(replay_path, player_name)