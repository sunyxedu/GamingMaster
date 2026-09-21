"""Hidden-rule grid puzzles — the tier-1 test bed (ARC-AGI-3 in spirit).

The agent sees a coloured grid and 6 abstract actions (like ARC-AGI-3's
ACTION1..6) but is *not told the rules*: which tiles are keys, which are
doors, whether a tile teleports, what "interact" does. Rules are drawn per
seed. Score = levels completed; each level has an action budget so wasting
actions matters (cf. ARC's human-action-efficiency metric).

Rule families (sampled per seed):
  * key_door:   step on the KEY colour, then the DOOR colour opens the exit
  * toggle:     'interact' flips walls of colour T within radius 1
  * teleport:   stepping on colour P jumps to its twin
  * mirror:     left/right (or up/down) actions are swapped this seed
"""
from __future__ import annotations

import random
from typing import Any

from ..games.base import ActionSpec

ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
COLOURS = ["black", "blue", "red", "green", "yellow", "grey", "magenta", "orange", "cyan", "white"]
EMPTY, WALL, AGENT, EXIT = 0, 5, 9, 4       # black, grey, white, yellow


class RulePuzzleGame:
    name = "rulepuzzle"
    action_spec = ActionSpec(
        names=ACTIONS,
        descriptions={a: "unknown — discover by trying" for a in ACTIONS},
        notes="ACTION5/ACTION6 may be context-dependent. Effects are fixed for the whole episode.",
    )

    def __init__(self, size: int = 10, n_levels: int = 3, budget_per_level: int = 60) -> None:
        self.size, self.n_levels, self.budget_per_level = size, n_levels, budget_per_level
        self.reset(0)

    # -- rules ----------------------------------------------------------
    def _draw_rules(self) -> None:
        r = self.rng
        moves = ["up", "down", "left", "right"]
        r.shuffle(moves)
        self.move_map = dict(zip(ACTIONS[:4], moves))         # which ACTIONk moves where
        self.family = r.choice(["key_door", "toggle", "teleport", "mirror"])
        self.key_colour = r.choice([1, 2, 3])                   # blue/red/green
        self.door_colour = r.choice([c for c in (1, 2, 3) if c != self.key_colour])
        self.toggle_colour = r.choice([6, 7, 8])                # magenta/orange/cyan
        self.tele_colour = r.choice([6, 7, 8])
        self.interact_action = r.choice(ACTIONS[4:])
        if self.family == "mirror":
            a, b = ACTIONS[0], ACTIONS[1]
            self.move_map[a], self.move_map[b] = self.move_map[b], self.move_map[a]

    def reset(self, seed: int) -> dict[str, Any]:
        self.rng = random.Random(seed)
        self._draw_rules()
        self.level = 0
        self.levels_done = 0
        self.actions_used = 0
        self.actions_this_level = 0
        self._done = False
        self.have_key = False
        self.history: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
        self._build_level()
        return self.observe()

    def _build_level(self) -> None:
        n = self.size
        g = [[EMPTY] * n for _ in range(n)]
        r = self.rng
        for _ in range(n * n // 6):
            x, y = r.randrange(n), r.randrange(n)
            g[y][x] = WALL
        self.pos = (1, 1)
        self.exit = (n - 2, n - 2)
        g[1][1] = EMPTY
        g[n - 2][n - 2] = EXIT
        self.specials: dict[tuple[int, int], int] = {}
        if self.family == "key_door":
            kx, ky = self._free(g)
            g[ky][kx] = self.key_colour
            # door ring around exit
            for dx, dy in ((-1, 0), (0, -1), (-1, -1)):
                g[self.exit[1] + dy][self.exit[0] + dx] = self.door_colour
            self.have_key = False
        elif self.family == "toggle":
            for dx, dy in ((-1, 0), (0, -1), (-1, -1)):
                g[self.exit[1] + dy][self.exit[0] + dx] = self.toggle_colour
        elif self.family == "teleport":
            ax, ay = self._free(g)
            g[ay][ax] = self.tele_colour
            bx, by = self.exit[0] - 1, self.exit[1]
            g[by][bx] = self.tele_colour
            self.tele_pair = {(ax, ay): (bx, by), (bx, by): (ax, ay)}
            for dx, dy in ((-1, 0), (0, -1), (-1, -1)):
                if (self.exit[0] + dx, self.exit[1] + dy) != (bx, by):
                    g[self.exit[1] + dy][self.exit[0] + dx] = WALL
        # guarantee a corridor from start to exit-neighbourhood so levels are solvable
        self._carve(g, self.pos, (self.exit[0] - 2, self.exit[1] - 2))
        self.grid = g
        self.actions_this_level = 0

    def _free(self, g) -> tuple[int, int]:
        while True:
            x, y = self.rng.randrange(2, self.size - 3), self.rng.randrange(2, self.size - 3)
            if g[y][x] == EMPTY:
                return x, y

    def _carve(self, g, a, b) -> None:
        x, y = a
        while (x, y) != b:
            if x != b[0]:
                x += 1 if b[0] > x else -1
            elif y != b[1]:
                y += 1 if b[1] > y else -1
            if g[y][x] in (WALL,):
                g[y][x] = EMPTY

    # -- dynamics -------------------------------------------------------
    def step(self, action: Any) -> dict[str, Any]:
        if self._done:
            return self.observe()
        a = str(action)
        if a not in ACTIONS:
            raise ValueError(f"unknown action {a!r}")
        before = self.pos
        self.actions_used += 1
        self.actions_this_level += 1
        if a in self.move_map:
            self._move(self.move_map[a])
        elif a == self.interact_action:
            self._interact()
        self.history.append((a, before, self.pos))
        if self.pos == self.exit:
            self.levels_done += 1
            self.level += 1
            if self.level >= self.n_levels:
                self._done = True
            else:
                self._build_level()
        elif self.actions_this_level >= self.budget_per_level:
            self._done = True
        return self.observe()

    def _passable(self, x, y) -> bool:
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        c = self.grid[y][x]
        if c == WALL:
            return False
        if self.family == "key_door" and c == self.door_colour and not self.have_key:
            return False
        if self.family == "toggle" and c == self.toggle_colour:
            return False
        return True

    def _move(self, d: str) -> None:
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if not self._passable(nx, ny):
            return
        self.pos = (nx, ny)
        c = self.grid[ny][nx]
        if self.family == "key_door" and c == self.key_colour:
            self.have_key = True
            self.grid[ny][nx] = EMPTY
        if self.family == "teleport" and c == self.tele_colour:
            self.pos = self.tele_pair.get(self.pos, self.pos)

    def _interact(self) -> None:
        if self.family != "toggle":
            return
        x, y = self.pos
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny][nx] == self.toggle_colour:
                    self.grid[ny][nx] = EMPTY

    # -- views ----------------------------------------------------------
    @property
    def done(self) -> bool:
        return self._done

    @property
    def score(self) -> float:
        return float(self.levels_done)

    def observe(self) -> dict[str, Any]:
        rows = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                row.append(AGENT if (x, y) == self.pos else self.grid[y][x])
            rows.append(row)
        return {
            "level": self.level + 1, "levels_total": self.n_levels,
            "actions_left_this_level": self.budget_per_level - self.actions_this_level,
            "grid": rows, "legend": {str(i): c for i, c in enumerate(COLOURS)},
            "agent_colour": COLOURS[AGENT], "done": self._done,
        }

    def info(self) -> dict[str, Any]:
        return {"levels_done": self.levels_done, "actions_used": self.actions_used,
                "family": self.family, "move_map": self.move_map, "interact": self.interact_action}

    def describe(self) -> str:
        return (
            f"A {self.size}x{self.size} colour-grid puzzle with {self.n_levels} levels and a budget of "
            f"{self.budget_per_level} actions per level. You control the white cell. Six actions "
            "ACTION1..ACTION6 exist; their effects are NOT given and must be discovered by experiment "
            "(they are fixed within an episode). Reaching the yellow cell completes a level. Some colours "
            "may be walls, keys, doors, toggles or teleports. Score = levels completed."
        )


from ..games.registry import register  # noqa: E402

register("rulepuzzle", "tier1")(RulePuzzleGame)
