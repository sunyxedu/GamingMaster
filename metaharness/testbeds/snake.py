"""Real-time Snake — the tier-2 test bed.

Classic rules on a W×H grid: eat food (+1), die on wall / self. The snake
keeps moving in its current direction every tick whether or not the agent
decided; reversing is ignored. Observation is a plain dict a harness can pass
to Jev directly or render for Astra.
"""
from __future__ import annotations

import random
from typing import Any

from ..games.base import ActionSpec

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class SnakeGame:
    name = "snake"
    action_spec = ActionSpec(
        names=["up", "down", "left", "right", "keep"],
        descriptions={
            "up": "face up (y-1)", "down": "face down (y+1)",
            "left": "face left (x-1)", "right": "face right (x+1)",
            "keep": "keep current heading",
        },
        noop="keep",
        notes="Reversing into yourself is ignored (heading unchanged). Walls kill. "
              "Grid coordinates: x to the right, y downward, origin top-left.",
    )

    def __init__(self, width: int = 12, height: int = 12, tick_s: float = 0.25,
                 max_ticks: int = 400, n_food_target: int | None = None) -> None:
        self.width, self.height = width, height
        self.tick_s = tick_s
        self.max_ticks = max_ticks
        self.n_food_target = n_food_target
        self.reset(0)

    # -- lifecycle ------------------------------------------------------
    def reset(self, seed: int) -> dict[str, Any]:
        self.rng = random.Random(seed)
        cx, cy = self.width // 2, self.height // 2
        self.body: list[tuple[int, int]] = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.heading = "right"
        self._ticks = 0
        self._score = 0.0
        self._done = False
        self._death: str | None = None
        self.food = self._spawn_food()
        return self.observe()

    def _spawn_food(self) -> tuple[int, int]:
        free = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.body]
        return self.rng.choice(free) if free else (-1, -1)

    # -- dynamics -------------------------------------------------------
    def tick(self, action: Any) -> None:
        if self._done:
            return
        a = str(action) if action is not None else "keep"
        if a in DIRS and a != OPPOSITE[self.heading]:
            self.heading = a
        dx, dy = DIRS[self.heading]
        hx, hy = self.body[0]
        nx, ny = hx + dx, hy + dy
        self._ticks += 1
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            self._done, self._death = True, "wall"
            return
        grow = (nx, ny) == self.food
        tail = self.body[:-1] if not grow else self.body
        if (nx, ny) in tail:
            self._done, self._death = True, "self"
            return
        self.body = [(nx, ny)] + list(tail)
        if grow:
            self._score += 1
            self.food = self._spawn_food()
            if self.n_food_target and self._score >= self.n_food_target:
                self._done, self._death = True, "target_reached"
        if self._ticks >= self.max_ticks:
            self._done, self._death = True, "timeout"

    # -- views ----------------------------------------------------------
    @property
    def done(self) -> bool:
        return self._done

    @property
    def score(self) -> float:
        return self._score

    @property
    def ticks(self) -> int:
        return self._ticks

    def danger(self) -> dict[str, bool]:
        hx, hy = self.body[0]
        out = {}
        for d, (dx, dy) in DIRS.items():
            nx, ny = hx + dx, hy + dy
            out[d] = not (0 <= nx < self.width and 0 <= ny < self.height) or (nx, ny) in self.body[:-1]
        return out

    def observe(self) -> dict[str, Any]:
        hx, hy = self.body[0]
        fx, fy = self.food
        return {
            "tick": self._ticks,
            "grid_size": [self.width, self.height],
            "head": [hx, hy],
            "heading": self.heading,
            "body": [list(p) for p in self.body],
            "length": len(self.body),
            "food": [fx, fy],
            "food_delta": [fx - hx, fy - hy],
            "danger": self.danger(),
            "score": self._score,
            "grid": self.render_ascii(),
        }

    def render_ascii(self) -> str:
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                p = (x, y)
                if p == self.body[0]:
                    row.append({"up": "^", "down": "v", "left": "<", "right": ">"}[self.heading])
                elif p in self.body:
                    row.append("o")
                elif p == self.food:
                    row.append("*")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)

    def info(self) -> dict[str, Any]:
        return {"ticks": self._ticks, "score": self._score, "length": len(self.body), "death": self._death}

    def describe(self) -> str:
        return (
            f"Snake on a {self.width}x{self.height} grid, one tick every {self.tick_s:.3f}s, "
            f"max {self.max_ticks} ticks. +1 per food eaten; episode ends on wall/self collision. "
            "The snake advances one cell per tick in its heading regardless of whether you decided. "
            "Observation: dict with head, heading, body, food, food_delta, danger{up,down,left,right} "
            "(True = that cell kills), and an ASCII grid (head=arrow, body=o, food=*)."
        )


from ..games.registry import register  # noqa: E402

register("snake", "tier2")(SnakeGame)
