"""Lane Runner — a second tier-2 game (Subway-Surfers-like), used as the
*held-out game* for tier-2 harness generality.

Three lanes; obstacles scroll toward the runner one row per tick. Actions:
``left`` / ``right`` (change lane), ``jump`` (clears a *low* obstacle in the
current lane this tick), ``stay``. Hitting a ``high`` obstacle, or a ``low``
one without jumping, ends the episode. Coins give +1; surviving a tick gives
+0.1. Observation keys deliberately differ from snake's so a harness that
hard-codes snake fields cannot work here.
"""
from __future__ import annotations

import random
from typing import Any

from .base import ActionSpec
from .registry import register

LANES = 3


@register("lanerunner", "tier2")
class LaneRunnerGame:
    name = "lanerunner"
    action_spec = ActionSpec(
        names=["left", "right", "jump", "stay"],
        descriptions={"left": "move one lane left", "right": "move one lane right",
                      "jump": "jump this tick (clears a LOW obstacle in your lane)", "stay": "do nothing"},
        noop="stay",
        notes="Actions apply on the tick they are read. Moving off the edge is ignored. "
              "A HIGH obstacle can only be avoided by changing lane.",
    )

    def __init__(self, tick_s: float = 0.5, max_ticks: int = 300, lookahead: int = 6, density: float = 0.35) -> None:
        self.tick_s, self.max_ticks, self.lookahead, self.density = tick_s, max_ticks, lookahead, density
        self.reset(0)

    def reset(self, seed: int) -> dict[str, Any]:
        self.rng = random.Random(seed)
        self.lane = 1
        self._ticks = 0
        self._score = 0.0
        self._done = False
        self._death = None
        self.rows: list[list[str]] = [self._empty_row() for _ in range(self.lookahead)]
        for i in range(2, self.lookahead):
            self.rows[i] = self._gen_row()
        return self.observe()

    def _empty_row(self) -> list[str]:
        return ["."] * LANES

    def _gen_row(self) -> list[str]:
        row = self._empty_row()
        for i in range(LANES):
            r = self.rng.random()
            if r < self.density * 0.5:
                row[i] = "low"
            elif r < self.density:
                row[i] = "high"
            elif r < self.density + 0.15:
                row[i] = "coin"
        if all(c == "high" for c in row):           # always leave an escape
            row[self.rng.randrange(LANES)] = "."
        return row

    def tick(self, action: Any) -> None:
        if self._done:
            return
        a = str(action) if action is not None else "stay"
        if a == "left":
            self.lane = max(0, self.lane - 1)
        elif a == "right":
            self.lane = min(LANES - 1, self.lane + 1)
        self._ticks += 1
        incoming = self.rows.pop(0)
        self.rows.append(self._gen_row())
        cell = incoming[self.lane]
        if cell == "high" or (cell == "low" and a != "jump"):
            self._done, self._death = True, f"hit_{cell}"
            return
        if cell == "coin":
            self._score += 1.0
        self._score += 0.1
        if self._ticks >= self.max_ticks:
            self._done, self._death = True, "timeout"

    @property
    def done(self) -> bool:
        return self._done

    @property
    def score(self) -> float:
        return round(self._score, 2)

    @property
    def ticks(self) -> int:
        return self._ticks

    def observe(self) -> dict[str, Any]:
        return {
            "t": self._ticks, "lane": self.lane, "n_lanes": LANES,
            "upcoming": [{"in_ticks": i + 1, "lanes": list(r)} for i, r in enumerate(self.rows)],
            "next_row": list(self.rows[0]),
            "points": self.score,
            "view": self.render_ascii(),
        }

    def render_ascii(self) -> str:
        sym = {".": ".", "low": "_", "high": "#", "coin": "$"}
        lines = [" ".join(sym[c] for c in r) + f"   <- in {i + 1}" for i, r in enumerate(self.rows)]
        lines.append(" ".join("R" if i == self.lane else " " for i in range(LANES)) + "   <- you")
        return "\n".join(lines)

    def info(self) -> dict[str, Any]:
        return {"ticks": self._ticks, "score": self.score, "death": self._death}

    def describe(self) -> str:
        return (
            f"Lane runner with {LANES} lanes, one tick every {self.tick_s}s, max {self.max_ticks} ticks. Rows of "
            "obstacles approach one row per tick; 'upcoming[0]' is the row you meet on the NEXT tick. 'low' "
            "obstacles are cleared by 'jump' on that tick, 'high' only by not being in that lane, 'coin' gives +1, "
            "each survived tick +0.1. Observation: lane, upcoming rows (nearest first) and an ASCII view "
            "(# high, _ low, $ coin, R you)."
        )
