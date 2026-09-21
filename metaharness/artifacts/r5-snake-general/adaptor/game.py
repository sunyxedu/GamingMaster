"""Original classic Snake instance; the Meta-Harness runner owns the clock."""
from __future__ import annotations

import hashlib
import threading
from copy import deepcopy
from typing import Any

from metaharness.games.base import ActionSpec

TIER = 2
DETERMINISTIC = True  # Reset and applied-action replay, not model arrival timing.
RULESET = "classic-walled-v1"
SCENARIOS = "snake-scenarios-v1"
DIRECTIONS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
PROFILES = {
    "demo-v1": (((10, 10),), 0.200),
    "primary-v1": (((8, 8), (10, 10), (12, 12), (16, 12)), 0.200),
    "generalization-odd-v1": (((9, 9), (11, 11), (13, 13)), 0.200),
    "stress-fast-v1": (((8, 8), (10, 10), (12, 12), (16, 12)), 0.100),
    "stress-slow-v1": (((8, 8), (10, 10), (12, 12), (16, 12)), 0.300),
}


class _RNG:
    """Version-stable uniform draws, independent of Python's random implementation."""

    def __init__(self, seed: int):
        self.key = f"{SCENARIOS}:{seed}:".encode("ascii")
        self.counter = 0

    def below(self, n: int) -> int:
        limit = (1 << 256) - ((1 << 256) % n)
        while True:
            value = int.from_bytes(hashlib.sha256(
                self.key + str(self.counter).encode("ascii")).digest(), "big")
            self.counter += 1
            if value < limit:
                return value % n


class SnakeGame:
    name = "Snake"
    action_spec = ActionSpec(
        names=("UP", "RIGHT", "DOWN", "LEFT", "NOOP"),
        descriptions={
            "UP": "Request one cell upward (y - 1).",
            "RIGHT": "Request one cell rightward (x + 1).",
            "DOWN": "Request one cell downward (y + 1).",
            "LEFT": "Request one cell leftward (x - 1).",
            "NOOP": "Continue in the current heading for this tick.",
        },
        noop="NOOP",
        notes="Absolute directions, one movement per tick; immediate reversal is ignored.",
    )

    def __init__(self, *, profile: str = "demo-v1"):
        if not isinstance(profile, str) or profile not in PROFILES:
            raise ValueError(f"profile must be one of {tuple(PROFILES)}")
        self.profile = profile
        self.tick_s = PROFILES[profile][1]
        self._lock = threading.RLock()
        self.reset(0)

    def reset(self, seed: int) -> dict[str, Any]:
        if type(seed) is not int:
            raise TypeError("seed must be an integer (not bool)")
        with self._lock:
            self._seed = seed
            self._rng = _RNG(seed)
            boards = PROFILES[self.profile][0]
            self.width, self.height = boards[self._rng.below(len(boards))]
            # D4 transform of centered offsets, keeping the selected rectangle fixed.
            transform = self._rng.below(8)
            offsets = [(1, 0), (0, 0), (-1, 0)]
            if transform >= 4:
                offsets = [(-x, y) for x, y in offsets]
            for _ in range(transform % 4):
                offsets = [(-y, x) for x, y in offsets]
            cx, cy = (self.width - 1) // 2, (self.height - 1) // 2
            self._body = [(cx + x, cy + y) for x, y in offsets]
            self._direction = next(k for k, v in DIRECTIONS.items() if v == offsets[0])
            self._ticks = 0
            self._status = "ready"
            self._reason = None
            self._food = self._draw_food()
            self._attempted = None
            self._last_requested = None
            self._last_applied = None
            self._reverse_ignored = False
            self._records: list[dict[str, Any]] = []
            return self.observe()

    def _draw_food(self) -> tuple[int, int]:
        occupied = set(self._body)
        empty = [(x, y) for y in range(self.height) for x in range(self.width)
                 if (x, y) not in occupied]
        return empty[self._rng.below(len(empty))]

    def tick(self, action: Any) -> None:
        with self._lock:
            if self.done:
                return
            if action is None:
                action = "NOOP"
            if not isinstance(action, str) or action not in self.action_spec.names:
                raise ValueError("action must be UP, RIGHT, DOWN, LEFT or NOOP")
            heading = self._direction if action == "NOOP" else action
            dx, dy = DIRECTIONS[self._direction]
            self._reverse_ignored = DIRECTIONS[heading] == (-dx, -dy)
            if self._reverse_ignored:
                heading = self._direction
            self._direction = heading
            self._last_requested = action
            self._last_applied = heading
            dx, dy = DIRECTIONS[heading]
            x, y = self._body[0]
            destination = (x + dx, y + dy)
            self._attempted = destination
            growing = destination == self._food
            self._ticks += 1
            self._status = "running"
            nx, ny = destination
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                self._status, self._reason = "dead", "wall_collision"
            elif destination in (self._body if growing else self._body[:-1]):
                self._status, self._reason = "dead", "body_collision"
            else:
                self._body.insert(0, destination)
                if not growing:
                    self._body.pop()
                elif len(self._body) == self.width * self.height:
                    self._food = None
                    self._status, self._reason = "won", "board_full"
                else:
                    self._food = self._draw_food()
            self._records.append({"tick": self._ticks, "requested": action,
                                  "applied": heading,
                                  "reverse_ignored": self._reverse_ignored})

    def truncate(self, reason: str = "external_tick_cap") -> None:
        """Evaluator hook; never part of the gameplay action vocabulary."""
        if reason not in ("external_tick_cap", "external_cancel", "infrastructure_failure"):
            raise ValueError("invalid truncation reason")
        with self._lock:
            if not self.done:
                self._status, self._reason = "truncated", reason

    @property
    def done(self) -> bool:
        with self._lock:
            return self._status in ("dead", "won", "truncated")

    @property
    def ticks(self) -> int:
        with self._lock:
            return self._ticks

    @property
    def score(self) -> float:
        with self._lock:
            return (len(self._body) - 3) / (self.width * self.height - 3)

    @property
    def recommended_tick_cap(self) -> int:
        with self._lock:
            return 2 * (self.width * self.height) ** 2

    def observe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1, "ruleset_id": RULESET,
                "scenario_version": SCENARIOS, "profile": self.profile,
                "tick": self._ticks, "status": self._status,
                "width": self.width, "height": self.height, "initial_length": 3,
                "body": [list(cell) for cell in self._body],
                "direction": self._direction,
                "food": list(self._food) if self._food is not None else None,
                "length": len(self._body), "food_eaten": len(self._body) - 3,
                "score": len(self._body) - 3, "completion": self.score,
                "tick_interval_s": self.tick_s,
                "last_requested_action": self._last_requested,
                "last_applied_action": self._last_applied,
                "reverse_ignored": self._reverse_ignored,
                "attempted_destination": list(self._attempted) if self._attempted else None,
                "terminated": self._status in ("dead", "won"),
                "truncated": self._status == "truncated", "won": self._status == "won",
                "end_reason": self._reason,
            }

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {"ruleset_id": RULESET, "scenario_version": SCENARIOS,
                    "profile": self.profile, "clock_owner": "runner",
                    "width": self.width, "height": self.height,
                    "food_eaten": len(self._body) - 3, "raw_score": len(self._body) - 3,
                    "completion": self.score, "score": self.score,
                    "won": self._status == "won", "ticks": self._ticks,
                    "terminated": self._status in ("dead", "won"),
                    "truncated": self._status == "truncated", "end_reason": self._reason,
                    "recommended_tick_cap": self.recommended_tick_cap}

    def replay(self) -> dict[str, Any]:
        """Evaluator-only replay data; seed never appears in player observations."""
        with self._lock:
            return {"ruleset_id": RULESET, "scenario_version": SCENARIOS,
                    "profile": self.profile, "seed": self._seed,
                    "actions": deepcopy(self._records), "final": self.observe()}

    def describe(self) -> str:
        return (
            "Classic walled Snake. Eat food and fill the board. Coordinates are [x,y], "
            "origin top left; x increases right, y down. Body is head first, tail last. "
            "Start length is 3. Every tick moves exactly one cell. UP/RIGHT/DOWN/LEFT "
            "are absolute directions; NOOP continues straight. Reversals are ignored. "
            "The runner applies its current mailbox action each tick, even while the "
            "controller is thinking; late decisions can affect a later tick. "
            "Eating grows by one and spawns food uniformly in empty cells. Otherwise "
            "the tail moves away, and entering that departing tail is legal. Walls "
            "and the retained body kill; a fatal attempt increments tick but preserves "
            "body and score. Filling the board wins immediately. There is no wrapping, "
            "starvation, obstacle, opponent or acceleration. Terminal states freeze. "
            f"Profile {self.profile}; movement interval {self.tick_s:.3f} seconds. "
            "Default demo boards are 10x10; primary boards are 8x8, 10x10, 12x12 or "
            "16x12, with central rotated/reflected straight starts. "
            "Game.score is completion = food_eaten/(width*height-3), from 0 to 1. "
            "Observation score is raw food_eaten = length-3; completion is normalized. "
            "Surviving without eating earns nothing. Evaluators cap at 2*(width*height)^2 "
            "ticks, marking truncation separately and preserving earned score. "
            "Observations expose schema/ruleset/scenario versions, profile, tick, status, "
            "width/height/initial_length, body/direction/food (null on victory), length, "
            "score/food_eaten/completion, tick_interval_s, last_requested_action, "
            "last_applied_action, reverse_ignored, attempted_destination (null before "
            "first tick), terminated/truncated/won and end_reason. No future food or "
            "RNG state is exposed. Ready means reset with no tick yet; running means "
            "at least one tick applied. No clock deadlines are owned by the adaptor."
        )


def make_game(**kwargs: Any) -> SnakeGame:
    return SnakeGame(**kwargs)
