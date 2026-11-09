"""Headless, deterministic, clock-driven classic Snake.

This adaptor implements the local symbolic Snake contract selected during the
study phase.  It does not launch a GUI, process, server, or network client:
``tick`` is the one-cell game clock that the tier-2 runner calls at the
scenario's cadence.  The class is deliberately small enough that all collision,
food, and seeded-reset rules are visible here rather than delegated to an
unrelated Snake package with different semantics.
"""
from __future__ import annotations

import copy
import math
import numbers
import random
import threading
from typing import Any

try:
    from metaharness.games.base import ActionSpec
except ImportError:  # pragma: no cover - permits a standalone local smoke test
    from dataclasses import dataclass, field

    @dataclass
    class ActionSpec:  # type: ignore[no-redef]
        names: list[str]
        descriptions: dict[str, str] = field(default_factory=dict)
        noop: str | None = None
        notes: str = ""

        def describe(self) -> str:
            lines = [f"- {name}: {self.descriptions.get(name, '')}".rstrip()
                     for name in self.names]
            if self.noop:
                lines.append(
                    f"(real-time games: ticks apply '{self.noop}' until the first action is posted; "
                    "afterwards a tick without a fresh action repeats the last posted action)"
                )
            if self.notes:
                lines.append(self.notes)
            return "\n".join(lines)


TIER = 2
DETERMINISTIC = True

# The evaluator uses these as the ordinary training/held-out seed split.  The
# seed is not a map ID: it drives the initial valid opening and the episode RNG
# used for every future food placement.
SEEDS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "heldout": [101, 102, 103, 104, 105, 106, 107],
}

# Reference profile from study.json.  rows/cols/tick_s/max_ticks are also
# constructor options for declared rectangular/timing generalisation cases.
EVAL_KWARGS = {
    "rows": 12,
    "cols": 12,
    "tick_s": 0.10,
    "max_ticks": 4000,
}

# Explicitly named optional evaluation profiles.  They change only the fixed
# board/timing budget; all Snake rules remain identical.  The odd-by-odd
# profile is intentionally included because a simple rectangular Hamiltonian
# cycle is not available there.
SCENARIO_OPTIONS = {
    "reference-12x12": {"rows": 12, "cols": 12, "tick_s": 0.10, "max_ticks": 4000},
    "even-8x8-fast": {"rows": 8, "cols": 8, "tick_s": 0.08, "max_ticks": 4000},
    "rect-10x14": {"rows": 10, "cols": 14, "tick_s": 0.15, "max_ticks": 6000},
    "rect-14x10": {"rows": 14, "cols": 10, "tick_s": 0.25, "max_ticks": 6000},
    "odd-11x11": {"rows": 11, "cols": 11, "tick_s": 0.10, "max_ticks": 5000},
}

_VERSION = "local-classic-v1"
_INITIAL_LENGTH = 3
_DIRECTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
_DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
}
_OPPOSITE = {"UP": "DOWN", "RIGHT": "LEFT", "DOWN": "UP", "LEFT": "RIGHT"}
_INTEGER_ACTIONS = {index: name for index, name in enumerate(_DIRECTION_NAMES)}

_ACTION_DESCRIPTIONS = {
    "UP": "Steer the head one cell toward decreasing row on the next tick.",
    "RIGHT": "Steer the head one cell toward increasing column on the next tick.",
    "DOWN": "Steer the head one cell toward increasing row on the next tick.",
    "LEFT": "Steer the head one cell toward decreasing column on the next tick.",
    "NOOP": "Do not change heading; the snake continues in its current direction.",
}


class SnakeGame:
    """A symbolic RealtimeGame for bounded, single-food classic Snake.

    Coordinates are ``[row, column]`` with row zero at the top.  The body is
    stored head-first.  Calls are protected by an RLock so observe/reset/tick
    remain safe if a runner elects to inspect from a helper thread; there is no
    hidden background clock.
    """

    name = "Snake (local classic)"
    action_spec = ActionSpec(
        names=list(_DIRECTION_NAMES) + ["NOOP"],
        descriptions=dict(_ACTION_DESCRIPTIONS),
        noop="NOOP",
        notes=(
            "Actions are absolute names (or integer aliases 0=UP, 1=RIGHT, "
            "2=DOWN, 3=LEFT); a dict {'action': name, 'state_tick': t} is also "
            "accepted. NOOP and a missing action retain the current heading. "
            "An immediate reverse is ignored, never converted into a death. "
            "Malformed or stale actions are ignored and reported in the next "
            "observation. One call to tick advances exactly one game cell."
        ),
    )

    def __init__(
        self,
        *,
        rows: int = 12,
        cols: int = 12,
        tick_s: float = 0.10,
        max_ticks: int = 4000,
        scenario: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._lock = threading.RLock()
        if "profile" in kwargs:
            if scenario is not None:
                raise ValueError("specify only one of profile or scenario")
            scenario = kwargs.pop("profile")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected constructor option(s): {unknown}")
        if scenario is not None:
            if not isinstance(scenario, str) or scenario not in SCENARIO_OPTIONS:
                raise ValueError(f"scenario must be one of {tuple(SCENARIO_OPTIONS)}")
            profile = SCENARIO_OPTIONS[scenario]
            # Explicit numeric kwargs may be used to define an ad-hoc evaluation
            # profile, but a named scenario cannot be silently contradicted.
            supplied = {"rows": rows, "cols": cols, "tick_s": tick_s, "max_ticks": max_ticks}
            defaults = {"rows": 12, "cols": 12, "tick_s": 0.10, "max_ticks": 4000}
            for key, value in profile.items():
                if supplied[key] != defaults[key] and supplied[key] != value:
                    raise ValueError(f"{key} conflicts with named scenario {scenario!r}")
            rows, cols = profile["rows"], profile["cols"]
            tick_s, max_ticks = profile["tick_s"], profile["max_ticks"]
        self.rows = self._positive_int(rows, "rows", minimum=4)
        self.cols = self._positive_int(cols, "cols", minimum=4)
        if self.rows * self.cols > 10_000:
            raise ValueError("rows * cols must be at most 10000 so observations stay compact")
        if isinstance(tick_s, bool) or not isinstance(tick_s, numbers.Real):
            raise TypeError("tick_s must be a finite positive number")
        self.tick_s = float(tick_s)
        if not math.isfinite(self.tick_s) or self.tick_s <= 0.0 or self.tick_s >= 10.0:
            raise ValueError("tick_s must be finite, greater than 0, and below 10 seconds")
        self.max_ticks = self._positive_int(max_ticks, "max_ticks", minimum=1)
        if scenario is not None:
            if not isinstance(scenario, str) or not scenario.strip() or len(scenario) > 100:
                raise ValueError("scenario must be a short non-empty string or None")
            self._scenario_override = scenario.strip()
        else:
            self._scenario_override = None
        self._closed = False
        self._seed = 0
        self._rng = random.Random(0)
        self._snake: list[tuple[int, int]] = []
        self._food: tuple[int, int] | None = None
        self._direction = "RIGHT"
        self._ticks = 0
        self._score = 0
        self._last_reward = 0
        self._done = False
        self._terminated = False
        self._truncated = False
        self._terminal_reason: str | None = None
        self._last_action = "NOOP"
        self._last_action_error: str | None = None
        self._last_event = "reset"
        self._attempted_head: tuple[int, int] | None = None
        self._action_tick = 0
        self._invalid_action_count = 0
        self._reverse_ignored_count = 0
        self._stale_action_count = 0
        self._pending_direction: str | None = None
        self._pending_state_tick: int | None = None
        self.reset(0)

    @staticmethod
    def _positive_int(value: Any, name: str, *, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer")
        result = int(value)
        if result < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return result

    @property
    def scenario(self) -> str:
        if self._scenario_override is not None:
            return self._scenario_override
        if self.rows == 12 and self.cols == 12 and self.tick_s == 0.10 and self.max_ticks == 4000:
            return "reference-12x12"
        return f"rect-{self.rows}x{self.cols}-tick-{self.tick_s:g}"

    @property
    def initial_length(self) -> int:
        return _INITIAL_LENGTH

    @property
    def _cell_count(self) -> int:
        return self.rows * self.cols

    @property
    def _max_score(self) -> int:
        return self._cell_count - _INITIAL_LENGTH

    @staticmethod
    def _in_bounds(cell: tuple[int, int], rows: int, cols: int) -> bool:
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols

    def _central_openings(self) -> list[tuple[tuple[int, int], str]]:
        """Enumerate valid near-centre length-three openings deterministically."""
        row_start = max(0, self.rows // 2 - 1)
        col_start = max(0, self.cols // 2 - 1)
        central_rows = range(row_start, min(self.rows, row_start + 3))
        central_cols = range(col_start, min(self.cols, col_start + 3))
        candidates: list[tuple[tuple[int, int], str]] = []
        for row in central_rows:
            for col in central_cols:
                head = (row, col)
                for direction in _DIRECTION_NAMES:
                    dr, dc = _DIRECTION_VECTORS[direction]
                    neck = (row - dr, col - dc)
                    tail = (row - 2 * dr, col - 2 * dc)
                    forward = (row + dr, col + dc)
                    if all(self._in_bounds(cell, self.rows, self.cols)
                           for cell in (head, neck, tail, forward)):
                        candidates.append((head, direction))
        if not candidates:  # guarded by the constructor's dimensions, but explicit
            raise ValueError("board has no valid length-three central opening")
        return candidates

    def _sample_food(self) -> tuple[int, int] | None:
        occupied = set(self._snake)
        free = [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) not in occupied
        ]
        if not free:
            return None
        # The row-major list and randrange make the RNG mapping independent of
        # set iteration and stable near a full board.
        return free[self._rng.randrange(len(free))]

    def reset(self, seed: int, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Start a fresh seeded episode and return its first JSON observation.

        ``options`` is accepted for Gymnasium-shaped callers but does not
        silently alter the fixed constructor scenario.  The constructor is the
        explicit place to select a board/timing profile.
        """
        with self._lock:
            if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
                raise TypeError("seed must be an integer")
            if options is not None and not isinstance(options, dict):
                raise TypeError("options must be a dict or None")
            self._closed = False
            self._seed = int(seed)
            self._rng = random.Random(self._seed)
            self._ticks = 0
            self._score = 0
            self._last_reward = 0
            self._done = False
            self._terminated = False
            self._truncated = False
            self._terminal_reason = None
            self._last_action = "NOOP"
            self._last_action_error = None
            self._last_event = "reset"
            self._attempted_head = None
            self._action_tick = 0
            self._invalid_action_count = 0
            self._reverse_ignored_count = 0
            self._stale_action_count = 0
            self._pending_direction = None
            self._pending_state_tick = None

            openings = self._central_openings()
            head, self._direction = openings[self._rng.randrange(len(openings))]
            dr, dc = _DIRECTION_VECTORS[self._direction]
            self._snake = [head, (head[0] - dr, head[1] - dc),
                           (head[0] - 2 * dr, head[1] - 2 * dc)]
            self._food = self._sample_food()
            if self._food is None:  # impossible for dimensions >=4 and length 3
                raise RuntimeError("failed to place initial food")
            return self.observe()

    @staticmethod
    def _safe_action_text(action: Any) -> str:
        if action is None:
            return "<none>"
        if isinstance(action, (str, int, float, bool)):
            return repr(action)[:80]
        return f"<{type(action).__name__}>"

    def _normalise_action(self, action: Any) -> tuple[str | None, int | None, str | None]:
        """Return (direction/noop, state_tick, error-kind)."""
        state_tick: int | None = None
        if action is None:
            return "NOOP", None, None
        if isinstance(action, dict):
            if "state_tick" in action:
                raw_tick = action.get("state_tick")
                if isinstance(raw_tick, bool) or not isinstance(raw_tick, numbers.Integral):
                    return None, None, "invalid_state_tick"
                state_tick = int(raw_tick)
            if "action" in action:
                action = action.get("action")
            elif "direction" in action:
                action = action.get("direction")
            else:
                return None, state_tick, "invalid_action"
        if isinstance(action, str):
            name = action.strip().upper()
            if name in _DIRECTION_NAMES or name == "NOOP":
                return name, state_tick, None
            # These are harmless spellings for the documented NOOP concept;
            # they are not advertised as additional strategic actions.
            if name in {"KEEP", "STAY", "STRAIGHT"}:
                return "NOOP", state_tick, None
            return None, state_tick, "invalid_action"
        if isinstance(action, numbers.Integral) and not isinstance(action, bool):
            return _INTEGER_ACTIONS.get(int(action)), state_tick, (
                None if int(action) in _INTEGER_ACTIONS else "invalid_action"
            )
        return None, state_tick, "invalid_action"

    def set_direction(self, direction: Any, state_tick: int) -> bool:
        """Queue one steering request for the next tick if its snapshot is current.

        This optional low-level helper mirrors the study API.  Normal harnesses
        should call ``tick(action)``; a queued request is consumed by the next
        ``tick(None)``.  It never rewinds a request to an old game state.
        """
        with self._lock:
            if isinstance(state_tick, bool) or not isinstance(state_tick, numbers.Integral):
                self._last_action_error = "invalid_state_tick"
                self._last_event = "invalid_state_tick"
                self._invalid_action_count += 1
                return False
            if int(state_tick) != self._ticks:
                self._last_action_error = "stale_action"
                self._last_event = "stale_action"
                self._stale_action_count += 1
                return False
            parsed, parsed_tick, error = self._normalise_action(direction)
            if error is not None or parsed is None:
                self._last_action_error = error or "invalid_action"
                self._last_event = self._last_action_error
                self._invalid_action_count += 1
                return False
            if parsed_tick is not None and parsed_tick != self._ticks:
                self._last_action_error = "stale_action"
                self._last_event = "stale_action"
                self._stale_action_count += 1
                return False
            self._pending_direction = parsed
            self._pending_state_tick = self._ticks
            return True

    def _is_safe_direction(self, direction: str) -> bool:
        if self._done:
            return False
        dr, dc = _DIRECTION_VECTORS[direction]
        head = self._snake[0]
        next_head = (head[0] + dr, head[1] + dc)
        if not self._in_bounds(next_head, self.rows, self.cols):
            return False
        if next_head == self._food:
            # Food keeps the current tail, so every body cell remains occupied.
            return next_head not in set(self._snake)
        # On a non-food transition the tail leaves at the same instant.
        return next_head not in set(self._snake[:-1])

    def _commit_terminal(self, reason: str, *, truncated: bool = False) -> None:
        self._done = True
        self._terminal_reason = reason
        self._truncated = bool(truncated)
        self._terminated = not self._truncated

    def _advance_one_tick(self, requested: str | None, error: str | None) -> None:
        self._last_reward = 0
        self._attempted_head = None
        self._last_action_error = error
        if error is not None:
            self._last_event = error
            if error == "stale_action":
                self._stale_action_count += 1
            else:
                self._invalid_action_count += 1
        else:
            self._last_event = "move"

        # A malformed/stale request and NOOP both preserve the committed heading.
        if requested in _DIRECTION_NAMES:
            self._last_action = requested
            if requested == _OPPOSITE[self._direction]:
                self._last_action_error = "reverse_ignored"
                self._last_event = "reverse_ignored"
                self._reverse_ignored_count += 1
            elif requested != self._direction:
                self._direction = requested
        elif requested == "NOOP" or requested is None:
            self._last_action = "NOOP"
        else:
            self._last_action = "NOOP"

        dr, dc = _DIRECTION_VECTORS[self._direction]
        head = self._snake[0]
        next_head = (head[0] + dr, head[1] + dc)
        self._attempted_head = next_head
        self._ticks += 1
        self._action_tick = self._ticks

        if not self._in_bounds(next_head, self.rows, self.cols):
            self._commit_terminal("wall_collision")
            return

        ate_food = next_head == self._food
        occupied = set(self._snake if ate_food else self._snake[:-1])
        if next_head in occupied:
            self._commit_terminal("self_collision")
            return

        if ate_food:
            self._snake = [next_head] + self._snake
            self._score += 1
            self._last_reward = 1
            self._last_event = "food_eaten"
            if len(self._snake) == self._cell_count:
                self._food = None
                self._commit_terminal("filled_board")
                return
            self._food = self._sample_food()
        else:
            self._snake = [next_head] + self._snake[:-1]

        # A collision or full-board win on the limit tick has already returned;
        # only a still-live episode becomes an external truncation.
        if self._ticks >= self.max_ticks:
            self._commit_terminal("time_limit", truncated=True)

    def tick(self, action: Any) -> None:
        """Apply one action and advance one fixed-cadence game tick.

        The runner owns the wall clock.  This method intentionally performs no
        sleep and never advances a terminal or closed episode.
        """
        with self._lock:
            if self._closed or self._done:
                return
            if action is None and self._pending_direction is not None:
                requested = self._pending_direction
                requested_tick = self._pending_state_tick
                self._pending_direction = None
                self._pending_state_tick = None
                if requested_tick != self._ticks:
                    requested, error = None, "stale_action"
                else:
                    requested, error = requested, None
            else:
                self._pending_direction = None
                self._pending_state_tick = None
                requested, requested_tick, error = self._normalise_action(action)
                if error is None and requested_tick is not None and requested_tick != self._ticks:
                    requested, error = None, "stale_action"
            self._advance_one_tick(requested, error)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Deterministic one-tick compatibility API for replay/unit tests."""
        with self._lock:
            before = self._score
            self.tick(action)
            reward = float(self._score - before)
            return self.observe(), reward, self._terminated, self._truncated, self.info()

    def _observation(self) -> dict[str, Any]:
        head = self._snake[0]
        tail = self._snake[-1]
        nonreverse = [] if self._done else [
            direction for direction in _DIRECTION_NAMES
            if direction != _OPPOSITE[self._direction]
        ]
        safe = [] if self._done else [
            direction for direction in _DIRECTION_NAMES
            if direction != _OPPOSITE[self._direction] and self._is_safe_direction(direction)
        ]
        max_score = self._max_score
        return {
            "game": "snake",
            "version": _VERSION,
            "seed": self._seed,
            "scenario": self.scenario,
            "rows": self.rows,
            "cols": self.cols,
            "tick": self._ticks,
            "sim_time_s": round(self._ticks * self.tick_s, 6),
            "tick_s": self.tick_s,
            "deadline_in_s": 0.0 if self._done else self.tick_s,
            "snake": [[row, col] for row, col in self._snake],
            "head": [head[0], head[1]],
            "tail": [tail[0], tail[1]],
            "direction": self._direction,
            "food": None if self._food is None else [self._food[0], self._food[1]],
            "score": self._score,
            "length": len(self._snake),
            "free_cells": self._cell_count - len(self._snake),
            "max_score": max_score,
            "score_fraction": round(self._score / max_score, 12),
            "nonreverse_directions": nonreverse,
            "safe_directions_now": safe,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "done": self._done,
            "terminal_reason": self._terminal_reason,
            "filled_board": self._terminal_reason == "filled_board",
            "last_reward": self._last_reward,
            "last_action": self._last_action,
            "last_action_error": self._last_action_error,
            "last_event": self._last_event,
            "attempted_head": None if self._attempted_head is None else [
                self._attempted_head[0], self._attempted_head[1]
            ],
            "input_age_s": 0.0,
            "action_tick": self._action_tick,
            "ticks_survived": self._ticks,
        }

    def observe(self) -> dict[str, Any]:
        with self._lock:
            # Never expose the mutable body/RNG state to the controller.
            return copy.deepcopy(self._observation())

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._done)

    @property
    def score(self) -> float:
        with self._lock:
            return float(self._score)

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            max_score = self._max_score
            return {
                "food_count": float(self._score),
                "final_length": float(len(self._snake)),
                "ticks_survived": float(self._ticks),
                "score_fraction": float(self._score / max_score),
                "filled_board": float(self._terminal_reason == "filled_board"),
                "wall_collision": float(self._terminal_reason == "wall_collision"),
                "self_collision": float(self._terminal_reason == "self_collision"),
                "time_limit": float(self._terminal_reason == "time_limit"),
            }

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "valid_trial": True,
                "invalid_reason": None,
                "rules_version": _VERSION,
                "seed": self._seed,
                "scenario": self.scenario,
                "rows": self.rows,
                "cols": self.cols,
                "tick_s": self.tick_s,
                "max_ticks": self.max_ticks,
                "initial_length": _INITIAL_LENGTH,
                "score": float(self._score),
                "foods_eaten": self._score,
                "final_length": len(self._snake),
                "ticks_survived": self._ticks,
                "terminal_reason": self._terminal_reason,
                "terminated": self._terminated,
                "truncated": self._truncated,
                "filled_board": self._terminal_reason == "filled_board",
                "max_score": self._max_score,
                "score_fraction": float(self._score / self._max_score),
                "invalid_action_count": self._invalid_action_count,
                "reverse_ignored_count": self._reverse_ignored_count,
                "stale_action_count": self._stale_action_count,
                "last_action_error": self._last_action_error,
                "attempted_head": None if self._attempted_head is None else list(self._attempted_head),
            }

    def describe(self) -> str:
        return (
            "This is a local, headless classic Snake on a bounded rectangular grid. "
            "The reference profile is 12 rows by 12 columns, zero-based [row, column] "
            "coordinates, no wall wrapping, one food item, initial length three near the "
            "centre, tick_s=0.10 seconds, and max_ticks=4000. The configured rows, cols, "
            "tick_s, and max_ticks are fixed for an episode and are shown in every observation. "
            "The objective is to eat food: score is exactly the cumulative food count and each "
            "food adds one body cell. Survival without food earns no score. The episode ends on "
            "wall_collision, self_collision, or filled_board; a still-live snake at max_ticks is "
            "truncated with terminal_reason=time_limit. A filled board wins even on the limit tick.\n\n"
            "The snake list is head-first and contains [row, column] cells. On every tick the "
            "head advances exactly one cell in the current absolute cardinal direction. UP is "
            "row-1, RIGHT is column+1, DOWN is row+1, and LEFT is column-1. An action request "
            "in the immediate opposite direction is ignored (the snake keeps going); a same "
            "direction, NOOP, missing action, malformed action, or stale state-tick request does "
            "not stop the clock. On a non-food move the current tail vacates simultaneously, so "
            "entering that exact tail cell is legal. On a food move the tail stays and the body "
            "grows; the next food is sampled uniformly from row-major empty cells using a private "
            "random.Random(seed). No obstacles, opponents, power-ups, acceleration, starvation "
            "timer, or implicit collision avoidance exists.\n\n"
            "action_spec names are UP, RIGHT, DOWN, LEFT, and NOOP. Integer aliases 0, 1, 2, 3 "
            "map to the four directions. For the optional low-level API, a dictionary such as "
            "{'action': 'RIGHT', 'state_tick': 12} carries a request derived from observation "
            "tick 12 and is rejected if it is stale; set_direction(direction, state_tick) queues "
            "the same kind of request for the next tick. The normal runner calls tick(action) at "
            "its own wall/virtual-clock cadence; tick_s is metadata for that cadence and tick() "
            "does not sleep. Each applied call increments ticks, records action_tick, and returns "
            "quickly. observe() only snapshots state and never advances time. Observations also "
            "include food, length, free_cells, max_score, score_fraction, nonreverse directions, "
            "one-step safe directions using exact tail-vacating semantics, terminal flags, last "
            "reward/event/error, attempted head, and timing fields. info() adds final score, "
            "length, ticks, terminal reason, completion, and rejected-action diagnostics."
        )

    def close(self) -> None:
        """Close this in-process instance; idempotent and resource-free."""
        with self._lock:
            self._closed = True
            self._pending_direction = None
            self._pending_state_tick = None


def make_game(**kwargs: Any) -> SnakeGame:
    """Construct a fresh local game; no credentials or external services are used."""
    return SnakeGame(**kwargs)
