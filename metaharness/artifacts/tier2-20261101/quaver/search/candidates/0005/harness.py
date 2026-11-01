"""Timestamped chart scheduler with a small Jev advisory and an idle fallback.

The simulator exposes exact target times for the currently fair-visible chart.  The
critical path therefore belongs to this module, not to a language model: it queues
presses and LN releases ahead of the virtual clock and emits a harmless action on
every tick while those events are pending.  Jev is used once as a typed check on
how far the visible schedule should be trusted; a failed/ambiguous check simply
falls back to the conservative scheduler (or to noop), never to a guessed lane.
"""
from __future__ import annotations

import json
import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


class Harness:
    """Run one episode using fair-window observations as a rolling event queue."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._presses: set[tuple[str, str]] = set()
        self._releases: set[tuple[str, str]] = set()
        self._advice_done = False
        # The conservative value is still comfortably longer than model latency
        # and is refreshed every tick.  Jev may expand it to the whole fair view.
        self._horizon_ms = 1800

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            result = float(value)
            if math.isfinite(result):
                return result
        return None

    @classmethod
    def _time(cls, value: Any) -> int | None:
        number = cls._number(value)
        if number is None:
            return None
        return int(round(number))

    @staticmethod
    def _note_key(note: dict[str, Any], lane: int, start: int, end: int) -> str:
        """Make a stable key even for a deliberately minimal stand-in observation."""
        ident = note.get("id")
        if isinstance(ident, (str, int, float)) and not isinstance(ident, bool):
            return str(ident)
        return f"{lane}:{start}:{end}"

    def _compact_state(self, obs: Any, now: int) -> dict[str, Any]:
        """Send Jev facts, rather than the large raw observation."""
        if not isinstance(obs, dict):
            return {
                "clock_ms": now,
                "visible_pending": 0,
                "visible_held": 0,
                "nearest_target": "unknown",
                "patterns": [],
                "scheduler_horizon": self._horizon_ms,
            }

        pending = 0
        held = 0
        starts: list[int] = []
        patterns: list[str] = []
        for raw in obs.get("visible_notes", []):
            if not isinstance(raw, dict):
                continue
            state = raw.get("state")
            if state == "pending":
                pending += 1
            elif state == "held":
                held += 1
            start = self._time(raw.get("start_ms"))
            if start is not None and state in ("pending", "held"):
                starts.append(start)
            pattern = raw.get("pattern")
            if isinstance(pattern, str) and pattern not in patterns:
                patterns.append(pattern)
        starts.sort()
        delta = starts[0] - now if starts else None
        if delta is None:
            bucket = "unknown"
        elif delta <= 180:
            bucket = "imminent"
        elif delta <= 700:
            bucket = "near"
        else:
            bucket = "far"
        return {
            "clock_ms": now,
            "visible_pending": pending,
            "visible_held": held,
            "nearest_target": bucket,
            "patterns": patterns[:6],
            "scheduler_horizon": self._horizon_ms,
            "timestamp_targets_are_explicit": True,
        }

    @staticmethod
    def _advice_call(ctx, state: dict[str, Any], question: Choice):
        """Run Jev off the game thread; all failures become advisory absence."""
        try:
            return ctx.decide(state, {"schedule": question}, timeout_s=1.0)
        except Exception:
            return None

    def _apply_schedule_advice(self, answers: Any) -> None:
        answer = answers.get("schedule") if isinstance(answers, dict) else None
        value = getattr(answer, "value", None)
        probabilities = getattr(answer, "probabilities", {})
        selected_probability = 0.0
        if isinstance(probabilities, dict):
            selected_probability = self._number(probabilities.get(value)) or 0.0
        # Either response is safe: the short value is refreshed each tick.
        # Require a little evidence before allowing the broader queue.
        if value == "wide" and selected_probability >= 0.45:
            self._horizon_ms = 2000
        elif value == "near_term":
            self._horizon_ms = 650
        else:
            self._horizon_ms = 1800
        try:
            self.ctx.log(
                "jev_schedule_advice",
                value=value if isinstance(value, str) else "fallback",
                probability=selected_probability,
                horizon_ms=self._horizon_ms,
            )
        except Exception:
            pass

    def _ask_schedule_advice(self, obs: Any, now: int) -> None:
        """Launch/poll one typed judgment without blocking the note path."""
        ctx = self.ctx
        future = getattr(self, "_advice_future", None)
        if future is not None:
            try:
                if future.done():
                    self._advice_future = None
                    self._apply_schedule_advice(future.result())
            except Exception:
                self._advice_future = None
                self._horizon_ms = 1800
            return
        if self._advice_done or getattr(ctx, "system1", None) is None:
            return
        self._advice_done = True
        question = Choice(
            instructions=(
                "The observation facts describe a local deterministic rhythm simulator. "
                "Choose wide when the explicit target timestamps and pending/held states "
                "are sufficient for code to queue every visible press and long-note tail; "
                "choose near_term only when a conservative short rolling queue is safer. "
                "Do not invent a lane or a timestamp."
            ),
            criteria={
                "wide": (
                    "Use the entire visible timestamp schedule (up to about 2 seconds); "
                    "the code can validate lane and note state itself."
                ),
                "near_term": (
                    "Use only the next short slice and rescan frequently; this is a "
                    "fallback for ambiguous or incomplete visible state."
                ),
            },
        )
        state = self._compact_state(obs, now)
        try:
            self._advice_future = ctx.spawn(
                lambda: self._advice_call(ctx, state, question)
            )
        except Exception:
            # A missing thread facility is equivalent to an absent advisory.
            self._advice_future = None
            self._horizon_ms = 1800

    def _events_for(self, obs: Any, now: int) -> tuple[list[dict[str, Any]], set[tuple[str, str]], set[tuple[str, str]]]:
        """Derive new head/tail events from only the current fair-visible notes."""
        if not isinstance(obs, dict):
            return [], set(), set()
        lane_count = self._time(obs.get("lane_count", obs.get("lanes")))
        if lane_count is None or lane_count not in (4, 7):
            lane_count = 7
        horizon = now + int(self._horizon_ms)
        events: list[tuple[int, int, int, dict[str, Any], tuple[str, str]]] = []
        new_presses: set[tuple[str, str]] = set()
        new_releases: set[tuple[str, str]] = set()
        visible = obs.get("visible_notes", [])
        if not isinstance(visible, list):
            return [], new_presses, new_releases

        for raw in visible:
            if not isinstance(raw, dict):
                continue
            lane_value = self._time(raw.get("lane"))
            start = self._time(raw.get("start_ms"))
            end = self._time(raw.get("end_ms", 0))
            if lane_value is None or start is None:
                continue
            lane = int(lane_value)
            if lane < 1 or lane > lane_count:
                continue
            if end is None:
                end = 0
            long_note = end > start
            state = raw.get("state")
            note_key = self._note_key(raw, lane, start, end)

            # A target just outside the past edge can still be rescued by a
            # current-time event.  Normally all of these are queued well ahead.
            press_time = start if start >= now else now
            if state == "pending" and start <= horizon:
                key = (note_key, "press")
                if key not in self._presses:
                    event = {"type": "press", "lane": lane, "time_ms": press_time}
                    # 1 is after releases at an equal timestamp.
                    events.append((press_time, 1, lane, event, key))
                    new_presses.add(key)

            if long_note and state in ("pending", "held") and end <= horizon:
                release_time = end if end >= now else now
                key = (note_key, "release")
                if key not in self._releases:
                    event = {"type": "release", "lane": lane, "time_ms": release_time}
                    # Release first when an LN tail and a new head coincide.
                    events.append((release_time, 0, lane, event, key))
                    new_releases.add(key)

        events.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in events], new_presses, new_releases

    def _post(self, action: str) -> bool:
        try:
            self.ctx.act(action)
            return True
        except (TimeoutError, RuntimeError, TypeError, ValueError):
            # A realtime adaptor treats malformed/duplicate events as inert, but
            # keeping this guard makes pre-flight stand-ins harmless too.
            return False

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if not getattr(game, "tick_s", None):
            return

        noop = getattr(getattr(game, "action_spec", None), "noop", None) or "noop"
        while not ctx.done and not getattr(game, "done", False):
            obs = game.observe()
            now = self._time(obs.get("clock_ms", 0)) if isinstance(obs, dict) else 0
            now = 0 if now is None else now

            events, presses, releases = self._events_for(obs, now)
            if events:
                # JSON is the simulator's timestamped lookahead protocol.  It
                # remains a single game action even for a chord or LN pair.
                action = "events " + json.dumps(events, separators=(",", ":"))
                if self._post(action):
                    self._presses.update(presses)
                    self._releases.update(releases)
            else:
                # This deliberately keeps a fresh harmless action on each tick.
                # It also replaces the previous event batch after that batch has
                # had one tick in which to enter the simulator's future queue.
                self._post(noop)

            ctx.wait_for_tick()
            if not self._advice_done:
                # Do not let the advisory call repeat a timestamp batch while it
                # is thinking.  The first wait above has armed the queue; noop is
                # now the safe repeated action.
                self._post(noop)
            self._ask_schedule_advice(obs, now)


def build(ctx):
    return Harness(ctx)
