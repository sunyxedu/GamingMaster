"""Jev-controlled osu!mania scheduler.

The parent asked Jev to choose an abstract command (``tap <lane>``), which
cannot contain a lane or a timestamp.  This variant keeps the same synchronous
Jev loop, but makes the only meaningful choice a *precomputed scheduling
policy*.  Once the policy is selected, the bridge turns the newest structured
observation into legal future event batches.  Exact note arithmetic stays in
this module; Jev never has to generate command text.
"""
from __future__ import annotations

import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


# The visible horizon is 2000 ms in this adaptor.  This fallback is deliberately
# below the smallest miss windows seen in the supplied training observations.
_DEFAULT_HORIZON_MS = 2000.0
_DEFAULT_MISS_MS = 188.0


def _number(value: Any, default: float | None = None) -> float | None:
    """Return a finite float, without allowing bad model/game data into state."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _event_time(value: Any) -> float | None:
    return _number(value)


def _format_time(value: float) -> str:
    # The simulator accepts chart milliseconds as decimal numbers.  Keep chart
    # precision while avoiding scientific notation in the action grammar.
    if abs(value - round(value)) < 1e-7:
        return str(int(round(value)))
    return (f"{value:.6f}").rstrip("0").rstrip(".")


def _unique_notes(observation: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten lane-local upcoming notes and deduplicate chord copies."""
    by_id: dict[str, dict[str, Any]] = {}
    for lane_record in observation.get("lanes", []) or []:
        lane_record = _as_dict(lane_record)
        lane_fallback = lane_record.get("lane")
        for raw in lane_record.get("upcoming", []) or []:
            note = _as_dict(raw)
            lane = note.get("lane", lane_fallback)
            start = _event_time(note.get("start_ms"))
            if lane is None or start is None:
                continue
            try:
                lane_i = int(lane)
            except (TypeError, ValueError):
                continue
            note_id = note.get("id")
            # IDs are supplied by the adaptor.  The fallback key only protects
            # the pre-flight stand-in from a malformed observation.
            key = str(note_id) if note_id is not None else f"{lane_i}:{start}:{note.get('kind', 'tap')}"
            normalized = {
                "id": key,
                "lane": lane_i,
                "kind": "hold" if note.get("kind") == "hold" else "tap",
                "start_ms": start,
                "end_ms": _event_time(note.get("end_ms")),
                "time_to_start_ms": _number(note.get("time_to_start_ms")),
                "chord_lanes": list(note.get("chord_lanes", []) or []),
                "is_chord": bool(note.get("is_chord", False)),
                "pattern": note.get("pattern"),
            }
            # A chord is repeated once per lane.  Preserve the first complete
            # record; lane and start are also retained in the key fallback.
            by_id.setdefault(key, normalized)
    notes = list(by_id.values())
    notes.sort(key=lambda n: (n["start_ms"], n["lane"], n["id"]))
    return notes


def _compact_observation(observation: dict[str, Any], scheduled: int) -> dict[str, Any]:
    """State sent to Jev: relevant facts and no inaccessible game object."""
    notes = _unique_notes(observation)
    compact_notes = []
    for note in notes:
        compact_notes.append({
            "id": note["id"],
            "lane": note["lane"],
            "kind": note["kind"],
            "start_ms": note["start_ms"],
            "end_ms": note["end_ms"],
            "time_to_start_ms": note["time_to_start_ms"],
            "is_chord": note["is_chord"],
            "chord_lanes": note["chord_lanes"],
            "pattern": note["pattern"],
        })
    return {
        "ruleset": observation.get("ruleset", "stable-scorev1-native-mania"),
        "current_time_ms": _number(observation.get("current_time_ms"), 0.0),
        "planning_horizon_ms": _number(
            observation.get("planning_horizon_ms",
                           _as_dict(observation.get("timers")).get("planning_horizon_ms")),
            _DEFAULT_HORIZON_MS,
        ),
        "key_count": observation.get("key_count"),
        "judgement_windows_ms": observation.get("judgement_windows_ms", {}),
        "visible_notes": compact_notes,
        "visible_patterns": [
            {
                "start_ms": p.get("start_ms"),
                "time_to_start_ms": p.get("time_to_start_ms"),
                "lanes": list(p.get("lanes", []) or []),
                "pattern": p.get("pattern"),
                "is_chord": bool(p.get("is_chord", False)),
            }
            for p in (observation.get("visible_patterns", []) or [])[:24]
            if isinstance(p, dict)
        ],
        "active_holds": [
            {
                "id": h.get("id"),
                "lane": h.get("lane"),
                "release_due": bool(h.get("release_due", False)),
                "hold_break_risk": h.get("hold_break_risk"),
                "release_window_ms": h.get("release_window_ms"),
            }
            for h in (observation.get("active_holds", []) or [])[:16]
            if isinstance(h, dict)
        ],
        "recent_results": list(observation.get("recent_results", []) or [])[-8:],
        "progress": {
            "score": observation.get("score"),
            "accuracy": observation.get("accuracy"),
            "combo": observation.get("combo"),
            "miss_rate": observation.get("miss_rate"),
            "hold_break_rate": observation.get("hold_break_rate"),
        },
        "scheduler": {
            "already_scheduled_objects": scheduled,
            "action_rule": "A selected play policy causes code to queue exact transitions for visible notes; Jev does not invent timestamps.",
        },
    }


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.head_scheduled: set[str] = set()
        self.tail_scheduled: set[str] = set()
        self.posted_event_keys: set[tuple[float, int, str]] = set()
        self.last_policy = "execute_window"

    @staticmethod
    def _release_window(observation: dict[str, Any]) -> float:
        windows = observation.get("judgement_windows_ms")
        if isinstance(windows, dict):
            value = _number(windows.get("MISS"))
            if value is not None:
                return max(1.0, value)
        return _DEFAULT_MISS_MS

    @staticmethod
    def _horizon(observation: dict[str, Any]) -> float:
        timers = _as_dict(observation.get("timers"))
        value = _number(observation.get("planning_horizon_ms"))
        if value is None:
            value = _number(timers.get("planning_horizon_ms"))
        return max(16.0, value if value is not None else _DEFAULT_HORIZON_MS)

    @staticmethod
    def _now(observation: dict[str, Any]) -> float:
        value = _number(observation.get("current_time_ms"))
        return value if value is not None else 0.0

    def _active_hold_notes(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        """Recover a tail if the adaptor has moved a hold to active_holds."""
        result: list[dict[str, Any]] = []
        for raw in observation.get("active_holds", []) or []:
            hold = _as_dict(raw)
            lane = hold.get("lane")
            if lane is None:
                continue
            try:
                lane_i = int(lane)
            except (TypeError, ValueError):
                continue
            tail = None
            for key in ("end_ms", "tail_ms", "tail_time_ms", "release_time_ms"):
                tail = _event_time(hold.get(key))
                if tail is not None:
                    break
            if tail is None:
                window = _as_dict(hold.get("release_window_ms"))
                tail = _event_time(window.get("centre_ms"))
                if tail is None:
                    earliest = _event_time(window.get("earliest_ms"))
                    latest = _event_time(window.get("latest_ms"))
                    if earliest is not None and latest is not None:
                        tail = (earliest + latest) / 2.0
            if tail is None:
                continue
            ident = hold.get("id")
            if ident is None:
                ident = f"active:{lane_i}:{tail}"
            result.append({
                "id": str(ident),
                "lane": lane_i,
                "kind": "hold",
                "start_ms": self._now(observation),
                "end_ms": tail,
                "time_to_start_ms": 0.0,
                "is_chord": False,
                "chord_lanes": [lane_i],
                "pattern": "hold",
            })
        return result

    def _has_scheduleable(self, observation: dict[str, Any]) -> bool:
        """Whether the newest observation contains an unscheduled legal event."""
        notes = _unique_notes(observation)
        notes.extend(self._active_hold_notes(observation))
        now = self._now(observation)
        upper = now + self._horizon(observation)
        miss = self._release_window(observation)
        for note in notes:
            ident = str(note["id"])
            start = _event_time(note.get("start_ms"))
            if start is None or not (now - miss <= start <= upper):
                continue
            if ident not in self.head_scheduled:
                return True
            if note.get("kind") == "hold":
                end = _event_time(note.get("end_ms"))
                if end is not None and now - miss <= end <= upper and ident not in self.tail_scheduled:
                    return True
        return False

    def _events_for_latest(self, observation: dict[str, Any], policy: str) -> str:
        """Create a legal batch from the post-answer observation.

        The observation is refreshed after Jev returns.  This is important in a
        realtime game: event timestamps are checked against the *current* clock,
        not the clock at which the model request began.
        """
        if policy == "keep_existing":
            return "noop"

        notes = _unique_notes(observation)
        notes.extend(self._active_hold_notes(observation))
        # Do not duplicate an active copy of an upcoming note.
        merged: dict[str, dict[str, Any]] = {}
        for note in notes:
            merged.setdefault(str(note["id"]), note)
        notes = list(merged.values())
        notes.sort(key=lambda n: (n["start_ms"], n["lane"], n["id"]))

        now = self._now(observation)
        horizon = self._horizon(observation)
        miss = self._release_window(observation)
        upper = now + horizon
        events: list[tuple[float, int, int, str, str, str]] = []
        # event tuple: time, lane, ordering, transition, object id, phase
        for note in notes:
            ident = str(note["id"])
            lane = int(note["lane"])
            start = _event_time(note.get("start_ms"))
            if start is None:
                continue
            # A just-passed head can still be submitted at the current time and
            # judged late.  Older heads are already irrecoverable and are not
            # allowed to pollute a valid batch.
            head_eligible = start >= now - miss and start <= upper
            head_new = ident not in self.head_scheduled
            if head_new and head_eligible:
                events.append((start, lane, 0, "down", ident, "head"))

            if note.get("kind") == "hold":
                end = _event_time(note.get("end_ms"))
                # The release may be scheduled in a later window after the head
                # was queued.  It is safe only if its head is in this action or
                # was posted previously.
                head_known = ident in self.head_scheduled or (head_new and head_eligible)
                tail_eligible = end is not None and end >= now - miss and end <= upper
                if ident not in self.tail_scheduled and head_known and tail_eligible:
                    events.append((end, lane, 1, "up", ident, "tail"))
            else:
                # A same-time up makes a tap a real down transition without
                # leaving the key held for a later jack.
                if head_new and head_eligible:
                    events.append((start, lane, 1, "up", ident, "tail"))

        if not events:
            return "noop"

        # The game requires nondecreasing timestamps.  Deduping also protects
        # against a repeated stale mailbox action or duplicate chord records.
        events.sort(key=lambda e: (e[0], e[1], e[2], e[3]))
        tokens: list[str] = ["events"]
        emitted: list[tuple[float, int, str, str, str]] = []
        for timestamp, lane, _ordering, transition, ident, phase in events:
            key = (round(timestamp, 6), lane, transition)
            if key in self.posted_event_keys:
                continue
            tokens.extend((_format_time(timestamp), str(lane), transition))
            emitted.append((timestamp, lane, transition, ident, phase))
        if not emitted:
            return "noop"

        for timestamp, lane, transition, ident, phase in emitted:
            self.posted_event_keys.add((round(timestamp, 6), lane, transition))
            if phase == "head":
                self.head_scheduled.add(ident)
            else:
                self.tail_scheduled.add(ident)
        return " ".join(tokens)

    def _question(self, observation: dict[str, Any]) -> tuple[dict[str, Any], Choice]:
        state = _compact_observation(observation, len(self.head_scheduled))
        count = len(state["visible_notes"])
        return state, Choice(
            """Choose the scheduling policy for this osu!mania observation.  The
            action bridge, not you, supplies legal lane numbers and chart
            timestamps.  If visible_notes is nonempty, prefer the executable
            policy that queues every not-yet-scheduled visible tap and hold,
            preserving simultaneous chords and releasing holds at their tails.
            Do not output a command or invent a time; select exactly one named
            policy.  keep_existing is appropriate only when the existing future
            queue already covers the visible section.""",
            {
                "execute_window": (
                    f"Queue exact down/up transitions for all {count} visible note objects not already scheduled. "
                    "This includes all lanes of chords, tap releases, and hold tails."
                ),
                "keep_existing": (
                    "Submit no new transitions; rely only on events already queued. "
                    "Use this only if no newly visible object needs scheduling."
                ),
            },
        )

    def _post_fallback(self, game: Any) -> None:
        latest = _as_dict(game.observe())
        policy = self.last_policy
        if policy == "keep_existing" and self._has_scheduleable(latest):
            policy = "execute_window"
        action = self._events_for_latest(latest, policy)
        self.ctx.act(action)

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        if getattr(ctx, "system1", True) is None:
            return

        while not ctx.done and not getattr(game, "done", False):
            observation = _as_dict(game.observe())
            state, question = self._question(observation)
            try:
                answers = ctx.decide(state, {"schedule": question}, timeout_s=5.0)
                answer = answers.get("schedule") if isinstance(answers, dict) else None
                value = getattr(answer, "value", None)
                probabilities = getattr(answer, "probabilities", {}) or {}
                keep_probability = _number(probabilities.get("keep_existing"), 0.0) if isinstance(probabilities, dict) else 0.0
                # Garbage/stand-in answers cannot crash the harness.  The
                # executable default is the only safe recovery for an empty
                # queue; a deliberate keep decision is honoured only after a
                # queue has already been posted and Jev assigns it a 0.60+
                # top-option probability.
                if value == "keep_existing" and self.head_scheduled and keep_probability >= 0.60:
                    self.last_policy = "keep_existing"
                else:
                    self.last_policy = "execute_window"
                latest = _as_dict(game.observe())
                # A keep answer is a valid optimization only when the newest
                # clock still has no unscheduled event.  This recheck prevents
                # a semantic keep decision made on the older request state from
                # starving a note that entered the horizon during inference.
                effective_policy = self.last_policy
                if effective_policy == "keep_existing" and self._has_scheduleable(latest):
                    effective_policy = "execute_window"
                action = self._events_for_latest(latest, effective_policy)
                ctx.act(action)
                ctx.log(
                    "jev_schedule",
                    selected=value if isinstance(value, str) else "invalid_default_execute",
                    action_kind="events" if action.startswith("events ") else "noop",
                    visible=len(_unique_notes(latest)),
                    scheduled_heads=len(self.head_scheduled),
                    scheduled_tails=len(self.tail_scheduled),
                )
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # Keep the loop alive through a transient/late Jev call.  The
                # previous plan is executable and the newest observation avoids
                # submitting timestamps that went stale during inference.
                if ctx.done:
                    break
                self._post_fallback(game)


def build(ctx: Any) -> Harness:
    return Harness(ctx)
