"""Visible-window event compiler with a nonblocking protocol-repair supervisor.

Only public observations are consumed.  Models never generate note events or
change the chart's timing offset.  Their exceptional job is interpreting a new
protocol error, not recognizing lanes that are already explicitly decoded.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


def observation(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return int(round(value)) if math.isfinite(value) else None
    except (OverflowError, ValueError):
        return None


def probability(value):
    """Model output is untrusted, including an unexpected confidence type."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) and 0.0 <= value <= 1.0 else None
    except (OverflowError, ValueError):
        return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.sent = set()
        self.transport = "string"
        self.repair = None
        self.repair_version = None
        self.seen_errors = set()
        self.astra_used = False
        self.scheduled_parts = 0
        self.late_parts = 0

    def compile(self, obs):
        now = integer(obs.get("clock_ms"))
        lanes = integer(obs.get("lane_count", obs.get("lanes")))
        if now is None or lanes not in (4, 7):
            return [], []
        # All normal events are discovered roughly two seconds before the head.
        # A newly discovered overdue event is not backdated: try a future tick.
        tick_ms = max(1, int(round(float(obs.get("tick_s", 0.018)) * 1000)))
        miss_window = float(obs.get("judgement_window_ms", {}).get("Miss", 164))
        events, keys = [], []
        batch_seen = set()
        notes = obs.get("visible_notes", [])
        for note in notes if isinstance(notes, list) else []:
            if not isinstance(note, dict):
                continue
            lane = integer(note.get("lane"))
            start, end = integer(note.get("start_ms")), integer(note.get("end_ms", 0))
            if lane is None or not 1 <= lane <= lanes or start is None or end is None:
                continue
            state = note.get("state")
            identity = (str(note.get("id")), lane, start, end)
            parts = []
            if state == "pending":
                parts.append(("press", start, miss_window))
            if end > start and state in ("pending", "held"):
                parts.append(("release", end, miss_window * float(obs.get("release_window_multiplier", 1.5))))
            for kind, target, window in parts:
                key = identity + (kind,)
                if key in self.sent:
                    continue
                when = target
                if target < now + tick_ms:
                    when = now + tick_ms
                    if when - target > window:
                        continue
                    self.late_parts += 1
                event_key = (kind, lane, when)
                if event_key in batch_seen:
                    continue
                batch_seen.add(event_key)
                events.append({"type": kind, "lane": lane, "time_ms": when})
                keys.append(key)
        # Releasing an old hold precedes a new head at an identical timestamp.
        events.sort(key=lambda e: (e["time_ms"], e["type"] != "release", e["lane"]))
        return events, keys

    def launch_diagnostic(self, obs):
        """Only unexpected protocol prose goes to Jev; never per-note arithmetic."""
        error = obs.get("last_action_error")
        if not error or self.repair is not None:
            return
        text = str(error)[:1200]
        if text in self.seen_errors or self.ctx.budget.remaining()["system1_calls"] <= 1:
            return
        self.seen_errors.add(text)
        version = (obs.get("chart_hash"), self.transport, text)
        self.repair_version = version
        state = {
            "goal": "Keep exact timestamped presses and LN releases running; repair only the event transport.",
            "protocol": {
                "backend": "local_sim", "mapping": "virtual lanes, not physical keys",
                "legal_transports": ["events JSON-list string", "direct JSON event list"],
                "duplicates": "Duplicate or impossible events are ignored.",
                "late_events": "Past timestamps are not a valid way to recover a judgement.",
                "timing": "Visible start_ms/end_ms already include rate and offset; never change them.",
            },
            "plan": {"transport": self.transport, "policy": "compile visible notes, deduplicate, heartbeat each tick"},
            "error": text,
            "input_mapping": obs.get("input_mapping", {}),
            "timing_error_summary": obs.get("timing_error_summary", {}),
            "version": list(version),
        }
        criteria = {
            "replay": "The error only reports a duplicate, already-judged event, or no pending/held object. Keep compiling new notes.",
            "late": "The error specifically says an event timestamp is already past. Keep future scheduling, never backdate.",
            "encoding": "The error explicitly rejects parsing/encoding of the whole batch, not lane state or timing. Try the other documented transport.",
            "unknown": "Another error, contradictory evidence, or insufficient information. Preserve the executable policy and request a slow diagnosis.",
        }
        instructions = (
            "Classify `error` against `protocol`. Select encoding only for an explicit batch syntax/serialization failure; "
            "a duplicate, expired time, invalid lane, or absent held note is not an encoding failure. Do not calculate timing."
        )
        questions = {
            "error_kind": Choice(instructions, criteria),
            "error_check": Choice(instructions, dict(reversed(list(criteria.items())))),
        }
        def work():
            try:
                return self.ctx.decide(state, questions, timeout_s=1.5)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                return None
        self.repair = ("jev", self.ctx.spawn(work), state)
        self.ctx.log("protocol_diagnostic_started", version=list(version))

    def collect_diagnostic(self, obs):
        if self.repair is None or not self.repair[1].done():
            return
        kind, future, state = self.repair
        self.repair = None
        try:
            result = future.result()
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError):
            return
        current = (obs.get("chart_hash"), self.transport, str(obs.get("last_action_error"))[:1200])
        if current != self.repair_version:
            self.ctx.log("protocol_diagnostic_discarded", reason="evidence or transport changed")
            return
        if kind == "astra":
            text = str(getattr(result, "text", "")).strip().lower()
            # Astra can select a documented transport only, never author code or events.
            if text in ("string", "list"):
                self.change_transport(text)
            else:
                self.ctx.log("astra_diagnostic_ignored", reason="unrecognized response")
            return
        if not isinstance(result, dict):
            return
        votes = []
        for name in ("error_kind", "error_check"):
            answer = result.get(name)
            value = getattr(answer, "value", None)
            probabilities = getattr(answer, "probabilities", None)
            confidence = probability(getattr(answer, "confidence", None))
            if not isinstance(value, str) or value not in ("replay", "late", "encoding", "unknown"):
                return
            if not isinstance(probabilities, dict):
                return
            winning = probability(probabilities.get(value))
            if confidence is None or confidence < 0.6 or winning is None or winning < 0.85:
                return
            votes.append(value)
        if votes[0] != votes[1]:
            return
        self.ctx.log("protocol_diagnostic_accepted", diagnosis=votes[0], action="retain exact event timing")
        if votes[0] == "encoding":
            self.change_transport("list" if self.transport == "string" else "string")
        elif votes[0] == "unknown" and not self.astra_used and self.ctx.budget.remaining()["system2_calls"] > 1:
            self.astra_used = True
            prompt = (
                "Diagnose this offline Quaver event transport error. Both 'events '+JSON text and a direct JSON event list "
                "are documented actions. Never change timing, lanes, offsets or gameplay rules. Reply with exactly "
                "one token: string, list, or keep. Select a different transport only for a demonstrated serialization "
                "failure; for semantic errors or uncertainty select keep. Existing deterministic play continues.\n" +
                json.dumps(state, separators=(",", ":"))
            )
            def think():
                try:
                    return self.ctx.think(prompt, timeout_s=120.0)
                except (BudgetExceeded, ModelUnavailable):
                    raise
                except (TimeoutError, RuntimeError):
                    return None
            self.repair = ("astra", self.ctx.spawn(think), state)
            self.ctx.log("astra_protocol_diagnostic_started")

    def change_transport(self, transport):
        if transport != self.transport:
            old = self.transport
            self.transport = transport
            # If parsing failed, rebuild from currently visible pending/held notes.
            # Already-resolved parts disappear from compile() without being retried.
            self.sent.clear()
            self.ctx.log("event_transport_changed", old=old, new=transport)

    def run_episode(self, game):
        ctx = self.ctx
        ctx.log("scheduler_started", policy="visible_timestamp_compiler", model_policy="exceptions_only")
        while not ctx.done:
            obs = observation(ctx.observe())
            if obs.get("terminal"):
                break
            self.collect_diagnostic(obs)
            events, keys = self.compile(obs)
            if events:
                action = events if self.transport == "list" else "events " + json.dumps(events, separators=(",", ":"))
                ctx.act(action)
                self.sent.update(keys)
                self.scheduled_parts += len(events)
                ctx.log("visible_batch_posted", clock_ms=obs.get("clock_ms"), events=len(events),
                        first_ms=events[0]["time_ms"], last_ms=events[-1]["time_ms"])
            else:
                # A fresh heartbeat replaces the last batch after it has had a tick
                # to be consumed, preventing avoidable repeated lane events.
                ctx.act("noop")
            self.launch_diagnostic(obs)
            ctx.wait_for_tick()
        ctx.log("scheduler_finished", scheduled_parts=self.scheduled_parts, late_parts=self.late_parts)


def build(ctx):
    return Harness(ctx)
