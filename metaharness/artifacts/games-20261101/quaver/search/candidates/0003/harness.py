"""Refine 0000: the same synchronous Jev-only loop, with queued batch actions.

Jev approves a particular visible schedule. Code implements documented timestamp
and LN semantics; it does not discover hidden notes or play without approval.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


def observation(value):
    # The public interface can return JSON text rather than an already parsed dict.
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError, OverflowError):
        return default


def supported(obs):
    return (
        obs.get("interface") == "quaver-local-sim-v1"
        and obs.get("input_mapping", {}).get("kind") == "virtual_lane_events"
        and obs.get("timing", {}).get("scroll", {}).get("velocity_model")
        == "timestamp_authoritative"
        and obs.get("lane_count", obs.get("lanes")) in (4, 7)
    )


def identity(obs):
    return (obs.get("interface"), obs.get("chart_hash"),
            obs.get("lane_count", obs.get("lanes")))


def event_key(note, part):
    return (note["id"], part, note["lane"], note["start_ms"], note["end_ms"])


def compile_visible(obs, submitted):
    """Expand only observed, unfinished objects into legal timestamped events.

    The returned keys refer to immutable note identities, not to transient clocks.
    A pending LN's observed tail may lie beyond the head's lookahead window.
    """
    if not supported(obs):
        return []
    now = int(number(obs.get("clock_ms")))
    tick_ms = max(1, int(round(number(obs.get("tick_s"), 0.018) * 1000)))
    earliest = now + tick_ms
    lane_count = int(obs.get("lane_count", obs.get("lanes")))
    windows = obs.get("judgement_window_ms", {})
    press_window = number(windows.get("Miss"), 164.0)
    release_window = press_window * number(obs.get("release_window_multiplier"), 1.5)
    result = []
    seen = set()
    for raw in obs.get("visible_notes", []):
        if not isinstance(raw, dict) or raw.get("state") not in ("pending", "held"):
            continue
        try:
            note = {k: int(raw.get(k, 0)) for k in ("id", "lane", "start_ms", "end_ms")}
        except (ValueError, TypeError, OverflowError):
            continue
        if not 1 <= note["lane"] <= lane_count:
            continue
        head = event_key(note, "press")
        head_available = head in submitted or raw["state"] == "held"
        if raw["state"] == "pending" and head not in submitted:
            when = max(note["start_ms"], earliest)
            if when <= note["start_ms"] + press_window:
                if head not in seen:
                    result.append((head, {"type": "press", "lane": note["lane"],
                                          "time_ms": when}))
                    seen.add(head)
                head_available = True
        # Taps do not need artificial releases. An LN release is linked to its head.
        tail = event_key(note, "release")
        if note["end_ms"] > note["start_ms"] and head_available and tail not in submitted:
            when = max(note["end_ms"], earliest)
            if when <= note["end_ms"] + release_window and tail not in seen:
                result.append((tail, {"type": "release", "lane": note["lane"],
                                      "time_ms": when}))
                seen.add(tail)
    # Resolve a tail before a new same-lane head when their timestamps coincide.
    result.sort(key=lambda item: (item[1]["time_ms"],
                                 0 if item[1]["type"] == "release" else 1,
                                 item[1]["lane"]))
    return result


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        submitted = set()
        revision = 0
        questions = {"act": Choice(
            instructions=(
                "Quaver local simulator: maximize timing accuracy and complete the chart. "
                "Choose whether to submit `schedule`, a code-generated timestamped event batch. "
                "This is a LOOKAHEAD QUEUE decision, not a physical press right now. "
                "Submitting future timestamps now does NOT hit early: the game stores them "
                "and executes each at its target time, including simultaneous chords and LN tails. "
                "Code has already checked lanes, note state and duplicate submissions. "
                "Use `schedule.status`: ready means new observed events can be queued; "
                "empty means all currently observed events are already covered or finished; "
                "unsupported means the timestamp interface could not be verified. "
                "Which action should be posted? Do not wait merely because notes are in the future."
            ),
            criteria={
                "queue_visible": (
                    "Submit the ready schedule now to hit its observed notes at their timestamps. "
                    "Preserve simultaneous presses and complete long-note holds/releases."
                ),
                "wait": "Post noop when the schedule is empty; existing queued events still execute.",
                "unknown": "The interface is unsupported or the stated queue semantics are contradicted.",
            },
        )}
        while not ctx.done:
            obs = observation(ctx.observe())
            if obs.get("terminal"):
                return
            if not supported(obs) or getattr(ctx, "system1", True) is None:
                ctx.act("noop")
                ctx.wait_for_tick()
                continue
            revision += 1
            proposed = compile_visible(obs, submitted)
            state = {
                # Preserve the parent's full observation rather than introduce a new recognizer.
                "observation": obs,
                "schedule": {
                    "revision": revision,
                    "observed_clock_ms": obs.get("clock_ms"),
                    "status": "ready" if proposed else "empty",
                    "events": [event for _, event in proposed],
                    "semantics": (
                        "These are new, not-yet-submitted events from visible notes only. "
                        "Times already include the chart rate and offset. An LN head and its "
                        "known tail are queued together; a held LN needs only its tail. "
                        "After approval code rechecks the same note identities on a fresh "
                        "observation, discards resolved events, and moves late salvageable "
                        "events to the next tick rather than backdating them."
                    ),
                },
            }
            try:
                answers = ctx.decide(state, questions, timeout_s=5.0)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # The one-shot bridge leaves noop in the last-action register.
                # Already queued events continue while this synchronous call retries.
                ctx.log("batch_model_retry", revision=revision)
                ctx.wait_for_tick()
                continue
            if ctx.done:
                return
            answer = answers.get("act") if isinstance(answers, dict) else None
            choice = getattr(answer, "value", None)
            probabilities = getattr(answer, "probabilities", {})
            probability = number(probabilities.get("queue_visible"), 0.0) if isinstance(probabilities, dict) else 0.0
            # Begin the one-shot handoff immediately after a tick boundary, leaving
            # a full tick for act() -> wait_for_tick() before we clear the register.
            ctx.wait_for_tick()
            if ctx.done:
                return
            latest = observation(ctx.observe())
            approved = []
            if (choice == "queue_visible" and probability >= 0.60
                    and identity(latest) == identity(obs)):
                # Approval authorizes only this snapshot, never new notes revealed during inference.
                authorized = {key for key, _ in proposed}
                approved = [(key, event) for key, event in compile_visible(latest, submitted)
                            if key in authorized]
            if latest.get("terminal") or ctx.done:
                return
            action = "events " + json.dumps([e for _, e in approved], separators=(",", ":")) if approved else "noop"
            ctx.act(action)
            # Timestamped batches are NOT idempotent while still in the future:
            # replayed batches enqueue duplicate events that can hit later notes.
            # Deliver for one tick, then leave noop during the next Jev request.
            ctx.wait_for_tick()
            if ctx.done:
                return
            if approved:
                ctx.act("noop")
            submitted.update(key for key, _ in approved)
            ctx.log("jev_batch_action", revision=revision,
                    choice=choice if isinstance(choice, str) else "invalid",
                    queue_probability=probability,
                    confidence=getattr(answer, "confidence", None),
                    source_clock_ms=obs.get("clock_ms"),
                    action_clock_ms=latest.get("clock_ms"),
                    proposed_events=len(proposed), posted_events=len(approved),
                    last_action_error=latest.get("last_action_error"))


def build(ctx):
    return Harness(ctx)
