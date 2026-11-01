"""Timestamped chart executor with a small Jev-controlled dispatch policy.

The emulator exposes the actual upcoming chart, so note timing and hold
bookkeeping are deterministic code responsibilities rather than model tasks.
Jev only chooses how far ahead to commit; an invalid/late answer falls back to
incremental dispatch, while a noop remains the safe action between batches.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


FULL_HORIZON = "full_horizon"
NEXT_GROUP = "next_group"


def _object(value):
    """Normalize the adaptor observation without assuming every field exists."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _confidence(answer):
    try:
        value = float(getattr(answer, "confidence", None))
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _dispatch_policy(ctx, obs):
    """Ask one compact, typed question; never make play depend on prose."""
    episode = obs.get("episode", {})
    chart = obs.get("chart", {})
    upcoming = obs.get("upcoming", [])
    holds = obs.get("active_holds", [])
    controls = obs.get("controls", {})
    judgment = obs.get("last_judgments", [])
    horizon = []
    if isinstance(upcoming, list):
        for note in upcoming[:16]:
            if not isinstance(note, dict):
                continue
            horizon.append({
                "id": note.get("id"),
                "column": note.get("column"),
                "start_ms": note.get("start_ms"),
                "end_ms": note.get("end_ms"),
                "kind": note.get("kind"),
                "pattern": note.get("pattern"),
                "group_id": note.get("group_id"),
                "group_size": note.get("group_size"),
                "group_columns": note.get("group_columns"),
            })
    state = {
        "task": "Dispatch known Malody Key chart events using exact timestamps.",
        "observation": {
            "time_ms": episode.get("time_ms"),
            "tick_s": episode.get("tick_s"),
            "columns": episode.get("columns"),
            "judge": episode.get("judge"),
            "windows_ms": episode.get("judgment_windows_ms"),
            "mod": chart.get("mod"),
            "offset_ms": chart.get("offset_ms"),
            "bpm_segments": chart.get("bpm_segments"),
            "column_mapping": controls.get("column_mapping"),
            "upcoming": horizon,
            "active_holds": holds[:8] if isinstance(holds, list) else [],
            "recent_judgments": judgment[-6:] if isinstance(judgment, list) else [],
        },
        "execution_constraints": {
            "events_are_scheduled_at_their_absolute_start_ms": True,
            "equal_start_group_columns_must_be_pressed_together": True,
            "hold_release_ms_must_not_precede_end_ms": True,
            "fallback_between_event_batches": "noop",
        },
    }
    question = Choice(
        "Choose a dispatch horizon for this deterministic chart. Select full_horizon "
        "to schedule every visible upcoming head now; hold tails are released later "
        "from active_holds. Select next_group to schedule only the earliest equal-start "
        "group, then refresh the observation before committing later groups. This is a "
        "scheduling commitment choice, not a request to name individual key presses.",
        {
            FULL_HORIZON: {
                "what": "Schedule all notes already visible in the 2.2-second chart horizon.",
                "not_for": "A chart field that is malformed or contradictory.",
                "examples": "Trust valid start_ms and end_ms; retain simultaneous chord timestamps.",
            },
            NEXT_GROUP: {
                "what": "Schedule only the nearest simultaneous start-time group.",
                "not_for": "Choosing separate presses for notes in that group.",
                "examples": "Keep a chord atomic, then refresh the state for later groups.",
            },
        },
    )
    if getattr(ctx, "system1", True) is None:
        return NEXT_GROUP
    try:
        answers = ctx.decide(state, {"dispatch": question}, timeout_s=1.0)
        answer = answers.get("dispatch") if isinstance(answers, dict) else None
    except (TimeoutError, RuntimeError):
        return NEXT_GROUP
    except (BudgetExceeded, EpisodeOver, ModelUnavailable):
        return NEXT_GROUP
    if getattr(answer, "value", None) == FULL_HORIZON and _confidence(answer) >= 0.60:
        return FULL_HORIZON
    # Incremental dispatch is the conservative deterministic default for bad or
    # ambiguous typed output; arbitrary text is never parsed into an action.
    return NEXT_GROUP


def _events_from_observation(obs, scheduled, policy):
    """Commit the next safe timestamped group, including discovered hold tails.

    A future batch advances the emulator through its last timestamp. Therefore a
    batch must never end after an unobserved note. Heads are taken from
    ``upcoming`` and hold tails from ``active_holds``; selecting the earliest
    event/group preserves that invariant. Hold releases get a tiny positive lead
    over the reported end to avoid floating-point ``released_before_end`` ties.
    """
    episode = obs.get("episode", {})
    upcoming = obs.get("upcoming", [])
    active_holds = obs.get("active_holds", [])
    if not isinstance(upcoming, list):
        upcoming = []
    if not isinstance(active_holds, list):
        active_holds = []
    try:
        columns = int(episode.get("columns"))
    except (TypeError, ValueError, OverflowError):
        columns = 0
    now = _number(episode.get("time_ms"), 0.0)
    # The adaptor exposes a 2.2-second upcoming horizon. Do not post a hold
    # tail beyond that frontier: an unseen head can lie just before the tail.
    visible_until = now + 2200.0
    candidates = []
    for note in upcoming:
        if not isinstance(note, dict) or note.get("id") is None:
            continue
        note_id = str(note["id"])
        start = _number(note.get("start_ms"))
        try:
            lane = int(note.get("column"))
        except (TypeError, ValueError, OverflowError):
            continue
        if start is None or lane < 0 or lane >= columns:
            continue
        key = (note_id, "head")
        if key not in scheduled:
            candidates.append({"type": "press", "column": lane,
                               "at_ms": start, "_key": key})

    for hold in active_holds:
        if not isinstance(hold, dict) or hold.get("id") is None:
            continue
        end = _number(hold.get("end_ms"))
        try:
            lane = int(hold.get("column"))
        except (TypeError, ValueError, OverflowError):
            continue
        if end is None or lane < 0 or lane >= columns or end + 1.0 > visible_until:
            continue
        key = (str(hold["id"]), "tail")
        if key not in scheduled:
            candidates.append({"type": "release", "column": lane,
                               # 1 ms is inside neither a timing head window
                               # nor the reported tail instant, and avoids an
                               # equality/rounding ambiguity in the adaptor.
                               "at_ms": end + 1.0, "_key": key})

    if not candidates:
        return []

    incremental = policy == NEXT_GROUP
    # Holds require incremental dispatch even if Jev selected the broad policy:
    # a tail can lie beyond the visible horizon and would otherwise skip unseen
    # heads as the batch is processed.
    if any(isinstance(n, dict) and _number(n.get("end_ms")) is not None
           for n in upcoming) or active_holds:
        incremental = True

    if incremental:
        first = min(e["at_ms"] for e in candidates)
        # All heads at the earliest start are one chord. A release at that exact
        # dispatch instant is also safe to include.
        head_group = [e for e in candidates
                      if e["type"] == "press" and abs(e["at_ms"] - first) <= 0.01]
        if head_group:
            candidates = [e for e in candidates if
                          (e["type"] == "press" and abs(e["at_ms"] - first) <= 0.01) or
                          (e["type"] == "release" and e["at_ms"] <= first + 0.01)]
        else:
            candidates = [e for e in candidates if abs(e["at_ms"] - first) <= 0.01]

    candidates.sort(key=lambda event: (
        event["at_ms"], 0 if event["type"] == "release" else 1,
        event["column"],
    ))
    result = []
    for event in candidates:
        scheduled.add(event["_key"])
        result.append({"type": event["type"], "column": event["column"],
                       "at_ms": event["at_ms"]})
    return result


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if not getattr(game, "tick_s", None):
            return
        scheduled = set()
        policy = FULL_HORIZON
        try:
            policy = _dispatch_policy(ctx, _object(game.observe()))
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            if ctx.done:
                return
        while not ctx.done:
            obs = _object(game.observe())
            if obs.get("episode", {}).get("done") is True:
                return
            upcoming = obs.get("upcoming", [])
            ctx.log("chart_horizon", time_ms=obs.get("episode", {}).get("time_ms"),
                    note_ids=[n.get("id") for n in upcoming if isinstance(n, dict)] if isinstance(upcoming, list) else [])
            events = _events_from_observation(obs, scheduled, policy)
            ctx.log("dispatch_batch", event_count=len(events),
                    events=[{k: e[k] for k in ("type", "column", "at_ms")} for e in events])
            try:
                if events:
                    ctx.act({"type": "batch", "events": events})
                else:
                    # Replaces the prior one-tick batch so the real-time clock
                    # cannot keep replaying an old press/release action.
                    ctx.act("noop")
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return


def build(ctx):
    return Harness(ctx)
