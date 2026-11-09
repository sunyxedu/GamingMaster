"""Jev-only timing-policy bridge with chart-time micro-batches.

The parent asked Jev to choose one primitive action at the instant of a slow
request.  This controller keeps Jev as the only model, but makes its answer a
small policy choice: code turns the already exposed note horizon into an
absolute-time batch.  Exact chart arithmetic and hold bookkeeping stay in
code; Jev never has to race a 40 ms tick or invent a timestamp.
"""
from __future__ import annotations

import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


class Harness:
    # A decision is useful as a timing-policy refresh, not as a per-note gate.
    POLICY_REFRESH_MS = 10000.0
    HORIZON_GUARD_MS = 160.0

    def __init__(self, ctx):
        self.ctx = ctx

    @staticmethod
    def _number(value: Any):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @classmethod
    def _compact_state(cls, obs: Any) -> dict[str, Any]:
        """Keep the Jev request about policy, not a full repeated game dump."""
        if not isinstance(obs, dict):
            return {"episode": {}, "upcoming": [], "holds": [], "recent": {}}
        episode = obs.get("episode")
        chart = obs.get("chart")
        stats = obs.get("stats")
        upcoming = obs.get("upcoming")
        active = obs.get("active_holds")
        recent = obs.get("last_judgments")
        if not isinstance(episode, dict):
            episode = {}
        if not isinstance(chart, dict):
            chart = {}
        if not isinstance(stats, dict):
            stats = {}
        if not isinstance(upcoming, list):
            upcoming = []
        if not isinstance(active, list):
            active = []
        if not isinstance(recent, list):
            recent = []
        notes = []
        for note in upcoming[:64]:
            if not isinstance(note, dict):
                continue
            # These are the fields needed to decide whether exact scheduling is
            # safe; all arithmetic remains in the harness.
            notes.append({
                "id": note.get("id"),
                "column": note.get("column"),
                "start_ms": note.get("start_ms"),
                "end_ms": note.get("end_ms"),
                "kind": note.get("kind"),
                "pattern": note.get("pattern"),
                "group_size": note.get("group_size"),
                "group_columns": note.get("group_columns"),
                "in_ms": note.get("in_ms"),
            })
        holds = []
        for hold in active:
            if isinstance(hold, dict):
                holds.append({
                    "column": hold.get("column"),
                    "end_ms": hold.get("end_ms"),
                    "remaining_ms": hold.get("remaining_ms"),
                    "head_judgment": hold.get("head_judgment"),
                })
        return {
            "episode": {
                "time_ms": episode.get("time_ms"),
                "tick": episode.get("tick"),
                "columns": episode.get("columns"),
                "judge": episode.get("judge"),
                "mod": episode.get("mod"),
                "judgment_windows_ms": episode.get("judgment_windows_ms"),
            },
            "chart": {
                "pattern_family": chart.get("pattern_family"),
                "offset_ms": chart.get("offset_ms"),
                "bpm_segments": chart.get("bpm_segments"),
                "snap_resolution": chart.get("snap_resolution"),
            },
            "upcoming": notes,
            "holds": holds,
            "recent": recent[-8:],
            "stats": {
                "judgment_histogram": stats.get("judgment_histogram"),
                "tail_judgment_histogram": stats.get("tail_judgment_histogram"),
                "misses": stats.get("misses"),
                "broken_holds": stats.get("broken_holds"),
                "accuracy": stats.get("accuracy"),
                "pending_heads": stats.get("pending_heads"),
                "pending_holds": stats.get("pending_holds"),
            },
        }

    @staticmethod
    def _policy_from_answer(answer: Any) -> float:
        """Return a bounded timing offset in milliseconds.

        The exact option is deliberately first and is also the safe fallback.
        A low-confidence or malformed stand-in response cannot make a chart
        late by hundreds of milliseconds.
        """
        value = getattr(answer, "value", None)
        if value == "target_exactly":
            return 0.0
        if value == "slightly_early":
            return -8.0
        if value == "slightly_late":
            return 8.0
        return 0.0

    def _ask_policy(self, obs: Any) -> float:
        ctx = self.ctx
        question = Choice(
            """Choose the timing policy for the deterministic chart-time batch.
The harness will press every valid upcoming note at its absolute timestamp and
release each hold once its tail is safe to include in the visible timeline;
you are choosing only a small global calibration,
not an individual lane action. Prefer target_exactly unless the recent signed
judgments in `recent` show a consistent timing bias. Never choose an offset to
compensate for model latency: events are timestamped in chart time.""",
            {
                "target_exactly": {
                    "what": "Use each note's reported start_ms and each hold's reported end_ms.",
                    "not_for": "Do not add a latency compensation when timestamps are absolute.",
                },
                "slightly_early": {
                    "what": "Shift all scheduled events 8 ms before their reported target.",
                    "not_for": "Do not use this for isolated random signed offsets.",
                },
                "slightly_late": {
                    "what": "Shift all scheduled events 8 ms after their reported target.",
                    "not_for": "Do not use this when recent offsets are already centered.",
                },
            },
        )
        try:
            answers = ctx.decide(self._compact_state(obs), {"timing": question}, timeout_s=1.0)
            if not isinstance(answers, dict):
                return 0.0
            return self._policy_from_answer(answers.get("timing"))
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError):
            return 0.0

    @classmethod
    def _events_for_observation(
        cls,
        obs: Any,
        scheduled_heads: set[Any],
        scheduled_tails: set[Any],
        hold_info: dict[Any, tuple[int, float]],
        timing_offset_ms: float,
    ) -> tuple[list[dict[str, Any]], float]:
        """Build a safe batch from the visible horizon.

        The adaptor treats a timestamped batch as a mini timeline: advancing to
        a release can resolve/miss intervening notes.  Therefore a hold tail
        beyond the last currently visible head is deliberately deferred.  The
        head is remembered in ``hold_info`` and its release is emitted once the
        rolling horizon exposes the intervening notes (or the tail is imminent).
        """
        if not isinstance(obs, dict):
            return [], -math.inf
        episode = obs.get("episode")
        if not isinstance(episode, dict):
            episode = {}
        now = cls._number(episode.get("time_ms"))
        if now is None:
            now = 0.0
        columns = episode.get("columns")
        try:
            columns = int(columns)
        except (TypeError, ValueError):
            columns = 7
        upcoming = obs.get("upcoming")
        if not isinstance(upcoming, list):
            upcoming = []

        # Normalize only the public fields needed for scheduling.  In
        # particular, do not let a malformed note prevent other lanes from
        # being scheduled.
        notes: list[tuple[Any, int, float, float | None]] = []
        for note in upcoming:
            if not isinstance(note, dict):
                continue
            note_id = note.get("id")
            try:
                hash(note_id)
            except TypeError:
                note_id = repr(note_id)
            col = note.get("column")
            try:
                col = int(col)
            except (TypeError, ValueError):
                continue
            if col < 0 or col >= columns:
                continue
            start = cls._number(note.get("start_ms"))
            if start is None:
                continue
            end = cls._number(note.get("end_ms"))
            notes.append((note_id, col, start, end))
            if end is not None:
                # Keep this even after the head disappears from upcoming.
                hold_info[note_id] = (col, end)

        # Every head at or before this frontier is represented in this
        # observation.  A tail at/before it can be included without skipping
        # an unseen head between the current clock and that tail.
        frontier = max((item[2] for item in notes), default=now)
        events: list[tuple[float, int, dict[str, Any]]] = []
        planned_until = max(now, frontier)
        for note_id, col, start, _end in notes:
            target = start + timing_offset_ms
            if note_id not in scheduled_heads and target >= now - 2.0:
                events.append((target, 1, {
                    "type": "press",
                    "column": col,
                    "at_ms": round(target, 3),
                }))
                scheduled_heads.add(note_id)

        # Also learn tails from active_holds.  This covers a hold whose head
        # was scheduled in a prior batch and is no longer in `upcoming`.
        active = obs.get("active_holds")
        if isinstance(active, list):
            for hold in active:
                if not isinstance(hold, dict):
                    continue
                col = hold.get("column")
                end = cls._number(hold.get("end_ms"))
                if end is None:
                    continue
                try:
                    col = int(col)
                except (TypeError, ValueError):
                    continue
                if 0 <= col < columns:
                    hold_id = hold.get("id")
                    try:
                        hash(hold_id)
                    except TypeError:
                        hold_id = repr(hold_id)
                    if hold_id is not None:
                        hold_info[hold_id] = (col, end)

        # Do not append a long tail merely because its head is visible: that
        # would make the batch jump past heads that have not entered the 2.2 s
        # observation yet.  Once the tail is near or behind the current
        # frontier, all intervening visible heads are included first.
        for note_id, (col, end) in list(hold_info.items()):
            if note_id in scheduled_tails:
                continue
            tail = max(end + timing_offset_ms + 1.0, end + 1.0)
            safe = tail <= frontier + 2.0 or tail - now <= cls.HORIZON_GUARD_MS
            if not safe:
                continue
            # A late release is still legal for a hold.  Never send an event
            # with an absolute time in the past: the adaptor would apply it at
            # the current clock and report a malformed/late timestamp.
            tail = max(tail, end + 1.0, now)
            events.append((tail, 0, {
                "type": "release",
                "column": col,
                "at_ms": round(tail, 3),
            }))
            scheduled_tails.add(note_id)

        events.sort(key=lambda item: (item[0], item[1], item[2]["column"]))
        return [item[2] for item in events], planned_until

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if ctx.system1 is None:
            return

        scheduled_heads: set[Any] = set()
        scheduled_tails: set[Any] = set()
        hold_info: dict[Any, tuple[int, float]] = {}
        timing_offset_ms = 0.0
        next_policy_refresh = -math.inf
        planned_until = 0.0
        last_obs: Any = None

        while not ctx.done and not getattr(game, "done", False):
            try:
                obs = ctx.observe()
            except Exception:
                # The public game observation is the documented fallback during
                # a realtime tick; this is intentionally broad for pre-flight
                # stand-ins, but never masks model/backend exceptions.
                obs = game.observe()
            if not isinstance(obs, dict):
                obs = game.observe()
            last_obs = obs
            episode = obs.get("episode") if isinstance(obs, dict) else {}
            if not isinstance(episode, dict):
                episode = {}
            now = self._number(episode.get("time_ms"))
            if now is None:
                now = 0.0

            # Jev is a coarse calibration author, not a gate for the rolling
            # chart window.  The exact scheduler continues to add newly visible
            # notes with the current policy between these long refreshes.
            if now >= next_policy_refresh:
                timing_offset_ms = self._ask_policy(obs)
                next_policy_refresh = now + self.POLICY_REFRESH_MS
                # The call advances a wall-clock realtime game; use a fresh
                # observation below rather than scheduling from stale state.
                try:
                    obs = ctx.observe()
                except Exception:
                    obs = game.observe()
                if isinstance(obs, dict):
                    last_obs = obs
                    ep2 = obs.get("episode")
                    if isinstance(ep2, dict):
                        now2 = self._number(ep2.get("time_ms"))
                        if now2 is not None:
                            now = now2

            events, horizon = self._events_for_observation(
                last_obs, scheduled_heads, scheduled_tails, hold_info, timing_offset_ms
            )
            planned_until = max(planned_until, horizon)
            if events:
                try:
                    # A batch is one fresh post containing absolute chart-time
                    # events; it covers latency instead of asking Jev to race
                    # one 40 ms tick.
                    ctx.act({"type": "batch", "events": events})
                except (BudgetExceeded, ModelUnavailable):
                    raise
                except (TimeoutError, RuntimeError, ValueError, TypeError):
                    # A malformed stand-in action must degrade to a legal noop;
                    # never let a bad optional batch end a training episode.
                    try:
                        ctx.act("noop")
                    except (BudgetExceeded, ModelUnavailable):
                        raise
                    except Exception:
                        pass
            else:
                # Do not spin while waiting for the game clock.  A plain noop
                # also prevents an old primitive decision from being replayed.
                try:
                    ctx.wait_for_tick()
                    if not ctx.done and not getattr(game, "done", False):
                        # The batch has already been consumed on the tick it
                        # was posted; make the following tick explicitly
                        # neutral instead of replaying that batch.
                        ctx.act("noop")
                except (BudgetExceeded, ModelUnavailable):
                    raise
                except (TimeoutError, RuntimeError):
                    continue
                continue

            # Let the realtime adaptor consume ticks.  We do not post the same
            # batch every tick, because scheduled IDs are already in flight.
            try:
                ctx.wait_for_tick()
                if not ctx.done and not getattr(game, "done", False):
                    # A fresh noop is the fallback between timestamped
                    # batches; it prevents a prior batch from being replayed.
                    ctx.act("noop")
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                continue


def build(ctx):
    return Harness(ctx)
