"""Lease-based realtime controller for the stable native osu!mania simulator.

The two earlier successful controllers had complementary weaknesses: one kept a
very conservative object approval set, while the other kept a durable timestamp
mailbox but could lose the first mailbox to a slow Jev request.  This controller
uses a short semantic lease.  Jev approves a scheduling policy, then the
post-answer observation (not the stale request observation) becomes the source
of the lease.  The lease expands the complete visible horizon into a persistent
mailbox, and the mailbox also retains hold tails after their heads disappear.

The model never supplies an action string, lane, or time.  It only selects one
of two named policies; all timestamp arithmetic and input-state bookkeeping are
owned by this module.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


# A lease is shorter than the two-second visibility window.  This leaves a
# future-event buffer while avoiding the old controller's stale candidate set.
_LEASE_MS = 700.0
_RETRY_MS = 220.0
_DECIDE_TIMEOUT_S = 1.8
_DEFAULT_HORIZON_MS = 2000.0
_DEFAULT_MISS_MS = 188.0


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _strict(value: Any) -> Any:
    """Copy a value into the strict JSON subset accepted by ctx.decide."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    return str(value)


def _answer_value(answer: Any) -> Any:
    try:
        return answer.value
    except Exception:
        if isinstance(answer, Mapping):
            return answer.get("value", answer.get("choice"))
        return None


def _answer_probability(answer: Any, option: str) -> float:
    try:
        probabilities = answer.probabilities
    except Exception:
        probabilities = answer.get("probabilities") if isinstance(answer, Mapping) else None
    if not isinstance(probabilities, Mapping):
        return 0.0
    value = _finite(probabilities.get(option), 0.0)
    return max(0.0, min(1.0, value or 0.0))


def _time_text(value: float) -> str:
    """Format chart milliseconds without scientific notation."""
    if abs(value - round(value)) < 1e-7:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _kind(raw: Any) -> str:
    return "hold" if str(raw).lower() == "hold" else "tap"


class Harness:
    """One episode of a policy-lease plus timestamp-mailbox controller."""

    def __init__(self, ctx: Any):
        self.ctx = ctx

        # `records` is deliberately retained after an object leaves the visible
        # list.  That is what lets a previously approved hold acquire its tail
        # event in a later lease.
        self.records: dict[str, dict[str, Any]] = {}
        self.leased: set[str] = set()
        self.head_posted: set[str] = set()
        self.tail_posted: set[str] = set()
        self.expired_heads: set[str] = set()
        self.event_keys: set[tuple[float, int, str]] = set()
        self.pending: list[tuple[float, int, bool]] = []

        self.last_action: str | None = None
        # This is an executable emergency policy, not a model-generated command:
        # if a Jev request times out, the previous lease is still the safest
        # useful action.  It is also the preflight/garbage-answer fallback.
        self.policy = "arm_visible"
        self.next_query_ms = 0.0
        self.queried = False
        self.last_now = 0.0

    @staticmethod
    def _now(obs: Mapping[str, Any]) -> float:
        value = _finite(obs.get("current_time_ms", obs.get("time_ms")), 0.0)
        return value or 0.0

    @staticmethod
    def _horizon(obs: Mapping[str, Any]) -> float:
        value = _finite(obs.get("planning_horizon_ms"))
        if value is None:
            timers = obs.get("timers")
            if isinstance(timers, Mapping):
                value = _finite(timers.get("planning_horizon_ms", timers.get("lookahead_ms")))
        return max(32.0, value if value is not None else _DEFAULT_HORIZON_MS)

    @staticmethod
    def _miss_window(obs: Mapping[str, Any]) -> float:
        windows = obs.get("judgement_windows_ms")
        if isinstance(windows, Mapping):
            miss = _finite(windows.get("MISS"))
            if miss is not None:
                return max(1.0, miss)
        od = _finite(obs.get("od"), 0.0) or 0.0
        return max(1.0, _DEFAULT_MISS_MS - 3.0 * od)

    def _remember_visible(self, obs: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Flatten lane-local observations into unique, validated objects."""
        found: dict[str, dict[str, Any]] = {}
        lanes = obs.get("lanes", [])
        if not isinstance(lanes, list):
            return []

        for lane_row in lanes:
            if not isinstance(lane_row, Mapping):
                continue
            fallback_lane = _integer(lane_row.get("lane"))
            upcoming = lane_row.get("upcoming", [])
            if not isinstance(upcoming, list):
                continue
            for raw in upcoming:
                if not isinstance(raw, Mapping):
                    continue
                lane = _integer(raw.get("lane", fallback_lane))
                start = _finite(raw.get("start_ms"))
                if lane is None or lane < 0 or start is None:
                    continue
                note_kind = _kind(raw.get("kind", "tap"))
                end = _finite(raw.get("end_ms"))
                if note_kind == "hold" and (end is None or end < start):
                    continue

                raw_id = raw.get("id")
                if raw_id is None:
                    ident = f"{lane}:{start:.3f}:{note_kind}:{end if end is not None else ''}"
                else:
                    ident = str(raw_id)
                chord = raw.get("chord_lanes", [lane])
                if not isinstance(chord, list):
                    chord = [lane]
                chord_lanes = []
                for item in chord:
                    parsed = _integer(item)
                    if parsed is not None and parsed >= 0:
                        chord_lanes.append(parsed)
                found.setdefault(
                    ident,
                    {
                        "id": ident,
                        "lane": lane,
                        "kind": note_kind,
                        "start_ms": start,
                        "end_ms": end,
                        "pattern": str(raw.get("pattern", "single")),
                        "chord_lanes": chord_lanes or [lane],
                        "chord_size": max(1, _integer(raw.get("chord_size", 1)) or 1),
                        "same_lane_run_length": max(
                            1, _integer(raw.get("same_lane_run_length", 1)) or 1
                        ),
                    },
                )

        self.records.update(found)
        return sorted(found.values(), key=lambda item: (item["start_ms"], item["lane"], item["id"]))

    def _remember_active_holds(self, obs: Mapping[str, Any], now: float) -> None:
        """Retain a tail if the adaptor has moved a hold to active_holds."""
        active = obs.get("active_holds", [])
        if not isinstance(active, list):
            return
        for raw in active:
            if not isinstance(raw, Mapping):
                continue
            lane = _integer(raw.get("lane"))
            if lane is None or lane < 0:
                continue
            tail = None
            for key in ("end_ms", "tail_ms", "tail_time_ms", "tail_time"):  # public variants
                tail = _finite(raw.get(key))
                if tail is not None:
                    break
            release_window = raw.get("release_window_ms")
            if tail is None and isinstance(release_window, Mapping):
                centre = _finite(release_window.get("centre_ms"))
                if centre is not None:
                    tail = centre
                else:
                    earliest = _finite(release_window.get("earliest_ms"))
                    latest = _finite(release_window.get("latest_ms"))
                    if earliest is not None and latest is not None:
                        tail = (earliest + latest) / 2.0
            if tail is None or tail < now - self._miss_window(obs):
                continue
            ident = raw.get("id")
            if ident is None:
                ident = f"active:{lane}:{tail:.3f}"
            key = str(ident)
            existing = self.records.get(key)
            if existing is None:
                self.records[key] = {
                    "id": key,
                    "lane": lane,
                    "kind": "hold",
                    "start_ms": now,
                    "end_ms": tail,
                    "pattern": "hold",
                    "chord_lanes": [lane],
                    "chord_size": 1,
                    "same_lane_run_length": 1,
                }
            elif existing.get("end_ms") is None:
                existing["end_ms"] = tail

    def _prune_pending(self, now: float) -> None:
        self.pending = [item for item in self.pending if item[0] > now + 0.25]

    def _coverage_until(self, now: float) -> float:
        future = [event[0] for event in self.pending if event[0] > now + 0.25]
        return max(future, default=now)

    def _compact_state(self, obs: Mapping[str, Any], notes: list[dict[str, Any]], now: float) -> dict[str, Any]:
        """Build a small derived state; Jev only chooses a policy lease."""
        # Rows are more useful to recognition than a long lane-local dump.  The
        # scheduler itself has the complete records and never asks Jev to count.
        rows: dict[float, dict[str, Any]] = {}
        for note in notes:
            row = rows.setdefault(
                note["start_ms"],
                {"start_ms": note["start_ms"], "lanes": [], "patterns": [], "holds": 0},
            )
            if note["lane"] not in row["lanes"]:
                row["lanes"].append(note["lane"])
            if note["pattern"] not in row["patterns"]:
                row["patterns"].append(note["pattern"])
            if note["kind"] == "hold":
                row["holds"] += 1
        row_list = sorted(rows.values(), key=lambda row: row["start_ms"])
        for row in row_list:
            row["lanes"].sort()
            row["is_chord"] = len(row["lanes"]) > 1
        kinds = {"tap": 0, "hold": 0}
        for note in notes:
            kinds[note["kind"]] += 1

        recent = obs.get("recent_results", [])
        if not isinstance(recent, list):
            recent = []
        counts = obs.get("counts", {})
        if not isinstance(counts, Mapping):
            counts = {}
        return _strict(
            {
                "ruleset": str(obs.get("ruleset", "stable-scorev1-native-mania")),
                "now_ms": now,
                "key_count": _integer(obs.get("key_count")),
                "od": _finite(obs.get("od"), 0.0),
                "windows_ms": obs.get("judgement_windows_ms", {}),
                "horizon_ms": self._horizon(obs),
                "visible": {
                    "object_count": len(notes),
                    "tap_count": kinds["tap"],
                    "hold_count": kinds["hold"],
                    "first_start_ms": notes[0]["start_ms"] if notes else None,
                    "last_start_ms": notes[-1]["start_ms"] if notes else None,
                    "rows": row_list[:48],
                },
                "active_holds": [
                    {
                        "id": h.get("id"),
                        "lane": h.get("lane"),
                        "release_due": bool(h.get("release_due", False)),
                        "hold_break_risk": str(h.get("hold_break_risk", "unknown")),
                    }
                    for h in (obs.get("active_holds", []) if isinstance(obs.get("active_holds", []), list) else [])[:12]
                    if isinstance(h, Mapping)
                ],
                "queue": {
                    "leased_objects": len(self.leased),
                    "heads_posted": len(self.head_posted),
                    "tails_posted": len(self.tail_posted),
                    "coverage_until_ms": self._coverage_until(now),
                    "policy": self.policy,
                    "lease_rule": "arm_visible approves the complete newest visible horizon; keep_queue adds nothing",
                },
                "telemetry": {
                    "score": _finite(obs.get("score"), 0.0),
                    "accuracy": _finite(obs.get("accuracy"), 0.0),
                    "combo": _integer(obs.get("combo")) or 0,
                    "miss_rate": _finite(obs.get("miss_rate"), 0.0),
                    "hold_break_rate": _finite(obs.get("hold_break_rate"), 0.0),
                    "counts": dict(counts),
                    "recent_results": recent[-6:],
                },
            }
        )

    def _ask(self, state: dict[str, Any]) -> Any:
        question = Choice(
            """Choose the mailbox policy for this structured osu!mania state.  The
controller, not Jev, supplies all lane numbers, chart timestamps, down/up
transitions, chord grouping, and hold tails.  Select `arm_visible` when the
newest visible section should be admitted to the timestamp mailbox.  Select
`keep_queue` only when the existing future queue already covers the visible
section and no newly visible object needs admission.  Do not output an action,
number, timestamp, or prose; select exactly one named option.""",
            {
                "arm_visible": {
                    "what": "Approve a lease for every validated object in the newest post-answer visible horizon, including chord lanes and hold tails.",
                    "not_for": "Do not invent objects or timing; the controller reads the refreshed observation.",
                },
                "keep_queue": {
                    "what": "Add no new objects because the persistent timestamp queue already covers the visible section.",
                    "not_for": "Do not choose this when an unleased visible object is approaching its judgement window.",
                },
            },
        )
        try:
            return self.ctx.decide(state, {"mailbox": question}, timeout_s=_DECIDE_TIMEOUT_S)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            try:
                self.ctx.log("jev lease unavailable", error=type(exc).__name__)
            except Exception:
                pass
            return None

    def _choose_policy(self, answers: Any) -> str:
        """Interpret only the typed Choice; garbage never escapes the bridge."""
        answer = answers.get("mailbox") if isinstance(answers, Mapping) else None
        value = _answer_value(answer)
        if value == "keep_queue":
            # A weak keep answer is not allowed to starve a realtime chart.
            # This probability gate also neutralizes arbitrary stand-in values.
            if _answer_probability(answer, "keep_queue") >= 0.60:
                return "keep_queue"
        if value == "arm_visible":
            return "arm_visible"
        return "arm_visible"

    def _needs_admission(self, notes: list[dict[str, Any]], now: float, horizon: float, miss: float) -> bool:
        """Check whether keep_queue would leave a near-term object uncovered."""
        upper = now + min(horizon - 2.0, 1200.0)
        for note in notes:
            ident = note["id"]
            start = note["start_ms"]
            if ident not in self.leased and now - miss <= start <= upper:
                return True
            if note["kind"] == "hold" and ident in self.leased and ident in self.head_posted:
                end = note.get("end_ms")
                if end is not None and ident not in self.tail_posted and now - miss <= end <= upper:
                    return True
        return False

    def _arm(self, notes: list[dict[str, Any]]) -> None:
        for note in notes:
            self.leased.add(str(note["id"]))

    def _target(self, wanted: float, now: float, miss: float, horizon: float) -> float | None:
        # A requested past event is still useful if it remains in the miss
        # window.  Submit it just in the future so the mailbox accepts it; the
        # simulator then judges the resulting late transition honestly.
        if wanted < now:
            if now - wanted > miss:
                return None
            wanted = now + 0.5
        if wanted > now + horizon - 1.0:
            return None
        return max(wanted, now + 0.5)

    def _event_candidates(self, obs: Mapping[str, Any], now: float) -> list[tuple[float, int, bool, str, str]]:
        """Return new legal transitions for all currently leased records."""
        horizon = self._horizon(obs)
        miss = self._miss_window(obs)
        specs: list[tuple[float, int, bool, str, str]] = []
        ordered = sorted(
            (self.records[key] for key in self.leased if key in self.records),
            key=lambda note: (note["start_ms"], note["lane"], note["id"]),
        )
        for note in ordered:
            ident = str(note["id"])
            lane = _integer(note.get("lane"))
            if lane is None or lane < 0:
                continue

            if ident not in self.head_posted and ident not in self.expired_heads:
                target = self._target(float(note["start_ms"]), now, miss, horizon)
                if target is None:
                    self.expired_heads.add(ident)
                else:
                    specs.append((target, lane, True, ident, "head"))
                    # Do not produce another head while this action is being
                    # assembled, even if a duplicated lane-local record exists.
                    self.head_posted.add(ident)

            if note["kind"] == "tap":
                if ident in self.head_posted and ident not in self.tail_posted:
                    target = self._target(float(note["start_ms"]) + 1.0, now, miss, horizon)
                    if target is not None:
                        specs.append((target, lane, False, ident, "tail"))
                        self.tail_posted.add(ident)
            else:
                end = _finite(note.get("end_ms"))
                # A hold tail is legal only after its head was admitted.  If the
                # head is in this very batch, the game will see both transitions
                # in the same timestamped mailbox action.
                if end is not None and ident in self.head_posted and ident not in self.tail_posted:
                    target = self._target(end, now, miss, horizon)
                    if target is not None:
                        specs.append((target, lane, False, ident, "tail"))
                        self.tail_posted.add(ident)
        return specs

    def _rollback_unposted(self, specs: list[tuple[float, int, bool, str, str]]) -> None:
        """Undo optimistic bookkeeping if ctx.act rejects a batch."""
        for _time, _lane, _down, ident, phase in specs:
            if phase == "head":
                self.head_posted.discard(ident)
            else:
                self.tail_posted.discard(ident)

    def _post_events(self, specs: list[tuple[float, int, bool, str, str]]) -> bool:
        if not specs:
            return False
        # Deduplicate by physical transition.  Release-first ordering makes a
        # same-lane tail/head boundary legal even in a chart with no gap.
        unique: dict[tuple[float, int, bool], tuple[float, int, bool, str, str]] = {}
        for timestamp, lane, down, ident, phase in specs:
            key = (round(timestamp, 3), lane, down)
            if key not in self.event_keys:
                unique[key] = (timestamp, lane, down, ident, phase)
        ordered = sorted(
            unique.values(),
            key=lambda item: (item[0], item[1], 1 if item[2] else 0),
        )
        if not ordered:
            return False

        tokens = ["events"]
        emitted: list[tuple[float, int, bool, str, str]] = []
        for timestamp, lane, down, ident, phase in ordered:
            tokens.extend((_time_text(timestamp), str(lane), "down" if down else "up"))
            emitted.append((timestamp, lane, down, ident, phase))
        action = " ".join(tokens)
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self._rollback_unposted(emitted)
            try:
                self.ctx.log("mailbox action rejected", error=type(exc).__name__)
            except Exception:
                pass
            return False

        for timestamp, lane, down, ident, phase in emitted:
            self.event_keys.add((round(timestamp, 3), lane, down))
            self.pending.append((timestamp, lane, down))
        self.last_action = action
        return True

    def _post_current_plan(self, obs: Mapping[str, Any], now: float) -> None:
        self._prune_pending(now)
        specs = self._event_candidates(obs, now)
        if self._post_events(specs):
            return
        # A timestamp batch is a durable mailbox.  Reposting it is harmless and
        # documented idempotent; it also supplies a fresh action on ticks while
        # the next Jev request is in flight.
        if self.last_action is not None:
            try:
                self.ctx.act(self.last_action)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                pass
        else:
            # This starts the mailbox cleanly in preflight and is the exact
            # conservative fallback while the first policy answer is pending.
            try:
                self.ctx.act("noop")
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                pass

    def _query_due(self, now: float) -> bool:
        return (not self.queried) or now >= self.next_query_ms

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        if not getattr(game, "tick_s", None):
            return
        if getattr(ctx, "system1", True) is None:
            return

        while not ctx.done and not getattr(game, "done", False):
            observation = _dict(game.observe())
            now = self._now(observation)
            self.last_now = now
            notes = self._remember_visible(observation)
            self._remember_active_holds(observation, now)
            self._prune_pending(now)

            if self._query_due(now):
                # The request state is intentionally compact.  After the answer
                # we re-read the game, so a response never authorizes a stale
                # object list merely because it was visible at request time.
                state = self._compact_state(observation, notes, now)
                self.queried = True
                try:
                    answers = self._ask(state)
                    selected = self._choose_policy(answers)
                    self.policy = selected
                    latest = _dict(game.observe())
                    latest_now = self._now(latest)
                    latest_notes = self._remember_visible(latest)
                    self._remember_active_holds(latest, latest_now)
                    if selected == "arm_visible" or self._needs_admission(
                        latest_notes,
                        latest_now,
                        self._horizon(latest),
                        self._miss_window(latest),
                    ):
                        self._arm(latest_notes)
                    self.next_query_ms = latest_now + _LEASE_MS
                    self.ctx.log(
                        "lease decision",
                        selected=selected,
                        requested_at_ms=now,
                        applied_at_ms=latest_now,
                        visible_objects=len(latest_notes),
                        leased_objects=len(self.leased),
                    )
                    observation = latest
                    now = latest_now
                except (BudgetExceeded, ModelUnavailable):
                    raise
                except Exception as exc:
                    # _ask catches ordinary model failures, but keep this outer
                    # guard for malformed stand-ins and unexpected answer types.
                    self.policy = "arm_visible"
                    latest = _dict(game.observe())
                    now = self._now(latest)
                    latest_notes = self._remember_visible(latest)
                    self._remember_active_holds(latest, now)
                    self._arm(latest_notes)
                    self.next_query_ms = now + _RETRY_MS
                    try:
                        self.ctx.log("lease fallback", error=type(exc).__name__)
                    except Exception:
                        pass
                    observation = latest

            self._post_current_plan(observation, now)
            if ctx.done:
                break
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                if ctx.done:
                    break
                # A transient wait failure is no reason to abandon an already
                # posted mailbox; retry the loop and let the contract decide the
                # eventual episode boundary.
                continue


def build(ctx: Any) -> Harness:
    return Harness(ctx)
