"""A leased-window controller for the realtime native mania simulator.

Jev is used as a fast semantic gate for a small, prevalidated execution lease.
The lease is not a stream of model-generated commands: once a window is granted,
this module owns timestamp arithmetic, chord grouping, key transitions, and the
long-lived tail of every granted hold.  The game is kept alive with an idempotent
mailbox heartbeat while the next lease decision is being made.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


# A lease is refreshed before the two-second public horizon is exhausted.  The
# compact request is intentionally much smaller than the parent's note dump, so
# the first cold Jev request has a chance to return before early heads expire.
_REVIEW_GAP_MS = 900.0
_DENSE_REVIEW_GAP_MS = 650.0
_RETRY_GAP_MS = 120.0
_DECIDE_TIMEOUT_S = 2.0
_DEFAULT_HORIZON_MS = 2000.0
_DEFAULT_MISS_MS = 188.0


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    """Produce strict JSON without allowing a malformed stand-in to escape."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _choice_value(answer: Any) -> str | None:
    value = getattr(answer, "value", None)
    if value is None and isinstance(answer, Mapping):
        value = answer.get("value", answer.get("choice"))
    return value.strip().lower() if isinstance(value, str) else None


def _probability(answer: Any, option: str) -> float:
    probabilities = getattr(answer, "probabilities", None)
    if probabilities is None and isinstance(answer, Mapping):
        probabilities = answer.get("probabilities", {})
    if not isinstance(probabilities, Mapping):
        return 0.0
    number = _finite(probabilities.get(option), 0.0)
    return max(0.0, min(1.0, number if number is not None else 0.0))


def _truth(answer: Any) -> float | None:
    """Read the diagnostic Noul without making it a safety-critical veto."""
    value = getattr(answer, "value", None)
    if value is None and isinstance(answer, Mapping):
        value = answer.get("value")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = _finite(value)
    if number is not None and 0.0 <= number <= 1.0:
        return number
    probabilities = getattr(answer, "probabilities", None)
    if isinstance(probabilities, Mapping):
        number = _finite(probabilities.get("true"))
        if number is not None and 0.0 <= number <= 1.0:
            return number
    return None


def _format_ms(value: float) -> str:
    if abs(value - round(value)) < 1e-7:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        # catalog survives visibility changes.  In particular, a granted hold
        # keeps its tail here after its head leaves the lane's upcoming list.
        self.catalog: dict[str, dict[str, Any]] = {}
        self.granted: set[str] = set()
        self.head_sent: set[str] = set()
        self.finish_sent: set[str] = set()
        self.last_mailbox: str | None = None
        self.next_review_ms = 0.0
        self.review_started = False
        self.late_decisions = 0
        self.last_shape = "ordinary"

    def _now(self, obs: Mapping[str, Any]) -> float:
        return _finite(obs.get("current_time_ms", obs.get("time_ms")), 0.0) or 0.0

    def _horizon(self, obs: Mapping[str, Any]) -> float:
        timers = _as_mapping(obs.get("timers"))
        value = _finite(obs.get("planning_horizon_ms"))
        if value is None:
            value = _finite(obs.get("lookahead_ms"))
        if value is None:
            value = _finite(timers.get("planning_horizon_ms"))
        return max(32.0, value if value is not None else _DEFAULT_HORIZON_MS)

    def _miss_window(self, obs: Mapping[str, Any]) -> float:
        windows = _as_mapping(obs.get("judgement_windows_ms"))
        value = _finite(windows.get("MISS"))
        return max(1.0, value if value is not None else _DEFAULT_MISS_MS)

    def _read_visible(self, obs: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Decode only public lane records; never inspect chart files or globals."""
        found: dict[str, dict[str, Any]] = {}
        rows = obs.get("lanes", [])
        if not isinstance(rows, list):
            return []
        for row in rows:
            row = _as_mapping(row)
            row_lane = _as_int(row.get("lane"))
            upcoming = row.get("upcoming", [])
            if not isinstance(upcoming, list):
                continue
            for raw in upcoming:
                note = _as_mapping(raw)
                lane = _as_int(note.get("lane", row_lane))
                start = _finite(note.get("start_ms"))
                if lane is None or start is None:
                    continue
                kind = "hold" if str(note.get("kind", "tap")).lower() == "hold" else "tap"
                end = _finite(note.get("end_ms"))
                if kind == "hold" and (end is None or end < start):
                    continue
                raw_id = note.get("id")
                ident = str(raw_id) if raw_id is not None else (
                    f"fallback:{lane}:{_format_ms(start)}:{kind}"
                )
                chord = note.get("chord_lanes", [lane])
                if not isinstance(chord, list):
                    chord = [lane]
                chord_lanes = [parsed for item in chord if (parsed := _as_int(item)) is not None]
                record = {
                    "id": ident,
                    "lane": lane,
                    "kind": kind,
                    "start_ms": start,
                    "end_ms": end,
                    "pattern": str(note.get("pattern", "single")),
                    "chord_lanes": chord_lanes or [lane],
                    "is_chord": bool(note.get("is_chord", False)),
                    "chord_size": max(1, _as_int(note.get("chord_size", 1)) or 1),
                    "run_length": max(1, _as_int(note.get("same_lane_run_length", 1)) or 1),
                }
                # A lane-local observation may repeat a chord object.  The
                # object id, not the lane row, is the lease unit.
                found.setdefault(ident, record)
        visible = list(found.values())
        visible.sort(key=lambda item: (item["start_ms"], item["lane"], item["id"]))
        self.catalog.update(found)
        return visible

    def _active_tail_fallback(self, obs: Mapping[str, Any]) -> None:
        """Recover a tail only when the public active-hold record names it."""
        active = obs.get("active_holds", [])
        if not isinstance(active, list):
            return
        now = self._now(obs)
        for raw in active:
            hold = _as_mapping(raw)
            ident = hold.get("id")
            lane = _as_int(hold.get("lane"))
            tail = None
            for field in ("end_ms", "tail_ms", "tail_time_ms", "release_time_ms"):
                tail = _finite(hold.get(field))
                if tail is not None:
                    break
            window = _as_mapping(hold.get("release_window_ms"))
            if tail is None:
                tail = _finite(window.get("centre_ms"))
            if tail is None or lane is None:
                continue
            if ident is None:
                ident = f"active:{lane}:{_format_ms(tail)}"
            key = str(ident)
            self.catalog.setdefault(key, {
                "id": key,
                "lane": lane,
                "kind": "hold",
                "start_ms": now,
                "end_ms": tail,
                "pattern": "hold",
                "chord_lanes": [lane],
                "is_chord": False,
                "chord_size": 1,
                "run_length": 1,
            })

    def _pending(self, visible: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in visible if item["id"] not in self.granted]

    def _shape_digest(
        self, obs: Mapping[str, Any], visible: list[dict[str, Any]], pending: list[dict[str, Any]]
    ) -> dict[str, Any]:
        starts: dict[float, set[int]] = {}
        kinds = {"tap": 0, "hold": 0}
        patterns: dict[str, int] = {}
        for note in pending:
            start = note["start_ms"]
            starts.setdefault(start, set()).add(note["lane"])
            kinds[note["kind"]] += 1
            pattern = note["pattern"]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        groups = list(starts.values())
        chord_groups = sum(1 for lanes in groups if len(lanes) > 1)
        earliest = pending[0]["start_ms"] if pending else None
        latest = pending[-1]["start_ms"] if pending else None
        first = []
        now = self._now(obs)
        for note in pending[:6]:
            first.append({
                "id": note["id"],
                "lane": note["lane"],
                "kind": note["kind"],
                "head_in_ms": max(-_DEFAULT_MISS_MS, round(note["start_ms"] - now, 1)),
                "tail_in_ms": (
                    None if note["end_ms"] is None else round(note["end_ms"] - now, 1)
                ),
            })
        counts = _as_mapping(obs.get("counts"))
        return _json_safe({
            "ruleset": obs.get("ruleset", "stable-scorev1-native-mania"),
            "clock_ms": round(now, 1),
            "horizon_ms": round(self._horizon(obs), 1),
            "key_count": _as_int(obs.get("key_count")),
            "od": _finite(obs.get("od"), 0.0),
            "pending_visible": len(pending),
            "pending_ids": [note["id"] for note in pending],
            "shape": {
                "object_count": len(pending),
                "head_groups": len(groups),
                "chord_groups": chord_groups,
                "tap_count": kinds["tap"],
                "hold_count": kinds["hold"],
                "patterns": patterns,
                "earliest_head_in_ms": None if earliest is None else round(earliest - now, 1),
                "latest_head_in_ms": None if latest is None else round(latest - now, 1),
            },
            "first_objects": first,
            "active_holds": [
                {
                    "lane": _as_int(_as_mapping(item).get("lane")),
                    "release_due": bool(_as_mapping(item).get("release_due", False)),
                    "hold_break_risk": str(_as_mapping(item).get("hold_break_risk", "unknown")),
                }
                for item in (obs.get("active_holds", []) or [])[:8]
                if isinstance(item, Mapping)
            ],
            "coverage": {
                "granted_objects": len(self.granted),
                "heads_sent": len(self.head_sent),
                "finishes_sent": len(self.finish_sent),
                "last_mailbox_exists": self.last_mailbox is not None,
            },
            "telemetry": {
                "score": _finite(obs.get("score"), 0.0),
                "accuracy": _finite(obs.get("accuracy"), 0.0),
                "combo": _as_int(obs.get("combo")) or 0,
                "miss_rate": _finite(obs.get("miss_rate"), 0.0),
                "hold_break_rate": _finite(obs.get("hold_break_rate"), 0.0),
                "counts": counts,
            },
        })

    def _questions(self) -> dict[str, Any]:
        return {
            "lease": Choice(
                """Choose how the controller should use this visible execution
window.  `pending_ids` names the exact note objects represented by the window;
the controller, not Jev, will emit their legal timestamped transitions.  Do not
write an action, calculate a time, or add a lane.  Activate the window when the
public records are coherent.  Preserve the existing lease only when it already
covers the pending section or when the input is genuinely unclear.""",
                {
                    "activate_visible": "Grant the controller permission to play every listed pending object, including hold heads/tails and all lanes of chords.",
                    "preserve_lease": "Do not grant new objects; the already granted mailbox is sufficient for the visible section.",
                    "pause_uncertain": "Do not grant new objects because the supplied records are malformed or explicitly unsafe.",
                },
            ),
            "shape": Choice(
                """Using only `shape`, `first_objects`, and `active_holds`, classify
what requires reading attention next.  This is recognition, not arithmetic:
choose the named family whose description best matches the supplied facts.""",
                {
                    "ordinary": "Mostly isolated taps with no meaningful simultaneous or repeated-lane load.",
                    "chords": "Several head groups contain simultaneous different lanes.",
                    "repeated_lane": "The visible family is dominated by repeated same-lane or anchor timing.",
                    "hold_transition": "A hold head/tail or active release is the main immediate concern.",
                    "mixed_dense": "Several families are interleaved or the visible section is unusually dense.",
                },
            ),
            "hold_attention": Noul(
                "Is any active hold explicitly near its release or marked high break risk in the supplied public fields?",
                {
                    "true": "At least one active hold has release_due or high hold_break_risk.",
                    "false": "No active hold is explicitly due or high risk.",
                },
            ),
        }

    def _request_lease(
        self, obs: Mapping[str, Any], visible: list[dict[str, Any]], pending: list[dict[str, Any]]
    ) -> tuple[set[str], str, bool]:
        """Ask one compact typed request and return (grants, shape, was_late)."""
        snapshot = list(pending)
        state = self._shape_digest(obs, visible, snapshot)
        try:
            answers = self.ctx.decide(state, self._questions(), timeout_s=_DECIDE_TIMEOUT_S)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError) as exc:
            # A late lease must not erase a previous mailbox.  Granting this
            # already decoded snapshot is the executable timeout fallback; it
            # is never a raw model string or an unvalidated chart command.
            self.late_decisions += 1
            self._log("jev_lease_late", error=type(exc).__name__, pending=len(snapshot))
            return {item["id"] for item in snapshot}, self.last_shape, True

        answers = answers if isinstance(answers, Mapping) else {}
        lease_answer = answers.get("lease")
        shape_answer = answers.get("shape")
        selected = _choice_value(lease_answer)
        confidence = _probability(lease_answer, selected or "")
        shape = _choice_value(shape_answer) or self.last_shape
        if shape not in {"ordinary", "chords", "repeated_lane", "hold_transition", "mixed_dense"}:
            shape = self.last_shape
        self.last_shape = shape

        # `preserve_lease` is honored only when the actual timestamp mailbox
        # reaches the whole pending section.  A clock-distance heuristic is
        # unsafe: in seed 100 the next head was 2,288 ms while the old mailbox
        # ended at 1,576 ms, even though that head was more than one second away.
        # This coverage test defeats option-order bias without discarding Jev's
        # useful no-refresh decision.
        now = self._now(obs)
        mailbox_end = max(
            (event[0] for event in self._mailbox_events()),
            default=now - 1.0,
        )
        pending_end = max(
            (
                (item["end_ms"] if item["kind"] == "hold" else item["start_ms"] + 1.0)
                for item in snapshot
            ),
            default=now,
        )
        can_preserve = (
            selected == "preserve_lease"
            and confidence >= 0.65
            and bool(self.granted)
            and mailbox_end >= pending_end
        )
        if can_preserve:
            grants: set[str] = set()
        else:
            # Invalid stand-in values and the two conservative/activation
            # choices both use the executable default.  The only deliberate
            # no-new-object branch is the high-confidence, already-covered
            # preserve case above.
            grants = {item["id"] for item in snapshot}
        guard = _truth(answers.get("hold_attention"))
        self._log(
            "jev_lease",
            selected=selected or "invalid_default_activate",
            selected_probability=round(confidence, 3),
            shape=shape,
            pending=len(snapshot),
            grants=len(grants),
            hold_attention=guard,
        )
        return grants, shape, False

    def _event_specs(self, obs: Mapping[str, Any]) -> list[tuple[float, int, bool, str, str]]:
        """Turn the granted object ledger into only currently legal events."""
        now = self._now(obs)
        upper = now + self._horizon(obs)
        miss = self._miss_window(obs)
        specs: list[tuple[float, int, bool, str, str]] = []
        ordered = sorted(
            (self.catalog[key] for key in self.granted if key in self.catalog),
            key=lambda item: (item["start_ms"], item["lane"], item["id"]),
        )
        # Local virtual state prevents a newly granted tap from generating a
        # second down behind a still-pending hold on the same lane.
        virtual: dict[int, bool] = {}
        for row in obs.get("lanes", []) if isinstance(obs.get("lanes", []), list) else []:
            row = _as_mapping(row)
            lane = _as_int(row.get("lane"))
            if lane is not None:
                virtual[lane] = bool(row.get("key_down", False))
        for timestamp, lane, down in sorted(self._mailbox_events()):
            if timestamp > now + 0.25:
                virtual[lane] = down

        for note in ordered:
            ident = note["id"]
            lane = note["lane"]
            head = note["start_ms"]
            head_ok = now - miss <= head <= upper
            if note["kind"] == "tap":
                if ident in self.head_sent or not head_ok or virtual.get(lane, False):
                    continue
                specs.append((head, lane, True, ident, "head"))
                # Keep the closing transition inside the accepted horizon even
                # for a head on its last visible millisecond.  A same-time
                # up/down pair is still an explicit tap transition.
                finish = head + 1.0 if head + 1.0 <= upper else head
                specs.append((finish, lane, False, ident, "finish"))
                virtual[lane] = False
                continue

            if ident not in self.head_sent and head_ok and not virtual.get(lane, False):
                specs.append((head, lane, True, ident, "head"))
                virtual[lane] = True
            head_known = ident in self.head_sent or any(
                item[3] == ident and item[4] == "head" for item in specs
            )
            tail = _finite(note.get("end_ms"))
            tail_ok = tail is not None and now - miss <= tail <= upper
            if ident not in self.finish_sent and head_known and tail_ok:
                specs.append((tail, lane, False, ident, "finish"))
                virtual[lane] = False
        return specs

    def _mailbox_events(self) -> list[tuple[float, int, bool]]:
        # The posted event list is only a local duplicate-suppression aid.  It
        # is intentionally reconstructed from the ledger each time rather than
        # treating a model response as authoritative key state.
        return list(getattr(self, "_future_events", []))

    def _post(self, specs: list[tuple[float, int, bool, str, str]]) -> None:
        if specs:
            unique: dict[tuple[float, int, bool], tuple[float, int, bool, str, str]] = {}
            for item in specs:
                timestamp, lane, down, ident, phase = item
                unique[(round(timestamp, 6), lane, down)] = item
            ordered = sorted(unique.values(), key=lambda item: (item[0], item[1], 0 if item[2] else 1))
            tokens = ["events"]
            for timestamp, lane, down, _ident, _phase in ordered:
                tokens.extend((_format_ms(timestamp), str(lane), "down" if down else "up"))
            action = " ".join(tokens)
            self.ctx.act(action)
            self.last_mailbox = action
            self._future_events = [(item[0], item[1], item[2]) for item in ordered]
            for _timestamp, _lane, _down, ident, phase in ordered:
                if phase == "head":
                    self.head_sent.add(ident)
                else:
                    self.finish_sent.add(ident)
            self._log("post_lease_events", events=len(ordered), granted=len(self.granted))
        elif self.last_mailbox is not None:
            # The adaptor explicitly treats repeated timestamp mailboxes as
            # idempotent.  This heartbeat gives the realtime evaluator a fresh
            # action while its clock advances through already queued events.
            self.ctx.act(self.last_mailbox)
        else:
            self.ctx.act("noop")

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        if getattr(ctx, "system1", True) is None:
            return
        while not ctx.done and not getattr(game, "done", False):
            obs = _as_mapping(game.observe())
            visible = self._read_visible(obs)
            self._active_tail_fallback(obs)
            pending = self._pending(visible)
            now = self._now(obs)
            should_review = (not self.review_started) or now >= self.next_review_ms
            if should_review and getattr(ctx, "system1", True) is not None:
                self.review_started = True
                grants, shape, late = self._request_lease(obs, visible, pending)
                self.granted.update(grants)
                latest = _as_mapping(game.observe())
                latest_visible = self._read_visible(latest)
                self._active_tail_fallback(latest)
                latest_now = self._now(latest)
                if late:
                    self.next_review_ms = latest_now + _RETRY_GAP_MS
                else:
                    gap = _DENSE_REVIEW_GAP_MS if shape in {"mixed_dense", "hold_transition"} else _REVIEW_GAP_MS
                    self.next_review_ms = latest_now + gap
                specs = self._event_specs(latest)
            else:
                specs = self._event_specs(obs)
            self._post(specs)
            if ctx.done:
                break
            ctx.wait_for_tick()


def build(ctx: Any) -> Harness:
    return Harness(ctx)
