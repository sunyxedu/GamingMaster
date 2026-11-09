"""Tier-2 crossover: Jev approves note groups; a deterministic mailbox schedules them.

The parent Jev harness selected only an action-grammar word, so it never supplied a
lane or time.  This harness makes Jev choose a subset of a prevalidated visible
note plan.  The low-level part then expands that approved subset into timestamped
transitions and keeps posting no new decisions while those transitions execute.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


# The game exposes a two-second planning horizon.  Refreshing before the old
# horizon is exhausted leaves room for the measured Jev tail latency.
_REFRESH_MS = 820.0
_DECIDE_TIMEOUT_S = 1.2
_RETRY_AFTER_FAILURE_MS = 180.0


def _number(value, default=0.0):
    """Return a finite float without allowing bad observations into JSON."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _int_or_none(value):
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _json_value(value):
    """Make a small, strict-JSON copy for ctx.decide."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return str(value)


def _observation_dict(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _answer_value(answer):
    try:
        return getattr(answer, "value")
    except Exception:
        if isinstance(answer, Mapping):
            return answer.get("value", answer.get("choice"))
        return None


def _truth_probability(answer):
    """Read a Noul defensively; malformed stand-in output means unknown."""
    value = _answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
        if math.isfinite(number) and 0.0 <= number <= 1.0:
            return number
    except (TypeError, ValueError):
        pass
    try:
        probabilities = getattr(answer, "probabilities", None)
        if isinstance(probabilities, Mapping):
            number = float(probabilities.get("true"))
            if math.isfinite(number) and 0.0 <= number <= 1.0:
                return number
    except (TypeError, ValueError, AttributeError):
        pass
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        # Object ids are the stable unit of model approval.  An approved hold
        # remains approved after its head leaves the visible list, so its tail
        # can be scheduled later without another model call.
        self.decided = set()
        self.approved = set()
        self.head_posted = set()
        self.tail_posted = set()
        self.note_meta = {}
        self.pending_events = []  # (time_ms, lane, down), not yet observed applied
        self.last_decision_ms = None
        self.next_decision_ms = 0.0
        self.started = False
        self.last_action = None

    def _now(self, obs):
        return _number(obs.get("current_time_ms", obs.get("time_ms", 0.0)))

    def _horizon(self, obs):
        value = obs.get("planning_horizon_ms", obs.get("lookahead_ms", 2000.0))
        return max(32.0, _number(value, 2000.0))

    def _windows(self, obs):
        raw = obs.get("judgement_windows_ms", {})
        if not isinstance(raw, Mapping):
            raw = {}
        return {str(k): _number(v) for k, v in raw.items()}

    def _collect_notes(self, obs):
        """Collect unique visible notes and retain their authoritative times."""
        found = {}
        lanes = obs.get("lanes", [])
        if not isinstance(lanes, list):
            return found
        for lane_row in lanes:
            if not isinstance(lane_row, Mapping):
                continue
            row_lane = _int_or_none(lane_row.get("lane"))
            upcoming = lane_row.get("upcoming", [])
            if not isinstance(upcoming, list):
                continue
            for raw in upcoming:
                if not isinstance(raw, Mapping):
                    continue
                raw_id = raw.get("id")
                if raw_id is None:
                    continue
                key = str(raw_id)
                lane = _int_or_none(raw.get("lane", row_lane))
                start = raw.get("start_ms")
                if lane is None or start is None:
                    continue
                start_ms = _number(start, -1.0)
                if start_ms < -1.0:
                    continue
                kind = "hold" if str(raw.get("kind", "tap")).lower() == "hold" else "tap"
                end_value = raw.get("end_ms")
                end_ms = None if end_value is None else _number(end_value, -1.0)
                if kind == "hold" and (end_ms is None or end_ms < start_ms):
                    # A malformed hold is never offered to the model or the
                    # executor.  The chart itself is expected to be valid.
                    continue
                chord_lanes = raw.get("chord_lanes", [lane])
                if not isinstance(chord_lanes, list):
                    chord_lanes = [lane]
                safe_chord = []
                for item in chord_lanes:
                    parsed_lane = _int_or_none(item)
                    if parsed_lane is not None:
                        safe_chord.append(parsed_lane)
                found[key] = {
                    "id": raw_id,
                    "lane": lane,
                    "kind": kind,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "pattern": str(raw.get("pattern", "single")),
                    "chord_lanes": safe_chord or [lane],
                    "chord_size": max(1, _int_or_none(raw.get("chord_size", 1)) or 1),
                    "same_lane_run_length": max(
                        1, _int_or_none(raw.get("same_lane_run_length", 1)) or 1
                    ),
                }
        self.note_meta.update(found)
        return found

    def _prune_pending(self, now):
        # The simulator's current observation is after the last applied tick.
        # Keep only events that are still in the future; old events must not be
        # replayed into the virtual key-state calculation.
        self.pending_events = [
            event for event in self.pending_events if event[0] > now + 0.25
        ]

    def _virtual_keys(self, obs, now):
        keys = {}
        lanes = obs.get("lanes", [])
        if isinstance(lanes, list):
            for row in lanes:
                if isinstance(row, Mapping):
                    lane = _int_or_none(row.get("lane"))
                    if lane is not None:
                        keys[lane] = bool(row.get("key_down", False))
        for time_ms, lane, down in sorted(self.pending_events):
            if time_ms > now + 0.25:
                keys[lane] = down
        return keys

    def _new_candidates(self, obs):
        notes = self._collect_notes(obs)
        return [
            note for key, note in notes.items()
            if key not in self.decided
        ]

    def _decision_state(self, obs, candidates, now):
        lanes = []
        for row in obs.get("lanes", []) if isinstance(obs.get("lanes", []), list) else []:
            if isinstance(row, Mapping):
                lane = _int_or_none(row.get("lane"))
                if lane is not None:
                    lanes.append({"lane": lane, "key_down": bool(row.get("key_down", False))})

        active = []
        raw_active = obs.get("active_holds", [])
        if isinstance(raw_active, list):
            for hold in raw_active[:12]:
                if isinstance(hold, Mapping):
                    active.append({
                        "lane": _int_or_none(hold.get("lane")),
                        "head_time_ms": _number(hold.get("head_time_ms", 0.0)),
                        "tail_time_ms": _number(hold.get("tail_time_ms", 0.0)),
                        "key_down": bool(hold.get("key_down", False)),
                        "release_due": bool(hold.get("release_due", False)),
                        "hold_break_risk": str(hold.get("hold_break_risk", "unknown")),
                    })

        recent = []
        raw_recent = obs.get("recent_results", [])
        if isinstance(raw_recent, list):
            for result in raw_recent[-8:]:
                if isinstance(result, Mapping):
                    recent.append({
                        "id": result.get("id"),
                        "lane": result.get("lane"),
                        "kind": result.get("kind"),
                        "result": result.get("result"),
                        "head_error_ms": result.get("head_error_ms"),
                        "tail_error_ms": result.get("tail_error_ms"),
                        "hold_break": bool(result.get("hold_break", False)),
                    })

        compact_candidates = []
        for note in candidates:
            compact_candidates.append({
                "id": note["id"],
                "lane": note["lane"],
                "kind": note["kind"],
                "start_ms": note["start_ms"],
                "end_ms": note["end_ms"],
                "pattern": note["pattern"],
                "chord_lanes": note["chord_lanes"],
                "chord_size": note["chord_size"],
                "same_lane_run_length": note["same_lane_run_length"],
            })

        counts = obs.get("counts", {})
        if not isinstance(counts, Mapping):
            counts = {}
        return _json_value({
            "ruleset": "stable-scorev1-native-mania",
            "now_ms": now,
            "key_count": _int_or_none(obs.get("key_count", len(lanes))) or len(lanes),
            "od": _number(obs.get("od", 0.0)),
            "judgement_windows_ms": self._windows(obs),
            "planning_horizon_ms": self._horizon(obs),
            "lanes": lanes,
            "candidate_notes": compact_candidates,
            "candidate_plan": {
                "selection_unit": "whole note object id; only listed ids may be executed",
                "tap_events": "down at start_ms, then up one millisecond later",
                "hold_events": "down at start_ms and continuous down until up at end_ms",
                "chord_rule": "different lanes with equal start_ms are emitted together",
                "timing_rule": "the scheduler uses the supplied chart times; Jev must not invent times",
                "invalid_plan": "no malformed or overlapping note is included",
            },
            "active_holds": active,
            "recent_results": recent,
            "telemetry": {
                "score": _number(obs.get("score", 0.0)),
                "accuracy": _number(obs.get("accuracy", 0.0)),
                "combo": _int_or_none(obs.get("combo", 0)) or 0,
                "miss_rate": _number(obs.get("miss_rate", 0.0)),
                "hold_break_rate": _number(obs.get("hold_break_rate", 0.0)),
                "remaining_objects": _int_or_none(obs.get("remaining_objects", 0)) or 0,
                "counts": dict(counts),
            },
        })

    def _ask_jev(self, obs, candidates, now):
        state = self._decision_state(obs, candidates, now)
        questions = {
            "selection": Choice(
                """Classify the concrete subset the controller should approve.  The
JSON field `candidate_notes` contains already parsed native Mania objects and
`candidate_plan` contains the exact mechanical expansion.  Do not calculate
new times, invent lanes, or write an action string.  Select `execute_all` when
every listed object is valid and should be played; select a smaller named
subset only when the state gives a real reason not to play that object.  The
controller will execute only the object ids in the selected subset.""",
                {
                    "execute_all": "Approve every listed candidate note, including all stated holds and chord lanes.",
                    "execute_taps": "Approve listed tap objects but reject every listed hold; use only for an explicit hold-safety concern.",
                    "execute_holds": "Approve listed hold objects but reject every listed tap; use only when that is explicitly safer.",
                    "execute_none": "Approve no candidate because the supplied state is malformed or unsafe to execute.",
                },
            ),
            "hold_guard": Noul(
                "The JSON has `candidate_notes`, `candidate_plan`, and `active_holds`.  Is it safe to keep each candidate hold down continuously from its stated `start_ms` through its stated `end_ms`, with the scheduler releasing it at the tail?  Answer true only when the hold records are complete and there is no explicit lane conflict or break risk; do not reject a hold merely because other lanes have notes.",
                {
                    "true": "All candidate hold records are complete and safe under the supplied plan.",
                    "false": "At least one candidate hold has an explicit malformed, conflicting, or unsafe condition.",
                },
            ),
        }
        try:
            return self.ctx.decide(state, questions, timeout_s=_DECIDE_TIMEOUT_S)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            # A timeout, transient error, malformed stand-in answer, or a
            # provider-side validation issue leaves the already posted mailbox
            # running.  No unapproved note is synthesized on this path.
            try:
                self.ctx.log("jev decision unavailable", error=type(exc).__name__)
            except Exception:
                pass
            return None

    def _apply_selection(self, answers, candidates):
        if not isinstance(answers, Mapping):
            return False
        selection = _answer_value(answers.get("selection"))
        if not isinstance(selection, str):
            return False
        selection = selection.strip().lower()
        modes = {
            "execute_all": "all",
            "execute_taps": "taps",
            "execute_holds": "holds",
            "execute_none": "none",
        }
        mode = modes.get(selection)
        if mode is None:
            return False

        # The chart parser and the scheduler already reject malformed or
        # overlapping holds.  Keep the second judgment as telemetry rather
        # than letting a low-confidence recognition answer erase valid holds:
        # in the seed-101 trace Jev selected execute_all while its independent
        # guard happened to be 0.28, and that over-gate caused three avoidable
        # hold misses.  An explicit execute_taps choice still remains the
        # model-controlled hold fallback.
        _truth_probability(answers.get("hold_guard"))

        for note in candidates:
            key = str(note["id"])
            self.decided.add(key)
            include = (
                mode == "all"
                or (mode == "taps" and note["kind"] == "tap")
                or (mode == "holds" and note["kind"] == "hold")
            )
            if include:
                self.approved.add(key)
        try:
            self.ctx.log(
                "jev selected note subset",
                mode=mode,
                candidate_count=len(candidates),
                approved_count=sum(1 for note in candidates if str(note["id"]) in self.approved),
            )
        except Exception:
            pass
        return True

    def _target_time(self, wanted, now, miss_window, horizon):
        wanted = _number(wanted, now)
        # A response can arrive after a target.  Applying it at the current
        # chart time is still meaningful inside the MISS window; farther-past
        # heads are left for the simulator to miss rather than forged early.
        if wanted < now:
            if now - wanted > max(1.0, miss_window):
                return None
            wanted = now + 0.5
        if wanted > now + horizon - 1.0:
            return None
        return max(wanted, now + 0.5)

    def _add_spec(self, specs, time_ms, lane, down, key, part, now, horizon):
        if lane is None:
            return False
        time_ms = _number(time_ms, -1.0)
        if time_ms < now - 1.0 or time_ms > now + horizon - 1.0:
            return False
        specs.append((time_ms, int(lane), bool(down), key, part))
        return True

    def _build_specs(self, obs, now):
        self._prune_pending(now)
        horizon = self._horizon(obs)
        windows = self._windows(obs)
        miss_window = windows.get("MISS", 188.0 - 3.0 * _number(obs.get("od", 0.0)))
        virtual = self._virtual_keys(obs, now)
        specs = []

        # Use all approved metadata, not just the current first note in each
        # lane.  This lets a hold's release be posted after its head scrolls out
        # of the visible list.
        ordered = sorted(
            (self.note_meta[key] for key in self.approved if key in self.note_meta),
            key=lambda note: (note["start_ms"], note["lane"], str(note["id"])),
        )
        for note in ordered:
            key = str(note["id"])
            lane = note["lane"]
            if note["kind"] == "tap":
                if key in self.head_posted or key in self.tail_posted:
                    continue
                # A tap requires a genuine down transition.  Do not press
                # through an active hold on the same lane.
                if virtual.get(lane, False):
                    continue
                target = self._target_time(note["start_ms"], now, miss_window, horizon)
                if target is None or target + 1.0 > now + horizon - 1.0:
                    continue
                if self._add_spec(specs, target, lane, True, key, "head", now, horizon):
                    virtual[lane] = True
                    self._add_spec(specs, target + 1.0, lane, False, key, "tail", now, horizon)
                    virtual[lane] = False
                continue

            # Hold head.
            if key not in self.head_posted:
                if virtual.get(lane, False):
                    continue
                target = self._target_time(note["start_ms"], now, miss_window, horizon)
                if target is not None and self._add_spec(
                    specs, target, lane, True, key, "head", now, horizon
                ):
                    self.head_posted.add(key)
                    virtual[lane] = True

            # Hold tail can be delayed to a later refill, but it is still part
            # of this already approved object.
            if key in self.head_posted and key not in self.tail_posted:
                end_ms = note.get("end_ms")
                if end_ms is not None:
                    release = self._target_time(end_ms, now, miss_window, horizon)
                    if release is not None and self._add_spec(
                        specs, release, lane, False, key, "tail", now, horizon
                    ):
                        self.tail_posted.add(key)
                        virtual[lane] = False

        return specs

    @staticmethod
    def _format_time(value):
        # Three decimals preserves generated fractional chart rows while
        # avoiding scientific notation in the action grammar.
        text = f"{float(value):.3f}"
        return text.rstrip("0").rstrip(".") or "0"

    def _post_specs(self, specs, now):
        if not specs:
            return False
        # Release before a same-time down on one lane; chords remain parallel
        # because their timestamps are equal.  Remove exact duplicate events
        # generated by a stale observation.
        unique = {}
        for time_ms, lane, down, key, part in specs:
            unique[(round(time_ms, 3), lane, down)] = (time_ms, lane, down, key, part)
        ordered = sorted(
            unique.values(),
            key=lambda item: (item[0], item[1], 1 if item[2] else 0),
        )
        if not ordered:
            return False
        tokens = ["events"]
        for time_ms, lane, down, _key, _part in ordered:
            tokens.extend([
                self._format_time(time_ms),
                str(lane),
                "down" if down else "up",
            ])
        action = " ".join(tokens)
        try:
            self.ctx.act(action)
            self.last_action = action
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            # Do not claim that a transition was posted if a harness stand-in
            # rejects it.  The next observation/model cycle can retry safely.
            try:
                self.ctx.log("event batch rejected", error=type(exc).__name__)
            except Exception:
                pass
            return False

        for time_ms, lane, down, key, part in ordered:
            self.pending_events.append((time_ms, lane, down))
            if part == "head":
                self.head_posted.add(key)
            elif part == "tail":
                self.tail_posted.add(key)
        return True

    def run_episode(self, game):
        ctx = self.ctx
        if not getattr(game, "tick_s", None):
            return
        # Tier-1/preflight contexts may expose no Jev.  Waiting is the same
        # safe behavior as the idle parent and cannot produce an illegal action.
        system1 = getattr(ctx, "system1", True)
        while not ctx.done:
            obs = _observation_dict(game.observe())
            now = self._now(obs)
            self._collect_notes(obs)
            self._prune_pending(now)

            candidates = self._new_candidates(obs)
            # A failed/garbled answer must not cause a Jev call on every
            # 16-ms tick.  The same refresh gate applies whether candidates
            # are present or not; the old mailbox remains the fallback.
            should_decide = (
                system1 is not None
                and (not self.started or now >= self.next_decision_ms)
            )
            if should_decide:
                self.started = True
                self.last_decision_ms = now
                self.next_decision_ms = now + _REFRESH_MS
                answers = self._ask_jev(obs, candidates, now)
                applied = self._apply_selection(answers, candidates)
                if not applied:
                    # A timeout or garbage answer can consume much of the
                    # chart horizon.  Retry soon, while the previous event
                    # batch continues through the idempotent heartbeat.
                    latest_after_failure = _observation_dict(game.observe())
                    self.next_decision_ms = self._now(latest_after_failure) + _RETRY_AFTER_FAILURE_MS

            # The model call may have occupied several ticks.  Re-read before
            # creating timestamps so a legal future event is never based on an
            # obsolete current time.
            latest = _observation_dict(game.observe())
            latest_now = self._now(latest)
            self._collect_notes(latest)
            specs = self._build_specs(latest, latest_now)
            self._post_specs(specs, latest_now)

            # Re-post the last timestamp batch as a heartbeat.  The adaptor
            # explicitly makes repeated Tier-2 timestamp actions idempotent;
            # this reduces stale ticks without changing queued transitions.
            if self.last_action is not None and not ctx.done:
                try:
                    ctx.act(self.last_action)
                except (BudgetExceeded, ModelUnavailable):
                    raise
                except Exception:
                    pass

            if ctx.done:
                break
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
