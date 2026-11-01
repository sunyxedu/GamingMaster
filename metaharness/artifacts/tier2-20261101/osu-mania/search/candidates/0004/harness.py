"""Candidate 0004: pipelined note approval with a deterministic event scheduler.

The simulator exposes exact future note timestamps.  Code owns the clock,
transition construction, chord grouping, and hold bookkeeping.  Jev is asked
only to approve concrete visible note objects; it is never asked to emit a
partially specified action such as ``event <time> ...``.  Astra, when
available, writes a bounded policy asynchronously while Jev and the scheduler
continue playing.
"""
from __future__ import annotations

import json
import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Noul


class Harness:
    """A real-time controller for the structured osu!mania observation."""

    JEV_TIMEOUT_S = 2.0
    ASTRA_TIMEOUT_S = 180.0
    MAX_CANDIDATES = 64
    MAX_EVENTS_PER_ACTION = 96

    def __init__(self, ctx):
        self.ctx = ctx
        self.pending_jev = None
        self.pending_jev_meta = None
        self.astra_future = None
        self.astra_consumed = False
        self.reviewed = set()
        self.approved = {}
        self.sent_events = set()
        self.next_jev_ms = 0.0
        self.policy = {
            # Exact chart timestamps are the safest default.  Astra may make
            # only a small, explicitly bounded change after inspecting facts.
            "timing_bias_ms": 0.0,
            "approval_threshold": 0.45,
            "tail_bias_ms": 0.0,
            "mode": "all_visible_valid_notes",
        }

    # ---------- defensive observation/state helpers ----------

    @staticmethod
    def _finite_number(value, default=None):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    @classmethod
    def _json_safe(cls, value):
        """Make a compact, strict-JSON copy for ctx.decide."""
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(v) for v in value]
        return str(value)

    def _observe(self):
        try:
            observation = self.ctx.observe()
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return {}
        if isinstance(observation, dict):
            return observation
        if isinstance(observation, str):
            try:
                parsed = json.loads(observation)
            except (TypeError, ValueError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _now(obs):
        return Harness._finite_number(
            obs.get("current_time_ms", obs.get("time_ms", 0.0)), 0.0
        )

    @staticmethod
    def _horizon(obs):
        value = Harness._finite_number(
            obs.get("planning_horizon_ms", obs.get("lookahead_ms", 2000.0)),
            2000.0,
        )
        return max(0.0, value)

    def _miss_window(self, obs):
        windows = obs.get("judgement_windows_ms", {})
        miss = self._finite_number(windows.get("MISS"), 188.0)
        return max(1.0, miss)

    def _extract_notes(self, obs):
        """Return normalized visible notes keyed by stable object id."""
        lanes = obs.get("lanes", [])
        if isinstance(lanes, dict):
            normalized_lanes = []
            for key, value in lanes.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("lane", key)
                    normalized_lanes.append(item)
            lanes = normalized_lanes

        notes = {}
        if not isinstance(lanes, list):
            return notes

        for lane_data in lanes:
            if not isinstance(lane_data, dict):
                continue
            lane_value = lane_data.get("lane")
            try:
                lane = int(lane_value)
            except (TypeError, ValueError):
                continue

            upcoming = lane_data.get("upcoming", lane_data.get("notes", []))
            if not isinstance(upcoming, list):
                continue

            for raw in upcoming:
                if not isinstance(raw, dict) or raw.get("id") is None:
                    continue
                start = self._finite_number(raw.get("start_ms"))
                if start is None:
                    continue
                key = str(raw.get("id"))
                kind = str(raw.get("kind", "tap")).lower()
                end = self._finite_number(raw.get("end_ms"))
                if kind == "hold" and (end is None or end <= start):
                    # A malformed hold cannot be scheduled safely.  Treating
                    # it as a tap is preferable to an invalid release pair.
                    kind = "tap"
                    end = None

                chord_lanes = raw.get("chord_lanes", [lane])
                if not isinstance(chord_lanes, list):
                    chord_lanes = [lane]
                clean_chord_lanes = []
                for chord_lane in chord_lanes:
                    try:
                        clean_chord_lanes.append(int(chord_lane))
                    except (TypeError, ValueError):
                        pass
                if not clean_chord_lanes:
                    clean_chord_lanes = [lane]

                note = {
                    "id": key,
                    "lane": lane,
                    "kind": kind,
                    "start_ms": start,
                    "end_ms": end,
                    "pattern": str(raw.get("pattern", "single")),
                    "chord_lanes": clean_chord_lanes,
                    "chord_size": len(clean_chord_lanes),
                    "same_lane_run_length": int(
                        self._finite_number(raw.get("same_lane_run_length"), 1)
                    ),
                    "head_window_ms": self._json_safe(
                        raw.get("head_window_ms", {})
                    ),
                }
                # If duplicate copies occur in a malformed observation, keep
                # the earliest copy rather than expanding the action.
                previous = notes.get(key)
                if previous is None or note["start_ms"] < previous["start_ms"]:
                    notes[key] = note

        return notes

    def _compact_state(self, obs, candidates):
        patterns = []
        for row in obs.get("visible_patterns", [])[:16]:
            if not isinstance(row, dict):
                continue
            patterns.append({
                "row_index": row.get("row_index"),
                "start_ms": row.get("start_ms"),
                "time_to_start_ms": row.get("time_to_start_ms"),
                "lanes": row.get("lanes", []),
                "pattern": row.get("pattern"),
                "is_chord": row.get("is_chord", False),
                "anchor_lane": row.get("anchor_lane"),
            })

        recent = obs.get("recent_results", [])
        if not isinstance(recent, list):
            recent = []

        state = {
            "schema": "mania-note-approval-v1",
            "ruleset": obs.get("ruleset", "stable-scorev1-native-mania"),
            "now_ms": self._now(obs),
            "key_count": obs.get("key_count"),
            "od": obs.get("od"),
            "planning_horizon_ms": self._horizon(obs),
            "judgement_windows_ms": obs.get("judgement_windows_ms", {}),
            "policy": dict(self.policy),
            "candidate_notes": list(candidates.values()),
            "visible_patterns": patterns,
            "active_holds": obs.get("active_holds", [])[:16],
            "recent_results": recent[-8:],
            "counts": obs.get("counts", {}),
            "accuracy": obs.get("accuracy"),
            "score": obs.get("score"),
            "combo": obs.get("combo"),
            "miss_rate": obs.get("miss_rate"),
            "hold_break_rate": obs.get("hold_break_rate"),
            "chart_metrics": obs.get("chart_metrics", {}),
            "instruction": (
                "The code has already decoded legal note lanes and exact chart "
                "times. Approve visible candidate objects so every valid note "
                "can be scored. Do not invent objects, timestamps, lanes, or "
                "action strings. Chords are approved independently by lane and "
                "will be emitted simultaneously by code; holds are approved as "
                "one object and their tail release is emitted later by code."
            ),
        }
        return self._json_safe(state)

    # ---------- Astra: asynchronous policy author ----------

    def _start_astra(self, obs):
        if self.astra_future is not None:
            return
        if getattr(self.ctx, "system2", None) is None:
            return

        initial = self._compact_state(obs, self._extract_notes(obs))
        prompt = (
            "You are the slow policy author for a real-time offline osu!mania "
            "controller. The chart clock advances while you answer. Code will "
            "parse notes and schedule exact timestamped down/up transitions; "
            "Jev will approve concrete visible note IDs. Do not output game "
            "actions or timestamps. Return compact JSON only with optional "
            "fields timing_bias_ms, tail_bias_ms, approval_threshold, and mode. "
            "Use zero timing bias unless recent signed errors justify a small "
            "bounded correction. Keep approval_threshold between 0.45 and 0.60. "
            "The stable ruleset has one judgement per hold, continuous key-down "
            "state, simultaneous chords, and zero-based lanes.\n\nSTATE:\n"
            + json.dumps(initial, separators=(",", ":"))
        )

        def call_astra():
            try:
                return self.ctx.think(prompt, timeout_s=self.ASTRA_TIMEOUT_S).text
            except Exception as exc:  # cancelled/garbage Astra must not kill play
                return {"error": type(exc).__name__}

        try:
            self.astra_future = self.ctx.spawn(call_astra)
        except Exception:
            self.astra_future = None

    def _poll_astra(self):
        if self.astra_future is None or self.astra_consumed:
            return
        try:
            if not self.astra_future.done():
                return
            raw = self.astra_future.result()
        except Exception:
            self.astra_consumed = True
            return

        self.astra_consumed = True
        if not isinstance(raw, str):
            return
        left = raw.find("{")
        right = raw.rfind("}")
        if left < 0 or right <= left:
            return
        try:
            data = json.loads(raw[left:right + 1])
        except (TypeError, ValueError):
            return
        if not isinstance(data, dict):
            return

        bias = self._finite_number(data.get("timing_bias_ms"))
        tail_bias = self._finite_number(data.get("tail_bias_ms"))
        threshold = self._finite_number(data.get("approval_threshold"))
        if bias is not None:
            self.policy["timing_bias_ms"] = max(-4.0, min(4.0, bias))
        if tail_bias is not None:
            self.policy["tail_bias_ms"] = max(-4.0, min(4.0, tail_bias))
        if threshold is not None:
            self.policy["approval_threshold"] = max(0.45, min(0.60, threshold))
        mode = data.get("mode")
        if isinstance(mode, str) and mode:
            self.policy["mode"] = mode[:80]
        self.ctx.log("astra_policy", policy=dict(self.policy))

    # ---------- Jev: batched typed approval ----------

    @staticmethod
    def _answer_probability(answer):
        value = getattr(answer, "value", None)
        probabilities = getattr(answer, "probabilities", None)
        if isinstance(answer, dict):
            value = answer.get("value", answer.get("probability", value))
            probabilities = answer.get("probabilities", probabilities)

        if isinstance(value, bool):
            return 1.0 if value else 0.0
        number = Harness._finite_number(value)
        if number is not None and 0.0 <= number <= 1.0:
            return number
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "approve", "approved", "play"}:
                return 1.0
            if lowered in {"false", "no", "reject", "rejected", "skip"}:
                return 0.0
        if isinstance(probabilities, dict):
            return Harness._finite_number(
                probabilities.get("true", probabilities.get("yes")), 0.0
            )
        return None

    def _launch_jev(self, obs):
        if self.pending_jev is not None:
            return
        if getattr(self.ctx, "system1", None) is None:
            return
        now = self._now(obs)
        if now < self.next_jev_ms:
            return

        visible = self._extract_notes(obs)
        candidates = {
            key: note for key, note in visible.items()
            if key not in self.reviewed and key not in self.approved
        }
        if not candidates:
            return

        ordered = sorted(
            candidates.items(),
            key=lambda pair: (pair[1]["start_ms"], pair[1]["lane"], pair[0]),
        )[: self.MAX_CANDIDATES]
        candidates = dict(ordered)

        questions = {}
        qmap = {}
        for index, (key, note) in enumerate(ordered):
            qid = "approve_%d" % index
            qmap[qid] = key
            questions[qid] = Noul(
                (
                    "Should the controller execute candidate note id %s? "
                    "The candidate is already visible in `candidate_notes`, its "
                    "lane and timestamp were decoded by code, and the objective "
                    "is to score every valid visible note. Answer true for a "
                    "normal valid note, including notes in chords, streams, "
                    "jacks, anchors, and holds. Answer false only if this exact "
                    "candidate is not a valid object to execute. Do not compute "
                    "milliseconds or invent an action."
                ) % key,
                {
                    "true": "Execute this visible valid note.",
                    "false": "Do not execute this candidate.",
                },
            )

        state = self._compact_state(obs, candidates)
        request = {
            "state": state,
            "questions": questions,
            "qmap": qmap,
            "candidate_ids": list(candidates),
            "requested_at_ms": now,
        }

        def call_jev():
            try:
                answers = self.ctx.decide(
                    request["state"],
                    request["questions"],
                    timeout_s=self.JEV_TIMEOUT_S,
                )
                return {"answers": answers}
            except Exception as exc:
                return {"error": type(exc).__name__}

        try:
            self.pending_jev = self.ctx.spawn(call_jev)
            self.pending_jev_meta = request
        except Exception:
            self.pending_jev = None
            self.pending_jev_meta = None
            self.next_jev_ms = now + 500.0

    def _finish_jev(self, obs):
        if self.pending_jev is None:
            return
        try:
            if not self.pending_jev.done():
                return
            result = self.pending_jev.result()
        except Exception:
            result = {"error": "future_failure"}

        meta = self.pending_jev_meta or {}
        self.pending_jev = None
        self.pending_jev_meta = None
        now = self._now(obs)

        if not isinstance(result, dict) or "answers" not in result:
            self.next_jev_ms = now + 500.0
            return

        answers = result.get("answers")
        if not isinstance(answers, dict):
            self.next_jev_ms = now + 500.0
            return

        qmap = meta.get("qmap", {})
        candidates = self._extract_notes(obs)
        # Mark a completed batch as reviewed even if the note has just left
        # the visible window.  It was still a model-approved candidate at the
        # time of the request, and approved hold tails remain in self.approved.
        for qid, key in qmap.items():
            if qid not in answers:
                continue
            self.reviewed.add(key)
            probability = self._answer_probability(answers[qid])
            if probability is None or probability < self.policy["approval_threshold"]:
                continue
            note = candidates.get(key)
            if note is None:
                # Use the original normalized candidate if latency moved it
                # out of the latest observation.
                for old_key, old_note in zip(
                    meta.get("candidate_ids", []),
                    meta.get("candidate_ids", []),
                ):
                    if old_key == key:
                        break
                # The original notes are reconstructed from the compact state.
                for original in meta.get("state", {}).get("candidate_notes", []):
                    if isinstance(original, dict) and str(original.get("id")) == key:
                        note = original
                        break
            if isinstance(note, dict):
                self.approved[key] = note

        self.ctx.log(
            "jev_note_batch",
            reviewed=list(qmap.values()),
            approved=[key for key in qmap.values() if key in self.approved],
        )

    # ---------- deterministic transition scheduler ----------

    def _pending_events(self, obs):
        now = self._now(obs)
        horizon_end = now + self._horizon(obs)
        miss = self._miss_window(obs)
        head_bias = self.policy["timing_bias_ms"]
        tail_bias = self.policy["tail_bias_ms"]
        pending = []

        for key, note in self.approved.items():
            try:
                lane = int(note["lane"])
                start = float(note["start_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            end = self._finite_number(note.get("end_ms"))
            kind = note.get("kind", "tap")

            head_time = int(round(start + head_bias))
            head_key = (key, "head")
            # A past event is legal and is judged at current time.  Keep a
            # small recovery window so a slow response can still earn a lower
            # judgement instead of abandoning the object.
            head_recoverable = now <= start + miss + 1.0
            if head_key not in self.sent_events and head_time <= horizon_end:
                if head_time >= now or head_recoverable:
                    pending.append((
                        head_time, lane, "down", head_key, key
                    ))

            if kind == "hold" and end is not None:
                tail_time = int(round(end + tail_bias))
                tail_key = (key, "tail")
                if tail_key not in self.sent_events and tail_time <= horizon_end:
                    # Even a late release cleans up the held lane.  The judge
                    # itself determines whether that release is a miss.
                    pending.append((
                        tail_time, lane, "up", tail_key, key
                    ))
            elif kind != "hold":
                # A one-millisecond distinct up transition is enough for a
                # tap and avoids holding a lane into the next jack.
                up_time = max(head_time + 1, int(round(start + head_bias + 1)))
                up_key = (key, "tap_up")
                if up_key not in self.sent_events and up_time <= horizon_end:
                    pending.append((
                        up_time, lane, "up", up_key, key
                    ))

        # Release before press when two legal objects share a boundary.
        pending.sort(key=lambda item: (
            item[0], 0 if item[2] == "up" else 1, item[1], item[3]
        ))
        return pending[: self.MAX_EVENTS_PER_ACTION]

    def _emit_events(self, obs):
        pending = self._pending_events(obs)
        if not pending:
            return

        fields = []
        for timestamp, lane, direction, _event_key, _note_key in pending:
            fields.extend((str(timestamp), str(lane), direction))

        if len(pending) == 1:
            action = "event " + " ".join(fields)
        else:
            action = "events " + " ".join(fields)

        # The event grammar is generated entirely from validated numeric
        # fields, so malformed model text cannot produce malformed actions.
        self.ctx.act(action)
        for _timestamp, _lane, _direction, event_key, _note_key in pending:
            self.sent_events.add(event_key)
        self.ctx.log(
            "scheduled_events",
            count=len(pending),
            first_time=pending[0][0],
            last_time=pending[-1][0],
        )

    # ---------- real-time loop ----------

    def run_episode(self, game) -> None:
        if not getattr(game, "tick_s", None):
            return

        obs = self._observe()
        self._start_astra(obs)
        self._launch_jev(obs)

        while not self.ctx.done and not game.done:
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                break

            if self.ctx.done or game.done:
                break

            obs = self._observe()
            self._poll_astra()
            self._finish_jev(obs)
            self._emit_events(obs)
            self._launch_jev(obs)


def build(ctx):
    return Harness(ctx)
