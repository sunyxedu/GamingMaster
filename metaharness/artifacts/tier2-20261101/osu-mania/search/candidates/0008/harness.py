"""Hedged asynchronous Jev approvals for the realtime mania chart.

The controller deliberately separates three jobs.  A typed Jev answer approves a
set of *visible object ids*; this module keeps those approvals as a small event
reservoir; and the timestamp builder turns approved native objects into legal
future transitions.  The distinctive part of this candidate is the decision
transport: a slow Jev request is started in the background and, only when it
misses a short latency target, a second identical request is hedged against it.
The first usable answer wins while the already approved reservoir keeps playing.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


# The adaptor exposes a 2,000 ms fixed lookahead.  A successful answer should
# arrive long before that, but a hedge is useful for the observed long tail.
_REFRESH_MS = 700.0
_HEDGE_AFTER_MS = 270.0
_RETRY_MS = 180.0
_DECIDE_TIMEOUT_S = 5.0
_MAX_BACKGROUND_CALLS = 4
_DEFAULT_HORIZON_MS = 2000.0


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> dict[str, Any]:
    """Obtain a mapping from either form used by the public observation API."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _strict_json(value: Any) -> Any:
    """Make the decision state strict JSON, including hostile stand-ins."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    return str(value)


def _answer_value(answer: Any) -> Any:
    value = getattr(answer, "value", None)
    if value is not None:
        return value
    if isinstance(answer, Mapping):
        return answer.get("value", answer.get("choice"))
    return None


def _answer_probability(answer: Any, option: str) -> float | None:
    probabilities = getattr(answer, "probabilities", None)
    if probabilities is None and isinstance(answer, Mapping):
        probabilities = answer.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return None
    result = _number(probabilities.get(option))
    return result if result is not None and 0.0 <= result <= 1.0 else None


class Harness:
    """A realtime controller with a timestamped, model-approved reservoir."""

    def __init__(self, ctx: Any):
        self.ctx = ctx

        # Stable object ids are the only approval unit.  Metadata is retained
        # after an object leaves the visible window so an approved hold can be
        # released later without asking Jev about its tail again.
        self.note_meta: dict[str, dict[str, Any]] = {}
        self.decided: set[str] = set()
        self.approved: set[str] = set()
        self.head_posted: set[str] = set()
        self.tail_posted: set[str] = set()
        self.pending_events: list[tuple[float, int, bool]] = []

        # A round contains one primary request and, if necessary, one latency
        # hedge.  Old hedge futures are left cancellable/observable but never
        # allowed to overwrite a newer winning answer.
        self.round_counter = 0
        self.current_round: dict[str, Any] | None = None
        self.background: list[dict[str, Any]] = []
        self.next_round_ms = 0.0

        self.last_action: str | None = None
        self.last_policy = "execute_all"

    def _log(self, message: str, **fields: Any) -> None:
        try:
            self.ctx.log(message, **fields)
        except Exception:
            pass

    @staticmethod
    def _now(obs: Mapping[str, Any]) -> float:
        return _number(obs.get("current_time_ms", obs.get("time_ms")), 0.0) or 0.0

    @staticmethod
    def _horizon(obs: Mapping[str, Any]) -> float:
        timers = _mapping(obs.get("timers"))
        value = _number(obs.get("planning_horizon_ms"))
        if value is None:
            value = _number(obs.get("lookahead_ms"))
        if value is None:
            value = _number(timers.get("planning_horizon_ms"))
        return max(32.0, value if value is not None else _DEFAULT_HORIZON_MS)

    @staticmethod
    def _miss_window(obs: Mapping[str, Any]) -> float:
        windows = obs.get("judgement_windows_ms")
        if isinstance(windows, Mapping):
            value = _number(windows.get("MISS"))
            if value is not None:
                return max(1.0, value)
        od = _number(obs.get("od"), 0.0) or 0.0
        return max(1.0, 188.0 - 3.0 * od)

    def _collect_notes(self, obs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Flatten lane-local upcoming records into unique native objects."""
        found: dict[str, dict[str, Any]] = {}
        raw_lanes = obs.get("lanes", [])
        if not isinstance(raw_lanes, list):
            return found
        for lane_row in raw_lanes:
            if not isinstance(lane_row, Mapping):
                continue
            row_lane = _integer(lane_row.get("lane"))
            raw_notes = lane_row.get("upcoming", [])
            if not isinstance(raw_notes, list):
                continue
            for raw in raw_notes:
                if not isinstance(raw, Mapping):
                    continue
                lane = _integer(raw.get("lane", row_lane))
                start = _number(raw.get("start_ms"))
                if lane is None or start is None:
                    continue
                kind = "hold" if str(raw.get("kind", "tap")).lower() == "hold" else "tap"
                end = _number(raw.get("end_ms"))
                if kind == "hold" and (end is None or end < start):
                    continue
                raw_id = raw.get("id")
                key = str(raw_id) if raw_id is not None else f"{lane}:{start}:{kind}"

                chord = raw.get("chord_lanes", [lane])
                if not isinstance(chord, list):
                    chord = [lane]
                chord_lanes = []
                for item in chord:
                    parsed = _integer(item)
                    if parsed is not None and parsed not in chord_lanes:
                        chord_lanes.append(parsed)
                if not chord_lanes:
                    chord_lanes = [lane]

                record = {
                    "id": key,
                    "lane": lane,
                    "kind": kind,
                    "start_ms": start,
                    "end_ms": end,
                    "pattern": str(raw.get("pattern", "single")),
                    "chord_lanes": chord_lanes,
                    "chord_size": max(1, _integer(raw.get("chord_size"), len(chord_lanes)) or 1),
                    "same_lane_run_length": max(
                        1, _integer(raw.get("same_lane_run_length"), 1) or 1
                    ),
                }
                found[key] = record
                self.note_meta[key] = record
        return found

    def _new_candidates(self, obs: Mapping[str, Any]) -> list[dict[str, Any]]:
        found = self._collect_notes(obs)
        candidates = [record for key, record in found.items() if key not in self.decided]
        candidates.sort(key=lambda item: (item["start_ms"], item["lane"], item["id"]))
        return candidates

    def _decision_state(
        self, obs: Mapping[str, Any], candidates: list[dict[str, Any]], now: float
    ) -> dict[str, Any]:
        lane_state = []
        raw_lanes = obs.get("lanes", [])
        if isinstance(raw_lanes, list):
            for row in raw_lanes:
                if isinstance(row, Mapping):
                    lane = _integer(row.get("lane"))
                    if lane is not None:
                        lane_state.append({"lane": lane, "key_down": bool(row.get("key_down", False))})

        active = []
        raw_active = obs.get("active_holds", [])
        if isinstance(raw_active, list):
            for hold in raw_active[:16]:
                if not isinstance(hold, Mapping):
                    continue
                active.append({
                    "id": hold.get("id"),
                    "lane": _integer(hold.get("lane")),
                    "head_time_ms": _number(hold.get("head_time_ms")),
                    "tail_time_ms": _number(hold.get("tail_time_ms")),
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
        for item in candidates:
            compact_candidates.append({
                "id": item["id"],
                "lane": item["lane"],
                "kind": item["kind"],
                "start_ms": item["start_ms"],
                "end_ms": item["end_ms"],
                "time_to_head_ms": item["start_ms"] - now,
                "pattern": item["pattern"],
                "chord_lanes": item["chord_lanes"],
                "chord_size": item["chord_size"],
                "same_lane_run_length": item["same_lane_run_length"],
            })

        counts = obs.get("counts", {})
        if not isinstance(counts, Mapping):
            counts = {}
        return _strict_json({
            "ruleset": obs.get("ruleset", "stable-scorev1-native-mania"),
            "current_time_ms": now,
            "planning_horizon_ms": self._horizon(obs),
            "key_count": _integer(obs.get("key_count"), len(lane_state)) or len(lane_state),
            "od": _number(obs.get("od"), 0.0),
            "judgement_windows_ms": obs.get("judgement_windows_ms", {}),
            "lanes": lane_state,
            "candidate_notes": compact_candidates,
            "active_holds": active,
            "recent_results": recent,
            "telemetry": {
                "score": _number(obs.get("score"), 0.0),
                "accuracy": _number(obs.get("accuracy"), 0.0),
                "combo": _integer(obs.get("combo"), 0) or 0,
                "miss_rate": _number(obs.get("miss_rate"), 0.0),
                "hold_break_rate": _number(obs.get("hold_break_rate"), 0.0),
                "remaining_objects": _integer(obs.get("remaining_objects"), 0) or 0,
                "counts": dict(counts),
            },
            "execution_contract": {
                "approval_unit": "listed object ids only",
                "tap": "down at start_ms and up 1 ms later",
                "hold": "down at start_ms, continuous key-down, up at end_ms",
                "chord": "equal start_ms on different lanes is simultaneous",
                "timing": "the controller supplies parsed chart times; do not invent times",
                "invalid_records": "never approve a malformed record",
            },
            "reservoir": {
                "already_approved_objects": len(self.approved),
                "already_decided_objects": len(self.decided),
                "policy": "an approval remains executable while Jev is unavailable",
            },
        })

    @staticmethod
    def _question(candidate_count: int) -> Choice:
        return Choice(
            """Choose the execution policy for this already parsed osu!mania
observation.  `candidate_notes` is the complete set of new object ids offered
for this round.  Select exactly one named policy.  Do not write an action, do
not calculate milliseconds, and do not invent lanes: the timestamp executor
will use only the listed records.  The chart is deterministic and valid, so
play every complete candidate unless the supplied state explicitly makes a
record unsafe.  A policy is allowed to cover multiple simultaneous chord lanes
and both transitions of a hold.""",
            {
                "execute_all": (
                    f"Approve all {candidate_count} listed complete objects.  Preserve every chord lane, "
                    "press each head, and release each hold at its stated tail."
                ),
                "execute_taps": (
                    "Approve only listed tap objects and omit holds.  Use this only when the state gives "
                    "a concrete hold-safety or lane-conflict reason."
                ),
                "execute_holds": (
                    "Approve only listed hold objects and omit taps.  Use this only when the state gives "
                    "a concrete reason that taps cannot be executed."
                ),
                "execute_none": (
                    "Approve no object because the candidate records are malformed or the execution state "
                    "is explicitly unsafe."
                ),
            },
        )

    def _call(self, state: dict[str, Any], question: Choice) -> Any:
        # This function is intentionally tiny: ctx.spawn owns the thread and
        # ctx.decide remains the only route to Jev.
        return self.ctx.decide(
            state,
            {"selection": question},
            timeout_s=_DECIDE_TIMEOUT_S,
        )

    def _active_background(self) -> int:
        return sum(1 for entry in self.background if not self._future_done(entry["future"]))

    @staticmethod
    def _future_done(future: Any) -> bool:
        try:
            return bool(future.done())
        except Exception:
            return False

    def _spawn_round_call(self, round_info: dict[str, Any], role: str) -> bool:
        state = round_info["state"]
        question = round_info["question"]
        try:
            future = self.ctx.spawn(
                lambda state=state, question=question: self._call(state, question)
            )
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self._log("jev spawn unavailable", error=type(exc).__name__, role=role)
            return False
        self.background.append({"future": future, "round": round_info, "role": role})
        return True

    def _start_round(self, obs: Mapping[str, Any], now: float, candidates: list[dict[str, Any]]) -> None:
        if not candidates or self._active_background() >= _MAX_BACKGROUND_CALLS:
            return
        self.round_counter += 1
        state = self._decision_state(obs, candidates, now)
        round_info: dict[str, Any] = {
            "id": self.round_counter,
            "state": state,
            "question": self._question(len(candidates)),
            "candidates": candidates,
            "started_ms": now,
            "hedge_started": False,
            "winner": False,
        }
        self.current_round = round_info
        if not self._spawn_round_call(round_info, "primary"):
            self.current_round = None
            self.next_round_ms = now + _RETRY_MS
            return
        self._log(
            "jev round started",
            round_id=round_info["id"],
            candidate_count=len(candidates),
            hedge_after_ms=now + _HEDGE_AFTER_MS,
        )

    def _maybe_hedge(self, now: float) -> None:
        round_info = self.current_round
        if round_info is None or round_info["winner"] or round_info["hedge_started"]:
            return
        if now < round_info["started_ms"] + _HEDGE_AFTER_MS:
            return
        if self._active_background() >= _MAX_BACKGROUND_CALLS:
            return
        round_info["hedge_started"] = True
        if not self._spawn_round_call(round_info, "hedge"):
            self._log("jev hedge unavailable", round_id=round_info["id"])
        else:
            self._log("jev latency hedge started", round_id=round_info["id"])

    def _selection_from_result(self, result: Any) -> str | None:
        if not isinstance(result, Mapping):
            return None
        answer = result.get("selection")
        value = _answer_value(answer)
        if not isinstance(value, str):
            return None
        value = value.strip().lower()
        if value not in {"execute_all", "execute_taps", "execute_holds", "execute_none"}:
            return None
        # A low-probability refusal is treated as an undecided answer.  The
        # all-object policy is safe for a validated native chart and is accepted
        # without a confidence gate; this prevents a confidence field missing
        # from a pre-flight stand-in from crashing or starving the mailbox.
        probability = _answer_probability(answer, value)
        if value != "execute_all" and probability is not None and probability < 0.50:
            return None
        return value

    def _apply_selection(self, round_info: dict[str, Any], selection: str) -> None:
        mode = {
            "execute_all": "all",
            "execute_taps": "taps",
            "execute_holds": "holds",
            "execute_none": "none",
        }[selection]
        for item in round_info["candidates"]:
            key = str(item["id"])
            self.decided.add(key)
            if mode == "all":
                self.approved.add(key)
            elif mode == "taps" and item["kind"] == "tap":
                self.approved.add(key)
            elif mode == "holds" and item["kind"] == "hold":
                self.approved.add(key)
        self.last_policy = selection
        self._log(
            "jev round accepted",
            round_id=round_info["id"],
            policy=selection,
            candidate_count=len(round_info["candidates"]),
            approved_count=sum(
                1 for item in round_info["candidates"] if str(item["id"]) in self.approved
            ),
        )

    def _cancel_round(self, round_info: dict[str, Any]) -> None:
        for entry in self.background:
            if entry["round"] is not round_info:
                continue
            future = entry["future"]
            if not self._future_done(future):
                try:
                    future.cancel()
                except Exception:
                    pass

    def _poll_answers(self, now: float) -> None:
        remaining = []
        for entry in self.background:
            future = entry["future"]
            if not self._future_done(future):
                remaining.append(entry)
                continue

            round_info = entry["round"]
            try:
                result = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError) as exc:
                self._log(
                    "jev background call failed",
                    round_id=round_info["id"],
                    role=entry["role"],
                    error=type(exc).__name__,
                )
                result = None
            except Exception as exc:
                # Garbage stand-ins and provider formatting failures are normal
                # recovery cases for this candidate; they never become actions.
                self._log(
                    "jev background answer unusable",
                    round_id=round_info["id"],
                    role=entry["role"],
                    error=type(exc).__name__,
                )
                result = None

            if round_info.get("winner"):
                continue
            selection = self._selection_from_result(result)
            if selection is None:
                # The other member of a hedge may still provide a usable answer.
                continue

            round_info["winner"] = True
            self._apply_selection(round_info, selection)
            self._cancel_round(round_info)
            if self.current_round is round_info:
                self.current_round = None
            self.next_round_ms = now + _REFRESH_MS

        self.background = remaining

        # If every member of an unresolved round has finished, retry with a
        # fresh observation rather than leaving the controller permanently idle.
        round_info = self.current_round
        if round_info is not None and not round_info.get("winner"):
            members = [entry for entry in self.background if entry["round"] is round_info]
            if not members:
                self.current_round = None
                self.next_round_ms = now + _RETRY_MS
                self._log("jev round produced no usable answer", round_id=round_info["id"])

    def _maybe_start_round(self, obs: Mapping[str, Any], now: float) -> None:
        if self.current_round is not None or now < self.next_round_ms:
            return
        candidates = self._new_candidates(obs)
        if not candidates:
            # Poll frequently for a new object entering the fixed horizon, but
            # do not waste a Jev request on an empty Choice.
            self.next_round_ms = now + 160.0
            return
        self._start_round(obs, now, candidates)

    def _prune_pending(self, now: float) -> None:
        self.pending_events = [item for item in self.pending_events if item[0] > now + 0.25]

    def _virtual_keys(self, obs: Mapping[str, Any], now: float) -> dict[int, bool]:
        result: dict[int, bool] = {}
        raw_lanes = obs.get("lanes", [])
        if isinstance(raw_lanes, list):
            for row in raw_lanes:
                if isinstance(row, Mapping):
                    lane = _integer(row.get("lane"))
                    if lane is not None:
                        result[lane] = bool(row.get("key_down", False))
        future = [item for item in self.pending_events if item[0] > now + 0.25]
        future.sort(key=lambda item: (item[0], 0 if not item[2] else 1))
        for timestamp, lane, down in future:
            result[lane] = down
        return result

    @staticmethod
    def _target_time(wanted: Any, now: float, miss_window: float, horizon: float) -> float | None:
        value = _number(wanted)
        if value is None:
            return None
        if value < now:
            if now - value > miss_window:
                return None
            value = now + 0.5
        if value > now + horizon - 1.0:
            return None
        return max(value, now + 0.5)

    @staticmethod
    def _add_spec(
        specs: list[tuple[float, int, bool, str, str]],
        timestamp: float | None,
        lane: int,
        down: bool,
        key: str,
        phase: str,
        now: float,
        horizon: float,
    ) -> bool:
        if timestamp is None:
            return False
        if timestamp < now + 0.25 or timestamp > now + horizon - 1.0:
            return False
        specs.append((timestamp, lane, down, key, phase))
        return True

    def _build_specs(self, obs: Mapping[str, Any], now: float) -> list[tuple[float, int, bool, str, str]]:
        self._prune_pending(now)
        horizon = self._horizon(obs)
        miss_window = self._miss_window(obs)
        virtual = self._virtual_keys(obs, now)
        specs: list[tuple[float, int, bool, str, str]] = []
        planned_heads: set[str] = set()

        ordered = sorted(
            (self.note_meta[key] for key in self.approved if key in self.note_meta),
            key=lambda item: (item["start_ms"], item["lane"], item["id"]),
        )
        for item in ordered:
            key = str(item["id"])
            lane = int(item["lane"])
            kind = item["kind"]
            if kind == "tap":
                if key in self.head_posted or key in self.tail_posted:
                    continue
                if virtual.get(lane, False):
                    continue
                target = self._target_time(item["start_ms"], now, miss_window, horizon)
                if target is None:
                    continue
                if not self._add_spec(specs, target, lane, True, key, "head", now, horizon):
                    continue
                virtual[lane] = True
                if self._add_spec(specs, target + 1.0, lane, False, key, "tail", now, horizon):
                    virtual[lane] = False
                continue

            # Holds are emitted in two independently windowed phases.  A head
            # approval remains in the reservoir even after it scrolls away.
            if key not in self.head_posted:
                if not virtual.get(lane, False):
                    target = self._target_time(item["start_ms"], now, miss_window, horizon)
                    if self._add_spec(specs, target, lane, True, key, "head", now, horizon):
                        planned_heads.add(key)
                        virtual[lane] = True
            if key in self.head_posted or key in planned_heads:
                end = item.get("end_ms")
                release = self._target_time(end, now, miss_window, horizon)
                if key not in self.tail_posted and self._add_spec(
                    specs, release, lane, False, key, "tail", now, horizon
                ):
                    virtual[lane] = False

        return specs

    @staticmethod
    def _format_time(value: float) -> str:
        text = f"{float(value):.3f}"
        return text.rstrip("0").rstrip(".") or "0"

    def _post_specs(self, specs: list[tuple[float, int, bool, str, str]]) -> bool:
        if not specs:
            return False
        unique: dict[tuple[float, int, bool], tuple[float, int, bool, str, str]] = {}
        for item in specs:
            timestamp, lane, down, key, phase = item
            unique[(round(timestamp, 3), lane, down)] = item
        ordered = sorted(unique.values(), key=lambda item: (item[0], item[1], 1 if item[2] else 0))
        tokens = ["events"]
        for timestamp, lane, down, _key, _phase in ordered:
            tokens.extend((self._format_time(timestamp), str(lane), "down" if down else "up"))
        action = " ".join(tokens)
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self._log("event post failed", error=type(exc).__name__)
            return False

        for timestamp, lane, down, key, phase in ordered:
            self.pending_events.append((timestamp, lane, down))
            if phase == "head":
                self.head_posted.add(key)
            else:
                self.tail_posted.add(key)
        self.last_action = action
        return True

    def _heartbeat(self) -> None:
        if self.ctx.done:
            return
        # Timestamped batches are idempotent in this adaptor.  Reposting the
        # mailbox keeps the realtime tick fresh without asking Jev per tick.
        self.ctx.act(self.last_action or "noop")

    def _shutdown_background(self) -> None:
        for entry in self.background:
            future = entry.get("future")
            if future is not None and not self._future_done(future):
                try:
                    future.cancel()
                except Exception:
                    pass
        self.background.clear()

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        if getattr(ctx, "system1", True) is None:
            return

        try:
            while not ctx.done and not getattr(game, "done", False):
                obs = _mapping(game.observe())
                now = self._now(obs)
                self._poll_answers(now)
                self._maybe_hedge(now)
                self._maybe_start_round(obs, now)

                # Refresh after any completed model response.  This prevents a
                # timestamp accepted against an old clock from being submitted
                # as though model latency had not elapsed.
                latest = _mapping(game.observe())
                latest_now = self._now(latest)
                self._collect_notes(latest)
                specs = self._build_specs(latest, latest_now)
                if specs:
                    self._post_specs(specs)
                else:
                    self._heartbeat()

                if ctx.done:
                    break
                ctx.wait_for_tick()
        finally:
            # The evaluator cancels model calls at episode close; explicit
            # cancellation also prevents a hedge from lingering in pre-flight.
            self._shutdown_background()


def build(ctx: Any) -> Harness:
    return Harness(ctx)
