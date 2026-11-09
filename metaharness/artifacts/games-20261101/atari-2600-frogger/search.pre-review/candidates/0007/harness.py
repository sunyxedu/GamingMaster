"""Temporal Frogger controller: one-frame pulses plus verified recovery cues.

The controller keeps the useful edge/debounce bridge from the strongest parent,
but does not wait for Jev on the 60 Hz loop.  It observes every frame, turns the
few repeatable RAM transitions into named temporal evidence, and uses Jev for a
small posture/action judgment.  Astra is an out-of-band advisory writer; an
unavailable or late card never stops the executable controller.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


_ACTIONS = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN")
_DIRECTIONS = frozenset(("UP", "RIGHT", "LEFT", "DOWN"))
_TACTIC_TO_ACTION = {
    "ADVANCE": "UP",
    "WAIT": "NOOP",
    "SHIFT_LEFT": "LEFT",
    "SHIFT_RIGHT": "RIGHT",
    "RETREAT": "DOWN",
}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _text(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    try:
        return str(value)[:limit]
    except Exception:
        return ""


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Make a strict-JSON decision state from arbitrary adaptor data."""
    if depth > 5:
        return _text(value, 160)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in value]
    return _text(value, 160)


def _raw(obs: Any) -> list[int]:
    if not isinstance(obs, dict):
        return []
    values = obs.get("raw_ram", [])
    if not isinstance(values, (list, tuple)):
        return []
    result: list[int] = []
    for value in values[:128]:
        result.append(_int(value, 0) & 0xFF)
    return result


def _byte(obs: Any, address: int) -> int:
    raw = _raw(obs)
    index = address - 0x80
    if 0 <= index < len(raw):
        return raw[index]
    fields = obs.get("ram_fields", {}) if isinstance(obs, dict) else {}
    if isinstance(fields, dict):
        for key in (hex(address), f"{address:02x}", f"{address:02X}"):
            if key in fields:
                return _int(fields[key], 0) & 0xFF
    return 0


def _frog(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    value = obs.get("frog", {})
    return value if isinstance(value, dict) else {}


def _fields(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    value = obs.get("ram_fields", {})
    return value if isinstance(value, dict) else {}


def _row(obs: Any) -> int:
    frog = _frog(obs)
    fields = _fields(obs)
    return _int(frog.get("row", fields.get("frog_row_ac", -1)), -1)


def _x(obs: Any) -> int:
    frog = _frog(obs)
    fields = _fields(obs)
    return _int(frog.get("x", fields.get("frog_x_b0", -1)), -1)


def _screen_x(obs: Any) -> int:
    frog = _frog(obs)
    fields = _fields(obs)
    return _int(
        frog.get("screen_x", fields.get("frog_screen_x_b4", -1)), -1
    )


def _phase(obs: Any) -> str:
    if not isinstance(obs, dict):
        return "UNKNOWN"
    if bool(obs.get("terminal", False)):
        return "TERMINAL"
    row = _row(obs)
    # These are deliberately broad labels.  The useful verified fact is that
    # AC changes from 255 to 0, then increments on forward hops; row 7 is the
    # first observed moving-platform transition in the archive.
    if row < 0:
        return "UNKNOWN"
    if row == 255:
        return "START"
    if row <= 6:
        return "ROAD_OR_MEDIAN"
    if row <= 11:
        return "RIVER"
    return "HOME_APPROACH"


def _answer_value(answer: Any) -> Any:
    try:
        return answer.value
    except Exception:
        if isinstance(answer, dict):
            return answer.get("value", answer.get("choice"))
        return None


def _probability(answer: Any, option: str) -> float:
    try:
        values = answer.probabilities
        return _number(values.get(option, 0.0), 0.0)
    except Exception:
        if isinstance(answer, dict):
            values = answer.get("probabilities", {})
            if isinstance(values, dict):
                return _number(values.get(option, 0.0), 0.0)
        return 0.0


def _valid_action(value: Any, legal: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    return value if value in legal else None


def _parse_advice(text: Any) -> dict[str, str]:
    """Keep only a bounded advisory card; never execute Astra prose."""
    raw = _text(text, 1200).strip()
    parsed: Any = None
    candidates = [raw]
    left, right = raw.find("{"), raw.rfind("}")
    if left >= 0 and right > left:
        candidates.append(raw[left : right + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                parsed = value
                break
        except Exception:
            continue

    def field(*names: str, limit: int = 240) -> str:
        if not isinstance(parsed, dict):
            return ""
        for name in names:
            value = parsed.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()[:limit]
        return ""

    focus = field("focus", "subgoal", "phase")
    rule = field("rule", "wait_rule", "forward_rule")
    lateral = field("lateral_rule", "alignment")
    if not focus and raw:
        upper = raw.upper()
        for word in ("RIVER", "ROAD", "HOME", "START"):
            if word in upper:
                focus = word
                break
    return {
        "focus": focus or "unknown",
        "rule": rule or "wait only when the landing is not established",
        "lateral_rule": lateral or "use lateral movement only for recovery or edge risk",
        "raw": raw[:700],
    }


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.legal: set[str] = set(_ACTIONS)
        self.noop = "NOOP"
        self.prev_obs: dict[str, Any] = {}
        self.prior_obs: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.life = 0
        self.last_deaths = 0
        self.last_score = 0.0
        self.last_row = 255
        self.last_x = 40
        self.last_screen_x = 40
        self.last_tick = 0
        self.last_progress_tick = 0
        self.last_direction_tick = -1000
        self.cooldown = 0
        self.current_action = "NOOP"
        self.shift_pending = False
        self.shift_done = False
        self.shift_dir = "RIGHT"
        self.restore_done = False
        self.last_death_phase = "UNKNOWN"
        self.last_death_row = -1
        self.hazard_seen = False
        self.hazard_rising = False
        self.hazard_escape_pending = False
        self.release_pending = False
        self.pending_jev: Any = None
        self.pending_astra: Any = None
        self.astra_calls = 0
        self.advice = {
            "focus": "unknown",
            "rule": "wait only when the landing is not established",
            "lateral_rule": "use lateral movement only for recovery or edge risk",
            "raw": "No Astra card yet.",
        }

    def _log(self, message: str, **fields: Any) -> None:
        try:
            self.ctx.log(message, **fields)
        except Exception:
            pass

    def _safe_observe(self, game: Any) -> dict[str, Any]:
        try:
            value = game.observe()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _update(self, obs: dict[str, Any], posted: str) -> None:
        previous = self.prev_obs
        tick = _int(obs.get("tick", obs.get("episode_frame_number", self.last_tick + 1)))
        row = _row(obs)
        x = _x(obs)
        sx = _screen_x(obs)
        score = _number(obs.get("score", 0.0))
        deaths = _int(obs.get("deaths", self.last_deaths))
        prev_row = _row(previous) if previous else self.last_row
        prev_x = _x(previous) if previous else self.last_x
        prev_sx = _screen_x(previous) if previous else self.last_screen_x
        prev_score = _number(previous.get("score", self.last_score)) if previous else self.last_score
        prev_deaths = _int(previous.get("deaths", self.last_deaths)) if previous else self.last_deaths
        e7 = _byte(obs, 0xE7)
        old_e7 = _byte(previous, 0xE7) if previous else 0
        self.hazard_rising = bool(e7 and not old_e7) or (e7 != old_e7 and bool(e7))
        self.hazard_seen = bool(e7)
        if self.hazard_rising and _phase(obs) == "RIVER":
            # Keep a one-frame release after the preceding direction, then
            # spend the next available pulse on an escape rather than dropping
            # the short-lived cue while Jev is in flight.
            self.hazard_escape_pending = True

        if deaths > prev_deaths:
            self.life += deaths - prev_deaths
            self.last_death_phase = _phase(previous) if previous else "UNKNOWN"
            self.last_death_row = prev_row
            # A column change is a recovery experiment, not a seed-specific
            # route.  Alternate sides after each lost life and restore at the
            # median so home alignment is not permanently sacrificed.
            self.shift_pending = self.life > 0
            self.shift_done = False
            self.restore_done = False
            self.shift_dir = "RIGHT" if self.life % 2 else "LEFT"
            self.hazard_escape_pending = False
            self.last_progress_tick = tick

        if score > prev_score or (row != prev_row and row not in (-1, 255)):
            self.last_progress_tick = tick

        event = "steady"
        if deaths > prev_deaths:
            event = "death_reset"
        elif bool(obs.get("terminal", False)):
            event = "terminal"
        elif self.hazard_rising:
            event = "river_hazard_rising"
        elif score > prev_score:
            event = "score_progress"
        elif row != prev_row:
            event = "row_transition"
        elif sx != prev_sx or x != prev_x:
            event = "carrier_or_motion"

        self.history.append(
            {
                "tick": tick,
                "phase": _phase(obs),
                "row": row,
                "x": x,
                "screen_x": sx,
                "score": score,
                "deaths": deaths,
                "posted": posted,
                "event": event,
                "e7": e7,
            }
        )
        self.history = self.history[-12:]
        self.prior_obs = previous
        self.prev_obs = obs
        self.last_tick = tick
        self.last_row = row
        self.last_x = x
        self.last_screen_x = sx
        self.last_score = score
        self.last_deaths = deaths

    def _state(self, obs: dict[str, Any]) -> dict[str, Any]:
        # _update has already installed `obs` as prev_obs; prior_obs preserves
        # the frame against which this state was actually compared.
        previous = self.prior_obs
        row = _row(obs)
        sx = _screen_x(obs)
        x = _x(obs)
        prev_row = _row(previous) if previous else row
        prev_sx = _screen_x(previous) if previous else sx
        raw_delta: list[str] = []
        old_raw, new_raw = _raw(previous), _raw(obs)
        if old_raw and new_raw:
            for i, (a, b) in enumerate(zip(old_raw, new_raw)):
                if a != b:
                    raw_delta.append(f"{0x80 + i:02X}:{a:02X}>{b:02X}")
                    if len(raw_delta) >= 32:
                        break

        phase = _phase(obs)
        score = _number(obs.get("score", 0.0))
        previous_score = _number(previous.get("score", score)) if previous else score
        tick = _int(obs.get("tick", self.last_tick))
        motion = "stable"
        if sx >= 0 and prev_sx >= 0 and sx != prev_sx:
            motion = "right" if sx > prev_sx else "left"
        row_motion = "same"
        if row != prev_row:
            row_motion = "forward" if row not in (-1, 255) and (prev_row in (255, -1) or row > prev_row) else "reset_or_back"

        state = {
            "game": "Atari 2600 Frogger",
            "rules": {
                "board_order": "start, five road lanes/median, five river rows, home bays",
                "direction": "one joystick hop in standard mode; it is released after one frame",
                "NOOP": "one frame of waiting while traffic and platforms continue",
                "priority": "preserve a frog, then reach an unoccupied home; no bonus chasing",
            },
            "time": {
                "tick": tick,
                "frame": _int(obs.get("frame_number", obs.get("frame", tick))),
                "timer_s": _number(obs.get("timer_s", 0)),
                "ticks_since_progress": max(0, tick - self.last_progress_tick),
                "ticks_since_direction": max(0, tick - self.last_direction_tick),
            },
            "life": {
                "life_index": self.life,
                "deaths": _int(obs.get("deaths", self.last_deaths)),
                "reserve_frogs": _int(obs.get("reserve_frogs", -1), -1),
                "last_death_phase": self.last_death_phase,
                "last_death_row": self.last_death_row,
                "recovery_shift_pending": self.shift_pending and not self.shift_done,
                "recovery_shift_direction": self.shift_dir,
                "restore_center_pending": self.shift_done and not self.restore_done,
            },
            "frog": {
                "row_candidate": row,
                "phase": phase,
                "x_candidate": x,
                "screen_x_candidate": sx,
                "screen_motion": motion,
                "row_motion": row_motion,
                "edge_bucket": (
                    "left_edge" if 0 <= sx < 20 else
                    "right_edge" if sx > 108 else
                    "interior" if sx >= 0 else "unknown"
                ),
            },
            "progress": {
                "score": score,
                "score_delta": score - previous_score,
                "frogs_home": _int(obs.get("frogs_home", 0)),
                "level": _int(obs.get("level", 1), 1),
                "terminal": bool(obs.get("terminal", False)),
            },
            "verified_temporal_evidence": {
                "current_event": self.history[-1].get("event", "steady") if self.history else "opening",
                "hazard_e7": _byte(obs, 0xE7),
                "hazard_e7_rising": self.hazard_rising,
                "collision_e5": _byte(obs, 0xE5),
                "state_e3": _byte(obs, 0xE3),
                "home_counter_da": _byte(obs, 0xDA),
                "raw_delta_addresses": raw_delta,
                "meaning_note": (
                    "The row transition 255->0->1... and score changes after UP are observed. "
                    "A nonzero/rising E7 is only a temporal hazard cue, not a named object."
                ),
            },
            "recent": list(self.history[-8:]),
            "astra_advice": dict(self.advice),
            "last_posted_action": self.current_action,
            "control": {
                "fresh_answer_must_be_single_frame": True,
                "never_hold_direction": True,
                "unknown_state_fallback": "NOOP, except a fresh start must eventually advance",
            },
        }
        return _jsonable(state)

    def _questions(self, tick: int) -> dict[str, Any]:
        # Rotate criteria order: Jev's documented first-option tendency should
        # not permanently turn a safety question into NOOP or UP.
        orders = (
            ("UP", "NOOP", "RIGHT", "LEFT", "DOWN"),
            ("LEFT", "UP", "NOOP", "RIGHT", "DOWN"),
            ("RIGHT", "DOWN", "UP", "NOOP", "LEFT"),
            ("NOOP", "RIGHT", "LEFT", "UP", "DOWN"),
        )
        descriptions = {
            "NOOP": "Wait for exactly one next emulator frame; do not move the frog.",
            "UP": "Make one forward hop now, toward the next row/home; never hold it.",
            "RIGHT": "Make one lateral hop right only as a useful alignment or recovery step.",
            "LEFT": "Make one lateral hop left only as a useful alignment or recovery step.",
            "DOWN": "Make one backward hop only to recover from an immediate hazard.",
        }
        action_criteria = {
            name: descriptions[name]
            for name in orders[tick % len(orders)]
            if name in self.legal
        }
        tactic_orders = (
            ("ADVANCE", "WAIT", "SHIFT_LEFT", "SHIFT_RIGHT", "RETREAT"),
            ("SHIFT_RIGHT", "ADVANCE", "WAIT", "RETREAT", "SHIFT_LEFT"),
            ("WAIT", "SHIFT_LEFT", "ADVANCE", "RETREAT", "SHIFT_RIGHT"),
            ("RETREAT", "SHIFT_RIGHT", "SHIFT_LEFT", "WAIT", "ADVANCE"),
        )
        tactic_desc = {
            "ADVANCE": "The current verified phase supports one UP hop as the next useful move.",
            "WAIT": "Wait one frame because moving platforms/traffic need to change before a hop.",
            "SHIFT_LEFT": "A single LEFT hop is the best alignment/recovery step before advancing.",
            "SHIFT_RIGHT": "A single RIGHT hop is the best alignment/recovery step before advancing.",
            "RETREAT": "A single DOWN hop is safer than entering the immediate hazard.",
        }
        tactic_criteria = {
            name: tactic_desc[name] for name in tactic_orders[tick % len(tactic_orders)]
        }
        return {
            "action": Choice(
                "Using only this current temporal JSON, choose the literal action for the "
                "next single Atari frame. The code releases every direction after one frame. "
                "Use row/phase, the recent event, E7 rising, motion, and life recovery; do "
                "not treat an uncertain raw byte as a named car or platform. A fresh start "
                "must not wait indefinitely.",
                action_criteria,
            ),
            "tactic": Choice(
                "Choose the immediate Frogger posture, not a multi-hop plan. `ADVANCE` means "
                "one UP pulse; SHIFT means one lateral pulse; WAIT means only one frame. "
                "The state includes a recovery shift after a repeated same-column death; use "
                "that evidence rather than always selecting the first option.",
                tactic_criteria,
            ),
            "forward_safe": Noul(
                "Given `frog.phase`, `verified_temporal_evidence.hazard_e7_rising`, "
                "`frog.screen_motion`, and the recent event, is one UP hop safer than waiting "
                "or shifting for the next single frame? True means this one hop, not a plan."
            ),
        }

    def _candidate_from_answers(self, answers: Any) -> str | None:
        if not isinstance(answers, dict):
            return None
        direct = _valid_action(_answer_value(answers.get("action")), self.legal)
        tactic_value = _answer_value(answers.get("tactic"))
        tactic_action = _TACTIC_TO_ACTION.get(
            tactic_value.strip().upper() if isinstance(tactic_value, str) else ""
        )
        if tactic_action not in self.legal:
            tactic_action = None
        # The action question is closest to execution.  Tactic wins only when
        # it explicitly names a lateral/recovery posture; this makes lateral
        # advice actionable without allowing an untrusted prose plan.
        if tactic_action in ("LEFT", "RIGHT", "DOWN"):
            if _probability(answers.get("tactic"), tactic_value) >= 0.30:
                return tactic_action
        if direct is not None:
            return direct
        if tactic_action is not None:
            return tactic_action
        # The Noul is only a rescue when the typed Choices are absent or
        # malformed.  It cannot recreate the parent's strong-wait starvation
        # because a valid action Choice always takes precedence.
        safe_probability = _probability(answers.get("forward_safe"), "true")
        if safe_probability >= 0.65 and "UP" in self.legal:
            return "UP"
        if safe_probability <= 0.25 and self.noop in self.legal:
            return self.noop
        return None

    def _fallback_action(self, obs: dict[str, Any], candidate: str | None) -> str:
        """Deterministic safety/progress bridge when Jev is late or ambiguous."""
        if bool(obs.get("terminal", False)):
            return self.noop
        row = _row(obs)
        phase = _phase(obs)
        tick = _int(obs.get("tick", self.last_tick))

        # Verified emergency cue: the archived river failure had E7=0x80 for
        # the frame before the frog died.  Preserve one release frame after a
        # prior direction, then spend the next pulse on an escape instead of
        # allowing the debounce interval to erase the cue.
        if (
            phase == "RIVER"
            and self.hazard_escape_pending
            and not self.release_pending
            and "UP" in self.legal
        ):
            self.hazard_escape_pending = False
            return "UP"

        if self.cooldown > 0:
            return self.noop

        # Keep a moving frog away from a carrier edge.  This is a guard, not a
        # fixed seed route; x is read from the live candidate and only used in
        # the river phase.
        sx = _screen_x(obs)
        if phase == "RIVER" and sx >= 0:
            if sx < 16 and "RIGHT" in self.legal:
                return "RIGHT"
            if sx > 112 and "LEFT" in self.legal:
                return "LEFT"

        # After a death, change the road column once, then undo that one-step
        # offset at the median.  It breaks the parent's same-column collision
        # loop while preserving a central home approach.
        if self.shift_pending and not self.shift_done and row == 255:
            if self.shift_dir in self.legal:
                return self.shift_dir
        if self.shift_pending and not self.shift_done and phase == "ROAD_OR_MEDIAN" and row <= 1:
            if self.shift_dir in self.legal:
                return self.shift_dir
        if self.shift_done and not self.restore_done and row >= 5 and row <= 7:
            opposite = "LEFT" if self.shift_dir == "RIGHT" else "RIGHT"
            if opposite in self.legal:
                return opposite

        if candidate in self.legal:
            # A high-level WAIT answer is not allowed to consume a whole life
            # at the start or after a long no-progress interval.
            if candidate == self.noop:
                if phase == "START" or tick - self.last_progress_tick >= 22:
                    return "UP" if "UP" in self.legal else self.noop
            return candidate

        # Model-free fallback: make progress from a fresh sidewalk and after
        # an extended wait, but otherwise retain a legal wait.
        if phase == "START" and "UP" in self.legal:
            return "UP"
        if tick - self.last_progress_tick >= 22 and row != 255 and "UP" in self.legal:
            return "UP"
        return self.noop

    def _mark_posted(self, action: str, obs: dict[str, Any]) -> None:
        self.current_action = action
        if action in _DIRECTIONS:
            self.last_direction_tick = _int(obs.get("tick", self.last_tick))
            self.cooldown = max(self.cooldown, 5)
            if self.shift_pending and not self.shift_done and action == self.shift_dir:
                self.shift_done = True
            elif self.shift_done and not self.restore_done:
                opposite = "LEFT" if self.shift_dir == "RIGHT" else "RIGHT"
                if action == opposite:
                    self.restore_done = True
        elif self.cooldown > 0:
            self.cooldown -= 1

    def _schedule_jev(self, state: dict[str, Any], questions: dict[str, Any]) -> None:
        if self.pending_jev is not None or self.ctx.done:
            return
        asked_tick = _int(state.get("time", {}).get("tick", self.last_tick))
        asked_life = self.life

        def call() -> Any:
            return self.ctx.decide(state, questions, timeout_s=0.70)

        try:
            self.pending_jev = (self.ctx.spawn(call), asked_tick, asked_life)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError):
            self.pending_jev = None

    def _harvest_jev(self, obs: dict[str, Any]) -> str | None:
        pending = self.pending_jev
        if pending is None or not pending[0].done():
            return None
        self.pending_jev = None
        try:
            answers = pending[0].result()
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError):
            return None
        except Exception:
            return None
        asked_tick, asked_life = pending[1], pending[2]
        now = _int(obs.get("tick", self.last_tick))
        if asked_life != self.life or now - asked_tick > 16:
            return None
        return self._candidate_from_answers(answers)

    def _astra_prompt(self, game: Any, state: dict[str, Any], reason: str) -> str:
        try:
            description = _text(game.describe(), 1800)
        except Exception:
            description = "Atari 2600 Frogger with one-frame NOOP/UP/RIGHT/LEFT/DOWN actions."
        return (
            "You are an out-of-band System-2 supervisor for a real-time Atari 2600 Frogger "
            "controller. The fast loop observes every frame and releases every direction "
            "after one frame, so do not write an action sequence. Use only this live temporal "
            "snapshot and the public rules. Return at most 60 tokens as one JSON object with "
            "string fields focus, rule, lateral_rule. Mark RAM interpretations as hypotheses. "
            "Prefer a home and survival over score bonuses. This card is advisory to Jev; "
            "never assume it will be obeyed verbatim.\n"
            f"REASON={reason}\nGAME={description}\nSNAPSHOT={json.dumps(state, separators=(',', ':'), ensure_ascii=True)}"
        )

    def _start_astra(self, game: Any, state: dict[str, Any], reason: str) -> None:
        if self.pending_astra is not None or self.astra_calls >= 1 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        prompt = self._astra_prompt(game, state, reason)
        try:
            self.pending_astra = self.ctx.spawn(
                lambda p=prompt: self.ctx.think(p, timeout_s=40.0)
            )
            self.astra_calls += 1
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError):
            self.pending_astra = None

    def _poll_astra(self) -> None:
        future = self.pending_astra
        if future is None or not future.done():
            return
        self.pending_astra = None
        try:
            result = future.result()
            self.advice = _parse_advice(getattr(result, "text", result))
            self._log("astra_card_ready", focus=self.advice.get("focus", "unknown"))
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception:
            self.advice = dict(self.advice)

    def _cancel_background(self) -> None:
        for pending in (self.pending_jev, self.pending_astra):
            if pending is None:
                continue
            future = pending[0] if isinstance(pending, tuple) else pending
            try:
                future.cancel()
            except Exception:
                pass
        self.pending_jev = None
        self.pending_astra = None

    def run_episode(self, game: Any) -> None:
        # This is a tier-2 harness.  A harmless branch keeps malformed
        # pre-flight stand-ins from causing a contract error if tick_s is absent.
        if not getattr(game, "tick_s", None):
            try:
                if not self.ctx.done:
                    self.ctx.act(self.noop)
            except BudgetExceeded:
                return
            return

        try:
            spec = game.action_spec
            raw_names = getattr(spec, "names", _ACTIONS)
            self.legal = {str(name).upper() for name in raw_names}
            if not self.legal:
                self.legal = set(_ACTIONS)
            self.noop = str(getattr(spec, "noop", "NOOP")).upper()
            if self.noop not in self.legal:
                self.noop = "NOOP" if "NOOP" in self.legal else next(iter(self.legal))
        except Exception:
            self.legal = set(_ACTIONS)
            self.noop = "NOOP"

        initial = self._safe_observe(game)
        self.prev_obs = initial
        self.last_deaths = _int(initial.get("deaths", 0))
        self.last_score = _number(initial.get("score", 0.0))
        self.last_row = _row(initial)
        self.last_x = _x(initial)
        self.last_screen_x = _screen_x(initial)
        self.last_tick = _int(initial.get("tick", 0))
        self.last_progress_tick = self.last_tick
        self.current_action = self.noop

        try:
            self._start_astra(game, self._state(initial), "opening interface and route")
            # Explicit initial NOOP establishes the action slot before waiting.
            self.ctx.act(self.noop)
            self.current_action = self.noop

            while not self.ctx.done and not bool(getattr(game, "done", False)):
                # One live frame always passes; no spin loop while a model works.
                self.ctx.wait_for_tick()
                if self.ctx.done or bool(getattr(game, "done", False)):
                    break

                self._poll_astra()
                obs = self._safe_observe(game)
                self._update(obs, self.current_action)
                if bool(obs.get("terminal", False)):
                    break

                # Release a directional edge after exactly one consumed frame.
                self.release_pending = self.current_action in _DIRECTIONS
                if self.release_pending:
                    self.current_action = self.noop
                    self.cooldown = max(self.cooldown, 5)
                elif self.cooldown > 0:
                    self.cooldown -= 1

                # A death is a real interface event: the next Jev state gets
                # the recovery-column experiment and the new life id.
                if self.history and self.history[-1].get("event") == "death_reset":
                    self._log(
                        "life_reset",
                        life=self.life,
                        row=self.last_death_row,
                        phase=self.last_death_phase,
                    )

                candidate = self._harvest_jev(obs)
                state = self._state(obs)
                if self.pending_jev is None and not self.ctx.done:
                    self._schedule_jev(state, self._questions(self.last_tick))

                action = self._fallback_action(obs, candidate)
                if action not in self.legal:
                    action = self.noop
                if not self.ctx.done:
                    self.ctx.act(action)
                    self._mark_posted(action, obs)

        except (BudgetExceeded, ModelUnavailable):
            # Budget closure and model unavailability are normal evaluator
            # endings; no model exception should escape a late-frame cleanup.
            return
        finally:
            self._cancel_background()


def build(ctx: Any) -> Harness:
    return Harness(ctx)
