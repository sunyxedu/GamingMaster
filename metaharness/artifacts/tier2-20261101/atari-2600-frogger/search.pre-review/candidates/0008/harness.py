"""Phase-aware Frogger controller with action-specific Jev safety tests.

The controller deliberately does not ask Jev to generate one preferred action from
an opaque RAM dump.  It builds a small set of legal one-hop candidates and asks
independent typed questions about the safety/usefulness of each candidate.  Code
then combines those judgments with the verified forward-progress row counter and
runs a released one-hop bridge.  Astra is an asynchronous postmortem/world-model
writer; it never sits on the 60 Hz control path.
"""
from __future__ import annotations

import json
import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Noul


_ACTIONS = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN")
_DIRECTIONS = frozenset(("UP", "RIGHT", "LEFT", "DOWN"))
_JEV_TIMEOUT = 0.65
_ASTRA_TIMEOUT = 115.0
_MAX_ASTRA_CALLS = 2
_MAX_ANSWER_AGE = 28
_HISTORY_LIMIT = 18
_CARD_LIMIT = 1800


# The adaptor's candidate row is one of the few fields that is verified by the
# traces: 255 is the start/reset sentinel and 0,1,... advance after UP.  We do
# not pretend that it is a complete object decoder; it only selects which safety
# question deserves priority.
def _integer(value: Any, default: int = 0) -> int:
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
    """Keep every decide state strict JSON, including malformed stand-ins."""
    if depth > 6:
        return _text(value, 120)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in value]
    return _text(value, 180)


def _ram(obs: Any) -> list[int]:
    if not isinstance(obs, dict):
        return []
    raw = obs.get("raw_ram", [])
    if not isinstance(raw, (list, tuple)):
        return []
    result: list[int] = []
    for value in raw[:128]:
        result.append(_integer(value, 0) & 0xFF)
    return result


def _ram_field(obs: Any, address: str, default: int = 0) -> int:
    if isinstance(obs, dict):
        fields = obs.get("ram_fields", {})
        if isinstance(fields, dict):
            # The adaptor currently names these as e.g. frog_row_ac.  Accept
            # either the documented full name or an address-only key.
            aliases = {
                "AC": "frog_row_ac",
                "B0": "frog_x_b0",
                "B4": "frog_screen_x_b4",
                "D0": "reserve_d0",
                "DA": "home_counter_da",
                "D2": "level_counter_d2",
                "E0": "timer_e0",
                "E3": "state_e3",
                "E5": "collision_e5",
                "E7": "hazard_e7",
            }
            for key in (aliases.get(address.upper(), ""), address, address.lower()):
                if key in fields:
                    return _integer(fields.get(key), default)
        raw = _ram(obs)
        index = int(address, 16) - 0x80
        if 0 <= index < len(raw):
            return raw[index]
    return default


def _frog_value(obs: Any, key: str, default: int = 0) -> int:
    frog = obs.get("frog", {}) if isinstance(obs, dict) else {}
    if not isinstance(frog, dict):
        frog = {}
    aliases = {
        "row": ("row",),
        "x": ("x", "screen_x"),
        "screen_x": ("screen_x", "x"),
        "state": ("state",),
    }
    for name in aliases.get(key, (key,)):
        if name in frog:
            return _integer(frog.get(name), default)
    address = {"row": "AC", "x": "B0", "screen_x": "B4", "state": "E3"}.get(key)
    return _ram_field(obs, address, default) if address else default


def _observation_number(obs: Any, key: str, default: float = 0.0) -> float:
    if isinstance(obs, dict):
        return _number(obs.get(key), default)
    return default


def _observation_int(obs: Any, key: str, default: int = 0) -> int:
    if isinstance(obs, dict):
        return _integer(obs.get(key), default)
    return default


def _phase(obs: Any) -> str:
    if not isinstance(obs, dict):
        return "UNKNOWN"
    if bool(obs.get("terminal", False)):
        return "TERMINAL"
    row = _frog_value(obs, "row", -1)
    if row in (-1, 255):
        return "START"
    # These are route bands, not claimed screen object identities.  The first
    # five forward rows are traffic; the verified row counter then reaches the
    # median/river band where an unsupported UP is much more consequential.
    if row <= 4:
        return "ROAD"
    if row == 5:
        return "MEDIAN"
    if row <= 10:
        return "RIVER"
    return "HOME_APPROACH"


def _delta_bucket(before: int, after: int) -> str:
    if before == 255 and after != 255:
        return "spawned_or_started"
    if after == 255 and before != 255:
        return "reset_or_died"
    if after > before:
        return "forward_progress"
    if after < before:
        return "backward_or_reset"
    return "unchanged"


def _changed_bytes(previous: Any, current: Any) -> list[dict[str, Any]]:
    old = _ram(previous)
    new = _ram(current)
    if not old or not new:
        return []
    changed: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(old, new)):
        if a != b:
            diff = b - a
            changed.append({
                "address": f"{0x80 + i:02X}",
                "value_hex": f"{b:02X}",
                "motion": "increased" if diff > 0 else "decreased",
            })
    return changed[:64]


def _motion_view(previous: Any, current: Any) -> dict[str, Any]:
    """Expose temporal RAM evidence without calling numeric byte arithmetic."""
    old = _ram(previous)
    new = _ram(current)
    if not new:
        return {"changed_addresses": [], "known_fields": {}}
    # The 0x80-0xAB area is the dense moving work area in the supplied traces.
    # It is retained as labelled hex, while the named fields carry the semantic
    # candidates.  Jev is told not to treat these bytes as a decoder guarantee.
    dynamic = [f"{value:02X}" for value in new[:44]]
    return {
        "moving_ram_80_AB_hex": dynamic,
        "changed_addresses": _changed_bytes(previous, current),
        "known_fields": {
            "AC_row_candidate": _ram_field(current, "AC", 255),
            "B0_x_candidate": _ram_field(current, "B0", 0),
            "B4_screen_x_candidate": _ram_field(current, "B4", 0),
            "D0_reserve_raw": _ram_field(current, "D0", 0),
            "DA_home_counter_candidate": _ram_field(current, "DA", 0),
            "D2_level_candidate": _ram_field(current, "D2", 1),
            "E0_timer_candidate": _ram_field(current, "E0", 0),
            "E3_state_candidate": _ram_field(current, "E3", 0),
            "E5_collision_candidate": _ram_field(current, "E5", 0),
            "E7_hazard_candidate": _ram_field(current, "E7", 0),
        },
        "note": (
            "Hex RAM is temporal evidence only. AC/score/life/timer fields are "
            "named candidates; no byte is asserted to be a car or platform list."
        ),
    }


def _safe_action(value: Any, legal: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().upper()
    return candidate if candidate in legal else None


def _answer_probability(answer: Any) -> float:
    """Read a Noul defensively; garbage stand-ins become an undecided answer."""
    try:
        value = getattr(answer, "value", None)
        number = float(value)
        if math.isfinite(number) and 0.0 <= number <= 1.0:
            return number
    except (TypeError, ValueError):
        pass
    try:
        probabilities = getattr(answer, "probabilities", None)
        if isinstance(probabilities, dict):
            number = float(probabilities.get("true", 0.0))
            return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        pass
    return 0.0


def _answer_has_true(answer: Any) -> bool:
    return _answer_probability(answer) >= 0.5


def _candidate_questions(phase: str, card: str, preferred_side: str) -> dict[str, Any]:
    side_hint = preferred_side if preferred_side in ("LEFT", "RIGHT") else "either side"
    common = (
        "Judge exactly one next Atari frame, not a multi-hop plan. The controller "
        "will release every direction after one applied frame; NOOP waits while "
        "traffic and river objects continue. `true` means this candidate is both "
        "reasonably safe and useful now. If the RAM evidence does not establish "
        "the landing, answer false rather than hallucinating a car/platform. "
        f"Current route band is {phase}; an Astra card is advisory and may be stale: {card}."
    )
    return {
        "up_safe": Noul(
            common +
            " Is a single UP hop safe and useful for the next forward row? In ROAD, "
            "this specifically means the next lane landing is likely not occupied; "
            "in RIVER it means the next row has support."
        ),
        "left_safe": Noul(
            common +
            f" Is a single LEFT hop safe and useful for horizontal alignment? Prefer "
            f"it only when it avoids a threat or finds support; the code's scan side is {side_hint}."
        ),
        "right_safe": Noul(
            common +
            f" Is a single RIGHT hop safe and useful for horizontal alignment? Prefer "
            f"it only when it avoids a threat or finds support; the code's scan side is {side_hint}."
        ),
        "down_safe": Noul(
            common +
            " Is a single DOWN hop the safer recovery from an immediate hazard? "
            "False when it merely throws away a safe forward position."
        ),
        "wait_safe": Noul(
            common +
            " Is one frame of NOOP safer than every directional candidate while "
            "the moving gap or platform phase changes? This is a one-frame wait, "
            "not permission to idle indefinitely."
        ),
    }


def _extract_plan(text: Any) -> str:
    """Astra is advisory prose; never execute or eval its output."""
    result = _text(text, _CARD_LIMIT).strip()
    if not result:
        return "No usable Astra card. Treat unknown object identity as unsafe."
    # If Astra emitted a huge JSON/markdown answer, preserve only text.  This
    # avoids making the Jev state pathological and makes arbitrary stand-ins safe.
    return result


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.legal: set[str] = set(_ACTIONS)
        self.noop = "NOOP"
        self.card = "No usable Astra card. Treat unknown object identity as unsafe."
        self.card_version = 0
        self.astra_calls = 0
        self.pending_astra: Any = None
        self.astra_session: str | None = None
        self.last_obs: Any = None
        self.last_posted = "NOOP"
        self.last_pulse = "NOOP"
        self.pulse_cooldown = 0
        self.last_pulse_tick = -1
        self.pending_jev: Any = None
        self.pending_jev_tick = -1
        self.pending_jev_deaths = -1
        self.history: list[dict[str, Any]] = []
        self.scan_side = "RIGHT"
        self.river_stall = 0
        self.last_row = 255
        self.last_deaths = 0
        self.last_level = 1

    def _log(self, event: str, **fields: Any) -> None:
        try:
            if not self.ctx.done:
                self.ctx.log(event, **fields)
        except Exception:
            pass

    def _snapshot(self, obs: Any, previous: Any, action_consumed: str) -> dict[str, Any]:
        row = _frog_value(obs, "row", -1)
        deaths = _observation_int(obs, "deaths", 0)
        score = _observation_number(obs, "score", 0.0)
        old_row = _frog_value(previous, "row", row) if previous is not None else row
        old_deaths = _observation_int(previous, "deaths", deaths) if previous is not None else deaths
        phase = _phase(obs)
        row_change = _delta_bucket(old_row, row)
        if deaths > old_deaths:
            row_change = "death_reset"
        x = _frog_value(obs, "x", 0)
        sx = _frog_value(obs, "screen_x", x)
        snap = {
            "tick": _observation_int(obs, "tick", _observation_int(obs, "frame_number", 0)),
            "frame": _observation_int(obs, "frame_number", _observation_int(obs, "frame", 0)),
            "phase": phase,
            "frog": {
                "row_candidate": row,
                "x_candidate": x,
                "screen_x_candidate": sx,
                "state_candidate": _frog_value(obs, "state", 0),
            },
            "score": score,
            "score_delta": _observation_number(obs, "score_delta", 0.0),
            "deaths": deaths,
            "reserve_frogs": _observation_int(obs, "reserve_frogs", -1),
            "timer_s": _observation_number(obs, "timer_s", 0.0),
            "level": _observation_int(obs, "level", 1),
            "terminal": bool(obs.get("terminal", False)) if isinstance(obs, dict) else False,
            "terminal_reason": obs.get("terminal_reason") if isinstance(obs, dict) else None,
            "last_action_from_game": _text(obs.get("last_action", ""), 20) if isinstance(obs, dict) else "",
            "action_consumed": action_consumed,
            "last_pulse": self.last_pulse,
            "pulse_cooldown": self.pulse_cooldown,
            "row_transition": row_change,
            "raw_motion": _motion_view(previous, obs),
            "astra_card_version": self.card_version,
            "astra_card": self.card,
            "recent_transitions": self.history[-_HISTORY_LIMIT:],
            "control_rule": (
                "A direction is a one-hop edge. It is never held: after a directional "
                "tick code inserts NOOP cooldown frames. Choose based on the candidate "
                "questions, not on a repeated previous direction."
            ),
        }
        return _jsonable(snap)

    def _start_astra(self, game: Any, obs: Any, reason: str) -> None:
        if self.astra_calls >= _MAX_ASTRA_CALLS or self.pending_astra is not None:
            return
        if self.ctx.done:
            return
        # Keep the slow call out of the control path.  The initial and postmortem
        # calls have different evidence, which is the point of the revision hook.
        snap = self._snapshot(obs, self.last_obs, self.last_posted)
        try:
            description = _text(game.describe(), 2800)
        except Exception:
            description = "Atari 2600 Frogger: cross traffic and moving river platforms into a home."
        prompt = (
            "You are Astra, the slow route analyst for a live Atari 2600 Frogger "
            "controller. Do not act and do not output code. Write a compact policy "
            "card (under 180 words) for a fast typed judge. The judge receives five "
            "independent Noul questions: whether UP, LEFT, RIGHT, DOWN, or one-frame "
            "NOOP is safe and useful. Code releases joystick directions after one "
            "frame. Use the verified AC row candidate only as a route band (255 is "
            "start/reset); treat other RAM bytes as evidence, not named objects. "
            "Explain how to recognize a road gap, a supported river landing, and a "
            "home approach from the supplied temporal evidence. Prefer survival and "
            "a home over bonuses. Mark uncertainty explicitly. Do not give a fixed "
            "seed route. This is a revision after the named event, so correct the "
            "old card if the transition history contradicts it.\n\n"
            f"EVENT: {reason}\nGAME: {description}\n"
            "JEV CONTRACT: questions are independent, typed, and judged from the "
            "current JSON only; code combines their probabilities and uses a safe "
            "fallback.\nOLD_CARD: " + self.card[-1200:] +
            "\nCURRENT_SNAPSHOT_JSON: " + json.dumps(snap, separators=(",", ":"), ensure_ascii=True) +
            "\nRECENT_TRANSITIONS_JSON: " + json.dumps(self.history[-_HISTORY_LIMIT:], separators=(",", ":"), ensure_ascii=True)
        )
        session = self.astra_session
        try:
            if session:
                self.pending_astra = self.ctx.spawn(
                    lambda p=prompt, s=session: self.ctx.think(
                        p, session_id=s, timeout_s=_ASTRA_TIMEOUT
                    )
                )
            else:
                self.pending_astra = self.ctx.spawn(
                    lambda p=prompt: self.ctx.think(p, timeout_s=_ASTRA_TIMEOUT)
                )
            self.astra_calls += 1
            self._log("astra_started", reason=reason, call=self.astra_calls)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self._log("astra_start_failed", error=type(exc).__name__)
            self.pending_astra = None

    def _poll_astra(self) -> None:
        future = self.pending_astra
        if future is None:
            return
        try:
            if not future.done():
                return
            result = future.result()
            text = getattr(result, "text", result)
            card = _extract_plan(text)
            if card:
                self.card = card
                self.card_version += 1
                sid = getattr(result, "session_id", None)
                if isinstance(sid, str) and sid:
                    self.astra_session = sid
                self._log("astra_card_ready", version=self.card_version, chars=len(card))
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError) as exc:
            self._log("astra_card_failed", error=type(exc).__name__)
        except Exception as exc:
            # Pre-flight stand-ins and malformed text are advisory failures only.
            self._log("astra_card_unusable", error=type(exc).__name__)
        finally:
            self.pending_astra = None

    def _start_jev(self, game: Any, snapshot: dict[str, Any]) -> None:
        if self.pending_jev is not None or self.ctx.done:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        phase = _text(snapshot.get("phase"), 30).upper()
        state = _jsonable(snapshot)
        questions = _candidate_questions(phase, self.card, self.scan_side)
        tick = _integer(snapshot.get("tick"), 0)
        deaths = _integer(snapshot.get("deaths"), 0)
        try:
            self.pending_jev = self.ctx.spawn(
                lambda s=state, q=questions: self.ctx.decide(
                    s, q, timeout_s=_JEV_TIMEOUT
                )
            )
            self.pending_jev_tick = tick
            self.pending_jev_deaths = deaths
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.pending_jev = None
            self._log("jev_start_failed", error=type(exc).__name__)

    def _fallback(self, snapshot: dict[str, Any]) -> str:
        """Progress fallback used for latency, errors, or arbitrary pre-flight answers."""
        if snapshot.get("terminal"):
            return self.noop
        phase = _text(snapshot.get("phase"), 30).upper()
        row = _integer(snapshot.get("frog", {}).get("row_candidate"), -1)
        if phase == "START" or row == 255:
            return "UP" if "UP" in self.legal else self.noop
        if phase in ("ROAD", "MEDIAN"):
            return "UP" if "UP" in self.legal else self.noop
        if phase == "RIVER":
            # A side probe is safer than blindly repeating UP when river support
            # is unknown.  Alternate sides so this is a bounded search, not drift.
            return self.scan_side if self.scan_side in self.legal else self.noop
        if phase == "HOME_APPROACH":
            return "UP" if "UP" in self.legal else self.noop
        return self.noop

    def _choose_from_answers(self, answers: Any, snapshot: dict[str, Any]) -> str:
        fallback = self._fallback(snapshot)
        if not isinstance(answers, dict):
            return fallback
        probabilities = {
            "UP": _answer_probability(answers.get("up_safe")),
            "LEFT": _answer_probability(answers.get("left_safe")),
            "RIGHT": _answer_probability(answers.get("right_safe")),
            "DOWN": _answer_probability(answers.get("down_safe")),
            "NOOP": _answer_probability(answers.get("wait_safe")),
        }
        for key in list(probabilities):
            if key not in self.legal:
                probabilities[key] = 0.0
        phase = _text(snapshot.get("phase"), 30).upper()
        if phase == "TERMINAL":
            return self.noop

        # The bridge has a structural preference, rather than trusting a single
        # free-form top choice: forward movement is valuable on road, while the
        # river must first compare horizontal support candidates.  Thresholds are
        # deliberately modest because an uncertain answer falls back to a pulse,
        # not to a held direction.
        wait = probabilities["NOOP"]
        directional = [name for name in ("UP", "LEFT", "RIGHT", "DOWN") if name in self.legal]
        best = max(directional, key=lambda name: probabilities[name], default=self.noop)
        best_p = probabilities.get(best, 0.0)

        if phase in ("START", "ROAD", "MEDIAN"):
            # Lateral corrections are accepted only when Jev distinguishes them
            # from both waiting and the forward landing.  This prevents the model
            # from turning an arbitrary road gap into a sideways car collision.
            if best in ("LEFT", "RIGHT") and best_p >= 0.56 and best_p >= probabilities["UP"] + 0.05:
                return best
            if probabilities["UP"] >= 0.43 and probabilities["UP"] >= wait + 0.02:
                return "UP" if "UP" in self.legal else self.noop
            return self.noop

        if phase == "RIVER":
            side_best = max(("LEFT", "RIGHT"), key=lambda name: probabilities[name], default=self.noop)
            side_p = probabilities.get(side_best, 0.0)
            # A lateral answer can be useful even when UP is the model's generic
            # prior.  Require it to beat waiting and have a real margin.
            if side_best in self.legal and side_p >= 0.46 and side_p >= wait + 0.03 and side_p >= probabilities["UP"] + 0.02:
                return side_best
            if probabilities["UP"] >= 0.46 and probabilities["UP"] >= wait + 0.02:
                return "UP" if "UP" in self.legal else self.noop
            # If all candidates are ambiguous, periodically use the deterministic
            # alternating probe.  It prevents an entire life being consumed by
            # NOOP while preserving the one-hop release invariant.
            if self.river_stall >= 24 and self.scan_side in self.legal:
                return self.scan_side
            return self.noop

        if phase == "HOME_APPROACH":
            if probabilities["UP"] >= 0.40 and probabilities["UP"] >= wait:
                return "UP" if "UP" in self.legal else self.noop
            if best in ("LEFT", "RIGHT") and best_p >= 0.52 and best_p >= wait + 0.03:
                return best
            return self.noop
        return fallback

    def _harvest_jev(self, snapshot: dict[str, Any]) -> str | None:
        future = self.pending_jev
        if future is None or not future.done():
            return None
        self.pending_jev = None
        try:
            answers = future.result()
            tick = _integer(snapshot.get("tick"), 0)
            deaths = _integer(snapshot.get("deaths"), 0)
            if tick - self.pending_jev_tick > _MAX_ANSWER_AGE or deaths != self.pending_jev_deaths:
                self._log("jev_answer_dropped", age=tick - self.pending_jev_tick)
                return None
            return self._choose_from_answers(answers, snapshot)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError) as exc:
            self._log("jev_failed", error=type(exc).__name__)
            return None
        except Exception as exc:
            self._log("jev_unusable", error=type(exc).__name__)
            return None

    def _post_next(self, action: str) -> bool:
        if action not in self.legal:
            action = self.noop
        try:
            self.ctx.act(action)
            self.last_posted = action
            return True
        except BudgetExceeded:
            return False

    def _record_transition(self, snapshot: dict[str, Any]) -> None:
        frog = snapshot.get("frog", {})
        self.history.append({
            "tick": _integer(snapshot.get("tick"), 0),
            "phase": _text(snapshot.get("phase"), 30),
            "action_consumed": _text(snapshot.get("action_consumed"), 20),
            "row": _integer(frog.get("row_candidate"), -1) if isinstance(frog, dict) else -1,
            "x": _integer(frog.get("x_candidate"), 0) if isinstance(frog, dict) else 0,
            "row_transition": _text(snapshot.get("row_transition"), 30),
            "score": _number(snapshot.get("score"), 0.0),
            "deaths": _integer(snapshot.get("deaths"), 0),
            "timer_s": _number(snapshot.get("timer_s"), 0.0),
        })
        if len(self.history) > _HISTORY_LIMIT:
            del self.history[:-_HISTORY_LIMIT]

    def run_episode(self, game: Any) -> None:
        try:
            tick_s = getattr(game, "tick_s", None)
            if not tick_s:
                # This is a tier-2 harness; a benign single action keeps accidental
                # pre-flight reuse legal without inventing a turn-game policy.
                if not self.ctx.done:
                    try:
                        self.ctx.act(self.noop)
                    except BudgetExceeded:
                        pass
                return
            names = getattr(game.action_spec, "names", _ACTIONS)
            self.legal = {str(name).upper() for name in names} or set(_ACTIONS)
            requested_noop = str(getattr(game.action_spec, "noop", "NOOP")).upper()
            self.noop = requested_noop if requested_noop in self.legal else ("NOOP" if "NOOP" in self.legal else next(iter(self.legal)))
        except Exception:
            self.legal = set(_ACTIONS)
            self.noop = "NOOP"

        self.last_obs = game.observe()
        self.last_row = _frog_value(self.last_obs, "row", 255)
        self.last_deaths = _observation_int(self.last_obs, "deaths", 0)
        self.last_level = _observation_int(self.last_obs, "level", 1)
        try:
            self._start_astra(game, self.last_obs, "opening calibration")
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self._log("astra_opening_skipped", error=type(exc).__name__)

        # Make the implicit initial NOOP explicit before the first live tick.
        if not self._post_next(self.noop):
            return

        try:
            while not self.ctx.done and not bool(getattr(game, "done", False)):
                self.ctx.wait_for_tick()
                if self.ctx.done or bool(getattr(game, "done", False)):
                    break
                latest = game.observe()
                consumed = self.last_posted
                snapshot = self._snapshot(latest, self.last_obs, consumed)
                self._record_transition(snapshot)
                self._poll_astra()

                deaths = _integer(snapshot.get("deaths"), 0)
                level = _integer(snapshot.get("level"), 1)
                phase = _text(snapshot.get("phase"), 30).upper()
                row = _integer(snapshot.get("frog", {}).get("row_candidate"), -1)
                if deaths > self.last_deaths:
                    self.river_stall = 0
                    self.scan_side = "RIGHT" if self.scan_side == "LEFT" else "LEFT"
                    # The first failure is the useful postmortem boundary; a
                    # second call is used only for a later failure/level shift.
                    if self.pending_astra is None and self.astra_calls < _MAX_ASTRA_CALLS:
                        self._start_astra(game, latest, "a life was lost; revise the landing policy")
                elif level > self.last_level:
                    self.river_stall = 0
                    if self.pending_astra is None and self.astra_calls < _MAX_ASTRA_CALLS:
                        self._start_astra(game, latest, "the level increased; re-estimate hazards and support")
                elif phase == "RIVER":
                    if row > self.last_row:
                        self.river_stall = 0
                    else:
                        self.river_stall += 1
                else:
                    self.river_stall = 0
                self.last_deaths = deaths
                self.last_level = level
                self.last_row = row

                # A direction consumed on the just-finished frame is now released.
                # Two further NOOP frames make the edge semantics explicit even if
                # a decision future completes immediately.
                if consumed in _DIRECTIONS:
                    self.pulse_cooldown = 2
                    self.last_pulse = consumed
                    self.scan_side = "LEFT" if consumed == "RIGHT" else "RIGHT" if consumed == "LEFT" else self.scan_side
                elif self.pulse_cooldown > 0:
                    self.pulse_cooldown -= 1

                chosen: str | None = None
                if self.pulse_cooldown <= 0:
                    chosen = self._harvest_jev(snapshot)
                else:
                    # Do not consume a completed answer while a release edge is
                    # active; the answer remains fresh for the next safe slot.
                    chosen = None

                if chosen is None and self.pulse_cooldown <= 0:
                    # A pending Jev call gets a real-time-safe NOOP fallback.  A
                    # blind phase pulse is allowed only after a long gap, so the
                    # first 0.2 s of model latency cannot turn the road policy
                    # into rapid repeated UP hops.
                    tick = _integer(snapshot.get("tick"), 0)
                    pulse_gap = tick - self.last_pulse_tick
                    if self.last_pulse_tick < 0 or pulse_gap >= 18:
                        chosen = self._fallback(snapshot)
                    else:
                        chosen = self.noop

                if self.pulse_cooldown > 0 or chosen is None:
                    chosen = self.noop

                # Schedule the next typed judgment after harvesting the previous
                # one.  It sees this frame's result, action edge, and RAM motion.
                if self.pending_jev is None and not self.ctx.done:
                    self._start_jev(game, snapshot)

                if chosen in _DIRECTIONS:
                    self.last_pulse_tick = _integer(snapshot.get("tick"), 0)
                    self.river_stall = 0 if chosen == "UP" else self.river_stall
                self._log("decision", action=chosen, phase=phase,
                          tick=_integer(snapshot.get("tick"), 0),
                          card_version=self.card_version,
                          cooldown=self.pulse_cooldown)
                if not self._post_next(chosen):
                    break
                self.last_obs = latest
        finally:
            for future in (self.pending_jev, self.pending_astra):
                if future is not None:
                    try:
                        future.cancel()
                    except Exception:
                        pass


def build(ctx: Any) -> Harness:
    return Harness(ctx)
