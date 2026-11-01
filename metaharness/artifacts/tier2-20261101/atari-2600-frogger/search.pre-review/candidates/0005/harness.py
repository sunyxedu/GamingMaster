"""Astra-supervised, one-frame Jev controller for real-time Frogger.

The important execution detail is that a directional answer is an *edge*: after
one tick the bridge posts NOOP, so a slow Jev answer cannot turn one hop into a
long held joystick input.  Astra is advisory and runs out of band; the game is
never stopped waiting for it.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


_ACTIONS = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN")
_DIRECTIONS = frozenset(("UP", "RIGHT", "LEFT", "DOWN"))


def _text(value: Any, limit: int = 700) -> str:
    """Turn arbitrary model/adaptor values into bounded, JSON-safe text."""
    if value is None:
        return ""
    try:
        result = str(value)
    except Exception:
        return ""
    return result[:limit]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _jsonable(value: Any) -> Any:
    """Recursively remove values that would make ctx.decide reject state."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else 0.0
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return _text(value, 180)


def _valid_action(value: Any, legal: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().upper()
    return candidate if candidate in legal else None


def _answer_value(answer: Any) -> Any:
    try:
        return answer.value
    except Exception:
        return None


def _answer_probability(answer: Any, key: str) -> float:
    try:
        probabilities = answer.probabilities
        return _number(probabilities.get(key, 0.0), 0.0)
    except Exception:
        return 0.0


def _answer_confidence(answer: Any) -> float:
    try:
        confidence = answer.confidence
        return _number(confidence, 0.0)
    except Exception:
        return 0.0


def _plan_from_text(text: Any, legal: set[str]) -> dict[str, Any]:
    """Extract only a tiny advisory interface from arbitrary Astra output."""
    raw = _text(text, 900).strip()
    parsed: Any = None
    if raw:
        candidates = [raw]
        if "```" in raw:
            candidates.append(raw.replace("```json", "").replace("```", "").strip())
        left, right = raw.find("{"), raw.rfind("}")
        if left >= 0 and right > left:
            candidates.append(raw[left : right + 1])
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    parsed = obj
                    break
            except Exception:
                continue

    def first_string(*keys: str) -> str:
        if not isinstance(parsed, dict):
            return ""
        for key in keys:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return _text(value, 260).strip()
        return ""

    preferred = first_string("preferred_action", "next_action", "action").upper()
    if preferred not in legal:
        preferred = ""
    goal = first_string("goal", "subgoal", "route")
    wait_rule = first_string("wait", "wait_rule", "cadence")
    avoid = parsed.get("avoid", parsed.get("constraints", "")) if isinstance(parsed, dict) else ""
    if isinstance(avoid, (list, tuple)):
        avoid = ", ".join(_text(item, 100) for item in avoid[:5])
    else:
        avoid = _text(avoid, 260)

    # A terse non-JSON response is still useful if it names one legal action.
    if not preferred and raw:
        match = re.search(r"\b(NOOP|UP|RIGHT|LEFT|DOWN)\b", raw.upper())
        if match:
            preferred = match.group(1)

    return {
        "preferred_action": preferred or "NOOP",
        "goal": goal or "preserve a frog, then make deliberate forward hops",
        "wait_rule": wait_rule or "wait when the current facts do not establish a safe landing",
        "avoid": avoid or "blind lateral hops, long held directions, and optional bonuses",
        "advice": raw[:500],
    }


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.legal: set[str] = set(_ACTIONS)
        self.noop = "NOOP"
        self.plan: dict[str, Any] = {
            "preferred_action": "NOOP",
            "goal": "preserve a frog, then make deliberate forward hops",
            "wait_rule": "wait when the current facts do not establish a safe landing",
            "avoid": "blind lateral hops, long held directions, and optional bonuses",
            "advice": "No supervisor answer yet.",
        }
        self.plan_version = 0
        self.pending = None
        self.supervisor_calls = 0
        self.last_obs: dict[str, Any] | None = None
        self.last_posted = "NOOP"
        self.history: list[dict[str, Any]] = []
        self.revision_requested_for: set[str] = set()

    def _log(self, message: str, **fields: Any) -> None:
        try:
            self.ctx.log(message, **fields)
        except Exception:
            # Logging must never turn a survivable model/backend event into a game error.
            pass

    def _observe(self, game: Any) -> dict[str, Any]:
        try:
            obs = game.observe()
        except Exception:
            return {}
        return obs if isinstance(obs, dict) else {}

    def _summary(self, obs: dict[str, Any], event: str = "") -> dict[str, Any]:
        ram_fields = obs.get("ram_fields")
        if not isinstance(ram_fields, dict):
            ram_fields = {}
        frog = obs.get("frog")
        if not isinstance(frog, dict):
            frog = {}
        raw_ram = obs.get("raw_ram")
        if not isinstance(raw_ram, list):
            raw_ram = []
        # Keep a few audit bytes, rather than sending an opaque 128-byte blob on
        # every decision.  The adaptor's named candidates remain explicit.
        ram_focus = {}
        for address in ("b0", "b4", "ac", "d0", "da", "d2", "e0", "e3", "e5", "e7"):
            if address in ram_fields:
                ram_focus[address] = _integer(ram_fields.get(address), 0)
        if not ram_focus and raw_ram:
            for address in (0xB0, 0xB4, 0xAC, 0xD0, 0xDA, 0xD2, 0xE0, 0xE3, 0xE5, 0xE7):
                index = address - 0x80
                if 0 <= index < len(raw_ram):
                    ram_focus[hex(address)] = _integer(raw_ram[index], 0)

        score = _number(obs.get("score"), 0.0)
        deaths = _integer(obs.get("deaths"), 0)
        prev = self.last_obs or {}
        prev_deaths = _integer(prev.get("deaths"), deaths)
        prev_score = _number(prev.get("score"), score)
        prev_level = _integer(prev.get("level"), _integer(obs.get("level"), 1))
        level = _integer(obs.get("level"), 1)
        if not event:
            if deaths > prev_deaths:
                event = "death_or_reset"
            elif level > prev_level:
                event = "level_change"
            elif score > prev_score:
                event = "score_progress"
            else:
                event = "steady"

        row = _integer(frog.get("row", ram_fields.get("frog_row_ac", -1)), -1)
        if row == 255:
            zone_hint = "start_or_reset_candidate"
        elif row < 0:
            zone_hint = "unknown"
        else:
            zone_hint = "advanced_candidate_row"

        summary = {
            "game": "Atari 2600 Frogger",
            "mode": _integer(obs.get("mode"), 0),
            "difficulty": _integer(obs.get("difficulty"), 0),
            "tick": _integer(obs.get("tick"), 0),
            "frame": _integer(obs.get("frame_number", obs.get("frame", 0)), 0),
            "frog": {
                "row_candidate": row,
                "x_candidate": _integer(frog.get("x", ram_fields.get("frog_x_b0", 0)), 0),
                "screen_x_candidate": _integer(frog.get("screen_x", ram_fields.get("frog_screen_x_b4", 0)), 0),
                "state_candidate": _integer(frog.get("state", ram_fields.get("state_e3", 0)), 0),
                "zone_hint": zone_hint,
            },
            "score": score,
            "frogs_home": _integer(obs.get("frogs_home"), 0),
            "deaths": deaths,
            "reserve_frogs": _integer(obs.get("reserve_frogs"), 0),
            "timer_s": _integer(obs.get("timer_s"), 0),
            "level": level,
            "terminal": bool(obs.get("terminal", False)),
            "decoder_confidence": _text(obs.get("decoder_confidence", "unknown"), 80),
            "field_legend": {
                "frog": "conservative RAM candidates; row 255 indicates a likely start/reset, other rows are not a full object map",
                "deaths": "frogs lost; an increase means the current frog ended and a reserve may have spawned",
                "timer_s": "seconds remaining for the current frog; preserve the frog before chasing bonuses",
                "ram_focus": "raw/hypothesis bytes for recognition, not guaranteed named cars or platforms",
                "supervisor": "slow advisory policy, checked against this observation and never an action sequence",
            },
            "ram_focus": ram_focus,
            "last_observed_action": _text(obs.get("last_action", self.last_posted), 20).upper(),
            "last_posted_action": self.last_posted,
            "event": event,
            "score_delta_since_summary": score - prev_score,
            "deaths_delta_since_summary": deaths - prev_deaths,
            "recent": list(self.history[-4:]),
            "supervisor": dict(self.plan),
            "supervisor_version": self.plan_version,
            "control_contract": (
                "The selected direction will be posted for exactly one next tick and then "
                "released to NOOP. Choose a direction only when it is useful now; do not "
                "assume a previous direction remains intentionally held."
            ),
        }
        return _jsonable(summary)

    def _supervisor_prompt(self, game: Any, summary: dict[str, Any], reason: str) -> str:
        try:
            description = _text(game.describe(), 3200)
        except Exception:
            description = "Atari 2600 Frogger: cross traffic and the river into an unoccupied home."
        try:
            actions = _text(game.action_spec.describe(), 1200)
        except Exception:
            actions = "NOOP, UP, RIGHT, LEFT, DOWN"
        previous = json.dumps(self.plan, separators=(",", ":"), ensure_ascii=True)
        current = json.dumps(summary, separators=(",", ":"), ensure_ascii=True)
        return (
            "You are the slow strategic supervisor for a real-time Atari 2600 Frogger "
            "controller. The fast controller calls Jev every decision and code releases a "
            "direction after one emulator frame; you must not assume a held joystick. "
            "Use only the supplied observation and public rules. Do not invent cars, logs, "
            "or pixels that are not present. Return one compact JSON object, no markdown and "
            "at most 100 output tokens, with string fields: goal, preferred_action (one of "
            "NOOP/UP/RIGHT/LEFT/DOWN), wait_rule, avoid. This is advice for Jev, not a long "
            "action sequence. Reconsider the previous advice after the named event.\n\n"
            f"Event requiring supervision: {reason}\n"
            f"Game: {description}\nActions: {actions}\n"
            f"Previous advice: {previous}\nCurrent normalized observation: {current}"
        )

    def _start_supervisor(self, game: Any, summary: dict[str, Any], reason: str) -> None:
        if getattr(self.ctx, "system2", None) is None:
            return
        if self.pending is not None or self.supervisor_calls >= 3 or self.ctx.done:
            return
        prompt = self._supervisor_prompt(game, summary, reason)
        try:
            self.pending = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=120.0)
            )
            self.supervisor_calls += 1
            self._log("supervisor_started", reason=reason, call=self.supervisor_calls)
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
            self.pending = None

    def _poll_supervisor(self) -> None:
        future = self.pending
        if future is None:
            return
        try:
            if not future.done():
                return
            result = future.result()
        except Exception as exc:
            self.pending = None
            self._log("supervisor_unavailable", error=_text(exc, 160))
            return
        self.pending = None
        text = getattr(result, "text", result)
        self.plan = _plan_from_text(text, self.legal)
        self.plan_version += 1
        self._log("supervisor_updated", version=self.plan_version,
                  preferred=self.plan.get("preferred_action", ""))

    def _fallback(self, summary: dict[str, Any]) -> str:
        if summary.get("terminal"):
            return self.noop
        event = _text(summary.get("event"), 40)
        frog = summary.get("frog", {})
        row = _integer(frog.get("row_candidate"), -1) if isinstance(frog, dict) else -1
        # A fresh frog is the one case where waiting is predictably unproductive.
        if row == 255 or event == "death_or_reset":
            return "UP" if "UP" in self.legal else self.noop
        # A supervisor recommendation is only a fallback, never an override of Jev.
        preferred = _valid_action(self.plan.get("preferred_action"), self.legal)
        if preferred is not None:
            return preferred
        return self.noop

    def _questions(self, tick: int) -> dict[str, Any]:
        # Rotate the primary option order so the first-option tendency cannot make
        # one action permanent.  The descriptions explicitly define a one-frame press.
        orders = (
            ("NOOP", "UP", "RIGHT", "LEFT", "DOWN"),
            ("UP", "NOOP", "LEFT", "RIGHT", "DOWN"),
            ("RIGHT", "UP", "NOOP", "DOWN", "LEFT"),
            ("LEFT", "NOOP", "UP", "DOWN", "RIGHT"),
        )
        order = [name for name in orders[tick % len(orders)] if name in self.legal]
        descriptions = {
            "NOOP": "Wait for exactly this next emulator frame; do not move the frog.",
            "UP": "Make one deliberate one-frame hop toward the river/home bays.",
            "RIGHT": "Make one deliberate one-frame hop to the right while retaining a safe landing.",
            "LEFT": "Make one deliberate one-frame hop to the left while retaining a safe landing.",
            "DOWN": "Make one deliberate one-frame hop toward the starting sidewalk for recovery.",
        }
        criteria = {name: descriptions.get(name, name) for name in order}
        return {
            "action": Choice(
                "Using only the current JSON state, choose the best literal action for the "
                "next single Atari frame. A direction is a one-frame press, not a held "
                "joystick. Prefer NOOP when the state does not establish a safe landing. "
                "Follow the supervisor only when its preconditions match the current frog "
                "candidate and event. Do not calculate or plan many hops.",
                criteria,
            ),
            "wait_is_safe": Noul(
                "Is waiting for this one next frame safer than making any directional "
                "hop, given the current frog candidate, event, timer, RAM focus, "
                "and supervisor constraints? True means one-frame waiting is safer, not that "
                "the controller should wait forever."
            ),
            "current_phase": Choice(
                "Which phase is best supported by the named current fields? Choose unknown "
                "if the decoder does not establish it.",
                {
                    "START": "frog.row_candidate is 255 or the event says death_or_reset.",
                    "ADVANCED": "frog.row_candidate is a non-255 candidate and the frog is alive.",
                    "TERMINAL": "terminal is true.",
                    "UNKNOWN": "The fields do not support a phase confidently.",
                },
            ),
        }

    def _choose(self, answers: Any, summary: dict[str, Any]) -> str:
        fallback = self._fallback(summary)
        if not isinstance(answers, dict):
            return fallback
        answer = answers.get("action")
        chosen = _valid_action(_answer_value(answer), self.legal)
        if chosen is None:
            return fallback

        # Jev's independent wait judgment is a safety veto only when it is strong
        # and the action answer itself is weak.  Otherwise movement remains possible.
        wait_answer = answers.get("wait_is_safe")
        wait_probability = _answer_probability(wait_answer, "true")
        action_probability = _answer_probability(answer, chosen)
        if chosen in _DIRECTIONS and wait_probability >= 0.78 and action_probability < 0.56:
            return self.noop
        if summary.get("terminal"):
            return self.noop
        # On a newly spawned frog, an unconfident NOOP would recreate the idle
        # baseline.  Permit the safe forward start unless Jev explicitly says
        # that even this one-frame wait is safer.
        frog = summary.get("frog", {})
        row = _integer(frog.get("row_candidate"), -1) if isinstance(frog, dict) else -1
        if chosen == self.noop and row == 255 and wait_probability < 0.78:
            return "UP" if "UP" in self.legal else self.noop
        # A malformed/unknown phase answer is not a reason to invent a lateral move.
        phase = _text(_answer_value(answers.get("current_phase")), 30).upper()
        if phase in {"TERMINAL", "UNKNOWN"} and chosen in ("LEFT", "RIGHT"):
            return self.noop
        return chosen

    def _decide(self, summary: dict[str, Any]) -> str:
        if getattr(self.ctx, "system1", None) is None:
            return self._fallback(summary)
        questions = self._questions(_integer(summary.get("tick"), 0))
        try:
            answers = self.ctx.decide(summary, questions, timeout_s=0.65)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError):
            self._log("jev_unavailable")
            return self._fallback(summary)
        return self._choose(answers, summary)

    def _post_edge(self, game: Any, action: str) -> bool:
        """Post action, wait for its one tick, and release a direction."""
        if action not in self.legal:
            action = self.noop
        try:
            self.ctx.act(action)
            self.last_posted = action
            self.ctx.wait_for_tick()
            if action in _DIRECTIONS and not self.ctx.done and not bool(getattr(game, "done", False)):
                self.ctx.act(self.noop)
                self.last_posted = self.noop
            return True
        except (BudgetExceeded, ModelUnavailable):
            return False

    def run_episode(self, game: Any) -> None:
        if not getattr(game, "tick_s", None):
            return
        try:
            self.legal = {str(name).upper() for name in game.action_spec.names}
            self.noop = _valid_action(getattr(game.action_spec, "noop", "NOOP"), self.legal) or "NOOP"
        except Exception:
            self.legal = set(_ACTIONS)
            self.noop = "NOOP"

        initial = self._observe(game)
        initial_summary = self._summary(initial, "opening")
        self._start_supervisor(game, initial_summary, "opening")
        self.last_obs = initial

        while not self.ctx.done and not bool(getattr(game, "done", False)):
            self._poll_supervisor()
            obs = self._observe(game)
            summary = self._summary(obs)
            event = _text(summary.get("event"), 40)
            if event in {"death_or_reset", "level_change"} and event not in self.revision_requested_for:
                self.revision_requested_for.add(event)
                self._start_supervisor(game, summary, event)

            try:
                action = self._decide(summary)
            except (BudgetExceeded, ModelUnavailable):
                break
            if action not in self.legal:
                action = self._fallback(summary)
            self._log("decision", action=action, tick=summary.get("tick"),
                      plan_version=self.plan_version)
            self.history.append({
                "tick": _integer(summary.get("tick"), 0),
                "event": event,
                "action": action,
                "score": _number(summary.get("score"), 0.0),
                "deaths": _integer(summary.get("deaths"), 0),
            })
            self.history = self.history[-8:]
            if not self._post_edge(game, action):
                break
            self.last_obs = self._observe(game)


def build(ctx: Any) -> Harness:
    return Harness(ctx)
