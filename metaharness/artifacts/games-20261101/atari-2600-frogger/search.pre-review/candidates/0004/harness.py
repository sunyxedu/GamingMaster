"""Astra-authored phase policy with a pipelined, pulse-based Jev controller.

The game is real time.  The main thread owns the clock and posts a legal action
for every tick; model calls are deliberately moved off that thread.  Astra
writes a short policy card, while Jev judges the current one-frame choice from a
small temporal state.  Code owns freshness, debounce, and all fallbacks.
"""
from __future__ import annotations

import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul, Score


_ACTIONS = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN")
_DIRECTIONS = frozenset(("UP", "RIGHT", "LEFT", "DOWN"))
_MAX_HISTORY = 8
_MAX_PLAN_CHARS = 3500
_JEV_TIMEOUT = 1.5
_ASTRA_TIMEOUT = 100.0
_MAX_ASTRA_CALLS = 3
_MAX_ANSWER_AGE_TICKS = 18


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_value(value: Any, depth: int = 0) -> Any:
    """Make arbitrary adaptor values safe for ctx.decide's strict JSON check."""
    if depth > 5:
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_value(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, depth + 1) for v in value]
    return str(value)


def _raw_ram(obs: Any) -> list[int]:
    if not isinstance(obs, dict):
        return []
    raw = obs.get("raw_ram", [])
    if not isinstance(raw, (list, tuple)):
        return []
    result: list[int] = []
    for value in raw[:128]:
        result.append(_int(value, 0) & 0xFF)
    return result


def _field(obs: Any, name: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, default)
    return default


def _frog(obs: Any) -> dict[str, Any]:
    frog = _field(obs, "frog", {})
    return frog if isinstance(frog, dict) else {}


def _ram_fields(obs: Any) -> dict[str, Any]:
    fields = _field(obs, "ram_fields", {})
    return fields if isinstance(fields, dict) else {}


def _frog_row(obs: Any) -> int:
    frog = _frog(obs)
    fields = _ram_fields(obs)
    return _int(frog.get("row", fields.get("frog_row_ac", -1)), -1)


def _frog_x(obs: Any) -> int:
    frog = _frog(obs)
    fields = _ram_fields(obs)
    return _int(frog.get("screen_x", frog.get("x", fields.get("frog_screen_x_b4", -1))), -1)


def _zone(row: int, terminal: bool = False) -> str:
    if terminal:
        return "TERMINAL"
    if row in (-1, 255):
        return "START_OR_RESPAWN"
    # The adaptor's row candidate is deliberately only a hypothesis.  These
    # broad labels keep Jev from treating an uncertain byte as exact geometry.
    if row <= 1:
        return "LOW_ROAD_OR_FIRST_HOP"
    if row <= 5:
        return "ROAD_OR_MEDIAN"
    if row <= 10:
        return "RIVER_OR_HOME_APPROACH"
    return "UNKNOWN_ACTIVE_ROW"


def _ram_delta(old: list[int], new: list[int]) -> list[str]:
    if not old or not new:
        return []
    changed: list[str] = []
    for index, (before, after) in enumerate(zip(old, new)):
        if before != after:
            # Use the 6502 address rather than an unexplained list index.
            changed.append(f"{0x80 + index:02X}:{before:02X}->{after:02X}")
    return changed[:48]


def _candidate_fields(obs: Any) -> dict[str, Any]:
    """Keep any adaptor-provided semantic candidates without assuming names."""
    if not isinstance(obs, dict):
        return {}
    wanted = (
        "road", "roads", "cars", "traffic", "river", "platforms",
        "river_platforms", "home", "home_bays", "hazards", "candidates",
        "timer", "level", "terminal_reason", "invalid_action",
    )
    result: dict[str, Any] = {}
    for key in wanted:
        if key in obs:
            result[key] = _json_value(obs[key])
    return result


def _compact_observation(obs: Any, previous: Any = None) -> dict[str, Any]:
    """Build the state Jev needs; raw RAM remains available as temporal evidence."""
    if not isinstance(obs, dict):
        return {"observation_text": str(obs), "zone": "UNKNOWN"}
    raw = _raw_ram(obs)
    old_raw = _raw_ram(previous)
    frog = _frog(obs)
    fields = _ram_fields(obs)
    terminal = bool(obs.get("terminal", False))
    row = _frog_row(obs)
    compact = {
        "tick": _int(obs.get("tick", obs.get("episode_frame_number", 0))),
        "frame": _int(obs.get("frame", obs.get("frame_number", 0))),
        "mode": _int(obs.get("mode", 0)),
        "difficulty": _int(obs.get("difficulty", 0)),
        "seed_phase_frames": _int(obs.get("seed_phase_frames", 0)),
        "score": _finite_number(obs.get("score", 0)),
        "score_delta": _finite_number(obs.get("score_delta", 0)),
        "deaths": _int(obs.get("deaths", 0)),
        "reserve_frogs": _int(obs.get("reserve_frogs", -1), -1),
        "timer_s": _finite_number(obs.get("timer_s", obs.get("timer", 0))),
        "level": _int(obs.get("level", 1), 1),
        "terminal": terminal,
        "terminal_reason": obs.get("terminal_reason"),
        "frog": {
            "x": _frog_x(obs),
            "row": row,
            "state": _int(frog.get("state", fields.get("state_e3", -1)), -1),
        },
        "zone": _zone(row, terminal),
        "ram_fields": _json_value(fields),
        "ram_changed_since_previous": _ram_delta(old_raw, raw),
        "candidate_fields": _candidate_fields(obs),
    }
    return _json_value(compact)


def _planner_observation(obs: Any, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Astra gets the complete RAM once, but with an explicit uncertainty label."""
    if not isinstance(obs, dict):
        return {"observation": str(obs), "recent": _json_value(history[-_MAX_HISTORY:])}
    result = _compact_observation(obs)
    raw = _raw_ram(obs)
    result["raw_ram_80_ff"] = raw
    result["raw_ram_note"] = (
        "raw_ram is the exact $80-$FF image; ram_fields and frog row are "
        "conservative adaptor hypotheses, not guaranteed object labels"
    )
    result["recent"] = _json_value(history[-_MAX_HISTORY:])
    return _json_value(result)


def _answer_value(answer: Any) -> Any:
    if hasattr(answer, "value"):
        return getattr(answer, "value")
    if isinstance(answer, dict):
        if "value" in answer:
            return answer.get("value")
        if "choice" in answer:
            return answer.get("choice")
    return None


def _probability(answer: Any, option: str) -> float:
    probabilities = getattr(answer, "probabilities", None)
    if probabilities is None and isinstance(answer, dict):
        probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        return 0.0
    return _finite_number(probabilities.get(option, 0.0), 0.0)


def _safe_action(value: Any, legal: set[str], noop: str) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    return value if value in legal else None


def _make_questions(game: Any, state: dict[str, Any]) -> dict[str, Any]:
    spec = game.action_spec
    names = [str(name) for name in getattr(spec, "names", _ACTIONS)]
    legal = [name for name in _ACTIONS if name in names]
    if not legal:
        legal = names or list(_ACTIONS)
    descriptions = getattr(spec, "descriptions", {})
    if not isinstance(descriptions, dict):
        descriptions = {}
    criteria_a = {
        name: str(descriptions.get(name, name)) for name in
        ("UP", "LEFT", "RIGHT", "DOWN", "NOOP") if name in legal
    }
    criteria_b = {
        name: str(descriptions.get(name, name)) for name in
        ("NOOP", "DOWN", "RIGHT", "LEFT", "UP") if name in legal
    }
    common = (
        "The state is a current Frogger snapshot, not a plan for many seconds. "
        "Choose the action for the next single Atari frame. In modes 0/1 a "
        "direction is one hop; do not repeat it merely because last_action was "
        "the same. NOOP is a deliberate wait, not a missing answer. Use the "
        "planner_card only as a constraint and use the temporal evidence in "
        "state. If object identity is uncertain, preserve the frog rather than "
        "inventing a car or platform. "
    )
    return {
        "action_primary": Choice(
            common + "Which legal action best serves the current subgoal?",
            criteria_a,
        ),
        "action_check": Choice(
            common + "Independently select the safest useful action; agree only "
            "when the current state supports it.",
            criteria_b,
        ),
        "risk": Score(
            "Using state.ram_fields, state.candidate_fields, state.zone, and "
            "state.ram_changed_since_previous, rate the immediate collision or "
            "water risk of acting now. Do not calculate a numeric score.",
            [
                "CLEAR: no visible immediate hazard and a hop or wait is viable.",
                "UNCERTAIN: timing or object identity is ambiguous; wait or use a conservative hop.",
                "DANGEROUS: a current hop is likely to collide, enter water, or leave a carrier.",
            ],
        ),
        "subgoal": Choice(
            "Which immediate posture matches planner_card and the current frog "
            "zone: advance one row, align laterally, wait for motion, retreat, "
            "or escape a current hazard?",
            {
                "ADVANCE": "A single forward UP hop is the useful next subgoal.",
                "ALIGN_LEFT": "A single LEFT hop is needed before advancing.",
                "ALIGN_RIGHT": "A single RIGHT hop is needed before advancing.",
                "WAIT": "Stay still for the next frame while traffic/platforms move.",
                "RETREAT": "A DOWN hop is safer than advancing right now.",
                "ESCAPE": "Use a lateral or backward hop to leave an immediate hazard.",
            },
        ),
    }


def _choose_action(answers: Any, legal: set[str], noop: str, cooldown: int) -> str:
    """Turn typed judgments into one pulse; never trust generated text as an action."""
    if not isinstance(answers, dict):
        return noop
    first = answers.get("action_primary")
    second = answers.get("action_check")
    a = _safe_action(_answer_value(first), legal, noop)
    b = _safe_action(_answer_value(second), legal, noop)
    risk = _answer_value(answers.get("risk"))
    if isinstance(risk, str):
        risk_level = {"CLEAR": 0, "UNCERTAIN": 1, "DANGEROUS": 2}.get(risk.upper(), 1)
    else:
        # Score.value is a fractional level index, not the displayed label.
        risk_level = _finite_number(risk, 1.0)
    subgoal = _answer_value(answers.get("subgoal"))

    # A fresh, agreeing pair is required for an irreversible direction.  A
    # disagreement is useful information but not permission to guess.
    if a is not None and a == b:
        p = min(_probability(first, a), _probability(second, a))
        if a == noop:
            return noop
        if cooldown > 0:
            return noop
        if p >= 0.40:
            if risk_level >= 1.5 and a == "UP" and subgoal not in ("ESCAPE", "RETREAT"):
                return noop
            return a
        return noop

    # If the two independent views disagree, use their semantic subgoal only
    # when both views are explicit and the choice is a cheap lateral correction.
    # This avoids the baseline's habitual UP winning every ambiguous frame.
    if cooldown <= 0 and risk_level < 1.5:
        if subgoal == "ALIGN_LEFT" and "LEFT" in legal:
            if max(_probability(first, "LEFT"), _probability(second, "LEFT")) >= 0.58:
                return "LEFT"
        if subgoal == "ALIGN_RIGHT" and "RIGHT" in legal:
            if max(_probability(first, "RIGHT"), _probability(second, "RIGHT")) >= 0.58:
                return "RIGHT"
    return noop


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

    def _log(self, *args: Any, **fields: Any) -> None:
        try:
            if not self.ctx.done:
                self.ctx.log(*args, **fields)
        except Exception:
            # Logging must never make an otherwise valid episode fail.
            pass

    def _schedule_jev(self, game: Any, obs: Any, previous: Any, plan: str,
                      history: list[dict[str, Any]], current_action: str):
        state = _compact_observation(obs, previous)
        state["game_rules"] = (
            "Frogger has five traffic lanes, a safe median, five river rows, and "
            "unoccupied home bays. Cars and water kill; a direction is one hop in "
            "standard modes; moving objects continue during NOOP. Survival and a "
            "home outrank optional bonuses."
        )
        state["planner_card"] = plan[-_MAX_PLAN_CHARS:]
        state["recent_history"] = _json_value(history[-_MAX_HISTORY:])
        state["last_action"] = current_action
        state["action_pulse_rule"] = (
            "A direction may be posted for exactly one tick; code inserts NOOP "
            "cooldown frames after it. Do not ask for a held joystick."
        )
        state = _json_value(state)
        questions = _make_questions(game, state)
        asked_tick = _int(state.get("tick", 0))

        def call():
            return ctx_decide(self.ctx, state, questions)

        return self.ctx.spawn(call), asked_tick, state

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        spec = game.action_spec
        names = {str(name) for name in getattr(spec, "names", _ACTIONS)}
        noop = str(getattr(spec, "noop", "NOOP"))
        if noop not in names:
            noop = "NOOP" if "NOOP" in names else (next(iter(names), "NOOP"))
        legal = set(names) if names else set(_ACTIONS)
        if not getattr(game, "tick_s", None):
            # This candidate targets tier 2, but retaining a safe turn-game
            # branch makes pre-flight and accidental reuse harmless.
            if not ctx.done:
                try:
                    ctx.act(noop)
                except BudgetExceeded:
                    return
            return

        obs = game.observe()
        previous_obs = None
        history: list[dict[str, Any]] = []
        current_action = noop
        cooldown = 0
        pending_jev = None
        pending_plan = None
        planner_card = (
            "Astra planner is pending. Until it arrives, use the current row/x, "
            "recent RAM changes, and one-hop/no-repeat rules; NOOP is the safe fallback."
        )
        planner_session = None
        planner_calls = 0
        last_planned_deaths = _int(_field(obs, "deaths", 0))

        # Astra is intentionally not on the clock.  Jev and the deterministic
        # pulse controller keep the frog alive while the route card is written.
        def planner_prompt(snapshot: Any, reason: str, old_card: str) -> str:
            return (
                "You are the slow System-2 route planner for Atari 2600 Parker "
                "Brothers Frogger. Do not act and do not write a long essay. "
                "Write a compact planner card (at most 220 words) for a fast "
                "typed judge. The controller posts one-frame pulses and inserts "
                "NOOP between directional hops. Use the current live snapshot, "
                "raw RAM only as evidence, and recent transitions; never assume "
                "a fixed seed route. Prefer survival and an available home over "
                "optional bonuses. Explicitly state: SUBGOAL, WAIT_RULE, "
                "FORWARD_RULE, LATERAL_RULE, and REPLAN_IF. Mark uncertain RAM "
                "interpretations as hypotheses. The judge will see this card as "
                "text, so describe recognizable facts rather than inventing APIs.\n\n"
                f"REASON: {reason}\n"
                f"PREVIOUS_CARD: {old_card[-1800:]}\n"
                f"LIVE_SNAPSHOT_JSON: {snapshot}\n"
            )

        try:
            try:
                initial_prompt = planner_prompt(
                    _planner_observation(obs, history), "initial route and interface design", "none"
                )
                pending_plan = ctx.spawn(
                    lambda: ctx.think(initial_prompt, timeout_s=_ASTRA_TIMEOUT)
                )
                planner_calls = 1
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self._log("astra_schedule_failed", error=type(exc).__name__)

            # Post before the first wait: otherwise the first live frame gets
            # the adaptor's implicit NOOP rather than our explicit fallback.
            try:
                ctx.act(noop)
            except BudgetExceeded:
                return

            while not ctx.done and not getattr(game, "done", False):
                # One actual emulator frame passes here.  No busy loop is used.
                ctx.wait_for_tick()
                if ctx.done or getattr(game, "done", False):
                    break

                latest = game.observe()
                tick = _int(_field(latest, "tick", _field(latest, "frame_number", 0)))
                deaths = _int(_field(latest, "deaths", 0))
                score = _finite_number(_field(latest, "score", 0))
                row = _frog_row(latest)
                history.append({
                    "tick": tick,
                    "score": score,
                    "deaths": deaths,
                    "frog_row": row,
                    "frog_x": _frog_x(latest),
                    "timer_s": _finite_number(_field(latest, "timer_s", 0)),
                    "action_posted_for_previous_tick": current_action,
                    "zone": _zone(row, bool(_field(latest, "terminal", False))),
                })
                if len(history) > _MAX_HISTORY:
                    del history[:-_MAX_HISTORY]

                # Harvest an Astra card without ever waiting for it.
                if pending_plan is not None and pending_plan.done():
                    try:
                        result = pending_plan.result()
                        text = getattr(result, "text", result)
                        if not isinstance(text, str):
                            text = str(text)
                        text = text.strip()
                        if text:
                            planner_card = text[:_MAX_PLAN_CHARS]
                            sid = getattr(result, "session_id", None)
                            if isinstance(sid, str) and sid:
                                planner_session = sid
                            last_planned_deaths = deaths
                            self._log("astra_card_ready", calls=planner_calls, chars=len(planner_card))
                    except (BudgetExceeded, ModelUnavailable):
                        raise
                    except (TimeoutError, RuntimeError) as exc:
                        self._log("astra_card_failed", error=type(exc).__name__)
                        last_planned_deaths = deaths
                    except Exception as exc:
                        # Stand-in/pre-flight results may not look like a normal
                        # System-2 result; treat them as an absent card.
                        self._log("astra_card_unusable", error=type(exc).__name__)
                        last_planned_deaths = deaths
                    pending_plan = None

                # After a death, ask Astra to revise only when the previous call
                # has returned.  This makes adaptation event-driven rather than
                # spending the System-2 budget on every frame.
                if (
                    pending_plan is None
                    and planner_calls < _MAX_ASTRA_CALLS
                    and deaths > last_planned_deaths
                ):
                    prompt = planner_prompt(
                        _planner_observation(latest, history),
                        f"the frog has now recorded {deaths} deaths; revise after the failure",
                        planner_card,
                    )
                    session = planner_session
                    try:
                        if session:
                            pending_plan = ctx.spawn(
                                lambda p=prompt, s=session: ctx.think(
                                    p, session_id=s, timeout_s=_ASTRA_TIMEOUT
                                )
                            )
                        else:
                            pending_plan = ctx.spawn(
                                lambda p=prompt: ctx.think(p, timeout_s=_ASTRA_TIMEOUT)
                            )
                        planner_calls += 1
                    except (BudgetExceeded, ModelUnavailable):
                        raise
                    except Exception as exc:
                        self._log("astra_reschedule_failed", error=type(exc).__name__)
                        last_planned_deaths = deaths

                # A direction posted for the preceding tick is now replaced
                # by NOOP.  The counter is a real debounce interval, not a
                # second direction held on the joystick.  A completed answer
                # can choose the *next* pulse only after this gate.
                if current_action in _DIRECTIONS:
                    cooldown = max(cooldown, 3)
                    current_action = noop
                elif cooldown > 0:
                    cooldown -= 1
                    current_action = noop

                # A completed Jev answer is used only if it was judged on the
                # current life and is not too old.  Otherwise NOOP remains the
                # executable fallback while a fresh request is in flight.
                if pending_jev is not None and pending_jev[0].done():
                    future, asked_tick, _asked_state = pending_jev
                    pending_jev = None
                    try:
                        answers = future.result()
                        age = tick - asked_tick
                        judged_deaths = _int(_asked_state.get("deaths", -1), -1)
                        same_life = judged_deaths == deaths
                        if age <= _MAX_ANSWER_AGE_TICKS and same_life:
                            candidate = _choose_action(answers, legal, noop, cooldown)
                            if candidate in legal:
                                current_action = candidate
                            else:
                                current_action = noop
                        else:
                            current_action = noop
                    except (BudgetExceeded, ModelUnavailable):
                        raise
                    except (TimeoutError, RuntimeError) as exc:
                        self._log("jev_failed", error=type(exc).__name__)
                        current_action = noop
                    except Exception as exc:
                        # Includes malformed/garbage stand-in answers.
                        self._log("jev_unusable", error=type(exc).__name__)
                        current_action = noop

                # Schedule the next fast judgment only after harvesting the old
                # one.  It sees a compact temporal state rather than a stale
                # full game description alone.
                if pending_jev is None and not ctx.done:
                    try:
                        pending_jev = self._schedule_jev(
                            game, latest, previous_obs, planner_card, history, current_action
                        )
                    except (BudgetExceeded, ModelUnavailable):
                        raise
                    except Exception as exc:
                        self._log("jev_schedule_failed", error=type(exc).__name__)
                        pending_jev = None

                # Always refresh the action slot for the next emulator frame.
                # This eliminates stale ticks even while either model is slow.
                try:
                    ctx.act(current_action)
                except BudgetExceeded:
                    return

                previous_obs = latest

        finally:
            # A pending model request is safe to cancel; ctx also cancels model
            # calls when the game closes.  No custom background loop survives.
            for pending in (pending_jev, pending_plan):
                if pending is not None:
                    try:
                        pending[0].cancel() if isinstance(pending, tuple) else pending.cancel()
                    except Exception:
                        pass


def ctx_decide(ctx: Any, state: dict[str, Any], questions: dict[str, Any]) -> Any:
    """Small indirection keeps worker closures easy to inspect in traces."""
    return ctx.decide(state, questions, timeout_s=_JEV_TIMEOUT)


def build(ctx: Any) -> Harness:
    return Harness(ctx)
