"""A two-level Frogger controller: RAM-derived candidates, typed judging, and a
slow policy card.

The adaptor deliberately exposes only conservative candidate RAM fields.  This
controller therefore does not ask Jev to decode 128 bytes or to plan a whole
crossing.  It builds a small, auditable lane/carrier view from the moving RAM
tracks, asks Jev to judge the immediate maneuver, and lets code enforce a
one-frame joystick pulse.  Astra is an asynchronous policy reviewer: its
compact card changes the route bias and the fallback posture, never the action
syntax or the realtime loop.
"""
from __future__ import annotations

import json
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


# The first twenty moving RAM bytes are treated as five groups of four traffic
# candidates.  The fast-changing run beginning at $96 is treated as five
# groups of four carrier candidates; the intervening row is the safe median.
# This is intentionally a *candidate* decoder, not a claim that ALE provides
# object identities.  Raw RAM and its confidence stay in Astra's calibration
# input, and Jev receives only the resulting semantic buckets.
_ROAD_BASE = 0
_RIVER_BASE = 22
_GROUP_WIDTH = 4
_MAX_ROW = 10
_DIRECTIONS = {"LEFT", "RIGHT", "UP", "DOWN"}
_MANEUVER_TO_ACTION = {
    "FORWARD": "UP",
    "WAIT": "NOOP",
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "BACK": "DOWN",
}
_MANEUVER_NAMES = ("FORWARD", "WAIT", "LEFT", "RIGHT", "BACK", "UNSURE")
_HAZARD_NAMES = ("CLEAR", "TRAFFIC", "WATER", "EDGE", "UNKNOWN")


def _integer(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _text(value: Any, default: str = "", limit: int = 240) -> str:
    if isinstance(value, str):
        return value[:limit]
    return default


def _raw_ram(obs: Any) -> list[int]:
    if not isinstance(obs, dict):
        return []
    raw = obs.get("raw_ram")
    if not isinstance(raw, list):
        return []
    result: list[int] = []
    for value in raw:
        value = _integer(value, 0)
        result.append(max(0, min(255, value)))
    return result


def _safe_json(value: Any, depth: int = 0) -> Any:
    """Return a strict-JSON tree even when a stand-in gives odd values."""
    if depth > 5:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, (list, tuple)):
        return [_safe_json(item, depth + 1) for item in value[:80]]
    if isinstance(value, dict):
        return {
            str(key)[:80]: _safe_json(item, depth + 1)
            for key, item in list(value.items())[:120]
        }
    return _text(value, "", 120)


def _answer_value(answer: Any) -> Any:
    if answer is None:
        return None
    value = getattr(answer, "value", None)
    if value is not None:
        return value
    # This is only for harmless pre-flight stand-ins; real local answers use
    # .value.  Never interpret generated prose as an action.
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice"))
    return getattr(answer, "choice", None)


def _answer_probability(answer: Any, option: str) -> float:
    probabilities = getattr(answer, "probabilities", None)
    if isinstance(answer, dict):
        probabilities = answer.get("probabilities", probabilities)
    if not isinstance(probabilities, dict):
        return 0.0
    try:
        value = float(probabilities.get(option, 0.0))
        return value if value == value and value > 0.0 else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _answer_probability_of_value(answer: Any) -> float:
    value = _answer_value(answer)
    return _answer_probability(answer, value) if isinstance(value, str) else 0.0


def _circular_delta(target: int, origin: int) -> int:
    delta = (target - origin) % 256
    return delta - 256 if delta > 127 else delta


def _circular_distance(target: int, origin: int) -> int:
    return abs(_circular_delta(target, origin))


def _group_for_row(raw: list[int], row: int) -> list[int]:
    if row < 0 or row > _MAX_ROW:
        return []
    if row <= 4:
        start = _ROAD_BASE + row * _GROUP_WIDTH
    elif 6 <= row <= 10:
        start = _RIVER_BASE + (row - 6) * _GROUP_WIDTH
    else:
        # Row 5 is the riverbank/median, not a moving-object row.
        return []
    return raw[start : start + _GROUP_WIDTH]


def _side(delta: int) -> str:
    if delta < -8:
        return "LEFT"
    if delta > 8:
        return "RIGHT"
    return "CENTER"


def _track_bucket(previous: list[int], current: list[int], x: int) -> str:
    if not previous or not current or len(previous) != len(current):
        return "UNKNOWN"
    changes: list[int] = []
    for before, after in zip(previous, current):
        old_d = _circular_distance(before, x)
        new_d = _circular_distance(after, x)
        if new_d + 2 < old_d:
            changes.append(-1)  # approaching the frog
        elif new_d > old_d + 2:
            changes.append(1)   # receding from the frog
        else:
            changes.append(0)
    if not changes:
        return "UNKNOWN"
    if all(item == -1 for item in changes):
        return "APPROACHING"
    if all(item == 1 for item in changes):
        return "RECEDING"
    if all(item == 0 for item in changes):
        return "STABLE"
    return "MIXED"


def _object_view(group: list[int], x: int, kind: str,
                 previous: list[int] | None = None) -> dict[str, Any]:
    if not group:
        return {
            "kind": kind,
            "status": "UNKNOWN",
            "side": "UNKNOWN",
            "breadth": "UNKNOWN",
            "motion": "UNKNOWN",
            "escape": "UNKNOWN",
            "seek": "UNKNOWN",
        }
    distances = [_circular_distance(value, x) for value in group]
    nearest_index = min(range(len(group)), key=lambda index: distances[index])
    nearest_distance = distances[nearest_index]
    nearest_delta = _circular_delta(group[nearest_index], x)
    near_count = sum(distance <= 18 for distance in distances)
    motion = _track_bucket(previous or [], group, x)

    if kind == "traffic":
        if nearest_distance <= 14:
            status = "BLOCKED"
        elif nearest_distance <= 28:
            status = "TIGHT"
        else:
            status = "OPEN"
        # Move away from the nearest candidate car when a lateral escape is
        # needed.  A center/unknown result is deliberately not guessed.
        escape = {"LEFT": "RIGHT", "RIGHT": "LEFT"}.get(
            _side(nearest_delta), "UNKNOWN"
        )
        return {
            "kind": kind,
            "status": status,
            "side": _side(nearest_delta),
            "breadth": "MANY" if near_count >= 2 else ("ONE" if near_count else "NONE"),
            "motion": motion,
            "escape": escape,
            "seek": "UNKNOWN",
        }

    # For a carrier row, overlap is useful support, while a near edge is a
    # reason to steer toward the candidate before making a forward hop.
    if nearest_distance <= 18:
        status = "SUPPORTED"
    elif nearest_distance <= 36:
        status = "EDGE"
    else:
        status = "MISSING"
    seek = _side(nearest_delta)
    return {
        "kind": kind,
        "status": status,
        "side": seek,
        "breadth": "BROAD" if near_count >= 2 else ("NARROW" if near_count else "NONE"),
        "motion": motion,
        "escape": "UNKNOWN",
        "seek": seek,
    }


def _row_bucket(row: int) -> str:
    if row >= 200:
        return "START"
    if 0 <= row <= 4:
        return "ROAD_LANE_" + str(row + 1)
    if row == 5:
        return "MEDIAN"
    if 6 <= row <= 10:
        return "RIVER_ROW_" + str(row - 5)
    if row > 10:
        return "HOME_APPROACH"
    return "UNKNOWN"


def _x_bucket(x: int) -> str:
    # The frog candidate is in the same 8-bit horizontal coordinate family as
    # the moving candidates.  This coarse label is all Jev needs.
    if x < 56:
        return "LEFT_OF_CENTER"
    if x > 104:
        return "RIGHT_OF_CENTER"
    return "CENTER_BAND"


def _planner_groups(raw: list[int]) -> list[list[int]]:
    return [raw[index : index + 4] for index in range(0, min(len(raw), 44), 4)]


def _make_snapshot(obs: Any, previous_raw: list[int] | None,
                   previous_row: int | None, previous_deaths: int | None,
                   plan: dict[str, str], stall_count: int) -> dict[str, Any]:
    """Create both the controller facts and the small Jev-facing state."""
    if not isinstance(obs, dict):
        obs = {}
    raw = _raw_ram(obs)
    fields = obs.get("ram_fields") if isinstance(obs.get("ram_fields"), dict) else {}
    frog = obs.get("frog") if isinstance(obs.get("frog"), dict) else {}
    row = _integer(frog.get("row", fields.get("frog_row_ac", 255)), 255)
    x = _integer(frog.get("x", fields.get("frog_x_b0", 80)), 80)
    deaths = _integer(obs.get("deaths", 0), 0)
    reserve = _integer(obs.get("reserve_frogs", fields.get("reserve_d0", 0)), 0)
    timer_value = _integer(obs.get("timer_s"), -1)
    if timer_value >= 20:
        timer_bucket = "PLENTY"
    elif timer_value >= 10:
        timer_bucket = "LIMITED"
    elif timer_value >= 0:
        timer_bucket = "URGENT"
    else:
        timer_bucket = "UNKNOWN"
    row_change = "NEW_OR_UNKNOWN"
    if previous_row is not None:
        if deaths > (previous_deaths if previous_deaths is not None else deaths):
            row_change = "LIFE_LOST"
        elif row > previous_row:
            row_change = "FORWARD_PROGRESS"
        elif row < previous_row:
            row_change = "RESET_OR_BACKTRACK"
        else:
            row_change = "NO_ROW_CHANGE"

    current_group = _group_for_row(raw, row)
    target_row = row + 1 if row < 200 else 0
    target_group = _group_for_row(raw, target_row)
    current_kind = "traffic" if 0 <= row <= 4 else "carrier"
    if row >= 200:
        current_kind = "start"
    target_kind = "traffic" if 0 <= target_row <= 4 else "carrier"
    if row == 5:
        current_kind = "safe_median"
    if target_row == 5:
        target_kind = "safe_median"
    if row >= 10 and target_row > 10:
        target_kind = "home_bay"

    current_view = _object_view(
        current_group, x, current_kind,
        _group_for_row(previous_raw or [], row),
    )
    target_view = _object_view(
        target_group, x, target_kind,
        _group_for_row(previous_raw or [], target_row),
    )
    if row == 5:
        current_view = {
            "kind": "safe_median",
            "status": "SAFE",
            "side": "CENTER",
            "breadth": "BROAD",
            "motion": "NONE",
            "escape": "UNKNOWN",
            "seek": "UNKNOWN",
        }
    if target_row == 5:
        target_view = {
            "kind": "safe_median",
            "status": "SAFE",
            "side": "CENTER",
            "breadth": "BROAD",
            "motion": "NONE",
            "escape": "UNKNOWN",
            "seek": "UNKNOWN",
        }
    if target_row > _MAX_ROW:
        target_view = {
            "kind": "home_bay",
            "status": "AVAILABLE_OR_UNKNOWN",
            "side": "CENTER",
            "breadth": "UNKNOWN",
            "motion": "NONE",
            "escape": "UNKNOWN",
            "seek": "UNKNOWN",
        }

    if row_change == "LIFE_LOST":
        progress = "A_NEW_FROG_HAS_SPAWNED"
    elif row_change == "FORWARD_PROGRESS":
        progress = "THE_LAST_OBSERVED_FROG_ROW_ADVANCED"
    elif stall_count <= 1:
        progress = "RECENTLY_ACTIVE"
    elif stall_count <= 4:
        progress = "SHORT_STALL"
    else:
        progress = "LONG_STALL_NEEDS_PROGRESS"

    # The code-level candidate is computed before Jev judges it.  In standard
    # modes every direction is still one hop; this field is not a held command.
    candidate = _heuristic_action(
        row, target_row, current_view, target_view, plan, stall_count, deaths
    )
    homes = _integer(obs.get("frogs_home", 0), 0)
    level = _integer(obs.get("level", 1), 1)
    home_progress = "NO_HOME" if homes <= 0 else ("PARTIAL" if homes < 5 else "LEVEL_COMPLETE")
    level_bucket = "FIRST_LEVEL" if level <= 1 else "HARDER_LEVEL"
    return {
        "tick": _integer(obs.get("tick", 0), 0),
        "frame": _integer(obs.get("frame", obs.get("frame_number", 0)), 0),
        "score": _integer(obs.get("score", 0), 0),
        "homes": home_progress,
        "level": level_bucket,
        "deaths": deaths,
        "reserve_frogs": reserve,
        "timer": timer_bucket,
        "terminal": bool(obs.get("terminal", False) or obs.get("truncated", False)),
        "zone": _row_bucket(row),
        "horizontal_band": _x_bucket(x),
        "row_change": row_change,
        "progress": progress,
        "current": current_view,
        "target": target_view,
        "target_zone": _row_bucket(target_row),
        "plan": dict(plan),
        "decoder": "RAM track candidates only; object identity is uncertain",
        "last_action": _text(obs.get("last_action"), "NOOP", 20).upper(),
        "candidate_action": candidate,
        "stall_bucket": progress,
        # Private numeric/context fields used by the bridge, not sent to Jev.
        "_row": row,
        "_x": x,
        "_raw": raw,
        "_target_row": target_row,
        "_stall": stall_count,
    }


def _weave_direction(bias: str, row: int, deaths: int) -> str:
    if bias in ("LEFT", "RIGHT"):
        return bias
    # WEAVE is a generic fallback, not a seed-specific route.  Alternating by
    # coarse board progress gives later lives a different horizontal probe.
    return "LEFT" if (row + deaths) % 2 == 0 else "RIGHT"


def _heuristic_action(row: int, target_row: int, current: dict[str, Any],
                      target: dict[str, Any], plan: dict[str, str],
                      stall_count: int, deaths: int) -> str:
    """Produce a progress-preserving candidate without model prose."""
    if row >= 200:
        return "UP"
    if row > _MAX_ROW:
        return "UP"
    if row == 5:
        return "UP"

    current_status = _text(current.get("status"), "UNKNOWN")
    current_kind = _text(current.get("kind"), "unknown")
    target_status = _text(target.get("status"), "UNKNOWN")
    target_kind = _text(target.get("kind"), "unknown")
    bias = _text(plan.get("bias"), "WEAVE")
    road_rule = _text(plan.get("road_rule"), "ADVANCE_WHEN_OPEN")
    river_rule = _text(plan.get("river_rule"), "FOLLOW_CARRIER")
    fallback = _text(plan.get("fallback"), "STEER")

    if target_kind == "safe_median":
        return "UP"

    if target_kind == "traffic":
        # A car overlapping the current lane is more urgent than a blocked
        # landing lane.  This is the key difference from a blind wait: leave
        # the current collision interval when a candidate escape is visible.
        if current_kind == "traffic" and current_status == "BLOCKED":
            escape = _text(current.get("escape"), "UNKNOWN")
            if escape in ("LEFT", "RIGHT"):
                return escape
            if target_status == "OPEN":
                return "UP"
            if stall_count >= 1:
                return _weave_direction(bias, row, deaths)
            return "NOOP"
        if target_status == "OPEN":
            if road_rule == "WEAVE" and stall_count >= 3:
                return _weave_direction(bias, row, deaths)
            # Never make a gratuitous lateral hop on a fresh lane.
            return "UP"
        if road_rule == "WAIT_FOR_GAP" and stall_count < 4:
            return "NOOP"
        if stall_count < 4:
            return "NOOP"
        escape = _text(target.get("escape"), "UNKNOWN")
        if escape in ("LEFT", "RIGHT"):
            return escape
        return _weave_direction(bias, row, deaths)

    if target_kind == "carrier":
        # Once in the river, preserving the current carrier takes precedence
        # over the tempting fact that a later row happens to be supported.
        if current_kind == "carrier" and current_status in ("MISSING", "EDGE"):
            seek = _text(current.get("seek"), "UNKNOWN")
            if seek in ("LEFT", "RIGHT"):
                return seek
            if stall_count >= 3:
                return _weave_direction(bias, row, deaths)
            return "NOOP"
        if target_status == "SUPPORTED":
            return "UP"
        seek = _text(target.get("seek"), "UNKNOWN")
        if seek in ("LEFT", "RIGHT") and river_rule != "WAIT_FOR_CARRIER":
            return seek
        if stall_count >= 4:
            if fallback == "UP":
                return "UP"
            if fallback == "NOOP":
                return "NOOP"
            return _weave_direction(bias, row, deaths)
        return "NOOP"

    # Home entry is deliberately not optimized for bonuses: complete the
    # crossing first and let the semantic judge veto an occupied bay.
    if target_kind == "home_bay":
        return "UP"
    if current_kind == "carrier":
        seek = _text(current.get("seek"), "UNKNOWN")
        if seek in ("LEFT", "RIGHT") and stall_count >= 2:
            return seek
    if stall_count >= 5:
        return _weave_direction(bias, row, deaths)
    return "UP"


def _compact_for_jev(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in snapshot.items()
        if not key.startswith("_")
    }


def _planner_state(snapshot: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    raw = snapshot.get("_raw", [])
    return {
        "zone": snapshot.get("zone", "UNKNOWN"),
        "frog_row_candidate": snapshot.get("_row", 255),
        "frog_x_candidate": snapshot.get("_x", 80),
        "raw_ram_80_to_ab": raw[:44],
        "ram_fields": {
            "score": snapshot.get("score", 0),
            "homes": snapshot.get("homes", "NO_HOME"),
            "level": snapshot.get("level", "FIRST_LEVEL"),
            "deaths": snapshot.get("deaths", 0),
            "reserve_frogs": snapshot.get("reserve_frogs", 0),
            "timer": snapshot.get("timer", "unknown"),
        },
        "derived_candidate_tracks": {
            "current": snapshot.get("current", {}),
            "target": snapshot.get("target", {}),
            "target_zone": snapshot.get("target_zone", "UNKNOWN"),
        },
        "recent_evidence": history[-24:],
        "decoder_warning": (
            "The object groups are hypotheses from observations, not a first-party ALE object API. "
            "Improve the policy conservatively and do not invent seed-specific schedules."
        ),
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    candidates = [text.strip()]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_plan(text: Any, old: dict[str, str]) -> dict[str, str]:
    """Accept only a tiny allow-listed card; arbitrary Astra prose is inert."""
    card = dict(old)
    obj = _extract_json_object(_text(text, "", 5000))
    if obj is None:
        return card

    def choose(keys: tuple[str, ...], allowed: set[str], current: str) -> str:
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str):
                normalized = value.strip().upper().replace("-", "_")
                if normalized in allowed:
                    return normalized
        return current

    card["bias"] = choose(
        ("bias", "horizontal_bias", "river_bias"),
        {"LEFT", "RIGHT", "CENTER", "WEAVE"},
        card.get("bias", "WEAVE"),
    )
    card["road_rule"] = choose(
        ("road_rule", "traffic_rule"),
        {"WAIT_FOR_GAP", "ADVANCE_WHEN_OPEN", "WEAVE"},
        card.get("road_rule", "ADVANCE_WHEN_OPEN"),
    )
    card["river_rule"] = choose(
        ("river_rule", "carrier_rule"),
        {"FOLLOW_CARRIER", "WAIT_FOR_CARRIER", "ADVANCE_WHEN_SUPPORTED"},
        card.get("river_rule", "FOLLOW_CARRIER"),
    )
    card["fallback"] = choose(
        ("fallback", "late_fallback"),
        {"NOOP", "UP", "STEER"},
        card.get("fallback", "STEER"),
    )
    note = obj.get("note", obj.get("reason", ""))
    if isinstance(note, str):
        card["note"] = note[:180]
    return card


def _plan_prompt(game: Any, snapshot: dict[str, Any], history: list[dict[str, Any]],
                 reason: str, old: dict[str, str]) -> str:
    return (
        "You are Astra, the slow route-policy reviewer for a realtime Atari 2600 Frogger controller. "
        "Do not drive individual frames and do not output a long essay. The controller already owns "
        "the literal action map, score bookkeeping, one-hop debounce, and timing. Inspect the supplied "
        "RAM snapshots and failure evidence, then return exactly one compact JSON object with keys "
        "bias, road_rule, river_rule, fallback, note. Allowed values are: bias LEFT, RIGHT, CENTER, "
        "or WEAVE; road_rule WAIT_FOR_GAP, ADVANCE_WHEN_OPEN, or WEAVE; river_rule FOLLOW_CARRIER, "
        "WAIT_FOR_CARRIER, or ADVANCE_WHEN_SUPPORTED; fallback NOOP, UP, or STEER. Keep note under "
        "180 characters. Choose a generic policy that can transfer to another reset phase; never encode "
        "a training seed or a fixed wall-clock schedule. Object groups are hypotheses, so say WAIT or "
        "STEER rather than pretending an uncertain byte is a car identity. Jev will receive your parsed "
        "card and a semantic one-frame state, and will judge an immediate maneuver.\n\n"
        f"Reason for review: {reason}\n"
        f"Game: {_text(game.describe(), 'Atari 2600 Frogger', 2500)}\n"
        f"Previous card: {json.dumps(old, separators=(',', ':'), ensure_ascii=True)}\n"
        f"Current planner input: {json.dumps(_planner_state(snapshot, history), separators=(',', ':'), ensure_ascii=True)}\n"
    )


def _make_questions(decision_index: int) -> dict[str, Choice]:
    # Reorder the duplicate maneuver Choices so Jev's documented first-option
    # bias cannot permanently turn uncertainty into FORWARD.
    shift = decision_index % len(_MANEUVER_NAMES)
    order_a = list(_MANEUVER_NAMES[shift:]) + list(_MANEUVER_NAMES[:shift])
    reversed_names = tuple(reversed(_MANEUVER_NAMES))
    order_b = list(reversed_names[shift:]) + list(reversed_names[:shift])
    maneuver_descriptions = {
        "FORWARD": "One UP hop into the stated target row; use only if the target facts support it.",
        "WAIT": "One NOOP frame while traffic or carriers move; this is a deliberate wait, not missing input.",
        "LEFT": "One LEFT hop to seek the stated left-side gap or carrier; never hold it.",
        "RIGHT": "One RIGHT hop to seek the stated right-side gap or carrier; never hold it.",
        "BACK": "One DOWN hop only to escape a stated immediate hazard; do not use for ordinary planning.",
        "UNSURE": "The semantic facts do not support a maneuver; choose the code candidate conservatively.",
    }
    risk_descriptions = {
        "CLEAR": "The target/current facts support a useful hop or a controlled wait.",
        "TRAFFIC": "The target road lane has a near traffic candidate; do not blindly enter it.",
        "WATER": "The current or target river row lacks established carrier support.",
        "EDGE": "The frog appears near a carrier edge or screen edge and needs preservation.",
        "UNKNOWN": "The RAM-derived identity is uncertain; prefer the code candidate and one-frame safety.",
    }
    return {
        "maneuver_primary": Choice(
            "Using only the semantic fields in state, choose the best immediate maneuver. "
            "This is one Atari frame, not a multi-hop plan. Respect state.target and state.current; "
            "do not calculate from raw byte magnitudes. If evidence is uncertain, choose UNSURE or WAIT.",
            {name: maneuver_descriptions[name] for name in order_a},
        ),
        "maneuver_check": Choice(
            "Independently judge the same single-frame maneuver from state.target, state.current, "
            "state.progress, and state.plan. Agree with a directional maneuver only when its stated "
            "precondition is visible; a direction is released after one tick.",
            {name: maneuver_descriptions[name] for name in order_b},
        ),
        "hazard": Choice(
            "Classify the immediate hazard relevant to the next hop, not the eventual route. "
            "Use the named status buckets and do not infer a hazard from arithmetic alone.",
            {name: risk_descriptions[name] for name in _HAZARD_NAMES},
        ),
    }


def _valid_maneuver(answer: Any) -> str | None:
    value = _answer_value(answer)
    return value if isinstance(value, str) and value in _MANEUVER_TO_ACTION else None


def _choose_action(snapshot: dict[str, Any], answers: Any, decision_index: int) -> str:
    """Combine typed recognition with an auditable safety/progress bridge."""
    candidate = _text(snapshot.get("candidate_action"), "NOOP").upper()
    if candidate not in {"NOOP", "UP", "LEFT", "RIGHT", "DOWN"}:
        candidate = "NOOP"
    row = _integer(snapshot.get("_row"), 255)
    target = snapshot.get("target") if isinstance(snapshot.get("target"), dict) else {}
    target_kind = _text(target.get("kind"), "unknown")
    target_status = _text(target.get("status"), "UNKNOWN")

    if not isinstance(answers, dict):
        return candidate
    primary = answers.get("maneuver_primary")
    check = answers.get("maneuver_check")
    hazard = answers.get("hazard")
    p_value = _valid_maneuver(primary)
    c_value = _valid_maneuver(check)
    p_prob = _answer_probability_of_value(primary) if p_value else 0.0
    c_prob = _answer_probability_of_value(check) if c_value else 0.0
    h_value = _answer_value(hazard)
    h_prob = _answer_probability_of_value(hazard) if h_value else 0.0

    # A strong typed hazard veto prevents the old raw-RAM UP bias from entering
    # a clearly blocked road or unsupported river row.
    if h_value in ("TRAFFIC", "WATER", "EDGE") and h_prob >= 0.58:
        if target_kind == "traffic" and h_value == "TRAFFIC":
            if target_status != "OPEN":
                if candidate in ("LEFT", "RIGHT"):
                    return candidate
                return "NOOP"
        if target_kind == "carrier" and h_value == "WATER":
            if candidate in ("LEFT", "RIGHT"):
                return candidate
            return "NOOP"
        if h_value == "EDGE" and candidate in ("LEFT", "RIGHT", "DOWN"):
            return candidate

    # A consensus is useful, but a weak consensus must not erase a code-level
    # carrier/traffic candidate.  The thresholds are probabilities of the
    # selected option, not the answer confidence field.
    if p_value is not None and p_value == c_value and min(p_prob, c_prob) >= 0.34:
        model_action = _MANEUVER_TO_ACTION[p_value]
        if model_action == "UP":
            if target_kind == "traffic" and target_status == "BLOCKED" and min(p_prob, c_prob) < 0.62:
                return candidate if candidate != "UP" else "NOOP"
            if target_kind == "carrier" and target_status == "MISSING" and min(p_prob, c_prob) < 0.62:
                return candidate if candidate != "UP" else "NOOP"
        if model_action == "DOWN" and row < 6 and h_value != "EDGE":
            return candidate
        return model_action

    # If one judge agrees with the computed candidate and is not very weak, use
    # it.  This makes Jev's current-frame recognition matter without allowing a
    # single WAIT answer to starve the timer forever.
    candidate_maneuver = {
        "UP": "FORWARD", "NOOP": "WAIT", "LEFT": "LEFT",
        "RIGHT": "RIGHT", "DOWN": "BACK",
    }[candidate]
    if (p_value == candidate_maneuver and p_prob >= 0.42) or (
        c_value == candidate_maneuver and c_prob >= 0.42
    ):
        return candidate

    # Periodically honor a directional typed judgment when code has no clear
    # object identity.  This is the exploration path that the old controller
    # lacked: it can enter the river's horizontal route rather than issuing
    # only UP/NOOP forever.
    if p_value == c_value and p_value in ("LEFT", "RIGHT") and min(p_prob, c_prob) >= 0.40:
        return _MANEUVER_TO_ACTION[p_value]

    return candidate


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            # Logging is diagnostic and must never make a realtime episode fail.
            pass

    def _start_astra(self, game: Any, snapshot: dict[str, Any], history: list[dict[str, Any]],
                     reason: str, old: dict[str, str], session: str | None,
                     calls: int, pending: Any) -> tuple[Any, int]:
        if pending is not None or calls >= 5 or getattr(self.ctx, "system2", None) is None:
            return pending, calls
        prompt = _plan_prompt(game, snapshot, history, reason, old)
        try:
            if session:
                pending = self.ctx.spawn(
                    lambda p=prompt, s=session: self.ctx.think(
                        p, session_id=s, timeout_s=78.0
                    )
                )
            else:
                pending = self.ctx.spawn(
                    lambda p=prompt: self.ctx.think(p, timeout_s=78.0)
                )
            calls += 1
            self._log("astra_review_started", reason=reason, call=calls)
        except (BudgetExceeded, EpisodeOver):
            pending = None
        except (TimeoutError, RuntimeError, ModelUnavailable):
            pending = None
        except Exception:
            pending = None
        return pending, calls

    def _collect_astra(self, pending: Any, plan: dict[str, str], session: str | None) -> tuple[Any, dict[str, str], str | None]:
        if pending is None:
            return None, plan, session
        try:
            if not pending.done():
                return pending, plan, session
            result = pending.result()
            text = getattr(result, "text", "")
            new_plan = _parse_plan(text, plan)
            new_session = getattr(result, "session_id", session)
            self._log("astra_card_ready", bias=new_plan.get("bias"), rule=new_plan.get("river_rule"))
            return None, new_plan, new_session
        except Exception as exc:
            self._log("astra_card_failed", error=type(exc).__name__)
            return None, plan, session

    def _pulse(self, action: str, names: set[str], noop: str) -> bool:
        """Post one action frame, then explicitly release a direction."""
        ctx = self.ctx
        if action not in names:
            action = noop
        try:
            ctx.act(action)
            if ctx.done:
                return False
            ctx.wait_for_tick()
            if ctx.done:
                return False
            if action in _DIRECTIONS:
                ctx.act(noop)
            return not ctx.done
        except (BudgetExceeded, EpisodeOver):
            return False
        except (TimeoutError, RuntimeError, ModelUnavailable):
            return False

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        if getattr(ctx, "system1", None) is None:
            return

        spec = game.action_spec
        names = {str(name).upper() for name in getattr(spec, "names", ())}
        if not names:
            return
        raw_noop = getattr(spec, "noop", "NOOP")
        noop = str(raw_noop).upper() if raw_noop is not None else "NOOP"
        if noop not in names:
            noop = "NOOP" if "NOOP" in names else next(iter(names))
        # The benchmark's names are uppercase; normalize only our bridge and
        # pass the exact advertised spelling when a variant uses one.
        name_lookup = {str(name).upper(): str(name) for name in getattr(spec, "names", ())}

        plan: dict[str, str] = {
            "bias": "WEAVE",
            "road_rule": "ADVANCE_WHEN_OPEN",
            "river_rule": "FOLLOW_CARRIER",
            "fallback": "STEER",
            "note": "pending Astra review; preserve a frog before optional bonuses",
        }
        previous_raw: list[int] | None = None
        previous_row: int | None = None
        previous_deaths: int | None = None
        previous_home = 0
        stall_count = 0
        decision_index = 0
        review_for_deaths = -1
        history: list[dict[str, Any]] = []
        session: str | None = None
        astra_calls = 0
        pending = None

        try:
            initial = game.observe()
        except Exception:
            initial = {}
        snapshot = _make_snapshot(
            initial, previous_raw, previous_row, previous_deaths,
            plan, stall_count,
        )
        pending, astra_calls = self._start_astra(
            game, snapshot, history, "initial route policy", plan,
            session, astra_calls, pending,
        )

        while not ctx.done and not bool(getattr(game, "done", False)):
            pending, plan, session = self._collect_astra(pending, plan, session)
            try:
                obs = game.observe()
            except Exception:
                obs = initial
            snapshot = _make_snapshot(
                obs, previous_raw, previous_row, previous_deaths,
                plan, stall_count,
            )
            row = _integer(snapshot.get("_row"), 255)
            deaths = _integer(snapshot.get("deaths"), 0)
            home = _integer(obs.get("frogs_home", 0), previous_home) if isinstance(obs, dict) else previous_home
            if previous_row is not None:
                if deaths > (previous_deaths if previous_deaths is not None else deaths):
                    stall_count = 0
                elif row == previous_row:
                    stall_count += 1
                else:
                    stall_count = 0
                snapshot = _make_snapshot(
                    obs, previous_raw, previous_row, previous_deaths,
                    plan, stall_count,
                )
            if deaths > 0 and deaths > review_for_deaths and pending is None:
                # If the initial review was still in flight at the death, wait
                # for it to finish and then submit this revision; never create
                # two concurrent Astra conversations for one episode.
                pending, astra_calls = self._start_astra(
                    game, snapshot, history, "a life was lost; revise the immediate route", plan,
                    session, astra_calls, pending,
                )
                review_for_deaths = deaths

            compact = _compact_for_jev(snapshot)
            questions = _make_questions(decision_index)
            try:
                answers = ctx.decide(compact, questions, timeout_s=0.62)
            except (BudgetExceeded, EpisodeOver):
                return
            except ModelUnavailable:
                return
            except (TimeoutError, RuntimeError):
                # A late typed answer must not leave a directional action
                # latched.  A single code candidate is posted only after the
                # explicit pulse/release bridge, and a failed call retries.
                fallback = _text(snapshot.get("candidate_action"), "NOOP").upper()
                if fallback not in {"UP", "LEFT", "RIGHT", "DOWN", "NOOP"}:
                    fallback = "NOOP"
                if fallback in ("LEFT", "RIGHT") and stall_count < 5:
                    fallback = "NOOP"
                if not self._pulse(name_lookup.get(fallback, noop), names, name_lookup.get(noop, noop)):
                    return
                decision_index += 1
                previous_raw = list(snapshot.get("_raw", []))
                previous_row = row
                previous_deaths = deaths
                previous_home = home
                continue

            action_upper = _choose_action(snapshot, answers, decision_index)
            if action_upper not in {"UP", "LEFT", "RIGHT", "DOWN", "NOOP"}:
                action_upper = "NOOP"
            if action_upper == "NOOP" and stall_count >= 6:
                # Code owns liveness.  Never let a stream of low-confidence
                # WAIT judgments consume all four reserves on the sidewalk.
                forced = _text(snapshot.get("candidate_action"), "UP").upper()
                if forced in {"UP", "LEFT", "RIGHT", "DOWN"}:
                    action_upper = forced
            action = name_lookup.get(action_upper, name_lookup.get(noop, noop))
            self._log(
                "maneuver",
                tick=_integer(snapshot.get("tick"), 0),
                action=action_upper,
                candidate=_text(snapshot.get("candidate_action"), "NOOP"),
                zone=_text(snapshot.get("zone"), "UNKNOWN"),
                target=_text((snapshot.get("target") or {}).get("status"), "UNKNOWN"),
            )
            if not self._pulse(action, names, name_lookup.get(noop, noop)):
                return

            history.append({
                "zone": _text(snapshot.get("zone"), "UNKNOWN"),
                "target_status": _text((snapshot.get("target") or {}).get("status"), "UNKNOWN"),
                "candidate": _text(snapshot.get("candidate_action"), "NOOP"),
                "action": action_upper,
                "progress": _text(snapshot.get("progress"), "UNKNOWN"),
                "deaths": deaths,
            })
            if len(history) > 40:
                del history[:-40]
            previous_raw = list(snapshot.get("_raw", []))
            previous_row = row
            previous_deaths = deaths
            previous_home = home
            decision_index += 1
        # Do not join an Astra future.  The contract cancels model work when a
        # game closes, and no plain background loop is owned by this harness.


def build(ctx: Any) -> Harness:
    return Harness(ctx)
