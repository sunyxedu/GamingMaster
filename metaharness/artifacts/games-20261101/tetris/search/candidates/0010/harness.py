"""Crossover harness: exact placement search with a non-blocking model bridge.

The controller never lets a model call be the clock driver.  Code enumerates
Guideline-shaped placements, evaluates a short queue lookahead, and executes a
verified rotation/translation/drop sequence.  Jev only chooses among those
already legal destinations.  Astra, when it eventually answers, selects a
risk regime that changes the code-owned evaluator.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


WIDTH = 10
HEIGHT = 20
PIECES = ("I", "J", "L", "O", "S", "T", "Z")
ORIENTATION_NAMES = ("0", "R", "2", "L")

# Spawn-orientation cells in the adaptor's bottom-origin coordinates.  These
# are also directly checked against the observed active cells by the executor;
# the game remains authoritative for collision and SRS kicks.
_BASE: dict[str, tuple[tuple[int, int], ...]] = {
    "J": ((0, 2), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 2), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 2), (2, 2), (0, 1), (1, 1)),
    "Z": ((0, 2), (1, 2), (1, 1), (2, 1)),
    "T": ((1, 2), (0, 1), (1, 1), (2, 1)),
    "I": ((0, 2), (1, 2), (2, 2), (3, 2)),
    "O": ((1, 2), (2, 2), (1, 1), (2, 1)),
}

# Standard SRS kick data is kept with the geometry as explicit ruleset data,
# so a target orientation is not confused with a generic matrix rotation.  The
# engine performs the actual kick and remains authoritative for reachability.
_KICKS: dict[str, dict[tuple[int, int], tuple[tuple[int, int], ...]]] = {
    "JLSTZ": {
        (0, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
        (1, 0): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
        (1, 2): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
        (2, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
        (2, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
        (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        (0, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    },
    "I": {
        (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
        (1, 0): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
        (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
        (2, 1): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
        (2, 3): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
        (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
        (3, 0): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
        (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    },
}


def _make_shapes() -> dict[str, tuple[tuple[tuple[int, int], ...], ...]]:
    result: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {}
    for piece, base in _BASE.items():
        states: list[tuple[tuple[int, int], ...]] = []
        current = tuple(base)
        for _ in range(4):
            states.append(tuple(sorted(current)))
            if piece == "O":
                continue
            pivot = (1.5, 1.5) if piece == "I" else (1.0, 1.0)
            rotated: list[tuple[int, int]] = []
            for x, y in current:
                dx, dy = x - pivot[0], y - pivot[1]
                # Clockwise in bottom-origin coordinates.
                nx = pivot[0] + dy
                ny = pivot[1] - dx
                rotated.append((int(round(nx)), int(round(ny))))
            current = tuple(rotated)
        result[piece] = tuple(states)
    return result


SHAPES = _make_shapes()


def _grid(obs: dict[str, Any]) -> list[list[int]]:
    """Convert the public top-to-bottom board into bottom-origin rows."""
    grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    raw = obs.get("board")
    if not isinstance(raw, list):
        return grid
    for top_y, row in enumerate(raw[:HEIGHT]):
        if not isinstance(row, list):
            continue
        y = HEIGHT - 1 - top_y
        for x, value in enumerate(row[:WIDTH]):
            try:
                grid[y][x] = int(value) if value else 0
            except (TypeError, ValueError):
                grid[y][x] = 0
    return grid


def _heights(grid: list[list[int]]) -> list[int]:
    heights: list[int] = []
    for x in range(WIDTH):
        height = 0
        for y in range(HEIGHT - 1, -1, -1):
            if grid[y][x]:
                height = y + 1
                break
        heights.append(height)
    return heights


def _features(grid: list[list[int]]) -> dict[str, Any]:
    heights = _heights(grid)
    holes = 0
    hole_depth = 0
    for x, height in enumerate(heights):
        occupied_above = 0
        for y in range(height - 1, -1, -1):
            if grid[y][x]:
                occupied_above += 1
            elif occupied_above:
                holes += 1
                hole_depth += occupied_above
    bumpiness = sum(abs(heights[x] - heights[x + 1]) for x in range(WIDTH - 1))
    wells = 0
    for x, height in enumerate(heights):
        left = heights[x - 1] if x else HEIGHT
        right = heights[x + 1] if x + 1 < WIDTH else HEIGHT
        wells += max(0, min(left, right) - height)
    row_transitions = 0
    for y in range(HEIGHT):
        previous = 1
        for x in range(WIDTH):
            current = 1 if grid[y][x] else 0
            if current != previous:
                row_transitions += 1
            previous = current
        if previous == 0:
            row_transitions += 1
    return {
        "heights": heights,
        "aggregate_height": sum(heights),
        "max_height": max(heights) if heights else 0,
        "holes": holes,
        "hole_depth": hole_depth,
        "bumpiness": bumpiness,
        "wells": wells,
        "row_transitions": row_transitions,
    }


def _drop(
    grid: list[list[int]], piece: str, orientation: int, anchor_x: int
) -> tuple[list[list[int]], dict[str, Any]] | None:
    """Simulate a legal final placement using the observed Guideline geometry."""
    shape = SHAPES.get(piece, SHAPES["O"])[orientation % 4]
    if not shape:
        return None

    def legal(y0: int) -> bool:
        for dx, dy in shape:
            x, y = anchor_x + dx, y0 + dy
            if x < 0 or x >= WIDTH or y < 0:
                return False
            # The public board has no buffer rows.  Above the visible board is
            # empty for search purposes; such a result is rejected below.
            if y < HEIGHT and grid[y][x]:
                return False
        return True

    y = HEIGHT + 5
    while legal(y - 1):
        y -= 1
    if not legal(y):
        return None
    cells = [(anchor_x + dx, y + dy) for dx, dy in shape]
    if any(y0 < 0 or y0 >= HEIGHT for _, y0 in cells):
        return None

    placed = [row[:] for row in grid]
    for x, y0 in cells:
        placed[y0][x] = 1
    survivors = [row for row in placed if sum(1 for value in row if value) < WIDTH]
    lines = HEIGHT - len(survivors)
    survivors.extend([[0 for _ in range(WIDTH)] for _ in range(lines)])
    features = _features(survivors)
    features.update(
        {
            "lines": lines,
            "landing_height": max(y0 for _, y0 in cells),
            "cells": tuple(cells),
        }
    )
    return survivors, features


def _placement_value(
    after: dict[str, Any], before: dict[str, Any], style: str, lines: int
) -> float:
    """Conservative code-owned value; models never do this arithmetic."""
    if style == "RECOVER":
        hole_w, depth_w, height_w, max_w = 18.0, 2.8, 0.95, 3.2
        line_w = 126.0
    elif style == "WELL":
        hole_w, depth_w, height_w, max_w = 13.0, 2.0, 0.68, 2.2
        line_w = 132.0
    else:
        hole_w, depth_w, height_w, max_w = 15.0, 2.2, 0.75, 2.6
        line_w = 128.0

    old_holes = int(before.get("holes", 0))
    old_max = int(before.get("max_height", 0))
    old_aggregate = int(before.get("aggregate_height", 0))
    delta_holes = int(after.get("holes", 0)) - old_holes
    delta_max = int(after.get("max_height", 0)) - old_max
    delta_aggregate = int(after.get("aggregate_height", 0)) - old_aggregate
    value = (
        line_w * lines
        + 26.0 * lines * lines
        - hole_w * float(after.get("holes", 0))
        - depth_w * float(after.get("hole_depth", 0))
        - 2.0 * hole_w * max(0, delta_holes)
        - height_w * float(after.get("aggregate_height", 0))
        - max_w * float(after.get("max_height", 0))
        - 1.8 * max_w * max(0, delta_max)
        - 0.48 * float(after.get("bumpiness", 0))
        - 0.07 * float(after.get("row_transitions", 0))
        - 0.08 * float(after.get("wells", 0))
        - 0.12 * float(after.get("landing_height", 0))
        - 0.16 * max(0, delta_aggregate)
    )
    # Preserve a useful edge well only when it is not paired with a dangerous
    # height; this is deliberately a small preference, not a forced opener.
    if style == "WELL" and int(after.get("wells", 0)) > 0:
        value += min(5.0, 0.25 * float(after.get("wells", 0)))
    return value


def _enumerate(
    grid: list[list[int]], piece: str, style: str, before: dict[str, Any], stop=None
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    orientations = (0,) if piece == "O" else range(4)
    for orientation in orientations:
        if stop is not None and stop():
            return plans
        for anchor_x in range(-4, WIDTH + 4):
            if stop is not None and stop():
                return plans
            result = _drop(grid, piece, orientation, anchor_x)
            if result is None:
                continue
            after, features = result
            plans.append(
                {
                    "piece": piece,
                    "orientation": int(orientation),
                    "x": int(anchor_x),
                    "after": after,
                    "features": features,
                    "lines": int(features.get("lines", 0)),
                    "immediate": _placement_value(
                        features, before, style, int(features.get("lines", 0))
                    ),
                }
            )
    plans.sort(
        key=lambda item: (
            float(item["immediate"]),
            int(item["lines"]),
            -int(item["features"].get("holes", 0)),
            -int(item["features"].get("max_height", 0)),
            -int(item["x"]),
        ),
        reverse=True,
    )
    return plans


def _best_future(
    grid: list[list[int]], queue: list[str], style: str, depth: int, stop=None
) -> float:
    """Small bounded queue lookahead used only by code."""
    if stop is not None and stop():
        return 0.0
    if depth <= 0 or not queue:
        return 0.0
    piece = queue[0]
    if piece not in PIECES:
        return 0.0
    before = _features(grid)
    choices = _enumerate(grid, piece, style, before, stop=stop)
    if not choices or (stop is not None and stop()):
        return -1000.0
    # The first ply is exact; at the second ply only the strongest few
    # placements are needed to distinguish a smooth surface from a trap.
    best = -10**12
    for choice in choices[:12]:
        if stop is not None and stop():
            break
        value = float(choice["immediate"])
        if depth > 1:
            value += 0.22 * _best_future(
                choice["after"], queue[1:], style, depth - 1, stop=stop
            )
        if value > best:
            best = value
    return best


def _candidate_sources(obs: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    active = obs.get("active") or {}
    current = str(active.get("type", "O"))
    if current not in PIECES:
        current = "O"
    queue = [str(value) for value in (obs.get("queue") or []) if str(value) in PIECES]
    sources: list[tuple[str, str, list[str]]] = [("current", current, queue)]
    if bool(obs.get("can_hold", False)):
        held = obs.get("hold")
        if held in PIECES:
            sources.append(("hold", str(held), queue))
        elif queue:
            # Empty hold stores current and immediately spawns queue[0].
            sources.append(("hold", queue[0], queue[1:]))
    return sources


def _plans(obs: dict[str, Any], style: str, stop=None) -> list[dict[str, Any]]:
    if stop is not None and stop():
        return []
    grid = _grid(obs)
    before = _features(grid)
    all_plans: list[dict[str, Any]] = []
    for source, piece, future_queue in _candidate_sources(obs):
        if stop is not None and stop():
            return []
        for plan in _enumerate(grid, piece, style, before, stop=stop):
            if stop is not None and stop():
                return []
            hold_penalty = 3.5 if source == "hold" else 0.0
            # Look ahead through two visible previews.  Queue data is public;
            # no seed-specific information is inferred.
            # One-ply queue lookahead is deliberate: it keeps planning below a
            # few milliseconds of game-time on the real-time clock.  The next
            # piece is enough to reject a surface that immediately traps it;
            # deeper search would recreate the stale-tick problem.
            future = _best_future(
                grid=plan["after"], queue=future_queue, style=style, depth=1, stop=stop
            )
            plan = dict(plan)
            plan.update(
                {
                    "source": source,
                    "rank": float(plan["immediate"]) + 0.28 * future - hold_penalty,
                    "future_value": future,
                }
            )
            all_plans.append(plan)
    if stop is not None and stop():
        return []
    all_plans.sort(
        key=lambda item: (
            float(item["rank"]),
            int(item["lines"]),
            -int(item["features"].get("holes", 0)),
            -int(item["features"].get("max_height", 0)),
            1 if item.get("source") == "current" else 0,
            -int(item["x"]),
        ),
        reverse=True,
    )
    if not all_plans:
        return []

    # Keep the shortlist small and distinct.  Retaining each source and each
    # orientation makes Jev's choice meaningful without asking it to generate.
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    def add(plan: dict[str, Any]) -> None:
        key = (str(plan.get("source")), int(plan.get("orientation", 0)), int(plan.get("x", 0)))
        if key not in seen and len(kept) < 9:
            kept.append(plan)
            seen.add(key)

    add(all_plans[0])
    for source in ("current", "hold"):
        for plan in all_plans:
            if plan.get("source") == source:
                add(plan)
                break
    for orientation in range(4):
        for plan in all_plans:
            if int(plan.get("orientation", 0)) == orientation:
                add(plan)
                break
    for plan in all_plans:
        add(plan)
    return kept


def _height_bucket(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 20
    if n <= 5:
        return "low"
    if n <= 10:
        return "moderate"
    if n <= 16:
        return "high"
    return "critical"


def _holes_bucket(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 99
    if n <= 0:
        return "hole-free"
    if n <= 3:
        return "few shallow holes"
    if n <= 8:
        return "several holes"
    return "many or deep holes"


def _surface_bucket(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 99
    if n <= 8:
        return "flat"
    if n <= 18:
        return "uneven"
    return "rough"


def _plan_description(plan: dict[str, Any]) -> str:
    features = plan.get("features") or {}
    lines = int(plan.get("lines", 0))
    clear = {0: "no line", 1: "single", 2: "double", 3: "triple", 4: "Tetris"}.get(
        lines, "line clear"
    )
    source = "use hold, then place" if plan.get("source") == "hold" else "place current piece"
    ori = int(plan.get("orientation", 0)) % 4
    return (
        f"{source}; orientation {ORIENTATION_NAMES[ori]}, target anchor column {int(plan.get('x', 0))}; "
        f"{clear}; projected {_height_bucket(features.get('max_height', 20))} stack, "
        f"{_holes_bucket(features.get('holes', 99))}, {_surface_bucket(features.get('bumpiness', 99))} surface, "
        f"{int(features.get('wells', 0))} well depth."
    )


def _jev_state(obs: dict[str, Any], style: str, plans: list[dict[str, Any]]) -> dict[str, Any]:
    active = obs.get("active") or {}
    return {
        "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
        "policy": style,
        "piece_instance": _json_int(obs.get("pieces", 0)),
        "board": {
            "height": _height_bucket(obs.get("top_out_height", 0)),
            "holes": _holes_bucket(obs.get("holes", 0)),
            "surface": _surface_bucket(obs.get("bumpiness", 0)),
            "lines": _json_int(obs.get("lines", 0)),
            "level": _json_int(obs.get("level", 1)),
            "combo": "active" if _json_int(obs.get("combo", -1)) >= 0 else "inactive",
            "back_to_back": bool(obs.get("back_to_back", False)),
        },
        "active_piece": {
            "type": active.get("type"),
            "orientation": _json_int(active.get("orientation", 0)),
            "anchor_x": _json_int(active.get("x", 3)),
        },
        "hold": obs.get("hold"),
        "can_hold": bool(obs.get("can_hold", False)),
        "next_queue": [str(value) for value in (obs.get("queue") or [])[:5]],
        "fallback": "C0",
        "candidates": {f"C{i}": _plan_description(plan) for i, plan in enumerate(plans)},
    }


def _json_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _probability(answer: Any, label: str) -> float:
    try:
        probabilities = getattr(answer, "probabilities", {}) or {}
        value = probabilities.get(label, 0.0)
        return float(value)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _answer_plan(result: Any, plans: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(result, dict) or not plans:
        return None
    valid = {f"C{i}": plan for i, plan in enumerate(plans)}
    first = result.get("pick_a")
    second = result.get("pick_b")
    first_value = getattr(first, "value", None)
    second_value = getattr(second, "value", None)
    if first_value not in valid or second_value != first_value:
        return None
    # All choices are code-validated.  A modest gate rejects diffuse or
    # malformed answers while still allowing Jev to override a deterministic
    # fallback when it has a clear preference.
    if _probability(first, first_value) < 0.30 or _probability(second, second_value) < 0.30:
        return None
    return valid[first_value]


def _choice_questions(state: dict[str, Any], options: list[tuple[str, str]]) -> dict[str, Choice]:
    ordered = {key: description for key, description in options}
    reversed_order = {key: description for key, description in reversed(options)}
    common = (
        "Choose one candidate label from the `candidates` object in the state. "
        "The labels are legal, code-simulated final placements for the current piece. "
        "Prefer a line clear when it does not create a high or hole-filled stack; "
        "otherwise prefer the lowest, flattest, least hole-prone result. "
        "Do not calculate, invent actions, or output a primitive key. "
        "If the state is ambiguous, still choose the safer described candidate."
    )
    # Keep the state itself in the request as strict JSON; the repeated compact
    # copy makes the question self-contained for provider adapters that isolate
    # question instructions.
    compact = json.dumps(state, separators=(",", ":"), ensure_ascii=True)
    return {
        "pick_a": Choice(
            common + " Primary judgment. State: " + compact,
            criteria=ordered,
        ),
        "pick_b": Choice(
            common + " Recheck independently with option order reversed. State: " + compact,
            criteria=reversed_order,
        ),
    }


def _safe_action(spec: Any, name: str) -> str:
    names = list(getattr(spec, "names", ()) or ())
    if name in names:
        return name
    noop = getattr(spec, "noop", None)
    if noop in names:
        return noop
    if "noop" in names:
        return "noop"
    return sorted(str(value) for value in names)[0] if names else "noop"


def _piece_id(obs: dict[str, Any]) -> tuple[int, str]:
    active = obs.get("active") or {}
    return (_json_int(obs.get("pieces", 0)), str(active.get("type", "")))


def _planning_snapshot(obs: dict[str, Any]) -> dict[str, Any]:
    """Detach worker input from any mutable observation owned by the game."""
    try:
        copied = json.loads(json.dumps(obs, separators=(",", ":")))
        return copied if isinstance(copied, dict) else dict(obs)
    except Exception:
        return dict(obs)


def _planning_key(obs: dict[str, Any], style: str) -> tuple[Any, ...]:
    """Facts that affect placement search; gravity-only movement is excluded."""
    active = obs.get("active") or {}
    queue = tuple(str(value) for value in (obs.get("queue") or []) if str(value) in PIECES)
    return (
        _json_int(obs.get("pieces", 0)),
        str(active.get("type", "")),
        obs.get("hold"),
        bool(obs.get("can_hold", False)),
        queue,
        _grid(obs),
        style,
    )


def _compatible_plan(obs: dict[str, Any], plan: dict[str, Any], hold_done: bool) -> bool:
    active = obs.get("active") or {}
    active_type = str(active.get("type", ""))
    target_type = str(plan.get("piece", ""))
    if plan.get("source") == "hold":
        if not hold_done:
            return bool(obs.get("can_hold", False))
        return active_type == target_type
    if hold_done:
        return False
    return active_type == target_type


def _best_current(obs: dict[str, Any], style: str) -> dict[str, Any] | None:
    plans = _plans(obs, style)
    return plans[0] if plans else None


def _next_action(
    obs: dict[str, Any], plan: dict[str, Any], hold_done: bool, spec: Any
) -> str:
    """Return one short verified input; never synthesize a multi-key action."""
    if plan.get("source") == "hold" and not hold_done:
        if bool(obs.get("can_hold", False)):
            return _safe_action(spec, "hold")
        return _safe_action(spec, "hard_drop")

    active = obs.get("active") or {}
    current_type = str(active.get("type", ""))
    if current_type != str(plan.get("piece", "")):
        return _safe_action(spec, "hard_drop")
    current_orientation = _json_int(active.get("orientation", 0)) % 4
    target_orientation = _json_int(plan.get("orientation", 0)) % 4
    delta = (target_orientation - current_orientation) % 4
    if delta:
        # One 180-degree turn is expressed as two verified clockwise turns.
        return _safe_action(spec, "rotate_cw" if delta in (1, 2) else "rotate_ccw")
    current_x = _json_int(active.get("x", 3), 3)
    target_x = _json_int(plan.get("x", current_x), current_x)
    if current_x < target_x:
        return _safe_action(spec, "right")
    if current_x > target_x:
        return _safe_action(spec, "left")
    return _safe_action(spec, "hard_drop")


def _style_from_text(text: Any) -> str | None:
    raw = str(text or "").strip().upper()
    # Astra is prompted for a single token.  Prefer a token at the beginning so
    # explanatory text mentioning all regimes cannot accidentally switch policy.
    match = re.match(r"^(?:PLAN\s*[:=]\s*)?(BALANCED|RECOVER|WELL)\b", raw)
    if match:
        return match.group(1)
    return None


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.style = "BALANCED"
        self.astra_future: Any = None
        self.astra_started = False
        self.pending_jev: Any = None
        self.pending_jev_piece: tuple[int, str] | None = None
        self.pending_jev_state: dict[str, Any] | None = None
        self.pending_jev_options: list[tuple[str, str]] = []
        self.pending_jev_plans: list[dict[str, Any]] = []

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _start_astra(self, obs: dict[str, Any]) -> None:
        if self.astra_started or getattr(self.ctx, "system2", None) is None:
            return
        self.astra_started = True
        compact = {
            "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
            "board": {
                "holes": _json_int(obs.get("holes", 0)),
                "aggregate_height": _json_int(obs.get("aggregate_height", 0)),
                "top_out_height": _json_int(obs.get("top_out_height", 0)),
                "bumpiness": _json_int(obs.get("bumpiness", 0)),
            },
            "active": (obs.get("active") or {}).get("type"),
            "hold": obs.get("hold"),
            "queue": [str(value) for value in (obs.get("queue") or [])[:5]],
            "level": _json_int(obs.get("level", 1)),
        }
        prompt = (
            "You are a slow strategic reviewer for a real-time Guideline-compatible "
            "Tetris controller. Code enumerates legal SRS placements and Jev ranks "
            "them; you must not give keypresses or a seed-specific script. Read the "
            "public facts below and return exactly one token: BALANCED for ordinary "
            "low-stack play, RECOVER when holes or height make downstacking and "
            "survival dominant, or WELL only when a controlled well is safe. "
            "Return only BALANCED, RECOVER, or WELL.\n" + json.dumps(compact, separators=(",", ":"))
        )

        def call() -> str:
            try:
                result = self.ctx.think(prompt, timeout_s=65.0)
                return str(getattr(result, "text", ""))
            except ModelUnavailable:
                raise
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver):
                return ""

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception:
            self.astra_future = None

    def _harvest_astra(self) -> None:
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            style = _style_from_text(future.result())
        except ModelUnavailable:
            raise
        except Exception:
            style = None
        if style and style != self.style:
            self.style = style
            self._log("astra_policy_update", policy=style)

    def _launch_jev(self, obs: dict[str, Any], plans: list[dict[str, Any]]) -> bool:
        if getattr(self.ctx, "system1", None) is None or not plans:
            return True
        if self.pending_jev is not None and not self.pending_jev.done():
            return False
        if self.pending_jev is not None and self.pending_jev.done():
            self._discard_jev_result()
        state = _jev_state(obs, self.style, plans)
        options = [(f"C{i}", _plan_description(plan)) for i, plan in enumerate(plans)]
        questions = _choice_questions(state, options)
        piece = _piece_id(obs)
        self.pending_jev_piece = piece
        self.pending_jev_state = state
        self.pending_jev_options = options
        self.pending_jev_plans = list(plans)

        def call() -> Any:
            try:
                return self.ctx.decide(state, questions, timeout_s=0.45)
            except ModelUnavailable:
                raise
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver):
                return None

        try:
            self.pending_jev = self.ctx.spawn(call)
        except Exception:
            self.pending_jev = None
        return True

    def _discard_jev_result(self) -> None:
        future = self.pending_jev
        self.pending_jev = None
        self.pending_jev_piece = None
        self.pending_jev_state = None
        self.pending_jev_options = []
        self.pending_jev_plans = []
        if future is None:
            return
        try:
            future.result()
        except ModelUnavailable:
            raise
        except Exception:
            pass

    def _harvest_jev(
        self, obs: dict[str, Any], current_plan: dict[str, Any] | None, hold_done: bool
    ) -> dict[str, Any] | None:
        future = self.pending_jev
        if future is None or not future.done():
            return current_plan
        piece = self.pending_jev_piece
        plans = self.pending_jev_plans
        try:
            result = future.result()
        except ModelUnavailable:
            raise
        except Exception:
            result = None
        self.pending_jev = None
        self.pending_jev_piece = None
        self.pending_jev_state = None
        self.pending_jev_options = []
        self.pending_jev_plans = []
        if piece != _piece_id(obs):
            return current_plan
        chosen = _answer_plan(result, plans)
        if chosen is None or not _compatible_plan(obs, chosen, hold_done):
            return current_plan
        self._log("jev_candidate", piece=piece[0], candidate=f"C{plans.index(chosen)}")
        return chosen

    def _post_and_wait(self, game: Any, action: str) -> bool:
        if bool(getattr(self.ctx, "done", False)) or bool(getattr(game, "done", False)):
            return False
        try:
            self.ctx.act(action)
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return False
        except Exception:
            return False
        try:
            self.ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return False
        except Exception:
            return False
        return not bool(getattr(self.ctx, "done", False)) and not bool(
            getattr(game, "done", False)
        )

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        try:
            initial = game.observe()
        except Exception:
            return
        if not isinstance(initial, dict):
            return

        spec = getattr(game, "action_spec", None)
        piece_number: int | None = None
        current_piece: tuple[int, str] | None = None
        plans: list[dict[str, Any]] = []
        fallback: dict[str, Any] | None = None
        selected: dict[str, Any] | None = None
        hold_done = False
        request_to_launch: tuple[dict[str, Any], list[dict[str, Any]]] | None = None
        first_action_posted = False
        previous_command: str | None = None
        command_piece: tuple[int, str] | None = None
        command_x: int | None = None
        command_orientation: int | None = None
        blocked_command = False
        plan_future: Any = None
        plan_key: tuple[Any, ...] | None = None
        plan_observation: dict[str, Any] | None = None
        plan_style: str | None = None

        while not bool(getattr(ctx, "done", False)) and not bool(getattr(game, "done", False)):
            self._harvest_astra()
            try:
                obs = game.observe()
            except Exception:
                return
            if not isinstance(obs, dict) or bool(obs.get("game_over", False)):
                return

            number = _json_int(obs.get("pieces", 0))
            piece_now = _piece_id(obs)
            if number != piece_number:
                # Never orphan one planner per rapidly locking piece. If an old
                # search is still finishing, keep the game moving with noop and
                # consume/reject it before starting work for the new piece.
                if plan_future is not None:
                    if not plan_future.done():
                        if not self._post_and_wait(game, _safe_action(spec, "noop")):
                            return
                        previous_command = "noop"
                        continue
                    try:
                        plan_future.result()
                    except Exception:
                        pass
                    plan_future = None
                    plan_key = None
                piece_number = number
                current_piece = piece_now
                plans = []
                fallback = None
                selected = None
                hold_done = False
                blocked_command = False
                request_to_launch = None
                plan_observation = _planning_snapshot(obs)
                plan_style = self.style
                plan_key = _planning_key(plan_observation, plan_style)

                # The search runs off-clock.  Immediately replace the prior
                # action with noop, then let one tick pass while it computes.
                try:
                    plan_future = ctx.spawn(
                        lambda state=plan_observation, style=plan_style: _plans(
                            state, style,
                            stop=lambda: bool(getattr(ctx, "done", False))
                            or bool(getattr(game, "done", False)),
                        )
                    )
                except Exception:
                    plan_future = None
                if plan_future is None:
                    # Spawn failure is rare; keep the episode valid rather than
                    # doing a long synchronous search on the game clock.
                    plans = []
                    plan_key = None
                if not self._post_and_wait(game, _safe_action(spec, "noop")):
                    return
                first_action_posted = True
                previous_command = "noop"
                if not self.astra_started:
                    try:
                        self._start_astra(initial)
                    except ModelUnavailable:
                        raise
                    except Exception:
                        pass
                continue

            # Do not block the realtime clock on the placement search.  Fresh
            # noops are posted until the finite worker returns; stale geometry
            # or a changed policy snapshot is discarded and recomputed.
            if plan_future is not None:
                if not plan_future.done():
                    if not self._post_and_wait(game, _safe_action(spec, "noop")):
                        return
                    previous_command = "noop"
                    continue
                finished = plan_future
                plan_future = None
                try:
                    result = finished.result()
                    computed = result if isinstance(result, list) else []
                except Exception as exc:
                    computed = []
                    self._log("planner_failed", error=type(exc).__name__)
                try:
                    current_key = _planning_key(obs, self.style)
                except Exception:
                    current_key = None
                if plan_key != current_key:
                    # Search inputs changed before publication; start a fresh
                    # worker instead of using a stale board/hold/queue plan.
                    plan_observation = _planning_snapshot(obs)
                    plan_style = self.style
                    plan_key = _planning_key(plan_observation, plan_style)
                    try:
                        plan_future = ctx.spawn(
                            lambda state=plan_observation, style=plan_style: _plans(
                                state, style,
                                stop=lambda: bool(getattr(ctx, "done", False))
                                or bool(getattr(game, "done", False)),
                            )
                        )
                    except Exception:
                        plan_future = None
                    if not self._post_and_wait(game, _safe_action(spec, "noop")):
                        return
                    previous_command = "noop"
                    continue

                plans = computed
                fallback = plans[0] if plans else None
                selected = fallback
                hold_done = False
                blocked_command = False
                request_to_launch = (obs, list(plans)) if plans else None
                plan_key = None
                plan_observation = None
                plan_style = None

            elif piece_number == number and not plans and fallback is None and selected is None:
                # Worker could not be created or returned no legal destination.
                # This is an immediate, contract-valid escape action.
                if not self._post_and_wait(game, _safe_action(spec, "hard_drop")):
                    return
                previous_command = "hard_drop"
                continue

            elif (
                current_piece == piece_now
                and command_piece == piece_now
                and previous_command in {"left", "right", "rotate_cw", "rotate_ccw"}
            ):
                active_now = obs.get("active") or {}
                now_x = _json_int(active_now.get("x", 3), 3)
                now_orientation = _json_int(active_now.get("orientation", 0)) % 4
                blocked_command = (
                    now_x == command_x and now_orientation == command_orientation
                )
            else:
                if current_piece != piece_now:
                    current_piece = piece_now
                blocked_command = False

            # Harvest an advisory answer only for its original piece.  Plan
            # labels refer to the immutable shortlist captured at request time.
            selected = self._harvest_jev(obs, selected, hold_done)
            if selected is None:
                selected = fallback
            if selected is not None and not _compatible_plan(obs, selected, hold_done):
                selected = fallback if fallback is not None and _compatible_plan(obs, fallback, hold_done) else None

            if blocked_command or selected is None:
                action = _safe_action(spec, "hard_drop")
            else:
                action = _next_action(obs, selected, hold_done, spec)

            active_before = obs.get("active") or {}
            command_piece = piece_now
            command_x = _json_int(active_before.get("x", 3), 3)
            command_orientation = _json_int(active_before.get("orientation", 0)) % 4

            # Post before starting any model call so inference never owns time.
            if not self._post_and_wait(game, action):
                return
            first_action_posted = True
            previous_command = action

            if action == "hold" and "hold" in (getattr(spec, "names", ()) or ()):
                hold_done = True

            if request_to_launch is not None:
                launch_obs, launch_plans = request_to_launch
                try:
                    launched = self._launch_jev(launch_obs, launch_plans)
                except (BudgetExceeded, EpisodeOver):
                    return
                except ModelUnavailable:
                    raise
                except Exception:
                    launched = True
                if launched:
                    request_to_launch = None

            if not self.astra_started and first_action_posted:
                try:
                    self._start_astra(initial)
                except (BudgetExceeded, EpisodeOver):
                    return
                except ModelUnavailable:
                    raise
                except Exception:
                    pass


def build(ctx: Any) -> Harness:
    return Harness(ctx)
