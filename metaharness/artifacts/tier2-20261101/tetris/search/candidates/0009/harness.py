"""Staged queue search with a verified real-time Tetris executor.

The controller never lets a model call be the clock driver.  Code enumerates
Guideline-shaped placements, evaluates a bounded queue continuation, and
executes a verified rotation/translation/drop sequence.  Jev only chooses
among already legal destinations; its absence is harmless.
"""
from __future__ import annotations

import json
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
    """Safety-first value.  The engine, not a model, does all arithmetic."""
    # Keep one stable evaluator in this candidate.  A model-authored style
    # cannot silently change the ranking, and the value is intentionally close
    # to 0007's proven low-hole/low-height regime.
    old_holes = int(before.get("holes", 0))
    old_max = int(before.get("max_height", 0))
    old_aggregate = int(before.get("aggregate_height", 0))
    delta_holes = int(after.get("holes", 0)) - old_holes
    delta_max = int(after.get("max_height", 0)) - old_max
    delta_aggregate = int(after.get("aggregate_height", 0)) - old_aggregate
    value = (
        128.0 * lines
        + 26.0 * lines * lines
        - 15.0 * float(after.get("holes", 0))
        - 2.2 * float(after.get("hole_depth", 0))
        - 30.0 * max(0, delta_holes)
        - 0.75 * float(after.get("aggregate_height", 0))
        - 2.6 * float(after.get("max_height", 0))
        - 4.7 * max(0, delta_max)
        - 0.48 * float(after.get("bumpiness", 0))
        - 0.07 * float(after.get("row_transitions", 0))
        - 0.08 * float(after.get("wells", 0))
        - 0.12 * float(after.get("landing_height", 0))
        - 0.16 * max(0, delta_aggregate)
    )
    return value


def _enumerate(
    grid: list[list[int]], piece: str, style: str, before: dict[str, Any]
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    orientations = (0,) if piece == "O" else range(4)
    for orientation in orientations:
        shape = SHAPES.get(piece, SHAPES["O"])[orientation % 4]
        min_dx = min(dx for dx, _ in shape)
        max_dx = max(dx for dx, _ in shape)
        for anchor_x in range(-min_dx, WIDTH - max_dx):
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
    grid: list[list[int]], queue: list[str], style: str, depth: int,
    width: int = 6,
) -> float:
    """Bounded queue continuation; never expands the whole tree."""
    if depth <= 0 or not queue:
        return 0.0
    piece = queue[0]
    if piece not in PIECES:
        return 0.0
    before = _features(grid)
    choices = _enumerate(grid, piece, style, before)
    if not choices:
        return -1000.0
    best = -10**12
    for choice in choices[:max(1, width)]:
        value = float(choice["immediate"])
        if depth > 1:
            value += 0.24 * _best_future(
                choice["after"], queue[1:], style, depth - 1,
                width=max(3, width // 2),
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


def _plans(obs: dict[str, Any], style: str) -> list[dict[str, Any]]:
    """Rank broadly, then spend queue-search work only on a diverse shortlist.

    0008's deep beam was allowed to run before an action was posted, so stale
    hard drops became the controller.  This staged form keeps the useful part
    of that beam while bounding the expensive continuation work.
    """
    grid = _grid(obs)
    before = _features(grid)
    all_plans: list[dict[str, Any]] = []
    for source, piece, future_queue in _candidate_sources(obs):
        for plan in _enumerate(grid, piece, style, before):
            item = dict(plan)
            item.update({
                "source": source,
                "future_queue": tuple(future_queue),
                "future_value": 0.0,
                "rank": float(plan["immediate"])
                       - (3.5 if source == "hold" else 0.0),
            })
            all_plans.append(item)
    if not all_plans:
        return []

    # Immediate ranking is cheap and lets the continuation stage ignore most
    # geometrically legal but plainly inferior destinations.  Preserve both
    # hold sources if possible; an empty-hold opener is still a real option.
    all_plans.sort(key=lambda item: (
        float(item["immediate"]), int(item["lines"]),
        -int(item["features"].get("holes", 0)),
        -int(item["features"].get("max_height", 0)),
        1 if item.get("source") == "current" else 0,
        -int(item["x"]),
    ), reverse=True)
    focus: list[dict[str, Any]] = []
    focus_keys: set[tuple[str, int, int]] = set()

    def focus_add(item: dict[str, Any]) -> None:
        key = (str(item.get("source")), int(item.get("orientation", 0)),
               int(item.get("x", 0)))
        if key not in focus_keys and len(focus) < 8:
            focus_keys.add(key)
            focus.append(item)

    for item in all_plans[:8]:
        focus_add(item)
    for source in ("current", "hold"):
        for item in all_plans:
            if item.get("source") == source:
                focus_add(item)
                break

    # At low gravity the second continuation is affordable; at fast gravity
    # retain only a one-piece check.  Only the first three focus plans receive
    # the second-ply check, keeping the maximum work bounded.
    try:
        gravity = float(obs.get("gravity_s_per_cell", 1.0))
    except (TypeError, ValueError):
        gravity = 1.0
    deep = gravity >= 0.12
    for index, item in enumerate(focus):
        remaining = list(item.get("future_queue") or ())
        depth = 2 if deep and index < 3 else 1
        future = _best_future(
            grid=item["after"], queue=remaining, style=style,
            depth=depth, width=6 if index < 3 else 4,
        )
        item["future_value"] = future
        item["rank"] = float(item["immediate"]) + 0.28 * future \
            - (3.5 if item.get("source") == "hold" else 0.0)

    all_plans.sort(key=lambda item: (
        float(item.get("rank", item["immediate"])),
        int(item["lines"]),
        -int(item["features"].get("holes", 0)),
        -int(item["features"].get("max_height", 0)),
        1 if item.get("source") == "current" else 0,
        -int(item["x"]),
    ), reverse=True)

    # Keep a small, diverse named interface for Jev.  Unevaluated plans retain
    # their immediate rank and can still be the deterministic fallback only if
    # the staged ranking says they are best.
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    def add(item: dict[str, Any]) -> None:
        key = (str(item.get("source")), int(item.get("orientation", 0)),
               int(item.get("x", 0)))
        if key not in seen and len(kept) < 9:
            seen.add(key)
            kept.append(item)

    add(all_plans[0])
    for source in ("current", "hold"):
        for item in all_plans:
            if item.get("source") == source:
                add(item)
                break
    for orientation in range(4):
        for item in all_plans:
            if int(item.get("orientation", 0)) == orientation:
                add(item)
                break
    for item in all_plans:
        add(item)
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
    future_value = float(plan.get("future_value", 0.0) or 0.0)
    future = ("strong future support" if future_value > 25.0 else
              "some future support" if future_value > -35.0 else
              "weak future support")
    return (
        f"{source}; orientation {ORIENTATION_NAMES[ori]}, target anchor column {int(plan.get('x', 0))}; "
        f"{clear}; projected {_height_bucket(features.get('max_height', 20))} stack, "
        f"{_holes_bucket(features.get('holes', 99))}, {_surface_bucket(features.get('bumpiness', 99))} surface, "
        f"{future}."
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
    if _probability(first, first_value) < 0.40 or _probability(second, second_value) < 0.40:
        return None
    return valid[first_value]


def _choice_questions(state: dict[str, Any], options: list[tuple[str, str]]) -> dict[str, Choice]:
    ordered = {key: description for key, description in options}
    reversed_order = {key: description for key, description in reversed(options)}
    common = (
        "Choose one candidate label from the `candidates` object in the state. "
        "The labels are legal, code-simulated final placements for the current piece. "
        "Prefer reducing dangerous holes and top-out risk while keeping a low, smooth "
        "surface; take a line clear when it does not leave a worse trapped surface. "
        "Use the described future support as a tie-breaker. Do not calculate, invent "
        "actions, or output a primitive key. If the state is ambiguous, still choose "
        "the safer described candidate."
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


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.style = "BALANCED"
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
        previous_command: str | None = None
        command_piece: tuple[int, str] | None = None
        command_x: int | None = None
        command_orientation: int | None = None
        blocked_command = False

        while not bool(getattr(ctx, "done", False)) and not bool(getattr(game, "done", False)):
            try:
                obs = game.observe()
            except Exception:
                return
            if not isinstance(obs, dict) or bool(obs.get("game_over", False)):
                return

            number = _json_int(obs.get("pieces", 0))
            piece_now = _piece_id(obs)
            if number != piece_number:
                piece_number = number
                current_piece = piece_now
                # Search takes about a tick on a normal interpreter.  Post a
                # safe noop before doing it so a just-locked hard drop cannot
                # be replayed into the new piece while plans are computed.
                try:
                    ctx.act(_safe_action(spec, "noop"))
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except Exception:
                    return
                plans = _plans(obs, self.style)
                try:
                    latest = game.observe()
                except Exception:
                    return
                if not isinstance(latest, dict) or bool(latest.get("game_over", False)):
                    return
                if _piece_id(latest) != piece_now:
                    # Gravity may have locked a very fast piece during search;
                    # the next loop will install a plan for the new instance.
                    continue
                obs = latest
                fallback = plans[0] if plans else None
                selected = fallback
                hold_done = False
                blocked_command = False
                request_to_launch = (obs, list(plans)) if plans else None
            elif (
                current_piece == piece_now
                and command_piece == piece_now
                and previous_command in {"left", "right", "rotate_cw", "rotate_ccw"}
            ):
                active_now = obs.get("active") or {}
                now_x = _json_int(active_now.get("x", 3), 3)
                now_orientation = _json_int(active_now.get("orientation", 0)) % 4
                # A legal movement or SRS rotation should change x or
                # orientation.  If it did not, abandon the target rather than
                # repeating a blocked input until lock.
                blocked_command = (
                    now_x == command_x and now_orientation == command_orientation
                )
            else:
                if current_piece != piece_now:
                    # A successful hold changes active.type without increasing
                    # pieces; the replacement is a fresh geometry instance.
                    current_piece = piece_now
                blocked_command = False

            # An answer can change only the still-falling piece.  The executor
            # rechecks hold/type compatibility before honoring it.
            selected = self._harvest_jev(obs, selected, hold_done)
            if selected is None:
                selected = fallback
            if selected is not None and not _compatible_plan(obs, selected, hold_done):
                refreshed = _best_current(obs, self.style)
                fallback = refreshed or fallback
                selected = fallback

            if blocked_command or selected is None:
                action = _safe_action(spec, "hard_drop")
            else:
                action = _next_action(obs, selected, hold_done, spec)

            active_before = obs.get("active") or {}
            command_piece = piece_now
            command_x = _json_int(active_before.get("x", 3), 3)
            command_orientation = _json_int(active_before.get("orientation", 0)) % 4

            # Post the deterministic action first.  Starting a background model
            # request after posting removes the boundary stall seen in parents.
            if not self._post_and_wait(game, action):
                return
            previous_command = action

            if action == "hold" and "hold" in (getattr(spec, "names", ()) or ()):
                hold_done = True
                # A hold changes active.type but not pieces; revalidate the
                # selected target on the next observation rather than assuming
                # the replacement spawned exactly where search predicted.

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

            # If a Jev request finished after the piece locked, consume and
            # discard it on the next iteration.  A new request is launched only
            # once the previous future is no longer in flight.


def build(ctx: Any) -> Harness:
    return Harness(ctx)
