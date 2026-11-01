"""Guardian/refinement Tetris controller for a real-time Guideline game.

Code owns exact placement simulation and input routing.  A cheap guardian
placement is executable immediately while a one-preview refinement search runs
in the background; Jev chooses a semantic placement role and Astra authors a
risk regime, but neither model ever supplies a primitive keypress.
"""
from __future__ import annotations

from collections import deque
import copy
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
    # The parent controller was excellent at surviving but usually converted
    # its safe surface into singles.  Give multi-line clears a stronger,
    # still-safe preference: this is a ranking signal, not a permission to
    # accept new holes or a tall landing.
    value = (
        line_w * lines
        + 70.0 * lines * lines
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
    grid: list[list[int]], piece: str, style: str, before: dict[str, Any]
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    orientations = (0,) if piece == "O" else range(4)
    for orientation in orientations:
        for anchor_x in range(-4, WIDTH + 4):
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
    grid: list[list[int]], queue: list[str], style: str, depth: int
) -> float:
    """Small bounded queue lookahead used only by code."""
    if depth <= 0 or not queue:
        return 0.0
    piece = queue[0]
    if piece not in PIECES:
        return 0.0
    before = _features(grid)
    choices = _enumerate(grid, piece, style, before)
    if not choices:
        return -1000.0
    # The first ply is exact; at the second ply only the strongest few
    # placements are needed to distinguish a smooth surface from a trap.
    best = -10**12
    for choice in choices[:12]:
        value = float(choice["immediate"])
        if depth > 1:
            value += 0.22 * _best_future(
                choice["after"], queue[1:], style, depth - 1
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
    grid = _grid(obs)
    before = _features(grid)
    all_plans: list[dict[str, Any]] = []
    for source, piece, future_queue in _candidate_sources(obs):
        for plan in _enumerate(grid, piece, style, before):
            hold_penalty = 3.5 if source == "hold" else 0.0
            # Look ahead through two visible previews.  Queue data is public;
            # no seed-specific information is inferred.
            # One-ply queue lookahead is deliberate: it keeps planning below a
            # few milliseconds of game-time on the real-time clock.  The next
            # piece is enough to reject a surface that immediately traps it;
            # deeper search would recreate the stale-tick problem.
            future = _best_future(grid=plan["after"], queue=future_queue, style=style, depth=1)
            plan = dict(plan)
            plan.update(
                {
                    "source": source,
                    "rank": float(plan["immediate"]) + 0.28 * future - hold_penalty,
                    "future_value": future,
                }
            )
            all_plans.append(plan)
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



def _coarse_plans(obs: dict[str, Any], style: str) -> list[dict[str, Any]]:
    """Return immediate placements quickly enough to guard a live piece.

    Unlike ``_plans`` this deliberately does not recurse through the preview.
    It is the foreground policy while the more discriminating one-preview
    search is running in a worker.  It is still a complete current/hold legal
    placement enumeration, not a spawn-column emergency guess.
    """
    grid = _grid(obs)
    before = _features(grid)
    result: list[dict[str, Any]] = []
    for source, piece, future_queue in _candidate_sources(obs):
        for raw in _enumerate(grid, piece, style, before):
            plan = dict(raw)
            plan.update(
                {
                    "source": source,
                    "future_queue": tuple(future_queue),
                    "future_value": 0.0,
                    "rank": float(raw.get("immediate", -10**9))
                    - (3.5 if source == "hold" else 0.0),
                }
            )
            result.append(plan)
    result.sort(
        key=lambda item: (
            float(item.get("rank", -10**9)),
            int(item.get("lines", 0)),
            -int((item.get("features") or {}).get("holes", 0)),
            -int((item.get("features") or {}).get("max_height", 0)),
            1 if item.get("source") == "current" else 0,
            -int(item.get("x", 0)),
        ),
        reverse=True,
    )
    return result[:12]


def _role_plans(obs: dict[str, Any], plans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build semantic portfolios from already simulated placements.

    Jev judges this small role set.  Code chooses the exact placement so the
    model is not asked to compare coordinates or do arithmetic.
    """
    if not plans:
        return {}

    def feats(plan: dict[str, Any]) -> dict[str, Any]:
        value = plan.get("features")
        return value if isinstance(value, dict) else {}

    # Keep the normal code ranking as the safe/balanced fallback.  It is the
    # known-good 0007 behavior; the other roles are explicit alternatives.
    safe = plans[0]
    recovery = min(
        plans,
        key=lambda item: (
            int(feats(item).get("holes", 99)),
            int(feats(item).get("hole_depth", 99)),
            int(feats(item).get("max_height", 99)),
            int(feats(item).get("aggregate_height", 999)),
            int(feats(item).get("bumpiness", 999)),
            -int(item.get("lines", 0)),
            -float(item.get("rank", -10**9)),
        ),
    )
    old_holes = _json_int(obs.get("holes", 0))
    old_max = _json_int(obs.get("top_out_height", obs.get("max_height", 0)))
    clear_pool = [
        item for item in plans
        if int(item.get("lines", 0)) > 0
        and int(feats(item).get("holes", 99)) <= old_holes + 1
        and int(feats(item).get("max_height", 99)) <= max(18, old_max + 5)
    ]
    if not clear_pool:
        clear_pool = [item for item in plans if int(item.get("lines", 0)) > 0]
    if not clear_pool:
        clear_pool = plans
    clear = max(
        clear_pool,
        key=lambda item: (
            int(item.get("lines", 0)),
            -int(feats(item).get("holes", 99)),
            -int(feats(item).get("hole_depth", 99)),
            -int(feats(item).get("max_height", 99)),
            float(item.get("rank", -10**9)),
        ),
    )
    future = max(
        plans,
        key=lambda item: (
            float(item.get("future_value", 0.0)),
            float(item.get("rank", -10**9)),
            -int(feats(item).get("holes", 99)),
            -int(feats(item).get("max_height", 99)),
        ),
    )
    return {
        "safe": safe,
        "clear": clear,
        "future": future,
        "recovery": recovery,
    }


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


def _role_description(role: str, plan: dict[str, Any]) -> str:
    purposes = {
        "safe": "ordinary low-risk placement",
        "clear": "take a clean line-clear opportunity",
        "future": "preserve the best surface for the visible queue",
        "recovery": "expose or reduce existing holes and top-out danger",
    }
    return purposes.get(role, "legal placement") + "; " + _plan_description(plan)


def _jev_state(
    obs: dict[str, Any],
    style: str,
    plans: list[dict[str, Any]],
    roles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    active = obs.get("active") or {}
    height = obs.get("max_height", obs.get("top_out_height", 0))
    return {
        "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
        "policy": style,
        "piece_instance": _json_int(obs.get("pieces", 0)),
        "board": {
            "height": _height_bucket(height),
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
        "fallback_role": "safe",
        "roles": {
            role: _role_description(role, plan) for role, plan in roles.items()
        },
        "candidate_count": len(plans),
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


def _answer_role(
    result: Any, roles: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(result, dict) or not roles:
        return None
    first = result.get("pick_a")
    second = result.get("pick_b")
    first_value = getattr(first, "value", None)
    second_value = getattr(second, "value", None)
    if first_value not in roles or second_value != first_value:
        return None
    if (
        _probability(first, first_value) < 0.45
        or _probability(second, second_value) < 0.45
    ):
        return None
    return str(first_value), roles[str(first_value)]


def _choice_questions(
    state: dict[str, Any], options: list[tuple[str, str]]
) -> dict[str, Choice]:
    ordered = {key: description for key, description in options}
    reversed_order = {key: description for key, description in reversed(options)}
    common = (
        "Choose one strategic role from the `roles` object. Code has already "
        "simulated every placement behind each role. Prefer `recovery` when "
        "the board has holes or dangerous height; prefer `clear` only when "
        "its described clear is not dangerous; otherwise prefer `safe` or "
        "`future` according to the visible queue. Do not count cells, compare "
        "raw numbers, calculate score, invent coordinates, or output a game "
        "action. Return one role label even if uncertain. State: "
        + json.dumps(state, separators=(",", ":"), ensure_ascii=True)
    )
    return {
        "pick_a": Choice(
            common + " Primary judgment.",
            criteria=ordered,
        ),
        "pick_b": Choice(
            common + " Recheck independently; the option order is reversed.",
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


def _pose_legal(
    grid: list[list[int]], piece: str, orientation: int, x: int, y: int
) -> bool:
    shape = SHAPES.get(piece, SHAPES["O"])[orientation % 4]
    for dx, dy in shape:
        cx, cy = x + dx, y + dy
        if cx < 0 or cx >= WIDTH or cy < 0:
            return False
        if cy < HEIGHT and grid[cy][cx]:
            return False
    return True


def _rotation_pose(
    grid: list[list[int]], piece: str, orientation: int, x: int, y: int,
    action: str,
) -> tuple[int, int, int] | None:
    if piece == "O":
        return None
    target = (orientation + (1 if action == "rotate_cw" else -1)) % 4
    table = _KICKS["I" if piece == "I" else "JLSTZ"]
    kicks = table.get((orientation % 4, target), ((0, 0),))
    for dx, dy in kicks:
        nx, ny = x + dx, y + dy
        if _pose_legal(grid, piece, target, nx, ny):
            return target, nx, ny
    return None


def _route_action(obs: dict[str, Any], plan: dict[str, Any], spec: Any) -> str:
    """Find one legal next input toward a named target, then replan next tick."""
    active = obs.get("active") or {}
    piece = str(active.get("type", ""))
    if piece not in PIECES or piece != str(plan.get("piece", "")):
        return _safe_action(spec, "hard_drop")
    start_o = _json_int(active.get("orientation", 0)) % 4
    start_x = _json_int(active.get("x", 3), 3)
    start_y = _json_int(active.get("y", 17), 17)
    target_o = _json_int(plan.get("orientation", start_o)) % 4
    target_x = _json_int(plan.get("x", start_x), start_x)
    if start_o == target_o and start_x == target_x:
        return _safe_action(spec, "hard_drop")

    grid = _grid(obs)
    # SRS kicks can make rotate-first fail at a crowded x.  A bounded BFS over
    # the public pose is enough to find a translate-then-rotate finesse route.
    queue = deque([(start_o, start_x, start_y, ())])
    seen = {(start_o, start_x, start_y)}
    max_depth = 9
    while queue:
        orientation, x, y, path = queue.popleft()
        if orientation == target_o and x == target_x and path:
            return _safe_action(spec, path[0])
        if len(path) >= max_depth:
            continue
        for action in ("left", "right", "rotate_cw", "rotate_ccw"):
            if action == "left":
                candidate = (orientation, x - 1, y)
                if not _pose_legal(grid, piece, *candidate):
                    continue
            elif action == "right":
                candidate = (orientation, x + 1, y)
                if not _pose_legal(grid, piece, *candidate):
                    continue
            else:
                rotated = _rotation_pose(grid, piece, orientation, x, y, action)
                if rotated is None:
                    continue
                candidate = rotated
            if candidate in seen:
                continue
            seen.add(candidate)
            next_path = path + (action,)
            if candidate[0] == target_o and candidate[1] == target_x:
                return _safe_action(spec, next_path[0])
            queue.append((candidate[0], candidate[1], candidate[2], next_path))

    # A route may disappear as gravity moves the piece.  Use a conservative
    # one-step fallback rather than repeating a blocked command indefinitely.
    if start_x < target_x and _pose_legal(grid, piece, start_o, start_x + 1, start_y):
        return _safe_action(spec, "right")
    if start_x > target_x and _pose_legal(grid, piece, start_o, start_x - 1, start_y):
        return _safe_action(spec, "left")
    if start_o != target_o:
        for action in ("rotate_cw", "rotate_ccw"):
            if _rotation_pose(grid, piece, start_o, start_x, start_y, action) is not None:
                return _safe_action(spec, action)
    return _safe_action(spec, "hard_drop")


def _next_action(
    obs: dict[str, Any], plan: dict[str, Any], hold_done: bool, spec: Any
) -> str:
    """Return one short input; target routing is verified from fresh state."""
    if plan.get("source") == "hold" and not hold_done:
        if bool(obs.get("can_hold", False)):
            return _safe_action(spec, "hold")
        return _safe_action(spec, "hard_drop")
    return _route_action(obs, plan, spec)


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
        self.astra_reviews = 0
        self.astra_last_piece = -1
        self.pending_jev: Any = None
        self.pending_jev_piece: tuple[int, str] | None = None
        self.pending_jev_state: dict[str, Any] | None = None
        self.pending_jev_options: list[tuple[str, str]] = []
        self.pending_jev_roles: dict[str, dict[str, Any]] = {}
        self.pending_search: Any = None
        self.pending_search_piece: tuple[int, str] | None = None

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _start_astra(self, obs: dict[str, Any], reason: str) -> None:
        if self.astra_reviews >= 2 or self.astra_future is not None:
            return
        if not callable(getattr(self.ctx, "think", None)):
            return
        try:
            remaining = self.ctx.budget.remaining()
            if int(remaining.get("system2_calls", 0)) <= 0:
                return
            if float(remaining.get("wall_s", 0.0)) < 70.0:
                return
        except Exception:
            pass
        self.astra_reviews += 1
        self.astra_last_piece = _json_int(obs.get("pieces", 0))
        active = obs.get("active") or {}
        events = []
        for event in (obs.get("events") or [])[-3:]:
            if isinstance(event, dict):
                events.append(str(event.get("type", "event")))
        payload = {
            "reason": reason,
            "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
            "pieces": _json_int(obs.get("pieces", 0)),
            "board": {
                "height": _height_bucket(obs.get("max_height", obs.get("top_out_height", 0))),
                "holes": _holes_bucket(obs.get("holes", 0)),
                "surface": _surface_bucket(obs.get("bumpiness", 0)),
                "lines": _json_int(obs.get("lines", 0)),
                "level": _json_int(obs.get("level", 1)),
                "combo": "active" if _json_int(obs.get("combo", -1)) >= 0 else "inactive",
                "back_to_back": bool(obs.get("back_to_back", False)),
            },
            "active": active.get("type"),
            "hold": obs.get("hold"),
            "queue": [str(value) for value in (obs.get("queue") or [])[:5]],
            "recent_events": events,
            "current_policy": self.style,
        }
        prompt = (
            "You are Astra, a slow strategic reviewer for a real-time, "
            "Guideline-compatible Tetris marathon. Code enumerates exact legal "
            "placements, a fast controller chooses among semantic roles, and a "
            "verified executor owns all keypresses. Review only the public derived "
            "facts below. Return exactly one leading token: BALANCED for ordinary "
            "low, smooth, hole-free play; RECOVER when exposing holes or reducing "
            "height must dominate line ambition; WELL only when a controlled side "
            "well is safe. Do not return coordinates, a sequence, arithmetic, or a "
            "seed-specific script. The token changes future code evaluator weights "
            "only.\n" + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
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

    def _maybe_start_astra(self, obs: dict[str, Any]) -> None:
        pieces = _json_int(obs.get("pieces", 0))
        holes = _json_int(obs.get("holes", 0))
        height = _json_int(obs.get("max_height", obs.get("top_out_height", 0)))
        danger = holes >= 2 or height >= 12
        if self.astra_reviews == 0 and pieces >= 6:
            self._start_astra(obs, "opening_review")
        elif (
            self.astra_reviews == 1
            and self.astra_future is None
            and pieces >= 20
            and danger
            and pieces > self.astra_last_piece + 8
        ):
            self._start_astra(obs, "danger_transition")

    def _emergency_plan(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        """A zero-search plan used only while the real shortlist is pending."""
        active = obs.get("active") or {}
        piece = str(active.get("type", "O"))
        if piece not in PIECES:
            piece = "O"
        orientation = _json_int(active.get("orientation", 0)) % 4
        x = _json_int(active.get("x", 3), 3)
        grid = _grid(obs)
        before = _features(grid)
        result = _drop(grid, piece, orientation, x)
        if result is None:
            return {
                "piece": piece,
                "orientation": orientation,
                "x": x,
                "source": "current",
                "after": grid,
                "features": before,
                "lines": 0,
                "rank": -10**9,
            }
        after, features = result
        return {
            "piece": piece,
            "orientation": orientation,
            "x": x,
            "source": "current",
            "after": after,
            "features": features,
            "lines": int(features.get("lines", 0)),
            "rank": -10**6,
        }

    def _launch_search(self, obs: dict[str, Any]) -> bool:
        """Start code-owned placement search without making the game wait."""
        if self.pending_search is not None:
            if not self.pending_search.done():
                return False
            self._harvest_search(None)
        piece = _piece_id(obs)
        snapshot = copy.deepcopy(obs)
        style = self.style

        def search() -> list[dict[str, Any]]:
            try:
                return _plans(snapshot, style)
            except Exception:
                # Search is advisory; the emergency placement remains executable.
                return []

        try:
            self.pending_search = self.ctx.spawn(search)
        except Exception:
            self.pending_search = None
            self.pending_search_piece = None
            return False
        self.pending_search_piece = piece
        return True

    def _harvest_search(self, current_piece: tuple[int, str] | None) -> list[dict[str, Any]] | None:
        future = self.pending_search
        if future is None or not future.done():
            return None
        piece = self.pending_search_piece
        self.pending_search = None
        self.pending_search_piece = None
        try:
            result = future.result()
        except Exception:
            result = []
        if piece is None or current_piece is None or piece != current_piece:
            return None
        if not isinstance(result, list):
            return []
        return [item for item in result if isinstance(item, dict)]

    def _launch_jev(self, obs: dict[str, Any], plans: list[dict[str, Any]]) -> bool:
        if getattr(self.ctx, "system1", None) is None or not plans:
            return True
        if self.pending_jev is not None and not self.pending_jev.done():
            return False
        if self.pending_jev is not None and self.pending_jev.done():
            self._discard_jev_result()
        roles = _role_plans(obs, plans)
        if not roles:
            return True
        state = _jev_state(obs, self.style, plans, roles)
        options = [
            (role, _role_description(role, plan))
            for role, plan in roles.items()
        ]
        piece = _piece_id(obs)
        self.pending_jev_piece = piece
        self.pending_jev_state = state
        self.pending_jev_options = options
        self.pending_jev_roles = dict(roles)

        def call() -> Any:
            try:
                return self.ctx.decide(
                    state, _choice_questions(state, options), timeout_s=0.60
                )
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
        self.pending_jev_roles = {}
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
        roles = self.pending_jev_roles
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
        self.pending_jev_roles = {}
        if piece != _piece_id(obs):
            return current_plan
        chosen = _answer_role(result, roles)
        if chosen is None:
            return current_plan
        role, plan = chosen
        if not _compatible_plan(obs, plan, hold_done):
            return current_plan
        self._log("jev_role", piece=piece[0], role=role)
        return plan

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
        guardian: dict[str, Any] | None = None
        selected: dict[str, Any] | None = None
        hold_done = False
        request_to_launch: tuple[dict[str, Any], list[dict[str, Any]]] | None = None
        previous_command: str | None = None
        command_piece: tuple[int, str] | None = None
        command_x: int | None = None
        command_orientation: int | None = None
        blocked_command = False

        def first_compatible(
            candidates: list[dict[str, Any]],
            obs: dict[str, Any],
            used_hold: bool,
        ) -> dict[str, Any] | None:
            for candidate in candidates:
                if _compatible_plan(obs, candidate, used_hold):
                    return candidate
            return None

        def urgent_drop(obs: dict[str, Any]) -> bool:
            try:
                remaining = obs.get("lock_remaining_s")
                if remaining is not None and float(remaining) <= 0.10:
                    return True
            except (TypeError, ValueError):
                pass
            active = obs.get("active") or {}
            cells = active.get("cells")
            ghost = obs.get("ghost_cells")
            if isinstance(cells, list) and isinstance(ghost, list) and cells and ghost:
                try:
                    return {tuple(cell) for cell in cells} == {tuple(cell) for cell in ghost}
                except (TypeError, ValueError):
                    return False
            return False

        try:
            while not bool(getattr(ctx, "done", False)) and not bool(
                getattr(game, "done", False)
            ):
                self._harvest_astra()
                try:
                    obs = game.observe()
                except Exception:
                    return
                if not isinstance(obs, dict) or bool(obs.get("game_over", False)):
                    return
                self._maybe_start_astra(obs)

                number = _json_int(obs.get("pieces", 0))
                piece_now = _piece_id(obs)

                # A worker result is a zero-wait event.  A result for the exact
                # falling piece can replace the guardian; a late result is
                # consumed and discarded rather than applied to a new piece.
                ready = self._harvest_search(piece_now)
                if (
                    ready is not None
                    and piece_number == number
                    and current_piece == piece_now
                ):
                    plans = ready
                    refined = first_compatible(plans, obs, hold_done)
                    if refined is not None:
                        selected = refined
                    request_to_launch = (obs, list(plans)) if plans else None
                    blocked_command = False
                    self._log("search_ready", piece=piece_now[0], candidates=len(plans))

                if number != piece_number:
                    piece_number = number
                    current_piece = piece_now
                    plans = []
                    hold_done = False
                    blocked_command = False
                    request_to_launch = None
                    guardian = self._emergency_plan(obs)
                    selected = guardian

                    # Cross the piece boundary with a harmless fresh action.
                    # The immediate guardian is then installed before the next
                    # policy tick, so a slow refinement never leaves a blind
                    # no-op policy in charge.
                    try:
                        ctx.act(_safe_action(spec, "noop"))
                    except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                        return
                    except Exception:
                        return
                    try:
                        coarse = _coarse_plans(obs, self.style)
                    except Exception:
                        coarse = []
                    try:
                        latest = game.observe()
                    except Exception:
                        return
                    if not isinstance(latest, dict) or bool(latest.get("game_over", False)):
                        return
                    if _piece_id(latest) != piece_now:
                        # Gravity locked during the cheap calculation.  Do not
                        # issue an action based on the old active piece.
                        continue
                    if coarse:
                        guardian = coarse[0]
                        selected = guardian
                    else:
                        guardian = self._emergency_plan(latest) or guardian
                        selected = guardian
                    obs = latest
                    self._launch_search(obs)
                elif current_piece != piece_now:
                    # Hold changes active.type without increasing pieces.  A
                    # selected hold plan remains valid; otherwise build an
                    # immediate guardian for the replacement piece.
                    current_piece = piece_now
                    blocked_command = False
                    if selected is None or not _compatible_plan(obs, selected, hold_done):
                        try:
                            coarse = _coarse_plans(obs, self.style)
                        except Exception:
                            coarse = []
                        guardian = coarse[0] if coarse else self._emergency_plan(obs)
                        selected = guardian
                elif (
                    command_piece == piece_now
                    and previous_command in {"left", "right", "rotate_cw", "rotate_ccw"}
                ):
                    active_now = obs.get("active") or {}
                    now_x = _json_int(active_now.get("x", 3), 3)
                    now_orientation = _json_int(active_now.get("orientation", 0)) % 4
                    blocked_command = (
                        now_x == command_x and now_orientation == command_orientation
                    )
                else:
                    blocked_command = False

                # If a prior worker ended after a lock, give the current piece a
                # refinement search as soon as the slot is free.  The guardian
                # remains usable while it runs.
                if (
                    piece_number == number
                    and self.pending_search is None
                    and not plans
                ):
                    self._launch_search(obs)

                selected = self._harvest_jev(obs, selected, hold_done)
                if selected is None or not _compatible_plan(obs, selected, hold_done):
                    replacement = first_compatible(plans, obs, hold_done)
                    if replacement is not None:
                        selected = replacement
                    elif guardian is not None and _compatible_plan(obs, guardian, hold_done):
                        selected = guardian
                    else:
                        selected = self._emergency_plan(obs)

                if blocked_command or selected is None:
                    action = _safe_action(spec, "hard_drop")
                elif (
                    urgent_drop(obs)
                    and not (selected.get("source") == "hold" and not hold_done)
                ):
                    action = _safe_action(spec, "hard_drop")
                else:
                    action = _next_action(obs, selected, hold_done, spec)

                active_before = obs.get("active") or {}
                command_piece = piece_now
                command_x = _json_int(active_before.get("x", 3), 3)
                command_orientation = _json_int(active_before.get("orientation", 0)) % 4

                if not self._post_and_wait(game, action):
                    return
                previous_command = action

                if action == "hold" and "hold" in (getattr(spec, "names", ()) or ()):
                    hold_done = True

                # Launch the fast judgment only after a real action has been
                # posted.  It can never become the clock driver.
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
        finally:
            # Search is finite ordinary Python work.  Give it a short bounded
            # chance to finish so no worker survives return; never join model
            # futures here because the evaluator cancels them at episode end.
            future = self.pending_search
            if future is not None and not future.done():
                try:
                    future.result(timeout=0.5)
                except Exception:
                    try:
                        future.cancel()
                    except Exception:
                        pass
            self.pending_search = None
            self.pending_search_piece = None


def build(ctx: Any) -> Harness:
    return Harness(ctx)
