"""Tactical-chain controller for the real-time Guideline Tetris game.

This controller separates three jobs.  The synchronous foreground computes a
small, safe placement immediately.  A bounded worker searches queue
continuations and labels *difficult-clear* opportunities (T-spins and
Tetrises).  Jev chooses a semantic track, rather than a coordinate, and Astra
chooses a long-lived phase.  The executor is always code-owned and verifies
SRS movement against the current observation.
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
ORIENTATIONS = ("0", "R", "2", "L")

# SRS spawn cells in the adaptor's bottom-origin coordinate system.
_BASE: dict[str, tuple[tuple[int, int], ...]] = {
    "J": ((0, 2), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 2), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 2), (2, 2), (0, 1), (1, 1)),
    "Z": ((0, 2), (1, 2), (1, 1), (2, 1)),
    "T": ((1, 2), (0, 1), (1, 1), (2, 1)),
    "I": ((0, 2), (1, 2), (2, 2), (3, 2)),
    "O": ((1, 2), (2, 2), (1, 1), (2, 1)),
}

# (old orientation, new orientation) -> SRS kick tests.
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
    shapes: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {}
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
                rotated.append((int(round(pivot[0] + dy)), int(round(pivot[1] - dx))))
            current = tuple(rotated)
        shapes[piece] = tuple(states)
    return shapes


SHAPES = _make_shapes()


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _board(obs: dict[str, Any]) -> list[list[int]]:
    """Return the public top-to-bottom board in bottom-origin form."""
    grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    raw = obs.get("board")
    if not isinstance(raw, list):
        return grid
    for top, row in enumerate(raw[:HEIGHT]):
        if not isinstance(row, list):
            continue
        y = HEIGHT - 1 - top
        for x, value in enumerate(row[:WIDTH]):
            try:
                grid[y][x] = int(value) if value else 0
            except (TypeError, ValueError):
                grid[y][x] = 0
    return grid


def _heights(grid: list[list[int]]) -> list[int]:
    result: list[int] = []
    for x in range(WIDTH):
        h = 0
        for y in range(HEIGHT - 1, -1, -1):
            if grid[y][x]:
                h = y + 1
                break
        result.append(h)
    return result


def _features(grid: list[list[int]]) -> dict[str, Any]:
    heights = _heights(grid)
    holes = 0
    hole_depth = 0
    for x, h in enumerate(heights):
        above = 0
        for y in range(h - 1, -1, -1):
            if grid[y][x]:
                above += 1
            elif above:
                holes += 1
                hole_depth += above
    bumpiness = sum(abs(heights[x] - heights[x + 1]) for x in range(WIDTH - 1))
    wells = 0
    for x, h in enumerate(heights):
        left = heights[x - 1] if x else HEIGHT
        right = heights[x + 1] if x + 1 < WIDTH else HEIGHT
        wells += max(0, min(left, right) - h)
    transitions = 0
    for y in range(HEIGHT):
        last = 1
        for x in range(WIDTH):
            now = 1 if grid[y][x] else 0
            transitions += int(now != last)
            last = now
        transitions += int(last == 0)
    return {
        "heights": heights,
        "aggregate_height": sum(heights),
        "max_height": max(heights) if heights else 0,
        "holes": holes,
        "hole_depth": hole_depth,
        "bumpiness": bumpiness,
        "wells": wells,
        "row_transitions": transitions,
    }


def _drop(
    grid: list[list[int]], piece: str, orientation: int, anchor_x: int
) -> tuple[list[list[int]], dict[str, Any]] | None:
    shape = SHAPES.get(piece, SHAPES["O"])[orientation % 4]

    def legal(anchor_y: int) -> bool:
        for dx, dy in shape:
            x, y = anchor_x + dx, anchor_y + dy
            if x < 0 or x >= WIDTH or y < 0:
                return False
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
    survivors = [row for row in placed if sum(bool(v) for v in row) < WIDTH]
    lines = HEIGHT - len(survivors)
    survivors.extend([[0] * WIDTH for _ in range(lines)])
    after = _features(survivors)
    after.update(
        {
            "lines": lines,
            "cells": tuple(cells),
            "anchor_y": y,
            "landing_height": max(y0 for _, y0 in cells),
        }
    )
    return survivors, after


def _occupied(grid: list[list[int]], x: int, y: int) -> bool:
    return x < 0 or x >= WIDTH or y < 0 or (y < HEIGHT and bool(grid[y][x]))


def _spin_corners(
    grid: list[list[int]], piece: str, orientation: int, x: int, cells: Any
) -> int:
    """Count occupied T diagonal corners before lock.

    A three-corner T is the full-spin opportunity used by the tactical
    planner.  The engine remains authoritative; the executor additionally
    requires the final approach rotation for such a plan.
    """
    if piece != "T" or not isinstance(cells, tuple) or len(cells) != 4:
        return 0
    # SRS's T pivot is anchor+(1,1) in all orientations.
    center_x, center_y = x + 1, _int(cells[0][1], 0)
    # Derive the pivot from the cells rather than assuming the landing anchor
    # survived a future change in shape ordering.
    xs = [int(c[0]) for c in cells]
    ys = [int(c[1]) for c in cells]
    # For a T, the unique cell with three orthogonal neighbours is the pivot.
    pivot = None
    cell_set = {(int(a), int(b)) for a, b in cells}
    for cx, cy in cell_set:
        if sum((cx + dx, cy + dy) in cell_set for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))) == 3:
            pivot = (cx, cy)
            break
    if pivot is not None:
        center_x, center_y = pivot
    return sum(
        _occupied(grid, center_x + dx, center_y + dy)
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1))
    )


def _source_options(obs: dict[str, Any]) -> list[tuple[str, str, list[str], int]]:
    active = obs.get("active") or {}
    current = str(active.get("type", "O"))
    if current not in PIECES:
        current = "O"
    start_o = _int(active.get("orientation", 0)) % 4
    queue = [str(v) for v in (obs.get("queue") or []) if str(v) in PIECES]
    result = [("current", current, queue, start_o)]
    if bool(obs.get("can_hold", False)):
        held = obs.get("hold")
        if held in PIECES:
            result.append(("hold", str(held), queue, 0))
        elif queue:
            result.append(("hold", queue[0], queue[1:], 0))
    return result


def _event_base(plan: dict[str, Any]) -> float:
    lines = _int(plan.get("lines", 0))
    spin = _int(plan.get("spin_corners", 0))
    # A corner count alone is not a T-spin: the final route must contain a
    # rotation, matching the engine's last-maneuver rule.
    qualified = bool(plan.get("spin_ready", False))
    if qualified and spin >= 3:
        return float({0: 400, 1: 800, 2: 1200, 3: 1600}.get(lines, 0))
    if qualified and spin == 2:
        return float({0: 100, 1: 200, 2: 400}.get(lines, 0))
    return float({0: 0, 1: 100, 2: 300, 3: 500, 4: 800}.get(lines, 0))


def _difficult(plan: dict[str, Any]) -> bool:
    return _int(plan.get("lines", 0)) == 4 or (
        bool(plan.get("spin_ready", False))
        and _int(plan.get("spin_corners", 0)) >= 3
        and _int(plan.get("lines", 0)) > 0
    )


def _plan_value(
    after: dict[str, Any], before: dict[str, Any], plan: dict[str, Any], obs: dict[str, Any], mode: str
) -> float:
    """Rank one exact placement without asking a model to do arithmetic."""
    level = max(1, _int(obs.get("level", 1)))
    lines = _int(plan.get("lines", 0))
    spin = _int(plan.get("spin_corners", 0))
    base = _event_base(plan)
    # A line-clear spin/Tetris is the high-value event.  A no-line spin is
    # useful only as a setup and is deliberately discounted.
    event = base * level
    if lines == 0 and spin >= 3:
        event *= 0.16
    elif lines == 0 and spin == 2:
        event *= 0.08
    if _difficult(plan) and bool(obs.get("back_to_back", False)):
        event *= 1.50
    combo = _int(obs.get("combo", -1))
    if lines > 0 and combo >= 0:
        event += 50.0 * max(1, combo + 1) * level

    old_holes = _int(before.get("holes", 0))
    new_holes = _int(after.get("holes", 0))
    delta_holes = new_holes - old_holes
    old_max = _int(before.get("max_height", 0))
    new_max = _int(after.get("max_height", 0))
    delta_max = new_max - old_max
    delta_aggregate = _int(after.get("aggregate_height", 0)) - _int(before.get("aggregate_height", 0))

    # The tactical event is one term in a safety envelope, not permission to
    # make a new buried hole.
    value = (
        0.52 * event
        + 40.0 * lines
        + 16.0 * lines * lines
        - 18.0 * _int(after.get("holes", 0))
        - 2.7 * _int(after.get("hole_depth", 0))
        - 0.82 * _int(after.get("aggregate_height", 0))
        - 2.7 * new_max
        - 0.54 * _int(after.get("bumpiness", 0))
        - 0.07 * _int(after.get("row_transitions", 0))
        - 0.08 * _int(after.get("wells", 0))
        - 0.10 * _int(after.get("landing_height", 0))
        - 35.0 * max(0, delta_holes)
        - 4.0 * max(0, delta_max)
        - 0.18 * max(0, delta_aggregate)
    )
    if _difficult(plan):
        value += 55.0
    if bool(plan.get("spin_ready", False)) and spin >= 3 and lines == 0:
        value += 20.0
    if mode == "RECOVER":
        value -= 14.0 * new_holes + 1.8 * _int(after.get("hole_depth", 0))
        value -= 2.0 * new_max
    elif mode == "ATTACK":
        value += 0.20 * event
        if (bool(plan.get("spin_ready", False)) and spin >= 3) or lines == 4:
            value += 75.0
    if plan.get("source") == "hold":
        value -= 4.0
    return value


def _one_piece_plans(
    grid: list[list[int]],
    piece: str,
    before: dict[str, Any],
    obs: dict[str, Any],
    mode: str,
    start_orientation: int,
    source: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    orientations = (0,) if piece == "O" else range(4)
    for orientation in orientations:
        for x in range(-4, WIDTH + 4):
            landed = _drop(grid, piece, orientation, x)
            if landed is None:
                continue
            after, facts = landed
            corners = _spin_corners(grid, piece, orientation, x, facts.get("cells"))
            # A spin needs a rotation as the last maneuver.  Excluding a
            # same-orientation target prevents a no-op placement from being
            # falsely called a spin; the live router enforces the final turn.
            spin_ready = piece == "T" and corners >= 2 and orientation != start_orientation
            plan = {
                "piece": piece,
                "orientation": int(orientation),
                "x": int(x),
                "source": source,
                "after": after,
                "features": facts,
                "lines": _int(facts.get("lines", 0)),
                "spin_corners": int(corners),
                "spin_ready": bool(spin_ready),
            }
            plan["difficult"] = bool(
                plan["lines"] == 4 or (corners >= 3 and plan["lines"] > 0 and spin_ready)
            )
            plan["value"] = _plan_value(facts, before, plan, obs, mode)
            result.append(plan)
    return result


def _future_value(
    grid: list[list[int]], queue: list[str], obs: dict[str, Any], mode: str,
    depth: int, width: int = 8,
) -> float:
    if depth <= 0 or not queue or queue[0] not in PIECES:
        return 0.0
    before = _features(grid)
    plans = _one_piece_plans(grid, queue[0], before, obs, mode, 0, "current")
    if not plans:
        return -500.0
    plans.sort(key=lambda p: float(p.get("value", -10**9)), reverse=True)
    best = -10**9
    limit = max(2, int(width))
    for plan in plans[:limit]:
        value = float(plan.get("value", -10**9))
        if depth > 1:
            value += 0.17 * _future_value(
                plan["after"], queue[1:], obs, mode, depth - 1,
                width=max(2, limit // 2),
            )
        best = max(best, value)
    return best


def _all_plans(obs: dict[str, Any], mode: str, depth: int) -> list[dict[str, Any]]:
    grid = _board(obs)
    before = _features(grid)
    all_plans: list[dict[str, Any]] = []
    for source, piece, future_queue, start_o in _source_options(obs):
        for plan in _one_piece_plans(grid, piece, before, obs, mode, start_o, source):
            item = dict(plan)
            item["future_queue"] = tuple(future_queue)
            item["future_value"] = 0.0
            item["rank"] = float(item["value"])
            all_plans.append(item)

    def sort_key(p: dict[str, Any]) -> tuple[Any, ...]:
        return (
            float(p.get("rank", -10**9)),
            _int(p.get("lines", 0)),
            _int(p.get("spin_corners", 0)),
            -_int((p.get("features") or {}).get("holes", 99)),
            -_int((p.get("features") or {}).get("max_height", 99)),
            1 if p.get("source") == "current" else 0,
            -_int(p.get("x", 0)),
        )

    all_plans.sort(key=sort_key, reverse=True)
    if depth and all_plans:
        # Lookahead is a refinement of a small ranked focus set, never a full
        # cross-product.  This keeps the worker bounded under the realtime GIL.
        focus: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()

        def add(item: dict[str, Any]) -> None:
            key = (
                str(item.get("source")), _int(item.get("orientation")),
                _int(item.get("x")),
            )
            if key not in seen and len(focus) < 10:
                seen.add(key)
                focus.append(item)

        for item in all_plans[:8]:
            add(item)
        for source in ("current", "hold"):
            for item in all_plans:
                if item.get("source") == source:
                    add(item)
                    break
        for orientation in range(4):
            for item in all_plans:
                if _int(item.get("orientation")) == orientation:
                    add(item)
                    break
        for item in focus:
            item["future_value"] = _future_value(
                item["after"], list(item.get("future_queue") or ()),
                obs, mode, depth, width=8,
            )
            item["rank"] = float(item["value"]) + 0.22 * float(item["future_value"])
        all_plans.sort(key=sort_key, reverse=True)
    return all_plans


def _safe_attack(plan: dict[str, Any], before: dict[str, Any]) -> bool:
    facts = plan.get("features") or {}
    old_holes = _int(before.get("holes", 0))
    old_max = _int(before.get("max_height", 0))
    return (
        _int(facts.get("holes", 99)) <= old_holes
        and _int(facts.get("max_height", 99)) <= max(18, old_max + 5)
        and _int(facts.get("hole_depth", 99)) <= _int(before.get("hole_depth", 0)) + 4
    )


def _tracks(obs: dict[str, Any], plans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Turn exact placements into four semantic commitments for Jev."""
    if not plans:
        return {}
    before = _features(_board(obs))
    smooth = max(
        plans,
        key=lambda p: (
            float(p.get("rank", -10**9))
            - (90.0 if _int(p.get("spin_corners", 0)) >= 3 and _int(p.get("lines", 0)) == 0 else 0.0),
            -_int((p.get("features") or {}).get("holes", 99)),
            -_int((p.get("features") or {}).get("max_height", 99)),
        ),
    )
    clear_pool = [
        p for p in plans
        if _int(p.get("lines", 0)) > 0 and _safe_attack(p, before)
    ]
    if not clear_pool:
        clear_pool = [p for p in plans if _int(p.get("lines", 0)) > 0]
    clear = max(
        clear_pool or plans,
        key=lambda p: (
            _event_base(p) * max(1, _int(obs.get("level", 1))),
            float(p.get("rank", -10**9)),
            -_int((p.get("features") or {}).get("holes", 99)),
        ),
    )
    attack_pool = [
        p for p in plans
        if _safe_attack(p, before)
        and (
            (_int(p.get("spin_corners", 0)) >= 3 and bool(p.get("spin_ready", False)))
            or _int(p.get("lines", 0)) == 4
        )
    ]
    attack = max(
        attack_pool or [smooth],
        key=lambda p: (
            1 if _difficult(p) else 0,
            _int(p.get("spin_corners", 0)),
            _event_base(p) * max(1, _int(obs.get("level", 1))),
            float(p.get("rank", -10**9)),
        ),
    )
    recover = min(
        plans,
        key=lambda p: (
            _int((p.get("features") or {}).get("holes", 99)),
            _int((p.get("features") or {}).get("hole_depth", 99)),
            _int((p.get("features") or {}).get("max_height", 99)),
            _int((p.get("features") or {}).get("aggregate_height", 999)),
            -_int(p.get("lines", 0)),
            -float(p.get("rank", -10**9)),
        ),
    )
    return {"attack": attack, "clear": clear, "smooth": smooth, "recover": recover}


def _bucket_height(value: Any) -> str:
    n = _int(value, 20)
    if n <= 5:
        return "low"
    if n <= 10:
        return "moderate"
    if n <= 16:
        return "high"
    return "critical"


def _bucket_holes(value: Any) -> str:
    n = _int(value, 99)
    if n <= 0:
        return "none"
    if n <= 3:
        return "few"
    if n <= 8:
        return "several"
    return "many"


def _bucket_surface(value: Any) -> str:
    n = _int(value, 99)
    if n <= 8:
        return "flat"
    if n <= 18:
        return "uneven"
    return "rough"


def _kind(plan: dict[str, Any]) -> str:
    lines = _int(plan.get("lines", 0))
    corners = _int(plan.get("spin_corners", 0))
    if corners >= 3 and bool(plan.get("spin_ready", False)):
        return "full T-spin" + (" line clear" if lines else " setup")
    if lines == 4:
        return "Tetris"
    if corners == 2 and bool(plan.get("spin_ready", False)):
        return "mini-spin opportunity"
    if lines:
        return {1: "single", 2: "double", 3: "triple"}.get(lines, "line clear")
    return "quiet placement"


def _description(label: str, plan: dict[str, Any], obs: dict[str, Any]) -> str:
    facts = plan.get("features") or {}
    purpose = {
        "attack": "pursue a verified difficult clear or a safe full-spin setup",
        "clear": "take the best safe ordinary line-clear opportunity",
        "smooth": "keep the surface low and flexible without forcing offense",
        "recover": "expose holes and reduce top-out risk before scoring",
    }.get(label, "place safely")
    source = "using hold" if plan.get("source") == "hold" else "using the current piece"
    chain = "continues the difficult-clear chain" if bool(obs.get("back_to_back", False)) and _difficult(plan) else "does not promise a chain"
    return (
        f"{purpose}; {source}; {_kind(plan)}; projected {_bucket_height(facts.get('max_height'))} height, "
        f"{_bucket_holes(facts.get('holes'))} holes, {_bucket_surface(facts.get('bumpiness'))} surface; {chain}."
    )


def _jev_state(obs: dict[str, Any], mode: str, tracks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    active = obs.get("active") or {}
    return {
        "ruleset": str(obs.get("ruleset", "guideline-marathon-v1")),
        "phase": mode,
        "piece": str(active.get("type", "unknown")),
        "hold": obs.get("hold") if obs.get("hold") in PIECES else None,
        "hold_available": bool(obs.get("can_hold", False)),
        "queue": [str(v) for v in (obs.get("queue") or [])[:5]],
        "stack": {
            "height": _bucket_height(obs.get("max_height", obs.get("top_out_height", 0))),
            "holes": _bucket_holes(obs.get("holes", 0)),
            "surface": _bucket_surface(obs.get("bumpiness", 0)),
        },
        "chain": "active" if bool(obs.get("back_to_back", False)) else "inactive",
        "combo": "active" if _int(obs.get("combo", -1)) >= 0 else "inactive",
        "alternatives": {key: _description(key, plan, obs) for key, plan in tracks.items()},
    }


def _questions(state: dict[str, Any], tracks: dict[str, dict[str, Any]]) -> dict[str, Choice]:
    criteria = {key: str(state["alternatives"].get(key, key)) for key in tracks}
    reversed_criteria = {key: criteria[key] for key in reversed(list(criteria))}
    common = (
        "Choose one tactical alternative in `alternatives`. The descriptions are already "
        "computed from legal placements. Prefer `recover` for holes or critical height; "
        "prefer `attack` only when its full-spin/Tetris opportunity is safe; preserve an "
        "active difficult-clear chain when possible. Do not count, calculate score, "
        "invent coordinates, or output a primitive action. Return exactly one label. State: "
        + json.dumps(state, separators=(",", ":"), ensure_ascii=True)
    )
    return {
        "first": Choice(common + " Make the primary judgment.", criteria=criteria),
        "second": Choice(
            common + " Recheck independently with the option order reversed.",
            criteria=reversed_criteria,
        ),
    }


def _probability(answer: Any, key: str) -> float | None:
    try:
        probs = getattr(answer, "probabilities", None)
        if not isinstance(probs, dict) or key not in probs:
            return None
        return float(probs[key])
    except (TypeError, ValueError, AttributeError):
        return None


def _answer_track(result: Any, tracks: dict[str, dict[str, Any]]) -> str | None:
    if not isinstance(result, dict):
        return None
    a = result.get("first")
    b = result.get("second")
    av = getattr(a, "value", None)
    bv = getattr(b, "value", None)
    if av not in tracks or av != bv:
        return None
    pa, pb = _probability(a, str(av)), _probability(b, str(av))
    if pa is not None and pa < 0.38:
        return None
    if pb is not None and pb < 0.38:
        return None
    return str(av)


def _safe_action(spec: Any, wanted: str) -> str:
    names = [str(v) for v in (getattr(spec, "names", ()) or ())]
    if wanted in names:
        return wanted
    noop = str(getattr(spec, "noop", "noop"))
    if noop in names:
        return noop
    if "noop" in names:
        return "noop"
    return sorted(names)[0] if names else "noop"


def _piece_id(obs: dict[str, Any]) -> tuple[int, str]:
    active = obs.get("active") or {}
    return (_int(obs.get("pieces", 0)), str(active.get("type", "")))


def _compatible(obs: dict[str, Any], plan: dict[str, Any], hold_done: bool) -> bool:
    active = obs.get("active") or {}
    active_type = str(active.get("type", ""))
    if plan.get("source") == "hold":
        if not hold_done:
            return bool(obs.get("can_hold", False))
        return active_type == str(plan.get("piece", ""))
    return not hold_done and active_type == str(plan.get("piece", ""))


def _pose_legal(grid: list[list[int]], piece: str, orientation: int, x: int, y: int) -> bool:
    for dx, dy in SHAPES.get(piece, SHAPES["O"])[orientation % 4]:
        cx, cy = x + dx, y + dy
        if cx < 0 or cx >= WIDTH or cy < 0:
            return False
        if cy < HEIGHT and grid[cy][cx]:
            return False
    return True


def _rotated_pose(
    grid: list[list[int]], piece: str, orientation: int, x: int, y: int, action: str
) -> tuple[int, int, int] | None:
    if piece == "O":
        return None
    target = (orientation + (1 if action == "rotate_cw" else -1)) % 4
    table = _KICKS["I" if piece == "I" else "JLSTZ"]
    for dx, dy in table.get((orientation % 4, target), ((0, 0),)):
        nx, ny = x + dx, y + dy
        if _pose_legal(grid, piece, target, nx, ny):
            return target, nx, ny
    return None


def _route_action(
    obs: dict[str, Any], plan: dict[str, Any], spec: Any, last_action: str | None
) -> str:
    active = obs.get("active") or {}
    piece = str(active.get("type", ""))
    if piece not in PIECES or piece != str(plan.get("piece", "")):
        return _safe_action(spec, "hard_drop")
    start_o = _int(active.get("orientation", 0)) % 4
    start_x = _int(active.get("x", 3), 3)
    start_y = _int(active.get("y", 17), 17)
    target_o = _int(plan.get("orientation", start_o)) % 4
    target_x = _int(plan.get("x", start_x), start_x)
    must_rotate_last = bool(plan.get("spin_ready", False))
    if start_o == target_o and start_x == target_x:
        return _safe_action(spec, "hard_drop")

    grid = _board(obs)
    # Search the public pose graph.  For a spin target, only accept a goal whose
    # final edge is a rotation, preserving the engine's last-maneuver test.
    queue = deque([(start_o, start_x, start_y, ())])
    seen = {(start_o, start_x, start_y)}
    while queue:
        orientation, x, y, path = queue.popleft()
        if orientation == target_o and x == target_x and path:
            if not must_rotate_last or path[-1] in ("rotate_cw", "rotate_ccw"):
                return _safe_action(spec, path[0])
        if len(path) >= 10:
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
                rotated = _rotated_pose(grid, piece, orientation, x, y, action)
                if rotated is None:
                    continue
                candidate = rotated
            if candidate in seen:
                continue
            seen.add(candidate)
            next_path = path + (action,)
            if candidate[0] == target_o and candidate[1] == target_x:
                if not must_rotate_last or next_path[-1] in ("rotate_cw", "rotate_ccw"):
                    return _safe_action(spec, next_path[0])
            queue.append((candidate[0], candidate[1], candidate[2], next_path))

    # A moving piece can lose a route as gravity advances.  One legal movement
    # is safer than repeating an impossible command; the next observation will
    # replan.  If the spin route disappeared, hard drop rather than risking a
    # late rotation at the lock boundary.
    if start_x < target_x and _pose_legal(grid, piece, start_o, start_x + 1, start_y):
        return _safe_action(spec, "right")
    if start_x > target_x and _pose_legal(grid, piece, start_o, start_x - 1, start_y):
        return _safe_action(spec, "left")
    if not must_rotate_last and start_o != target_o:
        for action in ("rotate_cw", "rotate_ccw"):
            if _rotated_pose(grid, piece, start_o, start_x, start_y, action) is not None:
                return _safe_action(spec, action)
    return _safe_action(spec, "hard_drop")


def _next_action(
    obs: dict[str, Any], plan: dict[str, Any], hold_done: bool, spec: Any, last_action: str | None
) -> str:
    if plan.get("source") == "hold" and not hold_done:
        return _safe_action(spec, "hold") if bool(obs.get("can_hold", False)) else _safe_action(spec, "hard_drop")
    return _route_action(obs, plan, spec, last_action)


def _urgent(obs: dict[str, Any]) -> bool:
    try:
        remaining = obs.get("lock_remaining_s")
        if remaining is not None and float(remaining) <= 0.10:
            return True
    except (TypeError, ValueError):
        pass
    active = obs.get("active") or {}
    cells, ghost = active.get("cells"), obs.get("ghost_cells")
    if isinstance(cells, list) and isinstance(ghost, list) and cells and ghost:
        try:
            return {tuple(c) for c in cells} == {tuple(c) for c in ghost}
        except (TypeError, ValueError):
            return False
    return False


def _default_track(obs: dict[str, Any], mode: str, tracks: dict[str, dict[str, Any]]) -> str:
    if not tracks:
        return "smooth"
    if mode == "RECOVER":
        return "recover"
    attack = tracks.get("attack", tracks.get("smooth"))
    attack_is_real = bool(attack) and (
        _difficult(attack)
        or (
            bool(attack.get("spin_ready", False))
            and _int(attack.get("spin_corners", 0)) >= 3
            and _int(attack.get("lines", 0)) == 0
        )
    )
    if attack_is_real and _safe_attack(attack, _features(_board(obs))):
        return "attack"
    clear = tracks.get("clear")
    smooth = tracks.get("smooth")
    if clear and _int(clear.get("lines", 0)) > 0 and float(clear.get("rank", -10**9)) >= float(smooth.get("rank", -10**9)) - 10.0:
        return "clear"
    return "smooth"


def _mode_from_text(text: Any) -> str | None:
    match = re.match(r"^(?:PHASE\s*[:=]\s*)?(ATTACK|BALANCED|RECOVER)\b", str(text or "").strip().upper())
    return match.group(1) if match else None


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.mode = "BALANCED"
        self.astra_future: Any = None
        self.astra_calls = 0
        self.astra_last_piece = -100
        self.search_future: Any = None
        self.search_piece: tuple[int, str] | None = None
        self.jev_future: Any = None
        self.jev_piece: tuple[int, str] | None = None
        self.jev_tracks: dict[str, dict[str, Any]] = {}

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _start_astra(self, obs: dict[str, Any], reason: str) -> None:
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if _int(remaining.get("system2_calls", 0)) <= 0 or float(remaining.get("wall_s", 0.0)) < 60.0:
                return
        except Exception:
            pass
        self.astra_calls += 1
        self.astra_last_piece = _int(obs.get("pieces", 0))
        payload = {
            "reason": reason,
            "ruleset": str(obs.get("ruleset", "guideline-marathon-v1")),
            "piece_count": _int(obs.get("pieces", 0)),
            "stack_height": _bucket_height(obs.get("max_height", obs.get("top_out_height", 0))),
            "holes": _bucket_holes(obs.get("holes", 0)),
            "surface": _bucket_surface(obs.get("bumpiness", 0)),
            "lines": _int(obs.get("lines", 0)),
            "level": _int(obs.get("level", 1)),
            "combo": "active" if _int(obs.get("combo", -1)) >= 0 else "inactive",
            "back_to_back": bool(obs.get("back_to_back", False)),
            "active": (obs.get("active") or {}).get("type"),
            "hold": obs.get("hold"),
            "queue": [str(v) for v in (obs.get("queue") or [])[:5]],
            "current_phase": self.mode,
        }
        prompt = (
            "You are Astra, the slow phase planner for a real-time, Guideline-compatible "
            "Tetris marathon. Code enumerates legal placements and guarantees safety; "
            "Jev chooses among tactical tracks. Review the derived public state below and "
            "return exactly one leading token: ATTACK when a safe difficult-clear/T-spin "
            "chain should be pursued, RECOVER when holes or height dominate, or BALANCED "
            "otherwise. The token is a phase preference for future pieces, not a keypress. "
            "Do not output coordinates, arithmetic, a sequence, or prose before the token.\n"
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        )

        def call() -> str:
            try:
                result = self.ctx.think(prompt, timeout_s=55.0)
                return str(getattr(result, "text", ""))
            except ModelUnavailable:
                raise
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver):
                return ""

        try:
            self.astra_future = self.ctx.spawn(call)
        except ModelUnavailable:
            raise
        except Exception:
            self.astra_future = None

    def _harvest_astra(self) -> None:
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            mode = _mode_from_text(future.result())
        except ModelUnavailable:
            raise
        except Exception:
            mode = None
        if mode and mode != self.mode:
            self.mode = mode
            self._log("astra_phase", phase=mode)

    def _maybe_astra(self, obs: dict[str, Any]) -> None:
        piece = _int(obs.get("pieces", 0))
        height = _int(obs.get("max_height", obs.get("top_out_height", 0)))
        holes = _int(obs.get("holes", 0))
        if self.astra_calls == 0:
            self._start_astra(obs, "initial_phase")
        elif (
            self.astra_calls == 1
            and self.astra_future is None
            and piece >= 30
            and piece > self.astra_last_piece + 20
            and (holes > 0 or height >= 12)
        ):
            self._start_astra(obs, "danger_phase")

    def _emergency(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        active = obs.get("active") or {}
        piece = str(active.get("type", "O"))
        if piece not in PIECES:
            piece = "O"
        orientation = _int(active.get("orientation", 0)) % 4
        x = _int(active.get("x", 3), 3)
        grid = _board(obs)
        before = _features(grid)
        landed = _drop(grid, piece, orientation, x)
        if landed is None:
            facts = dict(before)
            lines = 0
        else:
            _, facts = landed
            lines = _int(facts.get("lines", 0))
        return {
            "piece": piece,
            "orientation": orientation,
            "x": x,
            "source": "current",
            "features": facts,
            "lines": lines,
            "spin_corners": 0,
            "spin_ready": False,
            "rank": -10**9,
            "value": -10**9,
        }

    def _launch_search(self, obs: dict[str, Any]) -> None:
        if self.search_future is not None:
            return
        snapshot = copy.deepcopy(obs)
        identity = _piece_id(obs)
        mode = self.mode

        def search() -> dict[str, dict[str, Any]]:
            try:
                plans = _all_plans(snapshot, mode, depth=2)
                return _tracks(snapshot, plans)
            except Exception:
                return {}

        try:
            self.search_future = self.ctx.spawn(search)
            self.search_piece = identity
        except Exception:
            self.search_future = None
            self.search_piece = None

    def _harvest_search(self, identity: tuple[int, str]) -> dict[str, dict[str, Any]] | None:
        future = self.search_future
        if future is None or not future.done():
            return None
        expected = self.search_piece
        self.search_future = None
        self.search_piece = None
        try:
            result = future.result()
        except Exception:
            result = {}
        if expected != identity or not isinstance(result, dict):
            return None
        return {str(k): v for k, v in result.items() if isinstance(k, str) and isinstance(v, dict)}

    def _launch_jev(self, obs: dict[str, Any], tracks: dict[str, dict[str, Any]]) -> None:
        if getattr(self.ctx, "system1", None) is None or not tracks:
            return
        if self.jev_future is not None:
            return
        try:
            if _int(self.ctx.budget.remaining().get("system1_calls", 0)) <= 0:
                return
        except Exception:
            pass
        snapshot = copy.deepcopy(obs)
        state = _jev_state(snapshot, self.mode, tracks)
        questions = _questions(state, tracks)
        identity = _piece_id(snapshot)

        def call() -> Any:
            try:
                return self.ctx.decide(state, questions, timeout_s=0.80)
            except ModelUnavailable:
                raise
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver):
                return None

        try:
            self.jev_future = self.ctx.spawn(call)
            self.jev_piece = identity
            self.jev_tracks = tracks
        except ModelUnavailable:
            raise
        except Exception:
            self.jev_future = None
            self.jev_piece = None
            self.jev_tracks = {}

    def _harvest_jev(
        self, obs: dict[str, Any], hold_done: bool
    ) -> tuple[str, dict[str, Any]] | None:
        future = self.jev_future
        if future is None or not future.done():
            return None
        expected = self.jev_piece
        tracks = self.jev_tracks
        self.jev_future = None
        self.jev_piece = None
        self.jev_tracks = {}
        try:
            result = future.result()
        except ModelUnavailable:
            raise
        except Exception:
            result = None
        if expected != _piece_id(obs):
            return None
        chosen = _answer_track(result, tracks)
        if chosen is None:
            return None
        plan = tracks.get(chosen)
        if not isinstance(plan, dict) or not _compatible(obs, plan, hold_done):
            return None
        self._log("jev_track", piece=expected[0], track=chosen)
        return chosen, plan

    def _post_wait(self, game: Any, action: str) -> bool:
        if bool(getattr(self.ctx, "done", False)) or bool(getattr(game, "done", False)):
            return False
        try:
            self.ctx.act(action)
            self.ctx.wait_for_tick()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, EpisodeOver):
            return False
        except Exception:
            return False
        return not bool(getattr(self.ctx, "done", False)) and not bool(getattr(game, "done", False))

    def run_episode(self, game: Any) -> None:
        try:
            initial = game.observe()
        except Exception:
            return
        if not isinstance(initial, dict):
            return
        spec = getattr(game, "action_spec", None)
        self._start_astra(initial, "initial_phase")

        piece_number: int | None = None
        identity: tuple[int, str] | None = None
        tracks: dict[str, dict[str, Any]] = {}
        selected: dict[str, Any] | None = None
        selected_track: str | None = None
        hold_done = False
        last_action: str | None = None
        command_identity: tuple[int, str] | None = None
        command_x: int | None = None
        command_o: int | None = None
        blocked = False

        def compatible_fallback(observation: dict[str, Any]) -> None:
            nonlocal selected, selected_track
            if selected is not None and _compatible(observation, selected, hold_done):
                return
            for key in (selected_track, _default_track(observation, self.mode, tracks), "smooth", "recover"):
                if key and key in tracks and _compatible(observation, tracks[key], hold_done):
                    selected_track, selected = key, tracks[key]
                    return
            selected_track, selected = None, self._emergency(observation)

        try:
            while not bool(getattr(self.ctx, "done", False)) and not bool(getattr(game, "done", False)):
                self._harvest_astra()
                try:
                    obs = game.observe()
                except Exception:
                    return
                if not isinstance(obs, dict) or bool(obs.get("game_over", False)):
                    return
                self._maybe_astra(obs)
                current_number = _int(obs.get("pieces", 0))
                current_identity = _piece_id(obs)

                ready = self._harvest_search(current_identity)
                if ready is not None and piece_number == current_number and identity == current_identity:
                    tracks = ready
                    if selected_track in tracks and _compatible(obs, tracks[selected_track], hold_done):
                        selected = tracks[selected_track]
                    else:
                        selected_track = _default_track(obs, self.mode, tracks)
                        selected = tracks.get(selected_track)
                    self._log("tactical_search_ready", piece=current_number, tracks=list(tracks))

                new_piece = current_number != piece_number
                if new_piece:
                    piece_number = current_number
                    identity = current_identity
                    hold_done = False
                    blocked = False
                    selected_track = None
                    # Give the realtime clock a fresh, harmless command before
                    # foreground enumeration.  This preserves the parent's
                    # boundary guarantee without waiting for a model.  Re-read
                    # afterward so a fast gravity step cannot invalidate the
                    # placement snapshot.
                    try:
                        self.ctx.act(_safe_action(spec, "noop"))
                    except ModelUnavailable:
                        raise
                    except (BudgetExceeded, EpisodeOver):
                        return
                    except Exception:
                        return
                    try:
                        latest = game.observe()
                    except Exception:
                        return
                    if not isinstance(latest, dict) or bool(latest.get("game_over", False)):
                        return
                    if _piece_id(latest) != current_identity:
                        continue
                    obs = latest
                    current_identity = _piece_id(obs)
                    identity = current_identity
                    try:
                        fast = _all_plans(obs, self.mode, depth=0)
                        tracks = _tracks(obs, fast)
                    except Exception:
                        tracks = {}
                    selected_track = _default_track(obs, self.mode, tracks)
                    selected = tracks.get(selected_track) or self._emergency(obs)
                elif identity != current_identity:
                    # Hold swaps the active type without incrementing pieces.
                    identity = current_identity
                    blocked = False
                    compatible_fallback(obs)
                elif command_identity == current_identity and last_action in {
                    "left", "right", "rotate_cw", "rotate_ccw"
                }:
                    active = obs.get("active") or {}
                    now_x = _int(active.get("x", 3), 3)
                    now_o = _int(active.get("orientation", 0)) % 4
                    blocked = now_x == command_x and now_o == command_o
                else:
                    blocked = False

                decision = self._harvest_jev(obs, hold_done)
                if decision is not None:
                    selected_track, selected = decision
                compatible_fallback(obs)

                if blocked or selected is None:
                    action = _safe_action(spec, "hard_drop")
                else:
                    planned_action = _next_action(obs, selected, hold_done, spec, last_action)
                    # A grounded piece still has legal lock-reset moves.  Let a
                    # verified translation/rotation complete the target (most
                    # importantly the final T-spin rotation); if the route says
                    # the target is already reached, it returns hard_drop and
                    # the urgency guard retains its immediate-lock behavior.
                    if _urgent(obs) and not (
                        selected.get("source") == "hold" and not hold_done
                    ) and planned_action == _safe_action(spec, "hard_drop"):
                        action = _safe_action(spec, "hard_drop")
                    else:
                        action = planned_action

                active_before = obs.get("active") or {}
                command_identity = current_identity
                command_x = _int(active_before.get("x", 3), 3)
                command_o = _int(active_before.get("orientation", 0)) % 4
                if not self._post_wait(game, action):
                    return
                last_action = action
                if action == "hold" and "hold" in (getattr(spec, "names", ()) or ()):
                    hold_done = True

                # Workers are started only after a fresh action has been posted.
                # Neither worker can become the realtime clock driver.
                if new_piece:
                    self._launch_search(obs)
                    self._launch_jev(obs, tracks)

        finally:
            # Search is finite ordinary Python; bound its cleanup.  Model calls
            # are deliberately not joined because the evaluator cancels them.
            future = self.search_future
            if future is not None and not future.done():
                try:
                    future.result(timeout=0.5)
                except Exception:
                    try:
                        future.cancel()
                    except Exception:
                        pass
            self.search_future = None
            self.search_piece = None


def build(ctx: Any) -> Harness:
    return Harness(ctx)
