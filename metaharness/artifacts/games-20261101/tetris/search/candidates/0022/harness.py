"""Conditional I-well Tetris controller for a real-time Guideline marathon.

This module keeps the clock-owning guardian/executor pattern, but its background
planner is stateful about a one-column I well.  The game remains authoritative:
board arithmetic is only a ranking aid and model output is never an action.
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
ORI_NAMES = ("0", "R", "2", "L")

_BASE = {
    "J": ((0, 2), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 2), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 2), (2, 2), (0, 1), (1, 1)),
    "Z": ((0, 2), (1, 2), (1, 1), (2, 1)),
    "T": ((1, 2), (0, 1), (1, 1), (2, 1)),
    "I": ((0, 2), (1, 2), (2, 2), (3, 2)),
    "O": ((1, 2), (2, 2), (1, 1), (2, 1)),
}

_KICKS = {
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


def _shapes() -> dict[str, tuple[tuple[tuple[int, int], ...], ...]]:
    out: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {}
    for name, base in _BASE.items():
        states: list[tuple[tuple[int, int], ...]] = []
        cur = tuple(base)
        for _ in range(4):
            states.append(tuple(sorted(cur)))
            if name == "O":
                continue
            px, py = ((1.5, 1.5) if name == "I" else (1.0, 1.0))
            rotated = []
            for x, y in cur:
                dx, dy = x - px, y - py
                rotated.append((int(round(px + dy)), int(round(py - dx))))
            cur = tuple(rotated)
        out[name] = tuple(states)
    return out


SHAPES = _shapes()


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _board(obs: dict[str, Any]) -> list[list[int]]:
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


def _column_heights(grid: list[list[int]]) -> list[int]:
    result = []
    for x in range(WIDTH):
        h = 0
        for y in range(HEIGHT - 1, -1, -1):
            if grid[y][x]:
                h = y + 1
                break
        result.append(h)
    return result


def _metrics(grid: list[list[int]]) -> dict[str, Any]:
    heights = _column_heights(grid)
    holes = 0
    depth = 0
    for x, h in enumerate(heights):
        seen = 0
        for y in range(h - 1, -1, -1):
            if grid[y][x]:
                seen += 1
            elif seen:
                holes += 1
                depth += seen
    bump = sum(abs(heights[x] - heights[x + 1]) for x in range(WIDTH - 1))
    transitions = 0
    for y in range(HEIGHT):
        previous = 1
        for x in range(WIDTH):
            current = 1 if grid[y][x] else 0
            transitions += int(current != previous)
            previous = current
        transitions += int(previous == 0)
    wells = 0
    for x, h in enumerate(heights):
        left = heights[x - 1] if x else HEIGHT
        right = heights[x + 1] if x + 1 < WIDTH else HEIGHT
        wells += max(0, min(left, right) - h)
    return {
        "heights": heights,
        "aggregate_height": sum(heights),
        "max_height": max(heights) if heights else 0,
        "holes": holes,
        "hole_depth": depth,
        "bumpiness": bump,
        "row_transitions": transitions,
        "wells": wells,
    }


def _legal_pose(grid: list[list[int]], piece: str, ori: int, x: int, y: int) -> bool:
    for dx, dy in SHAPES.get(piece, SHAPES["O"])[ori % 4]:
        cx, cy = x + dx, y + dy
        if cx < 0 or cx >= WIDTH or cy < 0:
            return False
        if cy < HEIGHT and grid[cy][cx]:
            return False
    return True


def _land(
    grid: list[list[int]], piece: str, ori: int, x: int
) -> tuple[list[list[int]], dict[str, Any]] | None:
    shape = SHAPES.get(piece, SHAPES["O"])[ori % 4]

    def legal(y: int) -> bool:
        return all(
            0 <= x + dx < WIDTH
            and y + dy >= 0
            and (y + dy >= HEIGHT or not grid[y + dy][x + dx])
            for dx, dy in shape
        )

    y = HEIGHT + 5
    while legal(y - 1):
        y -= 1
    if not legal(y):
        return None
    cells = [(x + dx, y + dy) for dx, dy in shape]
    if any(cy < 0 or cy >= HEIGHT for _, cy in cells):
        return None
    placed = [row[:] for row in grid]
    for cx, cy in cells:
        placed[cy][cx] = 1
    survivors = [row for row in placed if sum(bool(v) for v in row) < WIDTH]
    lines = HEIGHT - len(survivors)
    survivors.extend([[0] * WIDTH for _ in range(lines)])
    facts = _metrics(survivors)
    facts.update(
        {
            "cells": tuple(cells),
            "landing_y": y,
            "landing_height": max(cy for _, cy in cells),
            "lines": lines,
        }
    )
    return survivors, facts


def _column_clean(grid: list[list[int]], x: int) -> bool:
    h = _column_heights(grid)[x]
    return all(grid[y][x] for y in range(h))


def _well_info(
    grid: list[list[int]], side: int | None, visible: list[str], policy: str
) -> dict[str, Any]:
    if side not in (0, 9):
        return {"side": "none", "status": "off", "i_visible": "I" in visible}
    m = _metrics(grid)
    heights = m["heights"]
    edge = heights[side]
    neighbor = heights[1] if side == 0 else heights[8]
    depth = max(0, neighbor - edge)
    clean = _column_clean(grid, side)
    i_visible = "I" in visible
    # Recovery and a dirty/tall surface take precedence over the opening.
    if policy == "RECOVER" or m["holes"] > 0 or m["max_height"] > 12:
        status = "off"
    elif not i_visible:
        status = "reserved" if clean and edge <= 2 else "off"
    elif depth >= 3 and edge <= 2 and clean:
        status = "ready"
    elif clean and edge <= max(2, min(4, neighbor)):
        status = "reserved"
    else:
        status = "off"
    return {
        "side": "left" if side == 0 else "right",
        "side_index": side,
        "status": status,
        "i_visible": i_visible,
        "depth": depth,
        "edge_height": edge,
        "neighbor_height": neighbor,
        "clean": clean,
    }


def _advance(state: dict[str, Any], lines: int, difficult: bool) -> dict[str, Any]:
    total = _integer(state.get("lines", 0)) + lines
    old_combo = _integer(state.get("combo", -1), -1)
    combo = (old_combo + 1) if lines else -1
    old_b2b = bool(state.get("back_to_back", False))
    if difficult:
        b2b = True
    elif lines:
        b2b = False
    else:
        b2b = old_b2b
    return {
        "lines": total,
        # Preserve a controlled scenario's already-started level while still
        # following the standard ten-lines progression in the normal start.
        "level": max(_integer(state.get("level", 1), 1), 1 + total // 10),
        "combo": combo,
        "back_to_back": b2b,
    }


def _event_utility(lines: int, state: dict[str, Any], difficult: bool) -> float:
    level = max(1, _integer(state.get("level", 1), 1))
    base = (0, 100, 300, 500, 800)[min(4, max(0, lines))] * level
    combo = _integer(state.get("combo", -1), -1)
    value = 0.16 * base
    if lines and combo >= 0:
        value += min(32.0, 5.0 * (combo + 1) * max(1, level))
    if difficult:
        value += 100.0 + (70.0 if bool(state.get("back_to_back", False)) else 0.0)
    if lines == 2:
        value += 12.0
    if lines == 3:
        value += 35.0
    return value


def _well_plan_facts(
    before_grid: list[list[int]],
    after_grid: list[list[int]],
    cells: tuple[tuple[int, int], ...],
    piece: str,
    lines: int,
    side: int | None,
    visible_i: bool,
) -> dict[str, Any]:
    if side not in (0, 9):
        return {"preserves_well": False, "breaks_well": False, "well_clear": False}
    before_h = _column_heights(before_grid)[side]
    after_h = _column_heights(after_grid)[side]
    touched = sum(1 for cx, _ in cells if cx == side)
    vertical_i = piece == "I" and touched == 4 and all(cx == side for cx, _ in cells)
    clean_after = _column_clean(after_grid, side)
    m = _metrics(after_grid)
    # A four-line I in the reserved edge is the targeted 9-0 event.  A
    # horizontal Tetris is still difficult, but is not called a well clear.
    well_clear = bool(piece == "I" and lines == 4 and vertical_i)
    preserve = bool(
        piece != "I"
        and visible_i
        and touched == 0
        and clean_after
        and after_h <= max(3, before_h + 1)
        and m["holes"] == 0
    )
    breaks = bool(piece != "I" and visible_i and touched > 0)
    return {
        "preserves_well": preserve,
        "breaks_well": breaks,
        "well_clear": well_clear,
        "well_depth_after": max(
            0,
            (m["heights"][1] if side == 0 else m["heights"][8]) - after_h,
        ),
    }


def _placement_rank(
    after: dict[str, Any],
    before: dict[str, Any],
    state: dict[str, Any],
    policy: str,
    well: dict[str, Any],
    piece: str,
    lines: int,
    source: str,
) -> float:
    holes = float(after.get("holes", 0))
    old_holes = _integer(before.get("holes", 0))
    max_h = float(after.get("max_height", 0))
    old_max = _integer(before.get("max_height", 0))
    aggregate = float(after.get("aggregate_height", 0))
    delta_agg = aggregate - float(before.get("aggregate_height", 0))
    delta_holes = _integer(after.get("holes", 0)) - old_holes
    delta_max = _integer(after.get("max_height", 0)) - old_max
    if policy == "RECOVER":
        hole_w, height_w, max_w, line_w = 20.0, 1.0, 3.5, 124.0
    elif policy == "WELL":
        hole_w, height_w, max_w, line_w = 15.0, 0.72, 2.5, 130.0
    else:
        hole_w, height_w, max_w, line_w = 16.0, 0.78, 2.7, 128.0
    value = (
        line_w * lines
        + 26.0 * lines * lines
        + _event_utility(lines, state, lines == 4)
        - hole_w * holes
        - 2.6 * float(after.get("hole_depth", 0))
        - 2.0 * hole_w * max(0, delta_holes)
        - height_w * aggregate
        - max_w * max_h
        - 1.8 * max_w * max(0, delta_max)
        - 0.48 * float(after.get("bumpiness", 0))
        - 0.07 * float(after.get("row_transitions", 0))
        - 0.08 * float(after.get("wells", 0))
        - 0.12 * float(after.get("landing_height", 0))
        - 0.15 * max(0.0, delta_agg)
    )
    if source == "hold":
        value -= 3.5
    if lines == 4:
        value += 105.0
    if policy != "RECOVER" and well.get("status") != "off":
        if well.get("well_clear"):
            value += 360.0
        elif well.get("preserves_well"):
            value += 24.0 if policy == "WELL" else 15.0
        elif well.get("breaks_well") and well.get("i_visible"):
            value -= 36.0
        if piece == "I" and lines == 4:
            value += 120.0
    return value


def _source_options(obs: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    active = obs.get("active") or {}
    current = str(active.get("type", "O"))
    if current not in PIECES:
        current = "O"
    queue = [str(x) for x in (obs.get("queue") or []) if str(x) in PIECES]
    result = [("current", current, queue)]
    if bool(obs.get("can_hold", False)):
        held = obs.get("hold")
        if held in PIECES:
            result.append(("hold", str(held), queue))
        elif queue:
            result.append(("hold", queue[0], queue[1:]))
    return result


def _one_piece(
    grid: list[list[int]],
    piece: str,
    before: dict[str, Any],
    state: dict[str, Any],
    policy: str,
    side: int | None,
    source: str,
    visible_i: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    orientations = (0,) if piece == "O" else range(4)
    for ori in orientations:
        for x in range(-4, WIDTH + 4):
            landed = _land(grid, piece, ori, x)
            if landed is None:
                continue
            after_grid, facts = landed
            well_facts = _well_plan_facts(
                grid, after_grid, facts["cells"], piece, int(facts["lines"]), side, visible_i
            )
            # Keep the pre-placement well status separate from the candidate's
            # after-state flags.  Without this separation a plan could receive
            # well credit even after the opening had been disabled for height or
            # holes.
            well_context = _well_info(
                grid, side, ["I"] if visible_i else [], policy
            )
            combined = dict(facts)
            combined.update(well_context)
            combined.update(well_facts)
            rank = _placement_rank(
                combined, before, state, policy, combined, piece,
                int(facts["lines"]), source,
            )
            result.append(
                {
                    "piece": piece,
                    "orientation": ori,
                    "x": x,
                    "after": after_grid,
                    "features": combined,
                    "lines": int(facts["lines"]),
                    "source": source,
                    "rank": rank,
                    "immediate": rank,
                    "future_value": 0.0,
                    "deep_value": None,
                }
            )
    result.sort(
        key=lambda p: (
            float(p.get("rank", -10**9)),
            int(p.get("lines", 0)),
            -_integer((p.get("features") or {}).get("holes", 99), 99),
            -_integer((p.get("features") or {}).get("max_height", 99), 99),
            1 if p.get("source") == "current" else 0,
            -_integer(p.get("x", 0)),
        ),
        reverse=True,
    )
    return result


def _future_state(state: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    lines = _integer(plan.get("lines", 0))
    return _advance(state, lines, lines == 4)


def _branch_value(
    grid: list[list[int]],
    queue: list[str],
    state: dict[str, Any],
    policy: str,
    side: int | None,
    depth: int,
    width: int,
    visible_i: bool,
) -> float:
    if depth <= 0 or not queue:
        return 0.0
    piece = queue[0]
    if piece not in PIECES:
        return 0.0
    before = _metrics(grid)
    choices = _one_piece(
        grid, piece, before, state, policy, side, "future", visible_i or "I" in queue[:3]
    )
    if not choices:
        return -500.0
    selected = list(choices[: max(1, width)])
    # Keep the future I-clear branch in the beam even if its immediate board
    # score is slightly below a single-clear branch.
    for special in choices:
        f = special.get("features") or {}
        if f.get("well_clear") or f.get("preserves_well"):
            if special not in selected:
                selected.append(special)
            if len(selected) >= width + 2:
                break
    best = -10**12
    for choice in selected:
        value = float(choice.get("rank", -10**9))
        next_state = _future_state(state, choice)
        if depth > 1:
            value += 0.34 * _branch_value(
                choice["after"], queue[1:], next_state, policy, side,
                depth - 1, max(3, width // 2), visible_i or "I" in queue[:3],
            )
        if value > best:
            best = value
    return best


def _make_plans(obs: dict[str, Any], policy: str, side: int | None) -> list[dict[str, Any]]:
    grid = _board(obs)
    before = _metrics(grid)
    state = {
        "lines": _integer(obs.get("lines", 0)),
        "level": _integer(obs.get("level", 1), 1),
        "combo": _integer(obs.get("combo", -1), -1),
        "back_to_back": bool(obs.get("back_to_back", False)),
    }
    active = obs.get("active") or {}
    queue_values = [str(x) for x in (obs.get("queue") or []) if str(x) in PIECES]
    visible = [str(active.get("type", "")), str(obs.get("hold", ""))] + queue_values[:5]
    i_visible = "I" in visible
    all_plans: list[dict[str, Any]] = []
    for source, piece, future_queue in _source_options(obs):
        for raw in _one_piece(grid, piece, before, state, policy, side, source, i_visible):
            plan = dict(raw)
            plan["future_queue"] = tuple(future_queue)
            # Start with the cheap value for every legal destination.  The
            # stateful beam is deliberately focused: doing a two-ply expansion
            # for every orientation/column made the worker slower than the
            # piece clock without adding useful candidates.
            plan["future_value"] = 0.0
            plan["deep_value"] = None
            plan["rank"] = float(plan["immediate"])
            all_plans.append(plan)
    all_plans.sort(
        key=lambda p: (
            float(p.get("rank", -10**9)),
            int(p.get("lines", 0)),
            -_integer((p.get("features") or {}).get("holes", 99), 99),
            -_integer((p.get("features") or {}).get("max_height", 99), 99),
            1 if p.get("source") == "current" else 0,
            -_integer(p.get("x", 0)),
        ),
        reverse=True,
    )
    if not all_plans:
        return []
    focus: list[dict[str, Any]] = []
    focus_keys: set[tuple[str, int, int]] = set()

    def focus_add(plan: dict[str, Any]) -> None:
        key = (str(plan.get("source")), _integer(plan.get("orientation")), _integer(plan.get("x")))
        if key not in focus_keys and len(focus) < 14:
            focus_keys.add(key)
            focus.append(plan)

    for plan in all_plans[:10]:
        focus_add(plan)
    # The point of this experiment is easy to miss if the best I-well branch
    # is just outside the ordinary feature ranking, so always expand a small
    # number of well and multi-line representatives too.
    for predicate in (
        lambda p: bool((p.get("features") or {}).get("well_clear")),
        lambda p: bool((p.get("features") or {}).get("preserves_well")),
        lambda p: _integer(p.get("lines", 0)) >= 2,
    ):
        for plan in all_plans:
            if predicate(plan):
                focus_add(plan)
                break

    for plan in focus:
        future_queue = list(plan.get("future_queue") or ())
        if not future_queue:
            continue
        try:
            future = _branch_value(
                plan["after"], future_queue, _future_state(state, plan), policy,
                side, depth=2, width=4, visible_i=i_visible,
            )
        except Exception:
            future = 0.0
        plan["future_value"] = future
        plan["deep_value"] = future
        plan["rank"] = float(plan["immediate"]) + 0.20 * float(future)
    all_plans.sort(
        key=lambda p: (
            float(p.get("rank", -10**9)),
            int(p.get("lines", 0)),
            -_integer((p.get("features") or {}).get("holes", 99), 99),
            -_integer((p.get("features") or {}).get("max_height", 99), 99),
            1 if p.get("source") == "current" else 0,
            -_integer(p.get("x", 0)),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    def add(plan: dict[str, Any] | None) -> None:
        if plan is None or len(kept) >= 10:
            return
        key = (str(plan.get("source")), _integer(plan.get("orientation")), _integer(plan.get("x")))
        if key not in seen:
            seen.add(key)
            kept.append(plan)

    add(all_plans[0])
    for predicate in (
        lambda p: bool((p.get("features") or {}).get("well_clear")),
        lambda p: bool((p.get("features") or {}).get("preserves_well")),
        lambda p: int(p.get("lines", 0)) >= 2,
    ):
        add(next((p for p in all_plans if predicate(p)), None))
    for source in ("current", "hold"):
        add(next((p for p in all_plans if p.get("source") == source), None))
    for ori in range(4):
        add(next((p for p in all_plans if _integer(p.get("orientation")) == ori), None))
    for p in all_plans:
        add(p)
    return kept


def _coarse(obs: dict[str, Any], policy: str, side: int | None) -> list[dict[str, Any]]:
    grid = _board(obs)
    before = _metrics(grid)
    state = {
        "lines": _integer(obs.get("lines", 0)),
        "level": _integer(obs.get("level", 1), 1),
        "combo": _integer(obs.get("combo", -1), -1),
        "back_to_back": bool(obs.get("back_to_back", False)),
    }
    active = obs.get("active") or {}
    visible = [str(active.get("type", "")), str(obs.get("hold", ""))]
    visible += [str(x) for x in (obs.get("queue") or [])[:5]]
    out: list[dict[str, Any]] = []
    for source, piece, future_queue in _source_options(obs):
        for plan in _one_piece(
            grid, piece, before, state, policy, side, source, "I" in visible
        ):
            plan = dict(plan)
            plan["future_queue"] = tuple(future_queue)
            out.append(plan)
    out.sort(key=lambda p: float(p.get("rank", -10**9)), reverse=True)
    return out[:14]


def _height_bucket(v: Any) -> str:
    n = _integer(v, 20)
    return "low" if n <= 5 else "moderate" if n <= 10 else "high" if n <= 16 else "critical"


def _hole_bucket(v: Any) -> str:
    n = _integer(v, 99)
    return "hole-free" if n == 0 else "few" if n <= 3 else "several" if n <= 8 else "many"


def _surface_bucket(v: Any) -> str:
    n = _integer(v, 99)
    return "flat" if n <= 8 else "uneven" if n <= 18 else "rough"


def _clear_word(n: Any) -> str:
    return {0: "no clear", 1: "single", 2: "double", 3: "triple", 4: "Tetris"}.get(_integer(n), "clear")


def _describe_plan(plan: dict[str, Any], well: dict[str, Any]) -> str:
    f = plan.get("features") or {}
    tags = []
    if f.get("well_clear"):
        tags.append("vertical I well clear")
    elif f.get("preserves_well"):
        tags.append("preserves the reserved I edge")
    elif f.get("breaks_well"):
        tags.append("uses the reserved edge")
    if int(plan.get("lines", 0)) == 4:
        tags.append("difficult clear")
    if not tags:
        tags.append("ordinary surface placement")
    source = "use hold then place" if plan.get("source") == "hold" else "place current piece"
    return (
        f"{source}; {_clear_word(plan.get('lines', 0))}; "
        + ", ".join(tags)
        + f"; projected {_height_bucket(f.get('max_height', 20))} stack, "
        + f"{_hole_bucket(f.get('holes', 99))} board, "
        + f"{_surface_bucket(f.get('bumpiness', 99))} surface."
    )


def _roles(obs: dict[str, Any], plans: list[dict[str, Any]], well: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not plans:
        return {}
    def f(p: dict[str, Any]) -> dict[str, Any]:
        return p.get("features") if isinstance(p.get("features"), dict) else {}
    safe = plans[0]
    recovery = min(
        plans,
        key=lambda p: (
            _integer(f(p).get("holes", 99), 99),
            _integer(f(p).get("hole_depth", 99), 99),
            _integer(f(p).get("max_height", 99), 99),
            _integer(f(p).get("aggregate_height", 999), 999),
            -_integer(p.get("lines", 0)),
            -float(p.get("rank", -10**9)),
        ),
    )
    clean_clears = [
        p for p in plans
        if _integer(p.get("lines", 0)) > 0
        and _integer(f(p).get("holes", 99), 99) <= _integer(obs.get("holes", 0)) + 1
        and _integer(f(p).get("max_height", 99), 99) <= max(18, _integer(obs.get("max_height", obs.get("top_out_height", 0))) + 5)
    ]
    clear = max(clean_clears or plans, key=lambda p: (int(p.get("lines", 0)), float(p.get("rank", -10**9))))
    result = {"safe": safe, "clear": clear, "recover": recovery}
    well_candidates = [
        p for p in plans
        if (f(p).get("well_clear") or f(p).get("preserves_well"))
        and _integer(f(p).get("holes", 99), 99) <= _integer(obs.get("holes", 0))
        and _integer(f(p).get("max_height", 99), 99) <= _integer(obs.get("max_height", 0)) + 5
    ]
    if well_candidates and well.get("status") != "off":
        result["well"] = max(well_candidates, key=lambda p: float(p.get("rank", -10**9)))
    return result


def _jev_state(obs: dict[str, Any], policy: str, well: dict[str, Any], roles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    active = obs.get("active") or {}
    return {
        "ruleset": str(obs.get("ruleset", "guideline-marathon-v1")),
        "policy": policy,
        "piece_instance": _integer(obs.get("pieces", 0)),
        "board": {
            "height": _height_bucket(obs.get("max_height", obs.get("top_out_height", 0))),
            "holes": _hole_bucket(obs.get("holes", 0)),
            "surface": _surface_bucket(obs.get("bumpiness", 0)),
            "lines": _integer(obs.get("lines", 0)),
            "level": _integer(obs.get("level", 1), 1),
            "combo": "active" if _integer(obs.get("combo", -1), -1) >= 0 else "inactive",
            "back_to_back": bool(obs.get("back_to_back", False)),
        },
        "active_piece": {
            "type": str(active.get("type", "")),
            "orientation": _integer(active.get("orientation", 0)),
            "anchor_x": _integer(active.get("x", 3), 3),
        },
        "hold": obs.get("hold") if obs.get("hold") in PIECES else None,
        "can_hold": bool(obs.get("can_hold", False)),
        "next_queue": [str(x) for x in (obs.get("queue") or [])[:5]],
        "well": {
            "side": str(well.get("side", "none")),
            "status": str(well.get("status", "off")),
            "i_visible": bool(well.get("i_visible", False)),
        },
        "fallback_role": "safe",
        "roles": {name: _describe_plan(plan, well) for name, plan in roles.items()},
        "candidate_count": len(roles),
    }


def _prob(answer: Any, label: Any) -> float:
    try:
        return float((getattr(answer, "probabilities", {}) or {}).get(label, 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _answer_role(result: Any, roles: dict[str, dict[str, Any]]) -> str | None:
    if not isinstance(result, dict):
        return None
    a, b = result.get("pick_a"), result.get("pick_b")
    av, bv = getattr(a, "value", None), getattr(b, "value", None)
    if not isinstance(av, str) or av not in roles or bv != av:
        return None
    if _prob(a, av) < 0.45 or _prob(b, bv) < 0.45:
        return None
    return av


def _questions(state: dict[str, Any], options: list[tuple[str, str]]) -> dict[str, Choice]:
    first = {k: v for k, v in options}
    second = {k: v for k, v in reversed(options)}
    common = (
        "Choose one code-generated placement role from `roles`. Recovery wins "
        "when the board is dangerous. Use `well` only when its description says "
        "the reserved edge or I clear is safe and useful; otherwise prefer safe "
        "or clear. Do not count, calculate score, compare raw numbers, invent a "
        "coordinate, or output an action. Select a label even if uncertain. "
        "State=" + json.dumps(state, separators=(",", ":"), ensure_ascii=True)
    )
    return {
        "pick_a": Choice(common + " Primary judgment.", criteria=first),
        "pick_b": Choice(common + " Independent recheck with reversed option order.", criteria=second),
    }


def _action(spec: Any, desired: str) -> str:
    names = list(getattr(spec, "names", ()) or ())
    if desired in names:
        return desired
    noop = getattr(spec, "noop", None)
    if noop in names:
        return noop
    if "noop" in names:
        return "noop"
    return str(names[0]) if names else "noop"


def _piece_id(obs: dict[str, Any]) -> tuple[int, str]:
    return (_integer(obs.get("pieces", 0)), str((obs.get("active") or {}).get("type", "")))


def _compatible(obs: dict[str, Any], plan: dict[str, Any], held: bool) -> bool:
    active = str((obs.get("active") or {}).get("type", ""))
    target = str(plan.get("piece", ""))
    if plan.get("source") == "hold":
        return bool(obs.get("can_hold", False)) if not held else active == target
    return not held and active == target


def _rotate_pose(grid: list[list[int]], piece: str, ori: int, x: int, y: int, action: str) -> tuple[int, int, int] | None:
    if piece == "O":
        return None
    target = (ori + (1 if action == "rotate_cw" else -1)) % 4
    table = _KICKS["I" if piece == "I" else "JLSTZ"]
    for dx, dy in table.get((ori % 4, target), ((0, 0),)):
        if _legal_pose(grid, piece, target, x + dx, y + dy):
            return target, x + dx, y + dy
    return None


def _route(obs: dict[str, Any], plan: dict[str, Any], spec: Any) -> str:
    active = obs.get("active") or {}
    piece = str(active.get("type", ""))
    if piece not in PIECES or piece != str(plan.get("piece", "")):
        return _action(spec, "hard_drop")
    start = (
        _integer(active.get("orientation", 0)) % 4,
        _integer(active.get("x", 3), 3),
        _integer(active.get("y", 17), 17),
    )
    target = (_integer(plan.get("orientation", start[0])) % 4, _integer(plan.get("x", start[1]), start[1]))
    if (start[0], start[1]) == target:
        return _action(spec, "hard_drop")
    grid = _board(obs)
    pending = deque([(start, ())])
    visited = {start}
    while pending:
        (ori, x, y), path = pending.popleft()
        if (ori, x) == target and path:
            return _action(spec, path[0])
        if len(path) >= 9:
            continue
        for move in ("left", "right", "rotate_cw", "rotate_ccw"):
            if move == "left" or move == "right":
                nx = x + (-1 if move == "left" else 1)
                candidate = (ori, nx, y)
                if not _legal_pose(grid, piece, *candidate):
                    continue
            else:
                rotated = _rotate_pose(grid, piece, ori, x, y, move)
                if rotated is None:
                    continue
                candidate = rotated
            if candidate in visited:
                continue
            visited.add(candidate)
            pending.append((candidate, path + (move,)))
    if start[1] < target[1] and _legal_pose(grid, piece, start[0], start[1] + 1, start[2]):
        return _action(spec, "right")
    if start[1] > target[1] and _legal_pose(grid, piece, start[0], start[1] - 1, start[2]):
        return _action(spec, "left")
    for move in ("rotate_cw", "rotate_ccw"):
        if start[0] != target[0] and _rotate_pose(grid, piece, *start, move) is not None:
            return _action(spec, move)
    return _action(spec, "hard_drop")


def _next(obs: dict[str, Any], plan: dict[str, Any], held: bool, spec: Any) -> str:
    if plan.get("source") == "hold" and not held:
        return _action(spec, "hold") if bool(obs.get("can_hold", False)) else _action(spec, "hard_drop")
    return _route(obs, plan, spec)


def _policy_token(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    match = re.match(r"^(?:PLAN\s*[:=]\s*)?(WELL|BALANCED|RECOVER)\b", text)
    return match.group(1) if match else None


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.policy = "BALANCED"
        self.well_side: int | None = None
        self.astra_future: Any = None
        self.astra_reviews = 0
        self.astra_last_piece = -1
        self.search_future: Any = None
        self.search_piece: tuple[int, str] | None = None
        self.jev_future: Any = None
        self.jev_piece: tuple[int, str] | None = None
        self.jev_roles: dict[str, dict[str, Any]] = {}
        self.jev_state: dict[str, Any] | None = None

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _choose_side(self, obs: dict[str, Any]) -> int:
        if self.well_side in (0, 9):
            return int(self.well_side)
        heights = _column_heights(_board(obs))
        # A tie goes right, the conventional 9-0 side; otherwise reserve the
        # edge that is already lower, minimizing the opening correction.
        self.well_side = 0 if heights[0] < heights[9] else 9
        return int(self.well_side)

    def _astra_start(self, obs: dict[str, Any], reason: str) -> None:
        if self.astra_future is not None or self.astra_reviews >= 2:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if int(remaining.get("system2_calls", 0)) <= 0 or float(remaining.get("wall_s", 0.0)) < 70.0:
                return
        except Exception:
            pass
        self.astra_reviews += 1
        self.astra_last_piece = _integer(obs.get("pieces", 0))
        active = obs.get("active") or {}
        visible = [str(active.get("type", "")), str(obs.get("hold", ""))]
        visible += [str(x) for x in (obs.get("queue") or [])[:5]]
        well = _well_info(_board(obs), self.well_side, visible, self.policy)
        payload = {
            "reason": reason,
            "ruleset": str(obs.get("ruleset", "guideline-marathon-v1")),
            "pieces": _integer(obs.get("pieces", 0)),
            "height": _height_bucket(obs.get("max_height", obs.get("top_out_height", 0))),
            "holes": _hole_bucket(obs.get("holes", 0)),
            "surface": _surface_bucket(obs.get("bumpiness", 0)),
            "lines": _integer(obs.get("lines", 0)),
            "level": _integer(obs.get("level", 1), 1),
            "combo_active": _integer(obs.get("combo", -1), -1) >= 0,
            "back_to_back": bool(obs.get("back_to_back", False)),
            "active": str(active.get("type", "")),
            "hold": obs.get("hold"),
            "queue": [str(x) for x in (obs.get("queue") or [])[:5]],
            "well": {
                "side": well.get("side"),
                "status": well.get("status"),
                "i_visible": well.get("i_visible"),
            },
            "current_policy": self.policy,
        }
        prompt = (
            "You are Astra, a slow strategic reviewer for a real-time Guideline-compatible "
            "Tetris marathon. Code owns exact legal placement search and a verified "
            "executor. Review these derived facts only. Return exactly one leading token: "
            "WELL if a shallow clean edge well and visible I should be preserved, "
            "RECOVER if height/holes dominate, or BALANCED otherwise. Never return a "
            "coordinate, action, sequence, arithmetic, or seed-specific script. The token "
            "only changes future evaluator weights. Facts="
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
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

    def _astra_harvest(self) -> None:
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            policy = _policy_token(future.result())
        except ModelUnavailable:
            raise
        except Exception:
            policy = None
        if policy and policy != self.policy:
            self.policy = policy
            self._log("astra_policy_update", policy=policy)

    def _astra_maybe(self, obs: dict[str, Any]) -> None:
        pieces = _integer(obs.get("pieces", 0))
        height = _integer(obs.get("max_height", obs.get("top_out_height", 0)))
        holes = _integer(obs.get("holes", 0))
        if self.astra_reviews == 0 and pieces >= 7:
            self._astra_start(obs, "opening_well_review")
        elif (
            self.astra_reviews == 1
            and self.astra_future is None
            and pieces >= 25
            and (height >= 12 or holes >= 1)
            and pieces > self.astra_last_piece + 8
        ):
            self._astra_start(obs, "well_abandon_review")

    def _emergency(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        active = obs.get("active") or {}
        piece = str(active.get("type", "O"))
        if piece not in PIECES:
            piece = "O"
        ori = _integer(active.get("orientation", 0)) % 4
        x = _integer(active.get("x", 3), 3)
        landed = _land(_board(obs), piece, ori, x)
        if landed is None:
            return {"piece": piece, "orientation": ori, "x": x, "source": "current", "features": {}, "lines": 0, "rank": -10**9}
        after, facts = landed
        return {"piece": piece, "orientation": ori, "x": x, "source": "current", "after": after, "features": facts, "lines": int(facts.get("lines", 0)), "rank": -10**8, "immediate": -10**8}

    def _search_start(self, obs: dict[str, Any]) -> None:
        if self.search_future is not None and not self.search_future.done():
            return
        if self.search_future is not None:
            self._search_take(None)
        pid = _piece_id(obs)
        snapshot = copy.deepcopy(obs)
        policy = self.policy
        side = self.well_side

        def work() -> list[dict[str, Any]]:
            try:
                return _make_plans(snapshot, policy, side)
            except Exception:
                return []

        try:
            self.search_future = self.ctx.spawn(work)
            self.search_piece = pid
        except Exception:
            self.search_future = None
            self.search_piece = None

    def _search_take(self, current: tuple[int, str] | None) -> list[dict[str, Any]] | None:
        future = self.search_future
        if future is None or not future.done():
            return None
        tagged = self.search_piece
        self.search_future = None
        self.search_piece = None
        try:
            value = future.result()
        except Exception:
            value = []
        if tagged is None or current is None or tagged != current:
            return None
        return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []

    def _jev_start(self, obs: dict[str, Any], plans: list[dict[str, Any]], well: dict[str, Any]) -> bool:
        if getattr(self.ctx, "system1", None) is None or not plans:
            return True
        if self.jev_future is not None and not self.jev_future.done():
            return False
        if self.jev_future is not None:
            self._jev_discard()
        roles = _roles(obs, plans, well)
        if not roles:
            return True
        state = _jev_state(obs, self.policy, well, roles)
        options = [(name, _describe_plan(plan, well)) for name, plan in roles.items()]
        tagged = _piece_id(obs)
        self.jev_piece = tagged
        self.jev_roles = dict(roles)
        self.jev_state = state

        def call() -> Any:
            try:
                return self.ctx.decide(state, _questions(state, options), timeout_s=1.10)
            except ModelUnavailable:
                raise
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver):
                return None

        try:
            self.jev_future = self.ctx.spawn(call)
        except Exception:
            self.jev_future = None
        return True

    def _jev_discard(self) -> None:
        future = self.jev_future
        self.jev_future = None
        self.jev_piece = None
        self.jev_roles = {}
        self.jev_state = None
        if future is not None:
            try:
                future.result()
            except ModelUnavailable:
                raise
            except Exception:
                pass

    def _jev_harvest(self, obs: dict[str, Any], current: dict[str, Any] | None, held: bool) -> dict[str, Any] | None:
        future = self.jev_future
        if future is None or not future.done():
            return current
        tagged, roles = self.jev_piece, self.jev_roles
        try:
            result = future.result()
        except ModelUnavailable:
            raise
        except Exception:
            result = None
        self.jev_future = None
        self.jev_piece = None
        self.jev_roles = {}
        self.jev_state = None
        if tagged != _piece_id(obs):
            return current
        role = _answer_role(result, roles)
        if role is None:
            return current
        chosen = roles.get(role)
        if chosen is None or not _compatible(obs, chosen, held):
            return current
        if current is not None and float(chosen.get("rank", -10**9)) < float(current.get("rank", -10**9)) - 60.0:
            self._log("jev_role_rejected", piece=tagged[0], role=role, reason="value_envelope")
            return current
        self._log("jev_role", piece=tagged[0], role=role,
                  well=bool((chosen.get("features") or {}).get("well_clear")))
        return chosen

    def _post_wait(self, game: Any, action: str) -> bool:
        if bool(getattr(self.ctx, "done", False)) or bool(getattr(game, "done", False)):
            return False
        try:
            self.ctx.act(action)
            self.ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return False
        except Exception:
            return False
        return not bool(getattr(self.ctx, "done", False)) and not bool(getattr(game, "done", False))

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        try:
            initial = game.observe()
        except Exception:
            return
        if not isinstance(initial, dict):
            return
        self.well_side = None
        self.policy = "BALANCED"
        self.astra_future = None
        self.astra_reviews = 0
        self.astra_last_piece = -1
        self.search_future = None
        self.search_piece = None
        self.jev_future = None
        self.jev_piece = None
        self.jev_roles = {}
        self.jev_state = None
        spec = getattr(game, "action_spec", None)
        piece_number: int | None = None
        current_pid: tuple[int, str] | None = None
        plans: list[dict[str, Any]] = []
        guardian: dict[str, Any] | None = None
        selected: dict[str, Any] | None = None
        held = False
        request_jev: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None = None
        previous: str | None = None
        command_pid: tuple[int, str] | None = None
        command_x: int | None = None
        command_ori: int | None = None
        blocked = False

        def compatible_first(items: list[dict[str, Any]], ob: dict[str, Any], used: bool) -> dict[str, Any] | None:
            return next((p for p in items if _compatible(ob, p, used)), None)

        def urgent(ob: dict[str, Any]) -> bool:
            try:
                if ob.get("lock_remaining_s") is not None and float(ob.get("lock_remaining_s")) <= 0.10:
                    return True
            except (TypeError, ValueError):
                pass
            active = (ob.get("active") or {}).get("cells")
            ghost = ob.get("ghost_cells")
            if isinstance(active, list) and isinstance(ghost, list) and active and ghost:
                try:
                    return {tuple(x) for x in active} == {tuple(x) for x in ghost}
                except (TypeError, ValueError):
                    return False
            return False

        try:
            while not bool(getattr(ctx, "done", False)) and not bool(getattr(game, "done", False)):
                self._astra_harvest()
                try:
                    obs = game.observe()
                except Exception:
                    return
                if not isinstance(obs, dict) or bool(obs.get("game_over", False)):
                    return
                side = self._choose_side(obs)
                self._astra_maybe(obs)
                number = _integer(obs.get("pieces", 0))
                pid = _piece_id(obs)
                well = _well_info(
                    _board(obs), side,
                    [str((obs.get("active") or {}).get("type", "")), str(obs.get("hold", ""))]
                    + [str(x) for x in (obs.get("queue") or [])[:5]],
                    self.policy,
                )
                ready = self._search_take(pid)
                if ready is not None and number == piece_number and pid == current_pid:
                    plans = ready
                    replacement = compatible_first(plans, obs, held)
                    if replacement is not None:
                        selected = replacement
                    request_jev = (obs, list(plans), well) if plans else None
                    blocked = False
                    self._log(
                        "well_search_ready", piece=pid[0], candidates=len(plans),
                        well_status=well.get("status"),
                        well_clears=sum(1 for p in plans if (p.get("features") or {}).get("well_clear")),
                    )

                if number != piece_number:
                    piece_number = number
                    current_pid = pid
                    plans = []
                    held = False
                    request_jev = None
                    blocked = False
                    guardian = self._emergency(obs)
                    selected = guardian
                    try:
                        ctx.act(_action(spec, "noop"))
                    except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                        return
                    except Exception:
                        return
                    try:
                        coarse = _coarse(obs, self.policy, side)
                    except Exception:
                        coarse = []
                    try:
                        latest = game.observe()
                    except Exception:
                        return
                    if not isinstance(latest, dict) or bool(latest.get("game_over", False)):
                        return
                    if _piece_id(latest) != pid:
                        continue
                    if coarse:
                        guardian = coarse[0]
                        selected = guardian
                    else:
                        guardian = self._emergency(latest) or guardian
                        selected = guardian
                    obs = latest
                    self._search_start(obs)
                elif current_pid != pid:
                    current_pid = pid
                    blocked = False
                    if selected is None or not _compatible(obs, selected, held):
                        try:
                            coarse = _coarse(obs, self.policy, side)
                        except Exception:
                            coarse = []
                        guardian = coarse[0] if coarse else self._emergency(obs)
                        selected = guardian
                elif (
                    command_pid == pid
                    and previous in {"left", "right", "rotate_cw", "rotate_ccw"}
                ):
                    active_now = obs.get("active") or {}
                    now_x = _integer(active_now.get("x", 3), 3)
                    now_o = _integer(active_now.get("orientation", 0)) % 4
                    blocked = now_x == command_x and now_o == command_ori
                else:
                    blocked = False

                if piece_number == number and self.search_future is None and not plans:
                    self._search_start(obs)
                selected = self._jev_harvest(obs, selected, held)
                if selected is None or not _compatible(obs, selected, held):
                    selected = compatible_first(plans, obs, held) or (
                        guardian if guardian is not None and _compatible(obs, guardian, held) else self._emergency(obs)
                    )
                if blocked or selected is None or urgent(obs):
                    action = _action(spec, "hard_drop")
                else:
                    action = _next(obs, selected, held, spec)
                active_before = obs.get("active") or {}
                command_pid = pid
                command_x = _integer(active_before.get("x", 3), 3)
                command_ori = _integer(active_before.get("orientation", 0)) % 4
                if not self._post_wait(game, action):
                    return
                previous = action
                if action == "hold" and "hold" in (getattr(spec, "names", ()) or ()):
                    held = True
                if request_jev is not None:
                    launch_obs, launch_plans, launch_well = request_jev
                    try:
                        started = self._jev_start(launch_obs, launch_plans, launch_well)
                    except (BudgetExceeded, EpisodeOver):
                        return
                    except ModelUnavailable:
                        raise
                    except Exception:
                        started = True
                    if started:
                        request_jev = None
        finally:
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
