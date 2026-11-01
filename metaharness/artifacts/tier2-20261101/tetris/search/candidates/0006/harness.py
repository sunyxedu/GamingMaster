"""Beam-search placement controller for real-time Guideline Tetris.

The code searches a short known-queue continuation and owns all inputs. Jev is
an asynchronous, confidence-gated tie-breaker among safe named placements;
Astra can revise only a future risk mode after unusually long survival.
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

# The public spawn cells establish each piece's local origin. For an active
# piece, observations override the derived shape at its current orientation.
_BASE = {
    "J": ((0, 2), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 2), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 2), (2, 2), (0, 1), (1, 1)),
    "Z": ((0, 2), (1, 2), (1, 1), (2, 1)),
    "T": ((1, 2), (0, 1), (1, 1), (2, 1)),
    "I": ((0, 2), (1, 2), (2, 2), (3, 2)),
    "O": ((1, 2), (2, 2), (1, 1), (2, 1)),
}


def _rotate_shape(cells: tuple[tuple[int, int], ...], piece: str, cw: bool = True):
    if piece == "O":
        return tuple(sorted(cells))
    pivot = (1.5, 1.5) if piece == "I" else (1.0, 1.0)
    out = []
    for x, y in cells:
        dx, dy = x - pivot[0], y - pivot[1]
        nx = pivot[0] + (dy if cw else -dy)
        ny = pivot[1] + (-dx if cw else dx)
        out.append((int(round(nx)), int(round(ny))))
    return tuple(sorted(out))


def _make_shapes(piece: str):
    states = []
    cur = tuple(_BASE[piece])
    for _ in range(4):
        states.append(tuple(sorted(cur)))
        cur = _rotate_shape(cur, piece, True)
    return tuple(states)


SHAPES = {p: _make_shapes(p) for p in PIECES}


def _grid(obs: dict[str, Any]) -> list[list[int]]:
    grid = [[0] * WIDTH for _ in range(HEIGHT)]
    rows = obs.get("board")
    if not isinstance(rows, list):
        return grid
    for top_y, row in enumerate(rows[:HEIGHT]):
        if not isinstance(row, list):
            continue
        y = HEIGHT - 1 - top_y
        for x, cell in enumerate(row[:WIDTH]):
            try:
                grid[y][x] = 1 if int(cell) else 0
            except (TypeError, ValueError):
                grid[y][x] = 0
    return grid


def _observed_shapes(obs: dict[str, Any], piece: str):
    """Use observed active offsets as an orientation anchor where available."""
    active = obs.get("active") or {}
    if active.get("type") != piece:
        return SHAPES.get(piece, SHAPES["O"])
    try:
        ori = int(active.get("orientation", 0)) % 4
        ax, ay = int(active.get("x", 3)), int(active.get("y", 17))
    except (TypeError, ValueError):
        return SHAPES.get(piece, SHAPES["O"])
    cells = []
    for cell in active.get("cells") or []:
        try:
            cells.append((int(cell[0]) - ax, int(cell[1]) - ay))
        except (TypeError, ValueError, IndexError):
            pass
    if len(cells) != 4:
        return SHAPES.get(piece, SHAPES["O"])
    states = [None] * 4
    states[ori] = tuple(sorted(cells))
    cur = tuple(sorted(cells))
    for step in range(1, 4):
        cur = _rotate_shape(cur, piece, True)
        states[(ori + step) % 4] = cur
    return tuple(states)


def _features(grid: list[list[int]]) -> dict[str, Any]:
    heights = []
    holes = 0
    hole_depth = 0
    for x in range(WIDTH):
        top = 0
        for y in range(HEIGHT - 1, -1, -1):
            if grid[y][x]:
                top = y + 1
                break
        heights.append(top)
        occupied_above = 0
        for y in range(top - 1, -1, -1):
            if grid[y][x]:
                occupied_above += 1
            elif occupied_above:
                holes += 1
                hole_depth += occupied_above

    bump = sum(abs(heights[x] - heights[x + 1]) for x in range(WIDTH - 1))
    max_height = max(heights) if heights else 0
    wells = 0
    for x, h in enumerate(heights):
        left = heights[x - 1] if x else HEIGHT
        right = heights[x + 1] if x + 1 < WIDTH else HEIGHT
        wells += max(0, min(left, right) - h)

    row_transitions = 0
    near_rows = 0
    for y in range(HEIGHT):
        row = grid[y]
        filled = sum(1 for v in row if v)
        if filled >= 7 and filled < WIDTH:
            near_rows += 1
        prev = 1
        for x in range(WIDTH):
            cur = 1 if row[x] else 0
            if cur != prev:
                row_transitions += 1
            prev = cur
        if prev == 0:
            row_transitions += 1

    col_transitions = 0
    for x in range(WIDTH):
        prev = 1
        for y in range(heights[x]):
            cur = 1 if grid[y][x] else 0
            if cur != prev:
                col_transitions += 1
            prev = cur
        if prev == 0:
            col_transitions += 1

    return {
        "heights": heights,
        "aggregate_height": sum(heights),
        "max_height": max_height,
        "holes": holes,
        "hole_depth": hole_depth,
        "bumpiness": bump,
        "wells": wells,
        "transitions": row_transitions + col_transitions,
        "near_rows": near_rows,
    }


def _utility(f: dict[str, Any], policy: str, i_available: bool) -> float:
    """Board utility: safety dominates, but future row completion matters."""
    holes_w = 14.0 if policy == "RECOVER" else 11.5
    height_w = 0.62 if policy != "RECOVER" else 0.70
    bump_w = 0.48 if policy != "RECOVER" else 0.56
    value = (
        -holes_w * f["holes"]
        -1.25 * f["hole_depth"]
        -height_w * f["aggregate_height"]
        -1.8 * f["max_height"]
        -bump_w * f["bumpiness"]
        -0.10 * f["transitions"]
        - (0.12 if i_available else 0.30) * f["wells"]
        +2.6 * f["near_rows"]
    )
    # Once the tallest column is above a comfortable stack, danger rises
    # sharply; a small aggregate-height gain must not hide a spawn-level wall.
    if f["max_height"] > 8:
        value -= 2.4 * (f["max_height"] - 8) ** 2
    if policy == "CLEAR":
        value += 1.0 * f["near_rows"]
    return value


def _drop(grid: list[list[int]], piece: str, orientation: int, ax: int, shapes=None):
    shapes = shapes or SHAPES.get(piece, SHAPES["O"])
    shape = shapes[orientation % 4]
    min_dx = min(dx for dx, _ in shape)
    max_dx = max(dx for dx, _ in shape)
    if ax + min_dx < 0 or ax + max_dx >= WIDTH:
        return None

    def legal(y0: int) -> bool:
        for dx, dy in shape:
            x, y = ax + dx, y0 + dy
            if x < 0 or x >= WIDTH or y < 0:
                return False
            if y < HEIGHT and grid[y][x]:
                return False
        return True

    y = HEIGHT + 6
    while legal(y - 1):
        y -= 1
    if not legal(y):
        return None
    cells = [(ax + dx, y + dy) for dx, dy in shape]
    if any(cy < 0 or cy >= HEIGHT for _, cy in cells):
        return None
    placed = [row[:] for row in grid]
    for x, cy in cells:
        placed[cy][x] = 1
    kept = [row for row in placed if not all(row)]
    lines = HEIGHT - len(kept)
    if lines:
        kept.extend([[0] * WIDTH for _ in range(lines)])
    f = _features(kept)
    f["lines"] = lines
    f["landing_y"] = y
    f["cells"] = cells
    return kept, f


def _enumerate(grid, piece: str, shapes=None, orientations=None):
    shapes = shapes or SHAPES.get(piece, SHAPES["O"])
    oris = orientations if orientations is not None else (range(1) if piece == "O" else range(4))
    out = []
    for ori in oris:
        shape = shapes[ori % 4]
        min_dx = min(dx for dx, _ in shape)
        max_dx = max(dx for dx, _ in shape)
        for x in range(-min_dx, WIDTH - max_dx):
            result = _drop(grid, piece, ori, x, shapes)
            if result is None:
                continue
            after, f = result
            out.append({"piece": piece, "orientation": ori % 4, "x": x,
                        "board": after, "features": f, "lines": f["lines"]})
    return out


def _valid_piece(value: Any) -> bool:
    return isinstance(value, str) and value in PIECES


def _continuation_score(board, remaining, policy: str):
    """Cheap lower beam: score all second placements, then expand only its top two."""
    if not remaining:
        return 0.0
    seconds = _enumerate(board, remaining[0])
    scored = []
    for second in seconds:
        f2 = second["features"]
        v2 = _utility(f2, policy, "I" in remaining[1:3]) + 24.0 * second["lines"]
        scored.append((v2, second))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if not scored:
        return -150.0
    if len(remaining) == 1:
        return scored[0][0]
    best = -1.0e12
    # Expanding only the two best second placements retains a real three-piece
    # lookahead while keeping planning well below one gravity interval.
    for v2, second in scored[:2]:
        thirds = _enumerate(second["board"], remaining[1])
        if thirds:
            v3 = max(
                _utility(t["features"], policy, "I" in remaining[2:4])
                + 24.0 * t["lines"] for t in thirds
            )
        else:
            v3 = v2 - 150.0
        best = max(best, 0.48 * v2 + 0.52 * v3)
    return best


def _rank_fast(obs: dict[str, Any], policy: str):
    """Immediate fallback search; it finishes before the next 50 ms tick."""
    grid = _grid(obs)
    active = obs.get("active") or {}
    current = active.get("type") if _valid_piece(active.get("type")) else "O"
    queue = [p for p in (obs.get("queue") or []) if _valid_piece(p)]
    hold = obs.get("hold") if _valid_piece(obs.get("hold")) else None
    options = [("current", current, queue)]
    if bool(obs.get("can_hold", False)):
        if hold is not None:
            options.append(("hold", hold, queue))
        elif queue:
            options.append(("hold", queue[0], queue[1:]))

    ranked = []
    for source, piece, remaining in options:
        shapes = _observed_shapes(obs, piece) if source == "current" else SHAPES[piece]
        for first in _enumerate(grid, piece, shapes):
            immediate = (_utility(first["features"], policy, "I" in remaining[:3])
                         + 24.0 * first["lines"])
            if source == "hold":
                immediate -= 1.2
            first.update({"source": source, "rank": immediate, "future": 0.0,
                          "first_value": immediate, "shapes": shapes,
                          "remaining": remaining})
            ranked.append(first)

    # Expand only the ten immediate choices. This is the safe action available
    # while the full three-piece beam runs in a background worker.
    ranked.sort(key=lambda p: (p["rank"], p["lines"], -p["features"]["holes"],
                               -p["features"]["max_height"]), reverse=True)
    for first in ranked[:10]:
        first["future"] = _continuation_score(first["board"], first["remaining"][:1], policy)
        first["rank"] = 0.55 * first["first_value"] + 0.45 * first["future"]
    ranked.sort(key=lambda p: (p["rank"], p["lines"], -p["features"]["holes"],
                               -p["features"]["max_height"]), reverse=True)
    selected = []
    seen = set()
    for p in ranked:
        key = (p["source"], p["piece"], p["orientation"], p["x"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(p)
        if len(selected) >= 5:
            break
    return selected


def _rank_first(obs: dict[str, Any], policy: str):
    grid = _grid(obs)
    active = obs.get("active") or {}
    current = active.get("type") if _valid_piece(active.get("type")) else "O"
    queue = [p for p in (obs.get("queue") or []) if _valid_piece(p)]
    hold = obs.get("hold") if _valid_piece(obs.get("hold")) else None
    can_hold = bool(obs.get("can_hold", False))
    options = [("current", current, queue)]
    if can_hold:
        if hold is not None:
            options.append(("hold", hold, queue))
        elif queue:
            options.append(("hold", queue[0], queue[1:]))

    # First evaluate the immediate placement and the best two-piece continuation
    # for every first move. Only the strongest first moves get the more expensive
    # third-piece expansion; this is a bounded beam, not a full tree.
    ranked = []
    for source, piece, remaining in options:
        shapes = _observed_shapes(obs, piece) if source == "current" else SHAPES[piece]
        firsts = _enumerate(grid, piece, shapes)
        for first in firsts:
            f1 = first["features"]
            immediate = _utility(f1, policy, "I" in remaining[:3]) + 24.0 * first["lines"]
            if source == "hold":
                # Holding is a real resource decision; use it for a better
                # surface, not merely because the preview offers another legal
                # destination.
                immediate -= 1.2
            first.update({"source": source, "rank": immediate,
                          "future": 0.0, "first_value": immediate, "shapes": shapes,
                          "remaining": remaining})
            ranked.append(first)

    # A quick one-preview pass avoids a third-piece search for clearly inferior
    # first placements and is also useful if the queue is unusually short.
    for first in ranked:
        first["future"] = _continuation_score(first["board"], first["remaining"][:1], policy)
        first["rank"] = 0.42 * first["rank"] + 0.58 * first["future"]
    ranked.sort(key=lambda p: (p["rank"], p["lines"], -p["features"]["holes"],
                               -p["features"]["max_height"]), reverse=True)
    for first in ranked[:8]:
        first["future"] = _continuation_score(first["board"], first["remaining"][:2], policy)
        first["rank"] = 0.40 * first["first_value"] + 0.60 * first["future"]

    ranked.sort(key=lambda p: (p["rank"], p["lines"], -p["features"]["holes"],
                               -p["features"]["max_height"]), reverse=True)
    selected = []
    seen = set()
    for p in ranked:
        key = (p["source"], p["piece"], p["orientation"], p["x"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(p)
        if len(selected) >= 5:
            break
    return selected


def _rank_single(obs: dict[str, Any], policy: str, orientation: int | None = None):
    grid = _grid(obs)
    active = obs.get("active") or {}
    piece = active.get("type") if _valid_piece(active.get("type")) else "O"
    shapes = _observed_shapes(obs, piece)
    oris = [orientation % 4] if orientation is not None else None
    plans = _enumerate(grid, piece, shapes, oris)
    for p in plans:
        p["source"] = "current"
        p["rank"] = _utility(p["features"], policy, False) + 24.0 * p["lines"]
        p["future"] = 0.0
        p["shapes"] = shapes
    plans.sort(key=lambda p: p["rank"], reverse=True)
    return plans


def _emergency_plan(obs: dict[str, Any]):
    """A no-movement landing used after the engine rejects a planned input."""
    active = obs.get("active") or {}
    piece = active.get("type") if _valid_piece(active.get("type")) else "O"
    try:
        orientation = int(active.get("orientation", 0)) % 4
        x = int(active.get("x", 3))
    except (TypeError, ValueError):
        orientation, x = 0, 3
    return {
        "source": "current",
        "piece": piece,
        "orientation": orientation,
        "x": x,
        "rank": -1.0e12,
        "future": 0.0,
        "lines": 0,
        "features": _features(_grid(obs)),
    }


def _bucket_height(h: int) -> str:
    if h <= 5:
        return "low"
    if h <= 11:
        return "moderate"
    if h <= 16:
        return "high"
    return "critical"


def _bucket_holes(n: int, depth: int) -> str:
    if n == 0:
        return "hole-free"
    if n <= 3 and depth <= 5:
        return "few shallow holes"
    if n <= 8:
        return "several holes"
    return "many or deep holes"


def _plan_description(plan: dict[str, Any]) -> str:
    f = plan["features"]
    lines = int(plan.get("lines", 0))
    clear = {0: "no line clear", 1: "single", 2: "double", 3: "triple", 4: "Tetris"}.get(lines, "line clear")
    source = "use the available hold, then place" if plan.get("source") == "hold" else "place the current piece"
    rough = "smooth" if f["bumpiness"] <= 8 else ("uneven" if f["bumpiness"] <= 18 else "rough")
    fut = plan.get("future", 0.0)
    future_label = "strong future support" if fut > -35 else ("some future support" if fut > -100 else "weak future support")
    holes = _bucket_holes(f["holes"], f["hole_depth"])
    return (
        f"{source}; orientation {plan['orientation']}, landing column {plan['x']}; {clear}; "
        f"leaves {_bucket_height(f['max_height'])} maximum stack, {holes}, {rough} surface, "
        f"{future_label}."
    )


def _prob(answer: Any, choice: str) -> float:
    try:
        probs = getattr(answer, "probabilities", {}) or {}
        return float(probs.get(choice, 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _select_jev(result: Any, plans: list[dict[str, Any]]):
    if not isinstance(result, dict):
        return None
    valid = {f"P{i}": p for i, p in enumerate(plans)}
    a, b = result.get("pick_a"), result.get("pick_b")
    av, bv = getattr(a, "value", None), getattr(b, "value", None)
    if av not in valid or bv != av:
        return None
    if _prob(a, av) < 0.40 or _prob(b, bv) < 0.40:
        return None
    return valid[av]


def _safe_action(spec: Any, action: str) -> str:
    names = set(getattr(spec, "names", ()) or ())
    if action in names:
        return action
    noop = getattr(spec, "noop", None)
    if noop in names:
        return noop
    return next(iter(names), "noop")


def _pose(obs: dict[str, Any]):
    active = obs.get("active") or {}
    try:
        return (int(obs.get("pieces", 0)), str(active.get("type", "")),
                int(active.get("orientation", 0)) % 4, int(active.get("x", 3)),
                int(active.get("y", 17)))
    except (TypeError, ValueError):
        return (int(obs.get("pieces", 0) or 0), str(active.get("type", "")), 0, 3, 17)


def _state_for_jev(obs: dict[str, Any], plans: list[dict[str, Any]], policy: str):
    candidates = {f"P{i}": _plan_description(p) for i, p in enumerate(plans[:5])}
    active = obs.get("active") or {}
    try:
        holes = int(obs.get("holes", 0))
        height = int(obs.get("top_out_height", 0))
        bump = int(obs.get("bumpiness", 0))
        depth = int(obs.get("hole_depth", 0))
    except (TypeError, ValueError):
        holes, height, bump, depth = 0, 0, 0, 0
    try:
        combo = int(obs.get("combo", -1))
    except (TypeError, ValueError):
        combo = -1
    try:
        lines = int(obs.get("lines", 0))
    except (TypeError, ValueError):
        lines = 0
    try:
        level = int(obs.get("level", 1))
    except (TypeError, ValueError):
        level = 1
    try:
        near = int(obs.get("near_rows", 0))
    except (TypeError, ValueError):
        near = 0
    return {
        "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
        "policy": policy,
        "piece_instance": int(obs.get("pieces", 0) or 0),
        "board": {
            "height": _bucket_height(height),
            "holes": _bucket_holes(holes, depth),
            "surface": "smooth" if bump <= 8 else ("uneven" if bump <= 18 else "rough"),
            "near_rows": "many" if near >= 3 else "promising" if near else "none",
            "lines": lines,
            "level": level,
            "combo": "active" if combo >= 0 else "inactive",
            "back_to_back": bool(obs.get("back_to_back", False)),
        },
        "active_piece": active.get("type"),
        "hold": obs.get("hold"),
        "can_hold": bool(obs.get("can_hold", False)),
        "next_queue": list(obs.get("queue") or [])[:5],
        "fallback": "P0",
        "candidates": candidates,
    }


def _questions(state: dict[str, Any], plans: list[dict[str, Any]]):
    criteria = {f"P{i}": _plan_description(p) for i, p in enumerate(plans[:5])}
    reverse = dict(reversed(list(criteria.items())))
    instructions = (
        "Choose the safest useful legal placement label from `candidates`. These are "
        "precomputed landing results, not keypresses. Prefer reducing dangerous holes "
        "and top-out risk while keeping a low, smooth surface; take a line clear when "
        "it does not leave a worse trapped surface. Use the described future support "
        "as a tie-breaker. Do not count, calculate, invent a placement, or answer with "
        "anything except one candidate label. The code will revalidate the choice."
    )
    return {
        "pick_a": Choice(instructions + " Judge in the supplied order.", criteria),
        "pick_b": Choice(instructions + " Independently recheck with the options in reverse order.", reverse),
    }


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.policy = "BALANCED"
        self.astra_future = None
        self.astra_started = False
        self.policy_version = 0

    def _ask_jev(self, state, plans):
        try:
            return self.ctx.decide(state, _questions(state, plans), timeout_s=0.55)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
            return None
        except Exception:
            # Stand-in and malformed outputs are advisory failures; code's plan
            # remains executable.
            return None

    def _start_astra(self, obs):
        if self.astra_started or getattr(self.ctx, "system2", None) is None:
            return
        self.astra_started = True
        compact = {
            "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
            "reason": "controller has survived at least twenty placements; revise future risk mode only",
            "stack": {
                "holes": obs.get("holes", 0),
                "aggregate_height": obs.get("aggregate_height", 0),
                "top_out_height": obs.get("top_out_height", 0),
                "bumpiness": obs.get("bumpiness", 0),
            },
            "level": obs.get("level", 1),
            "hold": obs.get("hold"),
            "queue": list(obs.get("queue") or [])[:5],
            "current_mode": self.policy,
        }
        prompt = (
            "You are a slow risk-policy reviewer for a real-time Guideline Tetris "
            "controller. Code searches legal SRS-shaped placements over the visible "
            "queue; Jev only breaks ties among code-generated safe candidates. From "
            "these public facts, choose one future mode and return exactly one token: "
            "BALANCED for a low flexible stack, RECOVER when reducing holes/height "
            "dominates, or CLEAR when a safe near-complete row/well can be exploited. "
            "Do not give keypresses, coordinates, or a seed-specific script.\n" +
            json.dumps(compact, separators=(",", ":"))
        )

        def call():
            try:
                result = self.ctx.think(prompt, timeout_s=70.0)
                return getattr(result, "text", "")
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
                return None
            except Exception:
                return None

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception:
            self.astra_future = None

    def _harvest_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            text = str(future.result() or "").upper()
        except Exception:
            return
        match = re.search(r"\b(BALANCED|RECOVER|CLEAR)\b", text)
        if match and match.group(1) != self.policy:
            self.policy = match.group(1)
            self.policy_version += 1
            try:
                self.ctx.log("astra_mode_update", policy=self.policy, version=self.policy_version)
            except Exception:
                pass

    def _tick(self, game, action):
        ctx = self.ctx
        if ctx.done or getattr(game, "done", False):
            return False
        try:
            ctx.act(action)
            ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return False
        except Exception:
            return False
        return not ctx.done and not getattr(game, "done", False)

    def _action_for(self, obs, plan, hold_sent, spec):
        active = obs.get("active") or {}
        piece = active.get("type")
        if plan is None or not _valid_piece(piece):
            return _safe_action(spec, "hard_drop")
        if plan.get("source") == "hold" and not hold_sent:
            if bool(obs.get("can_hold", False)):
                return _safe_action(spec, "hold")
            return _safe_action(spec, "noop")
        if piece != plan.get("piece"):
            return _safe_action(spec, "noop")
        try:
            ori = int(active.get("orientation", 0)) % 4
            x = int(active.get("x", 3))
        except (TypeError, ValueError):
            return _safe_action(spec, "noop")
        delta = (int(plan["orientation"]) - ori) % 4
        if delta:
            return _safe_action(spec, "rotate_cw" if delta in (1, 2) else "rotate_ccw")
        if x < int(plan["x"]):
            return _safe_action(spec, "right")
        if x > int(plan["x"]):
            return _safe_action(spec, "left")
        return _safe_action(spec, "hard_drop")

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if not getattr(game, "tick_s", None):
            return
        spec = game.action_spec
        piece_id = None
        plans = []
        fallback = None
        selected = None
        decision_future = None
        decision_id = None
        hold_sent = False
        last_command = None
        last_before = None
        replan_count = 0

        while not ctx.done and not getattr(game, "done", False):
            self._harvest_astra()
            try:
                obs = game.observe()
            except Exception:
                return
            if obs.get("game_over") or getattr(game, "done", False):
                return

            current_id = int(obs.get("pieces", 0) or 0)
            if current_id != piece_id:
                # Replace any prior hard_drop immediately. Otherwise a long
                # board search could let one stale drop lock several pieces.
                try:
                    ctx.act(_safe_action(spec, "noop"))
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except Exception:
                    return
                piece_id = current_id
                hold_sent = False
                last_command = None
                last_before = None
                replan_count = 0
                plans = _rank_first(obs, self.policy)
                fallback = plans[0] if plans else None
                selected = fallback
                decision_future = None
                decision_id = current_id
                if getattr(ctx, "system1", None) is not None and plans:
                    state = _state_for_jev(obs, plans, self.policy)
                    try:
                        decision_future = ctx.spawn(lambda s=state, ps=plans[:]: self._ask_jev(s, ps))
                    except Exception:
                        decision_future = None
                if current_id >= 20:
                    self._start_astra(obs)

            # If a move/rotation did not actually change the pose, that target
            # is not reachable from here. Re-search only the current orientation.
            pose = _pose(obs)
            if last_command in ("left", "right", "rotate_cw", "rotate_ccw") and last_before is not None:
                unchanged = ((last_command in ("left", "right") and pose[3] == last_before[3]) or
                             (last_command in ("rotate_cw", "rotate_ccw") and pose[2] == last_before[2]))
                if pose[0] == last_before[0] and unchanged:
                    try:
                        emergency = _emergency_plan(obs)
                        plans = [emergency]
                        fallback = emergency
                        selected = emergency
                        decision_future = None
                        decision_id = None
                        replan_count += 1
                        ctx.log("placement_replan", reason="input_did_not_change_pose", piece=pose[0], count=replan_count)
                    except Exception:
                        pass
            last_command = None
            last_before = None

            # Accept only a current-piece answer before the chosen placement has
            # committed. Every answer is checked against the actual latest state.
            if decision_future is not None and decision_id == current_id and decision_future.done():
                try:
                    answer = decision_future.result()
                except Exception:
                    answer = None
                choice = _select_jev(answer, plans)
                decision_future = None
                if choice is not None and not hold_sent:
                    selected = choice
                    try:
                        ctx.log("jev_plan_selected", piece=current_id,
                                candidate=f"P{plans.index(choice)}", policy=self.policy)
                    except Exception:
                        pass

            target = selected or fallback
            # A hold failure or an unexpected active piece invalidates only the
            # plan, never the episode. Build a current-piece legal fallback.
            active = obs.get("active") or {}
            if target is not None and target.get("source") == "hold":
                if hold_sent and active.get("type") != target.get("piece"):
                    target = None
            elif target is not None and active.get("type") != target.get("piece"):
                target = None
            if target is None:
                plans = _rank_single(obs, self.policy)
                target = plans[0] if plans else None
                fallback = target
                selected = target
                hold_sent = False

            action = self._action_for(obs, target, hold_sent, spec)
            before_pose = _pose(obs)
            if action == "hold":
                hold_sent = True
            last_command = action
            last_before = before_pose
            if action == "hard_drop" and target is not None:
                # A target must match the observed pose before the irreversible
                # drop; otherwise use a fresh current-orientation placement.
                active = obs.get("active") or {}
                try:
                    matched = (active.get("type") == target.get("piece") and
                               int(active.get("orientation", 0)) % 4 == int(target["orientation"]) % 4 and
                               int(active.get("x", 3)) == int(target["x"]))
                except (TypeError, ValueError):
                    matched = False
                if not matched:
                    fresh = _rank_single(obs, self.policy)
                    target = fresh[0] if fresh else None
                    selected = fallback = target
                    action = self._action_for(obs, target, hold_sent, spec)
                    last_command = action
                    last_before = before_pose
            if not self._tick(game, action):
                return


def build(ctx):
    return Harness(ctx)
