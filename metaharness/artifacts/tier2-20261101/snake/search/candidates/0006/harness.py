"""Candidate 0006: a bounded portfolio controller for real-time Snake.

The important latency rule in this version is architectural: the tick path never
runs a body-state search.  It makes a few board-sized static routes, validates
them with the exact transition, and otherwise chooses one of a few O(board)
safety moves.  Jev judges route *families* asynchronously; it never supplies a
one-cell action.  Astra is reserved for a genuinely unusual late-game state.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from typing import Any

from metaharness.models.jev import Choice


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
VEC = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _cell(value: Any) -> tuple[int, int] | None:
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _body(obs: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(obs, dict) or not isinstance(obs.get("snake"), (list, tuple)):
        return ()
    result: list[tuple[int, int]] = []
    for value in obs["snake"]:
        point = _cell(value)
        if point is None:
            return ()
        result.append(point)
    return tuple(result)


def _heading(obs: Any) -> str:
    value = obs.get("direction") if isinstance(obs, dict) else None
    return value if value in DIRECTIONS else "RIGHT"


def _food(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict) or obs.get("food") is None:
        return None
    return _cell(obs.get("food"))


def _dimensions(obs: Any) -> tuple[int, int]:
    if not isinstance(obs, dict):
        return 12, 12
    try:
        rows, cols = int(obs.get("rows", 12)), int(obs.get("cols", 12))
        if rows >= 2 and cols >= 2:
            return rows, cols
    except (TypeError, ValueError, OverflowError):
        pass
    return 12, 12


def _inside(point: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= point[0] < rows and 0 <= point[1] < cols


def _transition(
    body: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """The published move, including simultaneous non-food tail removal."""
    if not body or heading not in DIRECTIONS:
        return None
    new_heading = request if request in DIRECTIONS and request != OPPOSITE[heading] else heading
    dr, dc = VEC[new_heading]
    nxt = body[0][0] + dr, body[0][1] + dc
    if not _inside(nxt, rows, cols):
        return None
    ate = food is not None and nxt == food
    occupied = set(body if ate else body[:-1])
    if nxt in occupied:
        return None
    return (nxt,) + (body if ate else body[:-1]), new_heading, ate


def _moves(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None,
    rows: int, cols: int,
):
    """Yield distinct legal requests and exact successor states."""
    seen: set[str] = set()
    for action in DIRECTIONS:
        if action == OPPOSITE.get(heading):
            continue
        result = _transition(body, heading, action, food, rows, cols)
        if result is None:
            continue
        new_body, new_heading, ate = result
        if new_heading in seen:
            continue
        seen.add(new_heading)
        yield action, new_body, new_heading, ate


def _area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not body or not _inside(body[0], rows, cols):
        return 0
    blocked = set(body[1:])
    start = body[0]
    queue = deque([start])
    visited = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in VEC.values():
            nxt = r + dr, c + dc
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return len(visited)


def _tail_access(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    """A static filter only; it is not treated as an infinite-survival proof."""
    if len(body) <= 1:
        return True
    start, target = body[0], body[-1]
    blocked = set(body[1:-1])
    queue = deque([start])
    visited = {start}
    while queue:
        point = queue.popleft()
        if point == target:
            return True
        for dr, dc in VEC.values():
            nxt = point[0] + dr, point[1] + dc
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return False


def _tail_distance(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if len(body) <= 1:
        return 0
    target = body[-1]
    blocked = set(body[1:-1])
    queue = deque([(body[0], 0)])
    visited = {body[0]}
    while queue:
        point, distance = queue.popleft()
        if point == target:
            return distance
        for dr, dc in VEC.values():
            nxt = point[0] + dr, point[1] + dc
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, distance + 1))
    return rows * cols + 1


def _post_growth_safe(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    if len(body) >= rows * cols:
        return True
    # Keep the two mechanical filters deliberately cheap.  They are only used
    # after a complete finite route has already been simulated.
    return _tail_access(body, rows, cols) and _area(body, rows, cols) >= 2


def _static_route(
    body: tuple[tuple[int, int], ...], heading: str, target: tuple[int, int],
    rows: int, cols: int, max_nodes: int | None = None,
) -> list[str] | None:
    """Find a short cell route while conservatively blocking the body.

    This is intentionally a position/heading BFS, not the expensive search over
    every future body arrangement used by the previous candidate.  Exact dynamic
    simulation below remains the authority before a route is executed.
    """
    if not body or not _inside(target, rows, cols):
        return None
    start = body[0]
    if start == target:
        return []
    # The old tail may vacate on the first move.  Allowing it here is useful;
    # the exact simulation rejects any route whose timing makes it unsafe.
    blocked = set(body[:-1])
    blocked.discard(start)
    queue = deque([(start, heading, ())])
    visited = {(start, heading)}
    limit = max_nodes if max_nodes is not None else max(64, rows * cols * 4)
    expanded = 0
    while queue and expanded < limit:
        point, current_heading, path = queue.popleft()
        expanded += 1
        for action in DIRECTIONS:
            if action == OPPOSITE.get(current_heading):
                continue
            dr, dc = VEC[action]
            nxt = point[0] + dr, point[1] + dc
            if not _inside(nxt, rows, cols) or nxt in blocked:
                continue
            new_path = path + (action,)
            if nxt == target:
                return list(new_path)
            key = (nxt, action)
            if key in visited:
                continue
            visited.add(key)
            queue.append((nxt, action, new_path))
    return None


def _simulate_route(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None,
    route: list[str] | tuple[str, ...], rows: int, cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    current_body, current_heading, ate_any = body, heading, False
    for index, action in enumerate(route):
        result = _transition(current_body, current_heading, action, food, rows, cols)
        if result is None:
            return None
        current_body, current_heading, ate = result
        if ate:
            if index != len(route) - 1:
                return None
            ate_any = True
    return current_body, current_heading, ate_any


def _food_options(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None,
    rows: int, cols: int, time_limit: float,
) -> list[dict[str, Any]]:
    """Return a small portfolio of fully simulated, post-growth-safe routes."""
    if food is None or not body:
        return []
    options: list[dict[str, Any]] = []
    # Each legal first turn gets one independent static BFS.  The total work on
    # the reference board is bounded by a few hundred cells, and the deadline
    # check makes the emergency fallback win if a larger profile is supplied.
    for first, after_body, after_heading, first_ate in _moves(body, heading, food, rows, cols):
        if time.perf_counter() >= time_limit:
            break
        if first_ate:
            route = [first]
        else:
            suffix = _static_route(after_body, after_heading, food, rows, cols)
            if suffix is None:
                continue
            route = [first] + suffix
        simulated = _simulate_route(body, heading, food, route, rows, cols)
        if simulated is None or not simulated[2]:
            continue
        final_body, final_heading, _ = simulated
        if not _post_growth_safe(final_body, rows, cols):
            continue
        options.append({
            "kind": "food",
            "route": route,
            "first": route[0],
            "length": len(route),
            "area": _area(final_body, rows, cols),
            "tail": _tail_access(final_body, rows, cols),
            "tail_distance": _tail_distance(final_body, rows, cols),
            "final_heading": final_heading,
        })
    # The same first direction can arise through a duplicate heading only on a
    # malformed state.  Keep the best few route records, never a large queue.
    unique: dict[str, dict[str, Any]] = {}
    for option in options:
        old = unique.get(option["first"])
        if old is None or (option["length"], -option["area"]) < (old["length"], -old["area"]):
            unique[option["first"]] = option
    return list(unique.values())


def _best_nonfood(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None,
    rows: int, cols: int,
) -> dict[str, Any] | None:
    """One-step tail/space fallback.  It never waits for a route search."""
    choices: list[tuple[tuple[int, int, int, int, int], dict[str, Any]]] = []
    for action, new_body, new_heading, ate in _moves(body, heading, food, rows, cols):
        if ate:
            # A safe food move is selected by _food_options; do not accidentally
            # grow while the fallback is buying time.
            continue
        access = 1 if _tail_access(new_body, rows, cols) else 0
        space = _area(new_body, rows, cols)
        tail_dist = _tail_distance(new_body, rows, cols)
        food_dist = 0 if food is None else abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
        straight = 1 if new_heading == heading else 0
        facts = {
            "kind": "tail",
            "route": [action],
            "first": action,
            "length": 1,
            "area": space,
            "tail": bool(access),
            "tail_distance": tail_dist,
            "final_heading": new_heading,
        }
        # Tail access and area dominate food distance.  Straight is only the
        # final tie-breaker, so a wall turn is selected as soon as needed.
        choices.append(((access, space, -tail_dist, -food_dist, straight), facts))
    if not choices:
        return None
    choices.sort(key=lambda item: item[0], reverse=True)
    return choices[0][1]


def _best_space(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None,
    rows: int, cols: int,
) -> dict[str, Any] | None:
    """The same legal frontier ranked for open area rather than tail access."""
    choices: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for action, new_body, new_heading, ate in _moves(body, heading, food, rows, cols):
        if ate:
            continue
        space = _area(new_body, rows, cols)
        access = 1 if _tail_access(new_body, rows, cols) else 0
        tail_dist = _tail_distance(new_body, rows, cols)
        straight = 1 if new_heading == heading else 0
        facts = {
            "kind": "space",
            "route": [action],
            "first": action,
            "length": 1,
            "area": space,
            "tail": bool(access),
            "tail_distance": tail_dist,
            "final_heading": new_heading,
        }
        choices.append(((space, access, -tail_dist, straight), facts))
    if not choices:
        return None
    choices.sort(key=lambda item: item[0], reverse=True)
    return choices[0][1]


def _even_cycle(rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Make and validate a simple cycle for the even-by-even reference board.

    The top row is held out as the return edge; the remaining columns are
    serpentine.  Odd profiles simply use the finite-search fallback below.
    """
    if rows < 2 or cols < 2 or rows % 2 or cols % 2:
        return None
    cycle: list[tuple[int, int]] = [(0, 0)]
    cycle.extend((r, 0) for r in range(1, rows))
    for col in range(1, cols):
        if col % 2:
            cycle.append((rows - 1, col))
            cycle.extend((r, col) for r in range(rows - 2, 0, -1))
        else:
            cycle.append((1, col))
            cycle.extend((r, col) for r in range(2, rows))
    cycle.extend((0, col) for col in range(cols - 1, 0, -1))
    expected = rows * cols
    if len(cycle) != expected or len(set(cycle)) != expected:
        return None
    for index, point in enumerate(cycle):
        nxt = cycle[(index + 1) % expected]
        if abs(point[0] - nxt[0]) + abs(point[1] - nxt[1]) != 1:
            return None
    return cycle


def _cycle_step(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None,
    rows: int, cols: int, cycle: list[tuple[int, int]] | None,
) -> str | None:
    if cycle is None or len(body) < 2:
        return None
    indices = {point: i for i, point in enumerate(cycle)}
    if any(point not in indices for point in body):
        return None
    size = len(cycle)
    for orientation in (1, -1):
        total_gap = 0
        ordered = True
        for index in range(1, len(body)):
            gap = ((indices[body[index - 1]] - indices[body[index]]) * orientation) % size
            if gap <= 0:
                ordered = False
                break
            total_gap += gap
        if not ordered or total_gap >= size:
            continue
        next_point = cycle[(indices[body[0]] + orientation) % size]
        dr = next_point[0] - body[0][0]
        dc = next_point[1] - body[0][1]
        request = next((name for name, vector in VEC.items() if vector == (dr, dc)), None)
        if request is None or request == OPPOSITE.get(heading):
            continue
        if _transition(body, heading, request, food, rows, cols) is not None:
            return request
    return None


def _bucket_area(value: int, rows: int, cols: int) -> str:
    total = max(1, rows * cols)
    if value * 3 >= total:
        return "large"
    if value * 8 >= total:
        return "open"
    return "tight"


def _bucket_length(value: int, rows: int, cols: int) -> str:
    total = max(1, rows * cols)
    if value * 2 >= total:
        return "high_occupancy"
    if value * 5 >= total:
        return "medium_occupancy"
    return "low_occupancy"


class SnakeController:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.rows, self.cols = 12, 12
        self.cycle: list[tuple[int, int]] | None = None
        self.plan: list[str] = []
        self.plan_food: tuple[int, int] | None = None
        self.expected: tuple[tuple[tuple[int, int], ...], str, tuple[int, int] | None, bool] | None = None
        self.mode_hint = "FOOD_FAST"
        self.mode_until = -1
        self.jev_future = None
        self.jev_meta: dict[str, Any] | None = None
        self.last_jev_tick = -10000
        self.astra_future = None
        self.astra_mode: str | None = None
        self.last_score: int | None = None
        self.no_food_streak = 0
        self.last_log_tick = -10000

    def _set_dimensions(self, obs: dict[str, Any]) -> None:
        dimensions = _dimensions(obs)
        if dimensions != (self.rows, self.cols):
            self.rows, self.cols = dimensions
            self.cycle = _even_cycle(self.rows, self.cols)

    def _clear_plan(self) -> None:
        self.plan = []
        self.plan_food = None

    def _portfolio(
        self, obs: dict[str, Any], body: tuple[tuple[int, int], ...], heading: str,
        food: tuple[int, int] | None,
    ) -> dict[str, dict[str, Any]]:
        # A hard local budget is intentionally much smaller than a game tick.
        # It protects the real-time loop even if a non-reference profile grows.
        deadline = time.perf_counter() + 0.008
        foods = _food_options(body, heading, food, self.rows, self.cols, deadline)
        portfolio: dict[str, dict[str, Any]] = {}
        if foods:
            fast = min(foods, key=lambda x: (x["length"], -x["area"], x["tail_distance"]))
            wide = max(foods, key=lambda x: (x["area"], x["tail"], -x["length"], -x["tail_distance"]))
            portfolio["FOOD_FAST"] = fast
            portfolio["FOOD_WIDE"] = wide
        cycle_action = _cycle_step(body, heading, food, self.rows, self.cols, self.cycle)
        if cycle_action is not None:
            portfolio["CYCLE"] = {
                "kind": "cycle", "route": [cycle_action], "first": cycle_action,
                "length": 1, "area": self.rows * self.cols - len(body),
                "tail": True, "tail_distance": 0, "final_heading": cycle_action,
            }
        fallback = _best_nonfood(body, heading, food, self.rows, self.cols)
        if fallback is not None:
            portfolio["TAIL"] = fallback
        space = _best_space(body, heading, food, self.rows, self.cols)
        if space is not None:
            portfolio["SPACE"] = space
        return portfolio

    def _current_mode(self, portfolio: dict[str, dict[str, Any]], body: tuple[tuple[int, int], ...]) -> str:
        if self.mode_until >= 0 and self._tick_value < self.mode_until and self.mode_hint in portfolio:
            chosen = self.mode_hint
        else:
            chosen = "FOOD_FAST" if "FOOD_FAST" in portfolio else "TAIL"
        # A model can request a cautious mode, but cannot veto a safe food route
        # at low occupancy: score is the objective and code owns the invariant.
        if chosen in ("TAIL", "CYCLE") and "FOOD_FAST" in portfolio:
            if len(body) * 3 < self.rows * self.cols and self.astra_mode not in ("TAIL", "CYCLE"):
                chosen = "FOOD_FAST"
        if chosen not in portfolio:
            chosen = "FOOD_FAST" if "FOOD_FAST" in portfolio else ("CYCLE" if "CYCLE" in portfolio else "TAIL")
        return chosen

    def _start_jev(self, obs: dict[str, Any], body: tuple[tuple[int, int], ...], portfolio: dict[str, dict[str, Any]]) -> None:
        if self.jev_future is not None or getattr(self.ctx, "system1", None) is None:
            return
        try:
            tick = int(obs.get("tick", 0))
            score = int(obs.get("score", 0))
        except (TypeError, ValueError, OverflowError):
            return
        significant = (
            tick == 0 or score != self.last_score or not self.plan or
            tick - self.last_jev_tick >= 14
        )
        if not significant or tick - self.last_jev_tick < 6:
            return
        labels = list(portfolio)
        if len(labels) < 2:
            return
        descriptions: dict[str, str] = {}
        for label in labels:
            option = portfolio[label]
            if label == "FOOD_FAST":
                descriptions[label] = (
                    "Use the shortest code-simulated route to the current food. "
                    "The final growth state passed exact transition checks, static tail access, "
                    "and a nontrivial connected-space check."
                )
            elif label == "FOOD_WIDE":
                descriptions[label] = (
                    "Use the code-simulated food route leaving the largest post-growth region. "
                    "It is preferable when a slightly longer safe route buys body clearance."
                )
            elif label == "CYCLE":
                descriptions[label] = (
                    "Follow one step of a validated Hamiltonian cycle. The complete body is in "
                    "cycle order, so this is the conservative non-search fallback."
                )
            elif label == "TAIL":
                descriptions[label] = (
                    "Take the code-selected non-food step with the best moving-tail access and "
                    "connected free area; use it to unwind instead of growing."
                )
            else:
                descriptions[label] = "Take the code-selected legal space-preserving step."
        state = {
            "task": "Choose a route family, not a cell action. Code rechecks every transition.",
            "rules": {
                "absolute_actions": True,
                "reverse_requests_ignored": True,
                "tail_vacates_on_nonfood": True,
                "tail_stays_on_food": True,
                "only_listed_modes": True,
            },
            "observation": {
                "tick": tick,
                "head": list(body[0]) if body else None,
                "tail": list(body[-1]) if body else None,
                "direction": _heading(obs),
                "food": list(_food(obs)) if _food(obs) is not None else None,
                "occupancy": _bucket_length(len(body), self.rows, self.cols),
                "free_space": _bucket_area(_area(body, self.rows, self.cols), self.rows, self.cols),
                "score": score,
            },
            "route_facts": {
                label: {
                    "food_route": label.startswith("FOOD"),
                    "post_growth_tail_access": bool(option.get("tail", False)),
                    "space": _bucket_area(int(option.get("area", 0)), self.rows, self.cols),
                    "horizon": "one_step" if int(option.get("length", 1)) <= 1 else "short_route",
                }
                for label, option in portfolio.items()
            },
            "available_route_families": labels,
        }
        criteria_a = {label: descriptions[label] for label in labels}
        criteria_b = {label: descriptions[label] for label in reversed(labels)}
        questions = {
            "route_a": Choice(
                "Choose the best available route family for score while preserving the stated safety constraints. "
                "Do not invent a label, count cells, or return a direction.",
                criteria_a,
            ),
            "route_b": Choice(
                "Choose the best available route family using the same criteria. Return exactly one listed label.",
                criteria_b,
            ),
        }

        def call_jev():
            return self.ctx.decide(state, questions, timeout_s=0.45)

        self.last_jev_tick = tick
        self.jev_meta = {"tick": tick, "score": score, "food": _food(obs), "labels": labels}
        try:
            self.jev_future = self.ctx.spawn(call_jev)
            self.ctx.log("jev_route_request", tick=tick, modes=labels)
        except Exception as exc:
            self.jev_future = None
            self.jev_meta = None
            self.ctx.log("jev_route_spawn_failed", tick=tick, error=type(exc).__name__)

    def _poll_jev(self, obs: dict[str, Any], portfolio: dict[str, dict[str, Any]]) -> None:
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        meta, self.jev_meta = self.jev_meta, None
        try:
            answers = future.result()
            first, second = answers.get("route_a"), answers.get("route_b")
            value_a, value_b = getattr(first, "value", None), getattr(second, "value", None)
            if value_a != value_b or value_a not in portfolio:
                self.ctx.log("jev_route_discarded", answer_a=value_a, answer_b=value_b)
                return
            probabilities = getattr(first, "probabilities", {})
            probability = float(probabilities.get(value_a, 0.0)) if isinstance(probabilities, dict) else 0.0
            if probability < 0.42:
                self.ctx.log("jev_route_uncertain", mode=value_a, probability=probability)
                return
            now_tick = int(obs.get("tick", 0))
            current_food = _food(obs)
            if not isinstance(meta, dict) or now_tick - int(meta.get("tick", now_tick)) > 4:
                self.ctx.log("jev_route_old", mode=value_a, tick=now_tick)
                return
            if current_food != meta.get("food") or int(obs.get("score", 0)) != int(meta.get("score", 0)):
                self.ctx.log("jev_route_invalidated", mode=value_a, tick=now_tick)
                return
            self.mode_hint = str(value_a)
            self.mode_until = now_tick + 18
            self.ctx.log("jev_route_advice", mode=self.mode_hint, probability=probability, tick=now_tick)
        except Exception as exc:
            self.ctx.log("jev_route_ignored", error=type(exc).__name__)

    def _start_astra(self, obs: dict[str, Any], body: tuple[tuple[int, int], ...], portfolio: dict[str, dict[str, Any]]) -> None:
        if self.astra_future is not None or getattr(self.ctx, "system2", None) is None:
            return
        try:
            tick = int(obs.get("tick", 0))
        except (TypeError, ValueError, OverflowError):
            return
        unusual = tick >= 120 and (
            self.no_food_streak >= 7 or (len(body) * 2 >= self.rows * self.cols and "CYCLE" not in portfolio)
        )
        if not unusual:
            return
        prompt = (
            "You are a slow supervisor for a deterministic real-time Snake controller. "
            "The controller owns collision legality, route simulation, tail-vacating, and all arithmetic. "
            "Choose only one policy token: FOOD_FAST, FOOD_WIDE, CYCLE, or TAIL. "
            "Use TAIL/CYCLE only when the compact state says a safe food route is unavailable or occupancy is high. "
            "Return a single token; no coordinates or prose.\n\n" +
            json.dumps({
                "tick": tick,
                "rows": self.rows,
                "cols": self.cols,
                "length": len(body),
                "food": obs.get("food"),
                "score": obs.get("score"),
                "no_food_streak": self.no_food_streak,
                "available_modes": list(portfolio),
                "head": list(body[0]) if body else None,
                "tail": list(body[-1]) if body else None,
            }, separators=(",", ":"))
        )

        def call_astra():
            return self.ctx.think(prompt, timeout_s=90.0)

        try:
            self.astra_future = self.ctx.spawn(call_astra)
            self.ctx.log("astra_late_supervisor_started", tick=tick)
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("astra_spawn_failed", error=type(exc).__name__)

    def _poll_astra(self, obs: dict[str, Any], portfolio: dict[str, dict[str, Any]]) -> None:
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = str(getattr(result, "text", result)).upper()
            match = re.search(r"\b(FOOD_FAST|FOOD_WIDE|CYCLE|TAIL)\b", text)
            if match and match.group(1) in portfolio:
                self.astra_mode = match.group(1)
                self.mode_hint = self.astra_mode
                self.mode_until = int(obs.get("tick", 0)) + 40
                self.ctx.log("astra_policy_applied", mode=self.astra_mode)
            else:
                self.ctx.log("astra_policy_ignored")
        except Exception as exc:
            self.ctx.log("astra_policy_failed", error=type(exc).__name__)

    def choose(self, obs: Any) -> str:
        if not isinstance(obs, dict):
            return "RIGHT"
        self._set_dimensions(obs)
        body, heading, food = _body(obs), _heading(obs), _food(obs)
        try:
            self._tick_value = int(obs.get("tick", 0))
        except (TypeError, ValueError, OverflowError):
            self._tick_value = 0
        if not body or any(not _inside(point, self.rows, self.cols) for point in body):
            return heading

        # A predicted one-tick state is a freshness check.  A skipped tick or a
        # stale observation invalidates the remaining route instead of trusting it.
        if self.expected is not None:
            expected_body, expected_heading, expected_food, ate = self.expected
            matches = body == expected_body and heading == expected_heading and (ate or food == expected_food)
            if not matches:
                self._clear_plan()
            self.expected = None
        if self.plan_food is not None and food != self.plan_food:
            self._clear_plan()

        portfolio = self._portfolio(obs, body, heading, food)
        self._poll_jev(obs, portfolio)
        self._poll_astra(obs, portfolio)
        score_now = int(obs.get("score", 0)) if isinstance(obs.get("score", 0), (int, float)) else 0
        if self.last_score is not None and score_now == self.last_score:
            self.no_food_streak += 1
        else:
            self.no_food_streak = 0
        self.last_score = score_now

        action: str | None = None
        if self.plan:
            # Validate the next planned edge against the latest exact body before
            # consuming it.  This is the cheap emergency shield.
            trial = _transition(body, heading, self.plan[0], food, self.rows, self.cols)
            if trial is not None:
                action = self.plan.pop(0)
            else:
                self._clear_plan()

        if action is None:
            mode = self._current_mode(portfolio, body)
            option = portfolio.get(mode)
            if option is None:
                option = portfolio.get("FOOD_FAST") or portfolio.get("CYCLE") or portfolio.get("TAIL")
            if option is not None:
                route = list(option.get("route", []))
                if route and _transition(body, heading, route[0], food, self.rows, self.cols) is not None:
                    action = route[0]
                    if len(route) > 1 and option.get("kind") == "food":
                        self.plan = route[1:]
                        self.plan_food = food

        # Starting a request happens after all facts are code-derived and is
        # nonblocking.  Its route-family answer will be applied only after a
        # freshness check; it cannot replace the current cell action.
        self._start_jev(obs, body, portfolio)
        self._start_astra(obs, body, portfolio)

        if action not in DIRECTIONS:
            # The observation's safety aid is a last resort, but transition is
            # checked locally so even a malformed aid cannot post an illegal turn.
            for candidate in obs.get("safe_directions_now", []) if isinstance(obs.get("safe_directions_now"), list) else []:
                if candidate in DIRECTIONS and _transition(body, heading, candidate, food, self.rows, self.cols) is not None:
                    action = candidate
                    break
        if action not in DIRECTIONS:
            legal = list(_moves(body, heading, food, self.rows, self.cols))
            action = legal[0][0] if legal else heading

        trial = _transition(body, heading, action, food, self.rows, self.cols)
        if trial is not None:
            next_body, next_heading, ate = trial
            self.expected = (next_body, next_heading, None if ate else food, ate)
        else:
            self.expected = None
            self._clear_plan()
        return action


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        if not getattr(game, "tick_s", None):
            return
        controller = SnakeController(self.ctx, game)
        obs = game.observe()
        while not self.ctx.done and not getattr(game, "done", False):
            action = controller.choose(obs)
            try:
                self.ctx.act(action)
            except (TimeoutError, RuntimeError):
                if self.ctx.done or getattr(game, "done", False):
                    return
                raise
            try:
                self.ctx.wait_for_tick()
            except Exception:
                if self.ctx.done or getattr(game, "done", False):
                    return
                raise
            if self.ctx.done or getattr(game, "done", False):
                return
            obs = game.observe()


def build(ctx):
    return Harness(ctx)
