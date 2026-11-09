"""Candidate 0009: cycle-invariant food routing with a non-blocking planner.

The controller has two layers.  A validated rectangular Hamiltonian cycle is
its safety contract and its immediate fallback.  Separately, a short-lived
worker searches the *full moving body* for a shortest route to the current
food, but only through states that preserve that cycle's head-to-tail order.
The worker can therefore improve travel time without weakening the invariant;
its result is accepted only when the live observation is an exact prefix of the
state from which it was planned.
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Any


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
VEC = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _body(obs: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(obs, dict) or not isinstance(obs.get("snake"), (list, tuple)):
        return ()
    out: list[tuple[int, int]] = []
    for raw in obs["snake"]:
        point = _cell(raw)
        if point is None:
            return ()
        out.append(point)
    return tuple(out)


def _food(obs: Any) -> tuple[int, int] | None:
    return _cell(obs.get("food")) if isinstance(obs, dict) and obs.get("food") is not None else None


def _dimensions(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict):
        return None
    try:
        rows, cols = int(obs.get("rows")), int(obs.get("cols"))
    except (TypeError, ValueError, OverflowError):
        return None
    return (rows, cols) if rows >= 2 and cols >= 2 else None


def _heading(obs: Any, body: tuple[tuple[int, int], ...]) -> str:
    value = obs.get("direction") if isinstance(obs, dict) else None
    if value in VEC:
        return value
    if len(body) >= 2:
        delta = (body[0][0] - body[1][0], body[0][1] - body[1][1])
        for name, vector in VEC.items():
            if delta == vector:
                return name
    return "RIGHT"


def _inside(point: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= point[0] < rows and 0 <= point[1] < cols


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _valid_cycle(cycle: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    size = rows * cols
    return (
        len(cycle) == size
        and len(set(cycle)) == size
        and all(_inside(point, rows, cols) for point in cycle)
        and all(_adjacent(cycle[i], cycle[(i + 1) % size]) for i in range(size))
    )


def _row_cycle(rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """A cycle that reserves column zero as the closing spine."""
    if rows < 2 or cols < 2 or rows % 2:
        return None
    path: list[tuple[int, int]] = [(0, c) for c in range(cols)]
    for r in range(1, rows):
        columns = range(cols - 1, 0, -1) if r % 2 else range(1, cols)
        path.extend((r, c) for c in columns)
    path.append((rows - 1, 0))
    path.extend((r, 0) for r in range(rows - 2, 0, -1))
    result = tuple(path)
    return result if _valid_cycle(result, rows, cols) else None


def _cycles(rows: int, cols: int) -> list[tuple[tuple[int, int], ...]]:
    bases: list[tuple[tuple[int, int], ...]] = []
    direct = _row_cycle(rows, cols)
    if direct is not None:
        bases.append(direct)
    if cols % 2 == 0:
        transposed = _row_cycle(cols, rows)
        if transposed is not None:
            bases.append(tuple((c, r) for r, c in transposed))

    result: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in bases:
        for flip_r in (False, True):
            for flip_c in (False, True):
                reflected = tuple(
                    (rows - 1 - r if flip_r else r,
                     cols - 1 - c if flip_c else c)
                    for r, c in base
                )
                for candidate in (reflected, tuple(reversed(reflected))):
                    if candidate not in seen and _valid_cycle(candidate, rows, cols):
                        seen.add(candidate)
                        result.append(candidate)
    return result


def _add_edge(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> None:
    graph.setdefault(a, set()).add(b)
    graph.setdefault(b, set()).add(a)


def _remove_edge(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> None:
    if a in graph:
        graph[a].discard(b)
    if b in graph:
        graph[b].discard(a)


def _walk_cycle(
    graph: dict[tuple[int, int], set[tuple[int, int]]], rows: int, cols: int
) -> tuple[tuple[int, int], ...] | None:
    if len(graph) != rows * cols or any(len(neighbors) != 2 for neighbors in graph.values()):
        return None
    start = (0, 0)
    if start not in graph:
        return None
    path = [start]
    previous: tuple[int, int] | None = None
    current = start
    while True:
        options = sorted(graph[current])
        if previous is not None:
            options = [cell for cell in options if cell != previous]
        if not options:
            return None
        nxt = options[0]
        if nxt == start:
            break
        if nxt in path:
            return None
        path.append(nxt)
        previous, current = current, nxt
        if len(path) > rows * cols:
            return None
    candidate = tuple(path)
    return candidate if _valid_cycle(candidate, rows, cols) else None


def _strip_cycle(
    rows: int, cols: int, seams: tuple[int, ...]
) -> tuple[tuple[int, int], ...] | None:
    """Join verified two-row rings at seams, then validate the full cycle."""
    if rows < 2 or cols < 2 or rows % 2 or len(seams) != rows // 2 - 1:
        return None
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for row in range(0, rows, 2):
        perimeter = (
            [(row, col) for col in range(cols)]
            + [(row + 1, col) for col in range(cols - 1, -1, -1)]
        )
        for i, cell in enumerate(perimeter):
            _add_edge(graph, cell, perimeter[(i + 1) % len(perimeter)])
    for strip, seam in enumerate(seams):
        if seam < 0 or seam >= cols - 1:
            return None
        upper, lower = 2 * strip + 1, 2 * strip + 2
        _remove_edge(graph, (upper, seam), (upper, seam + 1))
        _remove_edge(graph, (lower, seam), (lower, seam + 1))
        _add_edge(graph, (upper, seam), (lower, seam))
        _add_edge(graph, (upper, seam + 1), (lower, seam + 1))
    return _walk_cycle(graph, rows, cols)


def _pattern_bank(strip_count: int, width: int, count: int, seed: int) -> list[tuple[int, ...]]:
    if strip_count <= 0 or width <= 0:
        return [()]
    result: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(values: list[int] | tuple[int, ...]) -> None:
        pattern = tuple(int(value) % width for value in values)
        if pattern not in seen:
            seen.add(pattern)
            result.append(pattern)

    for value in (0, width - 1, width // 2):
        add([value] * strip_count)
    for phase in range(width):
        add([(phase + i) % width for i in range(strip_count)])
        add([(phase - i) % width for i in range(strip_count)])
    for i in range(strip_count):
        values = [0] * strip_count
        values[i] = (2 + 3 * i) % width
        add(values)
    state = (seed | 1) & 0x7FFFFFFF
    while len(result) < count:
        values = []
        for _ in range(strip_count):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            values.append(state % width)
        add(values)
    return result[:count]


def _cycle_family(rows: int, cols: int, pattern_count: int = 20,
                  limit: int = 192) -> list[tuple[tuple[int, int], ...]]:
    """Build a bounded, independently checked family for endpoint proofs."""
    if rows < 2 or cols < 2 or (rows % 2 and cols % 2):
        return []
    bases: list[tuple[tuple[int, int], ...]] = []
    direct = _row_cycle(rows, cols)
    if direct is not None:
        bases.append(direct)
    if cols % 2 == 0:
        transposed = _row_cycle(cols, rows)
        if transposed is not None:
            bases.append(tuple((c, r) for r, c in transposed))
    direct_patterns = _pattern_bank(rows // 2 - 1, cols - 1, pattern_count, 1729)
    transposed_patterns = _pattern_bank(cols // 2 - 1, rows - 1, pattern_count, 2718)
    for direct_pattern, transposed_pattern in zip(direct_patterns, transposed_patterns):
        candidate = _strip_cycle(rows, cols, direct_pattern)
        if candidate is not None:
            bases.append(candidate)
        if cols % 2 == 0:
            candidate = _strip_cycle(cols, rows, transposed_pattern)
            if candidate is not None:
                bases.append(tuple((c, r) for r, c in candidate))
    result: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in bases:
        for flip_r in (False, True):
            for flip_c in (False, True):
                reflected = tuple(
                    (rows - 1 - r if flip_r else r,
                     cols - 1 - c if flip_c else c)
                    for r, c in base
                )
                for oriented in (reflected, tuple(reversed(reflected))):
                    if oriented in seen or not _valid_cycle(oriented, rows, cols):
                        continue
                    seen.add(oriented)
                    result.append(oriented)
                    if len(result) >= limit:
                        return result
    return result


def _between(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    delta = b[0] - a[0], b[1] - a[1]
    for name, vector in VEC.items():
        if delta == vector:
            return name
    return None


def _ordered(body: tuple[tuple[int, int], ...], cycle: tuple[tuple[int, int], ...]) -> bool:
    """Return whether body cells occur in strict head-to-tail cycle order."""
    if not body or not cycle or len(set(body)) != len(body):
        return False
    positions = {point: i for i, point in enumerate(cycle)}
    if any(point not in positions for point in body):
        return False
    n = len(cycle)
    span = 0
    for previous, older in zip(body, body[1:]):
        gap = (positions[previous] - positions[older]) % n
        if gap <= 0:
            return False
        span += gap
    # This prevents the occupied body from wrapping around the tour.
    return span < n


def _transition(
    body: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """Apply exactly one public Snake transition, including tail semantics."""
    if not body or heading not in VEC:
        return None
    new_heading = heading if request == OPPOSITE.get(heading) else request
    if new_heading not in VEC:
        new_heading = heading
    dr, dc = VEC[new_heading]
    next_head = body[0][0] + dr, body[0][1] + dc
    if not _inside(next_head, rows, cols):
        return None
    ate = food is not None and next_head == food
    occupied = set(body if ate else body[:-1])
    if next_head in occupied:
        return None
    next_body = (next_head,) + (body if ate else body[:-1])
    return next_body, new_heading, ate


def _area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not body:
        return 0
    blocked = set(body[1:])
    q = deque([body[0]])
    seen = {body[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in VEC.values():
            nxt = r + dr, c + dc
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen)


def _repeatable(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    action: str,
    rows: int,
    cols: int,
    horizon: int = 4,
) -> bool:
    """Small emergency shield used only if cycle entry is unavailable."""
    current_body, current_heading, current_food = body, heading, food
    for _ in range(horizon):
        result = _transition(current_body, current_heading, action, current_food, rows, cols)
        if result is None:
            return False
        current_body, current_heading, ate = result
        if ate:
            # The new food is unknown, so this is no longer a valid repeated
            # action proof.  The already executed food move itself is valid.
            return True
    return True


def _fallback(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    advertised: tuple[str, ...],
) -> str:
    candidates: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action not in advertised or action == OPPOSITE.get(heading):
            continue
        result = _transition(body, heading, action, food, rows, cols)
        if result is None:
            continue
        new_body, new_heading, ate = result
        if ate and len(new_body) < rows * cols and _area(new_body, rows, cols) <= 1:
            continue
        rank = (
            int(ate),
            int(_repeatable(body, heading, food, action, rows, cols)),
            _area(new_body, rows, cols),
            int(new_heading == heading),
        )
        candidates.append(((rank[0], rank[1], rank[2], rank[3], action), action))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    if heading in advertised:
        return heading
    return advertised[0] if advertised else "NOOP"


def _cycle_action(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
) -> str | None:
    """Choose a single cycle-order-preserving neighbor.

    The first two rank terms reproduce the useful part of 0008: eat now, then
    reduce forward cycle distance to food.  The remaining terms favour the
    actual edge and heading continuity.  Every candidate is independently
    simulated before it can be returned.
    """
    if not _ordered(body, cycle):
        return None
    position = {point: i for i, point in enumerate(cycle)}
    n = len(cycle)
    next_on_tour = cycle[(position[body[0]] + 1) % n]
    options: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action == OPPOSITE.get(heading):
            continue
        result = _transition(body, heading, action, food, rows, cols)
        if result is None:
            continue
        new_body, new_heading, ate = result
        if not _ordered(new_body, cycle):
            continue
        if ate:
            forward = 0
        elif food is None or food not in position:
            forward = 0
        else:
            forward = (position[food] - position[new_body[0]]) % n
        rank = (
            int(ate),
            -forward,
            int(new_body[0] == next_on_tour),
            int(new_heading == heading),
            action,
        )
        options.append((rank, action))
    if not options:
        return None
    options.sort(reverse=True)
    return options[0][1]


def _greedy_food_path(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    first_action: str | None,
) -> tuple[str, ...] | None:
    """Reconstruct 0008's current greedy route as a local no-regression bound."""
    current_body, current_heading = body, heading
    path: list[str] = []
    limit = len(cycle) + len(body) + 4
    for _ in range(limit):
        action = _cycle_action(current_body, current_heading, food, rows, cols, cycle)
        if not path and first_action in DIRECTIONS:
            action = first_action
        if action is None:
            return None
        step = _transition(current_body, current_heading, action, food, rows, cols)
        if step is None or not _ordered(step[0], cycle):
            return None
        path.append(action)
        current_body, current_heading, ate = step
        if ate:
            return tuple(path)
    return None


def _food_search(
    origin_body: tuple[tuple[int, int], ...],
    origin_heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    first_action: str | None = None,
):
    """Bounded A* for a food route inside the cycle invariant.

    This is deliberately pure: it never touches the game or context, and its
    monotonic deadline means an episode cannot strand a worker.  The result is
    only an advisory route; the foreground rechecks every transition.
    """
    if not origin_body or not _ordered(origin_body, cycle):
        return None
    if origin_body[0] == food:
        return None
    positions = {point: i for i, point in enumerate(cycle)}
    n = len(cycle)
    deadline = time.monotonic() + 0.045
    baseline = _greedy_food_path(origin_body, origin_heading, food, rows, cols, cycle, first_action)
    serial = 0
    start_key = (origin_body, origin_heading)
    start_h = abs(origin_body[0][0] - food[0]) + abs(origin_body[0][1] - food[1])
    heap: list[tuple[float, int, int, int, tuple[tuple[int, int], ...], str, tuple[str, ...]]] = [
        (float(start_h), 0, 0, serial, origin_body, origin_heading, ())
    ]
    best: dict[tuple[tuple[tuple[int, int], ...], str], tuple[int, int]] = {
        start_key: (0, 0)
    }
    expansions = 0
    max_depth = min(rows * cols + len(origin_body), 220)

    while heap and expansions < 7000 and time.monotonic() < deadline:
        _, depth, turns, _, body, heading, path = heapq.heappop(heap)
        expansions += 1
        if depth >= max_depth:
            continue
        actions: list[tuple[int, int, str]] = []
        action_source = (first_action,) if depth == 0 and first_action in DIRECTIONS else DIRECTIONS
        for action in action_source:
            if action == OPPOSITE.get(heading):
                continue
            dr, dc = VEC[action]
            nr, nc = body[0][0] + dr, body[0][1] + dc
            distance = abs(nr - food[0]) + abs(nc - food[1])
            actions.append((distance, int(action != heading), action))
        actions.sort()
        for _, _, action in actions:
            result = _transition(body, heading, action, food, rows, cols)
            if result is None:
                continue
            new_body, new_heading, ate = result
            if not _ordered(new_body, cycle):
                continue
            new_path = path + (action,)
            if ate:
                return {
                    "origin_body": origin_body,
                    "origin_heading": origin_heading,
                    "origin_food": food,
                    "route": new_path,
                    "baseline_len": len(baseline) if baseline is not None else None,
                    "expansions": expansions,
                }
            new_depth = depth + 1
            new_turns = turns + int(new_heading != heading)
            key = (new_body, new_heading)
            old = best.get(key)
            if old is not None and old <= (new_depth, new_turns):
                continue
            best[key] = (new_depth, new_turns)
            serial += 1
            cycle_distance = (positions[food] - positions[new_body[0]]) % n
            heuristic = abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
            # Manhattan remains the main A* key; the small cycle term breaks
            # ties toward states that can reach the target sooner on the tour.
            priority = new_depth + heuristic + 0.002 * cycle_distance
            heapq.heappush(
                heap,
                (priority, new_depth, new_turns, serial, new_body, new_heading, new_path),
            )
    return None


def _food_search_reentry(
    origin_body: tuple[tuple[int, int], ...],
    origin_heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    active_cycle: tuple[tuple[int, int], ...],
    cycles: tuple[tuple[tuple[int, int], ...], ...],
):
    """Search a finite food route that can re-enter a validated cycle.

    This is deliberately used only by the late-game controller gate.  Unlike
    ``_food_search``, intermediate states need not belong to the current tour,
    but every edge is still the exact public transition.  The only goal that
    can be returned is a food transition whose grown body is ordered on a
    validated tour.  The route must strictly beat the current tour's greedy
    route, so this cannot silently replace the parent policy with an equal or
    slower rescue.
    """
    if not origin_body or not food or not active_cycle:
        return None
    if not _ordered(origin_body, active_cycle) or origin_body[0] == food:
        return None
    baseline = _greedy_food_path(
        origin_body, origin_heading, food, rows, cols, active_cycle, None
    )
    baseline_len = len(baseline) if baseline is not None else None
    if baseline_len is None or baseline_len <= 1:
        return None

    deadline = time.monotonic() + 0.045
    serial = 0
    start_key = (origin_body, origin_heading)
    start_h = abs(origin_body[0][0] - food[0]) + abs(origin_body[0][1] - food[1])
    heap: list[
        tuple[
            float, int, int, int,
            tuple[tuple[int, int], ...], str, tuple[str, ...]
        ]
    ] = [(float(start_h), 0, 0, serial, origin_body, origin_heading, ())]
    best: dict[tuple[tuple[tuple[int, int], ...], str], tuple[int, int]] = {
        start_key: (0, 0)
    }
    expanded = 0
    max_depth = min(rows * cols + len(origin_body), baseline_len - 1, 220)

    while heap and expanded < 12000 and time.monotonic() < deadline:
        _, depth, turns, _, body, heading, path = heapq.heappop(heap)
        expanded += 1
        if depth >= max_depth:
            continue
        ranked: list[tuple[int, int, str]] = []
        for action in DIRECTIONS:
            if action == OPPOSITE.get(heading):
                continue
            dr, dc = VEC[action]
            nr, nc = body[0][0] + dr, body[0][1] + dc
            ranked.append((abs(nr - food[0]) + abs(nc - food[1]),
                           int(action != heading), action))
        ranked.sort()
        for _, _, action in ranked:
            step = _transition(body, heading, action, food, rows, cols)
            if step is None:
                continue
            new_body, new_heading, ate = step
            new_path = path + (action,)
            new_depth = depth + 1
            if ate:
                if new_depth >= baseline_len:
                    continue
                endpoint = next(
                    (cycle for cycle in cycles if _ordered(new_body, cycle)),
                    None,
                )
                if endpoint is not None:
                    return {
                        "origin_body": origin_body,
                        "origin_heading": origin_heading,
                        "origin_food": food,
                        "route": new_path,
                        "baseline_len": baseline_len,
                        "endpoint_cycle": endpoint,
                        "reentry": True,
                        "expansions": expanded,
                    }
                continue

            remaining = abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
            if new_depth + remaining >= baseline_len:
                continue
            new_turns = turns + int(new_heading != heading)
            key = (new_body, new_heading)
            old = best.get(key)
            if old is not None and old <= (new_depth, new_turns):
                continue
            best[key] = (new_depth, new_turns)
            serial += 1
            heapq.heappush(
                heap,
                (new_depth + remaining + 0.002 * new_turns,
                 new_depth, new_turns, serial, new_body, new_heading, new_path),
            )
    return None


def _prefix_suffix(
    result: dict[str, Any],
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[str, ...] | None:
    """Return the unused suffix only if current state is an exact route prefix."""
    if not isinstance(result, dict) or result.get("origin_food") != food:
        return None
    origin_body = result.get("origin_body")
    origin_heading = result.get("origin_heading")
    route = result.get("route")
    if not isinstance(origin_body, tuple) or not isinstance(route, tuple) or not route:
        return None
    current_body, current_heading, current_food = origin_body, origin_heading, food
    for i, action in enumerate(route):
        if current_body == body and current_heading == heading and current_food == food:
            return route[i:]
        step = _transition(current_body, current_heading, action, current_food, rows, cols)
        if step is None:
            return None
        current_body, current_heading, ate = step
        if ate:
            # A plan ends on its food transition.  It must not be matched to a
            # later state whose newly sampled food is unknown here.
            return None
    if current_body == body and current_heading == heading and current_food == food:
        return ()
    return None


class _Controller:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.rows = 0
        self.cols = 0
        self.available_cycles: list[tuple[tuple[int, int], ...]] = []
        self.cycle: tuple[tuple[int, int], ...] | None = None
        self.plan: list[str] = []
        self.plan_food: tuple[int, int] | None = None
        self.plan_end_cycle: tuple[tuple[int, int], ...] | None = None
        self.plan_reentry = False
        self.expected: tuple[tuple[tuple[int, int], ...], str, tuple[int, int] | None] | None = None
        self.future: Any = None
        self.future_mode: str | None = None
        self.last_started_tick = -100000
        self.last_started_food: tuple[int, int] | None = None
        self.food_seen: tuple[int, int] | None = None
        self.reentry_ready = False
        self.last_reentry_tick = -100000

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _set_dimensions(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        family = _cycle_family(rows, cols, pattern_count=20, limit=192)
        self.available_cycles = family or _cycles(rows, cols)
        self.cycle = None
        self.plan = []
        self.plan_food = None
        self.plan_end_cycle = None
        self.plan_reentry = False
        self.expected = None
        self.future = None
        self.future_mode = None
        self.food_seen = None
        self.reentry_ready = False
        self.last_reentry_tick = -100000

    def _select_cycle(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
    ) -> str | None:
        choices: list[tuple[tuple[int, int, int], str, tuple[tuple[int, int], ...]]] = []
        for number, cycle in enumerate(self.available_cycles):
            action = _cycle_action(body, heading, food, self.rows, self.cols, cycle)
            if action is None:
                continue
            choices.append(((int(action == heading), int(action != OPPOSITE.get(heading)), -number), action, cycle))
        if not choices:
            return None
        choices.sort(reverse=True)
        _, action, cycle = choices[0]
        self.cycle = cycle
        self._log("cycle_selected", tick=0, variants=len(self.available_cycles), action=action,
                   planner="cycle_order_astar_late_reentry")
        return action

    def _clear_plan(self) -> None:
        self.plan = []
        self.plan_food = None
        self.plan_end_cycle = None
        self.plan_reentry = False
        self.expected = None

    def _late_gate(self, obs: Any) -> bool:
        if not isinstance(obs, dict) or self.rows < 2 or self.cols < 2:
            return False
        try:
            free = int(obs.get("free_cells"))
        except (TypeError, ValueError, OverflowError):
            return False
        return 0 <= free <= (self.rows * self.cols) // 4

    def _poll_plan(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        tick: int,
        late: bool,
    ) -> None:
        future = self.future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        mode = self.future_mode
        started_food = self.last_started_food
        self.future = None
        self.future_mode = None
        if mode == "cycle" and late and started_food == food:
            # The parent search has had its chance for this exact target.
            # Re-entry is only armed after that result is returned.
            self.reentry_ready = True
        try:
            result = future.result()
        except Exception as exc:
            self._log("food_plan_discarded", tick=tick, reason=type(exc).__name__, mode=mode)
            return
        if isinstance(result, dict):
            route = result.get("route")
            baseline = result.get("baseline_len")
            if isinstance(route, tuple) and isinstance(baseline, int) and len(route) >= baseline:
                self._log("food_plan_discarded", tick=tick,
                          reason="not_shorter_than_cycle", mode=mode)
                return
            if result.get("reentry"):
                endpoint = result.get("endpoint_cycle")
                if not isinstance(endpoint, tuple) or endpoint not in self.available_cycles:
                    self._log("food_plan_discarded", tick=tick,
                              reason="invalid_reentry_endpoint", mode=mode)
                    return
        suffix = _prefix_suffix(result, body, heading, food, self.rows, self.cols) if result else None
        if suffix:
            self.plan = list(suffix)
            self.plan_food = food
            self.plan_end_cycle = (
                result.get("endpoint_cycle")
                if isinstance(result, dict) and result.get("reentry")
                else self.cycle
            )
            self.plan_reentry = bool(isinstance(result, dict) and result.get("reentry"))
            self.reentry_ready = False
            self._log("food_plan_ready", tick=tick, remaining=len(self.plan),
                       mode=mode,
                       expansions=result.get("expansions", 0) if isinstance(result, dict) else 0)
        elif result:
            self._log("food_plan_discarded", tick=tick, reason="state_changed", mode=mode)

    def _start_plan(
        self,
        next_body: tuple[tuple[int, int], ...],
        next_heading: str,
        food: tuple[int, int] | None,
        tick: int,
        late: bool,
    ) -> None:
        """Start the parent's search, or one guarded late re-entry search."""
        if self.cycle is None or food is None or self.plan or self.future is not None:
            return
        if tick - self.last_started_tick < 2 and food == self.last_started_food:
            return

        use_reentry = (
            late and self.reentry_ready
            and tick - self.last_reentry_tick >= 10
            and bool(self.available_cycles)
        )
        if use_reentry:
            snapshot = (
                next_body, next_heading, food, self.rows, self.cols,
                self.cycle, tuple(self.available_cycles),
            )
            worker = _food_search_reentry
            self.future_mode = "reentry"
            self.last_reentry_tick = tick
            self.reentry_ready = False
            event = "food_reentry_started"
        else:
            snapshot = (next_body, next_heading, food, self.rows, self.cols, self.cycle, None)
            worker = _food_search
            self.future_mode = "cycle"
            event = "food_plan_started"
        try:
            self.future = self.ctx.spawn(lambda: worker(*snapshot))
            self.last_started_tick = tick
            self.last_started_food = food
            self._log(event, tick=tick, food=list(food), origin_tick=tick + 1,
                      first_action="free", late=late)
        except Exception as exc:
            self.future = None
            self.future_mode = None
            self._log("food_plan_spawn_failed", tick=tick, reason=type(exc).__name__)

    def choose(self, obs: Any, advertised: tuple[str, ...]) -> str:
        dimensions = _dimensions(obs)
        body = _body(obs)
        if dimensions is None or not body:
            return advertised[0] if advertised else "NOOP"
        self._set_dimensions(*dimensions)
        heading = _heading(obs, body)
        food = _food(obs)
        try:
            tick = int(obs.get("tick", 0)) if isinstance(obs, dict) else 0
        except (TypeError, ValueError, OverflowError):
            tick = 0
        late = self._late_gate(obs)

        if food != self.food_seen:
            self.food_seen = food
            self.reentry_ready = False
            self.last_reentry_tick = -100000

        self._poll_plan(body, heading, food, tick, late)
        if self.plan_food != food:
            self._clear_plan()

        # A planned non-food action carries an exact expected next state.  A
        # missed or externally advanced tick invalidates the suffix rather than
        # letting a merely legal action from an old route steer the snake.
        if self.expected is not None:
            expected_body, expected_heading, expected_food = self.expected
            if body != expected_body or heading != expected_heading or food != expected_food:
                self._log("food_plan_discarded", tick=tick, reason="expected_state_mismatch")
                self._clear_plan()
            self.expected = None

        # During a finite re-entry route the body is intentionally allowed to
        # leave the old cycle.  Its endpoint is checked on the food move below.
        if self.cycle is not None and not (self.plan and self.plan_reentry):
            if not _ordered(body, self.cycle):
                self._log("cycle_invalidated", tick=tick)
                self.cycle = None
                self._clear_plan()

        action: str | None = None
        if self.cycle is None:
            action = self._select_cycle(body, heading, food)

        if self.cycle is not None and self.plan:
            candidate = self.plan[0]
            step = _transition(body, heading, candidate, food, self.rows, self.cols)
            valid = step is not None and candidate in advertised
            if valid and not self.plan_reentry:
                valid = _ordered(step[0], self.cycle)  # type: ignore[index]
            if valid and step is not None and step[2]:
                endpoint = self.plan_end_cycle if self.plan_reentry else self.cycle
                valid = endpoint is not None and _ordered(step[0], endpoint)
            if valid:
                action = candidate
                self.plan.pop(0)
                if step is not None and step[2]:
                    endpoint = self.plan_end_cycle if self.plan_reentry else self.cycle
                    if endpoint is not None:
                        self.cycle = endpoint
                    self._clear_plan()
                else:
                    self.expected = (step[0], step[1], food)  # type: ignore[index]
            else:
                self._log("food_plan_discarded", tick=tick, reason="live_transition_mismatch",
                          mode="reentry" if self.plan_reentry else "cycle")
                self._clear_plan()

        if action is None and self.cycle is not None:
            action = _cycle_action(body, heading, food, self.rows, self.cols, self.cycle)
            if action not in advertised:
                action = None

        if action is None:
            action = _fallback(body, heading, food, self.rows, self.cols, advertised)

        if action not in advertised:
            action = heading if heading in advertised else (advertised[0] if advertised else "NOOP")

        # A non-food transition is an exact, known origin for the next
        # decision.  Search from there without pinning the next turn to the
        # current greedy cycle action; the route is still accepted only after
        # exact prefix checks and (for re-entry) endpoint validation.
        now = _transition(body, heading, action, food, self.rows, self.cols)
        if (self.cycle is not None and now is not None and not now[2]
                and not self.plan and action in DIRECTIONS):
            self._start_plan(now[0], now[1], food, tick, late)
        return action


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

    def run_episode(self, game: Any) -> None:
        if not getattr(game, "tick_s", None):
            return
        spec = getattr(game, "action_spec", None)
        names = tuple(name for name in (getattr(spec, "names", ()) or ()) if isinstance(name, str))
        advertised = tuple(direction for direction in DIRECTIONS if direction in names)
        if not advertised:
            noop = getattr(spec, "noop", None)
            if isinstance(noop, str) and noop in names:
                advertised = (noop,)
            else:
                advertised = names
        if not advertised:
            return

        controller = _Controller(self.ctx)
        while not self.ctx.done and not getattr(game, "done", False):
            observation = game.observe()
            action = controller.choose(observation, advertised)
            try:
                self.ctx.act(action)
            except Exception:
                if self.ctx.done or getattr(game, "done", False):
                    return
                raise
            if self.ctx.done or getattr(game, "done", False):
                return
            try:
                self.ctx.wait_for_tick()
            except Exception:
                if self.ctx.done or getattr(game, "done", False):
                    return
                raise


def build(ctx: Any) -> Harness:
    return Harness(ctx)
