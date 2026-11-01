"""Candidate 0014: portfolio-safe routing with delayed semantic arbitration.

The clock-facing controller is deterministic.  It keeps a validated Hamiltonian
invariant and uses exact body transitions to make a small portfolio of food
routes.  Jev is not asked to navigate the grid: it arbitrates among routes that
code has already proved safe, and only when a route is long enough for the
answer to be useful.  Astra is an asynchronous supervisor for the route
objective; an absent or malformed recommendation leaves the conservative code
policy in force.
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Any

from metaharness.models.jev import Choice


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
VECTORS = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _point(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _read_body(obs: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(obs, dict) or not isinstance(obs.get("snake"), (list, tuple)):
        return ()
    body: list[tuple[int, int]] = []
    for value in obs["snake"]:
        cell = _point(value)
        if cell is None:
            return ()
        body.append(cell)
    return tuple(body)


def _read_food(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict) or obs.get("food") is None:
        return None
    return _point(obs.get("food"))


def _read_dimensions(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict):
        return None
    try:
        rows, cols = int(obs.get("rows")), int(obs.get("cols"))
    except (TypeError, ValueError, OverflowError):
        return None
    if rows < 2 or cols < 2 or rows * cols > 4096:
        return None
    return rows, cols


def _read_heading(obs: Any, body: tuple[tuple[int, int], ...]) -> str:
    if isinstance(obs, dict) and obs.get("direction") in VECTORS:
        return str(obs["direction"])
    if len(body) >= 2:
        delta = (body[0][0] - body[1][0], body[0][1] - body[1][1])
        for name, vector in VECTORS.items():
            if delta == vector:
                return name
    return "RIGHT"


def _read_tick(obs: Any) -> int:
    try:
        return int(obs.get("tick", 0)) if isinstance(obs, dict) else 0
    except (TypeError, ValueError, OverflowError):
        return 0


def _inside(cell: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols


def _touching(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _valid_tour(tour: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    size = rows * cols
    if len(tour) != size or len(set(tour)) != size:
        return False
    if any(not _inside(cell, rows, cols) for cell in tour):
        return False
    return all(_touching(tour[i], tour[(i + 1) % size]) for i in range(size))


def _row_spine(rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """Make the even-height cycle with a reserved first-column return path."""
    if rows < 2 or cols < 2 or rows % 2:
        return None
    path: list[tuple[int, int]] = [(0, col) for col in range(cols)]
    for row in range(1, rows):
        column_order = range(cols - 1, 0, -1) if row % 2 else range(1, cols)
        path.extend((row, col) for col in column_order)
    path.append((rows - 1, 0))
    path.extend((row, 0) for row in range(rows - 2, 0, -1))
    candidate = tuple(path)
    return candidate if _valid_tour(candidate, rows, cols) else None


def _edge_add(graph: dict[tuple[int, int], set[tuple[int, int]]],
              a: tuple[int, int], b: tuple[int, int]) -> None:
    graph.setdefault(a, set()).add(b)
    graph.setdefault(b, set()).add(a)


def _edge_drop(graph: dict[tuple[int, int], set[tuple[int, int]]],
               a: tuple[int, int], b: tuple[int, int]) -> None:
    graph.get(a, set()).discard(b)
    graph.get(b, set()).discard(a)


def _walk_cycle(graph: dict[tuple[int, int], set[tuple[int, int]]],
                rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """Accept a seam construction only if it is one complete 2-regular cycle."""
    total = rows * cols
    if len(graph) != total or any(len(edges) != 2 for edges in graph.values()):
        return None
    start = (0, 0)
    if start not in graph:
        return None
    result = [start]
    previous: tuple[int, int] | None = None
    current = start
    while True:
        choices = sorted(graph[current])
        if previous is not None:
            choices = [cell for cell in choices if cell != previous]
        if not choices:
            return None
        following = choices[0]
        if following == start:
            break
        if following in result:
            return None
        result.append(following)
        previous, current = current, following
        if len(result) > total:
            return None
    candidate = tuple(result)
    return candidate if _valid_tour(candidate, rows, cols) else None


def _band_tour(rows: int, cols: int, seams: tuple[int, ...]) -> tuple[tuple[int, int], ...] | None:
    """Join two-row rings at checked vertical seams.

    This is only a candidate generator.  The degree-two walk and the complete
    adjacency check below are the proof boundary used by the controller.
    """
    if rows < 2 or cols < 2 or rows % 2:
        return None
    if len(seams) != rows // 2 - 1:
        return None
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for top in range(0, rows, 2):
        ring = [(top, col) for col in range(cols)]
        ring.extend((top + 1, col) for col in range(cols - 1, -1, -1))
        for index, cell in enumerate(ring):
            _edge_add(graph, cell, ring[(index + 1) % len(ring)])
    for band, seam in enumerate(seams):
        if not 0 <= seam < cols - 1:
            return None
        upper, lower = 2 * band + 1, 2 * band + 2
        _edge_drop(graph, (upper, seam), (upper, seam + 1))
        _edge_drop(graph, (lower, seam), (lower, seam + 1))
        _edge_add(graph, (upper, seam), (lower, seam))
        _edge_add(graph, (upper, seam + 1), (lower, seam + 1))
    return _walk_cycle(graph, rows, cols)


def _seam_choices(rows: int, cols: int) -> list[tuple[int, ...]]:
    count = rows // 2 - 1
    width = cols - 1
    if count <= 0:
        return [()]
    patterns: list[tuple[int, ...]] = []

    def add(values: list[int]) -> None:
        pattern = tuple(value % width for value in values)
        if pattern not in patterns:
            patterns.append(pattern)

    add([0] * count)
    add([width - 1] * count)
    add([width // 2] * count)
    for phase in range(min(width, 8)):
        add([(phase + 2 * index) % width for index in range(count)])
        add([(phase + width - 1 - 2 * index) % width for index in range(count)])
    # A few single-seam perturbations are useful on 12x12 without making the
    # tick path depend on a large cycle generator.
    for index in range(count):
        values = [0] * count
        values[index] = (2 + 3 * index) % width
        add(values)
    return patterns[:16]


def _tour_portfolio(rows: int, cols: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    bases: list[tuple[tuple[int, int], ...]] = []
    direct = _row_spine(rows, cols)
    if direct is not None:
        bases.append(direct)
    if cols % 2 == 0:
        transposed = _row_spine(cols, rows)
        if transposed is not None:
            bases.append(tuple((col, row) for row, col in transposed))
    for seams in _seam_choices(rows, cols):
        candidate = _band_tour(rows, cols, seams)
        if candidate is not None:
            bases.append(candidate)
    if cols % 2 == 0:
        for seams in _seam_choices(cols, rows):
            candidate = _band_tour(cols, rows, seams)
            if candidate is not None:
                bases.append(tuple((col, row) for row, col in candidate))

    result: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in bases:
        for flip_row in (False, True):
            for flip_col in (False, True):
                transformed = tuple(
                    (rows - 1 - row if flip_row else row,
                     cols - 1 - col if flip_col else col)
                    for row, col in base
                )
                for oriented in (transformed, tuple(reversed(transformed))):
                    if oriented in seen or not _valid_tour(oriented, rows, cols):
                        continue
                    seen.add(oriented)
                    result.append(oriented)
                    if len(result) >= 64:
                        return tuple(result)
    return tuple(result)


def _transition(
    body: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """Apply one exact public transition, including simultaneous tail vacating."""
    if not body or heading not in VECTORS or request not in VECTORS:
        return None
    actual = heading if request == OPPOSITE.get(heading) else request
    dr, dc = VECTORS[actual]
    next_head = body[0][0] + dr, body[0][1] + dc
    if not _inside(next_head, rows, cols):
        return None
    ate = food is not None and next_head == food
    occupied = set(body if ate else body[:-1])
    if next_head in occupied:
        return None
    return (next_head,) + (body if ate else body[:-1]), actual, ate


def _order_span(
    body: tuple[tuple[int, int], ...],
    tour: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int],
) -> int | None:
    """Return the occupied forward span when the body has tour order."""
    if not body or len(set(body)) != len(body):
        return None
    if any(cell not in positions for cell in body):
        return None
    size = len(tour)
    span = 0
    for newer, older in zip(body, body[1:]):
        gap = (positions[newer] - positions[older]) % size
        if gap <= 0:
            return None
        span += gap
    return span if span < size else None


def _free_area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not body:
        return 0
    blocked = set(body[1:])
    queue = deque([body[0]])
    reached = {body[0]}
    while queue:
        row, col = queue.popleft()
        for dr, dc in VECTORS.values():
            cell = row + dr, col + dc
            if _inside(cell, rows, cols) and cell not in blocked and cell not in reached:
                reached.add(cell)
                queue.append(cell)
    return len(reached)


def _forward_distance(
    head: tuple[int, int], food: tuple[int, int] | None,
    tour: tuple[tuple[int, int], ...], positions: dict[tuple[int, int], int],
) -> int:
    if food is None or food not in positions or head not in positions:
        return 0
    return (positions[food] - positions[head]) % len(tour)


def _tour_choice(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None,
    rows: int, cols: int, tour: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int], advertised: tuple[str, ...] | None = None,
) -> str | None:
    if _order_span(body, tour, positions) is None:
        return None
    next_cell = tour[(positions[body[0]] + 1) % len(tour)]
    options: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if advertised is not None and action not in advertised:
            continue
        if action == OPPOSITE.get(heading):
            continue
        step = _transition(body, heading, action, food, rows, cols)
        if step is None:
            continue
        new_body, new_heading, ate = step
        if _order_span(new_body, tour, positions) is None:
            continue
        distance = 0 if ate else _forward_distance(new_body[0], food, tour, positions)
        rank = (
            int(ate),
            -distance,
            int(new_body[0] == next_cell),
            int(new_heading == heading),
            action,
        )
        options.append((rank, action))
    if not options:
        return None
    options.sort(reverse=True)
    return options[0][1]


def _greedy_tour_route(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int],
    rows: int, cols: int, tour: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int], first: str,
) -> tuple[str, ...] | None:
    current_body, current_heading = body, heading
    route: list[str] = []
    for depth in range(len(tour) + len(body) + 6):
        action = first if depth == 0 else _tour_choice(
            current_body, current_heading, food, rows, cols, tour, positions
        )
        if action is None:
            return None
        step = _transition(current_body, current_heading, action, food, rows, cols)
        if step is None or _order_span(step[0], tour, positions) is None:
            return None
        route.append(action)
        current_body, current_heading, ate = step
        if ate:
            return tuple(route)
    return None


def _astar_tour_route(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int],
    rows: int, cols: int, tour: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int], first: str,
    deadline: float,
) -> tuple[str, ...] | None:
    if _order_span(body, tour, positions) is None:
        return None
    sequence = 0
    start_h = abs(body[0][0] - food[0]) + abs(body[0][1] - food[1])
    heap: list[tuple[float, int, int, int, tuple[tuple[int, int], ...], str, tuple[str, ...]]] = [
        (float(start_h), 0, 0, sequence, body, heading, ())
    ]
    best: dict[tuple[tuple[tuple[int, int], ...], str], tuple[int, int]] = {
        (body, heading): (0, 0)
    }
    expansions = 0
    max_depth = min(rows * cols + len(body) + 8, 240)
    while heap and expansions < 9000 and time.monotonic() < deadline:
        _, depth, turns, _, current_body, current_heading, path = heapq.heappop(heap)
        expansions += 1
        if depth >= max_depth:
            continue
        actions = (first,) if depth == 0 else DIRECTIONS
        ordered_actions: list[tuple[int, int, str]] = []
        for action in actions:
            if action == OPPOSITE.get(current_heading):
                continue
            dr, dc = VECTORS[action]
            nr, nc = current_body[0][0] + dr, current_body[0][1] + dc
            ordered_actions.append((abs(nr - food[0]) + abs(nc - food[1]),
                                    int(action != current_heading), action))
        ordered_actions.sort()
        for _, _, action in ordered_actions:
            step = _transition(current_body, current_heading, action, food, rows, cols)
            if step is None:
                continue
            next_body, next_heading, ate = step
            if _order_span(next_body, tour, positions) is None:
                continue
            next_path = path + (action,)
            if ate:
                return next_path
            next_depth = depth + 1
            next_turns = turns + int(next_heading != current_heading)
            key = (next_body, next_heading)
            prior = best.get(key)
            if prior is not None and prior <= (next_depth, next_turns):
                continue
            best[key] = (next_depth, next_turns)
            sequence += 1
            heuristic = abs(next_body[0][0] - food[0]) + abs(next_body[0][1] - food[1])
            cycle_hint = _forward_distance(next_body[0], food, tour, positions)
            priority = next_depth + heuristic + 0.001 * cycle_hint
            heapq.heappush(heap, (priority, next_depth, next_turns, sequence,
                                  next_body, next_heading, next_path))
    return None


def _simulate_route(
    origin_body: tuple[tuple[int, int], ...], origin_heading: str,
    food: tuple[int, int], route: tuple[str, ...], rows: int, cols: int,
    tour: tuple[tuple[int, int], ...], positions: dict[tuple[int, int], int],
) -> tuple[tuple[tuple[int, int], ...], str] | None:
    body, heading = origin_body, origin_heading
    for action in route:
        step = _transition(body, heading, action, food, rows, cols)
        if step is None or _order_span(step[0], tour, positions) is None:
            return None
        body, heading, ate = step
        if ate:
            return (body, heading)
    return None


def _safe_tour_ids(
    body: tuple[tuple[int, int], ...], tours: tuple[tuple[tuple[int, int], ...], ...],
    indexes: tuple[dict[tuple[int, int], int], ...],
) -> list[int]:
    return [i for i, tour in enumerate(tours)
            if _order_span(body, tour, indexes[i]) is not None]


def _route_worker(
    origin_body: tuple[tuple[int, int], ...], origin_heading: str,
    food: tuple[int, int], rows: int, cols: int,
    tours: tuple[tuple[tuple[int, int], ...], ...],
    indexes: tuple[dict[tuple[int, int], int], ...],
    active_id: int | None, first: str,
):
    """Search several fixed invariant choices, returning only checked routes."""
    if not origin_body or not tours:
        return None
    started = time.monotonic()
    deadline = started + 0.052
    compatible = _safe_tour_ids(origin_body, tours, indexes)
    if not compatible:
        return None
    compatible.sort(key=lambda i: (0 if i == active_id else 1,
                                  _forward_distance(origin_body[0], food, tours[i], indexes[i]), i))

    # The currently active tour's greedy route is the no-regression reference.
    reference: tuple[str, ...] | None = None
    if active_id in compatible:
        reference = _greedy_tour_route(origin_body, origin_heading, food, rows, cols,
                                       tours[active_id], indexes[active_id], first)
    reference_len = len(reference) if reference is not None else rows * cols + 1
    reference_end = _simulate_route(origin_body, origin_heading, food, reference,
                                    rows, cols, tours[active_id], indexes[active_id]) \
        if reference is not None and active_id in compatible else None
    reference_flex = 0
    if reference_end is not None:
        reference_flex = len(_safe_tour_ids(reference_end[0], tours, indexes))

    candidates: list[dict[str, Any]] = []
    seen_routes: set[tuple[str, ...]] = set()
    for tour_id in compatible[:24]:
        if time.monotonic() >= deadline:
            break
        tour, positions = tours[tour_id], indexes[tour_id]
        baseline = _greedy_tour_route(origin_body, origin_heading, food, rows, cols,
                                       tour, positions, first)
        route = _astar_tour_route(origin_body, origin_heading, food, rows, cols,
                                  tour, positions, first, deadline)
        if route is None or route in seen_routes:
            continue
        end = _simulate_route(origin_body, origin_heading, food, route, rows, cols,
                              tour, positions)
        if end is None:
            continue
        length = len(route)
        if length > reference_len:
            continue
        post_flex = len(_safe_tour_ids(end[0], tours, indexes))
        strict_gain = length < reference_len
        equal_gain = length == reference_len and post_flex > reference_flex
        if not strict_gain and not equal_gain:
            continue
        seen_routes.add(route)
        turns = sum(1 for a, b in zip(route, route[1:]) if a != b)
        candidates.append({
            "origin_body": origin_body,
            "origin_heading": origin_heading,
            "food": food,
            "route": route,
            "tour_id": tour_id,
            "length": length,
            "baseline": len(baseline) if baseline is not None else None,
            "post_flex": post_flex,
            "post_area": _free_area(end[0], rows, cols),
            "turns": turns,
            "strict_gain": strict_gain,
        })

    candidates.sort(key=lambda item: (item["length"], -item["post_flex"],
                                      -item["post_area"], item["turns"], item["tour_id"]))
    return {
        "origin_body": origin_body,
        "origin_heading": origin_heading,
        "food": food,
        "candidates": tuple(candidates[:5]),
        "reference_len": reference_len,
        "reference_flex": reference_flex,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


def _route_suffix(
    candidate: dict[str, Any], body: tuple[tuple[int, int], ...], heading: str,
    food: tuple[int, int] | None, rows: int, cols: int,
    tours: tuple[tuple[tuple[int, int], ...], ...],
    indexes: tuple[dict[tuple[int, int], int], ...],
) -> tuple[str, ...] | None:
    if not isinstance(candidate, dict) or food is None or candidate.get("food") != food:
        return None
    origin_body = candidate.get("origin_body")
    origin_heading = candidate.get("origin_heading")
    route = candidate.get("route")
    tour_id = candidate.get("tour_id")
    if not isinstance(origin_body, tuple) or not isinstance(route, tuple):
        return None
    if not isinstance(origin_heading, str) or not isinstance(tour_id, int):
        return None
    if not 0 <= tour_id < len(tours) or not route:
        return None
    tour, positions = tours[tour_id], indexes[tour_id]
    current_body, current_heading = origin_body, origin_heading
    for index, action in enumerate(route):
        if current_body == body and current_heading == heading and food == candidate.get("food"):
            return route[index:]
        step = _transition(current_body, current_heading, action, food, rows, cols)
        if step is None or _order_span(step[0], tour, positions) is None:
            return None
        current_body, current_heading, ate = step
        if ate:
            return None
    if current_body == body and current_heading == heading:
        return ()
    return None


def _safe_fallback(
    body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None,
    rows: int, cols: int, advertised: tuple[str, ...],
) -> str:
    options: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action not in advertised or action == OPPOSITE.get(heading):
            continue
        step = _transition(body, heading, action, food, rows, cols)
        if step is None:
            continue
        new_body, new_heading, ate = step
        area = _free_area(new_body, rows, cols)
        distance = 0 if food is None else abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
        options.append(((int(ate), area, -distance, int(new_heading == heading), action), action))
    if options:
        options.sort(reverse=True)
        return options[0][1]
    return heading if heading in advertised else (advertised[0] if advertised else "NOOP")


def _json_number(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
        return number if number >= 0 else default
    except (TypeError, ValueError, OverflowError):
        return default


def _metric_bucket(value: Any) -> str:
    number = _json_number(value)
    if number <= 1:
        return "minimal"
    if number <= 4:
        return "small"
    if number <= 12:
        return "moderate"
    return "large"


class _Controller:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.rows = 0
        self.cols = 0
        self.tours: tuple[tuple[tuple[int, int], ...], ...] = ()
        self.indexes: tuple[dict[tuple[int, int], int], ...] = ()
        self.active_id: int | None = None
        self.route: dict[str, Any] | None = None
        self.route_options: list[dict[str, Any]] = []
        self.search_future: Any = None
        self.judgment_future: Any = None
        self.judgment_labels: tuple[str, ...] = ()
        self.last_search_tick = -100000
        self.last_search_food: tuple[int, int] | None = None
        self.last_judged_food: tuple[int, int] | None = None
        self.supervisor_future: Any = None
        self.supervisor_started = False
        self.policy = "BALANCED"
        self.last_food: tuple[int, int] | None = None

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _set_board(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        self.tours = _tour_portfolio(rows, cols)
        self.indexes = tuple({cell: i for i, cell in enumerate(tour)} for tour in self.tours)
        self.active_id = None
        self.route = None
        self.route_options = []
        self.search_future = None
        self.judgment_future = None
        self.last_search_tick = -100000
        self.last_search_food = None
        self.last_judged_food = None
        self._log("tour_portfolio_ready", rows=rows, cols=cols, tours=len(self.tours))

    def _start_supervisor(self, obs: Any) -> None:
        if self.supervisor_started or self.ctx.done:
            return
        self.supervisor_started = True
        body = _read_body(obs)
        food = _read_food(obs)
        state = {
            "rows": self.rows,
            "cols": self.cols,
            "length": len(body),
            "free_cells_bucket": _metric_bucket((self.rows * self.cols) - len(body)),
            "food_present": food is not None,
            "invariant_candidates": len(_safe_tour_ids(body, self.tours, self.indexes)) if body else 0,
        }
        prompt = (
            "You are the slow supervisor for a bounded classic Snake controller. "
            "Code enforces exact wall/body/tail-vacating transitions and only executes "
            "routes whose every state preserves a validated Hamiltonian tour. Choose "
            "one route objective for the rest of this episode. Return exactly one token: "
            "SPEED, BALANCED, or FLEX. SPEED prefers the fewest steps to food; FLEX "
            "may accept a route no more than one step longer when it leaves more "
            "tour choices after growth; BALANCED prefers speed and uses flexibility only "
            "as a tie break. Never return an action or coordinates. Facts: " + repr(state)
        )
        try:
            self.supervisor_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=55.0)
            )
        except Exception as exc:
            self.supervisor_future = None
            self._log("supervisor_spawn_failed", reason=type(exc).__name__)

    def _poll_supervisor(self, tick: int) -> None:
        future = self.supervisor_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.supervisor_future = None
        try:
            result = future.result()
            text = str(getattr(result, "text", result)).upper()
        except Exception as exc:
            self._log("supervisor_ignored", tick=tick, reason=type(exc).__name__)
            return
        selected = None
        for token in ("SPEED", "FLEX", "BALANCED"):
            if token in text:
                selected = token
                break
        if selected is not None:
            self.policy = selected
            self._log("supervisor_policy", tick=tick, policy=selected)

    def _choose_tour(self, body: tuple[tuple[int, int], ...], heading: str,
                     food: tuple[int, int] | None, advertised: tuple[str, ...]) -> str | None:
        if not self.tours:
            return None
        choices: list[tuple[tuple[int, int, int, int], str, int]] = []
        for index, tour in enumerate(self.tours):
            action = _tour_choice(body, heading, food, self.rows, self.cols,
                                  tour, self.indexes[index], advertised)
            if action is None:
                continue
            rank = (
                int(index == self.active_id),
                int(action == heading),
                -_forward_distance(body[0], food, tour, self.indexes[index]),
                -index,
            )
            choices.append((rank, action, index))
        if not choices:
            return None
        choices.sort(reverse=True)
        _, action, index = choices[0]
        if index != self.active_id:
            self._log("tour_switch", old=self.active_id, new=index)
        self.active_id = index
        return action

    def _current_suffix(self, candidate: dict[str, Any], body: tuple[tuple[int, int], ...],
                        heading: str, food: tuple[int, int] | None) -> tuple[str, ...] | None:
        return _route_suffix(candidate, body, heading, food, self.rows, self.cols,
                             self.tours, self.indexes)

    def _best_route(self, options: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not options:
            return None
        if self.policy == "FLEX":
            shortest = min(_json_number(item.get("length"), 99999) for item in options)
            eligible = [item for item in options
                        if _json_number(item.get("length"), 99999) <= shortest + 1]
            return max(eligible, key=lambda item: (
                _json_number(item.get("post_flex")),
                _json_number(item.get("post_area")),
                -_json_number(item.get("length"), 99999),
            ))
        if self.policy == "SPEED":
            return min(options, key=lambda item: (
                _json_number(item.get("length"), 99999),
                -_json_number(item.get("post_flex")),
                -_json_number(item.get("post_area")),
            ))
        return min(options, key=lambda item: (
            _json_number(item.get("length"), 99999),
            -_json_number(item.get("post_flex")),
            -_json_number(item.get("post_area")),
        ))

    def _start_judgment(self, result: dict[str, Any], body: tuple[tuple[int, int], ...],
                        heading: str, food: tuple[int, int], tick: int) -> None:
        if self.judgment_future is not None or self.last_judged_food == food:
            return
        options = result.get("candidates")
        if not isinstance(options, tuple):
            return
        usable = [item for item in options if isinstance(item, dict) and isinstance(item.get("route"), tuple)]
        if len(usable) < 2:
            return
        usable = usable[:3]
        labels = tuple("R" + str(i) for i in range(len(usable)))
        for label, item in zip(labels, usable):
            item["label"] = label
        self.route_options = usable
        self.judgment_labels = labels
        criteria: dict[str, str] = {}
        for label, item in zip(labels, usable):
            strict = "beats the code fallback" if item.get("strict_gain") else "ties the fallback but leaves more tour choices"
            criteria[label] = (
                f"{strict}; reaches the current food in a {_metric_bucket(item.get('length'))} route; "
                f"post-food tour flexibility is {_metric_bucket(item.get('post_flex'))} and open area is {_metric_bucket(item.get('post_area'))}."
            )
        reversed_criteria = dict(reversed(list(criteria.items())))
        state = {
            "controller_policy": self.policy,
            "observation": {
                "tick": _json_number(tick),
                "length": len(body),
                "free_cells_bucket": _metric_bucket(self.rows * self.cols - len(body)),
                "food_visible": food is not None,
                "current_heading": heading if heading in DIRECTIONS else "unknown",
            },
            "candidates": [
                {
                    "label": label,
                    "steps_bucket": _metric_bucket(item.get("length")),
                    "beats_fallback": bool(item.get("strict_gain")),
                    "post_tour_choices_bucket": _metric_bucket(item.get("post_flex")),
                    "post_open_area_bucket": _metric_bucket(item.get("post_area")),
                }
                for label, item in zip(labels, usable)
            ],
        }
        questions = {
            "route_forward": Choice(
                "Select the safest useful route label from `candidates`. Prefer fewer steps to the visible food. "
                "Only use flexibility as a tie break unless `controller_policy` is FLEX.",
                criteria,
            ),
            "route_reversed": Choice(
                "Select the same route label independently. Do not invent a label and do not choose a route "
                "that is not in `candidates`.",
                reversed_criteria,
            ),
        }
        try:
            self.judgment_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.55)
            )
            self.last_judged_food = food
            self._log("route_judgment_started", tick=tick, labels=list(labels))
        except Exception as exc:
            self.judgment_future = None
            self._log("route_judgment_spawn_failed", tick=tick, reason=type(exc).__name__)

    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _answer_probability(answer: Any, label: str) -> float:
        try:
            probabilities = getattr(answer, "probabilities", {})
            value = probabilities.get(label, 0.0) if isinstance(probabilities, dict) else 0.0
            number = float(value)
            return number if 0.0 <= number <= 1.0 else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def _poll_judgment(self, body: tuple[tuple[int, int], ...], heading: str,
                       food: tuple[int, int] | None, tick: int) -> None:
        future = self.judgment_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.judgment_future = None
        try:
            answers = future.result()
            first = answers.get("route_forward") if isinstance(answers, dict) else None
            second = answers.get("route_reversed") if isinstance(answers, dict) else None
            a, b = self._answer_value(first), self._answer_value(second)
            if a is None or a != b or a not in self.judgment_labels:
                self._log("route_judgment_rejected", tick=tick, reason="disagreement_or_invalid")
                return
            if min(self._answer_probability(first, a), self._answer_probability(second, a)) < 0.52:
                self._log("route_judgment_rejected", tick=tick, reason="low_probability", label=a)
                return
            selected = next((item for item in self.route_options if item.get("label") == a), None)
            if selected is None or self._current_suffix(selected, body, heading, food) is None:
                self._log("route_judgment_rejected", tick=tick, reason="stale", label=a)
                return
            self.route = selected
            self._log("route_judgment_applied", tick=tick, label=a,
                       length=selected.get("length"))
        except Exception as exc:
            self._log("route_judgment_ignored", tick=tick, reason=type(exc).__name__)

    def _poll_search(self, body: tuple[tuple[int, int], ...], heading: str,
                     food: tuple[int, int] | None, tick: int) -> None:
        future = self.search_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.search_future = None
        try:
            result = future.result()
        except Exception as exc:
            self._log("route_search_ignored", tick=tick, reason=type(exc).__name__)
            return
        if not isinstance(result, dict) or result.get("food") != food:
            return
        candidates = result.get("candidates")
        if not isinstance(candidates, tuple):
            return
        current = [item for item in candidates
                   if isinstance(item, dict) and self._current_suffix(item, body, heading, food)]
        if not current:
            self._log("route_search_discarded", tick=tick, reason="prefix_or_food_mismatch")
            return
        self.route_options = current
        self.route = self._best_route(current)
        if self.route is not None:
            self._log("route_installed", tick=tick, length=self.route.get("length"),
                       tour=self.route.get("tour_id"), candidates=len(current),
                       elapsed_ms=result.get("elapsed_ms", 0))
        self._start_judgment(result, body, heading, food, tick)

    def _start_search(self, body: tuple[tuple[int, int], ...], heading: str,
                      food: tuple[int, int] | None, tick: int, first: str) -> None:
        if food is None or not self.tours or self.search_future is not None:
            return
        if tick - self.last_search_tick < 4 and food == self.last_search_food:
            return
        if self.ctx.done:
            return
        snapshot = (body, heading, food, self.rows, self.cols, self.tours,
                    self.indexes, self.active_id, first)
        try:
            self.search_future = self.ctx.spawn(lambda: _route_worker(*snapshot))
            self.last_search_tick = tick
            self.last_search_food = food
            self._log("route_search_started", tick=tick)
        except Exception as exc:
            self.search_future = None
            self._log("route_search_spawn_failed", tick=tick, reason=type(exc).__name__)

    def choose(self, obs: Any, advertised: tuple[str, ...]) -> str:
        dimensions = _read_dimensions(obs)
        body = _read_body(obs)
        if dimensions is None or not body:
            return advertised[0] if advertised else "NOOP"
        self._set_board(*dimensions)
        food = _read_food(obs)
        heading = _read_heading(obs, body)
        tick = _read_tick(obs)
        if not self.supervisor_started:
            total_cells = self.rows * self.cols
            high_occupancy = len(body) * 100 >= total_cells * 55
            late_unusual = tick >= 1800 and len(body) * 100 >= total_cells * 35
            if high_occupancy or late_unusual:
                self._start_supervisor(obs)
        self._poll_supervisor(tick)
        self._poll_search(body, heading, food, tick)
        self._poll_judgment(body, heading, food, tick)

        if food != self.last_food:
            self.route_options = []
            if self.route is not None and self.route.get("food") != food:
                self.route = None
            self.last_food = food
        if self.route is not None and self._current_suffix(self.route, body, heading, food) is None:
            self.route = None

        action: str | None = None
        if self.route is not None:
            suffix = self._current_suffix(self.route, body, heading, food)
            if suffix:
                candidate = suffix[0]
                if candidate in advertised:
                    action = candidate
            else:
                self.route = None

        if action is None:
            if self.active_id is not None:
                action = _tour_choice(body, heading, food, self.rows, self.cols,
                                      self.tours[self.active_id], self.indexes[self.active_id], advertised)
            if action is None:
                action = self._choose_tour(body, heading, food, advertised)
            if action is None:
                action = _safe_fallback(body, heading, food, self.rows, self.cols, advertised)

        if action not in advertised:
            action = heading if heading in advertised else (advertised[0] if advertised else "NOOP")
        # A route search always begins with the action posted here.  This makes
        # a completed worker result a valid prefix at the next observation.
        if self.route is None and action in DIRECTIONS:
            self._start_search(body, heading, food, tick, action)
        return action


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

    def run_episode(self, game: Any) -> None:
        if not getattr(game, "tick_s", None):
            return
        spec = getattr(game, "action_spec", None)
        names = tuple(name for name in (getattr(spec, "names", ()) or ())
                      if isinstance(name, str))
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
