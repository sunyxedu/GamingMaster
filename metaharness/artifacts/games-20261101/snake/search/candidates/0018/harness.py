"""Candidate 0018: food-epoch planning with a moving certificate.

The controller does not ask a model to steer individual ticks.  It maintains a
validated Hamiltonian certificate, chooses a certificate for a food *epoch*,
and lets a bounded worker search routes whose every intermediate state has at
least one certificate.  A route can therefore change certificates at a safe
state, but it can never leave the mechanically checked safety envelope.

Jev is used only as a durable tie-breaker between code-computed food policies.
Its answer changes a policy for the current food epoch (or a later one), never
an unverified action.  The default policy is deterministic when Jev is late or
unavailable.
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
POLICIES = ("SPEED", "BALANCE", "FLEX")


def _cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _body(observation: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(observation, dict):
        return ()
    raw = observation.get("snake")
    if not isinstance(raw, (list, tuple)):
        return ()
    result: list[tuple[int, int]] = []
    for value in raw:
        point = _cell(value)
        if point is None:
            return ()
        result.append(point)
    return tuple(result)


def _food(observation: Any) -> tuple[int, int] | None:
    if not isinstance(observation, dict) or observation.get("food") is None:
        return None
    return _cell(observation.get("food"))


def _shape(observation: Any) -> tuple[int, int] | None:
    if not isinstance(observation, dict):
        return None
    try:
        rows, cols = int(observation.get("rows")), int(observation.get("cols"))
    except (TypeError, ValueError, OverflowError):
        return None
    if rows < 2 or cols < 2 or rows * cols > 4096:
        return None
    return rows, cols


def _tick(observation: Any) -> int:
    if not isinstance(observation, dict):
        return 0
    try:
        return max(0, int(observation.get("tick", 0)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _heading(observation: Any, body: tuple[tuple[int, int], ...]) -> str:
    if isinstance(observation, dict) and observation.get("direction") in VECTORS:
        return str(observation["direction"])
    if len(body) >= 2:
        delta = body[0][0] - body[1][0], body[0][1] - body[1][1]
        for name, vector in VECTORS.items():
            if delta == vector:
                return name
    return "RIGHT"


def _inside(point: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= point[0] < rows and 0 <= point[1] < cols


def _adjacent(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1


def _advance(
    body: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """Apply one public tick, including ignored reverse and tail vacating."""
    if not body or heading not in VECTORS or request not in VECTORS:
        return None
    actual = heading if request == OPPOSITE.get(heading) else request
    dr, dc = VECTORS[actual]
    next_head = body[0][0] + dr, body[0][1] + dc
    if not _inside(next_head, rows, cols):
        return None
    grows = food is not None and next_head == food
    occupied = set(body) if grows else set(body[:-1])
    if next_head in occupied:
        return None
    next_body = (next_head,) + (body if grows else body[:-1])
    return next_body, actual, grows


# ---------------------------------------------------------------------------
# Validated cycle bank.  The bank is a geometric search space, not game data.


def _cycle_ok(cycle: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    size = rows * cols
    return (
        len(cycle) == size
        and len(set(cycle)) == size
        and all(_inside(point, rows, cols) for point in cycle)
        and all(_adjacent(cycle[i], cycle[(i + 1) % size]) for i in range(size))
    )


def _spine_cycle(rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """Make the standard even-height cycle with a returning left spine."""
    if rows < 2 or cols < 2 or rows % 2:
        return None
    path: list[tuple[int, int]] = [(0, column) for column in range(cols)]
    for row in range(1, rows):
        columns = range(cols - 1, 0, -1) if row % 2 else range(1, cols)
        path.extend((row, column) for column in columns)
    path.append((rows - 1, 0))
    path.extend((row, 0) for row in range(rows - 2, 0, -1))
    answer = tuple(path)
    return answer if _cycle_ok(answer, rows, cols) else None


def _add_edge(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    first: tuple[int, int],
    second: tuple[int, int],
) -> None:
    graph.setdefault(first, set()).add(second)
    graph.setdefault(second, set()).add(first)


def _remove_edge(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    first: tuple[int, int],
    second: tuple[int, int],
) -> None:
    graph.get(first, set()).discard(second)
    graph.get(second, set()).discard(first)


def _walk_graph(
    graph: dict[tuple[int, int], set[tuple[int, int]]], rows: int, cols: int
) -> tuple[tuple[int, int], ...] | None:
    size = rows * cols
    if len(graph) != size or any(len(neighbors) != 2 for neighbors in graph.values()):
        return None
    start = (0, 0)
    if start not in graph:
        return None
    path = [start]
    previous: tuple[int, int] | None = None
    current = start
    while True:
        next_options = sorted(graph[current])
        if previous is not None:
            next_options = [point for point in next_options if point != previous]
        if not next_options:
            return None
        following = next_options[0]
        if following == start:
            break
        if following in path:
            return None
        path.append(following)
        previous, current = current, following
        if len(path) > size:
            return None
    result = tuple(path)
    return result if _cycle_ok(result, rows, cols) else None


def _strip_cycle(rows: int, cols: int, seams: tuple[int, ...]) -> tuple[tuple[int, int], ...] | None:
    """Join two-row rings at vertical seams, then prove one complete cycle."""
    if rows < 2 or cols < 2 or rows % 2 or len(seams) != rows // 2 - 1:
        return None
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for top in range(0, rows, 2):
        ring = [(top, column) for column in range(cols)]
        ring.extend((top + 1, column) for column in range(cols - 1, -1, -1))
        for index, point in enumerate(ring):
            _add_edge(graph, point, ring[(index + 1) % len(ring)])
    for band, seam in enumerate(seams):
        if not 0 <= seam < cols - 1:
            return None
        upper, lower = 2 * band + 1, 2 * band + 2
        _remove_edge(graph, (upper, seam), (upper, seam + 1))
        _remove_edge(graph, (lower, seam), (lower, seam + 1))
        _add_edge(graph, (upper, seam), (lower, seam))
        _add_edge(graph, (upper, seam + 1), (lower, seam + 1))
    return _walk_graph(graph, rows, cols)


def _seam_bank(count: int, width: int, wanted: int, salt: int) -> list[tuple[int, ...]]:
    if count <= 0 or width <= 0:
        return [()]
    answer: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(values: list[int] | tuple[int, ...]) -> None:
        pattern = tuple(value % width for value in values)
        if pattern not in seen:
            seen.add(pattern)
            answer.append(pattern)

    for value in (0, width - 1, width // 2):
        add([value] * count)
    for phase in range(width):
        add([(phase + index) % width for index in range(count)])
        add([(phase - index) % width for index in range(count)])
    for index in range(count):
        values = [0] * count
        values[index] = (2 + 3 * index + salt) % width
        add(values)

    # A deterministic arithmetic bank supplies variety without using a seed.
    state = (salt | 1) & 0x7FFFFFFF
    while len(answer) < wanted:
        values: list[int] = []
        for _ in range(count):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            values.append(state % width)
        add(values)
    return answer[:wanted]


def _cycle_bank(rows: int, cols: int, limit: int = 96) -> tuple[tuple[tuple[int, int], ...], ...]:
    if rows < 2 or cols < 2 or (rows % 2 and cols % 2):
        return ()
    bases: list[tuple[tuple[int, int], ...]] = []
    direct = _spine_cycle(rows, cols)
    if direct is not None:
        bases.append(direct)
    if cols % 2 == 0:
        transposed = _spine_cycle(cols, rows)
        if transposed is not None:
            bases.append(tuple((column, row) for row, column in transposed))

    patterns = _seam_bank(rows // 2 - 1, cols - 1, 24, 17)
    for seams in patterns:
        candidate = _strip_cycle(rows, cols, seams)
        if candidate is not None:
            bases.append(candidate)
    if cols % 2 == 0:
        other_patterns = _seam_bank(cols // 2 - 1, rows - 1, 24, 31)
        for seams in other_patterns:
            candidate = _strip_cycle(cols, rows, seams)
            if candidate is not None:
                bases.append(tuple((column, row) for row, column in candidate))

    answer: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in bases:
        for flip_row, flip_col in ((False, False), (False, True), (True, False), (True, True)):
            reflected = tuple(
                (
                    rows - 1 - row if flip_row else row,
                    cols - 1 - column if flip_col else column,
                )
                for row, column in base
            )
            for candidate in (reflected, tuple(reversed(reflected))):
                if candidate in seen or not _cycle_ok(candidate, rows, cols):
                    continue
                seen.add(candidate)
                answer.append(candidate)
                if len(answer) >= limit:
                    return tuple(answer)
    return tuple(answer)


def _positions(cycle: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], int]:
    return {point: index for index, point in enumerate(cycle)}


def _ordered(
    body: tuple[tuple[int, int], ...], positions: dict[tuple[int, int], int], size: int
) -> bool:
    if not body or len(set(body)) != len(body) or any(point not in positions for point in body):
        return False
    span = 0
    for newer, older in zip(body, body[1:]):
        gap = (positions[newer] - positions[older]) % size
        if gap <= 0:
            return False
        span += gap
    return span < size


def _certificates(
    body: tuple[tuple[int, int], ...],
    indexes: tuple[dict[tuple[int, int], int], ...],
    size: int,
) -> tuple[int, ...]:
    if not body or len(set(body)) != len(body):
        return ()
    return tuple(index for index, positions in enumerate(indexes) if _ordered(body, positions, size))


# ---------------------------------------------------------------------------
# Exact cycle policy and conservative geometric helpers.


def _cycle_move(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int],
    advertised: tuple[str, ...],
) -> str | None:
    if not _ordered(body, positions, len(cycle)):
        return None
    current_index = positions[body[0]]
    next_cycle_point = cycle[(current_index + 1) % len(cycle)]
    choices: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action not in advertised or action == OPPOSITE.get(heading):
            continue
        step = _advance(body, heading, action, food, rows, cols)
        if step is None or not _ordered(step[0], positions, len(cycle)):
            continue
        new_body, new_heading, ate = step
        if ate or food is None or food not in positions:
            forward = 0
        else:
            forward = (positions[food] - positions[new_body[0]]) % len(cycle)
        choices.append(
            (
                (
                    int(ate),
                    -forward,
                    int(new_body[0] == next_cycle_point),
                    int(new_heading == heading),
                    action,
                ),
                action,
            )
        )
    if not choices:
        return None
    choices.sort(reverse=True)
    return choices[0][1]


def _cycle_food_route(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int],
    advertised: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...], str] | None:
    if not _ordered(body, positions, len(cycle)):
        return None
    current_body, current_heading = body, heading
    path: list[str] = []
    limit = len(cycle) + len(body) + 8
    for _ in range(limit):
        action = _cycle_move(
            current_body, current_heading, food, rows, cols, cycle, positions, advertised
        )
        if action is None:
            return None
        step = _advance(current_body, current_heading, action, food, rows, cols)
        if step is None or not _ordered(step[0], positions, len(cycle)):
            return None
        path.append(action)
        current_body, current_heading, ate = step
        if ate:
            return tuple(path), current_body, current_heading
    return None


def _reachable_area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not body:
        return 0
    blocked = set(body[1:])
    queue = deque([body[0]])
    seen = {body[0]}
    while queue:
        row, column = queue.popleft()
        for dr, dc in VECTORS.values():
            point = row + dr, column + dc
            if _inside(point, rows, cols) and point not in blocked and point not in seen:
                seen.add(point)
                queue.append(point)
    return len(seen)


def _tail_reachable(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    if len(body) < 2:
        return True
    target = body[-1]
    blocked = set(body[1:-1])
    queue = deque([body[0]])
    seen = {body[0]}
    while queue:
        point = queue.popleft()
        if point == target:
            return True
        row, column = point
        for dr, dc in VECTORS.values():
            next_point = row + dr, column + dc
            if _inside(next_point, rows, cols) and next_point not in blocked and next_point not in seen:
                seen.add(next_point)
                queue.append(next_point)
    return False


def _safe_fallback(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    advertised: tuple[str, ...],
    indexes: tuple[dict[tuple[int, int], int], ...],
) -> str:
    size = rows * cols
    candidates: list[tuple[tuple[int, int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action not in advertised or action == OPPOSITE.get(heading):
            continue
        step = _advance(body, heading, action, food, rows, cols)
        if step is None:
            continue
        new_body, new_heading, ate = step
        cert_count = len(_certificates(new_body, indexes, size))
        area = _reachable_area(new_body, rows, cols)
        tail = int(_tail_reachable(new_body, rows, cols))
        distance = 0 if food is None else abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
        # A certified continuation is preferred, then a food move, then room.
        candidates.append(
            (
                (int(cert_count > 0), int(ate), tail, cert_count, area, -distance, action),
                action,
            )
        )
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    if heading in advertised:
        return heading
    return advertised[0] if advertised else "NOOP"


def _bucket(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = 0
    if number <= 1:
        return "tiny"
    if number <= 4:
        return "small"
    if number <= 12:
        return "medium"
    if number <= 32:
        return "large"
    return "very_large"


def _route_suffix(
    result: Any,
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    indexes: tuple[dict[tuple[int, int], int], ...],
) -> tuple[tuple[str, ...], int] | None:
    """Match a route at an exact prefix and re-prove each intermediate state."""
    if not isinstance(result, dict) or result.get("food") != food:
        return None
    origin_body = result.get("origin_body")
    origin_heading = result.get("origin_heading")
    route = result.get("route")
    endpoint = result.get("endpoint")
    if not isinstance(origin_body, tuple) or not isinstance(origin_heading, str):
        return None
    if not isinstance(route, tuple) or not route or not isinstance(endpoint, int):
        return None
    if endpoint < 0 or endpoint >= len(indexes):
        return None
    current_body, current_heading = origin_body, origin_heading
    for offset, action in enumerate(route):
        if current_body == body and current_heading == heading:
            return route[offset:], endpoint
        step = _advance(current_body, current_heading, action, food, rows, cols)
        if step is None:
            return None
        current_body, current_heading, ate = step
        if not ate and not _certificates(current_body, indexes, rows * cols):
            return None
        if ate:
            if current_body != body or current_heading != heading:
                return None
            return route[offset + 1 :], endpoint
    if current_body == body and current_heading == heading:
        return (), endpoint
    return None


# ---------------------------------------------------------------------------
# Dynamic-certificate worker.  It searches only states that can immediately
# hand control back to at least one validated cycle.


def _dynamic_route(
    origin_body: tuple[tuple[int, int], ...],
    origin_heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    cycles: tuple[tuple[tuple[int, int], ...], ...],
    indexes: tuple[dict[tuple[int, int], int], ...],
    baseline: int | None,
    baseline_flex: int,
    stop: Any,
    active_id: int | None = None,
    policy: str = "BALANCE",
) -> dict[str, Any] | None:
    size = rows * cols
    if not origin_body or food in origin_body or origin_heading not in VECTORS:
        return None
    if not _certificates(origin_body, indexes, size):
        return None
    started = time.monotonic()
    # Baseline reconstruction belongs here, after ctx.act has posted the
    # current move.  Doing it in choose() made a new-food boundary miss a
    # 100-ms deadline in the live try.
    if baseline is None and isinstance(active_id, int) and 0 <= active_id < len(cycles):
        baseline_route = _cycle_food_route(
            origin_body,
            origin_heading,
            food,
            rows,
            cols,
            cycles[active_id],
            indexes[active_id],
            tuple(DIRECTIONS),
        )
        if baseline_route is not None:
            baseline = len(baseline_route[0])
            if policy == "FLEX":
                baseline_flex = len(_certificates(baseline_route[1], indexes, size))
    deadline = started + 0.058
    cap = min(size + len(origin_body) + 8, 240)
    if baseline is not None:
        cap = min(cap, baseline)
    if cap <= 0:
        return None

    def stopped() -> bool:
        try:
            return bool(stop())
        except Exception:
            return False

    serial = 0
    distance = abs(origin_body[0][0] - food[0]) + abs(origin_body[0][1] - food[1])
    frontier: list[tuple[float, int, int, int, tuple[tuple[int, int], ...], str, tuple[str, ...]]] = [
        (float(distance), 0, 0, serial, origin_body, origin_heading, ())
    ]
    best_seen: dict[tuple[tuple[tuple[int, int], ...], str], tuple[int, int]] = {
        (origin_body, origin_heading): (0, 0)
    }
    best_goal: tuple[tuple[int, int, int, int], tuple[str, ...], int] | None = None
    expanded = 0

    while frontier and expanded < 15000 and time.monotonic() < deadline and not stopped():
        priority, depth, turns, _, current_body, current_heading, path = heapq.heappop(frontier)
        if best_goal is not None and priority > best_goal[0][0] + 0.01:
            break
        expanded += 1
        if depth >= cap:
            continue

        ranked: list[tuple[int, int, str]] = []
        for action in DIRECTIONS:
            if action == OPPOSITE.get(current_heading):
                continue
            dr, dc = VECTORS[action]
            row = current_body[0][0] + dr
            column = current_body[0][1] + dc
            ranked.append((abs(row - food[0]) + abs(column - food[1]), int(action != current_heading), action))
        ranked.sort()

        for _, _, action in ranked:
            if stopped():
                return None
            step = _advance(current_body, current_heading, action, food, rows, cols)
            if step is None:
                continue
            next_body, next_heading, ate = step
            next_path = path + (action,)
            next_depth = depth + 1
            next_turns = turns + int(next_heading != current_heading)
            certs = _certificates(next_body, indexes, size)
            if ate:
                if not certs:
                    continue
                # Strictly shorter routes win.  An equal route is useful only
                # if it opens more continuation certificates than the baseline.
                if baseline is not None:
                    if next_depth > baseline:
                        continue
                    if next_depth == baseline and (
                        baseline_flex <= 0 or len(certs) <= baseline_flex
                    ):
                        continue
                # Prefer the first independently validated continuation.  The
                # foreground will re-check it at the food tick; an arbitrary
                # high-numbered certificate should not perturb the baseline.
                endpoint = certs[0]
                rank = (next_depth, -len(certs), -_reachable_area(next_body, rows, cols), next_turns)
                candidate = (rank, next_path, endpoint)
                if best_goal is None or candidate[0] < best_goal[0]:
                    best_goal = candidate
                continue

            if not certs:
                continue
            if next_depth >= cap:
                continue
            remaining = abs(next_body[0][0] - food[0]) + abs(next_body[0][1] - food[1])
            if baseline is not None and next_depth + remaining > baseline:
                continue
            key = (next_body, next_heading)
            old = best_seen.get(key)
            if old is not None and old <= (next_depth, next_turns):
                continue
            best_seen[key] = (next_depth, next_turns)
            serial += 1
            heapq.heappush(
                frontier,
                (
                    next_depth + remaining + 0.002 * next_turns,
                    next_depth,
                    next_turns,
                    serial,
                    next_body,
                    next_heading,
                    next_path,
                ),
            )

    if best_goal is None:
        return None
    rank, route, endpoint = best_goal
    return {
        "origin_body": origin_body,
        "origin_heading": origin_heading,
        "food": food,
        "route": route,
        "endpoint": endpoint,
        "baseline": baseline,
        "endpoint_flex": -rank[1],
        "expanded": expanded,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


class _Controller:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.rows = 0
        self.cols = 0
        self.cycles: tuple[tuple[tuple[int, int], ...], ...] = ()
        self.indexes: tuple[dict[tuple[int, int], int], ...] = ()
        self.active: int | None = None
        self.food_epoch: tuple[int, int] | None = None
        self.policy = "BALANCE"
        self.reselect = False

        self.plan: list[str] = []
        self.plan_food: tuple[int, int] | None = None
        self.plan_endpoint: int | None = None
        self.expected: tuple[tuple[tuple[int, int], ...], str, tuple[int, int] | None] | None = None

        self.route_future: Any = None
        self.policy_future: Any = None
        self.pending_route: tuple[Any, ...] | None = None
        self.pending_policy: tuple[Any, ...] | None = None
        self.last_route_start = -100000
        self.last_route_food: tuple[int, int] | None = None
        self.policy_food: tuple[int, int] | None = None
        self.policy_target_tick = -1
        self.metrics: list[dict[str, Any]] = []

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _reset_shape(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        bank = _cycle_bank(rows, cols)
        if not bank:
            spine = _spine_cycle(rows, cols)
            bank = (spine,) if spine is not None else ()
        self.cycles = bank
        self.indexes = tuple(_positions(cycle) for cycle in self.cycles)
        self.active = None
        self.food_epoch = None
        self.policy = "BALANCE"
        self.reselect = False
        self.plan = []
        self.plan_food = None
        self.plan_endpoint = None
        self.expected = None
        self.route_future = None
        self.policy_future = None
        self.pending_route = None
        self.pending_policy = None
        self.metrics = []
        self._log("certificate_bank_ready", rows=rows, cols=cols, certificates=len(self.cycles))

    def _compatible(self, body: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
        return _certificates(body, self.indexes, self.rows * self.cols)

    def _candidate_metrics(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        advertised: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        if food is None:
            return []
        result: list[dict[str, Any]] = []
        candidate_ids = list(range(min(6, len(self.cycles))))
        if self.active is not None and self.active not in candidate_ids:
            candidate_ids.append(self.active)
        for number in candidate_ids:
            cycle, positions = self.cycles[number], self.indexes[number]
            if not _ordered(body, positions, len(cycle)):
                continue
            route = _cycle_food_route(body, heading, food, self.rows, self.cols, cycle, positions, advertised)
            if route is None:
                continue
            actions, endpoint_body, _ = route
            result.append(
                {
                    "id": number,
                    "route": actions,
                    "length": len(actions),
                    "endpoint_flex": len(self._compatible(endpoint_body)),
                    "endpoint_area": _reachable_area(endpoint_body, self.rows, self.cols),
                    "keep": int(number == self.active),
                }
            )
        result.sort(key=lambda item: (item["length"], -item["endpoint_flex"], -item["endpoint_area"], -item["keep"], item["id"]))
        return result

    def _select_certificate(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        advertised: tuple[str, ...],
        tick: int,
    ) -> str | None:
        metrics = self._candidate_metrics(body, heading, food, advertised)
        self.metrics = metrics
        if not metrics:
            self.active = None
            return None
        shortest = metrics[0]["length"]
        current = next((item for item in metrics if item["id"] == self.active), None)
        if current is not None:
            # Keep the food epoch on its already-proven certificate.  Switching
            # a tour merely because another tour has a different tie-break is
            # the portfolio oscillation that hurt 0010/0014; the worker is the
            # mechanism allowed to change certificate at a checked endpoint.
            chosen = current
        elif self.active is None:
            # Match the safe baseline's initial preference for heading
            # continuity, not an arbitrary shortest-tour switch.
            chosen = max(
                metrics,
                key=lambda item: (
                    int(item["route"] and item["route"][0] == heading),
                    int(item["keep"]),
                    -item["id"],
                ),
            )
        elif self.policy == "FLEX":
            eligible = [item for item in metrics if item["length"] <= shortest + 1]
            chosen = max(eligible, key=lambda item: (item["endpoint_flex"], item["endpoint_area"], -item["length"], item["id"]))
        elif self.policy == "SPEED":
            chosen = min(metrics, key=lambda item: (item["length"], -item["endpoint_flex"], -item["endpoint_area"], item["id"]))
        else:
            chosen = min(metrics, key=lambda item: (item["length"], -item["endpoint_flex"], -item["endpoint_area"], item["id"]))
        old = self.active
        self.active = int(chosen["id"])
        if old != self.active:
            self._log("food_certificate_selected", tick=tick, old=old, new=self.active, policy=self.policy, candidates=len(metrics), route=chosen["length"])
        return chosen["route"][0] if chosen["route"] else None

    def _start_policy(self, food: tuple[int, int], tick: int) -> None:
        if self.policy_future is not None or self.ctx.done or len(self.metrics) < 2:
            return
        shortlist = self.metrics[: min(4, len(self.metrics))]
        state = {
            "purpose": "choose a durable Snake food-policy tie-break; do not return an action",
            "food_epoch": {"tick": int(tick), "target_present": True},
            "current_policy": self.policy,
            "candidates": [
                {
                    "certificate": int(item["id"]),
                    "route_steps": _bucket(item["length"]),
                    "post_food_certificates": _bucket(item["endpoint_flex"]),
                    "post_food_open_area": _bucket(item["endpoint_area"]),
                    "is_current_certificate": bool(item["keep"]),
                }
                for item in shortlist
            ],
        }
        criteria = {
            "SPEED": "Prefer the candidate with the fewest verified steps to the visible food; use post-food flexibility only as a tie break.",
            "BALANCE": "Prefer a verified short route while retaining a candidate with more post-food certificates or open area when the route lengths are close.",
            "FLEX": "Prefer a verified route within one step of the shortest when it leaves materially more post-food certificates or open area.",
        }
        reversed_criteria = {key: criteria[key] for key in reversed(POLICIES)}
        questions = {
            "policy_forward": Choice(
                "Choose one durable policy for this food epoch from `SPEED`, `BALANCE`, or `FLEX`. Code has already proven all candidate transitions; do not output a direction or certificate id.",
                criteria,
            ),
            "policy_reversed": Choice(
                "Independently choose the same durable policy from the supplied facts. Return exactly one of the three policy labels; do not invent an action.",
                reversed_criteria,
            ),
        }
        try:
            self.policy_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.55)
            )
            self.policy_food = food
            self.policy_target_tick = tick
            self._log("policy_judgment_started", tick=tick, food=list(food), candidates=len(shortlist))
        except Exception as exc:
            self.policy_future = None
            self._log("policy_judgment_spawn_failed", tick=tick, reason=type(exc).__name__)

    @staticmethod
    def _answer(answer: Any) -> tuple[str | None, float]:
        value = getattr(answer, "value", None)
        if not isinstance(value, str):
            return None, 0.0
        try:
            probabilities = getattr(answer, "probabilities", {})
            probability = float(probabilities.get(value, 0.0)) if isinstance(probabilities, dict) else 0.0
        except (TypeError, ValueError, OverflowError):
            probability = 0.0
        return value if value in POLICIES else None, probability if 0.0 <= probability <= 1.0 else 0.0

    def _poll_policy(self, food: tuple[int, int] | None, tick: int) -> None:
        future = self.policy_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.policy_future = None
        try:
            answers = future.result()
            first = answers.get("policy_forward") if isinstance(answers, dict) else None
            second = answers.get("policy_reversed") if isinstance(answers, dict) else None
            first_value, first_probability = self._answer(first)
            second_value, second_probability = self._answer(second)
            if (
                food is not None
                and food == self.policy_food
                and first_value is not None
                and first_value == second_value
                and min(first_probability, second_probability) >= 0.55
            ):
                self.policy = first_value
                self.reselect = True
                self._log("policy_judgment_applied", tick=tick, policy=self.policy, probability=min(first_probability, second_probability))
            else:
                self._log("policy_judgment_rejected", tick=tick, reason="disagreement_low_confidence_or_stale")
        except Exception as exc:
            self._log("policy_judgment_ignored", tick=tick, reason=type(exc).__name__)

    def _poll_route(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        tick: int,
    ) -> None:
        future = self.route_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.route_future = None
        try:
            result = future.result()
        except Exception as exc:
            self._log("dynamic_route_ignored", tick=tick, reason=type(exc).__name__)
            return
        if not isinstance(result, dict):
            return
        baseline = result.get("baseline")
        route = result.get("route")
        flex = result.get("endpoint_flex")
        if not isinstance(route, tuple) or not route or not isinstance(flex, int):
            return
        if isinstance(baseline, int):
            if len(route) > baseline or (len(route) == baseline and flex <= 0):
                self._log("dynamic_route_discarded", tick=tick, reason="not_better_than_epoch_route")
                return
        matched = _route_suffix(result, body, heading, food, self.rows, self.cols, self.indexes)
        if matched is None:
            self._log("dynamic_route_discarded", tick=tick, reason="prefix_or_certificate_mismatch")
            return
        suffix, endpoint = matched
        if not suffix:
            return
        self.plan = list(suffix)
        self.plan_food = food
        self.plan_endpoint = endpoint
        self.expected = None
        self._log("dynamic_route_ready", tick=tick, remaining=len(self.plan), endpoint=endpoint, expanded=result.get("expanded", 0), elapsed_ms=result.get("elapsed_ms", 0))

    def _launch_route(self) -> None:
        pending = self.pending_route
        self.pending_route = None
        if pending is None or self.route_future is not None or self.plan or self.ctx.done:
            return
        (
            origin_body,
            origin_heading,
            food,
            tick,
            baseline,
            baseline_flex,
        ) = pending
        if not isinstance(food, tuple) or tick <= self.last_route_start:
            return
        # A worker normally spans part of a tick.  Do not replace a still
        # useful deterministic certificate with a new search at every tick;
        # two-tick launches leave CPU for the clock and preserve a fresh
        # post-action origin when the result arrives.
        if (
            food == self.last_route_food
            and tick - self.last_route_start < 2
        ):
            return
        snapshot = (
            origin_body,
            origin_heading,
            food,
            self.rows,
            self.cols,
            self.cycles,
            self.indexes,
            baseline,
            baseline_flex,
            lambda: bool(getattr(self.ctx, "done", False)),
            self.active,
            self.policy,
        )
        try:
            self.route_future = self.ctx.spawn(lambda: _dynamic_route(*snapshot))
            self.last_route_start = tick
            self.last_route_food = food
            self._log("dynamic_route_started", tick=tick, food=list(food), baseline=baseline)
        except Exception as exc:
            self.route_future = None
            self._log("dynamic_route_spawn_failed", tick=tick, reason=type(exc).__name__)

    def _launch_policy(self) -> None:
        pending = self.pending_policy
        self.pending_policy = None
        if pending is None or self.policy_future is not None or self.ctx.done:
            return
        food, tick = pending
        self._start_policy(food, tick)

    def launch_background(self) -> None:
        # Route and policy inference are both off the clock-facing thread.
        self._launch_route()
        self._launch_policy()

    def _queue_after_action(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        action: str,
        tick: int,
    ) -> None:
        if food is None or self.plan or self.route_future is not None or action not in VECTORS:
            return
        step = _advance(body, heading, action, food, self.rows, self.cols)
        if step is None or step[2]:
            return
        next_body, next_heading, _ = step
        certificates = self._compatible(next_body)
        if not certificates:
            return
        # The route worker reconstructs the conservative baseline after the
        # current action is posted.  The foreground only performs this cheap
        # certificate gate, never a full food route simulation.
        self.pending_route = (next_body, next_heading, food, tick + 1, None, 0)

    def choose(self, observation: Any, advertised: tuple[str, ...]) -> str:
        dimensions = _shape(observation)
        body = _body(observation)
        if dimensions is None or not body:
            return advertised[0] if advertised else "NOOP"
        self._reset_shape(*dimensions)
        heading = _heading(observation, body)
        food = _food(observation)
        tick = _tick(observation)

        self._poll_route(body, heading, food, tick)
        self._poll_policy(food, tick)

        if self.plan_food != food:
            self.plan = []
            self.plan_food = None
            self.plan_endpoint = None
            self.expected = None

        if self.expected is not None:
            expected_body, expected_heading, expected_food = self.expected
            if body != expected_body or heading != expected_heading or food != expected_food:
                self._log("dynamic_route_discarded", tick=tick, reason="expected_state_mismatch")
                self.plan = []
                self.plan_food = None
                self.plan_endpoint = None
            self.expected = None

        new_food = food != self.food_epoch
        if new_food:
            self.food_epoch = food
            self.plan = []
            self.plan_food = None
            self.plan_endpoint = None
            self.reselect = False
            self.pending_route = None
            self.pending_policy = None
            if food is not None:
                self._select_certificate(body, heading, food, advertised, tick)
                self.pending_policy = (food, tick)
        elif self.reselect and food is not None and not self.plan:
            # Jev changes only a durable policy; code recomputes the cycle from
            # the latest exact observation before allowing it to affect a move.
            self.reselect = False
            self._select_certificate(body, heading, food, advertised, tick)

        # A planned action is accepted only after exact transition and a fresh
        # per-state certificate check.  The endpoint certificate is adopted on
        # the food transition, never before it.
        if self.plan and self.plan_food == food:
            action = self.plan[0]
            step = _advance(body, heading, action, food, self.rows, self.cols)
            valid = step is not None and action in advertised
            if valid and step is not None:
                if step[2]:
                    valid = self.plan_endpoint in self._compatible(step[0])
                else:
                    valid = bool(self._compatible(step[0]))
            if valid and step is not None:
                self.plan.pop(0)
                if step[2]:
                    self.active = self.plan_endpoint
                    self.plan = []
                    self.plan_food = None
                    self.plan_endpoint = None
                    self.expected = None
                else:
                    self.expected = (step[0], step[1], food)
                return action
            self._log("dynamic_route_discarded", tick=tick, reason="live_transition_rejected")
            self.plan = []
            self.plan_food = None
            self.plan_endpoint = None
            self.expected = None

        # A retained certificate is trusted only after checking the complete
        # current body.  If a certificate was lost, choose another compatible
        # one at this exact state rather than following stale cycle memory.
        if self.active is not None and self.active not in self._compatible(body):
            self.active = None
        action: str | None = None
        if self.active is not None:
            action = _cycle_move(
                body,
                heading,
                food,
                self.rows,
                self.cols,
                self.cycles[self.active],
                self.indexes[self.active],
                advertised,
            )
        if action not in advertised and food is not None:
            action = self._select_certificate(body, heading, food, advertised, tick)
        if action not in advertised:
            action = _safe_fallback(body, heading, food, self.rows, self.cols, advertised, self.indexes)
        if action not in advertised:
            action = heading if heading in advertised else (advertised[0] if advertised else "NOOP")

        self._queue_after_action(body, heading, food, action, tick)
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
            advertised = (noop,) if isinstance(noop, str) and noop in names else names
        if not advertised:
            return

        controller = _Controller(self.ctx)
        while not self.ctx.done and not getattr(game, "done", False):
            try:
                observation = game.observe()
                action = controller.choose(observation, advertised)
            except Exception as exc:
                controller._log("controller_recovered", reason=type(exc).__name__)
                action = advertised[0]
            if action not in advertised:
                action = advertised[0]
            try:
                self.ctx.act(action)
            except Exception:
                return
            if self.ctx.done or getattr(game, "done", False):
                return
            try:
                controller.launch_background()
            except Exception as exc:
                controller._log("background_launch_ignored", reason=type(exc).__name__)
            if self.ctx.done or getattr(game, "done", False):
                return
            try:
                self.ctx.wait_for_tick()
            except Exception:
                return


def build(ctx: Any) -> Harness:
    return Harness(ctx)
