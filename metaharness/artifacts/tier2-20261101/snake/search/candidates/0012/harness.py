"""Candidate 0012: per-food cycle switching with verified cycle re-entry.

The controller keeps a Hamiltonian-cycle fallback, but does not require a food
route to stay on the currently active cycle.  A bounded background search may
leave that cycle while the food is known.  It is executable only when the whole
route is simulated with exact tail-vacating rules *and* the post-growth body is
ordered on a validated cycle, so the next tick can return to a proven invariant.

The active cycle is selected when food changes, rather than being changed on
every tick.  This gives a food-specific cyclic order without the oscillation of
 a per-tick cycle portfolio.  No model is put in the 100 ms control path.
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Any


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
VEC = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


# ---------------------------------------------------------------------------
# Defensive observation and exact one-step mechanics


def _point(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _body(obs: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(obs, dict):
        return ()
    raw = obs.get("snake")
    if not isinstance(raw, (list, tuple)):
        return ()
    result: list[tuple[int, int]] = []
    for value in raw:
        cell = _point(value)
        if cell is None:
            return ()
        result.append(cell)
    return tuple(result)


def _food(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict) or obs.get("food") is None:
        return None
    return _point(obs.get("food"))


def _dimensions(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict):
        return None
    try:
        rows = int(obs.get("rows"))
        cols = int(obs.get("cols"))
    except (TypeError, ValueError, OverflowError):
        return None
    if rows < 2 or cols < 2 or rows * cols > 4096:
        return None
    return rows, cols


def _tick(obs: Any) -> int:
    if not isinstance(obs, dict):
        return 0
    try:
        return max(0, int(obs.get("tick", 0)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _heading(obs: Any, body: tuple[tuple[int, int], ...]) -> str:
    if isinstance(obs, dict) and obs.get("direction") in VEC:
        return obs["direction"]
    if len(body) >= 2:
        delta = body[0][0] - body[1][0], body[0][1] - body[1][1]
        for name, vector in VEC.items():
            if delta == vector:
                return name
    return "RIGHT"


def _inside(cell: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _transition(
    body: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """Simulate one public transition, including the food/tail distinction."""
    if not body or heading not in VEC or request not in VEC:
        return None
    accepted = heading if request == OPPOSITE.get(heading) else request
    dr, dc = VEC[accepted]
    new_head = body[0][0] + dr, body[0][1] + dc
    if not _inside(new_head, rows, cols):
        return None
    ate = food is not None and new_head == food
    occupied = set(body if ate else body[:-1])
    if new_head in occupied:
        return None
    return (new_head,) + (body if ate else body[:-1]), accepted, ate


def _area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not body:
        return 0
    blocked = set(body[1:])
    queue = deque([body[0]])
    seen = {body[0]}
    while queue:
        row, col = queue.popleft()
        for dr, dc in VEC.values():
            cell = row + dr, col + dc
            if _inside(cell, rows, cols) and cell not in blocked and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return len(seen)


# ---------------------------------------------------------------------------
# Validated cycle families


def _valid_cycle(
    cycle: tuple[tuple[int, int], ...], rows: int, cols: int
) -> bool:
    size = rows * cols
    return (
        len(cycle) == size
        and len(set(cycle)) == size
        and all(_inside(cell, rows, cols) for cell in cycle)
        and all(_adjacent(cycle[i], cycle[(i + 1) % size]) for i in range(size))
    )


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
    """Accept a strip splice only if its graph is one validated cycle."""
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
        if len(path) > size:
            return None
    candidate = tuple(path)
    return candidate if _valid_cycle(candidate, rows, cols) else None


def _row_cycle(rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """The boundary-spine cycle for a rectangle with an even row count."""
    if rows < 2 or cols < 2 or rows % 2:
        return None
    path: list[tuple[int, int]] = [(0, col) for col in range(cols)]
    for row in range(1, rows):
        columns = range(cols - 1, 0, -1) if row % 2 else range(1, cols)
        path.extend((row, col) for col in columns)
    path.append((rows - 1, 0))
    path.extend((row, 0) for row in range(rows - 2, 0, -1))
    result = tuple(path)
    return result if _valid_cycle(result, rows, cols) else None


def _strip_cycle(
    rows: int, cols: int, seams: tuple[int, ...]
) -> tuple[tuple[int, int], ...] | None:
    """Join two-row rings, then verify the resulting graph rather than trusting it."""
    if rows < 2 or cols < 2 or rows % 2:
        return None
    strips = rows // 2
    if len(seams) != max(0, strips - 1):
        return None
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for row in range(0, rows, 2):
        perimeter = (
            [(row, col) for col in range(cols)]
            + [(row + 1, col) for col in range(cols - 1, -1, -1)]
        )
        for i, cell in enumerate(perimeter):
            _add_edge(graph, cell, perimeter[(i + 1) % len(perimeter)])
    for strip in range(strips - 1):
        seam = seams[strip]
        if seam < 0 or seam >= cols - 1:
            return None
        upper = 2 * strip + 1
        lower = upper + 1
        _remove_edge(graph, (upper, seam), (upper, seam + 1))
        _remove_edge(graph, (lower, seam), (lower, seam + 1))
        _add_edge(graph, (upper, seam), (lower, seam))
        _add_edge(graph, (upper, seam + 1), (lower, seam + 1))
    return _walk_cycle(graph, rows, cols)


def _pattern_bank(strip_count: int, width: int, count: int, seed: int) -> list[tuple[int, ...]]:
    """Small deterministic seam bank; it is not tied to any evaluation seed."""
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

    # A local, deterministic generator gives varied seams without relying on
    # the episode seed or on a large random library call at tick zero.
    state = (seed | 1) & 0x7FFFFFFF
    while len(result) < count:
        values = []
        for _ in range(strip_count):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            values.append(state % width)
        add(values)
    return result[:count]


def _cycle_family(
    rows: int, cols: int, pattern_count: int = 20, limit: int = 192
) -> list[tuple[tuple[int, int], ...]]:
    """Build and independently validate a diverse, bounded cycle portfolio."""
    if rows < 2 or cols < 2 or (rows % 2 and cols % 2):
        return []
    bases: list[tuple[tuple[int, int], ...]] = []

    direct = _row_cycle(rows, cols)
    if direct is not None:
        bases.append(direct)
    if cols % 2 == 0:
        transposed = _row_cycle(cols, rows)
        if transposed is not None:
            bases.append(tuple((col, row) for row, col in transposed))

    # Interleave the two orientations so a bounded prefix contains both
    # row-wise and column-wise seam families.
    direct_patterns = _pattern_bank(rows // 2 - 1, cols - 1, pattern_count, 1729)
    transposed_patterns = _pattern_bank(cols // 2 - 1, rows - 1, pattern_count, 2718)
    for direct_pattern, transposed_pattern in zip(direct_patterns, transposed_patterns):
        candidate = _strip_cycle(rows, cols, direct_pattern)
        if candidate is not None:
            bases.append(candidate)
        if cols % 2 == 0:
            candidate = _strip_cycle(cols, rows, transposed_pattern)
            if candidate is not None:
                bases.append(tuple((col, row) for row, col in candidate))

    result: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in bases:
        for flip_row in (False, True):
            for flip_col in (False, True):
                reflected = tuple(
                    (
                        rows - 1 - row if flip_row else row,
                        cols - 1 - col if flip_col else col,
                    )
                    for row, col in base
                )
                for oriented in (reflected, tuple(reversed(reflected))):
                    if oriented in seen:
                        continue
                    if not _valid_cycle(oriented, rows, cols):
                        continue
                    seen.add(oriented)
                    result.append(oriented)
                    if len(result) >= limit:
                        return result
    return result


def _ordered(
    body: tuple[tuple[int, int], ...],
    cycle: tuple[tuple[int, int], ...],
    index: dict[tuple[int, int], int],
) -> bool:
    """Check strict head-to-tail cyclic order, allowing gaps from shortcuts."""
    if not body or len(set(body)) != len(body):
        return False
    if any(cell not in index for cell in body):
        return False
    size = len(cycle)
    span = 0
    for newer, older in zip(body, body[1:]):
        gap = (index[newer] - index[older]) % size
        if gap <= 0:
            return False
        span += gap
    return span < size


def _cycle_action(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    index: dict[tuple[int, int], int],
) -> str | None:
    """Pick a one-step cycle-safe move, including a checked food shortcut."""
    if not _ordered(body, cycle, index):
        return None
    size = len(cycle)
    next_cell = cycle[(index[body[0]] + 1) % size]
    choices: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action == OPPOSITE.get(heading):
            continue
        step = _transition(body, heading, action, food, rows, cols)
        if step is None:
            continue
        new_body, new_heading, ate = step
        if not _ordered(new_body, cycle, index):
            continue
        if ate or food is None:
            forward = 0
        else:
            forward = (index[food] - index[new_body[0]]) % size
        rank = (
            int(ate),
            -forward,
            int(new_body[0] == next_cell),
            int(new_heading == heading),
            action,
        )
        choices.append((rank, action))
    if not choices:
        return None
    choices.sort(reverse=True)
    return choices[0][1]


def _greedy_cycle_route(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    index: dict[tuple[int, int], int],
    first_action: str,
) -> tuple[str, ...] | None:
    """A conservative route length used as the no-regression baseline."""
    if not _ordered(body, cycle, index):
        return None
    path: list[str] = []
    limit = rows * cols + len(body) + 4
    current_body, current_heading = body, heading
    for _ in range(limit):
        action = _cycle_action(
            current_body, current_heading, food, rows, cols, cycle, index
        )
        if not path:
            action = first_action
        if action is None:
            return None
        step = _transition(current_body, current_heading, action, food, rows, cols)
        if step is None or not _ordered(step[0], cycle, index):
            return None
        path.append(action)
        current_body, current_heading, ate = step
        if ate:
            return tuple(path)
    return None


def _first_reentry_cycle(
    body: tuple[tuple[int, int], ...],
    pairs: tuple[tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], int]], ...],
) -> tuple[tuple[int, int], ...] | None:
    for cycle, index in pairs:
        if _ordered(body, cycle, index):
            return cycle
    return None


# ---------------------------------------------------------------------------
# Background search: leave the active cycle, then prove re-entry at growth


def _search_on_active_cycle(
    origin_body: tuple[tuple[int, int], ...],
    origin_heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    first_action: str,
    baseline_len: int | None,
    cycle: tuple[tuple[int, int], ...],
    index: dict[tuple[int, int], int],
    deadline: float,
    stop: Any = None,
) -> dict[str, Any] | None:
    """Recover 0009's fast invariant-preserving route before relaxing it."""
    if not _ordered(origin_body, cycle, index):
        return None
    max_depth = min(rows * cols + len(origin_body) + 8, 240)
    if baseline_len is not None:
        max_depth = min(max_depth, baseline_len - 1)
    if max_depth <= 0:
        return None

    def stopped() -> bool:
        if stop is None:
            return False
        try:
            return bool(stop())
        except Exception:
            return False

    serial = 0
    start_distance = abs(origin_body[0][0] - food[0]) + abs(origin_body[0][1] - food[1])
    heap: list[tuple[float, int, int, int, tuple[tuple[int, int], ...], str, tuple[str, ...]]] = [
        (float(start_distance), 0, 0, serial, origin_body, origin_heading, ())
    ]
    best: dict[tuple[tuple[tuple[int, int], ...], str], tuple[int, int]] = {
        (origin_body, origin_heading): (0, 0)
    }
    expanded = 0
    while heap and expanded < 7000 and time.monotonic() < deadline and not stopped():
        _, depth, turns, _, body, heading, path = heapq.heappop(heap)
        expanded += 1
        if depth >= max_depth:
            continue
        if depth == 0:
            actions = (first_action,)
        else:
            ranked: list[tuple[int, int, str]] = []
            for action in DIRECTIONS:
                if action == OPPOSITE.get(heading):
                    continue
                dr, dc = VEC[action]
                nr, nc = body[0][0] + dr, body[0][1] + dc
                ranked.append((abs(nr - food[0]) + abs(nc - food[1]), int(action != heading), action))
            ranked.sort()
            actions = tuple(action for _, _, action in ranked)
        for action in actions:
            if stopped():
                return None
            if action not in VEC or action == OPPOSITE.get(heading):
                continue
            step = _transition(body, heading, action, food, rows, cols)
            if step is None:
                continue
            new_body, new_heading, ate = step
            if not _ordered(new_body, cycle, index):
                continue
            new_path = path + (action,)
            new_depth = depth + 1
            if ate:
                return {
                    "origin_body": origin_body,
                    "origin_heading": origin_heading,
                    "origin_food": food,
                    "route": new_path,
                    "end_cycle": cycle,
                    "baseline_len": baseline_len,
                    "expanded": expanded,
                }
            remaining = abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
            if baseline_len is not None and new_depth + remaining >= baseline_len:
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
                (new_depth + remaining + 0.002 * new_turns, new_depth, new_turns,
                 serial, new_body, new_heading, new_path),
            )
    return None


def _search_food_with_reentry(
    origin_body: tuple[tuple[int, int], ...],
    origin_heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    first_action: str,
    baseline_len: int | None,
    entry_pairs: tuple[tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], int]], ...],
    active_cycle: tuple[tuple[int, int], ...] | None = None,
    active_index: dict[tuple[int, int], int] | None = None,
    stop: Any = None,
) -> dict[str, Any] | None:
    """Find a short exact route whose food endpoint is cycle-ordered.

    The search does not assume the current cycle remains valid during the
    approach.  Every intermediate state still goes through the exact public
    transition, and the only accepted goal is a post-growth state for which a
    fully validated cycle exists.  A Manhattan A* key gives a useful short
    route under the worker deadline; it is not used as a safety proof.
    """
    if not origin_body or food in origin_body or first_action not in VEC:
        return None
    if baseline_len is not None and baseline_len <= 1:
        return None

    # First spend a small slice on the invariant-preserving planner.  This
    # keeps the strong 0009 fallback when the more permissive search is busy.
    deadline = time.monotonic() + 0.070
    if active_cycle is not None and active_index is not None:
        recovered = _search_on_active_cycle(
            origin_body, origin_heading, food, rows, cols, first_action,
            baseline_len, active_cycle, active_index,
            min(deadline, time.monotonic() + 0.045), stop,
        )
        if recovered is not None:
            return recovered

    max_depth = min(rows * cols + len(origin_body) + 8, 260)
    if baseline_len is not None:
        # Equal-length routes are worth considering when they re-enter a
        # different cycle: they can make the *next* food visible sooner.
        max_depth = min(max_depth, baseline_len)
    if max_depth <= 0:
        return None

    serial = 0
    start_distance = abs(origin_body[0][0] - food[0]) + abs(origin_body[0][1] - food[1])
    heap: list[
        tuple[
            float,
            int,
            int,
            int,
            tuple[tuple[int, int], ...],
            str,
            tuple[str, ...],
        ]
    ] = [(float(start_distance), 0, 0, serial, origin_body, origin_heading, ())]
    best: dict[tuple[tuple[tuple[int, int], ...], str], tuple[int, int]] = {
        (origin_body, origin_heading): (0, 0)
    }
    expanded = 0

    def stopped() -> bool:
        if stop is None:
            return False
        try:
            return bool(stop())
        except Exception:
            return False

    while heap and expanded < 12000 and time.monotonic() < deadline and not stopped():
        _, depth, turns, _, body, heading, path = heapq.heappop(heap)
        expanded += 1
        if depth >= max_depth:
            continue
        if depth == 0:
            actions = (first_action,)
        else:
            scored: list[tuple[int, int, str]] = []
            for action in DIRECTIONS:
                if action == OPPOSITE.get(heading):
                    continue
                dr, dc = VEC[action]
                nr, nc = body[0][0] + dr, body[0][1] + dc
                scored.append((abs(nr - food[0]) + abs(nc - food[1]), int(action != heading), action))
            scored.sort()
            actions = tuple(action for _, _, action in scored)

        for action in actions:
            if stopped():
                return None
            if action not in VEC or action == OPPOSITE.get(heading):
                continue
            step = _transition(body, heading, action, food, rows, cols)
            if step is None:
                continue
            new_body, new_heading, ate = step
            new_path = path + (action,)
            new_depth = depth + 1
            if ate:
                endpoint_cycle = _first_reentry_cycle(new_body, entry_pairs)
                if endpoint_cycle is not None:
                    return {
                        "origin_body": origin_body,
                        "origin_heading": origin_heading,
                        "origin_food": food,
                        "route": new_path,
                        "end_cycle": endpoint_cycle,
                        "baseline_len": baseline_len,
                        "expanded": expanded,
                    }
                continue

            # A route longer than the known fallback cannot win.  Equal
            # length is allowed when it re-enters another cycle.  Manhattan
            # is a lower bound on the remaining number of moves.
            remaining_distance = abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
            if baseline_len is not None and new_depth + remaining_distance > baseline_len:
                continue
            key = (new_body, new_heading)
            old = best.get(key)
            new_turns = turns + int(new_heading != heading)
            if old is not None and old <= (new_depth, new_turns):
                continue
            best[key] = (new_depth, new_turns)
            serial += 1
            priority = new_depth + remaining_distance + 0.002 * new_turns
            heapq.heappush(
                heap,
                (priority, new_depth, new_turns, serial, new_body, new_heading, new_path),
            )
    return None


def _route_suffix(
    result: dict[str, Any],
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[str, ...] | None:
    """Match a worker result only at an exact prefix of its original state."""
    if not isinstance(result, dict) or result.get("origin_food") != food:
        return None
    origin_body = result.get("origin_body")
    origin_heading = result.get("origin_heading")
    route = result.get("route")
    if not isinstance(origin_body, tuple) or not isinstance(route, tuple) or not route:
        return None
    current_body, current_heading = origin_body, origin_heading
    for i, action in enumerate(route):
        if current_body == body and current_heading == heading:
            return route[i:]
        step = _transition(current_body, current_heading, action, food, rows, cols)
        if step is None:
            return None
        current_body, current_heading, ate = step
        if ate:
            return None
    if current_body == body and current_heading == heading:
        return ()
    return None


# ---------------------------------------------------------------------------
# Controller and real-time bridge


class _Controller:
    SEARCH_S = 0.070

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.rows = 0
        self.cols = 0
        self.entry_pairs: tuple[tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], int]], ...] = ()
        self.safe_pairs: list[tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], int]]] = []
        self.active_cycle: tuple[tuple[int, int], ...] | None = None
        self.active_index: dict[tuple[int, int], int] | None = None
        self.seen_food: tuple[int, int] | None = None

        self.future: Any = None
        self.plan: list[str] = []
        self.plan_food: tuple[int, int] | None = None
        self.plan_end_cycle: tuple[tuple[int, int], ...] | None = None
        self.expected: tuple[tuple[tuple[int, int], ...], str, tuple[int, int] | None] | None = None
        self.last_started_tick = -100000

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _set_dimensions(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        cycles = _cycle_family(rows, cols, pattern_count=20, limit=192)
        self.entry_pairs = tuple((cycle, {cell: i for i, cell in enumerate(cycle)}) for cycle in cycles)
        self.safe_pairs = list(self.entry_pairs[:96])
        self.active_cycle = None
        self.active_index = None
        self.seen_food = None
        self.future = None
        self.plan = []
        self.plan_food = None
        self.plan_end_cycle = None
        self.expected = None
        self._log("cycle_family_ready", rows=rows, cols=cols, cycles=len(cycles))

    def _adopt_cycle(self, cycle: tuple[tuple[int, int], ...] | None) -> None:
        if not isinstance(cycle, tuple) or not cycle:
            return
        for known, index in self.entry_pairs:
            if known == cycle:
                self.active_cycle, self.active_index = known, index
                if not any(known == pair[0] for pair in self.safe_pairs):
                    self.safe_pairs.insert(0, pair_for(known, index))
                return

    def _select_cycle_action(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        advertised: tuple[str, ...],
    ) -> str | None:
        """Choose a cycle once per food, scoring the food's forward distance."""
        pairs = list(self.safe_pairs)
        if self.active_cycle is not None and self.active_index is not None:
            if not any(pair[0] == self.active_cycle for pair in pairs):
                pairs.insert(0, (self.active_cycle, self.active_index))
        options: list[tuple[tuple[int, int, int, int, str], str, tuple[tuple[int, int], ...], dict[tuple[int, int], int]]] = []
        for number, (cycle, index) in enumerate(pairs):
            action = _cycle_action(body, heading, food, self.rows, self.cols, cycle, index)
            if action is None or action not in advertised:
                continue
            step = _transition(body, heading, action, food, self.rows, self.cols)
            if step is None:
                continue
            new_body, new_heading, ate = step
            if ate or food is None:
                forward = 0
            else:
                forward = (index[food] - index[new_body[0]]) % len(cycle)
            rank = (
                int(ate),
                -forward,
                int(cycle == self.active_cycle),
                int(new_heading == heading),
                action,
            )
            options.append((rank, action, cycle, index))
        if not options:
            self.active_cycle = None
            self.active_index = None
            return None
        options.sort(reverse=True)
        _, action, cycle, index = options[0]
        changed = cycle != self.active_cycle
        self.active_cycle, self.active_index = cycle, index
        if changed:
            self._log("food_cycle_selected", tick=0, action=action, candidates=len(options))
        return action

    def _clear_plan(self) -> None:
        self.plan = []
        self.plan_food = None
        self.plan_end_cycle = None
        self.expected = None

    def _poll(self, body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None, tick: int) -> None:
        future = self.future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.future = None
        try:
            result = future.result()
        except Exception as exc:
            self._log("food_search_discarded", tick=tick, reason=type(exc).__name__)
            return
        if not isinstance(result, dict):
            return
        route = result.get("route")
        baseline = result.get("baseline_len")
        if not isinstance(route, tuple) or not route:
            return
        if isinstance(baseline, int) and len(route) > baseline:
            self._log("food_search_discarded", tick=tick, reason="longer_than_cycle", route=len(route), baseline=baseline)
            return
        suffix = _route_suffix(result, body, heading, food, self.rows, self.cols)
        if suffix:
            self.plan = list(suffix)
            self.plan_food = food
            self.plan_end_cycle = result.get("end_cycle") if isinstance(result.get("end_cycle"), tuple) else None
            self._log("food_search_ready", tick=tick, remaining=len(self.plan), expanded=result.get("expanded", 0))
        elif result:
            self._log("food_search_discarded", tick=tick, reason="state_changed")

    def _start_search(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        action: str,
        tick: int,
    ) -> None:
        if food is None or self.future is not None or self.plan:
            return
        if not self.entry_pairs:
            return
        if tick <= self.last_started_tick or action not in VEC:
            return
        baseline: int | None = None
        if self.active_cycle is not None and self.active_index is not None:
            route = _greedy_cycle_route(
                body, heading, food, self.rows, self.cols,
                self.active_cycle, self.active_index, action,
            )
            baseline = len(route) if route is not None else None
        snapshot = (
            body,
            heading,
            food,
            self.rows,
            self.cols,
            action,
            baseline,
            self.entry_pairs,
            self.active_cycle,
            self.active_index,
        )
        try:
            self.future = self.ctx.spawn(
                lambda: _search_food_with_reentry(
                    *snapshot, stop=lambda: bool(getattr(self.ctx, "done", False))
                )
            )
            self.last_started_tick = tick
            self._log("food_search_started", tick=tick, baseline=baseline)
        except Exception as exc:
            self.future = None
            self._log("food_search_spawn_failed", tick=tick, reason=type(exc).__name__)

    def choose(self, obs: Any, advertised: tuple[str, ...]) -> str:
        dimensions = _dimensions(obs)
        body = _body(obs)
        if dimensions is None or not body:
            return advertised[0] if advertised else "NOOP"
        self._set_dimensions(*dimensions)
        heading = _heading(obs, body)
        food = _food(obs)
        tick = _tick(obs)

        self._poll(body, heading, food, tick)
        if self.plan_food != food:
            self._clear_plan()

        if self.expected is not None:
            expected_body, expected_heading, expected_food = self.expected
            if body != expected_body or heading != expected_heading or food != expected_food:
                self._log("food_search_discarded", tick=tick, reason="expected_state_mismatch")
                self._clear_plan()
            else:
                self.expected = None

        if food != self.seen_food:
            self.seen_food = food
            self._clear_plan()
            # A food-specific order is selected only at this boundary.  It is
            # not allowed to oscillate while traversing one food route.
            base_action = self._select_cycle_action(body, heading, food, advertised)
        else:
            base_action = None

        # A route is checked again immediately before every posted action.
        if self.plan and self.plan_food == food:
            candidate = self.plan[0]
            step = _transition(body, heading, candidate, food, self.rows, self.cols)
            if step is not None and candidate in advertised:
                self.plan.pop(0)
                if step[2]:
                    self._adopt_cycle(self.plan_end_cycle)
                    self._clear_plan()
                else:
                    self.expected = (step[0], step[1], food)
                return candidate
            self._clear_plan()

        if base_action is None and self.active_cycle is not None and self.active_index is not None:
            candidate = _cycle_action(
                body, heading, food, self.rows, self.cols,
                self.active_cycle, self.active_index,
            )
            if candidate in advertised:
                base_action = candidate
            else:
                self.active_cycle = None
                self.active_index = None

        if base_action is None:
            base_action = self._select_cycle_action(body, heading, food, advertised)

        if base_action not in advertised:
            candidates: list[tuple[tuple[int, int, int, int, str], str]] = []
            for action in DIRECTIONS:
                if action not in advertised or action == OPPOSITE.get(heading):
                    continue
                step = _transition(body, heading, action, food, self.rows, self.cols)
                if step is None:
                    continue
                new_body, new_heading, ate = step
                candidates.append((
                    (int(ate), _area(new_body, self.rows, self.cols), int(new_heading == heading), action),
                    action,
                ))
            if candidates:
                candidates.sort(reverse=True)
                base_action = candidates[0][1]
            else:
                base_action = heading if heading in advertised else (advertised[0] if advertised else "NOOP")

        # Do not search after an immediately eaten food; the next observation
        # contains a new target and a fresh cycle choice.
        first_step = _transition(body, heading, base_action, food, self.rows, self.cols)
        if first_step is not None and not first_step[2]:
            self._start_search(body, heading, food, base_action, tick)
        return base_action


def pair_for(cycle: tuple[tuple[int, int], ...], index: dict[tuple[int, int], int]):
    return cycle, index


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
            obs = game.observe()
            action = controller.choose(obs, advertised)
            if action not in names:
                noop = getattr(spec, "noop", None)
                action = noop if isinstance(noop, str) and noop in names else advertised[0]
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
