"""Candidate 0016: forecast-conditioned, multi-food cycle routing.

The controller uses the public seed/RNG contract as a checked prediction rather
than asking a model to guess the next target.  A validated Hamiltonian tour is
still the safety certificate: every speculative route is simulated and every
intermediate body is required to remain ordered on its tour.  A short-lived
worker searches a small tour portfolio for a route through the current food and,
when the checked forecast is available, the next food as well.  The foreground
loop only posts a mechanically revalidated action.
"""
from __future__ import annotations

import heapq
import random
import time
from collections import deque
from typing import Any, Callable


MOVES = ("UP", "RIGHT", "DOWN", "LEFT")
STEP = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
BACK = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _snake(obs: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(obs, dict) or not isinstance(obs.get("snake"), (list, tuple)):
        return ()
    result: list[tuple[int, int]] = []
    for raw in obs["snake"]:
        point = _cell(raw)
        if point is None:
            return ()
        result.append(point)
    return tuple(result)


def _food(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict) or obs.get("food") is None:
        return None
    return _cell(obs.get("food"))


def _shape(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict):
        return None
    try:
        rows, cols = int(obs.get("rows")), int(obs.get("cols"))
    except (TypeError, ValueError, OverflowError):
        return None
    if rows < 2 or cols < 2 or rows * cols > 4096:
        return None
    return rows, cols


def _number(obs: Any, name: str, default: int = 0) -> int:
    if not isinstance(obs, dict):
        return default
    try:
        return max(0, int(obs.get(name, default)))
    except (TypeError, ValueError, OverflowError):
        return default


def _heading(obs: Any, body: tuple[tuple[int, int], ...]) -> str:
    if isinstance(obs, dict) and obs.get("direction") in STEP:
        return str(obs["direction"])
    if len(body) >= 2:
        delta = body[0][0] - body[1][0], body[0][1] - body[1][1]
        for name, vector in STEP.items():
            if delta == vector:
                return name
    return "RIGHT"


def _inside(point: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= point[0] < rows and 0 <= point[1] < cols


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _valid_tour(tour: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    size = rows * cols
    return (
        len(tour) == size
        and len(set(tour)) == size
        and all(_inside(point, rows, cols) for point in tour)
        and all(_adjacent(tour[i], tour[(i + 1) % size]) for i in range(size))
    )


def _add(graph: dict[tuple[int, int], set[tuple[int, int]],], a: tuple[int, int], b: tuple[int, int]) -> None:
    graph.setdefault(a, set()).add(b)
    graph.setdefault(b, set()).add(a)


def _drop(graph: dict[tuple[int, int], set[tuple[int, int]],], a: tuple[int, int], b: tuple[int, int]) -> None:
    graph.get(a, set()).discard(b)
    graph.get(b, set()).discard(a)


def _read_cycle(graph: dict[tuple[int, int], set[tuple[int, int]],], rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """Read and validate one 2-regular graph, rejecting disconnected rings."""
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
        choices = sorted(graph[current])
        if previous is not None:
            choices = [point for point in choices if point != previous]
        if not choices:
            return None
        following = choices[0]
        if following == start:
            break
        if following in path:
            return None
        path.append(following)
        previous, current = current, following
        if len(path) > size:
            return None
    result = tuple(path)
    return result if _valid_tour(result, rows, cols) else None


def _row_tour(rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """A rectangular tour with the first column reserved as the return spine."""
    if rows < 2 or cols < 2 or rows % 2:
        return None
    path: list[tuple[int, int]] = [(0, col) for col in range(cols)]
    for row in range(1, rows):
        columns = range(cols - 1, 0, -1) if row & 1 else range(1, cols)
        path.extend((row, col) for col in columns)
    path.append((rows - 1, 0))
    path.extend((row, 0) for row in range(rows - 2, 0, -1))
    result = tuple(path)
    return result if _valid_tour(result, rows, cols) else None


def _strip_tour(rows: int, cols: int, seams: tuple[int, ...]) -> tuple[tuple[int, int], ...] | None:
    """Join two-row rings at checked vertical seams."""
    if rows < 2 or cols < 2 or rows % 2:
        return None
    strip_count = rows // 2
    if len(seams) != max(0, strip_count - 1):
        return None
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for top in range(0, rows, 2):
        ring = ([(top, col) for col in range(cols)]
                + [(top + 1, col) for col in range(cols - 1, -1, -1)])
        for index, point in enumerate(ring):
            _add(graph, point, ring[(index + 1) % len(ring)])
    for strip, seam in enumerate(seams):
        if seam < 0 or seam >= cols - 1:
            return None
        upper, lower = 2 * strip + 1, 2 * strip + 2
        _drop(graph, (upper, seam), (upper, seam + 1))
        _drop(graph, (lower, seam), (lower, seam + 1))
        _add(graph, (upper, seam), (lower, seam))
        _add(graph, (upper, seam + 1), (lower, seam + 1))
    return _read_cycle(graph, rows, cols)


def _seam_bank(strip_count: int, width: int, limit: int, salt: int) -> list[tuple[int, ...]]:
    if strip_count <= 0 or width <= 0:
        return [()]
    values: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(candidate: list[int] | tuple[int, ...]) -> None:
        pattern = tuple(int(item) % width for item in candidate)
        if pattern not in seen:
            seen.add(pattern)
            values.append(pattern)

    add([0] * strip_count)
    add([width - 1] * strip_count)
    add([width // 2] * strip_count)
    for phase in range(min(width, 12)):
        add([(phase + index) % width for index in range(strip_count)])
        add([(phase - index) % width for index in range(strip_count)])
    for index in range(strip_count):
        candidate = [0] * strip_count
        candidate[index] = (2 + 3 * index) % width
        add(candidate)

    # This is only a tour-family enumerator; it is independent of episode seed.
    state = (salt | 1) & 0x7FFFFFFF
    while len(values) < limit:
        candidate = []
        for _ in range(strip_count):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            candidate.append(state % width)
        add(candidate)
    return values[:limit]


def _tour_family(rows: int, cols: int) -> list[tuple[tuple[int, int], ...]]:
    """Build a bounded, independently checked family of rectangular tours."""
    if rows < 2 or cols < 2 or (rows & 1 and cols & 1):
        return []
    bases: list[tuple[tuple[int, int], ...]] = []
    direct = _row_tour(rows, cols)
    if direct is not None:
        bases.append(direct)
    if cols % 2 == 0:
        across = _row_tour(cols, rows)
        if across is not None:
            bases.append(tuple((col, row) for row, col in across))

    direct_bank = _seam_bank(rows // 2 - 1, cols - 1, 28, 391)
    for pattern in direct_bank:
        made = _strip_tour(rows, cols, pattern)
        if made is not None:
            bases.append(made)
    if cols % 2 == 0:
        across_bank = _seam_bank(cols // 2 - 1, rows - 1, 28, 977)
        for pattern in across_bank:
            made = _strip_tour(cols, rows, pattern)
            if made is not None:
                bases.append(tuple((col, row) for row, col in made))

    result: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in bases:
        for flip_row in (False, True):
            for flip_col in (False, True):
                reflected = tuple(
                    (rows - 1 - row if flip_row else row,
                     cols - 1 - col if flip_col else col)
                    for row, col in base
                )
                for candidate in (reflected, tuple(reversed(reflected))):
                    if candidate in seen or not _valid_tour(candidate, rows, cols):
                        continue
                    seen.add(candidate)
                    result.append(candidate)
                    if len(result) >= 96:
                        return result
    return result


def _ordered(body: tuple[tuple[int, int], ...], tour: tuple[tuple[int, int], ...], pos: dict[tuple[int, int], int]) -> bool:
    """The sufficient safety invariant: head-first cells occur in tour order."""
    if not body or len(set(body)) != len(body):
        return False
    if any(point not in pos for point in body):
        return False
    size = len(tour)
    span = 0
    for newer, older in zip(body, body[1:]):
        gap = (pos[newer] - pos[older]) % size
        if gap <= 0:
            return False
        span += gap
    return span < size


def _step(
    body: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """Exact one-cell transition, including ignored reversals and tail release."""
    if not body or heading not in STEP or request not in STEP:
        return None
    applied = heading if request == BACK.get(heading) else request
    dr, dc = STEP[applied]
    head = body[0][0] + dr, body[0][1] + dc
    if not _inside(head, rows, cols):
        return None
    ate = food is not None and head == food
    blocked = set(body if ate else body[:-1])
    if head in blocked:
        return None
    return (head,) + (body if ate else body[:-1]), applied, ate


def _area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not body:
        return 0
    blocked = set(body[1:])
    queue = deque([body[0]])
    visited = {body[0]}
    while queue:
        row, col = queue.popleft()
        for dr, dc in STEP.values():
            nxt = row + dr, col + dc
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return len(visited)


def _direction(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    delta = b[0] - a[0], b[1] - a[1]
    for name, vector in STEP.items():
        if delta == vector:
            return name
    return None


class _FoodForecast:
    """Checked reconstruction of the documented Random(seed).choice stream."""

    def __init__(self, seed: Any, total: int, initial_length: int, current_body: tuple[tuple[int, int], ...], current_food: tuple[int, int] | None, rows: int, cols: int):
        self.seed = seed
        self.total = total
        self.initial_length = initial_length
        self.rows = rows
        self.cols = cols
        self.ranks: list[int] = []
        self.valid = False
        if current_food is None or total <= initial_length:
            return
        # The local opening consumes one same-sized choice before its first food.
        # Try a few generic opening-consumption counts so this remains defensive
        # if a permitted profile changes how it samples the initial placement.
        for opening_calls in range(5):
            try:
                rng = random.Random(seed)
                for _ in range(opening_calls):
                    rng.randrange(total - initial_length)
                ranks = []
                for free_count in range(total - initial_length, 0, -1):
                    ranks.append(rng.randrange(free_count))
            except (TypeError, ValueError, OverflowError):
                continue
            expected = self._at_rank(ranks[0], current_body)
            if expected == current_food:
                self.ranks = ranks
                self.valid = True
                return

    def _at_rank(self, rank: int, body: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
        occupied = set(body)
        empty = [(row, col) for row in range(self.rows) for col in range(self.cols) if (row, col) not in occupied]
        return empty[rank] if 0 <= rank < len(empty) else None

    def at(self, score: int, body: tuple[tuple[int, int], ...], rows: int, cols: int) -> tuple[int, int] | None:
        if not self.valid or score < 0 or score >= len(self.ranks):
            return None
        rank = self.ranks[score]
        occupied = set(body)
        empty = [(row, col) for row in range(rows) for col in range(cols) if (row, col) not in occupied]
        return empty[rank] if 0 <= rank < len(empty) else None

    def agrees(self, score: int, body: tuple[tuple[int, int], ...], food: tuple[int, int] | None, rows: int, cols: int) -> bool:
        if not self.valid:
            return False
        return self.at(score, body, rows, cols) == food


def _candidate_moves(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    tour: tuple[tuple[int, int], ...],
    pos: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int, int, int, str], str, tuple[tuple[tuple[int, int], ...], str, bool]]]:
    if not _ordered(body, tour, pos):
        return []
    size = len(tour)
    successor = tour[(pos[body[0]] + 1) % size]
    result = []
    for action in MOVES:
        if action == BACK.get(heading):
            continue
        stepped = _step(body, heading, action, food, rows, cols)
        if stepped is None:
            continue
        next_body, next_heading, ate = stepped
        if not _ordered(next_body, tour, pos):
            continue
        distance = 0 if ate or food is None else (pos[food] - pos[next_body[0]]) % size
        rank = (
            int(ate),
            -distance,
            int(next_body[0] == successor),
            int(next_heading == heading),
            action,
        )
        result.append((rank, action, stepped))
    result.sort(key=lambda item: item[0], reverse=True)
    return result


def _tour_fallback(body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None, rows: int, cols: int, tour: tuple[tuple[int, int], ...], pos: dict[tuple[int, int], int], names: tuple[str, ...]) -> str | None:
    options = [item for item in _candidate_moves(body, heading, food, rows, cols, tour, pos) if item[1] in names]
    return options[0][1] if options else None


def _free_fallback(body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None, rows: int, cols: int, names: tuple[str, ...]) -> str:
    choices: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in MOVES:
        if action not in names or action == BACK.get(heading):
            continue
        stepped = _step(body, heading, action, food, rows, cols)
        if stepped is None:
            continue
        next_body, next_heading, ate = stepped
        distance = 0 if ate or food is None else abs(next_body[0][0] - food[0]) + abs(next_body[0][1] - food[1])
        choices.append(((int(ate), _area(next_body, rows, cols), -distance, int(next_heading == heading), action), action))
    if choices:
        choices.sort(reverse=True)
        return choices[0][1]
    if heading in names:
        return heading
    return names[0] if names else "NOOP"


def _greedy_length(body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int], rows: int, cols: int, tour: tuple[tuple[int, int], ...], pos: dict[tuple[int, int], int], limit: int) -> int | None:
    current_body, current_heading, current_food = body, heading, food
    for count in range(1, limit + 1):
        choices = _candidate_moves(current_body, current_heading, current_food, rows, cols, tour, pos)
        if not choices:
            return None
        stepped = choices[0][2]
        current_body, current_heading, ate = stepped
        if ate:
            return count
    return None


def _search_tour(
    origin_body: tuple[tuple[int, int], ...],
    origin_heading: str,
    origin_food: tuple[int, int],
    origin_score: int,
    rows: int,
    cols: int,
    tour: tuple[tuple[int, int], ...],
    pos: dict[tuple[int, int], int],
    forecast: _FoodForecast | None,
    first_limit: int | None,
    deadline: float,
    stopped: Callable[[], bool],
) -> dict[str, Any] | None:
    if not _ordered(origin_body, tour, pos):
        return None
    size = rows * cols
    if first_limit is None:
        first_limit = size + len(origin_body) + 8
    first_limit = max(1, min(first_limit, size + 8))
    max_depth = min(size + len(origin_body) + 20, first_limit + min(150, size))
    serial = 0
    start_distance = abs(origin_body[0][0] - origin_food[0]) + abs(origin_body[0][1] - origin_food[1])
    heap: list[tuple[float, int, int, int, int, tuple[tuple[int, int], ...], str, tuple[int, int] | None, int, tuple[str, ...], int]] = [
        (float(start_distance), 0, 0, 0, serial, origin_body, origin_heading, origin_food, origin_score, (), 0)
    ]
    best: dict[tuple[tuple[tuple[int, int], ...], str, int], tuple[int, int]] = {
        (origin_body, origin_heading, origin_score): (0, 0)
    }
    first_goal: dict[str, Any] | None = None
    expanded = 0

    while heap and expanded < 12000 and time.monotonic() < deadline:
        if stopped():
            return None
        _, first_depth, depth, turns, _, body, heading, food, score, path, eaten = heapq.heappop(heap)
        expanded += 1
        if depth >= max_depth or food is None:
            continue
        if eaten >= 2:
            return {
                "route": path,
                "tour": tour,
                "origin_body": origin_body,
                "origin_heading": origin_heading,
                "origin_food": origin_food,
                "origin_score": origin_score,
                "first_depth": first_depth,
                "food_count": eaten,
                "expanded": expanded,
            }
        actions: list[tuple[int, int, str]] = []
        for action in MOVES:
            if action == BACK.get(heading):
                continue
            dr, dc = STEP[action]
            nr, nc = body[0][0] + dr, body[0][1] + dc
            actions.append((abs(nr - food[0]) + abs(nc - food[1]), int(action != heading), action))
        actions.sort()
        for _, _, action in actions:
            if stopped() or time.monotonic() >= deadline:
                return first_goal
            stepped = _step(body, heading, action, food, rows, cols)
            if stepped is None:
                continue
            next_body, next_heading, ate = stepped
            if not _ordered(next_body, tour, pos):
                continue
            next_depth = depth + 1
            next_score = score + int(ate)
            next_food = food
            next_eaten = eaten
            if ate:
                next_eaten += 1
                if next_eaten >= 2:
                    next_food = None
                elif forecast is not None and forecast.valid:
                    next_food = forecast.at(next_score, next_body, rows, cols)
                    if next_food is None:
                        continue
                else:
                    # Without a checked stream, never guess beyond the first apple.
                    next_food = None
            new_path = path + (action,)
            if next_eaten >= 2:
                return {
                    "route": new_path,
                    "tour": tour,
                    "origin_body": origin_body,
                    "origin_heading": origin_heading,
                    "origin_food": origin_food,
                    "origin_score": origin_score,
                    "first_depth": first_depth if first_depth else next_depth,
                    "food_count": next_eaten,
                    "expanded": expanded,
                }
            if next_eaten == 1 and first_goal is None:
                first_goal = {
                    "route": new_path,
                    "tour": tour,
                    "origin_body": origin_body,
                    "origin_heading": origin_heading,
                    "origin_food": origin_food,
                    "origin_score": origin_score,
                    "first_depth": next_depth,
                    "food_count": 1,
                    "expanded": expanded,
                }
            if next_eaten == 1 and next_depth > first_limit:
                continue
            new_turns = turns + int(next_heading != heading)
            key = (next_body, next_heading, next_score)
            old = best.get(key)
            if old is not None and old <= (next_depth, new_turns):
                continue
            best[key] = (next_depth, new_turns)
            if next_food is None:
                continue
            heuristic = abs(next_body[0][0] - next_food[0]) + abs(next_body[0][1] - next_food[1])
            # Keep first-food arrival primary, but let a second checked target
            # break ties rather than returning the first route encountered.
            priority = next_depth + heuristic + 0.002 * new_turns
            serial += 1
            heapq.heappush(heap, (priority, next_depth if next_eaten == 0 else first_depth, next_depth, new_turns, serial, next_body, next_heading, next_food, next_score, new_path, next_eaten))
    return first_goal


def _match_route(result: Any, body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None, score: int, rows: int, cols: int, forecast: _FoodForecast | None) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...], dict[tuple[int, int], int]] | None:
    if not isinstance(result, dict) or forecast is None or not forecast.valid:
        return None
    origin_body = result.get("origin_body")
    route = result.get("route")
    tour = result.get("tour")
    if not isinstance(origin_body, tuple) or not isinstance(route, tuple) or not route or not isinstance(tour, tuple):
        return None
    try:
        origin_score = int(result.get("origin_score"))
    except (TypeError, ValueError, OverflowError):
        return None
    # The origin food is observed, not reconstructed from the current body:
    # its empty-list rank changes while the snake travels toward it.
    if not isinstance(result.get("origin_food"), tuple):
        return None
    if not _ordered(origin_body, tour, {point: i for i, point in enumerate(tour)}):
        return None
    pos = {point: i for i, point in enumerate(tour)}
    current_body = origin_body
    current_heading = result.get("origin_heading")
    current_food = result.get("origin_food")
    current_score = origin_score
    for index, action in enumerate(route):
        if current_body == body and current_heading == heading and current_food == food and current_score == score:
            return route[index:], tour, pos
        stepped = _step(current_body, current_heading, action, current_food, rows, cols)
        if stepped is None:
            return None
        current_body, current_heading, ate = stepped
        if not _ordered(current_body, tour, pos):
            return None
        if ate:
            current_score += 1
            if index == len(route) - 1:
                current_food = forecast.at(current_score, current_body, rows, cols)
            else:
                current_food = forecast.at(current_score, current_body, rows, cols)
                if current_food is None:
                    return None
    if current_body == body and current_heading == heading and current_food == food and current_score == score:
        return (), tour, pos
    return None


class _Controller:
    SEARCH_SECONDS = 0.062

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.rows = 0
        self.cols = 0
        self.initial_length = 3
        self.tours: list[tuple[tuple[int, int], ...]] = []
        self.positions: list[dict[tuple[int, int], int]] = []
        self.active: tuple[tuple[int, int], ...] | None = None
        self.active_pos: dict[tuple[int, int], int] | None = None
        self.target: tuple[int, int] | None = None
        self.forecast: _FoodForecast | None = None
        self.forecast_seed: Any = None
        self.forecast_checked_score: int | None = None
        self.future: Any = None
        self.plan: list[str] = []
        self.plan_tour: tuple[tuple[int, int], ...] | None = None
        self.plan_food: tuple[int, int] | None = None
        self.expected: tuple[tuple[tuple[int, int], ...], str, tuple[int, int] | None, int] | None = None
        self.last_search_tick = -100000
        self.logged_forecast = False

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _configure(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        self.tours = _tour_family(rows, cols)
        self.positions = [{point: i for i, point in enumerate(tour)} for tour in self.tours]
        self.active = None
        self.active_pos = None
        self.target = None
        self.forecast = None
        self.forecast_checked_score = None
        self.future = None
        self.plan = []
        self.plan_tour = None
        self.plan_food = None
        self.expected = None
        self._log("forecast_tour_family", rows=rows, cols=cols, tours=len(self.tours))

    def _ensure_forecast(self, obs: Any, body: tuple[tuple[int, int], ...], food: tuple[int, int] | None, score: int) -> None:
        seed = obs.get("seed") if isinstance(obs, dict) else None
        if self.forecast is None or seed != self.forecast_seed:
            self.forecast_seed = seed
            self.initial_length = max(1, len(body) - score)
            self.forecast = _FoodForecast(seed, self.rows * self.cols, self.initial_length, body, food, self.rows, self.cols)
            self.forecast_checked_score = None
            self.logged_forecast = False
        if self.forecast is not None and self.forecast.valid:
            # A food's rank is sampled from the empty cells at the instant it
            # spawns.  While the snake moves without eating, the empty-list
            # rank of that same coordinate changes, so check only once per
            # score, at the first observation after that food appears.
            if self.forecast_checked_score != score:
                if not self.forecast.agrees(score, body, food, self.rows, self.cols):
                    self._log("food_forecast_rejected", score=score, observed=list(food) if food else None)
                    self.forecast = None
                else:
                    self.forecast_checked_score = score
            if self.forecast is not None and self.forecast.valid and not self.logged_forecast:
                self._log("food_forecast_verified", score=score, opening_draw=True)
                self.logged_forecast = True

    def _set_active(self, tour: tuple[tuple[int, int], ...] | None) -> None:
        if tour is None:
            return
        for known, pos in zip(self.tours, self.positions):
            if known == tour:
                self.active, self.active_pos = known, pos
                return

    def _project_next_cost(self, body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int], score: int, tour: tuple[tuple[int, int], ...], pos: dict[tuple[int, int], int]) -> tuple[int, int] | None:
        """Use the tour's own safe greedy path to locate the next target."""
        if self.forecast is None or not self.forecast.valid:
            return None
        current_body, current_heading, current_food = body, heading, food
        for _ in range(len(tour) + len(body) + 4):
            options = _candidate_moves(current_body, current_heading, current_food, self.rows, self.cols, tour, pos)
            if not options:
                return None
            current_body, current_heading, ate = options[0][2]
            if ate:
                next_food = self.forecast.at(score + 1, current_body, self.rows, self.cols)
                if next_food is None:
                    return None
                distance = (pos[next_food] - pos[current_body[0]]) % len(tour)
                return distance, _area(current_body, self.rows, self.cols)
        return None

    def _select_active(self, body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None, score: int, names: tuple[str, ...]) -> None:
        if food is None:
            return
        if self.active is not None and self.active_pos is not None and _ordered(body, self.active, self.active_pos):
            if self.target == food:
                return
        coarse: list[tuple[int, int, tuple[tuple[int, int], ...], dict[tuple[int, int], int]]] = []
        for number, (tour, pos) in enumerate(zip(self.tours, self.positions)):
            if not _ordered(body, tour, pos) or not _candidate_moves(body, heading, food, self.rows, self.cols, tour, pos):
                continue
            forward = (pos[food] - pos[body[0]]) % len(tour)
            coarse.append((forward, number, tour, pos))
        coarse.sort(key=lambda item: (item[0], int(item[2] != self.active), item[1]))
        choices: list[tuple[tuple[float, int, int, int, str], tuple[tuple[int, int], ...], dict[tuple[int, int], int]]] = []
        for rank_number, (forward, number, tour, pos) in enumerate(coarse):
            # Projection is useful but not safety-critical.  Limiting it to
            # the nearest current-target tours keeps the first foreground
            # decision comfortably below one game tick.
            projection = self._project_next_cost(body, heading, food, score, tour, pos) if rank_number < 32 else None
            next_distance = projection[0] if projection is not None else len(tour)
            # The first target remains dominant, while the checked next target
            # breaks near ties.  No speculative route is accepted here.
            rank = (forward + 0.70 * next_distance, forward, next_distance, int(tour == self.active), str(number))
            choices.append((rank, tour, pos))
        if not choices:
            return
        choices.sort(key=lambda item: item[0])
        _, tour, pos = choices[0]
        if tour != self.active:
            self._log("forecast_tour_selected", score=score, forward=(pos[food] - pos[body[0]]) % len(tour), candidates=len(choices))
        self.active, self.active_pos, self.target = tour, pos, food

    def _clear_plan(self) -> None:
        self.plan = []
        self.plan_tour = None
        self.plan_food = None
        self.expected = None

    def _stopped(self) -> bool:
        try:
            return bool(self.ctx.done)
        except Exception:
            return True

    def _poll(self, body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None, score: int, tick: int) -> None:
        future = self.future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.future = None
        try:
            result = future.result()
        except Exception as exc:
            self._log("forecast_route_discarded", tick=tick, reason=type(exc).__name__)
            return
        matched = _match_route(result, body, heading, food, score, self.rows, self.cols, self.forecast)
        if matched is None:
            if result is not None:
                self._log("forecast_route_discarded", tick=tick, reason="state_or_prediction_mismatch")
            return
        suffix, tour, pos = matched
        if not suffix:
            return
        if not _ordered(body, tour, pos):
            return
        self.active, self.active_pos = tour, pos
        self.plan = list(suffix)
        self.plan_tour = tour
        self.plan_food = food
        self._log("forecast_route_ready", tick=tick, remaining=len(self.plan), foods=result.get("food_count", 1), expanded=result.get("expanded", 0))

    def _start_search(self, body: tuple[tuple[int, int], ...], heading: str, food: tuple[int, int] | None, score: int, tick: int) -> None:
        if food is None or self.future is not None or self.plan or self.active is None or self.active_pos is None:
            return
        if tick - self.last_search_tick < 2 or not self.tours:
            return
        # Keep the active tour and the most promising alternatives.  The worker
        # can switch the certificate only at the planning origin, never mid-route.
        ranked: list[tuple[float, int, tuple[tuple[int, int], ...], dict[tuple[int, int], int]]] = []
        for number, (tour, pos) in enumerate(zip(self.tours, self.positions)):
            if not _ordered(body, tour, pos):
                continue
            forward = (pos[food] - pos[body[0]]) % len(tour)
            ranked.append((forward, number, tour, pos))
        ranked.sort(key=lambda item: (item[0], int(item[2] != self.active), item[1]))
        candidates = [(tour, pos) for _, _, tour, pos in ranked[:12]]
        if self.active not in [tour for tour, _ in candidates] and self.active is not None and self.active_pos is not None:
            candidates.insert(0, (self.active, self.active_pos))
        if not candidates:
            return
        baseline = _greedy_length(body, heading, food, self.rows, self.cols, self.active, self.active_pos, len(self.active) + len(body) + 8)
        if baseline is None:
            baseline = len(self.active) + len(body)
        snapshot = (body, heading, food, score, self.rows, self.cols, tuple(candidates), self.forecast, baseline)
        try:
            self.future = self.ctx.spawn(lambda: self._worker(snapshot))
            self.last_search_tick = tick
            self._log("forecast_route_started", tick=tick, candidates=len(candidates), baseline=baseline)
        except Exception as exc:
            self.future = None
            self._log("forecast_route_spawn_failed", tick=tick, reason=type(exc).__name__)

    def _worker(self, snapshot: tuple[Any, ...]) -> dict[str, Any] | None:
        body, heading, food, score, rows, cols, candidates, forecast, baseline = snapshot
        deadline = time.monotonic() + self.SEARCH_SECONDS
        best: dict[str, Any] | None = None
        best_key: tuple[int, int, int] | None = None
        for tour, pos in candidates:
            if time.monotonic() >= deadline or self._stopped():
                break
            result = _search_tour(body, heading, food, score, rows, cols, tour, pos, forecast, baseline, deadline, self._stopped)
            if result is None:
                continue
            first = int(result.get("first_depth", 10**9))
            # A route must not make the first food slower than the active safe
            # fallback.  It may win by reaching a second checked food sooner.
            if first > baseline:
                continue
            key = (first, -int(result.get("food_count", 1)), len(result.get("route", ())))
            if best_key is None or key < best_key:
                best, best_key = result, key
        return best

    def choose(self, obs: Any, names: tuple[str, ...]) -> str:
        shape = _shape(obs)
        body = _snake(obs)
        if shape is None or not body:
            return names[0] if names else "NOOP"
        self._configure(*shape)
        heading = _heading(obs, body)
        food = _food(obs)
        score = _number(obs, "score")
        tick = _number(obs, "tick")
        self._ensure_forecast(obs, body, food, score)
        self._poll(body, heading, food, score, tick)

        if self.expected is not None:
            expected_body, expected_heading, expected_food, expected_score = self.expected
            if (body, heading, food, score) != (expected_body, expected_heading, expected_food, expected_score):
                self._log("forecast_route_invalidated", tick=tick, reason="expected_state_mismatch")
                self._clear_plan()
            else:
                self.expected = None

        if self.plan_food is not None and self.plan_food != food and self.plan:
            # A real food change is fine only when the route's predicted state
            # already matched it; otherwise its suffix is stale.
            self._clear_plan()
        if self.active is not None and self.active_pos is not None and not _ordered(body, self.active, self.active_pos):
            self._log("tour_certificate_lost", tick=tick)
            self.active = None
            self.active_pos = None
            self._clear_plan()

        if not self.plan:
            self._select_active(body, heading, food, score, names)

        action: str | None = None
        if self.plan and self.plan_tour is not None and self.active_pos is not None:
            candidate = self.plan[0]
            stepped = _step(body, heading, candidate, food, self.rows, self.cols)
            if candidate in names and stepped is not None and _ordered(stepped[0], self.plan_tour, self.active_pos):
                action = candidate
                self.plan.pop(0)
                if stepped[2] and not self.plan:
                    self.plan_food = None
                    self.plan_tour = None
                else:
                    next_food = food
                    next_score = score
                    if stepped[2]:
                        next_score += 1
                        next_food = self.forecast.at(next_score, stepped[0], self.rows, self.cols) if self.forecast is not None else None
                        # The route is allowed to cross a food only because
                        # this newly spawned target was reconstructed and checked.
                        self.plan_food = next_food
                    self.expected = (stepped[0], stepped[1], next_food, next_score)
            else:
                self._clear_plan()

        if action is None and self.active is not None and self.active_pos is not None:
            action = _tour_fallback(body, heading, food, self.rows, self.cols, self.active, self.active_pos, names)
        if action is None:
            action = _free_fallback(body, heading, food, self.rows, self.cols, names)

        if action not in names:
            action = heading if heading in names else (names[0] if names else "NOOP")

        # The posted move is the exact origin for a free-first-action search.
        stepped = _step(body, heading, action, food, self.rows, self.cols) if action in MOVES else None
        if stepped is not None:
            next_food = food
            next_score = score
            if stepped[2]:
                next_score += 1
                next_food = self.forecast.at(next_score, stepped[0], self.rows, self.cols) if self.forecast is not None else None
                if self.plan:
                    self._clear_plan()
            self.expected = (stepped[0], stepped[1], next_food, next_score)
            if not stepped[2] and not self.plan:
                self._start_search(stepped[0], stepped[1], food, score, tick)
        return action


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

    def run_episode(self, game: Any) -> None:
        if not getattr(game, "tick_s", None):
            return
        spec = getattr(game, "action_spec", None)
        names = tuple(name for name in (getattr(spec, "names", ()) or ()) if isinstance(name, str))
        directions = tuple(move for move in MOVES if move in names)
        if not directions:
            noop = getattr(spec, "noop", None)
            directions = (noop,) if isinstance(noop, str) and noop in names else names
        if not directions:
            return
        controller = _Controller(self.ctx)
        while not self.ctx.done and not getattr(game, "done", False):
            observation = game.observe()
            action = controller.choose(observation, directions)
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
