"""Candidate 0020: two-food value search inside one persistent cycle.

The safety certificate is deliberately boring: a validated Hamiltonian cycle,
checked after every simulated move.  The search objective is not boring: when
Random(seed) can be reconstructed from the public observation, it chooses a
route to the *current* apple by the time to that apple plus a small estimate of
the next apple's cycle distance.  Thus a route may be a little longer now when
it leaves the grown snake in a much better position for the next target.

All computation is local and deterministic.  The only asynchronous work is a
short, cancellable search for a route beginning at the predicted state after
the move currently being posted.
"""
from __future__ import annotations

import heapq
import random
import time
from collections import deque
from typing import Any, Callable


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
VECTOR = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _point(value: Any) -> tuple[int, int] | None:
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
    for item in raw:
        cell = _point(item)
        if cell is None:
            return ()
        result.append(cell)
    return tuple(result)


def _food(observation: Any) -> tuple[int, int] | None:
    if not isinstance(observation, dict) or observation.get("food") is None:
        return None
    return _point(observation.get("food"))


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


def _integer(observation: Any, name: str, default: int = 0) -> int:
    if not isinstance(observation, dict):
        return default
    try:
        return max(0, int(observation.get(name, default)))
    except (TypeError, ValueError, OverflowError):
        return default


def _heading(observation: Any, snake: tuple[tuple[int, int], ...]) -> str:
    value = observation.get("direction") if isinstance(observation, dict) else None
    if value in VECTOR:
        return str(value)
    if len(snake) >= 2:
        delta = snake[0][0] - snake[1][0], snake[0][1] - snake[1][1]
        for name, vector in VECTOR.items():
            if delta == vector:
                return name
    return "RIGHT"


def _inside(cell: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _valid_cycle(cycle: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    size = rows * cols
    return (
        len(cycle) == size
        and len(set(cycle)) == size
        and all(_inside(cell, rows, cols) for cell in cycle)
        and all(_adjacent(cycle[i], cycle[(i + 1) % size]) for i in range(size))
    )


def _even_row_cycle(rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """Make the standard serpentine tour with a reserved return column."""
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


def _cycle_family(rows: int, cols: int) -> list[tuple[tuple[int, int], ...]]:
    """Return checked orientations; do not assume either orientation is safe."""
    bases: list[tuple[tuple[int, int], ...]] = []
    direct = _even_row_cycle(rows, cols)
    if direct is not None:
        bases.append(direct)
    if cols % 2 == 0:
        across = _even_row_cycle(cols, rows)
        if across is not None:
            bases.append(tuple((col, row) for row, col in across))

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
                for candidate in (reflected, tuple(reversed(reflected))):
                    if candidate not in seen and _valid_cycle(candidate, rows, cols):
                        seen.add(candidate)
                        result.append(candidate)
    return result


def _positions(cycle: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], int]:
    return {cell: number for number, cell in enumerate(cycle)}


def _ordered(
    snake: tuple[tuple[int, int], ...],
    cycle: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int],
) -> bool:
    """Check strict head-to-tail order on the cyclic certificate."""
    if not snake or len(set(snake)) != len(snake):
        return False
    if any(cell not in positions for cell in snake):
        return False
    if len(snake) == 1:
        return True
    size = len(cycle)
    gaps = [
        (positions[newer] - positions[older]) % size
        for newer, older in zip(snake, snake[1:])
    ]
    return all(gap > 0 for gap in gaps) and sum(gaps) < size


def _advance(
    snake: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    apple: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """Exact one-tick transition, including ignored reverse and tail release."""
    if not snake or heading not in VECTOR or request not in VECTOR:
        return None
    actual = heading if request == OPPOSITE.get(heading) else request
    dr, dc = VECTOR[actual]
    new_head = snake[0][0] + dr, snake[0][1] + dc
    if not _inside(new_head, rows, cols):
        return None
    ate = apple is not None and new_head == apple
    occupied = set(snake) if ate else set(snake[:-1])
    if new_head in occupied:
        return None
    return (new_head,) + (snake if ate else snake[:-1]), actual, ate


def _reachable_area(snake: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not snake:
        return 0
    blocked = set(snake[1:])
    queue = deque([snake[0]])
    seen = {snake[0]}
    while queue:
        row, col = queue.popleft()
        for dr, dc in VECTOR.values():
            nxt = row + dr, col + dc
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


def _tail_reachable(snake: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    if len(snake) < 2:
        return True
    target = snake[-1]
    blocked = set(snake[1:-1])
    queue = deque([snake[0]])
    seen = {snake[0]}
    while queue:
        cell = queue.popleft()
        if cell == target:
            return True
        for dr, dc in VECTOR.values():
            nxt = cell[0] + dr, cell[1] + dc
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _safe_fallback(
    snake: tuple[tuple[int, int], ...],
    heading: str,
    apple: tuple[int, int] | None,
    rows: int,
    cols: int,
    advertised: tuple[str, ...],
) -> str:
    choices: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action not in advertised or action == OPPOSITE.get(heading):
            continue
        step = _advance(snake, heading, action, apple, rows, cols)
        if step is None:
            continue
        next_snake, next_heading, ate = step
        area = _reachable_area(next_snake, rows, cols)
        if ate and len(next_snake) < rows * cols and area <= 1:
            continue
        choices.append(
            (
                (
                    int(ate),
                    int(_tail_reachable(next_snake, rows, cols)),
                    area,
                    int(next_heading == heading),
                    action,
                ),
                action,
            )
        )
    if choices:
        choices.sort(reverse=True)
        return choices[0][1]
    if heading in advertised:
        return heading
    return advertised[0] if advertised else "NOOP"


def _cycle_move(
    snake: tuple[tuple[int, int], ...],
    heading: str,
    apple: tuple[int, int] | None,
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int],
    advertised: tuple[str, ...] | None = None,
) -> str | None:
    """Choose one checked cycle-compatible neighbor with target potential."""
    if not _ordered(snake, cycle, positions):
        return None
    size = len(cycle)
    next_cycle = cycle[(positions[snake[0]] + 1) % size]
    options: list[tuple[tuple[int, int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action == OPPOSITE.get(heading):
            continue
        if advertised is not None and action not in advertised:
            continue
        step = _advance(snake, heading, action, apple, rows, cols)
        if step is None or not _ordered(step[0], cycle, positions):
            continue
        new_snake, new_heading, ate = step
        if apple is None:
            forward = 0
            manhattan = 0
        else:
            forward = (positions[apple] - positions[new_snake[0]]) % size
            manhattan = abs(new_snake[0][0] - apple[0]) + abs(new_snake[0][1] - apple[1])
        # Forward cycle progress remains primary; Manhattan only chooses among
        # equally progressing safe shortcuts.  This prevents a direct-looking
        # turn from orbiting around the target without advancing the certificate.
        rank = (
            int(ate),
            -forward,
            -manhattan,
            int(new_snake[0] == next_cycle),
            int(new_heading == heading),
            action,
        )
        options.append((rank, action))
    if not options:
        return None
    options.sort(reverse=True)
    return options[0][1]


def _cycle_route(
    snake: tuple[tuple[int, int], ...],
    heading: str,
    apple: tuple[int, int],
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int],
) -> tuple[str, ...] | None:
    current_snake, current_heading = snake, heading
    path: list[str] = []
    for _ in range(len(cycle) + len(snake) + 12):
        action = _cycle_move(
            current_snake, current_heading, apple, rows, cols, cycle, positions
        )
        if action is None:
            return None
        step = _advance(current_snake, current_heading, action, apple, rows, cols)
        if step is None or not _ordered(step[0], cycle, positions):
            return None
        path.append(action)
        current_snake, current_heading, ate = step
        if ate:
            return tuple(path)
    return None


def _play_path(
    snake: tuple[tuple[int, int], ...],
    heading: str,
    path: tuple[str, ...],
    apple: tuple[int, int],
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    current_snake, current_heading = snake, heading
    for action in path:
        step = _advance(current_snake, current_heading, action, apple, rows, cols)
        if step is None:
            return None
        current_snake, current_heading, ate = step
        if ate:
            return current_snake, current_heading, True
    return current_snake, current_heading, False


class _FoodForecast:
    """A checked suffix of the documented row-major Random.choice stream."""

    def __init__(
        self,
        seed: Any,
        rows: int,
        cols: int,
        initial_length: int,
        initial_snake: tuple[tuple[int, int], ...],
        initial_food: tuple[int, int] | None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.total = rows * cols
        self.initial_length = initial_length
        self.ranks: list[int] = []
        self.valid = False
        if initial_food is None or self.total <= initial_length:
            return
        for opening_draws in range(7):
            try:
                rng = random.Random(seed)
                for _ in range(opening_draws):
                    rng.randrange(self.total - initial_length)
                ranks = [
                    rng.randrange(free_count)
                    for free_count in range(self.total - initial_length, 0, -1)
                ]
            except (TypeError, ValueError, OverflowError):
                continue
            if self._rank_cell(ranks[0], initial_snake) == initial_food:
                self.ranks = ranks
                self.valid = True
                return

    def _rank_cell(
        self, rank: int, snake: tuple[tuple[int, int], ...]
    ) -> tuple[int, int] | None:
        occupied = set(snake)
        empty = [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) not in occupied
        ]
        return empty[rank] if 0 <= rank < len(empty) else None

    def at(self, score: int, snake: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
        if not self.valid or score < 0 or score >= len(self.ranks):
            return None
        return self._rank_cell(self.ranks[score], snake)


def _future_proxy(
    snake: tuple[tuple[int, int], ...],
    target: tuple[int, int] | None,
    cycle: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int],
) -> int:
    if target is None or target not in positions or not snake:
        return len(cycle) * 2
    return (positions[target] - positions[snake[0]]) % len(cycle)


def _search_route(
    origin_snake: tuple[tuple[int, int], ...],
    origin_heading: str,
    apple: tuple[int, int],
    score: int,
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int],
    forecast: _FoodForecast | None,
    stop: Callable[[], bool],
) -> dict[str, Any] | None:
    """Find a current-apple route whose two-food cost beats the fallback.

    The worker never relaxes cycle order.  It keeps searching after finding a
    current-food goal, because the body endpoint determines how soon the next
    apple is reached on the retained cycle.
    """
    if not _ordered(origin_snake, cycle, positions) or origin_heading not in VECTOR:
        return None
    baseline_path = _cycle_route(
        origin_snake, origin_heading, apple, rows, cols, cycle, positions
    )
    baseline_len = len(baseline_path) if baseline_path is not None else None
    if baseline_len is not None and baseline_len <= 1:
        return None

    baseline_objective: float | None = None
    if baseline_path is not None:
        endpoint = _play_path(
            origin_snake, origin_heading, baseline_path, apple, rows, cols
        )
        if endpoint is not None and endpoint[2] and forecast is not None and forecast.valid:
            next_apple = forecast.at(score + 1, endpoint[0])
            baseline_objective = baseline_len + 0.35 * _future_proxy(
                endpoint[0], next_apple, cycle, positions
            )
    if baseline_objective is None and baseline_len is None:
        baseline_len = len(cycle) + len(origin_snake)

    deadline = time.monotonic() + 0.038
    max_depth = min(
        len(cycle) + len(origin_snake) + 10,
        (baseline_len + 14) if baseline_len is not None else len(cycle) + 12,
    )
    if max_depth <= 1:
        return None

    serial = 0
    start_distance = abs(origin_snake[0][0] - apple[0]) + abs(origin_snake[0][1] - apple[1])
    frontier: list[
        tuple[
            float,
            int,
            int,
            int,
            tuple[tuple[int, int], ...],
            str,
            tuple[str, ...],
        ]
    ] = [(float(start_distance), 0, 0, serial, origin_snake, origin_heading, ())]
    best_seen: dict[
        tuple[tuple[tuple[int, int], ...], str], tuple[int, int]
    ] = {(origin_snake, origin_heading): (0, 0)}
    best_goal: tuple[float, int, int, tuple[str, ...], tuple[tuple[int, int], ...], str] | None = None
    expanded = 0

    while frontier and expanded < 10000 and time.monotonic() < deadline:
        if stop():
            return None
        _, depth, turns, _, current_snake, current_heading, path = heapq.heappop(frontier)
        expanded += 1
        if depth >= max_depth:
            continue
        ranked: list[tuple[int, int, str]] = []
        for action in DIRECTIONS:
            if action == OPPOSITE.get(current_heading):
                continue
            dr, dc = VECTOR[action]
            candidate = current_snake[0][0] + dr, current_snake[0][1] + dc
            ranked.append(
                (
                    abs(candidate[0] - apple[0]) + abs(candidate[1] - apple[1]),
                    int(action != current_heading),
                    action,
                )
            )
        ranked.sort()
        for _, _, action in ranked:
            if time.monotonic() >= deadline or stop():
                break
            step = _advance(current_snake, current_heading, action, apple, rows, cols)
            if step is None:
                continue
            next_snake, next_heading, ate = step
            if not _ordered(next_snake, cycle, positions):
                continue
            next_path = path + (action,)
            next_depth = depth + 1
            next_turns = turns + int(next_heading != current_heading)
            if ate:
                next_apple = None
                if forecast is not None and forecast.valid:
                    next_apple = forecast.at(score + 1, next_snake)
                proxy = _future_proxy(next_snake, next_apple, cycle, positions)
                objective = float(next_depth) + (0.35 * proxy if next_apple is not None else 0.0)
                qualifies = (
                    baseline_objective is not None and objective + 0.001 < baseline_objective
                ) or (baseline_objective is None and baseline_len is not None and next_depth < baseline_len)
                if qualifies:
                    candidate_goal = (
                        objective,
                        next_depth,
                        next_turns,
                        next_path,
                        next_snake,
                        next_heading,
                    )
                    if best_goal is None or candidate_goal[:3] < best_goal[:3]:
                        best_goal = candidate_goal
                continue

            key = (next_snake, next_heading)
            old = best_seen.get(key)
            if old is not None and old <= (next_depth, next_turns):
                continue
            best_seen[key] = (next_depth, next_turns)
            remaining = abs(next_snake[0][0] - apple[0]) + abs(next_snake[0][1] - apple[1])
            forward = _future_proxy(next_snake, apple, cycle, positions)
            serial += 1
            heapq.heappush(
                frontier,
                (
                    next_depth + remaining + 0.003 * forward + 0.01 * next_turns,
                    next_depth,
                    next_turns,
                    serial,
                    next_snake,
                    next_heading,
                    next_path,
                ),
            )

    if best_goal is None:
        return None
    objective, depth, turns, route, _, _ = best_goal
    return {
        "origin_snake": origin_snake,
        "origin_heading": origin_heading,
        "origin_food": apple,
        "route": route,
        "baseline_len": baseline_len,
        "baseline_objective": baseline_objective,
        "objective": objective,
        "turns": turns,
        "expanded": expanded,
    }


def _route_suffix(
    result: Any,
    live_snake: tuple[tuple[int, int], ...],
    live_heading: str,
    live_food: tuple[int, int] | None,
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    positions: dict[tuple[int, int], int],
) -> tuple[str, ...] | None:
    if not isinstance(result, dict) or result.get("origin_food") != live_food:
        return None
    origin_snake = result.get("origin_snake")
    origin_heading = result.get("origin_heading")
    route = result.get("route")
    if not isinstance(origin_snake, tuple) or not isinstance(route, tuple) or not route:
        return None
    if not _ordered(origin_snake, cycle, positions):
        return None
    current_snake, current_heading = origin_snake, origin_heading
    for index, action in enumerate(route):
        if current_snake == live_snake and current_heading == live_heading:
            return route[index:]
        step = _advance(current_snake, current_heading, action, live_food, rows, cols)
        if step is None or not _ordered(step[0], cycle, positions):
            return None
        current_snake, current_heading, ate = step
        if ate:
            return None
    if current_snake == live_snake and current_heading == live_heading:
        return ()
    return None


class _Controller:
    SEARCH_SECONDS = 0.038

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.rows = 0
        self.cols = 0
        self.cycles: list[tuple[tuple[int, int], ...]] = []
        self.cycle: tuple[tuple[int, int], ...] | None = None
        self.positions: dict[tuple[int, int], int] | None = None
        self.forecast: _FoodForecast | None = None
        self.forecast_seed: Any = object()
        self.forecast_checked_score: int | None = None
        self.future: Any = None
        self.plan: list[str] = []
        self.plan_food: tuple[int, int] | None = None
        self.expected: tuple[tuple[tuple[int, int], ...], str, tuple[int, int] | None] | None = None
        self.last_launch_tick = -100000

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _reset_shape(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        self.cycles = _cycle_family(rows, cols)
        self.cycle = None
        self.positions = None
        self.forecast = None
        self.forecast_seed = object()
        self.forecast_checked_score = None
        self.future = None
        self.plan = []
        self.plan_food = None
        self.expected = None
        self._log("cycle_certificates_built", rows=rows, cols=cols, count=len(self.cycles))

    def _refresh_forecast(
        self,
        observation: Any,
        snake: tuple[tuple[int, int], ...],
        apple: tuple[int, int] | None,
        score: int,
    ) -> None:
        seed = observation.get("seed") if isinstance(observation, dict) else None
        if seed != self.forecast_seed:
            self.forecast_seed = seed
            initial_length = max(1, len(snake) - score)
            candidate = _FoodForecast(
                seed, self.rows, self.cols, initial_length, snake, apple
            )
            self.forecast = candidate if candidate.valid else None
            self.forecast_checked_score = None
            self._log("food_stream_checked", valid=bool(self.forecast), score=score)
        if (
            self.forecast is not None
            and self.forecast.valid
            and apple is not None
            and self.forecast_checked_score != score
        ):
            if self.forecast.at(score, snake) != apple:
                self._log("food_stream_invalidated", score=score)
                self.forecast = None
            else:
                # The current apple's row-major rank changes as the tail moves;
                # validate it only at the first observation of this score.
                self.forecast_checked_score = score

    def _choose_cycle(
        self,
        snake: tuple[tuple[int, int], ...],
        heading: str,
        apple: tuple[int, int] | None,
        score: int,
    ) -> None:
        if self.cycle is not None or not self.cycles:
            return
        candidates: list[tuple[tuple[float, int, int, int], tuple[tuple[int, int], ...], dict[tuple[int, int], int]]] = []
        for number, cycle in enumerate(self.cycles):
            positions = _positions(cycle)
            if not _ordered(snake, cycle, positions):
                continue
            first = _cycle_move(snake, heading, apple, self.rows, self.cols, cycle, positions)
            if first is None:
                continue
            route_len = len(cycle) * 3
            future = len(cycle)
            if apple is not None:
                route = _cycle_route(snake, heading, apple, self.rows, self.cols, cycle, positions)
                if route is not None:
                    route_len = len(route)
                    endpoint = _play_path(snake, heading, route, apple, self.rows, self.cols)
                    if endpoint is not None and endpoint[2] and self.forecast is not None:
                        next_apple = self.forecast.at(score + 1, endpoint[0])
                        future = _future_proxy(endpoint[0], next_apple, cycle, positions)
            candidates.append(
                (
                    (
                        float(route_len) + 0.35 * future,
                        -int(first == heading),
                        -int(first != OPPOSITE.get(heading)),
                        number,
                    ),
                    cycle,
                    positions,
                )
            )
        if candidates:
            candidates.sort(key=lambda item: item[0])
            _, self.cycle, self.positions = candidates[0]
            self._log("cycle_committed", candidates=len(candidates), lookahead=True)

    def _clear_plan(self) -> None:
        self.plan = []
        self.plan_food = None
        self.expected = None

    def _stopped(self) -> bool:
        try:
            return bool(self.ctx.done)
        except Exception:
            return True

    def _poll(
        self,
        snake: tuple[tuple[int, int], ...],
        heading: str,
        apple: tuple[int, int] | None,
        tick: int,
    ) -> None:
        future = self.future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.future = None
        try:
            result = future.result()
        except Exception as exc:
            self._log("lookahead_search_discarded", tick=tick, reason=type(exc).__name__)
            return
        if self.cycle is None or self.positions is None:
            return
        suffix = _route_suffix(
            result,
            snake,
            heading,
            apple,
            self.rows,
            self.cols,
            self.cycle,
            self.positions,
        )
        if suffix:
            self.plan = list(suffix)
            self.plan_food = apple
            self._log(
                "lookahead_route_ready",
                tick=tick,
                remaining=len(self.plan),
                objective=result.get("objective") if isinstance(result, dict) else None,
                baseline=result.get("baseline_objective") if isinstance(result, dict) else None,
                expanded=result.get("expanded", 0) if isinstance(result, dict) else 0,
            )
        elif result is not None:
            self._log("lookahead_search_discarded", tick=tick, reason="prefix_mismatch")

    def _start_search(
        self,
        next_snake: tuple[tuple[int, int], ...],
        next_heading: str,
        apple: tuple[int, int] | None,
        score: int,
        tick: int,
    ) -> None:
        if (
            apple is None
            or self.cycle is None
            or self.positions is None
            or self.future is not None
            or self.plan
            or tick - self.last_launch_tick < 1
        ):
            return
        cycle = self.cycle
        positions = self.positions
        forecast = self.forecast
        snapshot = (
            next_snake,
            next_heading,
            apple,
            score,
            self.rows,
            self.cols,
            cycle,
            positions,
            forecast,
            self._stopped,
        )
        try:
            self.future = self.ctx.spawn(lambda: _search_route(*snapshot))
            self.last_launch_tick = tick
            self._log("lookahead_search_started", tick=tick, origin_tick=tick + 1)
        except Exception as exc:
            self.future = None
            self._log("lookahead_search_spawn_failed", tick=tick, reason=type(exc).__name__)

    def choose(self, observation: Any, advertised: tuple[str, ...]) -> str:
        shape = _shape(observation)
        snake = _body(observation)
        if shape is None or not snake:
            return advertised[0] if advertised else "NOOP"
        self._reset_shape(*shape)
        heading = _heading(observation, snake)
        apple = _food(observation)
        score = _integer(observation, "score")
        tick = _integer(observation, "tick")
        self._refresh_forecast(observation, snake, apple, score)

        self._poll(snake, heading, apple, tick)
        if self.plan_food != apple:
            self._clear_plan()
        if self.expected is not None:
            expected_snake, expected_heading, expected_food = self.expected
            if (snake, heading, apple) != (expected_snake, expected_heading, expected_food):
                self._log("lookahead_route_invalidated", tick=tick, reason="state_mismatch")
                self._clear_plan()
            else:
                self.expected = None

        self._choose_cycle(snake, heading, apple, score)
        if self.cycle is not None and self.positions is not None:
            if not _ordered(snake, self.cycle, self.positions):
                self._log("cycle_certificate_lost", tick=tick)
                self.cycle = None
                self.positions = None
                self._clear_plan()

        action: str | None = None
        if self.plan and self.cycle is not None and self.positions is not None:
            candidate = self.plan[0]
            step = _advance(snake, heading, candidate, apple, self.rows, self.cols)
            if (
                candidate in advertised
                and step is not None
                and _ordered(step[0], self.cycle, self.positions)
            ):
                action = candidate
                self.plan.pop(0)
                if step[2]:
                    self._clear_plan()
                else:
                    self.expected = (step[0], step[1], apple)
            else:
                self._log("lookahead_route_invalidated", tick=tick, reason="live_transition")
                self._clear_plan()

        if action is None and self.cycle is not None and self.positions is not None:
            action = _cycle_move(
                snake,
                heading,
                apple,
                self.rows,
                self.cols,
                self.cycle,
                self.positions,
                advertised,
            )
        if action not in advertised:
            action = _safe_fallback(snake, heading, apple, self.rows, self.cols, advertised)
        if action not in advertised:
            action = heading if heading in advertised else (advertised[0] if advertised else "NOOP")

        posted_state = _advance(snake, heading, action, apple, self.rows, self.cols)
        if (
            posted_state is not None
            and not posted_state[2]
            and not self.plan
            and self.cycle is not None
            and self.positions is not None
            and _ordered(posted_state[0], self.cycle, self.positions)
        ):
            self._start_search(posted_state[0], posted_state[1], apple, score, tick)
        return action


class Harness:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def run_episode(self, game: Any) -> None:
        if not getattr(game, "tick_s", None):
            return
        spec = getattr(game, "action_spec", None)
        names = tuple(
            name for name in (getattr(spec, "names", ()) or ()) if isinstance(name, str)
        )
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
