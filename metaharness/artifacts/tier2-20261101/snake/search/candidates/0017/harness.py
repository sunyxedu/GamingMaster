"""Candidate 0017: food routes with cycle certificates and post-food ordering.

The controller keeps a validated Hamiltonian cycle as its emergency invariant,
but treats the cycle as a *certificate at the food endpoint*, not as a
restriction on every approach step.  A background search can therefore take a
short, exactly simulated route through currently empty cells.  Once it eats,
the resulting body must be ordered on a validated cycle before the route is
eligible.  Every route action is checked again in the foreground.
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Any


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
VECTORS = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT"}
OPPOSITE["RIGHT"] = "LEFT"


# ---------------------------------------------------------------------------
# Defensive observation handling and the exact public transition


def _point(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _body(obs: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(obs, dict) or not isinstance(obs.get("snake"), (list, tuple)):
        return ()
    cells: list[tuple[int, int]] = []
    for raw in obs["snake"]:
        cell = _point(raw)
        if cell is None:
            return ()
        cells.append(cell)
    return tuple(cells)


def _food(obs: Any) -> tuple[int, int] | None:
    return _point(obs.get("food")) if isinstance(obs, dict) and obs.get("food") is not None else None


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


def _tick(obs: Any) -> int:
    if not isinstance(obs, dict):
        return 0
    try:
        return max(0, int(obs.get("tick", 0)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _heading(obs: Any, body: tuple[tuple[int, int], ...]) -> str:
    if isinstance(obs, dict) and obs.get("direction") in VECTORS:
        return obs["direction"]
    if len(body) >= 2:
        delta = body[0][0] - body[1][0], body[0][1] - body[1][1]
        for name, vector in VECTORS.items():
            if delta == vector:
                return name
    return "RIGHT"


def _inside(cell: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols


def _advance(
    body: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """Apply one absolute request with the game's simultaneous tail rule."""
    if not body or heading not in VECTORS or request not in VECTORS:
        return None
    actual = heading if request == OPPOSITE.get(heading) else request
    dr, dc = VECTORS[actual]
    new_head = body[0][0] + dr, body[0][1] + dc
    if not _inside(new_head, rows, cols):
        return None
    ate = food is not None and new_head == food
    occupied = set(body) if ate else set(body[:-1])
    if new_head in occupied:
        return None
    next_body = (new_head,) + (body if ate else body[:-1])
    return next_body, actual, ate


# ---------------------------------------------------------------------------
# Validated rectangular cycle family


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _valid_cycle(path: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    size = rows * cols
    return (
        len(path) == size
        and len(set(path)) == size
        and all(_inside(cell, rows, cols) for cell in path)
        and all(_adjacent(path[i], path[(i + 1) % size]) for i in range(size))
    )


def _even_row_cycle(rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
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


def _cycle_family(rows: int, cols: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return independently checked reflected row/column-spine cycles."""
    bases: list[tuple[tuple[int, int], ...]] = []
    direct = _even_row_cycle(rows, cols)
    if direct is not None:
        bases.append(direct)
    transposed = _even_row_cycle(cols, rows)
    if transposed is not None:
        bases.append(tuple((col, row) for row, col in transposed))

    answer: list[tuple[tuple[int, int], ...]] = []
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
                        answer.append(candidate)
    return tuple(answer)


def _positions(cycle: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], int]:
    return {cell: index for index, cell in enumerate(cycle)}


def _ordered(
    body: tuple[tuple[int, int], ...],
    cycle: tuple[tuple[int, int], ...],
    index: dict[tuple[int, int], int] | None = None,
) -> bool:
    if not body or len(set(body)) != len(body) or not cycle:
        return False
    index = index if index is not None else _positions(cycle)
    if any(cell not in index for cell in body):
        return False
    size = len(cycle)
    gaps = [(index[a] - index[b]) % size for a, b in zip(body, body[1:])]
    return bool(gaps) and all(gap > 0 for gap in gaps) and sum(gaps) < size


def _cycle_move(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    index: dict[tuple[int, int], int],
    advertised: tuple[str, ...] | None = None,
) -> str | None:
    """Choose one transition that retains the complete cycle-order proof."""
    if not _ordered(body, cycle, index):
        return None
    size = len(cycle)
    next_cycle_cell = cycle[(index[body[0]] + 1) % size]
    choices: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action == OPPOSITE.get(heading) or (advertised and action not in advertised):
            continue
        step = _advance(body, heading, action, food, rows, cols)
        if step is None or not _ordered(step[0], cycle, index):
            continue
        new_body, new_heading, ate = step
        if ate or food is None or food not in index:
            forward = 0
        else:
            forward = (index[food] - index[new_body[0]]) % size
        choices.append(
            (
                (
                    int(ate),
                    -forward,
                    int(new_body[0] == next_cycle_cell),
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


def _greedy_cycle_route(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
    index: dict[tuple[int, int], int],
) -> tuple[str, ...] | None:
    """A conservative cycle-only route used as a no-regression bound."""
    current_body, current_heading = body, heading
    route: list[str] = []
    for _ in range(len(cycle) + len(body) + 8):
        action = _cycle_move(current_body, current_heading, food, rows, cols, cycle, index)
        if action is None:
            return None
        step = _advance(current_body, current_heading, action, food, rows, cols)
        if step is None or not _ordered(step[0], cycle, index):
            return None
        route.append(action)
        current_body, current_heading, ate = step
        if ate:
            return tuple(route)
    return None


def _cycle_for(
    body: tuple[tuple[int, int], ...],
    cycles: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], int]] | None:
    for cycle in cycles:
        index = _positions(cycle)
        if _ordered(body, cycle, index):
            return cycle, index
    return None


def _choose_cycle(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    cycles: tuple[tuple[tuple[int, int], ...], ...],
    advertised: tuple[str, ...],
    previous: tuple[tuple[int, int], ...] | None,
) -> tuple[tuple[int, int], ...] | None:
    """Select a cycle by an actual simulated route, only at food boundaries."""
    if food is None:
        return previous if previous is not None else None
    candidates: list[tuple[tuple[int, int, int, int], tuple[tuple[int, int], ...]]] = []
    for number, cycle in enumerate(cycles):
        index = _positions(cycle)
        if not _ordered(body, cycle, index):
            continue
        first = _cycle_move(body, heading, food, rows, cols, cycle, index, advertised)
        if first is None:
            continue
        route = _greedy_cycle_route(body, heading, food, rows, cols, cycle, index)
        length = len(route) if route is not None else rows * cols * 4
        candidates.append(
            (
                (
                    length,
                    -int(cycle == previous),
                    -int(first == heading),
                    number,
                ),
                cycle,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Conservative fallback used when no cycle certificate is available


def _reachable_area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not body:
        return 0
    blocked = set(body[1:])
    queue = deque([body[0]])
    seen = {body[0]}
    while queue:
        row, col = queue.popleft()
        for dr, dc in VECTORS.values():
            cell = row + dr, col + dc
            if _inside(cell, rows, cols) and cell not in blocked and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return len(seen)


def _can_reach_tail(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    if len(body) < 2:
        return True
    target = body[-1]
    blocked = set(body[1:-1])
    queue = deque([body[0]])
    seen = {body[0]}
    while queue:
        cell = queue.popleft()
        if cell == target:
            return True
        row, col = cell
        for dr, dc in VECTORS.values():
            nxt = row + dr, col + dc
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _safe_fallback(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    advertised: tuple[str, ...],
) -> str:
    choices: list[tuple[tuple[int, int, int, int, str], str]] = []
    for action in DIRECTIONS:
        if action not in advertised or action == OPPOSITE.get(heading):
            continue
        step = _advance(body, heading, action, food, rows, cols)
        if step is None:
            continue
        new_body, new_heading, ate = step
        area = _reachable_area(new_body, rows, cols)
        if ate and len(new_body) < rows * cols and area <= 1:
            continue
        choices.append(
            (
                (
                    int(ate),
                    int(_can_reach_tail(new_body, rows, cols)),
                    area,
                    int(new_heading == heading),
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


# ---------------------------------------------------------------------------
# Exact full-body search.  The cycle is a goal certificate, not an approach
# constraint.  This is the structural difference from the parent planners.


def _route_search(
    origin_body: tuple[tuple[int, int], ...],
    origin_heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    cycles: tuple[tuple[tuple[int, int], ...], ...],
    baseline: int | None,
    stop: Any,
) -> dict[str, Any] | None:
    if not origin_body or food in origin_body or origin_heading not in VECTORS:
        return None
    if baseline is not None and baseline <= 1:
        return None

    # The worker starts after the already-posted move.  It has a hard wall-clock
    # budget below one tick and a depth bound from the verified cycle fallback.
    deadline = time.monotonic() + 0.085
    board_size = rows * cols
    max_depth = min(board_size + len(origin_body) + 8, 190)
    if baseline is not None:
        max_depth = min(max_depth, baseline - 1)
    if max_depth <= 0:
        return None

    def stopped() -> bool:
        try:
            return bool(stop())
        except Exception:
            return False

    serial = 0
    start_distance = abs(origin_body[0][0] - food[0]) + abs(origin_body[0][1] - food[1])
    frontier: list[tuple[float, int, int, int, tuple[tuple[int, int], ...], str, tuple[str, ...]]] = [
        (float(start_distance), 0, 0, serial, origin_body, origin_heading, ())
    ]
    best_seen: dict[tuple[tuple[tuple[int, int], ...], str], tuple[int, int]] = {
        (origin_body, origin_heading): (0, 0)
    }
    best_goal: tuple[tuple[int, int, int, int], tuple[str, ...], tuple[tuple[int, int], ...]] | None = None
    best_depth: int | None = None
    expanded = 0

    while frontier and expanded < 18000 and time.monotonic() < deadline and not stopped():
        priority, depth, turns, _, current_body, current_heading, path = heapq.heappop(frontier)
        if best_depth is not None and priority > best_depth + 1.05:
            break
        expanded += 1
        if depth >= max_depth:
            continue

        ranked: list[tuple[int, int, str]] = []
        for action in DIRECTIONS:
            if action == OPPOSITE.get(current_heading):
                continue
            dr, dc = VECTORS[action]
            next_row = current_body[0][0] + dr
            next_col = current_body[0][1] + dc
            ranked.append((abs(next_row - food[0]) + abs(next_col - food[1]), int(action != current_heading), action))
        ranked.sort()

        for _, _, action in ranked:
            if stopped():
                return None
            step = _advance(current_body, current_heading, action, food, rows, cols)
            if step is None:
                continue
            new_body, new_heading, ate = step
            new_path = path + (action,)
            new_depth = depth + 1
            new_turns = turns + int(new_heading != current_heading)
            if ate:
                endpoint = _cycle_for(new_body, cycles)
                if endpoint is None:
                    continue
                endpoint_cycle, endpoint_index = endpoint
                area = _reachable_area(new_body, rows, cols)
                open_moves = 0
                for possible in DIRECTIONS:
                    if _advance(new_body, new_heading, possible, None, rows, cols) is not None:
                        open_moves += 1
                rank = (new_depth, -area, -open_moves, new_turns)
                candidate = (rank, new_path, endpoint_cycle)
                if best_goal is None or candidate[0] < best_goal[0]:
                    best_goal = candidate
                    best_depth = new_depth
                continue

            key = (new_body, new_heading)
            old = best_seen.get(key)
            if old is not None and (old[0] < new_depth or (old[0] == new_depth and old[1] <= new_turns)):
                continue
            best_seen[key] = (new_depth, new_turns)
            serial += 1
            remaining = abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
            heapq.heappush(
                frontier,
                (
                    new_depth + remaining + 0.01 * new_turns,
                    new_depth,
                    new_turns,
                    serial,
                    new_body,
                    new_heading,
                    new_path,
                ),
            )

    if best_goal is None:
        return None
    rank, route, endpoint_cycle = best_goal
    if baseline is not None and len(route) >= baseline:
        return None
    return {
        "origin_body": origin_body,
        "origin_heading": origin_heading,
        "origin_food": food,
        "route": route,
        "end_cycle": endpoint_cycle,
        "baseline_len": baseline,
        "expanded": expanded,
        "goal_rank": rank,
    }


def _route_suffix(
    result: Any,
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[str, ...] | None:
    if not isinstance(result, dict) or result.get("origin_food") != food:
        return None
    origin_body = result.get("origin_body")
    origin_heading = result.get("origin_heading")
    route = result.get("route")
    if not isinstance(origin_body, tuple) or not isinstance(route, tuple) or not route:
        return None

    current_body, current_heading = origin_body, origin_heading
    for offset, action in enumerate(route):
        if current_body == body and current_heading == heading:
            return route[offset:]
        step = _advance(current_body, current_heading, action, food, rows, cols)
        if step is None:
            return None
        current_body, current_heading, ate = step
        if ate:
            return None
    return () if current_body == body and current_heading == heading else None


# ---------------------------------------------------------------------------
# Real-time controller


class _Controller:
    SEARCH_SECONDS = 0.085

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.rows = 0
        self.cols = 0
        self.cycles: tuple[tuple[tuple[int, int], ...], ...] = ()
        self.active: tuple[tuple[int, int], ...] | None = None
        self.future: Any = None
        self.plan: list[str] = []
        self.plan_food: tuple[int, int] | None = None
        self.plan_end_cycle: tuple[tuple[int, int], ...] | None = None
        self.expected: tuple[tuple[tuple[int, int], ...], str, tuple[int, int] | None] | None = None
        self.seen_food: tuple[int, int] | None = None
        self.last_start = -100000
        self.pending: Any = None

    def log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def reset_shape(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        self.cycles = _cycle_family(rows, cols)
        self.active = None
        self.future = None
        self.pending = None
        self.plan = []
        self.plan_food = None
        self.plan_end_cycle = None
        self.expected = None
        self.seen_food = None
        self.log("cycle_certificates_ready", rows=rows, cols=cols, cycles=len(self.cycles))

    def adopt_cycle(self, candidate: Any) -> None:
        if not isinstance(candidate, tuple):
            return
        for cycle in self.cycles:
            if cycle == candidate:
                self.active = cycle
                return

    def clear_plan(self) -> None:
        self.plan = []
        self.plan_food = None
        self.plan_end_cycle = None
        self.expected = None

    def poll(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        tick: int,
    ) -> None:
        if self.future is None or not getattr(self.future, "done", lambda: False)():
            return
        future = self.future
        self.future = None
        try:
            result = future.result()
        except Exception as exc:
            self.log("food_route_discarded", tick=tick, reason=type(exc).__name__)
            return
        if not isinstance(result, dict):
            return
        route = result.get("route")
        if not isinstance(route, tuple) or not route:
            return
        if result.get("origin_food") != food:
            self.log("food_route_discarded", tick=tick, reason="food_changed")
            return
        baseline = result.get("baseline_len")
        if isinstance(baseline, int) and len(route) >= baseline:
            self.log("food_route_discarded", tick=tick, reason="not_shorter_than_cycle")
            return
        end_cycle = result.get("end_cycle")
        if not isinstance(end_cycle, tuple) or end_cycle not in self.cycles:
            self.log("food_route_discarded", tick=tick, reason="missing_cycle_certificate")
            return
        suffix = _route_suffix(result, body, heading, food, self.rows, self.cols)
        if not suffix:
            self.log("food_route_discarded", tick=tick, reason="prefix_mismatch")
            return
        self.plan = list(suffix)
        self.plan_food = food
        self.plan_end_cycle = end_cycle
        self.log(
            "food_route_ready",
            tick=tick,
            remaining=len(self.plan),
            expanded=result.get("expanded", 0),
            goal_rank=result.get("goal_rank"),
        )

    def queue_search(
        self,
        origin_body: tuple[tuple[int, int], ...],
        origin_heading: str,
        food: tuple[int, int] | None,
        tick: int,
    ) -> None:
        """Remember a post-action origin; launch it only after ctx.act returns."""
        if food is not None and not self.plan and self.cycles:
            self.pending = (origin_body, origin_heading, food, tick)

    def launch_search(self) -> None:
        """Start the bounded worker after the current action is posted.

        Spawning before ``ctx.act`` lets a CPU-heavy pure-Python worker contend
        for the GIL while the current action is being submitted.  The action is
        now already in the runner before this method can start that worker.
        """
        pending = self.pending
        self.pending = None
        if pending is None or self.future is not None or self.plan or not self.cycles:
            return
        origin_body, origin_heading, food, tick = pending
        if food is None or tick <= self.last_start:
            return
        baseline: int | None = None
        if self.active is not None:
            active_index = _positions(self.active)
            if _ordered(origin_body, self.active, active_index):
                baseline_route = _greedy_cycle_route(
                    origin_body,
                    origin_heading,
                    food,
                    self.rows,
                    self.cols,
                    self.active,
                    active_index,
                )
                baseline = len(baseline_route) if baseline_route is not None else None
        snapshot = (
            origin_body,
            origin_heading,
            food,
            self.rows,
            self.cols,
            self.cycles,
            baseline,
            lambda: bool(getattr(self.ctx, "done", False)),
        )
        try:
            self.future = self.ctx.spawn(lambda: _route_search(*snapshot))
            self.last_start = tick
            self.log("food_route_started", tick=tick, baseline=baseline, origin_tick=tick)
        except Exception as exc:
            self.future = None
            self.log("food_route_spawn_failed", tick=tick, reason=type(exc).__name__)

    def choose(self, obs: Any, advertised: tuple[str, ...]) -> str:
        dimensions = _shape(obs)
        body = _body(obs)
        if dimensions is None or not body:
            return advertised[0] if advertised else "NOOP"
        self.reset_shape(*dimensions)
        heading = _heading(obs, body)
        food = _food(obs)
        tick = _tick(obs)

        self.poll(body, heading, food, tick)
        if self.plan_food != food:
            self.clear_plan()
        if self.expected is not None:
            expected_body, expected_heading, expected_food = self.expected
            if body != expected_body or heading != expected_heading or food != expected_food:
                self.log("food_route_discarded", tick=tick, reason="expected_state_mismatch")
                self.clear_plan()
            else:
                self.expected = None

        # A planned route is permitted to leave the active cycle.  Its exact
        # transition check is therefore done before validating the old cycle.
        if self.plan and self.plan_food == food:
            action = self.plan[0]
            step = _advance(body, heading, action, food, self.rows, self.cols)
            if step is not None and action in advertised:
                self.plan.pop(0)
                if step[2]:
                    self.adopt_cycle(self.plan_end_cycle)
                    self.clear_plan()
                else:
                    self.expected = (step[0], step[1], food)
                return action
            self.log("food_route_discarded", tick=tick, reason="live_transition_rejected")
            self.clear_plan()

        # Select the safety cycle only when the target changes.  This avoids
        # per-tick portfolio oscillation while still choosing a food-specific
        # order by a complete simulated route length.
        if food is not None and food != self.seen_food:
            self.seen_food = food
            self.clear_plan()
            self.active = _choose_cycle(
                body,
                heading,
                food,
                self.rows,
                self.cols,
                self.cycles,
                advertised,
                self.active,
            )
            if self.active is not None:
                self.log("cycle_chosen_for_food", tick=tick, food=list(food))

        if self.active is not None:
            active_index = _positions(self.active)
            if not _ordered(body, self.active, active_index):
                self.log("cycle_certificate_lost", tick=tick)
                self.active = None

        action: str | None = None
        if self.active is not None:
            action = _cycle_move(
                body,
                heading,
                food,
                self.rows,
                self.cols,
                self.active,
                _positions(self.active),
                advertised,
            )
        if action not in advertised:
            action = _safe_fallback(body, heading, food, self.rows, self.cols, advertised)
        if action not in advertised:
            action = heading if heading in advertised else (advertised[0] if advertised else "NOOP")

        # The just-posted move is known exactly.  Search from that predicted
        # state, leaving the next action free (the 0013 timing improvement),
        # rather than spending the worker's budget reproving this move.
        next_state = _advance(body, heading, action, food, self.rows, self.cols)
        if next_state is not None and not next_state[2] and not self.plan:
            self.queue_search(next_state[0], next_state[1], food, tick + 1)
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
            controller.launch_search()
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
