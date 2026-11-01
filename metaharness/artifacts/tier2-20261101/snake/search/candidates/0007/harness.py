"""Candidate 0007: an asynchronous route planner with a deadline shield.

The controller deliberately separates two clocks.  A bounded planner searches
for a safe food (or tail) route in a worker, while the foreground loop only
performs small exact transition checks and posts an action immediately.  A
Jev judgment can break ties between the currently executable alternatives, but
never becomes a prerequisite for a tick.
"""
from __future__ import annotations

import heapq
import json
import time
from collections import deque
from typing import Any

from metaharness.models.jev import Choice, Noul


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
    out = []
    for value in obs["snake"]:
        point = _cell(value)
        if point is None:
            return ()
        out.append(point)
    return tuple(out)


def _food(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict) or obs.get("food") is None:
        return None
    return _cell(obs.get("food"))


def _heading(obs: Any) -> str:
    value = obs.get("direction") if isinstance(obs, dict) else None
    return value if value in DIRECTIONS else "RIGHT"


def _dims(obs: Any) -> tuple[int, int]:
    try:
        rows, cols = int(obs.get("rows", 12)), int(obs.get("cols", 12))
        if rows >= 2 and cols >= 2:
            return rows, cols
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass
    return 12, 12


def _inside(point: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= point[0] < rows and 0 <= point[1] < cols


def _transition(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    request: str,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """One exact game transition; the tail vacates only on a non-food move."""
    if not body:
        return None
    new_heading = request if request in DIRECTIONS and request != OPPOSITE.get(heading) else heading
    dr, dc = VEC.get(new_heading, VEC["RIGHT"])
    nxt = (body[0][0] + dr, body[0][1] + dc)
    if not _inside(nxt, rows, cols):
        return None
    ate = food is not None and nxt == food
    occupied = set(body if ate else body[:-1])
    if nxt in occupied:
        return None
    return (nxt,) + (body if ate else body[:-1]), new_heading, ate


def _successors(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
):
    for action in DIRECTIONS:
        if action == OPPOSITE.get(heading):
            continue
        result = _transition(body, heading, food, action, rows, cols)
        if result is not None:
            yield action, result[0], result[1], result[2]


def _area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not body:
        return 0
    blocked = set(body[1:])
    start = body[0]
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in VEC.values():
            nxt = (r + dr, c + dc)
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen)


def _tail_access(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    if len(body) <= 1:
        return True
    start, target = body[0], body[-1]
    blocked = set(body[1:-1])
    q = deque([start])
    seen = {start}
    while q:
        point = q.popleft()
        if point == target:
            return True
        r, c = point
        for dr, dc in VEC.values():
            nxt = (r + dr, c + dc)
            if _inside(nxt, rows, cols) and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def _post_growth_safe(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    if len(body) >= rows * cols:
        return True
    return _tail_access(body, rows, cols) and _area(body, rows, cols) >= 2


def _repeat_safe(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    request: str,
    rows: int,
    cols: int,
    ticks: int,
) -> bool:
    """Check what happens if this request is repeated while Jev is late."""
    cur_body, cur_heading, cur_food = body, heading, food
    for _ in range(max(1, ticks)):
        result = _transition(cur_body, cur_heading, cur_food, request, rows, cols)
        if result is None:
            return False
        cur_body, cur_heading, ate = result
        # The next food is private RNG state.  Retain the growth cell but do
        # not hallucinate a location for the new food in this short shield.
        if ate:
            cur_food = None
    return True


def _safe_sets(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    horizon: int = 3,
) -> tuple[list[str], list[str]]:
    one = []
    for action, new_body, _, ate in _successors(body, heading, food, rows, cols):
        # A legal food step is still rejected when its retained tail leaves a
        # sealed post-growth state.  If it is the only move, the final
        # heading fallback still lets the game report the unavoidable result.
        if ate and not _post_growth_safe(new_body, rows, cols):
            continue
        one.append(action)
    delayed = [
        action
        for action in one
        if _repeat_safe(body, heading, food, action, rows, cols, horizon)
    ]
    return one, delayed


def _food_route(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int],
    rows: int,
    cols: int,
    deadline: float,
) -> list[str] | None:
    """Bounded A* over full body states, with a post-growth safety gate."""
    if not body or body[0] == food:
        return None
    serial = 0
    start_key = (body, heading)
    heap: list[tuple[float, int, int, tuple[tuple[int, int], ...], str, tuple[str, ...]]] = []
    h0 = abs(body[0][0] - food[0]) + abs(body[0][1] - food[1])
    heapq.heappush(heap, (h0, 0, serial, body, heading, ()))
    seen = {start_key}
    expansions = 0
    max_depth = min(rows * cols + len(body), 180)
    while heap and expansions < 5000 and time.monotonic() < deadline:
        _, depth, _, cur_body, cur_heading, path = heapq.heappop(heap)
        expansions += 1
        if depth >= max_depth:
            continue
        for action, new_body, new_heading, ate in _successors(
            cur_body, cur_heading, food, rows, cols
        ):
            new_path = path + (action,)
            if ate:
                if _post_growth_safe(new_body, rows, cols):
                    return list(new_path)
                continue
            key = (new_body, new_heading)
            if key in seen:
                continue
            seen.add(key)
            distance = abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
            serial += 1
            heapq.heappush(
                heap,
                (len(new_path) + distance + (0.01 if new_heading != cur_heading else 0),
                 len(new_path), serial, new_body, new_heading, new_path),
            )
    return None


def _tail_route(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    deadline: float,
) -> list[str] | None:
    """Bounded route toward the old tail, refusing accidental growth."""
    if len(body) < 2:
        return None
    target = body[-1]
    q = deque([(body, heading, ())])
    seen = {(body, heading)}
    expansions = 0
    while q and expansions < 2600 and time.monotonic() < deadline:
        cur_body, cur_heading, path = q.popleft()
        expansions += 1
        if len(path) >= min(rows * cols, 120):
            continue
        for action, new_body, new_heading, ate in _successors(
            cur_body, cur_heading, food, rows, cols
        ):
            if ate:
                continue
            new_path = path + (action,)
            if new_body[0] == target:
                return list(new_path)
            key = (new_body, new_heading)
            if key not in seen:
                seen.add(key)
                q.append((new_body, new_heading, new_path))
    return None


def _planner_task(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
):
    """Pure worker task.  Its hard bound prevents lingering after an episode."""
    if not food:
        return None
    deadline = time.monotonic() + 0.055
    route = _food_route(body, heading, food, rows, cols, deadline)
    kind = "FOOD"
    if route is None:
        route = _tail_route(body, heading, food, rows, cols, deadline)
        kind = "TAIL"
    if not route:
        return None
    return {
        "origin_body": body,
        "origin_heading": heading,
        "origin_food": food,
        "route": tuple(route),
        "kind": kind,
    }


def _cycle(rows: int, cols: int) -> list[tuple[int, int]]:
    """Return a validated rectangular cycle when either dimension is even."""
    if rows % 2 == 0:
        path = [(r, 0) for r in range(rows)]
        for offset, r in enumerate(reversed(range(rows))):
            cs = range(1, cols) if offset % 2 == 0 else range(cols - 1, 0, -1)
            path.extend((r, c) for c in cs)
    elif cols % 2 == 0:
        transposed = _cycle(cols, rows)
        path = [(c, r) for r, c in transposed]
    else:
        return []
    expected = rows * cols
    if len(path) != expected or len(set(path)) != expected:
        return []
    if any(
        abs(path[i][0] - path[(i + 1) % expected][0])
        + abs(path[i][1] - path[(i + 1) % expected][1])
        != 1
        for i in range(expected)
    ):
        return []
    return path


def _cycle_step(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> str | None:
    if len(body) < 2:
        return None
    cycle = _cycle(rows, cols)
    if not cycle:
        return None
    index = {point: i for i, point in enumerate(cycle)}
    if any(point not in index for point in body):
        return None
    n = len(cycle)
    for orientation in (1, -1):
        total = 0
        ordered = True
        for i in range(1, len(body)):
            gap = ((index[body[i - 1]] - index[body[i]]) * orientation) % n
            if gap <= 0:
                ordered = False
                break
            total += gap
        if not ordered or total >= n:
            continue
        next_cell = cycle[(index[body[0]] + orientation) % n]
        dr = next_cell[0] - body[0][0]
        dc = next_cell[1] - body[0][1]
        action = next((name for name, vector in VEC.items() if vector == (dr, dc)), None)
        if action is None or action == OPPOSITE.get(heading):
            continue
        result = _transition(body, heading, food, action, rows, cols)
        if result is None:
            continue
        if result[2] and not _post_growth_safe(result[0], rows, cols):
            continue
        return action
    return None


def _probability(answer: Any, value: str) -> float:
    try:
        return float(getattr(answer, "probabilities", {}).get(value, 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


class SnakeController:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.rows, self.cols = 12, 12
        self.plan: list[str] = []
        self.plan_kind = ""
        self.plan_food: tuple[int, int] | None = None
        self.expected = None
        self.route_future = None
        self.route_started_tick = -1000
        self.route_started_food = None
        self.jev_future = None
        self.jev_tick = None
        self.last_jev_tick = -1000
        self.cycles_enabled = True

    def _update_dims(self, obs: dict) -> None:
        self.rows, self.cols = _dims(obs)

    def _clear_plan(self) -> None:
        self.plan = []
        self.plan_kind = ""
        self.plan_food = None

    def _poll_route(self, obs: dict, body, heading, food, tick: int) -> None:
        future = self.route_future
        if future is None or not future.done():
            return
        self.route_future = None
        try:
            result = future.result()
        except Exception as exc:
            self.ctx.log("route_worker_ignored", error=type(exc).__name__)
            return
        if not isinstance(result, dict):
            self.ctx.log("route_not_found", tick=tick)
            return
        origin_body = result.get("origin_body")
        origin_heading = result.get("origin_heading")
        origin_food = result.get("origin_food")
        route = list(result.get("route", ()))
        if not route or origin_food != food:
            self.ctx.log("route_discarded", reason="food_changed", tick=tick)
            return

        # A worker may finish one or two ticks after it started.  Accept only
        # an exact prefix match, so a late result can never smuggle a plan into
        # a different body geometry.
        cur_body, cur_heading, cur_food = origin_body, origin_heading, origin_food
        accepted_at = None
        for i, action in enumerate(route):
            if cur_body == body and cur_heading == heading and cur_food == food:
                accepted_at = i
                break
            result_step = _transition(cur_body, cur_heading, cur_food, action, self.rows, self.cols)
            if result_step is None:
                break
            cur_body, cur_heading, ate = result_step
            if ate:
                cur_food = None
        if accepted_at is None or accepted_at >= len(route):
            self.ctx.log("route_discarded", reason="state_changed", tick=tick)
            return
        self.plan = route[accepted_at:]
        self.plan_kind = str(result.get("kind", "FOOD"))
        self.plan_food = food
        self.ctx.log("route_ready", tick=tick, kind=self.plan_kind, remaining=len(self.plan))

    def _start_route(self, body, heading, food, tick: int) -> None:
        if food is None or self.route_future is not None or self.plan:
            return
        if tick - self.route_started_tick < 3 and food == self.route_started_food:
            return
        snapshot = (body, heading, food, self.rows, self.cols)
        try:
            self.route_future = self.ctx.spawn(lambda: _planner_task(*snapshot))
            self.route_started_tick = tick
            self.route_started_food = food
            self.ctx.log("route_started", tick=tick, food=list(food))
        except Exception as exc:
            self.route_future = None
            self.ctx.log("route_spawn_failed", tick=tick, error=type(exc).__name__)

    def _poll_jev(self, obs: dict, allowed: list[str], tick: int) -> str | None:
        future = self.jev_future
        if future is None or not future.done():
            return None
        self.jev_future = None
        try:
            answers = future.result()
            first = answers.get("pick_a") if isinstance(answers, dict) else None
            second = answers.get("pick_b") if isinstance(answers, dict) else None
            a = getattr(first, "value", None)
            b = getattr(second, "value", None)
            judged = int(self.jev_tick if self.jev_tick is not None else tick)
            age = tick - judged
            if age < 0 or age > 2 or a != b or a not in allowed:
                self.ctx.log("jev_discarded", judged_tick=judged, current_tick=tick, a=a, b=b)
                return None
            if _probability(first, a) < 0.34 or _probability(second, a) < 0.34:
                self.ctx.log("jev_low_confidence", tick=tick, answer=a)
                return None
            self.ctx.log("jev_advice", tick=tick, answer=a, age=age)
            return a
        except Exception as exc:
            self.ctx.log("jev_ignored", tick=tick, error=type(exc).__name__)
            return None

    def _start_jev(
        self,
        obs: dict,
        body,
        heading: str,
        food,
        candidates: list[str],
        delayed: list[str],
        fallback: str | None,
        tick: int,
    ) -> None:
        if (
            getattr(self.ctx, "system1", None) is None
            or self.jev_future is not None
            or self.route_future is not None
        ):
            return
        if len(candidates) < 2 or tick - self.last_jev_tick < 6:
            return
        facts = []
        for action in candidates:
            result = _transition(body, heading, food, action, self.rows, self.cols)
            if result is None:
                continue
            new_body, new_heading, ate = result
            facts.append({
                "action": action,
                "safe_for_repeated_ticks": action in delayed,
                "eats_now": bool(ate),
                "post_growth_safe": bool(not ate or _post_growth_safe(new_body, self.rows, self.cols)),
                "tail_access": bool(_tail_access(new_body, self.rows, self.cols)),
                "space": "open" if _area(new_body, self.rows, self.cols) >= (self.rows * self.cols) // 3 else "tight",
                "keeps_heading": bool(new_heading == heading),
            })
        labels = [fact["action"] for fact in facts]
        if len(labels) < 2:
            return
        descriptions = {
            fact["action"]: (
                "Code-legal candidate. "
                + ("It eats now and passed the post-growth test. " if fact["eats_now"] else "It does not eat now. ")
                + ("The next repeated ticks are shield-safe. " if fact["safe_for_repeated_ticks"] else "It needs a prompt turn. ")
                + ("Tail access is available and space is open." if fact["tail_access"] and fact["space"] == "open" else "Prefer it only if the alternatives are worse.")
            )
            for fact in facts
        }
        state = {
            "schema": "snake-junction-v2",
            "rule": "Choose exactly one listed action. Code has already checked bounds, reverse rejection, body collision, and tail-vacating.",
            "observation": {
                "tick": tick,
                "head": list(body[0]) if body else None,
                "tail": list(body[-1]) if body else None,
                "direction": heading,
                "food": list(food) if food is not None else None,
                "length": len(body),
                "score": obs.get("score", 0),
            },
            "candidates": facts,
            "fallback": fallback,
            "safety_pool": "repeated-tick-safe" if delayed else "one-step-safe",
        }
        questions = {
            "pick_a": Choice(
                "Select the safest useful action from `candidates`. Prefer a food action only when `post_growth_safe` is true; otherwise prefer tail access, open space, and a candidate safe while a next judgment is pending.",
                {key: descriptions[key] for key in labels},
            ),
            "pick_b": Choice(
                "Select exactly one action from `candidates` using the same safety criteria as pick_a. Do not invent an action.",
                {key: descriptions[key] for key in reversed(labels)},
            ),
            "urgent": Noul(
                "Does the current heading need a turn before another delayed decision can safely arrive?",
                {
                    "true": "The current heading is not in the repeated-tick-safe pool or is approaching a boundary/body.",
                    "false": "The current heading remains safe for the short delay and no immediate turn is required.",
                },
            ),
        }

        def call_jev():
            return self.ctx.decide(state, questions, timeout_s=0.55)

        self.last_jev_tick = tick
        self.jev_tick = tick
        try:
            self.jev_future = self.ctx.spawn(call_jev)
            self.ctx.log("jev_started", tick=tick, candidates=labels)
        except Exception as exc:
            self.jev_future = None
            self.ctx.log("jev_spawn_failed", tick=tick, error=type(exc).__name__)

    def choose(self, obs: Any) -> str:
        if not isinstance(obs, dict):
            return "RIGHT"
        self._update_dims(obs)
        body = _body(obs)
        heading = _heading(obs)
        food = _food(obs)
        if not body or not _inside(body[0], self.rows, self.cols):
            return heading
        try:
            tick = int(obs.get("tick", 0))
        except (TypeError, ValueError, OverflowError):
            tick = 0

        self._poll_route(obs, body, heading, food, tick)

        if self.expected is not None:
            expected_body, expected_heading, expected_food, food_was_eaten = self.expected
            matches = body == expected_body and heading == expected_heading
            if not food_was_eaten:
                matches = matches and food == expected_food
            if not matches or food_was_eaten:
                self._clear_plan()
            self.expected = None
        if self.plan and (self.plan_food != food or not self.plan):
            self._clear_plan()

        one, delayed = _safe_sets(body, heading, food, self.rows, self.cols, horizon=3)
        pool = delayed or one
        action = None
        used_plan = False
        if self.plan:
            candidate = self.plan[0]
            if candidate in one:
                action = candidate
                self.plan.pop(0)
                used_plan = True
            else:
                self._clear_plan()

        # The cycle is only an order-checked late-game fallback.  It is never
        # used to repair an arbitrary shortcut that broke the invariant.
        if action is None and len(body) * 3 >= self.rows * self.cols and self.cycles_enabled:
            cycle_action = _cycle_step(body, heading, food, self.rows, self.cols)
            if cycle_action in pool or (not pool and cycle_action in one):
                action = cycle_action

        if action is None:
            candidates = pool or one
            scored = []
            for candidate, new_body, new_heading, ate in _successors(
                body, heading, food, self.rows, self.cols
            ):
                if candidate not in candidates:
                    continue
                if ate and not _post_growth_safe(new_body, self.rows, self.cols):
                    continue
                space = _area(new_body, self.rows, self.cols)
                tail = 1 if _tail_access(new_body, self.rows, self.cols) else 0
                food_distance = 0 if food is None else abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
                scored.append((tail, space, -food_distance, int(new_heading == heading), candidate))
            if scored:
                scored.sort(reverse=True)
                action = scored[0][-1]

        # A Jev answer can only replace a code-selected fallback, never an
        # already verified route or an empty safety set.
        advice = self._poll_jev(obs, pool, tick)
        if advice is not None and not used_plan and advice in pool:
            action = advice

        if action is None:
            action = heading if heading in DIRECTIONS else "RIGHT"

        if not self.plan and food is not None:
            self._start_route(body, heading, food, tick)
        if not used_plan and len(pool) >= 2:
            self._start_jev(obs, body, heading, food, pool, delayed, action, tick)

        result = _transition(body, heading, food, action, self.rows, self.cols)
        if result is None:
            self.expected = None
        else:
            new_body, new_heading, ate = result
            self.expected = (new_body, new_heading, food, ate)
            if ate:
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
            except Exception as exc:
                self.ctx.log("act_ended", error=type(exc).__name__)
                return
            try:
                self.ctx.wait_for_tick()
            except Exception as exc:
                if self.ctx.done or getattr(game, "done", False):
                    return
                self.ctx.log("wait_ended", error=type(exc).__name__)
                return
            if self.ctx.done or getattr(game, "done", False):
                return
            obs = game.observe()


def build(ctx):
    return Harness(ctx)
