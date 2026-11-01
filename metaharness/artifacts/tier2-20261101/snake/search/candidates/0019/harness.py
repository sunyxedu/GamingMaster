"""Candidate 0019: concurrent invariant-route and cycle-reentry planning.

The active Hamiltonian tour remains the always-ready safety policy. Two bounded
workers may improve its food throughput: a parent-style search constrained to
the active tour, and (only on a crowded board) a search that proves exact body
transitions and rejoins some independently validated tour at the food move.
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Any


DIRS = ("UP", "RIGHT", "DOWN", "LEFT")
DELTA = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
BACK = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _point(x: Any) -> tuple[int, int] | None:
    if not isinstance(x, (list, tuple)) or len(x) < 2:
        return None
    try:
        return int(x[0]), int(x[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _state(obs: Any):
    if not isinstance(obs, dict):
        return None
    try:
        rows, cols = int(obs["rows"]), int(obs["cols"])
        tick = max(0, int(obs.get("tick", 0)))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if rows < 2 or cols < 2 or rows * cols > 4096:
        return None
    raw = obs.get("snake")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    body = tuple(_point(cell) for cell in raw)
    if any(cell is None for cell in body):
        return None
    body = tuple(body)
    food = _point(obs.get("food")) if obs.get("food") is not None else None
    heading = obs.get("direction")
    if heading not in DELTA:
        if len(body) >= 2:
            delta = body[0][0] - body[1][0], body[0][1] - body[1][1]
            heading = next((d for d, v in DELTA.items() if v == delta), "RIGHT")
        else:
            heading = "RIGHT"
    return rows, cols, tick, body, heading, food


def _inside(p: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= p[0] < rows and 0 <= p[1] < cols


def _near(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _valid_tour(tour: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    n = rows * cols
    return (len(tour) == n and len(set(tour)) == n
            and all(_inside(p, rows, cols) for p in tour)
            and all(_near(tour[i], tour[(i + 1) % n]) for i in range(n)))


def _serpentine(rows: int, cols: int):
    if rows < 2 or cols < 2 or rows % 2:
        return None
    cells = [(0, c) for c in range(cols)]
    for r in range(1, rows):
        cs = range(cols - 1, 0, -1) if r & 1 else range(1, cols)
        cells.extend((r, c) for c in cs)
    cells.append((rows - 1, 0))
    cells.extend((r, 0) for r in range(rows - 2, 0, -1))
    result = tuple(cells)
    return result if _valid_tour(result, rows, cols) else None


def _link(g, a, b):
    g.setdefault(a, set()).add(b)
    g.setdefault(b, set()).add(a)


def _unlink(g, a, b):
    g.get(a, set()).discard(b)
    g.get(b, set()).discard(a)


def _walk(g, rows, cols):
    if len(g) != rows * cols or any(len(neighbors) != 2 for neighbors in g.values()):
        return None
    start, prev, cur = (0, 0), None, (0, 0)
    tour = [start]
    while True:
        options = sorted(g[cur] - ({prev} if prev is not None else set()))
        if not options:
            return None
        nxt = options[0]
        if nxt == start:
            break
        if nxt in tour:
            return None
        tour.append(nxt)
        prev, cur = cur, nxt
        if len(tour) > rows * cols:
            return None
    result = tuple(tour)
    return result if _valid_tour(result, rows, cols) else None


def _strip_tour(rows: int, cols: int, joins: tuple[int, ...]):
    if rows < 2 or cols < 2 or rows % 2 or len(joins) != rows // 2 - 1:
        return None
    g = {}
    for r in range(0, rows, 2):
        ring = ([(r, c) for c in range(cols)]
                + [(r + 1, c) for c in range(cols - 1, -1, -1)])
        for i, cell in enumerate(ring):
            _link(g, cell, ring[(i + 1) % len(ring)])
    for strip, col in enumerate(joins):
        if not 0 <= col < cols - 1:
            return None
        top, bottom = 2 * strip + 1, 2 * strip + 2
        _unlink(g, (top, col), (top, col + 1))
        _unlink(g, (bottom, col), (bottom, col + 1))
        _link(g, (top, col), (bottom, col))
        _link(g, (top, col + 1), (bottom, col + 1))
    return _walk(g, rows, cols)


def _join_patterns(nstrips: int, width: int, limit: int):
    if nstrips <= 0 or width <= 0:
        return [()]
    patterns = []
    seen = set()

    def put(seq):
        val = tuple(int(x) % width for x in seq)
        if val not in seen:
            seen.add(val)
            patterns.append(val)

    for c in (0, width - 1, width // 2):
        put([c] * nstrips)
    for phase in range(width):
        put([(phase + i) % width for i in range(nstrips)])
        put([(phase - i) % width for i in range(nstrips)])
    for i in range(nstrips):
        p = [0] * nstrips
        p[i] = (i * 3 + 2) % width
        put(p)
    # The small fixed recurrence only diversifies geometry; it uses no episode seed.
    x = 0x4D595DF4
    while len(patterns) < limit:
        p = []
        for _ in range(nstrips):
            x = (1664525 * x + 1013904223) & 0xFFFFFFFF
            p.append(x % width)
        put(p)
    return patterns[:limit]


def _tour_family(rows: int, cols: int):
    bases = []
    normal = _serpentine(rows, cols)
    if normal is not None:
        bases.append(normal)
    if cols % 2 == 0:
        transposed = _serpentine(cols, rows)
        if transposed is not None:
            bases.append(tuple((c, r) for r, c in transposed))
    if rows % 2 == 0:
        for pat in _join_patterns(rows // 2 - 1, cols - 1, 20):
            t = _strip_tour(rows, cols, pat)
            if t is not None:
                bases.append(t)
    if cols % 2 == 0:
        for pat in _join_patterns(cols // 2 - 1, rows - 1, 20):
            t = _strip_tour(cols, rows, pat)
            if t is not None:
                bases.append(tuple((c, r) for r, c in t))

    out, seen = [], set()
    for base in bases:
        for fr in (False, True):
            for fc in (False, True):
                mapped = tuple((rows - 1 - r if fr else r,
                                cols - 1 - c if fc else c) for r, c in base)
                for candidate in (mapped, tuple(reversed(mapped))):
                    if candidate not in seen and _valid_tour(candidate, rows, cols):
                        seen.add(candidate)
                        out.append(candidate)
                        if len(out) >= 192:
                            return out
    return out


def _index(tour):
    return {cell: i for i, cell in enumerate(tour)}


def _in_order(body, positions, size):
    if not body or len(set(body)) != len(body):
        return False
    if any(p not in positions for p in body):
        return False
    span = 0
    for newer, older in zip(body, body[1:]):
        gap = (positions[newer] - positions[older]) % size
        if gap <= 0:
            return False
        span += gap
    return span < size


def _move(body, heading, request, food, rows, cols):
    if not body or heading not in DELTA or request not in DELTA:
        return None
    direction = heading if request == BACK.get(heading) else request
    dr, dc = DELTA[direction]
    head = body[0][0] + dr, body[0][1] + dc
    if not _inside(head, rows, cols):
        return None
    ate = food is not None and head == food
    if head in set(body if ate else body[:-1]):
        return None
    return (head,) + (body if ate else body[:-1]), direction, ate


def _space(body, rows, cols):
    blocked = set(body[1:])
    q, reached = deque([body[0]]), {body[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in DELTA.values():
            p = r + dr, c + dc
            if _inside(p, rows, cols) and p not in blocked and p not in reached:
                reached.add(p)
                q.append(p)
    return len(reached)


def _tour_action(body, heading, food, rows, cols, tour, positions):
    if not _in_order(body, positions, len(tour)):
        return None
    following = tour[(positions[body[0]] + 1) % len(tour)]
    choices = []
    for action in DIRS:
        if action == BACK.get(heading):
            continue
        moved = _move(body, heading, action, food, rows, cols)
        if moved is None:
            continue
        new_body, new_heading, ate = moved
        if not _in_order(new_body, positions, len(tour)):
            continue
        distance = 0 if ate or food is None else (positions[food] - positions[new_body[0]]) % len(tour)
        choices.append(((int(ate), -distance, int(new_body[0] == following),
                         int(new_heading == heading), action), action)
        )
    return max(choices)[1] if choices else None


def _tour_route(body, heading, food, rows, cols, tour, positions, first=None):
    if food is None or not _in_order(body, positions, len(tour)):
        return None
    path, b, h = [], body, heading
    for _ in range(len(tour) + len(body) + 5):
        action = _tour_action(b, h, food, rows, cols, tour, positions)
        if not path and first in DIRS:
            action = first
        if action is None:
            return None
        moved = _move(b, h, action, food, rows, cols)
        if moved is None or not _in_order(moved[0], positions, len(tour)):
            return None
        path.append(action)
        b, h, ate = moved
        if ate:
            return tuple(path)
    return None


def _find_food(origin, direction, food, rows, cols, active, family,
               mode, seconds=0.040, expansion_limit=6500, stop=None):
    """Bounded A*: exact full-body moves, plus an order certificate as requested."""
    if not origin or food is None or origin[0] == food:
        return None
    active_tour, active_pos = active
    base = _tour_route(origin, direction, food, rows, cols,
                       active_tour, active_pos)
    base_len = len(base) if base else None
    max_depth = min(rows * cols + len(origin) + 4, 220)
    if base_len is not None:
        max_depth = min(max_depth, base_len - 1)
    if max_depth <= 0:
        return None

    deadline = time.monotonic() + seconds
    start_h = abs(origin[0][0] - food[0]) + abs(origin[0][1] - food[1])
    heap = [(float(start_h), 0, 0, origin, direction, ())]
    best = {(origin, direction): (0, 0)}
    expanded = 0
    while heap and expanded < expansion_limit and time.monotonic() < deadline:
        if stop is not None:
            try:
                if stop():
                    return None
            except Exception:
                return None
        _, depth, turns, body, heading, path = heapq.heappop(heap)
        expanded += 1
        if depth >= max_depth:
            continue
        ranked = []
        for action in DIRS:
            if action == BACK.get(heading):
                continue
            dr, dc = DELTA[action]
            p = body[0][0] + dr, body[0][1] + dc
            ranked.append((abs(p[0] - food[0]) + abs(p[1] - food[1]),
                           int(action != heading), action))
        ranked.sort()
        for _, _, action in ranked:
            moved = _move(body, heading, action, food, rows, cols)
            if moved is None:
                continue
            new_body, new_heading, ate = moved
            if mode == "cycle" and not _in_order(new_body, active_pos, rows * cols):
                continue
            new_path = path + (action,)
            if ate:
                endpoint = None
                if mode == "cycle":
                    endpoint = active_tour
                else:
                    endpoint = next((tour for tour, pos in family
                                     if _in_order(new_body, pos, rows * cols)), None)
                if endpoint is not None and (base_len is None or len(new_path) < base_len):
                    return {"origin": origin, "heading": direction, "food": food,
                            "route": new_path, "baseline": base_len,
                            "end_tour": endpoint, "mode": mode, "expanded": expanded}
                continue
            new_depth = depth + 1
            remaining = abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
            # No strictly improving solution is possible beyond this admissible bound.
            if base_len is not None and new_depth + remaining >= base_len:
                continue
            new_turns = turns + int(new_heading != heading)
            key = (new_body, new_heading)
            old = best.get(key)
            if old is not None and old <= (new_depth, new_turns):
                continue
            best[key] = (new_depth, new_turns)
            # A tiny tour-distance tie breaker only orders nodes; legality is exact.
            tour_distance = 0
            if food in active_pos:
                tour_distance = (active_pos[food] - active_pos[new_body[0]]) % (rows * cols)
            heapq.heappush(heap, (new_depth + remaining + 0.001 * tour_distance,
                                  new_depth, new_turns, new_body, new_heading, new_path))
    return None


def _suffix(result, body, heading, food, rows, cols):
    if not isinstance(result, dict) or result.get("food") != food:
        return None
    b, h = result.get("origin"), result.get("heading")
    path = result.get("route")
    if not isinstance(b, tuple) or not isinstance(path, tuple) or not path:
        return None
    for i, action in enumerate(path):
        if b == body and h == heading:
            return path[i:]
        moved = _move(b, h, action, food, rows, cols)
        if moved is None:
            return None
        b, h, ate = moved
        if ate:
            return None
    return path[len(path):] if b == body and h == heading else None


class _Pilot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rows = self.cols = 0
        self.family = []
        self.active = None
        self.plan = []
        self.plan_mode = None
        self.plan_food = None
        self.plan_end = None
        self.expected = None
        self.safe_future = None
        self.reentry_future = None
        self.last_safe_tick = -100000
        self.last_reentry_tick = -100000
        self.last_reentry_key = None

    def log(self, event, **data):
        try:
            self.ctx.log(event, **data)
        except Exception:
            pass

    def configure(self, rows, cols):
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        self.family = [(t, _index(t)) for t in _tour_family(rows, cols)]
        self.active = None
        self.plan = []
        self.plan_mode = self.plan_food = self.plan_end = None
        self.expected = None
        self.safe_future = self.reentry_future = None
        self.log("tour_family_verified", rows=rows, cols=cols, count=len(self.family))

    def select_tour(self, body, heading, food):
        options = []
        for i, (tour, pos) in enumerate(self.family):
            action = _tour_action(body, heading, food, self.rows, self.cols, tour, pos)
            if action is None:
                continue
            moved = _move(body, heading, action, food, self.rows, self.cols)
            if moved is None:
                continue
            if food is None or moved[2]:
                distance = 0
            else:
                distance = (pos[food] - pos[moved[0][0]]) % len(tour)
            options.append(((int(moved[2]), -distance, int(tour == self.active),
                             int(moved[1] == heading), -i), action, tour, pos))
        if not options:
            self.active = None
            return None
        options.sort(reverse=True)
        _, action, tour, pos = options[0]
        if self.active != tour:
            self.log("tour_selected", alternatives=len(options), tick=0)
        self.active = tour
        return action

    def poll(self, future, mode, body, heading, food, tick):
        if future is None or not getattr(future, "done", lambda: False)():
            return None
        if mode == "cycle":
            self.safe_future = None
        else:
            self.reentry_future = None
        try:
            result = future.result()
        except Exception as exc:
            self.log("route_worker_failed", mode=mode, tick=tick, error=type(exc).__name__)
            return None
        if not isinstance(result, dict):
            return None
        if result.get("mode") != mode:
            return None
        route = result.get("route")
        baseline = result.get("baseline")
        if not isinstance(route, tuple) or not route or (isinstance(baseline, int) and len(route) >= baseline):
            self.log("route_rejected", mode=mode, reason="no_strict_gain", tick=tick)
            return None
        suffix = _suffix(result, body, heading, food, self.rows, self.cols)
        if not suffix:
            self.log("route_rejected", mode=mode, reason="not_live_prefix", tick=tick)
            return None
        end_tour = result.get("end_tour")
        if mode == "reentry" and not any(end_tour == tour for tour, _ in self.family):
            self.log("route_rejected", mode=mode, reason="unverified_endpoint", tick=tick)
            return None
        if mode == "cycle" and (self.active is None or not _in_order(body, _index(self.active), self.rows * self.cols)):
            return None
        return (len(suffix), list(suffix), mode, end_tour)

    def clear_plan(self):
        self.plan = []
        self.plan_mode = self.plan_food = self.plan_end = None
        self.expected = None

    def choose(self, obs, advertised):
        parsed = _state(obs)
        if parsed is None:
            return advertised[0] if advertised else "NOOP"
        rows, cols, tick, body, heading, food = parsed
        self.configure(rows, cols)
        score = obs.get("score", 0) if isinstance(obs, dict) else 0
        try:
            score = int(score)
        except (TypeError, ValueError, OverflowError):
            score = 0

        if self.active is not None and not _in_order(body, _index(self.active), rows * cols):
            self.log("active_tour_invalid", tick=tick)
            self.active = None
            self.clear_plan()

        if self.expected is not None:
            eb, eh, ef = self.expected
            if body != eb or heading != eh or food != ef:
                self.log("route_rejected", mode=self.plan_mode or "unknown",
                         reason="state_mismatch", tick=tick)
                self.clear_plan()
            else:
                self.expected = None
        if self.plan_food != food:
            self.clear_plan()

        found = []
        if not self.plan:
            a = self.poll(self.safe_future, "cycle", body, heading, food, tick)
            if a is not None:
                found.append(a)
            a = self.poll(self.reentry_future, "reentry", body, heading, food, tick)
            if a is not None:
                found.append(a)
            if found:
                found.sort(key=lambda x: x[0])
                _, self.plan, self.plan_mode, self.plan_end = found[0]
                self.plan_food = food
                self.log("route_installed", mode=self.plan_mode, tick=tick,
                         remaining=len(self.plan))

        action = None
        if self.plan:
            candidate = self.plan[0]
            moved = _move(body, heading, candidate, food, rows, cols)
            valid = moved is not None and candidate in advertised
            if valid and self.plan_mode == "cycle":
                valid = self.active is not None and _in_order(moved[0], _index(self.active), rows * cols)
            if valid and moved[2]:
                endpoint = self.active if self.plan_mode == "cycle'" else self.plan_end
                valid = endpoint is not None and _in_order(moved[0], _index(endpoint), rows * cols)
            if valid:
                action = candidate
                self.plan.pop(0)
                if moved[2]:
                    if self.plan_mode == "reentry":
                        self.active = self.plan_end
                    self.clear_plan()
                else:
                    self.expected = (moved[0], moved[1], food)
            else:
                self.log("route_rejected", mode=self.plan_mode or "unknown",
                         reason="live_transition", tick=tick)
                self.clear_plan()

        if action is None:
            if self.active is None:
                action = self.select_tour(body, heading, food)
            else:
                action = _tour_action(body, heading, food, rows, cols,
                                      self.active, _index(self.active))
            if action not in advertised:
                action = None
        if action is None:
            choices = []
            for candidate in DIRS:
                if candidate not in advertised or candidate == BACK.get(heading):
                    continue
                moved = _move(body, heading, candidate, food, rows, cols)
                if moved is None:
                    continue
                new_body, new_heading, ate = moved
                dist = 0 if food is None else abs(new_body[0][0] - food[0]) + abs(new_body[0][1] - food[1])
                choices.append(((int(ate), _space(new_body, rows, cols), -dist,
                                 int(new_heading == heading)), candidate))
            action = max(choices)[1] if choices else (heading if heading in advertised else (advertised[0] if advertised else "NOOP"))
            if self.active is not None and action == heading:
                moved = _move(body, heading, action, food, rows, cols)
                if moved is None or not _in_order(moved[0], _index(self.active), rows * cols):
                    self.active = None

        # Queue searches from the known post-action body, preserving the parent
        # controller's free-first-step freshness contract.
        next_state = _move(body, heading, action, food, rows, cols)
        if next_state is not None and not next_state[2] and self.active is not None and not self.plan:
            post_body, post_heading, _ = next_state
            active_pair = (self.active, _index(self.active))
            if (self.safe_future is None and tick - self.last_safe_tick >= 2
                    and food is not None):
                args = (post_body, post_heading, food, rows, cols,
                        active_pair, tuple(self.family), "cycle", 0.040, 6500,
                        lambda: bool(getattr(self.ctx, "done", False)))
                try:
                    self.safe_future = self.ctx.spawn(lambda a=args: _find_food(*a))
                    self.last_safe_tick = tick
                except Exception as exc:
                    self.log("route_spawn_failed", mode="cycle", error=type(exc).__name__)
            free = rows * cols - len(body)
            key = (score, food)
            if (free <= min(48, rows * cols // 3) and food is not None
                    and self.reentry_future is None
                    and tick - self.last_reentry_tick >= 8
                    and (self.last_reentry_key != key or tick - self.last_reentry_tick >= 16)):
                # The route worker itself checks strict improvement over the
                # active tour's exact greedy food distance.
                args = (post_body, post_heading, food, rows, cols,
                        active_pair, tuple(self.family), "reentry", 0.050, 7500,
                        lambda: bool(getattr(self.ctx, "done", False)))
                try:
                    self.reentry_future = self.ctx.spawn(lambda a=args: _find_food(*a))
                    self.last_reentry_tick = tick
                    self.last_reentry_key = key
                except Exception as exc:
                    self.log("route_spawn_failed", mode="reentry", error=type(exc).__name__)
        return action


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        if not getattr(game, "tick_s", None):
            return
        spec = getattr(game, "action_spec", None)
        names = tuple(x for x in (getattr(spec, "names", ()) or ()) if isinstance(x, str))
        advertised = tuple(d for d in DIRS if d in names)
        if not advertised:
            noop = getattr(spec, "noop", None)
            advertised = (noop,) if isinstance(noop, str) and noop in names else names
        if not advertised:
            return
        pilot = _Pilot(self.ctx)
        while not self.ctx.done and not getattr(game, "done", False):
            action = pilot.choose(game.observe(), advertised)
            if action not in names:
                action = advertised[0]
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
