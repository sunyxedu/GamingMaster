"""Candidate 0005: a clock-safe Snake controller with exact local search.

The parents either asked Jev for an unconstrained action (so the first useful
answer arrived after the snake had already travelled into a wall) or did not
post an action at all.  This controller makes the mechanical part explicit:
it simulates the published transition, plans a safe food route, and always
has a nonblocking tail/area fallback.  Jev and Astra are advisory only; a late
or malformed model answer can never be the only thing keeping the snake alive.
"""
from __future__ import annotations

import heapq
import json
from collections import deque
from typing import Any

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
VEC = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _cell(x: Any) -> tuple[int, int] | None:
    try:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            return (int(x[0]), int(x[1]))
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _body_from(obs: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(obs, dict):
        return ()
    raw = obs.get("snake", ())
    if not isinstance(raw, (list, tuple)):
        return ()
    out = []
    for x in raw:
        p = _cell(x)
        if p is None:
            return ()
        out.append(p)
    return tuple(out)


def _heading(obs: Any) -> str:
    d = obs.get("direction") if isinstance(obs, dict) else None
    return d if d in DIRECTIONS else "RIGHT"


def _food(obs: Any) -> tuple[int, int] | None:
    return _cell(obs.get("food")) if isinstance(obs, dict) and obs.get("food") is not None else None


def _in_bounds(p: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= p[0] < rows and 0 <= p[1] < cols


def _accepted_heading(request: str, heading: str) -> str:
    if request in DIRECTIONS and request != OPPOSITE.get(heading):
        return request
    return heading


def _transition(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    request: str,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """Apply one published Snake transition, including tail vacating."""
    if not body:
        return None
    new_heading = _accepted_heading(request, heading)
    dr, dc = VEC[new_heading]
    nxt = (body[0][0] + dr, body[0][1] + dc)
    if not _in_bounds(nxt, rows, cols):
        return None
    ate = food is not None and nxt == food
    occupied = set(body if ate else body[:-1])
    if nxt in occupied:
        return None
    new_body = (nxt,) + (body if ate else body[:-1])
    return new_body, new_heading, ate


def _successors(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
):
    seen_headings: set[str] = set()
    for action in DIRECTIONS:
        # Asking for the reverse is ignored by the game and duplicates straight.
        if action == OPPOSITE.get(heading):
            continue
        result = _transition(body, heading, food, action, rows, cols)
        if result is None:
            continue
        nb, nh, ate = result
        if nh in seen_headings:
            continue
        seen_headings.add(nh)
        yield action, nb, nh, ate


def _area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    """Free-space component of the head; the head itself is counted."""
    if not body:
        return 0
    blocked = set(body[1:])
    start = body[0]
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in VEC.values():
            p = (r + dr, c + dc)
            if _in_bounds(p, rows, cols) and p not in blocked and p not in seen:
                seen.add(p)
                q.append(p)
    return len(seen)


def _tail_reachable(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    """Static tail-access test used as a filter, never as a proof of infinity."""
    if len(body) <= 1:
        return True
    start, target = body[0], body[-1]
    blocked = set(body[1:-1])
    q = deque([start])
    seen = {start}
    while q:
        p = q.popleft()
        if p == target:
            return True
        for dr, dc in VEC.values():
            n = (p[0] + dr, p[1] + dc)
            if not _in_bounds(n, rows, cols) or n in blocked or n in seen:
                continue
            seen.add(n)
            q.append(n)
    return False


def _tail_distance(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if len(body) <= 1:
        return 0
    start, target = body[0], body[-1]
    blocked = set(body[1:-1])
    q = deque([(start, 0)])
    seen = {start}
    while q:
        p, d = q.popleft()
        if p == target:
            return d
        for dr, dc in VEC.values():
            n = (p[0] + dr, p[1] + dc)
            if _in_bounds(n, rows, cols) and n not in blocked and n not in seen:
                seen.add(n)
                q.append((n, d + 1))
    return rows * cols + 1


def _post_food_ok(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    if len(body) >= rows * cols:
        return True
    # Tail access is the principal test.  The small area guard rejects a
    # visibly sealed one-cell pocket but does not demand an impossible margin
    # late in a nearly filled board.
    return _tail_reachable(body, rows, cols) and _area(body, rows, cols) >= 2


def _food_plan(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    conservative: bool = False,
) -> list[str] | None:
    """A* over body states; return the first short route passing post-growth safety."""
    if not body or food is None or body[0] == food:
        return None
    # The full body is deliberately in the key: a static path is not enough
    # because tail-vacating and growth differ on the final move.
    start_key = (body, heading)
    heap: list[tuple[int, int, int, tuple[tuple[int, int], ...], str, tuple[str, ...]]] = []
    serial = 0
    h0 = abs(body[0][0] - food[0]) + abs(body[0][1] - food[1])
    heapq.heappush(heap, (h0, 0, serial, body, heading, ()))
    seen = {start_key}
    expansions = 0
    max_expansions = 9000 if len(body) < (rows * cols) // 3 else 3500
    max_depth = min(rows * cols + len(body), 220)
    accepted: list[tuple[tuple[int, int], tuple[str, ...], int, int]] = []
    first_goal_depth: int | None = None

    while heap and expansions < max_expansions:
        _, depth, _, cur_body, cur_heading, path = heapq.heappop(heap)
        expansions += 1
        if first_goal_depth is not None and depth > first_goal_depth + 8 and accepted:
            break
        if depth >= max_depth:
            continue
        for action, nb, nh, ate in _successors(cur_body, cur_heading, food, rows, cols):
            npath = path + (action,)
            if ate:
                if first_goal_depth is None:
                    first_goal_depth = depth + 1
                if _post_food_ok(nb, rows, cols):
                    # Larger reachable space breaks ties among similarly short
                    # routes.  Conservative supervision only raises the bar
                    # when occupancy is already substantial.
                    a = _area(nb, rows, cols)
                    td = _tail_distance(nb, rows, cols)
                    if not conservative or len(nb) * 2 < rows * cols or _tail_reachable(nb, rows, cols):
                        accepted.append((nb, npath, a, td))
                        if len(accepted) >= 5 and depth + 1 >= (first_goal_depth or 0) + 2:
                            break
                continue
            key = (nb, nh)
            if key in seen:
                continue
            seen.add(key)
            r, c = nb[0]
            heuristic = abs(r - food[0]) + abs(c - food[1])
            # A tiny turn cost is only a tie-breaker; never sacrifice distance
            # or safety for aesthetics.
            turn_cost = 0 if nh == cur_heading else 1
            serial += 1
            heapq.heappush(heap, (len(npath) + heuristic + turn_cost * 0.01,
                                  len(npath), serial, nb, nh, npath))
        if accepted and first_goal_depth is not None and depth >= first_goal_depth + 2:
            break

    if not accepted:
        return None
    accepted.sort(key=lambda x: (len(x[1]), -x[2], x[3]))
    return list(accepted[0][1])


def _tail_plan(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> list[str] | None:
    """Find a collision-free route toward the *moving* old tail."""
    if len(body) < 2:
        return None
    target = body[-1]
    start_key = (body, heading)
    q = deque([(body, heading, ())])
    seen = {start_key}
    expansions = 0
    while q and expansions < 5000:
        cur_body, cur_heading, path = q.popleft()
        expansions += 1
        for action, nb, nh, ate in _successors(cur_body, cur_heading, None, rows, cols):
            # Do not grow while the controller is explicitly buying time.
            if ate:
                continue
            npath = path + (action,)
            if nb[0] == target:
                return list(npath)
            key = (nb, nh)
            if key in seen or len(npath) >= min(rows * cols, 160):
                continue
            seen.add(key)
            q.append((nb, nh, npath))
    return None


def _fallback_action(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[str, tuple[tuple[int, int], ...], str, bool] | None:
    candidates = []
    for action, nb, nh, ate in _successors(body, heading, food, rows, cols):
        if ate and not _post_food_ok(nb, rows, cols):
            continue
        area = _area(nb, rows, cols)
        tail_ok = _tail_reachable(nb, rows, cols)
        td = _tail_distance(nb, rows, cols)
        fd = 0 if food is None else abs(nb[0][0] - food[0]) + abs(nb[0][1] - food[1])
        # Prefer a route with tail access and space; food distance is only a
        # secondary preference once the survival predicates agree.
        score = (1 if tail_ok else 0, area, -td, -fd, 1 if action == heading else 0)
        candidates.append((score, action, nb, nh, ate))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, action, nb, nh, ate = candidates[0]
    return action, nb, nh, ate


def _add_edge(adj: dict[tuple[int, int], set[tuple[int, int]]], a, b) -> None:
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)


def _remove_edge(adj, a, b) -> None:
    adj[a].discard(b)
    adj[b].discard(a)


def _pair_cycle(even_rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Build a cycle by merging the per-two-row perimeter cycles."""
    if even_rows < 2 or cols < 2 or even_rows % 2:
        return None
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for r in range(0, even_rows, 2):
        order = [(r, c) for c in range(cols)] + [(r + 1, c) for c in range(cols - 1, -1, -1)]
        for i, a in enumerate(order):
            _add_edge(adj, a, order[(i + 1) % len(order)])
    # Exchange one horizontal edge in each adjacent pair of strips for two
    # vertical boundary edges.  A 2-regular graph remains one merged cycle.
    for upper in range(1, even_rows - 1, 2):
        lower = upper + 1
        _remove_edge(adj, (upper, 0), (upper, 1))
        _remove_edge(adj, (lower, 0), (lower, 1))
        _add_edge(adj, (upper, 0), (lower, 0))
        _add_edge(adj, (upper, 1), (lower, 1))
    n = even_rows * cols
    if any(len(v) != 2 for v in adj.values()) or len(adj) != n:
        return None
    start = (0, 0)
    out = [start]
    prev = None
    cur = start
    while True:
        options = [x for x in adj[cur] if x != prev]
        if not options:
            return None
        nxt = options[0]
        if nxt == start:
            break
        if nxt in out:
            return None
        out.append(nxt)
        prev, cur = cur, nxt
    if len(out) != n:
        return None
    for i, p in enumerate(out):
        q = out[(i + 1) % n]
        if abs(p[0] - q[0]) + abs(p[1] - q[1]) != 1:
            return None
    return out


def _cycles(rows: int, cols: int) -> list[list[tuple[int, int]]]:
    result: list[list[tuple[int, int]]] = []
    if rows % 2 == 0:
        c = _pair_cycle(rows, cols)
        if c:
            result.append(c)
    if cols % 2 == 0:
        t = _pair_cycle(cols, rows)
        if t:
            result.append([(c, r) for r, c in t])
    return result


def _cycle_action(
    body: tuple[tuple[int, int], ...], heading: str, rows: int, cols: int,
    cycles: list[list[tuple[int, int]]], food: tuple[int, int] | None,
) -> str | None:
    if len(body) < 2:
        return None
    for cycle in cycles:
        index = {p: i for i, p in enumerate(cycle)}
        if any(p not in index for p in body):
            continue
        n = len(cycle)
        for orient in (1, -1):
            # Body cells may have gaps after a shortcut.  They must nevertheless
            # occur in one tail-to-head cyclic order without wrapping twice.
            total = 0
            ordered = True
            for i in range(1, len(body)):
                gap = ((index[body[i - 1]] - index[body[i]]) * orient) % n
                if gap <= 0:
                    ordered = False
                    break
                total += gap
            if not ordered or total >= n:
                continue
            hidx = index[body[0]]
            next_cell = cycle[(hidx + orient) % n]
            dr = next_cell[0] - body[0][0]
            dc = next_cell[1] - body[0][1]
            requested = next((d for d, (vr, vc) in VEC.items() if (vr, vc) == (dr, dc)), None)
            if requested is None or requested == OPPOSITE.get(heading):
                continue
            result = _transition(body, heading, food, requested, rows, cols)
            if result is not None:
                return requested
    return None


class SnakeController:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.rows = 12
        self.cols = 12
        self.plan: list[str] = []
        self.plan_food: tuple[int, int] | None = None
        self.expected: tuple[tuple[tuple[int, int], ...], str, tuple[int, int] | None, bool] | None = None
        self.cycles: list[list[tuple[int, int]]] = []
        self.policy = "balanced"
        self.astra_future = None
        self.jev_future = None
        self.jev_tick: int | None = None
        self.jev_alternatives: list[str] = []
        self.last_jev_start_tick = -1000
        self.last_score = 0
        self._init_models = False

    def _dimensions(self, obs: dict) -> None:
        try:
            r, c = int(obs.get("rows", 12)), int(obs.get("cols", 12))
            if r >= 2 and c >= 2:
                if (r, c) != (self.rows, self.cols):
                    self.rows, self.cols = r, c
                    self.cycles = _cycles(r, c)
        except (TypeError, ValueError, OverflowError):
            pass

    def _start_astra(self, obs: dict) -> None:
        if self._init_models or not getattr(self.ctx, "system1", None):
            return
        # Delay Astra until deterministic code has demonstrated that it can
        # keep the clock alive.  The call is advisory and never on the tick path.
        try:
            tick = int(obs.get("tick", 0))
        except (TypeError, ValueError):
            tick = 0
        if tick < 20 or self.astra_future is not None:
            return
        self._init_models = True
        prompt = (
            "You supervise a deterministic real-time Snake controller. The code already "
            "enforces walls, reverse rejection, exact tail-vacating, dynamic route simulation, "
            "and a post-food tail-access filter. Review this public state summary and return "
            "one short JSON object with risk_mode equal to conservative, balanced, or aggressive. "
            "Choose conservative only when occupancy is high or the state is cramped; choose "
            "aggressive only when occupancy is low and food routes have ample tail access. "
            "Do not give coordinates or an action. Invalid output is ignored.\n\nSTATE:\n" +
            json.dumps({
                "rows": obs.get("rows"), "cols": obs.get("cols"),
                "tick": obs.get("tick"), "length": obs.get("length"),
                "free_cells": obs.get("free_cells"), "score": obs.get("score"),
                "head": obs.get("head"), "tail": obs.get("tail"),
                "food": obs.get("food"), "direction": obs.get("direction"),
                "safe_directions_now": obs.get("safe_directions_now"),
            }, separators=(",", ":"))
        )

        def call_astra():
            return self.ctx.think(prompt, timeout_s=90.0)

        try:
            self.astra_future = self.ctx.spawn(call_astra)
            self.ctx.log("astra_started", tick=tick, role="risk_supervisor")
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("astra_spawn_failed", error=type(exc).__name__)

    def _poll_astra(self) -> None:
        f = self.astra_future
        if f is None or not f.done():
            return
        self.astra_future = None
        try:
            result = f.result()
            text = str(getattr(result, "text", result)).lower()
            if "conservative" in text:
                self.policy = "conservative"
            elif "aggressive" in text:
                self.policy = "aggressive"
            elif "balanced" in text:
                self.policy = "balanced"
            self.ctx.log("astra_policy", policy=self.policy)
        except Exception as exc:
            self.ctx.log("astra_ignored", error=type(exc).__name__)

    def _candidate_facts(self, obs, body, heading, food):
        facts = []
        for action, nb, nh, ate in _successors(body, heading, food, self.rows, self.cols):
            if ate and not _post_food_ok(nb, self.rows, self.cols):
                continue
            facts.append({
                "action": action,
                "food_now": bool(ate),
                "tail_access": bool(_tail_reachable(nb, self.rows, self.cols)),
                "space": "large" if _area(nb, self.rows, self.cols) >= (self.rows * self.cols) // 3 else "limited",
                "near_food": "near" if food is not None and abs(nb[0][0] - food[0]) + abs(nb[0][1] - food[1]) <= 3 else "far",
                "keeps_heading": nh == heading,
            })
        return facts

    def _maybe_start_jev(self, obs, facts, preferred: str | None) -> None:
        if not getattr(self.ctx, "system1", None) or self.jev_future is not None:
            return
        try:
            tick = int(obs.get("tick", 0))
        except (TypeError, ValueError):
            return
        if tick - self.last_jev_start_tick < 8 or len(facts) < 2:
            return
        labels = [x["action"] for x in facts]
        # An advisory call is useful only at a genuine junction; all options
        # have already passed the exact transition filter above.
        descriptions = {
            x["action"]: (
                "This action is code-legal. "
                + ("It eats the current food and passed the post-growth tail-access test. " if x["food_now"] else "It does not eat immediately. ")
                + ("The resulting head can reach the tail. " if x["tail_access"] else "The static tail-access check is uncertain. ")
                + ("The connected free region is large." if x["space"] == "large" else "The connected free region is limited.")
            ) for x in facts
        }
        reverse_labels = list(reversed(labels))
        state = {
            "rule": "Choose only among code-legal candidate actions. Do not invent an action or do arithmetic.",
            "observation": {
                "tick": tick, "head": obs.get("head"), "tail": obs.get("tail"),
                "direction": _heading(obs), "food": obs.get("food"),
                "length": obs.get("length"), "free_cells": obs.get("free_cells"),
                "score": obs.get("score"),
            },
            "policy": self.policy,
            "candidates": facts,
            "deterministic_preference": preferred,
        }
        questions = {
            "pick_a": Choice(
                "Select the safest useful action from `candidates`. Prefer a food_now candidate only when its description says it passed the post-growth safety test; otherwise prefer tail access and large space.",
                descriptions,
            ),
            "pick_b": Choice(
                "Select the safest useful action from `candidates`; use the same criteria as pick_a. Return only one listed action.",
                {k: descriptions[k] for k in reverse_labels},
            ),
            "urgent": Noul(
                "Does the current state require turning now rather than continuing the current heading?",
                {"true": "The current heading would soon approach a wall/body or a turn is needed for the safe candidate route.",
                 "false": "Continuing one more cell is not immediately dangerous."},
            ),
        }

        def call_jev():
            return self.ctx.decide(state, questions, timeout_s=0.6)

        self.last_jev_start_tick = tick
        self.jev_tick = tick
        self.jev_alternatives = labels
        try:
            self.jev_future = self.ctx.spawn(call_jev)
            self.ctx.log("jev_started", tick=tick, candidates=labels)
        except Exception as exc:
            self.jev_future = None
            self.ctx.log("jev_spawn_failed", tick=tick, error=type(exc).__name__)

    def _poll_jev(self, obs, facts) -> str | None:
        f = self.jev_future
        if f is None or not f.done():
            return None
        self.jev_future = None
        try:
            answers = f.result()
            a = answers.get("pick_a")
            b = answers.get("pick_b")
            va = getattr(a, "value", None)
            vb = getattr(b, "value", None)
            current_tick = int(obs.get("tick", 0))
            judged_tick = self.jev_tick if self.jev_tick is not None else current_tick
            # Do not let an old judgment steer a changed board.  A one-tick
            # grace is useful for a response racing the scheduler; otherwise
            # deterministic planning remains in charge.
            if current_tick - judged_tick > 1 or va != vb:
                self.ctx.log("jev_discarded", judged_tick=judged_tick, current_tick=current_tick, a=va, b=vb)
                return None
            legal = {x["action"] for x in facts}
            if va not in legal:
                self.ctx.log("jev_invalid", answer=va)
                return None
            prob = 0.0
            try:
                prob = float(getattr(a, "probabilities", {}).get(va, 0.0))
            except (TypeError, ValueError):
                prob = 0.0
            if prob < 0.34:
                self.ctx.log("jev_low_confidence", answer=va, probability=prob)
                return None
            self.ctx.log("jev_advice", tick=current_tick, answer=va, probability=prob)
            return va
        except Exception as exc:
            self.ctx.log("jev_ignored", error=type(exc).__name__)
            return None

    def choose(self, obs: Any) -> str:
        self._poll_astra()
        if not isinstance(obs, dict):
            return "NOOP"
        self._dimensions(obs)
        body = _body_from(obs)
        heading = _heading(obs)
        food = _food(obs)
        if not body or not _in_bounds(body[0], self.rows, self.cols):
            return heading

        # If a previous action was applied exactly as predicted, retain the
        # queued route.  Any skipped/stale tick invalidates it and triggers a
        # fresh simulation from the observed state.
        if self.expected is not None:
            eb, eh, ef, food_wild = self.expected
            matches = body == eb and heading == eh and (food_wild or food == ef)
            if not matches:
                self.plan = []
                self.plan_food = None
            self.expected = None

        if self.plan and self.plan_food != food:
            self.plan = []
            self.plan_food = None

        action: str | None = None
        if self.plan:
            trial = _transition(body, heading, food, self.plan[0], self.rows, self.cols)
            if trial is not None:
                action = self.plan.pop(0)
            else:
                self.plan = []
                self.plan_food = None

        facts = self._candidate_facts(obs, body, heading, food)
        advisory = self._poll_jev(obs, facts)

        if action is None:
            if food is not None:
                route = _food_plan(
                    body, heading, food, self.rows, self.cols,
                    conservative=(self.policy == "conservative"),
                )
                if route:
                    self.plan = route[1:]
                    self.plan_food = food
                    action = route[0]
            if action is None and len(body) * 2 >= self.rows * self.cols:
                action = _cycle_action(body, heading, self.rows, self.cols, self.cycles, food)
            if action is None:
                route = _tail_plan(body, heading, food, self.rows, self.cols)
                if route:
                    self.plan = route[1:]
                    self.plan_food = None
                    action = route[0]
            if action is None:
                fallback = _fallback_action(body, heading, food, self.rows, self.cols)
                if fallback is not None:
                    action = fallback[0]

        # Start Jev after code has built the candidate set.  It can only replace
        # a deterministic choice with a different member of that same set.
        if len(facts) >= 2:
            self._maybe_start_jev(obs, facts, action)
        if advisory is not None:
            # Advisory answers are never allowed to reverse or enter an unsafe
            # cell.  Prefer them only when they are among current candidates.
            if advisory in {x["action"] for x in facts}:
                action = advisory
                self.plan = []
                self.plan_food = None

        if action not in DIRECTIONS:
            action = heading if heading in DIRECTIONS else "RIGHT"
        trial = _transition(body, heading, food, action, self.rows, self.cols)
        if trial is not None:
            nb, nh, ate = trial
            self.expected = (nb, nh, None if ate else food, ate)
        else:
            # This should only occur in a terminal/invalid observation.  NOOP
            # preserves the API contract and lets the game report the terminal.
            self.expected = None
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
                # A real-time post is normally instant.  If the evaluator closes
                # the episode between checks, simply return with the score kept.
                if self.ctx.done or getattr(game, "done", False):
                    return
                raise
            self.ctx.wait_for_tick()
            if self.ctx.done or getattr(game, "done", False):
                return
            obs = game.observe()


def build(ctx):
    return Harness(ctx)
