"""Push-search front end with a latency-aware Astra fallback for Sokoban.

The search uses only the public JSON observation.  It is deliberately bounded:
when an exact push search does not finish quickly, Astra is asked for a short,
revalidated primitive plan instead of one call per action.
"""
from __future__ import annotations

import heapq
import itertools
import json
import re
import time
from collections import deque

from metaharness.baselines._generic import obs_text
from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


_DIRS = ((-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right"))
_ACTIONS = {"up", "down", "left", "right", "noop"}
_TOKEN = re.compile(r"(?<![A-Za-z0-9_])(up|down|left|right|noop)(?![A-Za-z0-9_])", re.I)


def _board_info(obs):
    board = str(obs.get("board", ""))
    rows = board.splitlines()
    nr, nc = int(obs.get("rows", len(rows))), int(obs.get("cols", 0))
    goals = {tuple(x) for x in obs.get("goals", [])}
    boxes = tuple(sorted(tuple(x) for x in obs.get("boxes", [])))
    player = tuple(obs.get("player", (0, 0)))

    def floor(p):
        r, c = p
        return 0 <= r < nr and 0 <= c < nc and c < len(rows[r]) and rows[r][c] != "#"

    return rows, nr, nc, goals, boxes, player, floor


def _reverse_maps(goals, floor):
    """Exact static reverse-push distances for each goal."""
    maps = []
    for goal in goals:
        dist = {goal: 0}
        q = deque([goal])
        while q:
            r, c = q.popleft()
            for dr, dc, _ in _DIRS:
                prev = (r - dr, c - dc)
                stand = (prev[0] - dr, prev[1] - dc)
                if prev in dist or not floor(prev) or not floor(stand):
                    continue
                dist[prev] = dist[(r, c)] + 1
                q.append(prev)
        maps.append(dist)
    return maps


def _reverse_reachable(goals, floor):
    """Cells from which a box can reach at least one goal."""
    seen = set()
    for dist in _reverse_maps(goals, floor):
        seen.update(dist)
    return seen


def _deadlock(boxes, goals, floor):
    """Cheap freeze test; only reject a cluster containing an off-goal box."""
    boxset = set(boxes)
    for r, c in boxset:
        if (r, c) in goals:
            continue
        cells = {(r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)}
        for rr, cc in cells:
            square = {(rr, cc), (rr + 1, cc), (rr, cc + 1), (rr + 1, cc + 1)}
            if all(x not in floor or x in boxset for x in square) and any(x in boxset and x not in goals for x in square):
                return True
    return False


def _reach_and_paths(player, boxes, floor):
    blocked = set(boxes)
    q = deque([player])
    parent = {player: None}
    how = {}
    while q:
        p = q.popleft()
        for dr, dc, name in _DIRS:
            n = (p[0] + dr, p[1] + dc)
            if n not in parent and n not in blocked and floor(n):
                parent[n] = p
                how[n] = name
                q.append(n)

    def path_to(target):
        if target not in parent:
            return None
        out = []
        p = target
        while parent[p] is not None:
            out.append(how[p])
            p = parent[p]
        out.reverse()
        return out

    return set(parent), path_to


def _matching_lb(boxes, goals, maps=None):
    if not goals:
        return 0
    if maps is not None and len(boxes) == len(maps):
        dp = {0: 0}
        for b in boxes:
            nd = {}
            for mask, value in dp.items():
                for j, dist in enumerate(maps):
                    if mask & (1 << j):
                        continue
                    cost = value + dist.get(b, 10 ** 6)
                    nm = mask | (1 << j)
                    nd[nm] = min(nd.get(nm, 10 ** 9), cost)
            dp = nd
        return dp.get((1 << len(boxes)) - 1, 10 ** 9)
    gs = tuple(goals)
    best = 10 ** 9
    for perm in itertools.permutations(gs, len(boxes)):
        cost = sum(abs(b[0] - g[0]) + abs(b[1] - g[1]) for b, g in zip(boxes, perm))
        if cost < best:
            best = cost
    return best


def _search(obs, seconds=20.0, max_expansions=60000):
    """Return a primitive solution, or None if bounded search gives up."""
    try:
        _rows, _nr, _nc, goals, start_boxes, start_player, floor = _board_info(obs)
        if not goals or len(goals) != len(start_boxes):
            return None
        maps = _reverse_maps(goals, floor)
        useful = set().union(*(m.keys() for m in maps)) if maps else set()
        if any(b not in useful and b not in goals for b in start_boxes):
            return None

        reach, _ = _reach_and_paths(start_player, start_boxes, floor)
        start_key = (start_boxes, min(reach) if reach else start_player)
        # heap priority is push lower bound first; walking is a tie-breaker.
        heap = []
        serial = itertools.count()
        heapq.heappush(heap, (_matching_lb(start_boxes, goals, maps), 0, 0, next(serial), start_key))
        states = {start_key: (start_boxes, start_player)}
        costs = {start_key: (0, 0)}
        parent = {start_key: None}
        segment = {}
        deadline = time.monotonic() + seconds
        expanded = 0

        while heap and expanded < max_expansions and time.monotonic() < deadline:
            _f, gp, gw, _serial, key = heapq.heappop(heap)
            if costs.get(key) != (gp, gw):
                continue
            boxes, player = states[key]
            if set(boxes) == goals:
                out = []
                cur = key
                while parent[cur] is not None:
                    out.extend(reversed(segment[cur]))
                    cur = parent[cur]
                out.reverse()
                return out
            expanded += 1
            reachable, path_to = _reach_and_paths(player, boxes, floor)
            boxset = set(boxes)
            for i, b in enumerate(boxes):
                for dr, dc, action in _DIRS:
                    stand = (b[0] - dr, b[1] - dc)
                    dest = (b[0] + dr, b[1] + dc)
                    if stand not in reachable or not floor(dest) or dest in boxset:
                        continue
                    if dest not in goals and dest not in useful:
                        continue
                    walk = path_to(stand)
                    if walk is None:
                        continue
                    nb = list(boxes)
                    nb[i] = dest
                    nb = tuple(sorted(nb))
                    # Canonicalize the player's connected component.  Any cell
                    # in it can reach the same future standing squares.
                    nrch, _ = _reach_and_paths(b, nb, floor)
                    nk = (nb, min(nrch) if nrch else b)
                    ncost = (gp + 1, gw + len(walk))
                    if ncost >= costs.get(nk, (10 ** 9, 10 ** 9)):
                        continue
                    states[nk] = (nb, b)
                    costs[nk] = ncost
                    parent[nk] = key
                    segment[nk] = tuple(walk + [action])
                    h = _matching_lb(nb, goals, maps)
                    heapq.heappush(heap, (gp + 1 + h, gp + 1, gw + len(walk), next(serial), nk))
    except Exception:
        # Search is an optimization; malformed stand-in observations must not
        # turn into a harness error during pre-flight.
        return None
    return None


def _static_safe(obs, action, useful):
    """Reject a model action that pushes a box into a statically dead cell."""
    if action not in _ACTIONS or action == "noop":
        return action == "noop"
    _rows, _nr, _nc, goals, boxes, player, floor = _board_info(obs)
    d = next((x for x in _DIRS if x[2] == action), None)
    if d is None:
        return False
    dr, dc, _ = d
    near = (player[0] + dr, player[1] + dc)
    if near not in set(boxes):
        return True
    dest = (near[0] + dr, near[1] + dc)
    return dest in goals or dest in useful


def _predict(obs, action):
    """Cheap exact transition for fallback exploration."""
    try:
        rows, nr, nc, goals, boxes, player, floor = _board_info(obs)
        d = next(x for x in _DIRS if x[2] == action)
        dr, dc, _ = d
        target = (player[0] + dr, player[1] + dc)
        if not floor(target):
            return player, boxes, False, False
        b = set(boxes); pushed = target in b
        if pushed:
            dest = (target[0] + dr, target[1] + dc)
            if not floor(dest) or dest in b:
                return player, boxes, False, False
            b.remove(target); b.add(dest)
        return target, tuple(sorted(b)), True, pushed
    except Exception:
        return (0, 0), (), False, False


def _extract_plan(text, limit=10):
    if not text:
        return []
    # Prefer the explicit compact line, since prose often mentions directions.
    marked = []
    for line in str(text).splitlines():
        if re.search(r"actions?|plan|moves?\s*:", line, re.I):
            marked.extend(m.group(1).lower() for m in _TOKEN.finditer(line))
    tokens = marked or [m.group(1).lower() for m in _TOKEN.finditer(str(text))]
    return tokens[:limit]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _fallback(self, obs, useful, visited=None):
        valid = list(obs.get("valid_actions", [])) if isinstance(obs, dict) else []
        visited = visited or set()
        choices = []
        for a in valid:
            if a == "noop" or not _static_safe(obs, a, useful):
                continue
            p, b, changed, pushed = _predict(obs, a)
            if not changed:
                continue
            key = (p, b)
            gain = (100 if pushed else 0) + (20 if key not in visited else -10)
            choices.append((gain, a))
        if choices:
            choices.sort(reverse=True)
            return choices[0][1]
        return valid[0] if valid else "noop"

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system2 is None:
            return
        obs = game.observe()
        try:
            useful = _reverse_reachable(set(tuple(x) for x in obs.get("goals", [])), _board_info(obs)[-1])
        except Exception:
            useful = set()

        # Spend a small, deterministic amount of CPU to eliminate model latency
        # on levels that fit the public push-state search.
        plan = _search(obs)
        if plan:
            for action in plan:
                if ctx.done:
                    return
                if action not in obs.get("valid_actions", ()) or not _static_safe(obs, action, useful):
                    plan = None
                    break
                try:
                    obs = ctx.act(action)
                except Exception:
                    plan = None
                    break
            if ctx.done or (isinstance(obs, dict) and obs.get("done")):
                return

        session = None
        calls = 0
        visited = set()
        while not ctx.done and calls < 40:
            if not isinstance(obs, dict):
                obs = game.observe()
            prompt = (
                "You are a careful Sokoban push planner. Recompute from the CURRENT JSON; "
                "the previous plan may be wrong. Do not explain. Return one line exactly "
                "ACTIONS: a1,a2,... with at most 8 primitive actions from up, down, left, "
                "right (noop only if no movement is legal). Stop before a speculative or "
                "deadlock push; every listed action must be legal in sequence. Prefer deep "
                "or constrained goals and preserve player access.\n"
                f"Static box-useful cells (push destinations outside this set are taboo): {sorted(useful)}\n"
                f"Current observation:\n{obs_text(obs, 7000)}"
            )
            try:
                res = ctx.think(prompt, session_id=session, timeout_s=240.0)
                session = res.session_id
                calls += 1
                actions = _extract_plan(res.text, 8)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except (TimeoutError, RuntimeError):
                calls += 1
                actions = []

            if not actions:
                    actions = [self._fallback(obs, useful, visited)]
            progressed = False
            for action in actions:
                if ctx.done:
                    return
                valid = obs.get("valid_actions", ()) if isinstance(obs, dict) else ()
                if action not in valid or not _static_safe(obs, action, useful):
                    break
                try:
                    before = obs
                    obs = ctx.act(action)
                    if isinstance(obs, dict):
                        visited.add((tuple(obs.get("player", ())), tuple(sorted(tuple(x) for x in obs.get("boxes", ())))))
                    progressed = True
                except Exception:
                    break
                if not isinstance(obs, dict) or obs.get("done"):
                    return
                # Replan immediately after a push; walking suffixes are safe to
                # amortize, while a new box arrangement needs fresh reasoning.
                if obs.get("last_event") == "push":
                    break
            if not progressed and not ctx.done:
                # A garbage response should still consume one safe action and
                # cannot create an unbounded tight loop.
                try:
                    a = self._fallback(obs, useful, visited)
                    if a in obs.get("valid_actions", ()):
                        obs = ctx.act(a)
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
