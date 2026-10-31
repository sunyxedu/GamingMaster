"""Latency-aware Sokoban controller.

The first controller is a bounded push-state A* solver.  Its heuristic uses
reverse-push distances (rather than Manhattan distance), and it prunes simple
freeze deadlocks.  Astra is used only when this deterministic search does not
finish; its output is treated as a short, checked continuation.
"""
from __future__ import annotations

import heapq
import itertools
import json
import re
import time
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable

DIRS = ((-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right"))
ACTION_NAMES = {x[2] for x in DIRS}
TOKEN = re.compile(r"(?<![A-Za-z0-9_])(up|down|left|right|u|d|l|r)(?![A-Za-z0-9_])", re.I)
ALIASES = {"u": "up", "d": "down", "l": "left", "r": "right",
           "up": "up", "down": "down", "left": "left", "right": "right"}


def _state(obs):
    rows = str(obs.get("board", "")).splitlines() if isinstance(obs, dict) else []
    h = int(obs.get("rows", len(rows))) if isinstance(obs, dict) else len(rows)
    w = int(obs.get("cols", max((len(x) for x in rows), default=0))) if isinstance(obs, dict) else 0
    rows = (rows + [""] * max(0, h - len(rows)))[:h]
    floor = {(r, c) for r in range(h) for c in range(w)
             if c < len(rows[r]) and rows[r][c] != "#"}
    def pairs(key):
        out = set()
        for p in (obs.get(key, ()) if isinstance(obs, dict) else ()):
            try: out.add((int(p[0]), int(p[1])))
            except Exception: pass
        return out
    try: player = (int(obs.get("player", [0, 0])[0]), int(obs.get("player", [0, 0])[1]))
    except Exception: player = (0, 0)
    return {"h": h, "w": w, "floor": floor, "walls": {(r, c) for r in range(h) for c in range(w)} - floor,
            "player": player, "boxes": frozenset(pairs("boxes")), "goals": frozenset(pairs("goals"))}


def _reach(player, boxes, floor):
    q, parent, how = deque([player]), {player: None}, {}
    while q:
        p = q.popleft()
        for dr, dc, a in DIRS:
            n = (p[0] + dr, p[1] + dc)
            if n in floor and n not in boxes and n not in parent:
                parent[n] = p; how[n] = a; q.append(n)
    def path(target):
        if target not in parent: return None
        out, p = [], target
        while parent[p] is not None:
            out.append(how[p]); p = parent[p]
        return out[::-1]
    return set(parent), path


def _reverse_maps(s):
    """For every goal, compute legal static pull distance for a box."""
    maps = []
    for goal in s["goals"]:
        d, q = {goal: 0}, deque([goal])
        while q:
            cur = q.popleft()
            for dr, dc, _ in DIRS:
                prev = (cur[0] - dr, cur[1] - dc)
                stand = (prev[0] - dr, prev[1] - dc)
                if prev in s["floor"] and stand in s["floor"] and prev not in d:
                    d[prev] = d[cur] + 1; q.append(prev)
        maps.append(d)
    return maps


def _matching(boxes, maps):
    n = len(boxes)
    if n != len(maps): return 10**6
    costs = [[m.get(b, 10**6) for m in maps] for b in boxes]
    dp = {0: 0}
    for row in costs:
        nd = {}
        for mask, val in dp.items():
            for j, c in enumerate(row):
                if mask & (1 << j): continue
                nm = mask | (1 << j)
                nd[nm] = min(nd.get(nm, 10**6), val + c)
        dp = nd
    return dp.get((1 << n) - 1, 10**6)


def _dead(s, boxes, good):
    """Conservative static corner and dynamic 2x2 freeze tests."""
    for b in boxes:
        if b in s["goals"] or b in good: continue
        r, c = b
        if ((r - 1, c) in s["walls"] or (r + 1, c) in s["walls"]) and ((r, c - 1) in s["walls"] or (r, c + 1) in s["walls"]):
            return True
    blocked = s["walls"] | set(boxes)
    for r, c in boxes:
        for rr, cc in ((r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)):
            square = {(rr, cc), (rr + 1, cc), (rr, cc + 1), (rr + 1, cc + 1)}
            if square <= blocked and any(x in boxes and x not in s["goals"] for x in square):
                return True
    return False


def _search(obs, seconds=9.0, expansions=30000):
    try:
        s = _state(obs); boxes0 = tuple(sorted(s["boxes"]))
        if not boxes0 or len(boxes0) != len(s["goals"]): return None
        maps = _reverse_maps(s)
        good = set().union(*(m.keys() for m in maps)) if maps else set()
        if any(b not in good and b not in s["goals"] for b in boxes0): return None
        reach, _ = _reach(s["player"], set(boxes0), s["floor"])
        anchor = min(reach) if reach else s["player"]
        start = (boxes0, anchor)
        states = {start: (boxes0, s["player"])}; cost = {start: (0, 0)}
        parent, segment = {start: None}, {}
        serial, heap = itertools.count(), []
        heapq.heappush(heap, (_matching(boxes0, maps), 0, 0, next(serial), start))
        deadline, done = time.monotonic() + seconds, 0
        while heap and done < expansions and time.monotonic() < deadline:
            _, gp, gw, _, key = heapq.heappop(heap)
            if cost.get(key) != (gp, gw): continue
            boxes, player = states[key]
            if set(boxes) == set(s["goals"]):
                out, cur = [], key
                while parent[cur] is not None:
                    out.extend(reversed(segment[cur])); cur = parent[cur]
                return out[::-1]
            done += 1; reach, path = _reach(player, set(boxes), s["floor"]); bset = set(boxes)
            for i, b in enumerate(boxes):
                for dr, dc, a in DIRS:
                    stand, dest = (b[0] - dr, b[1] - dc), (b[0] + dr, b[1] + dc)
                    if stand not in reach or dest not in s["floor"] or dest in bset: continue
                    if dest not in s["goals"] and dest not in good: continue
                    nb = list(boxes); nb[i] = dest; nb = tuple(sorted(nb))
                    if _dead(s, nb, good): continue
                    walk = path(stand)
                    if walk is None: continue
                    nr, _ = _reach(b, set(nb), s["floor"]); na = min(nr) if nr else b
                    nk, nc = (nb, na), (gp + 1, gw + len(walk))
                    if nc >= cost.get(nk, (10**9, 10**9)): continue
                    cost[nk] = nc; states[nk] = (nb, b); parent[nk] = key; segment[nk] = tuple(walk + [a])
                    heapq.heappush(heap, (gp + 1 + _matching(nb, maps), gp + 1, gw + len(walk), next(serial), nk))
    except Exception:
        return None
    return None


def _predict(s, action):
    if action not in ACTION_NAMES: return s, False, False
    dr, dc = next((x[0], x[1]) for x in DIRS if x[2] == action)
    target = (s["player"][0] + dr, s["player"][1] + dc)
    if target not in s["floor"]: return s, False, False
    boxes = set(s["boxes"]); pushed = target in boxes
    if pushed:
        dest = (target[0] + dr, target[1] + dc)
        if dest not in s["floor"] or dest in boxes: return s, False, False
        boxes.remove(target); boxes.add(dest)
    n = dict(s); n["player"] = target; n["boxes"] = frozenset(boxes)
    return n, True, pushed


def _same(a, b):
    try: return tuple(a.get("player")) == tuple(b["player"]) and {tuple(x) for x in a.get("boxes", ())} == set(b["boxes"])
    except Exception: return False


def _extract(text, limit=30):
    if not isinstance(text, str): return []
    lines = re.findall(r"(?im)\bactions?\s*[:=]\s*([^\n\r]+)", text)
    source = lines[-1] if lines else (text if len(text.strip()) <= 160 else "")
    return [ALIASES[m.group(1).lower()] for m in TOKEN.finditer(source)][:limit]


def _fallback(obs, s, good, visited):
    options = []
    for a in ACTION_NAMES:
        n, changed, pushed = _predict(s, a)
        if not changed or _dead(s, n["boxes"], good): continue
        score = (100 if pushed and n["boxes"] & s["goals"] else 0) + (4 if (n["player"], tuple(sorted(n["boxes"]))) not in visited else 0)
        options.append((score, a))
    if options: return max(options)[1]
    valid = obs.get("valid_actions", []) if isinstance(obs, dict) else []
    return next((a for a in valid if a in ACTION_NAMES), "noop")


class Harness:
    def __init__(self, ctx): self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if getattr(ctx, "system2", None) is None: return
        try: obs = game.observe()
        except Exception: return
        s = _state(obs); maps = _reverse_maps(s); good = set().union(*(m.keys() for m in maps)) if maps else set()
        plan = _search(obs); visited = {(s["player"], tuple(sorted(s["boxes"])))}
        session = None; calls = 0; failure = ""
        while not ctx.done and isinstance(obs, dict) and not obs.get("done"):
            if not plan:
                prompt = ("You are a Sokoban solver. Current JSON is authoritative. Return only one line "
                          "ACTIONS: followed by up to 30 primitive directions. Plan legal walking and a "
                          "safe next push; never push a box into a corner or off reverse-push cells. "
                          "Recompute after each push.\n" + json.dumps(obs, separators=(",", ":")) +
                          "\nREVERSE-PUSH CELLS:" + json.dumps(sorted(good)) + ("\nREJECTED:" + failure if failure else ""))
                if calls >= 40: break
                try:
                    rem = float(ctx.budget.remaining().get("wall_s", 1800.0)); timeout = max(5.0, min(240.0, rem - 2.0))
                    res = ctx.think(prompt, session_id=session, timeout_s=timeout); session = res.session_id; calls += 1
                    plan = _extract(res.text)
                except (BudgetExceeded, EpisodeOver, ModelUnavailable): break
                except (TimeoutError, RuntimeError): calls += 1; plan = []
                if not plan: plan = [_fallback(obs, s, good, visited)]
            action = plan.pop(0)
            if action not in (obs.get("valid_actions", []) if isinstance(obs, dict) else []):
                failure = "invalid proposed action"; plan = []; continue
            expected, changed, _ = _predict(s, action)
            if not changed:
                failure = "blocked transition"; plan = []; continue
            try: nxt = ctx.act(action)
            except Exception:
                plan = []; continue
            if not isinstance(nxt, dict) or not _same(nxt, expected):
                failure = "observation disagreed with prediction"; plan = []; obs = nxt if isinstance(nxt, dict) else obs; s = _state(obs); continue
            obs, s = nxt, _state(nxt); visited.add((s["player"], tuple(sorted(s["boxes"]))))
            if nxt.get("last_event") == "push": plan = []


def build(ctx): return Harness(ctx)
