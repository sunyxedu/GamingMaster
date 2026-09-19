"""Candidate 0004 — plan-and-commit with a one-step-lookahead VALUE: "expected detour to the next food".

Diagnosis of the parent (0003, mean 32, all episodes time out alive, 0 % stale, 1 S2 call):
  * Replaying 0003's planner offline on seeds 0-7 (no models) gives 31.25; ticks-per-food is
    Manhattan(head, food at spawn) + a *slack* of only 0.4-1.3 ticks/food on average.  The harness is
    at the shortest-path ceiling; the remaining loss is a handful of huge detours per episode (slack
    14, 16, 20, 22 ticks) that occur when the food spawns inside a pocket / behind a wall that the
    snake's own body formed a few foods earlier (seed 0, tick 189: food 2 cells away, only route is a
    dead-end pocket -> 25 ticks; tick 214: body seals the whole top half -> 23 ticks for a Manhattan-10
    food).  Neither the plan choice (length / area / perimeter) nor Jev's rule-following can see this,
    because "free cells after eating" is the same for a wall that splits the board and for an open board.
  * All 26 Jev calls in 0003 were on plans whose after-eat numbers hardly differed; Astra's TOLERANCE
    was irrelevant because plans for the 3 first moves only ever differ by 0 or 2 steps.

Hypothesis: the right one-step value of an after-eat state is the *expected extra distance to the next
food*: mean over all free cells c of (time-aware BFS distance(head, c) - Manhattan(head, c)), with
unreachable cells charged a penalty.  It is exact, cheap (one BFS per candidate plan), and it directly
measures "did my body split the board / build a pocket".  Plans are ranked by  steps + LAMBDA * detour.
Offline (mock, no models): seeds 0-7  31.25 -> 32.375, seeds 8-15  32.0 -> 32.875 (16-seed mean +0.9).

Astra / Jev bridge (the design under test):
  * Astra (1 call per evaluation, cached in ctx.memory) designs the OBJECTIVE, not the move: it returns
    LAMBDA (weight of the expected detour), PENALTY (cost of an unreachable cell), MARGIN (how close two
    plans' costs must be before the reflex model is consulted) and the rules text Jev reads.  All are
    clamped to ranges verified offline, so Astra can tune but not break the planner.
  * Jev is asked only on genuine close calls (cost gap <= MARGIN and different routes), on the PREDICTED
    next state (pipelined), and sees per-plan numbers incl. the expected detour.  The value-optimal plan
    is always posted first; Jev's pick replaces it at the next tick if it arrives in time.
  * Commitment / late policy / wander mode as in 0003: every tick posts a simulator action before any
    model work -> 0 stale ticks.
"""
from __future__ import annotations

import re
import time
from collections import deque

from metaharness.models.jev import Choice

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
ARROW = {"up": "^", "down": "v", "left": "<", "right": ">"}

# Objective parameters: defaults verified offline; Astra may move them inside these ranges.
DEFAULTS = {"lambda": 1.0, "penalty": 20.0, "margin": 0.5, "tol": 4}
RANGES = {"lambda": (0.5, 3.0), "penalty": (5.0, 40.0), "margin": (0.0, 3.0), "tol": (0, 6)}

DEFAULT_RULES = (
    "1. Every listed plan is a complete, collision-free route to the food.\n"
    "2. Prefer the plan with the smallest expected_detour_next_food: it means the body does not "
    "split the board or build pockets, so the next food (random cell) is reached without detours.\n"
    "3. Between plans with similar expected detour, prefer fewer steps, then more free cells.\n"
    "4. Never prefer a plan whose tail is CUT OFF or whose free cells are fewer than the snake's length."
)


def _dir_from(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    for d, (ex, ey) in DIRS.items():
        if (ex, ey) == (dx, dy):
            return d
    return None


# ----------------------------------------------------------------------------- simulator
class Sim:
    def __init__(self, w, h):
        self.w, self.h = w, h

    def inside(self, p):
        return 0 <= p[0] < self.w and 0 <= p[1] < self.h

    def step(self, body, heading, food, action):
        """Exact replica of SnakeGame.tick. Returns (body, heading, food, ate, dead)."""
        if action in DIRS and action != OPP[heading]:
            heading = action
        dx, dy = DIRS[heading]
        nxt = (body[0][0] + dx, body[0][1] + dy)
        if not self.inside(nxt):
            return body, heading, food, False, True
        grow = food is not None and nxt == food
        tail = body if grow else body[:-1]
        if nxt in tail:
            return body, heading, food, False, True
        return [nxt] + list(tail), heading, (None if grow else food), grow, False

    def legal_moves(self, body, heading, food):
        out = {}
        for d in DIRS:
            if d == OPP[heading]:
                continue
            nb, nh, nf, ate, dead = self.step(body, heading, food, d)
            if not dead:
                out[d] = (nb, nh, nf, ate)
        return out

    def flood(self, body):
        blocked = set(body[:-1])
        start = body[0]
        seen = {start}
        dq = deque([start])
        while dq:
            x, y = dq.popleft()
            for dx, dy in DIRS.values():
                p = (x + dx, y + dy)
                if self.inside(p) and p not in blocked and p not in seen:
                    seen.add(p)
                    dq.append(p)
        return len(seen) - 1

    def dist_map(self, body, order=None, target=None):
        """Time-aware BFS from the head: body index i is blocked at time t iff i < n - t (the tail
        vacates one cell per tick). Returns (dist, prev); stops early when `target` is reached."""
        order = order or list(DIRS.values())
        n = len(body)
        idx = {p: i for i, p in enumerate(body)}
        start = body[0]
        prev = {start: None}
        dist = {start: 0}
        dq = deque([start])
        while dq:
            cur = dq.popleft()
            if cur == target:
                break
            t = dist[cur]
            for dx, dy in order:
                p = (cur[0] + dx, cur[1] + dy)
                if not self.inside(p) or p in dist:
                    continue
                i = idx.get(p)
                if i is not None and i < n - (t + 1):
                    continue
                dist[p] = t + 1
                prev[p] = cur
                dq.append(p)
        return dist, prev

    def bfs_path(self, body, target, order=None):
        """Shortest time-aware route head -> target as a list of cells after the head, or None."""
        if target is None:
            return None
        dist, prev = self.dist_map(body, order, target)
        if target not in dist:
            return None
        path, cur = [], target
        while cur != body[0]:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def tail_reachable(self, body):
        if len(body) < 3:
            return True
        return self.bfs_path(body, body[-1]) is not None

    def expected_detour(self, body, penalty):
        """VALUE of a state for the *next* (uniformly random) food: mean over free cells of
        (time-aware BFS distance - Manhattan distance); unreachable cells cost `penalty`."""
        dist, _ = self.dist_map(body)
        bset = set(body)
        hx, hy = body[0]
        tot = cnt = 0
        for x in range(self.w):
            for y in range(self.h):
                p = (x, y)
                if p in bset:
                    continue
                cnt += 1
                d = dist.get(p)
                tot += penalty if d is None else d - abs(x - hx) - abs(y - hy)
        return tot / max(1, cnt)

    def simulate(self, body, heading, food, cells):
        """Follow a list of cells exactly. Returns (body, heading, food, ate_any) or None if dead."""
        ate_any = False
        for c in cells:
            d = _dir_from(body[0], c)
            if d is None:
                return None
            body, heading, food, ate, dead = self.step(body, heading, food, d)
            if dead:
                return None
            ate_any = ate_any or ate
        return body, heading, food, ate_any

    ORDERS = (
        [(0, -1), (0, 1), (-1, 0), (1, 0)],   # vertical-first
        [(1, 0), (-1, 0), (0, 1), (0, -1)],   # horizontal-first
    )

    def plans(self, body, heading, food, penalty):
        """Candidate plans: for each legal first move and BFS tie-break order, the shortest route to
        the food, simulated exactly, annotated with the after-eat state's numbers and VALUE."""
        out, seen = [], set()
        for d, (nb, nh, nf, ate) in self.legal_moves(body, heading, food).items():
            variants = []
            if ate:
                variants.append([nb[0]])
            else:
                for order in self.ORDERS:
                    p = self.bfs_path(nb, nf, order)
                    if p is not None:
                        variants.append([nb[0]] + p)
            for cells in variants:
                key = tuple(cells)
                if key in seen:
                    continue
                seen.add(key)
                sim = self.simulate(body, heading, food, cells)
                if sim is None or not sim[3]:
                    continue
                fb = sim[0]
                out.append({
                    "first": d, "cells": cells, "length": len(cells),
                    "tail_ok": self.tail_reachable(fb), "area": self.flood(fb), "final_len": len(fb),
                    "detour": self.expected_detour(fb, penalty),
                })
        return out

    def render(self, body, heading, food):
        bset = set(body)
        rows = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                p = (x, y)
                if p == body[0]:
                    row.append(ARROW[heading])
                elif p in bset:
                    row.append("o")
                elif food is not None and p == food:
                    row.append("*")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)


# ----------------------------------------------------------------------------- harness
class Harness:
    WANDER_ASTRA_TICKS = 40        # ask Astra once more if wandering this long
    JEV_MARGIN_S = 0.08            # s before the next boundary after which a Jev answer is dropped

    def __init__(self, ctx):
        self.ctx = ctx
        mem = ctx.memory
        self.rules = mem.get("rules", DEFAULT_RULES)
        self.params = dict(DEFAULTS)
        self.params.update(mem.get("params", {}))
        self.session = mem.get("session")
        self.astra_pending = None
        self.astra_calls = 0

    # ---------------------------------------------------------------- Astra: designs the objective
    def _astra_prompt(self, game, grid, length, reason):
        p = self.params
        return (
            f"{game.describe()}\nActions:\n{game.action_spec.describe()}\n\n"
            f"Board now (length {length}):\n{grid}\n\nContext: {reason}\n"
            "You are the slow planner; you design the OBJECTIVE, the harness executes it exactly every tick.\n"
            "Harness facts: for every possible first move it computes the shortest collision-free route to "
            "the food and simulates it. For the state right after eating it computes: steps (route length), "
            "free cells reachable, whether the tail stays reachable, and expected_detour = the average over "
            "all free cells of (real path length - Manhattan distance) to that cell, unreachable cells "
            "counted as PENALTY steps. Because the next food appears on a uniformly random free cell, "
            "expected_detour is the expected number of wasted ticks on the next food (it detects when the "
            "body splits the board or forms pockets). Plans are ranked by  cost = steps + LAMBDA * expected_detour  "
            "and the best is followed without further help. When the two best plans' costs differ by at most "
            "MARGIN, a fast reflex classifier (cannot reason) picks between them using your rules and the numbers.\n"
            f"Current values: LAMBDA {p['lambda']}, PENALTY {p['penalty']}, MARGIN {p['margin']}, TOL {p['tol']}.\n"
            "Offline calibration: with LAMBDA in [0.5, 3] and PENALTY 20 the planner eats ~32-33 food in 300 "
            "ticks; routes longer than the shortest almost never pay off, ties (equal steps) are where the value matters.\n"
            "Write at most 4 short imperative rules (one per line) for the classifier to choose between "
            "close-call plans. Then exactly these final lines (numbers only):\n"
            "LAMBDA: <0.5-3>\nPENALTY: <5-40>  (cost of an unreachable cell)\nMARGIN: <0-3>  (0 = never ask the classifier)\n"
            "TOL: <0-6>  (max extra steps a plan may cost to be considered at all)\nNo other prose."
        )

    def _ask_astra(self, game, grid, length, reason):
        try:
            res = self.ctx.think(self._astra_prompt(game, grid, length, reason), session_id=self.session)
        except Exception as e:
            self.ctx.log(event="astra_failed", error=str(e)[:300])
            return
        self.session = res.session_id
        text = (res.text or "").strip()
        if not text:
            return
        for key in ("lambda", "penalty", "margin", "tol"):
            m = re.search(key.upper() + r":\s*([0-9]+(?:\.[0-9]+)?)", text)
            if m:
                lo, hi = RANGES[key]
                v = max(lo, min(hi, float(m.group(1))))
                self.params[key] = int(v) if key == "tol" else v
        rules = re.sub(r"(LAMBDA|PENALTY|MARGIN|TOL):.*", "", text).strip()
        if rules:
            self.rules = rules[:800]
        mem = self.ctx.memory
        mem["rules"], mem["params"], mem["session"] = self.rules, dict(self.params), self.session
        self.ctx.log(event="plan_update", reason=reason, params=dict(self.params), rules=self.rules)

    def _spawn_astra(self, game, grid, length, reason, cap):
        if self.astra_calls >= cap:
            return
        if self.astra_pending is not None and not self.astra_pending.done():
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) < 1:
            return
        self.astra_calls += 1
        self.astra_pending = self.ctx.spawn(lambda: self._ask_astra(game, grid, length, reason))

    # ---------------------------------------------------------------- plan ranking (the value)
    def _cost(self, p):
        return p["length"] + self.params["lambda"] * p["detour"]

    def _offer(self, plans):
        """Safe plans ranked by cost = steps + LAMBDA * expected detour; at most 3, best first."""
        safe = [p for p in plans if p["tail_ok"]]
        if not safe:  # no plan keeps the tail reachable: accept only really roomy ones
            safe = [p for p in plans if p["area"] >= 2 * p["final_len"]]
        if not safe:
            return []
        best = min(p["length"] for p in safe)
        pool = sorted((p for p in safe if p["length"] <= best + self.params["tol"]),
                      key=lambda p: (self._cost(p), p["length"], -p["area"]))
        return pool[:3]

    def _close_call(self, offered):
        """Jev is consulted only when the runner-up is within MARGIN of the best and differs in route."""
        if len(offered) < 2 or self.params["margin"] <= 0:
            return False
        return self._cost(offered[1]) - self._cost(offered[0]) <= self.params["margin"]

    def _wander_move(self, sim, body, heading, food, opts):
        """No safe plan: the move that opens one next tick, else keeps the tail reachable, else the
        lowest expected detour (most open board)."""
        best, best_key = heading, None
        pen = self.params["penalty"]
        for d, (nb, nh, nf, ate) in opts.items():
            nxt = None if ate else self._offer(sim.plans(nb, nh, nf, pen))
            key = (1 if nxt else 0, 1 if sim.tail_reachable(nb) else 0, -sim.expected_detour(nb, pen),
                   sim.flood(nb))
            if best_key is None or key > best_key:
                best, best_key = d, key
        return best

    # ---------------------------------------------------------------- Jev: close-call arbiter
    def _ask_jev(self, pred, offered, grid):
        body, heading, food = pred
        crit, opts = {}, {}
        for i, p in enumerate(offered):
            label = f"plan_{i}"
            route = "".join(ARROW[_dir_from(a, b)] for a, b in zip([body[0]] + p["cells"], p["cells"]))
            crit[label] = (f"{p['length']} steps to the food (first move {p['first']}); after eating: "
                           f"expected detour to the next food {p['detour']:.1f} ticks, {p['area']} free cells "
                           f"reachable, tail {'reachable' if p['tail_ok'] else 'CUT OFF'}")
            opts[label] = {"steps": p["length"], "first_move": p["first"], "route": route[:40],
                           "expected_detour_next_food": round(p["detour"], 2),
                           "free_cells_after_eating": p["area"], "tail_reachable_after": p["tail_ok"],
                           "cost": round(self._cost(p), 2)}
        state = {"planner_rules": self.rules, "head": list(body[0]), "heading": heading,
                 "length": len(body), "food": list(food) if food else None, "plans": opts, "grid": grid}
        q = {"plan": Choice(instructions="Every plan is a verified collision-free route to the food; the "
                                         "harness rates them almost equally. Following planner_rules, pick "
                                         "the plan to commit to.",
                            criteria=crit)}
        try:
            ans = self.ctx.decide(state, q)
        except Exception as e:
            self.ctx.log(event="jev_failed", error=str(e)[:200])
            return None
        a = ans["plan"]
        try:
            i = int(str(a.value).split("_")[1])
        except Exception:
            return None
        if not 0 <= i < len(offered):
            return None
        return i, dict(a.probabilities or {})

    # ---------------------------------------------------------------- loop
    @staticmethod
    def _key(body, heading, food):
        return (tuple(body), heading, food)

    def _wait_next_tick(self, game, tick):
        while not self.ctx.done:
            obs = game.observe()
            if obs["tick"] != tick:
                return obs, time.time()
            time.sleep(0.004)
        return None, time.time()

    def run_episode(self, game):
        ctx = self.ctx
        obs = game.observe()
        w, h = obs["grid_size"]
        sim = Sim(w, h)
        tick_s = float(getattr(game, "tick_s", None) or ctx.tick_s or 0.5)

        if "params" not in ctx.memory:
            self._spawn_astra(game, obs["grid"], obs["length"], "episode start; design the objective", cap=1)

        plan = None            # remaining cells of the committed plan
        pending = None         # (tick, key, future, offered)
        boundary = time.time()
        wander_since = None
        n_jev_used = n_jev_late = n_mismatch = n_plans = n_wander = n_close = 0
        detours = []

        while not ctx.done and obs is not None:
            body = [tuple(p) for p in obs["body"]]
            heading = obs["heading"]
            food = tuple(obs["food"]) if obs.get("food") is not None else None
            tick = obs["tick"]
            pen = self.params["penalty"]

            # ---- validate the committed plan against reality
            if plan:
                d = _dir_from(body[0], plan[0])
                if d is None or sim.step(body, heading, food, d)[4] or food not in plan:
                    plan = None
            # ---- new plan needed: value-optimal fallback right now
            offered = None
            if not plan:
                offered = self._offer(sim.plans(body, heading, food, pen))
                if offered:
                    plan = list(offered[0]["cells"])
                    n_plans += 1
                    detours.append(round(offered[0]["detour"], 2))
            if plan:
                action = _dir_from(body[0], plan[0])
                wander_since = None
            else:
                opts = sim.legal_moves(body, heading, food)
                action = self._wander_move(sim, body, heading, food, opts) if opts else heading
                n_wander += 1
                wander_since = tick if wander_since is None else wander_since
                if tick - wander_since >= self.WANDER_ASTRA_TICKS:
                    wander_since = tick
                    self._spawn_astra(game, obs["grid"], len(body),
                                      "the snake has had no safe route to the food for 40 ticks; "
                                      "revise the objective", cap=2)
            ctx.act(action)  # never leave a tick stale

            # ---- late-decision policy: Jev's plan choice for THIS state, if it arrived in time
            if pending is not None:
                p_tick, p_key, fut, p_offered = pending
                pending = None
                if p_tick == tick and p_key == self._key(body, heading, food):
                    deadline = boundary + tick_s - self.JEV_MARGIN_S
                    res = None
                    try:
                        res = fut.result(timeout=max(0.0, deadline - time.time()))
                    except Exception:
                        n_jev_late += 1
                    if res is not None and game.observe()["tick"] == tick:
                        chosen = p_offered[res[0]]
                        n_jev_used += 1
                        ctx.log(event="jev_plan", tick=tick, pick=res[0], probs=res[1],
                                offered=[(p["length"], round(p["detour"], 2), p["area"], p["first"])
                                         for p in p_offered],
                                fallback_first=action, chosen_first=chosen["first"])
                        plan = list(chosen["cells"])
                        if chosen["first"] != action:
                            action = chosen["first"]
                            ctx.act(action)
                else:
                    n_mismatch += 1

            # ---- pipeline: if the NEXT state will be a close call, ask Jev about it now
            nb, nh, nf, ate, dead = sim.step(body, heading, food, action)
            fresh_plan = offered is not None and len(offered) >= 2
            if not dead and not ate and (fresh_plan or not plan) \
                    and ctx.budget.remaining().get("system1_calls", 0) > 5:
                offered_next = self._offer(sim.plans(nb, nh, nf, pen))
                if self._close_call(offered_next):
                    n_close += 1
                    pred = (nb, nh, nf)
                    grid = sim.render(nb, nh, nf)
                    fut = ctx.spawn(lambda p=pred, o=offered_next, g=grid: self._ask_jev(p, o, g))
                    pending = (tick + 1, self._key(nb, nh, nf), fut, offered_next)
            # advance the committed plan
            if plan and not dead:
                plan = plan[1:] if _dir_from(body[0], plan[0]) == action else None

            obs, boundary = self._wait_next_tick(game, tick)

        ctx.log(event="episode_summary", jev_used=n_jev_used, jev_late=n_jev_late, pred_mismatch=n_mismatch,
                close_calls=n_close, plans=n_plans, wander_ticks=n_wander, astra_calls=self.astra_calls,
                params=dict(self.params), chosen_detours=detours)
        ctx.memory.setdefault("stats", []).append(
            {"jev_used": n_jev_used, "jev_late": n_jev_late, "plans": n_plans, "wander": n_wander})


def build(ctx):
    return Harness(ctx)
