"""Candidate 0003 — "plan-and-commit": verified multi-tick paths, Jev picks between plans.

Diagnosis of the parent (0002, mean 24, every episode survives to the 300-tick timeout):
  * Score is limited by ticks-per-food, not by dying.  0002 decides ONE step at a time from a
    (steps_to_food, free_cells, tail_reachable) table; near corners / behind its own body the
    greedy step plus the `area >= 2*len` filter oscillates: seed 1 spent 70 ticks (152->222)
    circling a corner food at (0,0); seeds 0/2 have 16-23-tick gaps of the same kind.
  * Jev added nothing per tick: of 310 jev_pick events 304 chose the same distance as the
    fallback and 6 were worse (seed 0 ticks 171-175: it refused food 0 steps away because
    Astra's rule said "length >= 20 -> most free cells").  ~150 Jev calls/episode, 12-36 late.

Structure of this harness:
  1. Planner (harness, exact rules): for every legal first move, a time-aware BFS path to the
     food, then the path is SIMULATED exactly; a plan is "safe" if after eating the tail is
     still reachable (time-aware BFS).  If no safe plan exists the snake "wanders": each tick it
     takes the move that (a) makes a safe plan available next tick, else (b) keeps the tail
     reachable with the most room.
  2. Commitment: a chosen plan is followed to the food without further model calls (the world
     is deterministic until the food is eaten).  Every tick posts an action from the simulator
     -> 0 stale ticks.
  3. Jev chooses BETWEEN PLANS, not moves: when >= 2 distinct safe plans exist (shortest vs.
     roomier-after-eating), Jev is asked on the PREDICTED next state (pipelined, as in 0002) and
     its pick replaces the fallback plan at the next tick if it arrived in time.  -> about one
     Jev call per food instead of one per tick; a late answer just means the fallback plan
     (Astra's DEFAULT) is kept.
  4. Astra: one call per evaluation (cached in ctx.memory) that yields the rules Jev sees, the
     DEFAULT plan preference and a TOLERANCE (how many extra steps a roomier path may cost), plus
     at most one in-episode call when the snake has wandered for a long time.
"""
from __future__ import annotations

import re
import time
from collections import deque

from metaharness.models.jev import Choice

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
ARROW = {"up": "^", "down": "v", "left": "<", "right": ">"}

DEFAULT_RULES = (
    "1. Every listed plan is a complete, collision-free route to the food.\n"
    "2. Prefer the shortest plan unless it leaves the snake in a cramped region after eating.\n"
    "3. A plan that leaves many free cells and a reachable tail is worth a few extra steps, "
    "more so as the snake grows.\n"
    "4. Never prefer a plan whose free cells after eating are fewer than the snake's length."
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

    def bfs_path(self, body, target, order=None):
        """Time-aware BFS from head to target. Body index i is blocked at time t iff i < n - t
        (the tail vacates one cell per tick). Returns list of cells after the head, or None.
        `order` (a permutation of the 4 direction vectors) changes which shortest path is found."""
        if target is None:
            return None
        order = order or list(DIRS.values())
        n = len(body)
        idx = {p: i for i, p in enumerate(body)}
        start = body[0]
        prev = {start: None}
        dist = {start: 0}
        dq = deque([start])
        while dq:
            cur = dq.popleft()
            t = dist[cur]
            if cur == target:
                path = []
                while cur != start:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
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
        return None

    def tail_reachable(self, body):
        """Can the head reach the current tail cell (time-aware) without eating?"""
        if len(body) < 3:
            return True
        return self.bfs_path(body, body[-1]) is not None

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

    # ------------------------------------------------------------------ plans
    ORDERS = (
        [(0, -1), (0, 1), (-1, 0), (1, 0)],   # vertical-first
        [(1, 0), (-1, 0), (0, 1), (0, -1)],   # horizontal-first
    )

    def plans(self, body, heading, food):
        """Candidate plans: for each legal first move, exact-simulated shortest routes to the food
        (two BFS tie-break orders -> routes through different regions of the board)."""
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
                out.append(self._plan_entry(d, cells, fb))
        return out

    def _plan_entry(self, d, cells, fb):
        return {
                "first": d,
                "cells": cells,
                "length": len(cells),
                "tail_ok": self.tail_reachable(fb),
                "area": self.flood(fb),
                "final_len": len(fb),
                "perim": self.perimeter(fb),
        }

    def perimeter(self, body):
        """Free cells adjacent to the body: the smaller, the more compact the body (free space
        stays in one big region instead of being split by the snake)."""
        bset = set(body)
        per = set()
        for x, y in body:
            for dx, dy in DIRS.values():
                p = (x + dx, y + dy)
                if self.inside(p) and p not in bset:
                    per.add(p)
        return len(per)

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
    WANDER_ASTRA_TICKS = 40        # ask Astra once if wandering this long
    JEV_MARGIN = 0.08              # s before the next boundary after which a Jev answer is dropped

    def __init__(self, ctx):
        self.ctx = ctx
        mem = ctx.memory
        self.rules = mem.get("rules", DEFAULT_RULES)
        self.mode = mem.get("mode", "shortest")
        self.tol = int(mem.get("tol", 0))
        self.session = mem.get("session")
        self.astra_pending = None
        self.astra_calls = 0

    # ---------------------------------------------------------------- Astra
    def _astra_prompt(self, game, grid, length, reason):
        return (
            f"{game.describe()}\nActions:\n{game.action_spec.describe()}\n\n"
            f"Board now (length {length}):\n{grid}\n\nContext: {reason}\n"
            "You are the slow planner. The harness computes, for the snake, complete collision-free "
            "routes to the food (one per possible first move) and simulates each: it knows the route "
            "length, how many free cells the head can reach after eating, whether the tail stays "
            "reachable after eating (i.e. the snake cannot trap itself), and how many free cells touch "
            "the body after eating (lower = the body is compact and free space stays in one region). A fast reflex classifier, "
            "which cannot reason, picks one of those routes using your rules and those numbers; the "
            "snake then follows the chosen route without further help. When no route keeps the tail "
            "reachable the harness wanders safely on its own until one appears.\n"
            "Write at most 5 short imperative rules (one per line) for choosing between routes: when "
            "to accept a longer route for more room, as a function of snake length (grid is 144 cells, "
            "episode is 300 ticks, so speed matters). Then exactly two final lines:\n"
            "DEFAULT: shortest   or   DEFAULT: safest   (what to use when the classifier is late)\n"
            "TOLERANCE: <integer 0-6>   (max extra steps a roomier / more central route may cost to be offered)\n"
            "No other prose."
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
        m = re.search(r"DEFAULT:\s*(shortest|safest)", text, re.I)
        if m:
            self.mode = m.group(1).lower()
        m = re.search(r"TOLERANCE:\s*(\d+)", text, re.I)
        if m:
            self.tol = max(0, min(6, int(m.group(1))))
        rules = re.sub(r"(DEFAULT|TOLERANCE):.*", "", text, flags=re.I).strip()
        if rules:
            self.rules = rules[:800]
        mem = self.ctx.memory
        mem["rules"], mem["mode"], mem["tol"], mem["session"] = self.rules, self.mode, self.tol, self.session
        self.ctx.log(event="plan_update", reason=reason, mode=self.mode, tol=self.tol, rules=self.rules)

    def _spawn_astra(self, game, grid, length, reason, cap):
        if self.astra_calls >= cap:
            return
        if self.astra_pending is not None and not self.astra_pending.done():
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) < 1:
            return
        self.astra_calls += 1
        self.astra_pending = self.ctx.spawn(lambda: self._ask_astra(game, grid, length, reason))

    # ---------------------------------------------------------------- plan selection
    def _offer(self, plans, length):
        """Distinct safe plans worth offering: the shortest, and roomier ones within TOLERANCE."""
        safe = [p for p in plans if p["tail_ok"]]
        if not safe:  # no plan keeps the tail reachable: accept only really roomy ones
            safe = [p for p in plans if p["area"] >= 2 * p["final_len"]]
        if not safe:
            return []
        best = min(p["length"] for p in safe)
        pool = sorted((p for p in safe if p["length"] <= best + self.tol),
                      key=lambda p: (p["length"], -p["area"], p["perim"]))
        offered = []
        for p in pool:
            if any(q["area"] >= p["area"] and q["length"] <= p["length"]
                   and q["perim"] <= p["perim"] for q in offered):
                continue
            offered.append(p)
        return offered[:3]

    def _fallback(self, offered):
        if self.mode == "safest":
            return max(offered, key=lambda p: (p["area"], -p["length"], -p["perim"]))
        return offered[0]

    def _wander_move(self, sim, body, heading, food, opts):
        """No safe plan: pick the move that opens one next tick, else keeps the tail reachable."""
        best, best_key = heading, None
        for d, (nb, nh, nf, ate) in opts.items():
            if ate:
                nxt = None
            else:
                nxt = self._offer(sim.plans(nb, nh, nf), len(nb))
            key = (1 if nxt else 0, 1 if sim.tail_reachable(nb) else 0, sim.flood(nb),
                   -(abs(nb[0][0] - nf[0]) + abs(nb[0][1] - nf[1])) if nf else 0)
            if best_key is None or key > best_key:
                best, best_key = d, key
        return best

    # ---------------------------------------------------------------- Jev
    def _ask_jev(self, pred, offered, grid):
        body, heading, food = pred
        crit, opts = {}, {}
        for i, p in enumerate(offered):
            label = f"plan_{i}"
            route = "".join(ARROW[_dir_from(a, b)] for a, b in zip([body[0]] + p["cells"], p["cells"]))
            crit[label] = (f"{p['length']} steps to the food (first move {p['first']}); after eating "
                           f"{p['area']} free cells reachable, tail {'reachable' if p['tail_ok'] else 'CUT OFF'}, "
                           f"{p['perim']} free cells touch the body (lower = more compact)")
            opts[label] = {"steps": p["length"], "first_move": p["first"], "route": route[:40],
                           "free_cells_after_eating": p["area"], "tail_reachable_after": p["tail_ok"],
                           "free_cells_touching_body_after": p["perim"]}
        state = {"planner_rules": self.rules, "head": list(body[0]), "heading": heading,
                 "length": len(body), "food": list(food) if food else None, "plans": opts, "grid": grid}
        q = {"plan": Choice(instructions="Every plan is a verified collision-free route to the food. "
                                         "Following planner_rules, pick the plan to commit to.",
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

        if "rules" not in ctx.memory:
            self._spawn_astra(game, obs["grid"], obs["length"], "episode start; no rules yet", cap=1)

        plan = None            # remaining cells of the committed plan
        pending = None         # (tick, key, future, offered)
        boundary = time.time()
        wander_since = None
        n_jev_used = n_jev_late = n_mismatch = n_plans = n_wander = 0

        while not ctx.done and obs is not None:
            body = [tuple(p) for p in obs["body"]]
            heading = obs["heading"]
            food = tuple(obs["food"]) if obs.get("food") is not None else None
            tick = obs["tick"]

            # ---- validate the committed plan against reality
            if plan:
                d = _dir_from(body[0], plan[0])
                if d is None or sim.step(body, heading, food, d)[4] or food not in plan:
                    plan = None
            # ---- new plan needed: deterministic fallback right now
            offered = None
            if not plan:
                offered = self._offer(sim.plans(body, heading, food), len(body))
                if offered:
                    plan = list(self._fallback(offered)["cells"])
                    n_plans += 1
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
                                      "revise the rules", cap=2)
            ctx.act(action)  # never leave a tick stale

            # ---- late-decision policy: Jev's plan choice for THIS state, if it arrived in time
            if pending is not None:
                p_tick, p_key, fut, p_offered = pending
                pending = None
                if p_tick == tick and p_key == self._key(body, heading, food):
                    deadline = boundary + tick_s - self.JEV_MARGIN
                    res = None
                    try:
                        res = fut.result(timeout=max(0.0, deadline - time.time()))
                    except Exception:
                        n_jev_late += 1
                    if res is not None and game.observe()["tick"] == tick:
                        chosen = p_offered[res[0]]
                        n_jev_used += 1
                        ctx.log(event="jev_plan", tick=tick, pick=res[0], probs=res[1],
                                offered=[(p["length"], p["area"], p["perim"], p["first"]) for p in p_offered],
                                fallback_first=action, chosen_first=chosen["first"])
                        plan = list(chosen["cells"])
                        if chosen["first"] != action:
                            action = chosen["first"]
                            ctx.act(action)
                else:
                    n_mismatch += 1

            # ---- pipeline: if the NEXT state will need a plan choice, ask Jev about it now
            #   (only when a plan was just created from the fallback, or when wandering)
            nb, nh, nf, ate, dead = sim.step(body, heading, food, action)
            fresh_plan = offered is not None and len(offered) >= 2
            if not dead and not ate and (fresh_plan or not plan) \
                    and ctx.budget.remaining().get("system1_calls", 0) > 5:
                offered_next = self._offer(sim.plans(nb, nh, nf), len(nb))
                if len(offered_next) >= 2:
                    pred = (nb, nh, nf)
                    grid = sim.render(nb, nh, nf)
                    fut = ctx.spawn(lambda p=pred, o=offered_next, g=grid: self._ask_jev(p, o, g))
                    pending = (tick + 1, self._key(nb, nh, nf), fut, offered_next)
            # advance the committed plan
            if plan and not dead:
                plan = plan[1:] if _dir_from(body[0], plan[0]) == action else None

            obs, boundary = self._wait_next_tick(game, tick)

        ctx.log(event="episode_summary", jev_used=n_jev_used, jev_late=n_jev_late, pred_mismatch=n_mismatch,
                plans=n_plans, wander_ticks=n_wander, astra_calls=self.astra_calls, mode=self.mode, tol=self.tol)
        ctx.memory.setdefault("stats", []).append(
            {"jev_used": n_jev_used, "jev_late": n_jev_late, "plans": n_plans, "wander": n_wander})


def build(ctx):
    return Harness(ctx)
