"""Candidate 0002 — "predict-ahead, consequence-annotated options" harness for real-time Snake.

Diagnosis of the parents (see proposal.md):
  * 0000 (Jev per tick, raw obs): Jev latency 0.57-0.63 s > 0.5 s tick, so every answer is
    applied ONE TICK LATE to a state it was not computed for (seed 1: 'down' decided at [2,1]
    heading left became a reverse at [2,0] heading up -> ignored -> wall).  Jev's answers on the
    raw observation are also near coin-flips (0.44-0.6) and drift away from the food.
  * 0001 (Astra per tick): 2-3 s per call, 81% stale ticks.

Structure of this harness (the Astra -> Jev bridge):
  1. The harness owns an exact one-step simulator of the (fully known) snake rules and computes
     for every legal, non-fatal heading: steps-to-food (time-aware BFS), free cells reachable
     after the move (flood fill) and whether the tail stays reachable.  Fatal / trapped moves
     never reach a model.  This gives a deterministic fallback so NO tick is ever stale.
  2. Jev is asked about the PREDICTED next state (the game is deterministic given the action we
     posted), so its ~0.6 s answer lands inside the tick window where it is needed instead of one
     tick late.  It is asked only when >= 2 non-dominated options remain (distance vs. space
     trade-off) and its criteria are the concrete consequences of each move, plus Astra's rules.
  3. Astra is consulted once per evaluation (rules cached in ctx.memory across seeds) and again in
     the background only when the harness detects a trap (no option keeps the tail reachable).
     Its output: <=5 imperative rules for Jev + a DEFAULT preference (shortest|safest) that the
     harness heuristic uses whenever Jev is late.
  4. Tick detection polls obs['tick'] (the runner's ticker is not aligned to absolute multiples
     of tick_s, so ctx.wait_for_tick() would be off-phase).  Late-decision policy: post the
     heuristic action right at the boundary, then overwrite with Jev's answer if it arrives
     before the next boundary and the predicted state matched reality.
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
    "1. Never enter a cell that kills (wall or body); never reverse.\n"
    "2. Take the move with the fewest steps to the food when it keeps the tail reachable.\n"
    "3. If a move leaves fewer free cells than the snake's length, avoid it even if the food is close.\n"
    "4. When two moves tie, prefer the one that leaves more free space."
)


# ----------------------------------------------------------------------------- simulator
class Sim:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h

    def inside(self, p):
        return 0 <= p[0] < self.w and 0 <= p[1] < self.h

    def step(self, body, heading, food, action):
        """Exact copy of SnakeGame.tick. Returns (body, heading, food, ate, dead)."""
        if action in DIRS and action != OPP[heading]:
            heading = action
        dx, dy = DIRS[heading]
        hx, hy = body[0]
        nxt = (hx + dx, hy + dy)
        if not self.inside(nxt):
            return body, heading, food, False, True
        grow = food is not None and nxt == food
        tail = body if grow else body[:-1]
        if nxt in tail:
            return body, heading, food, False, True
        new_body = [nxt] + list(tail)
        return new_body, heading, (None if grow else food), grow, False

    def legal_moves(self, body, heading, food):
        """Headings that do not die on the next tick (reverse excluded — it is a no-op)."""
        out = {}
        for d in DIRS:
            if d == OPP[heading]:
                continue
            nb, nh, nf, ate, dead = self.step(body, heading, food, d)
            if not dead:
                out[d] = (nb, nh, nf, ate)
        return out

    def flood(self, body):
        """Free cells reachable from the head, treating body[:-1] as walls."""
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
        seen.discard(start)
        return seen

    def bfs_food(self, body, food):
        """Time-aware BFS from head to food: body index i is blocked at time t iff i < len-t."""
        if food is None:
            return None
        n = len(body)
        idx = {p: i for i, p in enumerate(body)}
        start = body[0]
        dist = {start: 0}
        dq = deque([start])
        while dq:
            cur = dq.popleft()
            t = dist[cur]
            if cur == food:
                return t
            for dx, dy in DIRS.values():
                p = (cur[0] + dx, cur[1] + dy)
                if not self.inside(p) or p in dist:
                    continue
                i = idx.get(p)
                if i is not None and i < n - (t + 1):
                    continue
                dist[p] = t + 1
                dq.append(p)
        return None

    def options(self, body, heading, food):
        """Consequence table for every non-fatal heading."""
        out = {}
        for d, (nb, nh, nf, ate) in self.legal_moves(body, heading, food).items():
            reach = self.flood(nb)
            tail = nb[-1]
            tail_ok = any((tail[0] + dx, tail[1] + dy) in reach or (tail[0] + dx, tail[1] + dy) == nb[0]
                          for dx, dy in DIRS.values())
            out[d] = {
                "dist": 0 if ate else self.bfs_food(nb, nf),
                "area": len(reach),
                "tail_ok": tail_ok,
                "ate": ate,
                "next": (nb, nh, nf),
            }
        return out

    def render(self, body, heading, food):
        rows = []
        bset = set(body)
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
    MAX_ASTRA_PER_EPISODE = 4
    TRAP_COOLDOWN_TICKS = 25

    def __init__(self, ctx):
        self.ctx = ctx
        mem = ctx.memory
        self.rules = mem.get("snake_rules", DEFAULT_RULES)
        self.mode = mem.get("snake_mode", "shortest")
        self.session = mem.get("snake_session")
        self.astra_pending = None
        self.astra_calls = 0

    # ---------------------------------------------------------------- Astra
    def _astra_prompt(self, game, grid, length, reason):
        return (
            f"{game.describe()}\nActions:\n{game.action_spec.describe()}\n\n"
            f"Board now (length {length}):\n{grid}\n\n"
            f"Context: {reason}\n"
            "You are the slow planner for a fast reflex classifier that, each tick, picks one of the "
            "SAFE headings the harness pre-computed. For each option it is told: steps to the food, "
            "number of free cells reachable after the move, and whether the tail stays reachable. "
            "It cannot reason; it only matches your rules against those numbers.\n"
            "Write at most 5 short imperative rules (one per line, most important first) about how to "
            "trade off distance-to-food against free space and tail reachability as the snake grows. "
            "Then a final line exactly 'DEFAULT: shortest' or 'DEFAULT: safest' telling the harness what "
            "to do when the classifier is late. No other prose."
        )

    def _ask_astra(self, game, grid, length, reason):
        if self.astra_calls >= self.MAX_ASTRA_PER_EPISODE:
            return
        self.astra_calls += 1
        try:
            res = self.ctx.think(self._astra_prompt(game, grid, length, reason), session_id=self.session)
        except Exception as e:  # budget or backend — keep old rules
            self.ctx.log(event="astra_failed", error=str(e)[:300])
            return
        self.session = res.session_id
        text = (res.text or "").strip()
        if not text:
            return
        m = re.search(r"DEFAULT:\s*(shortest|safest)", text, re.I)
        if m:
            self.mode = m.group(1).lower()
        rules = re.sub(r"DEFAULT:.*", "", text, flags=re.I | re.S).strip()
        if rules:
            self.rules = rules[:700]
        self.ctx.memory["snake_rules"] = self.rules
        self.ctx.memory["snake_mode"] = self.mode
        self.ctx.memory["snake_session"] = self.session
        self.ctx.log(event="plan_update", reason=reason, mode=self.mode, rules=self.rules)

    def _spawn_astra(self, game, grid, length, reason):
        if self.astra_pending is not None and not self.astra_pending.done():
            return
        self.astra_pending = self.ctx.spawn(lambda: self._ask_astra(game, grid, length, reason))

    # ---------------------------------------------------------------- heuristic / candidates
    def _rank_key(self, d, o, length):
        safe_enough = o["tail_ok"] or o["area"] >= 2 * length
        dist = o["dist"] if o["dist"] is not None else 10_000
        if self.mode == "safest":
            return (0 if safe_enough else 1, -min(o["area"], 3 * length), dist, -o["area"], d)
        return (0 if safe_enough else 1, dist, -o["area"], d)

    def heuristic(self, opts, length, heading):
        if not opts:
            return heading  # everything kills; keep going
        return min(opts, key=lambda d: self._rank_key(d, opts[d], length))

    def candidates(self, opts, length):
        """Non-dominated safe options (distance vs. area). Only these are offered to Jev."""
        good = {d: o for d, o in opts.items() if o["tail_ok"] or o["area"] >= 2 * length}
        pool = good or opts
        if len(pool) < 2:
            return list(pool)
        dists = [o["dist"] for o in pool.values() if o["dist"] is not None]
        best = min(dists) if dists else None
        nd = []
        for d, o in pool.items():
            od = o["dist"] if o["dist"] is not None else 10_000
            if best is not None and od > best + 6:
                continue
            dominated = False
            for d2, o2 in pool.items():
                if d2 == d:
                    continue
                o2d = o2["dist"] if o2["dist"] is not None else 10_000
                if o2d <= od and o2["area"] >= o["area"] and (o2["tail_ok"] or not o["tail_ok"]) \
                        and (o2d < od or o2["area"] > o["area"] or (o2["tail_ok"] and not o["tail_ok"])):
                    dominated = True
                    break
            if not dominated:
                nd.append(d)
        return nd

    # ---------------------------------------------------------------- Jev
    def _ask_jev(self, pred, opts, cands, grid):
        body, heading, food = pred
        head = body[0]
        crit = {}
        for d in cands:
            o = opts[d]
            dist = "eats the food now" if o["ate"] else (
                f"food {o['dist']} steps away" if o["dist"] is not None else "food unreachable")
            crit[d] = (f"{d}: {dist}; {o['area']} free cells reachable afterwards; "
                       f"tail {'reachable' if o['tail_ok'] else 'CUT OFF'}")
        state = {
            "planner_rules": self.rules,
            "head": list(head), "heading": heading, "length": len(body),
            "food_delta": [food[0] - head[0], food[1] - head[1]] if food else None,
            "options": {d: {"steps_to_food": opts[d]["dist"], "free_cells_after": opts[d]["area"],
                            "tail_reachable": opts[d]["tail_ok"]} for d in cands},
            "grid": grid,
        }
        q = {"move": Choice(
            instructions="All listed headings are safe this tick. Following planner_rules, pick the "
                         "heading whose consequences best match the rules.",
            criteria=crit)}
        try:
            ans = self.ctx.decide(state, q)
        except Exception as e:
            self.ctx.log(event="jev_failed", error=str(e)[:200])
            return None
        a = ans["move"]
        return (a.value, dict(a.probabilities or {}))

    # ---------------------------------------------------------------- loop helpers
    @staticmethod
    def _key(body, heading, food):
        return (tuple(body), heading, food)

    def _wait_next_tick(self, game, tick):
        """Poll until the game ticks (or ends). Returns (obs, boundary_time)."""
        while not self.ctx.done:
            obs = game.observe()
            if obs["tick"] != tick:
                return obs, time.time()
            time.sleep(0.004)
        return None, time.time()

    # ---------------------------------------------------------------- main loop
    def run_episode(self, game):
        ctx = self.ctx
        obs = game.observe()
        w, h = obs["grid_size"]
        sim = Sim(w, h)
        tick_s = float(getattr(game, "tick_s", None) or ctx.tick_s or 0.5)
        margin = 0.08  # seconds before the next boundary after which a Jev answer is not posted

        if "snake_rules" not in ctx.memory:
            self._spawn_astra(game, obs["grid"], obs["length"], "episode start; no rules yet")

        pending = None      # (tick, key, future, cands)
        boundary = time.time()  # best estimate of the last tick boundary
        last_trap_tick = -10_000
        n_jev_used = n_jev_late = n_mismatch = 0

        while not ctx.done and obs is not None:
            body = [tuple(p) for p in obs["body"]]
            heading = obs["heading"]
            food = tuple(obs["food"]) if obs.get("food") is not None else None
            length = len(body)
            opts = sim.options(body, heading, food)
            fallback = self.heuristic(opts, length, heading)
            action = fallback
            ctx.act(action)  # never leave a tick stale

            # ---- late-decision policy: use Jev's answer about the predicted state if it matches
            if pending is not None:
                p_tick, p_key, fut, cands = pending
                pending = None
                if p_tick == obs["tick"] and p_key == self._key(body, heading, food):
                    deadline = boundary + tick_s - margin
                    res = None
                    try:
                        res = fut.result(timeout=max(0.0, deadline - time.time()))
                    except Exception:
                        n_jev_late += 1
                    if res is not None and res[0] in opts and res[0] in cands \
                            and game.observe()["tick"] == obs["tick"]:
                        action = res[0]
                        if action != fallback:
                            ctx.act(action)
                        n_jev_used += 1
                        ctx.log(event="jev_pick", tick=obs["tick"], pick=action, fallback=fallback,
                                probs=res[1], cands=cands)
                else:
                    n_mismatch += 1

            # ---- trap detection -> Astra replan in background
            if opts and not any(o["tail_ok"] for o in opts.values()) \
                    and obs["tick"] - last_trap_tick >= self.TRAP_COOLDOWN_TICKS:
                last_trap_tick = obs["tick"]
                self._spawn_astra(game, obs["grid"], length,
                                  "the harness sees no move that keeps the tail reachable; "
                                  "revise the rules for escaping tight space")

            # ---- predict the next state and pre-ask Jev about it
            if action in opts:
                nb, nh, nf = opts[action]["next"]
                popts = sim.options(nb, nh, nf)
                cands = self.candidates(popts, len(nb))
                if len(cands) >= 2 and ctx.budget.remaining().get("system1_calls", 0) > 5:
                    grid = sim.render(nb, nh, nf)
                    pred = (nb, nh, nf)
                    fut = ctx.spawn(lambda p=pred, o=popts, c=cands, g=grid: self._ask_jev(p, o, c, g))
                    pending = (obs["tick"] + 1, self._key(nb, nh, nf), fut, cands)

            obs, boundary = self._wait_next_tick(game, obs["tick"])

        ctx.log(event="episode_summary", jev_used=n_jev_used, jev_late=n_jev_late,
                pred_mismatch=n_mismatch, astra_calls=self.astra_calls, mode=self.mode)
        stats = ctx.memory.setdefault("snake_stats", [])
        stats.append({"jev_used": n_jev_used, "jev_late": n_jev_late, "mismatch": n_mismatch})


def build(ctx):
    return Harness(ctx)
