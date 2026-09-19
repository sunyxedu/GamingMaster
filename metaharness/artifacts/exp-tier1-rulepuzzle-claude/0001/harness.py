"""Candidate 0001 — scripted hypothesis engine, Astra as stuck-advisor.

Parent (0000) burned its 15 Astra calls on 15 single actions and mis-parsed
Astra's commentary (first ACTION token instead of the decision).  Here the
harness itself runs a cheap System-0 loop:

  * action -> direction map learned by elimination (every unknown action is
    tried only when the planner *wants* a direction, so exploration is never
    wasted);
  * a colour-semantics belief: passable / blocked / consumable(key) /
    toggleable / teleport, updated from bumps and grid diffs;
  * Dijkstra over (pos, keys) with penalties for untested colours; blocked
    hypotheses are *soft* and get cleared whenever the world changes (key
    picked up, tile toggled, teleport discovered) — that is the refutation
    mechanism;
  * cross-seed memory keeps colour statistics only as priors.

Astra is consulted only when the scripted engine has no plan left; it sees an
ASCII map + legend + learned facts and must answer with a final `PLAN:` line
that is parsed from the *end* of the reply.
"""
from __future__ import annotations

import random
import re
from typing import Any, Optional

ACTIONS = [f"ACTION{i}" for i in range(1, 7)]
MOVE_CANDIDATES = ACTIONS[:4]
INTERACT_CANDIDATES = ACTIONS[4:]
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
VEC2DIR = {v: k for k, v in DIRS.items()}
EMPTY, EXIT, AGENT = 0, 4, 9
UNKNOWN_PENALTY = 2          # cost surcharge for stepping onto an untested colour
HARD_BUMPS = 3               # bumps in one episode after which a colour is a hard wall
MAX_ASTRA_PLAN = 10


def _find(grid, val):
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            if c == val:
                return (x, y)
    return None


class World:
    """Belief state for one episode (rules are fixed within an episode)."""

    def __init__(self, prior_walls: set[int]):
        self.size = 8
        self.grid: list[list[int]] = []
        self.pos = (0, 0)
        self.exit: Optional[tuple[int, int]] = None
        self.level = 0
        self.actions_left = 0
        # action semantics
        self.action_dir: dict[str, str] = {}                       # confirmed
        self.cands: dict[str, set[str]] = {a: set(DIRS) for a in MOVE_CANDIDATES}
        self.interact: Optional[str] = None
        self.interact_tried: dict[int, set[str]] = {}              # colour -> actions tried next to it
        # colour semantics
        self.passable: set[int] = {EMPTY, EXIT}
        self.blocked: dict[int, int] = {c: 1 for c in prior_walls}  # colour -> bump count (soft)
        self.consumable: set[int] = set()
        self.unlocks: dict[int, int] = {}                          # door colour -> key colour
        self.toggleable: set[int] = set()
        self.tele_colours: set[int] = set()
        self.tele: dict[tuple[int, int], tuple[int, int]] = {}
        # per-level
        self.keys: set[int] = set()
        self.visited: set[tuple[int, int]] = set()
        self.history: list[str] = []
        self.events: list[str] = []

    # -- observation ----------------------------------------------------
    def ingest(self, obs, under_agent: Optional[int] = None):
        g = obs["grid"]
        self.size = len(g)
        new_level = obs["level"] != self.level
        self.level = obs["level"]
        self.actions_left = obs.get("actions_left_this_level", 0)
        pos = _find(g, AGENT) or self.pos
        if new_level:
            self.keys = set()
            self.visited = set()
            self.tele = {}
            under_agent = EMPTY
            self.grid = [row[:] for row in g]
        else:
            for y in range(self.size):
                for x in range(self.size):
                    if (x, y) != pos:
                        self.grid[y][x] = g[y][x]
        self.pos = pos
        self.grid[pos[1]][pos[0]] = under_agent if under_agent is not None else EMPTY
        self.visited.add(pos)
        if self.exit is None or new_level:
            self.exit = _find(g, EXIT)
        return new_level

    def colour(self, p):
        x, y = p
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[y][x]
        return None

    def is_hard(self, c):
        return c is not None and self.blocked.get(c, 0) >= HARD_BUMPS and c not in self.passable

    def can_enter(self, c, keys):
        if c is None:
            return False
        if self.is_hard(c):
            return False
        if c in self.unlocks:
            return self.unlocks[c] in keys
        return c not in self.blocked

    def soft_clear_blocked(self, why):
        kept = {c: n for c, n in self.blocked.items() if self.is_hard(c)}
        dropped = set(self.blocked) - set(kept)
        self.blocked = kept
        if dropped:
            self.events.append(f"world changed ({why}) -> re-testing colours {sorted(dropped)}")

    # -- action selection -----------------------------------------------
    def action_for(self, d):
        for a, dd in self.action_dir.items():
            if dd == d:
                return a
        unknown = [a for a in MOVE_CANDIDATES if a not in self.action_dir and d in self.cands[a]]
        if unknown:
            return min(unknown, key=lambda a: len(self.cands[a]))
        unknown = [a for a in MOVE_CANDIDATES if a not in self.action_dir]
        return unknown[0] if unknown else random.choice(MOVE_CANDIDATES)

    def confirm(self, a, d):
        if self.action_dir.get(a) == d:
            return
        self.action_dir[a] = d
        self.cands[a] = {d}
        for b in MOVE_CANDIDATES:
            if b != a:
                self.cands[b].discard(d)
        # a direction can belong to one action only
        for b, dd in list(self.action_dir.items()):
            if b != a and dd == d:
                del self.action_dir[b]
                self.cands[b] = set(DIRS) - set(self.action_dir.values())

    # -- learning from one transition -----------------------------------
    def learn(self, a, before_pos, before_grid, intended: Optional[str]):
        dx, dy = self.pos[0] - before_pos[0], self.pos[1] - before_pos[1]
        note = ""
        moved = (dx, dy) != (0, 0)
        if a in MOVE_CANDIDATES:
            if (dx, dy) in VEC2DIR:
                d = VEC2DIR[(dx, dy)]
                self.confirm(a, d)
                c = before_grid[self.pos[1]][self.pos[0]]
                if c in self.blocked and c not in self.unlocks:
                    # a blocked colour became passable -> refuted; if we hold a key, attribute it
                    del self.blocked[c]
                    if self.keys:
                        self.unlocks[c] = sorted(self.keys)[-1]
                        self.events.append(f"colour {c} opened after picking colour {self.unlocks[c]}")
                self.passable.add(c)
                note = f"{a}={d} ok"
            elif moved:
                # non-unit displacement: teleport.  Which neighbour did we enter?
                land_c = before_grid[self.pos[1]][self.pos[0]]
                entered = [(d, (before_pos[0] + v[0], before_pos[1] + v[1])) for d, v in DIRS.items()
                           if 0 <= before_pos[0] + v[0] < self.size and 0 <= before_pos[1] + v[1] < self.size
                           and before_grid[before_pos[1] + v[1]][before_pos[0] + v[0]] == land_c]
                if len(entered) == 1:
                    d, cell = entered[0]
                    self.confirm(a, d)
                    self.tele[cell] = self.pos
                    self.tele[self.pos] = cell
                    self.tele_colours.add(land_c)
                    self.passable.add(land_c)
                    self.blocked.pop(land_c, None)
                    self.events.append(f"colour {land_c} teleports: {cell} <-> {self.pos}")
                    self.soft_clear_blocked("teleport")
                note = f"{a} teleported to {self.pos}"
            else:
                d = self.action_dir.get(a)
                if d is not None:
                    n = (before_pos[0] + DIRS[d][0], before_pos[1] + DIRS[d][1])
                    c = self.colour(n)
                    if c is not None:
                        if c == EMPTY:
                            # a confirmed direction bumped into empty: our map is wrong
                            del self.action_dir[a]
                            self.cands[a] = set(DIRS) - set(self.action_dir.values())
                            note = f"{a} contradiction, unlearned"
                        else:
                            self.blocked[c] = self.blocked.get(c, 0) + 1
                            self.passable.discard(c) if c not in self.unlocks else None
                            note = f"{a}={d} bumped colour {c}"
                    else:
                        note = f"{a}={d} bumped edge"
                else:
                    # eliminate directions whose neighbour is certainly passable
                    for dd, v in DIRS.items():
                        n = (before_pos[0] + v[0], before_pos[1] + v[1])
                        c = self.colour(n)
                        if c is not None and (c == EMPTY or (c in self.passable and self.can_enter(c, self.keys))):
                            self.cands[a].discard(dd)
                    if not self.cands[a]:
                        self.cands[a] = set(DIRS) - set(self.action_dir.values())
                    note = f"{a} no move, candidates {sorted(self.cands[a])}"
        # grid diff (excluding current agent cell): world events
        left_c = before_grid[before_pos[1]][before_pos[0]]
        changed = []
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) == self.pos:
                    continue
                if self.grid[y][x] != before_grid[y][x]:
                    changed.append(((x, y), before_grid[y][x], self.grid[y][x]))
        if moved and (before_pos, left_c) and any(p == before_pos for p, _, _ in changed):
            new_c = self.grid[before_pos[1]][before_pos[0]]
            if left_c not in (EMPTY, EXIT) and new_c == EMPTY:
                self.consumable.add(left_c)
                self.keys.add(left_c)
                self.events.append(f"colour {left_c} vanished when stepped on (key-like)")
                self.soft_clear_blocked("key")
            changed = [ch for ch in changed if ch[0] != before_pos]
        if changed and a in INTERACT_CANDIDATES:
            self.interact = a
            cols = {old for _, old, new in changed if new == EMPTY and old not in (EMPTY, EXIT)}
            self.toggleable |= cols
            for c in cols:
                self.blocked.pop(c, None)
            self.events.append(f"{a} is INTERACT: cleared colours {sorted(cols)} at {[p for p, _, _ in changed]}")
            self.soft_clear_blocked("toggle")
            note = f"{a} toggled {sorted(cols)}"
        elif a in INTERACT_CANDIDATES and not note:
            note = f"{a} no visible effect"
        self.history.append(f"{a}: {before_pos}->{self.pos} {note}".strip())
        return note

    # -- planning ---------------------------------------------------------
    def plan(self, goal_fn, keys=None):
        """Dijkstra over (pos, keys). Returns list of (dir, interact_first) or None."""
        import heapq
        keys = frozenset(self.keys if keys is None else keys)
        start = (self.pos, keys)
        dist = {start: 0}
        prev: dict = {}
        pq = [(0, 0, start)]
        cnt = 1
        while pq:
            d0, _, (p, ks) = heapq.heappop(pq)
            if dist.get((p, ks), 1e9) < d0:
                continue
            if goal_fn(p, ks) and p != self.pos:
                path = []
                cur = (p, ks)
                while cur in prev:
                    pcur, step = prev[cur]
                    path.append(step)
                    cur = pcur
                return path[::-1]
            for dname, v in DIRS.items():
                n = (p[0] + v[0], p[1] + v[1])
                c = self.colour(n)
                if c is None:
                    continue
                inter = False
                cost = 1
                if c in self.toggleable and self.interact and not self.can_enter(c, ks):
                    inter = True
                    cost = 2
                elif not self.can_enter(c, ks):
                    continue
                if c not in self.passable and c not in self.consumable and c not in self.tele_colours:
                    cost += UNKNOWN_PENALTY
                nks = ks | {c} if c in self.consumable else ks
                land = n
                if c in self.tele_colours:
                    twin = self.tele.get(n)
                    if twin is None:
                        others = [(x, y) for y in range(self.size) for x in range(self.size)
                                  if self.grid[y][x] == c and (x, y) != n]
                        if len(others) == 1:
                            twin = others[0]
                    if twin is not None:
                        land = twin
                st = (land, nks)
                nd = d0 + cost
                if nd < dist.get(st, 1e9):
                    dist[st] = nd
                    prev[st] = ((p, ks), (dname, inter))
                    cnt += 1
                    heapq.heappush(pq, (nd, cnt, st))
        return None

    def plan_to_exit(self):
        ex = self.exit
        if ex is None:
            return None
        return self.plan(lambda p, ks: p == ex)

    def plan_to_special(self):
        def goal(p, ks):
            c = self.colour(p)
            return (c not in (EMPTY, EXIT) and p not in self.visited and not self.is_hard(c)
                    and c not in self.toggleable)
        return self.plan(goal)

    # -- presentation -----------------------------------------------------
    def ascii_map(self):
        rows = ["    " + " ".join(str(x) for x in range(self.size))]
        for y in range(self.size):
            cells = []
            for x in range(self.size):
                cells.append("@" if (x, y) == self.pos else str(self.grid[y][x]))
            rows.append(f"y={y} " + " ".join(cells))
        return "\n".join(rows)

    def facts(self, legend):
        lines = []
        for a in MOVE_CANDIDATES:
            if a in self.action_dir:
                lines.append(f"{a} = move {self.action_dir[a]} (confirmed)")
            else:
                lines.append(f"{a} = move, direction one of {sorted(self.cands[a])} (unconfirmed)")
        if self.interact:
            lines.append(f"{self.interact} = INTERACT (removes toggleable tiles within 1 cell)")
        else:
            lines.append("ACTION5/ACTION6: no visible effect seen so far (maybe context-dependent)")
        def nm(c):
            return f"{c}({legend.get(str(c), '?')})"
        if self.passable - {EMPTY, EXIT}:
            lines.append("walkable colours: " + ", ".join(nm(c) for c in sorted(self.passable - {EMPTY, EXIT})))
        if self.blocked:
            lines.append("colours that blocked movement: " + ", ".join(f"{nm(c)} x{n}" for c, n in sorted(self.blocked.items())))
        if self.consumable:
            lines.append("consumed (key-like) colours: " + ", ".join(nm(c) for c in sorted(self.consumable)))
        if self.unlocks:
            lines.append("unlock rules: " + ", ".join(f"{nm(d)} opens after {nm(k)}" for d, k in self.unlocks.items()))
        if self.toggleable:
            lines.append("toggleable colours: " + ", ".join(nm(c) for c in sorted(self.toggleable)))
        if self.tele_colours:
            lines.append("teleport colours: " + ", ".join(nm(c) for c in sorted(self.tele_colours)) + f" pairs {self.tele}")
        return "\n".join(lines)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.astra_ok = True
        mem = ctx.memory
        mem.setdefault("colour_stats", {})
        mem.setdefault("episodes", 0)
        mem.setdefault("notes", [])

    # -- memory ---------------------------------------------------------
    def prior_walls(self) -> set[int]:
        out = set()
        for c, st in self.ctx.memory["colour_stats"].items():
            if st.get("blocked", 0) >= 3 and st.get("passable", 0) == 0:
                out.add(int(c))
        return out

    def record_stats(self, w: World):
        st = self.ctx.memory["colour_stats"]
        for c in w.passable - {EMPTY, EXIT}:
            st.setdefault(str(c), {"blocked": 0, "passable": 0})["passable"] += 1
        for c, n in w.blocked.items():
            if c not in w.passable:
                st.setdefault(str(c), {"blocked": 0, "passable": 0})["blocked"] += 1

    # -- acting ---------------------------------------------------------
    def act(self, w: World, a: str, intended: Optional[str] = None):
        before_pos, before_grid = w.pos, [r[:] for r in w.grid]
        obs = self.ctx.act(a)
        # colour under the agent after the step: what the cell showed before we entered it
        pos = _find(obs["grid"], AGENT) or before_pos
        under = before_grid[pos[1]][pos[0]] if obs["level"] == w.level else EMPTY
        new_level = w.ingest(obs, under)
        if new_level:
            w.history.append(f"{a}: level {w.level} started")
            return
        note = w.learn(a, before_pos, before_grid, intended)
        self.ctx.log(type="harness", step=a, note=note, pos=list(w.pos))

    def go(self, w: World, d: str):
        self.act(w, w.action_for(d), intended=d)

    # -- scripted policy ------------------------------------------------
    def scripted_step(self, w: World) -> bool:
        path = w.plan_to_exit()
        if path:
            d, inter = path[0]
            if inter and w.interact:
                self.act(w, w.interact)
                return True
            self.go(w, d)
            return True
        # stuck: (1) test interact next to a blocking colour
        for v in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            n = (w.pos[0] + v[0], w.pos[1] + v[1])
            c = w.colour(n)
            if c is None or c in (EMPTY, EXIT) or w.is_hard(c) or c in w.consumable or c in w.tele_colours:
                continue
            if c not in w.blocked and c not in w.toggleable:
                continue
            tried = w.interact_tried.setdefault(c, set())
            cand = [w.interact] if w.interact else INTERACT_CANDIDATES
            todo = [a for a in cand if a not in tried]
            if todo:
                tried.add(todo[0])
                self.act(w, todo[0])
                return True
        # (2) walk to an unvisited special tile (could be a key or a teleport)
        path = w.plan_to_special()
        if path:
            d, inter = path[0]
            if inter and w.interact:
                self.act(w, w.interact)
            else:
                self.go(w, d)
            return True
        return False

    # -- Astra ----------------------------------------------------------
    @staticmethod
    def parse_plan(text: str) -> list[str]:
        lines = [l for l in (text or "").splitlines() if l.strip()]
        for l in reversed(lines):
            if re.search(r"PLAN\s*:", l, re.I):
                acts = re.findall(r"ACTION\s*([1-6])", l.split(":", 1)[1], re.I)
                if acts:
                    return [f"ACTION{n}" for n in acts]
        for l in reversed(lines):
            acts = re.findall(r"ACTION\s*([1-6])", l, re.I)
            if acts:
                return [f"ACTION{n}" for n in acts]
        return []

    def ask_astra(self, w: World, game) -> bool:
        ctx = self.ctx
        if not self.astra_ok or ctx.budget.remaining()["system2_calls"] <= 0:
            return False
        legend = ctx.observe().get("legend", {})
        prompt = (
            f"{game.describe()}\n\n"
            "You are advising a scripted controller that got STUCK: with its current beliefs it finds no path "
            "to the yellow exit and no untested special tile to visit.\n"
            f"Coordinates are (x,y): x = column (0 = left), y = row (0 = top). '@' is you at {w.pos}; exit at {w.exit}.\n"
            f"Level {w.level}, actions left this level: {w.actions_left}.\n"
            f"Legend (digit=colour): {legend}\n\nMap:\n{w.ascii_map()}\n\n"
            f"What the controller has learned this episode:\n{w.facts(legend)}\n\n"
            "Recent transitions (action: from->to note):\n" + "\n".join(w.history[-12:]) + "\n\n"
            "Discoveries so far: " + ("; ".join(w.events[-6:]) or "none") + "\n\n"
            "Think briefly about which belief is probably wrong (a colour marked blocked may be a door needing a key, "
            "a tile may be a teleport, an interact action may only work next to a specific colour, a move may be "
            "mislabelled). Then give a short experiment or route.\n"
            f"Your LAST line must be exactly of the form `PLAN: ACTIONx ACTIONy ...` (1 to {MAX_ASTRA_PLAN} actions). "
            "Optionally add `RETRY_BLOCKED` on the line before it if you want the controller to forget its blocked-colour beliefs."
        )
        try:
            res = ctx.think(prompt, session_id=self.session)
        except RuntimeError as e:
            self.astra_ok = False
            ctx.log(type="harness", astra_error=str(e))
            return False
        self.session = res.session_id
        text = res.text or ""
        plan = self.parse_plan(text)[:MAX_ASTRA_PLAN]
        if re.search(r"RETRY_BLOCKED", text):
            w.blocked = {}
        ctx.log(type="harness", astra_plan=plan, retry="RETRY_BLOCKED" in text)
        if not plan:
            return False
        lvl = w.level
        for a in plan:
            if ctx.done or w.level != lvl:
                break
            self.act(w, a)
        return True

    def fallback(self, w: World):
        # last resort: forget blocked beliefs once, then take a random move
        if w.blocked:
            w.blocked = {}
            w.events.append("fallback: forgot all blocked colours")
            return
        self.act(w, random.choice(MOVE_CANDIDATES))

    # -- episode --------------------------------------------------------
    def run_episode(self, game):
        ctx = self.ctx
        w = World(self.prior_walls())
        w.ingest(ctx.observe(), EMPTY)
        ctx.log(type="harness", prior_walls=sorted(self.prior_walls()), episode=ctx.memory["episodes"])
        guard = 0
        while not ctx.done and guard < 400:
            guard += 1
            if self.scripted_step(w):
                continue
            ctx.log(type="harness", stuck=True, pos=list(w.pos), blocked=dict(w.blocked))
            if self.ask_astra(w, game):
                continue
            self.fallback(w)
        self.record_stats(w)
        ctx.memory["episodes"] += 1
        ctx.memory["notes"].append({"level_reached": w.level, "events": w.events[-8:],
                                    "moves": dict(w.action_dir), "interact": w.interact})
        ctx.log(type="harness", episode_end=True, moves=dict(w.action_dir), interact=w.interact,
                events=w.events, level=w.level)


def build(ctx):
    return Harness(ctx)
