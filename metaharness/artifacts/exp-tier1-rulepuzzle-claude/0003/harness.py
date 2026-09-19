"""Candidate 0003 — 0002 + cross-level transfer of colour semantics ("rules are episode constants").

Diagnosis of 0002 (300 local seeds, per level, actions minus full-knowledge oracle):
teleport L1 +3.9 / L2 +0.0, mirror +2.8 / +0.0, toggle +4.6 / +1.3, key_door +2.6 / **+6.5**
(worst seeds +20).  Level 2 is pure planning once the rules are known, yet in
key_door 0002 re-discovers the door from scratch: the door colour was walked
through in level 1 (key picked up incidentally) and is in `passable`; in level 2
it bumps it, spends 2 interact tests on it, walks to the key, stands ON the key
without knowing it holds it (vanish is only detected when leaving), tests
interact twice on a grey wall, reports "stuck", and only then walks through the
door.  Three inference rules fix this without touching the level-1 engine:

  * DOOR-FROM-HISTORY: a colour that was passable earlier in the episode and
    blocks now, while a consumable colour is known -> it is a door opened by
    that key (unlocks[c] = key); no interact tests on it.
  * KEY-UNDER-FEET: entering a cell whose colour is already known consumable
    means we hold that key immediately (no need to step off to see it vanish).
  * DOOR-WHEN-KEY-KNOWN: bumping a *rare* colour while a consumable colour
    exists and the bumped colour is not toggleable -> door hypothesis first,
    interact test second (the test is still run if the hypothesis is refuted).

Original 0002 description follows.


Parent (0001) is already perfect on the evaluation config (2.0, 0 Astra calls)
but wastes actions: it only tests an interact / key hypothesis when it has *no*
path at all, so whenever a detour exists it walks the detour (13 extra moves on
toggle seed 15, 11 on key_door seed 38).  Under a tighter budget (20/level)
that costs 0.26 levels per episode.  This candidate adds an explicit
cost comparison every step:

    detour = len(belief path) - len(optimistic path)   (optimistic = soft-blocked
                                                        colours assumed open)
    if detour > cost(experiment):  run the experiment instead of walking

Experiments, cheapest first: (1) untested interact actions while adjacent to the
blocking colour (<= 2 actions); (2) walk to an unvisited special tile (key
hypothesis) when  d(pos,tile) + d_opt(tile,exit) < len(belief path).  Every
experiment is a refutable hypothesis: no grid change -> the colour is marked
interact-tested; tile did not vanish -> it is visited and never retried.

Everything else (belief model, planner, refutation by world events, Astra as
stuck-advisor with end-of-reply PLAN parsing, cross-seed colour priors) is
inherited from 0001.


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
UNKNOWN_PENALTY = 2          # cost surcharge for stepping onto an untested *frequent* colour
RARE_COLOUR = 4              # <= this many cells on the grid -> "rare" colour (probably special, not wall)
RARE_STEP_COST = 0.6         # entering an unvisited rare unknown tile is *attractive* (curiosity)
FREQUENT_INTERACT_PENALTY = 4  # extra cost charged to an interact experiment next to a frequent colour
UNKNOWN_DIR_COST = 0.5       # expected extra action when a step needs a not-yet-identified move action
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
        self.bumped: set[int] = set()                              # colours bumped in THIS episode
        self.failed_at: set[tuple[str, tuple[int, int]]] = set()   # (action, pos) that produced no move
        self.no_move_streak = 0
        self.consumable: set[int] = set()
        self.unlocks: dict[int, int] = {}                          # door colour -> key colour
        self.door_refuted: set[int] = set()                        # colours whose door hypothesis failed
        self.enter_keys: dict[int, list[frozenset]] = {}           # colour -> keys held each time we entered it
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
            self.pos = pos
            self.level_start_inference()
        else:
            for y in range(self.size):
                for x in range(self.size):
                    if (x, y) != pos:
                        self.grid[y][x] = g[y][x]
        self.pos = pos
        if under_agent in self.consumable and under_agent not in self.keys:
            # KEY-UNDER-FEET: we know this colour is a key; we hold it the moment we step on it
            self.keys.add(under_agent)
            self.events.append(f"picked known key colour {under_agent} at {pos}")
            under_agent = EMPTY
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

    def count_colour(self, c):
        return sum(row.count(c) for row in self.grid)

    def is_rare(self, c):
        return self.count_colour(c) <= RARE_COLOUR

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

    def door_hypothesis(self, c):
        """Called when colour ``c`` just blocked us.  Rules are constant within the episode, so:
        (a) if ``c`` was passable earlier and a key colour is known, it is a door for that key;
        (b) if ``c`` is rare, not toggleable, and a key colour is known, hypothesise door first."""
        if c in self.door_refuted or c in self.toggleable or c in self.tele_colours or c in self.consumable:
            return False
        held_before = [k for k in sorted(self.consumable) if k not in self.keys]
        if not held_before:
            return False
        key = held_before[-1]
        if c in self.passable:
            self.unlocks[c] = key
            self.blocked.pop(c, None)
            self.interact_tried.setdefault(c, set()).update(INTERACT_CANDIDATES)   # no interact tests
            self.events.append(f"colour {c} was walkable earlier, blocks now without key -> door opened by key {key}")
            return True
        if self.is_rare(c):
            self.unlocks[c] = key
            self.blocked.pop(c, None)
            self.events.append(f"hypothesis: rare colour {c} is a door opened by key {key} (key colour known)")
            return True
        return False

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
        fresh = lambda a: (a, self.pos) not in self.failed_at
        unknown = [a for a in MOVE_CANDIDATES if a not in self.action_dir and d in self.cands[a] and fresh(a)]
        if unknown:
            return min(unknown, key=lambda a: len(self.cands[a]))
        unknown = [a for a in MOVE_CANDIDATES if a not in self.action_dir and fresh(a)]
        if unknown:
            return unknown[0]
        # every candidate already failed here: the wanted direction is blocked by an unknown colour,
        # so take a confirmed action that moves us somewhere (break the loop), else a random one
        known = [a for a in MOVE_CANDIDATES if a in self.action_dir and fresh(a)]
        return random.choice(known or MOVE_CANDIDATES)

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
        if a in MOVE_CANDIDATES and not moved:
            self.failed_at.add((a, before_pos))
        self.no_move_streak = 0 if moved else self.no_move_streak + 1
        if a in MOVE_CANDIDATES:
            if (dx, dy) in VEC2DIR:
                d = VEC2DIR[(dx, dy)]
                self.confirm(a, d)
                c = before_grid[self.pos[1]][self.pos[0]]
                if c in self.unlocks and self.unlocks[c] not in self.keys:
                    self.events.append(f"colour {c} passable without key -> unlock hypothesis dropped")
                    del self.unlocks[c]
                    self.door_refuted.add(c)
                    self.blocked.pop(c, None)
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
                            self.bumped.add(c)
                            if c in self.unlocks and self.unlocks[c] in self.keys:
                                self.events.append(f"colour {c} still blocked while holding {self.unlocks[c]} -> unlock hypothesis refuted")
                                del self.unlocks[c]
                                self.door_refuted.add(c)
                            elif c not in self.unlocks:
                                self.door_hypothesis(c)
                            if c not in self.unlocks:
                                self.passable.discard(c)
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
                    if len(self.cands[a]) == 1:
                        d1 = next(iter(self.cands[a]))
                        self.confirm(a, d1)
                        n1 = (before_pos[0] + DIRS[d1][0], before_pos[1] + DIRS[d1][1])
                        c1 = self.colour(n1)
                        if c1 is not None and c1 != EMPTY:
                            self.blocked[c1] = self.blocked.get(c1, 0) + 1
                            self.bumped.add(c1)
                            if c1 not in self.unlocks:
                                self.door_hypothesis(c1)
                            if c1 not in self.unlocks:
                                self.passable.discard(c1)
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
                for dc in list(self.blocked):
                    if dc in self.bumped and not self.is_hard(dc) and dc not in self.unlocks and dc != left_c and self.count_colour(dc) <= 4:
                        self.unlocks[dc] = left_c          # hypothesis: door dc opens with key left_c
                        self.events.append(f"hypothesis: colour {dc} is a door opened by key {left_c}")
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
        if a in MOVE_CANDIDATES and (dx, dy) in VEC2DIR:
            c_in = before_grid[self.pos[1]][self.pos[0]]
            if c_in not in (EMPTY, EXIT):
                self.enter_keys.setdefault(c_in, []).append(frozenset(self.keys))
        self.history.append(f"{a}: {before_pos}->{self.pos} {note}".strip())
        return note

    def level_start_inference(self):
        """CONDITIONAL-PASSABILITY: a rare colour that was only ever entered while holding key K is
        hypothesised to be a door for K at the start of a new level (keys are lost between levels)."""
        for c, entries in self.enter_keys.items():
            if c in self.unlocks or c in self.door_refuted or c in self.consumable \
                    or c in self.toggleable or c in self.tele_colours or not entries:
                continue
            common = frozenset.intersection(*entries)
            if common and self.is_rare(c):
                key = sorted(common)[-1]
                self.unlocks[c] = key
                self.blocked.pop(c, None)
                self.interact_tried.setdefault(c, set()).update(INTERACT_CANDIDATES)
                self.events.append(f"level start: colour {c} was only walkable while holding key {key} -> door hypothesis")

    # -- planning ---------------------------------------------------------
    def plan(self, goal_fn, keys=None, assume_open=None, start_pos=None):
        """Dijkstra over (pos, keys). Returns list of (dir, interact_first) or None.

        ``assume_open``: soft-blocked colours to treat as passable (optimistic plan)."""
        import heapq
        keys = frozenset(self.keys if keys is None else keys)
        assume_open = assume_open or set()
        start = (start_pos or self.pos, keys)
        dist = {start: 0}
        prev: dict = {}
        pq = [(0, 0, start)]
        cnt = 1
        while pq:
            d0, _, (p, ks) = heapq.heappop(pq)
            if dist.get((p, ks), 1e9) < d0:
                continue
            if goal_fn(p, ks) and p != start[0]:
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
                if dname not in self.action_dir.values():
                    cost += UNKNOWN_DIR_COST           # this direction still needs an action-discovery step
                assumed = False
                if c in self.toggleable and self.interact:
                    # a cell still showing a toggle colour is still solid (cleared cells turn EMPTY):
                    # always charge interact + move, never walk into it
                    inter = True
                    cost = 2
                elif not self.can_enter(c, ks):
                    if c in assume_open and not self.is_hard(c):
                        cost += UNKNOWN_PENALTY          # optimistic: assume this soft-blocked colour opens
                        assumed = True
                    else:
                        continue
                if not assumed and not inter and c not in self.passable and c not in self.consumable and c not in self.tele_colours:
                    if self.is_rare(c) and n not in self.visited and c not in self.bumped:
                        cost = RARE_STEP_COST          # curiosity: cheap to test a rare colour en route
                    else:
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

    def plan_to_exit(self, assume_open=None, start_pos=None):
        ex = self.exit
        if ex is None:
            return None
        return self.plan(lambda p, ks: p == ex, assume_open=assume_open, start_pos=start_pos)

    def soft_blocked(self):
        return {c for c in self.blocked if not self.is_hard(c) and c not in self.unlocks}

    def path_len(self, path):
        return None if path is None else sum(2 if inter else 1 for _, inter in path)

    def blocked_on_path(self, path):
        """(cell, colour, index) of the first soft-blocked cell an optimistic path enters."""
        p = self.pos
        for i, (d, _inter) in enumerate(path):
            v = DIRS[d]
            n = (p[0] + v[0], p[1] + v[1])
            c = self.colour(n)
            if c in self.soft_blocked():
                return n, c, i
            if c in self.tele_colours:
                twin = self.tele.get(n)
                p = twin if twin is not None else n
            else:
                p = n
        return None

    def colours_on_path(self, path):
        cols, p = set(), self.pos
        for d, _inter in path:
            v = DIRS[d]
            n = (p[0] + v[0], p[1] + v[1])
            c = self.colour(n)
            cols.add(c)
            twin = self.tele.get(n) if c in self.tele_colours else None
            p = twin if twin is not None else n
        return cols

    def special_tiles(self, exclude=()):
        out = []
        for y in range(self.size):
            for x in range(self.size):
                c = self.grid[y][x]
                if c in (EMPTY, EXIT) or (x, y) in self.visited or (x, y) == self.pos:
                    continue
                if self.is_hard(c) or c in self.toggleable or c in self.tele_colours or c in exclude:
                    continue
                if c in self.bumped and c not in self.consumable:
                    continue
                out.append(((x, y), c))
        return out

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
            if st.get("blocked", 0) >= 3 and st.get("passable", 0) == 0 and st.get("special", 0) == 0:
                out.add(int(c))
        return out

    def memory_says_inert(self, c: int) -> bool:
        st = self.ctx.memory["colour_stats"].get(str(c), {})
        return st.get("inert", 0) >= 2 and st.get("special", 0) == 0

    def record_stats(self, w: World):
        st0 = self.ctx.memory["colour_stats"]
        for c, tried in w.interact_tried.items():
            if len(tried) >= 2 and c not in w.toggleable:
                st0.setdefault(str(c), {"blocked": 0, "passable": 0})
                st0[str(c)]["inert"] = st0[str(c)].get("inert", 0) + 1
        st = self.ctx.memory["colour_stats"]
        for c in w.passable - {EMPTY, EXIT}:
            st.setdefault(str(c), {"blocked": 0, "passable": 0})["passable"] += 1
        for c, n in w.blocked.items():
            if c not in w.passable and c in w.bumped:
                st.setdefault(str(c), {"blocked": 0, "passable": 0})["blocked"] += 1
        for c in w.consumable | w.toggleable | w.tele_colours | set(w.unlocks):
            st.setdefault(str(c), {"blocked": 0, "passable": 0})
            st[str(c)]["special"] = st[str(c)].get("special", 0) + 1

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
        belief = w.plan_to_exit()
        soft = w.soft_blocked()
        opt = w.plan_to_exit(assume_open=soft) if soft else belief
        blen, olen = w.path_len(belief), w.path_len(opt)
        detour = (blen - olen) if (blen is not None and olen is not None) else (10 ** 6 if olen is not None else 0)
        # -- value of information: is an experiment cheaper than the detour? -----
        if opt is not None and (belief is None or detour > 0):
            hit = w.blocked_on_path(opt)
            # cheapest moment for an interact test is while already adjacent to such a tile
            route_cols = w.colours_on_path(opt) & soft
            for v in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (w.pos[0] + v[0], w.pos[1] + v[1])
                cn = w.colour(n)
                if cn in route_cols and (hit is None or cn != hit[1]):
                    hit = (n, cn, 0)
                    break
            if hit is not None:
                cell, c, idx = hit
                tried = w.interact_tried.setdefault(c, set())
                cand = [w.interact] if w.interact else INTERACT_CANDIDATES
                todo = [a for a in cand if a not in tried]
                exp_cost = idx + len(todo)                      # walk up to it + interact tries
                if not w.is_rare(c):
                    exp_cost += FREQUENT_INTERACT_PENALTY       # frequent colour: probably a plain wall
                if self.memory_says_inert(c):
                    exp_cost += 2 * FREQUENT_INTERACT_PENALTY   # earlier seeds: interact never touched it
                adjacent = max(abs(cell[0] - w.pos[0]), abs(cell[1] - w.pos[1])) <= 1
                if adjacent:
                    exp_cost = len(todo) + (0 if w.is_rare(c) else FREQUENT_INTERACT_PENALTY) \
                        + (2 * FREQUENT_INTERACT_PENALTY if self.memory_says_inert(c) else 0)
                if todo and (belief is None or detour >= exp_cost):   # tie -> experiment (pays off again in level 2)
                    if idx == 0 or adjacent:
                        tried.add(todo[0])
                        self.ctx.log(type="harness", experiment="interact", colour=c, action=todo[0], detour=detour)
                        self.act(w, todo[0])
                        return True
                    d, inter = opt[0]
                    if inter and w.interact:
                        self.act(w, w.interact)
                    else:
                        self.go(w, d)
                    return True
                # interact refuted for this colour -> key hypothesis: visit a special tile
                best = None
                for tile, tc in w.special_tiles(exclude={c}):
                    p1 = w.plan(lambda p, ks, t=tile: p == t)
                    if p1 is None:
                        continue
                    p2 = w.plan_to_exit(assume_open=soft, start_pos=tile)
                    if p2 is None:
                        continue
                    cost = w.path_len(p1) + w.path_len(p2) + UNKNOWN_PENALTY
                    if best is None or cost < best[0]:
                        best = (cost, tile, tc, p1)
                if best is not None and (belief is None or best[0] < blen):
                    cost, tile, tc, p1 = best
                    self.ctx.log(type="harness", experiment="key", tile=list(tile), colour=tc, cost=cost, belief=blen)
                    d, inter = p1[0]
                    if inter and w.interact:
                        self.act(w, w.interact)
                    else:
                        self.go(w, d)
                    return True
        if belief:
            d, inter = belief[0]
            if inter and w.interact:
                self.act(w, w.interact)
                return True
            self.go(w, d)
            return True
        # stuck ladder (as in 0001): interact next to a blocking colour, then unvisited specials
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
            if w.no_move_streak >= 4:
                # nothing we try moves us: forget soft-blocked beliefs once, then random known move
                w.no_move_streak = 0
                w.events.append("no-move streak: resetting soft-blocked colours")
                w.soft_clear_blocked("no-move streak")
                w.failed_at.clear()
                known = [a for a in MOVE_CANDIDATES if a in w.action_dir] or MOVE_CANDIDATES
                self.act(w, random.choice(known))
                continue
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
