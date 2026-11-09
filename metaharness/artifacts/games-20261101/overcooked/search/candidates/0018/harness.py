"""Candidate 0018: phase-locked complementary roles.

The controlled chef is not a general greedy worker.  It is assigned one side of
an observed production contract (supplier or runner), and the contract changes
only at a delivery boundary or a verified partner failure.  Exact state
interpretation and primitive execution stay in code; Jev recognizes whether a
partner handoff is real, while Astra writes a coarse opening contract.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


STEP = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
STEP_NAME = {v: k for k, v in STEP.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}


def point(v):
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            if isinstance(v[0], bool) or isinstance(v[1], bool):
                return None
            return int(v[0]), int(v[1])
        except Exception:
            return None
    return None


def json_value(v):
    if isinstance(v, dict):
        return {str(k): json_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_value(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def state(obs):
    return obs.get("state", {}) if isinstance(obs, dict) and isinstance(obs.get("state"), dict) else {}


def terrain(obs):
    t = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return t if isinstance(t, list) else []


def tick(obs):
    try:
        return int(obs.get("tick", state(obs).get("timestep", 0)))
    except Exception:
        return 0


def tile(t, p):
    p = point(p)
    if p is None:
        return "X"
    x, y = p
    if y < 0 or y >= len(t):
        return "X"
    try:
        row = t[y]
        if isinstance(row, (str, list, tuple)) and 0 <= x < len(row):
            return row[x]
    except Exception:
        pass
    return "X"


def open_cell(t, p):
    return tile(t, p) not in FEATURES and tile(t, p) != "X"


def cells(t, symbol):
    out = []
    for y, row in enumerate(t):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                out.append((x, y))
    return out


def around(p):
    x, y = p
    for d in STEP.values():
        yield (x + d[0], y + d[1]), d


def walk(t, start, goal, blocked=()):
    """Shortest static floor path, with actors treated as transient walls."""
    start, goal = point(start), point(goal)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    blocked = set(point(x) for x in blocked if point(x) is not None)
    blocked.discard(start)
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        for nxt, _ in around(cur):
            if nxt in prev or nxt in blocked or not open_cell(t, nxt):
                continue
            prev[nxt] = cur
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            q.append(nxt)
    return None


def poses(t, target):
    target = point(target)
    if target is None:
        return []
    result = []
    for d in STEP.values():
        p = (target[0] - d[0], target[1] - d[1])
        if open_cell(t, p) and p not in result:
            result.append(p)
    return result


def route(t, start, target, blocked=()):
    """Return the shortest route to an interaction square of target."""
    start, target = point(start), point(target)
    if start is None or target is None:
        return None, None
    blocked = set(blocked)
    choices = []
    for pose in poses(t, target):
        b = set(blocked)
        b.discard(pose)
        path = walk(t, start, pose, b)
        if path is not None:
            choices.append((len(path), pose, path))
    if not choices:
        # A partner may currently occupy the only approach.  Keep the job
        # rather than declaring it impossible; the motor will yield/retry.
        for pose in poses(t, target):
            path = walk(t, start, pose)
            if path is not None:
                choices.append((len(path), pose, path))
    if not choices:
        return None, None
    choices.sort(key=lambda x: (x[0], x[1]))
    return choices[0][1], choices[0][2]


def all_objects(obs):
    raw = state(obs).get("objects", [])
    return raw if isinstance(raw, list) else []


def object_name(obj):
    return str(obj.get("name", "") or "").lower() if isinstance(obj, dict) else ""


def object_at(obj):
    return point(obj.get("position")) if isinstance(obj, dict) else None


def actors(obs):
    players = state(obs).get("players", [])
    if not isinstance(players, list):
        players = []
    try:
        mine = int(obs.get("controlled_agent_index", 0))
    except Exception:
        mine = 0
    if mine not in (0, 1) or mine >= len(players):
        mine = 0
    other = 1 - mine if len(players) > 1 else mine
    me = players[mine] if mine < len(players) and isinstance(players[mine], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, partner, mine, other


def held(player):
    obj = player.get("held_object") if isinstance(player, dict) else None
    return str(obj.get("name", "") or "").lower() if isinstance(obj, dict) else ""


def ingredients(obj):
    if not isinstance(obj, dict):
        return []
    raw = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(raw, list):
        return []
    out = []
    for value in raw:
        if isinstance(value, dict):
            value = value.get("name", "")
        value = str(value or "").lower()
        if value in INGREDIENTS:
            out.append(value)
    return out


def recipes(obs):
    raw = state(obs).get("all_orders", [])
    result = []
    if isinstance(raw, list):
        for order in raw:
            values = order.get("ingredients") if isinstance(order, dict) else order
            if not isinstance(values, list):
                continue
            r = []
            for value in values:
                if isinstance(value, dict):
                    value = value.get("name", "")
                value = str(value or "").lower()
                if value in INGREDIENTS:
                    r.append(value)
            if r:
                result.append(r)
    return result or [["onion", "onion", "onion"]]


def recipe_gap(have, available):
    """Choose a legal active recipe and return its ordered missing multiset."""
    have_counts = Counter(have)
    chosen, gap = None, None
    for r in available:
        wanted = Counter(r)
        if any(have_counts[k] > wanted[k] for k in have_counts):
            continue
        left = Counter(have)
        missing = []
        for item in r:
            if left[item] < wanted[item]:
                missing.append(item)
                left[item] += 1
        if gap is None or len(missing) < len(gap):
            chosen, gap = r, missing
    return chosen or [], gap if gap is not None else []


def pot_states(obs):
    soups = {}
    for obj in all_objects(obs):
        p = object_at(obj)
        if p is not None and object_name(obj) == "soup":
            soups[p] = obj
    result = []
    active = recipes(obs)
    for p in cells(terrain(obs), "P"):
        soup = soups.get(p)
        have = ingredients(soup)
        r, missing = recipe_gap(have, active)
        ready = bool(soup and soup.get("is_ready"))
        raw_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        raw_time = soup.get("cook_time", -1) if soup else -1
        cooking = bool(soup and soup.get("is_cooking")) and not ready
        if not ready and not cooking and isinstance(raw_tick, (int, float)) and isinstance(raw_time, (int, float)):
            cooking = raw_tick >= 0 and raw_time >= 0 and not bool(soup and soup.get("is_idle"))
        full = bool(soup and have and not missing)
        if ready:
            phase = "ready"
        elif cooking:
            phase = "cooking"
        elif full:
            phase = "full_idle"
        elif have:
            phase = "partial"
        else:
            phase = "empty"
        remaining = None
        if cooking and isinstance(raw_tick, (int, float)) and isinstance(raw_time, (int, float)):
            remaining = max(0, int(raw_time - raw_tick))
        result.append({
            "pos": p, "phase": phase, "ingredients": have, "recipe": r,
            "missing": missing, "ready": ready, "cooking": cooking,
            "remaining": remaining,
        })
    return result


def occupied(obs):
    return {object_at(x) for x in all_objects(obs) if object_at(x) is not None}


def counters(obs):
    t = terrain(obs)
    used = occupied(obs)
    result = []
    for y, row in enumerate(t):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            p = (x, y)
            if value == "X" and p not in used and any(open_cell(t, q) for q, _ in around(p)):
                result.append(p)
    return result


def counter_objects(obs):
    t = terrain(obs)
    result = []
    for obj in all_objects(obs):
        p = object_at(obj)
        name = object_name(obj)
        if p is not None and name in INGREDIENTS and tile(t, p) == "X":
            result.append((name, p))
    return result


def partner_next(obs):
    _, partner, _, _ = actors(obs)
    p = point(partner.get("position"))
    action = obs.get("partner_action_last", "") if isinstance(obs, dict) else ""
    d = STEP.get(action, (0, 0))
    return (p[0] + d[0], p[1] + d[1]) if p is not None else None


def interaction_cells(obs):
    t = terrain(obs)
    out = set()
    for symbol in FEATURES:
        for p in cells(t, symbol):
            out.update(poses(t, p))
    return out


class RolePipeline:
    """A role contract plus a receding-horizon executor.

    Unlike a task-voting controller, this planner asks the models about the
    contract only.  Every primitive is selected by the current observation and
    checked again immediately before posting.
    """

    def __init__(self, ctx, game):
        self.ctx, self.game = ctx, game
        spec = getattr(game, "action_spec", None)
        self.legal = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.legal:
            self.noop = "stay" if "stay" in self.legal else (next(iter(self.legal), "stay"))

        self.role = None
        self.role_source = ""
        self.role_locked = False
        self.first_delivery = False
        self.dish_mode = "early"
        self.handoff_hint = None
        self.handoff_until = -1
        # A delivery-side chef may make one bounded refill contribution after
        # each team delivery, then must return to the delivery lane.
        self.assist_used = False

        self.task = None
        self.task_since = -1
        self.task_epoch = 0
        self.blocked = 0
        self.last_pos = None
        self.last_partner_pos = None
        self.last_action = self.noop
        self.previous_pos = None
        self.escape = 0
        self.escape_goal = None

        self.partner_still = 0
        self.partner_bad = 0
        self.partner_distance = None
        self.partner_goal = None

        self.jev_future = None
        self.jev_epoch = -1
        self.last_jev = -999
        self.astra_future = None
        self.astra_started = False
        self.astra_role = None
        self.astra_mode = None
        self.system1_off = getattr(ctx, "system1", object()) is None

    def log(self, name, **fields):
        try:
            self.ctx.log(name, **fields)
        except Exception:
            pass

    def act_name(self, value):
        if value in self.legal:
            return value
        if self.noop in self.legal:
            return self.noop
        return next(iter(self.legal), "stay")

    def initial_role(self, obs):
        """Use geometry only as a prior; the first sparse delivery is evidence."""
        if self.role is not None:
            return
        pots = pot_states(obs)
        me, partner, _, _ = actors(obs)
        mp, pp = point(me.get("position")), point(partner.get("position"))
        # The two-pot development layouts consistently benefit from the
        # controlled side being the ingredient/pot specialist.  A one-pot room
        # is decided by a route sum so reversed-agent starts do not hardcode an
        # index.
        if len(pots) > 1:
            role = "supplier"
        else:
            def runner_cost(start):
                if start is None:
                    return 999
                d1 = min((len(route(terrain(obs), start, d)[1]) - 1
                          for d in cells(terrain(obs), "D") if route(terrain(obs), start, d)[1] is not None), default=999)
                d2 = min((len(route(terrain(obs), start, s)[1]) - 1
                          for s in cells(terrain(obs), "S") if route(terrain(obs), start, s)[1] is not None), default=999)
                d3 = min((len(route(terrain(obs), start, p)[1]) - 1
                          for p in cells(terrain(obs), "P") if route(terrain(obs), start, p)[1] is not None), default=999)
                return d1 + d2 + d3
            role = "runner" if runner_cost(mp) <= runner_cost(pp) else "supplier"
        self.role = role
        self.role_source = "geometry_prior"
        self.log("role_prior", tick=tick(obs), role=role)

    def observe_reward(self, obs):
        me, partner, mine, other = actors(obs)
        rewards = state(obs).get("sparse_r_by_agent")
        if not isinstance(rewards, list):
            rewards = obs.get("sparse_r_by_agent", []) if isinstance(obs, dict) else []
        try:
            mine_r = float(rewards[mine])
            other_r = float(rewards[other])
        except Exception:
            mine_r = other_r = 0.0
        # The adaptor exposes the transition reward at the following observe.
        if mine_r > 0 or other_r > 0:
            if self.role_locked:
                # Team delivery boundaries are the only refill-assist reset.
                # A later partner delivery does not swap the role, but it does
                # open one more bounded assist window for the runner.
                if self.role == "runner":
                    self.assist_used = False
                return
            # Either chef may deliver later in a healthy pipeline.  The first
            # credit identifies the opening complementary role; subsequent
            # team credits are progress, not permission to swap roles.
            new_role = "runner" if mine_r >= other_r and mine_r > 0 else "supplier"
            self.role = new_role
            self.role_source = "delivery_credit"
            self.role_locked = True
            self.first_delivery = True
            self.assist_used = False
            self.task = None
            self.log("role_confirmed", tick=tick(obs), role=new_role,
                     mine_reward=mine_r, partner_reward=other_r)

    def partner_target(self, obs):
        _, p, _, _ = actors(obs)
        ph = held(p)
        ps = point(p.get("position"))
        pots = pot_states(obs)
        if ph == "soup":
            return ("serve", cells(terrain(obs), "S"))
        if ph == "dish":
            ready = [x["pos"] for x in pots if x["ready"]]
            if ready:
                return ("ready", ready)
            return ("cook", [x["pos"] for x in pots if x["cooking"]])
        if ph in INGREDIENTS:
            return ("pot", [x["pos"] for x in pots if x["phase"] in {"empty", "partial", "full_idle"}])
        return ("none", [])

    def update_partner(self, obs):
        _, p, _, _ = actors(obs)
        pp = point(p.get("position"))
        if pp is not None and pp == self.last_partner_pos:
            self.partner_still += 1
        else:
            self.partner_still = 0
        kind, goals = self.partner_target(obs)
        key = (kind, tuple(goals))
        if key != self.partner_goal:
            self.partner_goal = key
            self.partner_bad = 0
            self.partner_distance = None
        if pp is not None and goals:
            distances = []
            for target in goals:
                for pose in poses(terrain(obs), target):
                    path = walk(terrain(obs), pp, pose)
                    if path is not None:
                        distances.append(len(path) - 1)
            distance = min(distances, default=999)
            if self.partner_distance is not None:
                if distance < self.partner_distance:
                    self.partner_bad = 0
                elif self.partner_still >= 2 or distance > self.partner_distance + 1:
                    self.partner_bad += 1
            self.partner_distance = distance
        else:
            self.partner_bad = 0
            self.partner_distance = None
        self.last_partner_pos = pp

    def partner_facing(self, obs, target):
        _, p, _, _ = actors(obs)
        pp, facing, target = point(p.get("position")), point(p.get("orientation")), point(target)
        if pp is None or facing is None or target is None:
            return False
        return pp in poses(terrain(obs), target) and facing == (target[0] - pp[0], target[1] - pp[1])

    def partner_owns_handoff(self, obs, pot):
        _, p, _, _ = actors(obs)
        ph = held(p)
        if ph == "soup":
            return True
        if ph != "dish" or not pot:
            return False
        if self.partner_facing(obs, pot["pos"]):
            return True
        remaining = pot.get("remaining")
        urgent = pot.get("ready") or (isinstance(remaining, (int, float)) and remaining <= 7)
        if not urgent:
            return self.partner_bad < 6
        return self.partner_bad < 5 and self.partner_still < 8

    def safe_floor(self, obs, start, partner, near=None):
        t = terrain(obs)
        gates = interaction_cells(obs)
        if near is not None:
            gates.update(poses(t, near))
        candidates = []
        if start is None:
            return None
        for y, row in enumerate(t):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                q = (x, y)
                if not open_cell(t, q) or q in gates or q == partner:
                    continue
                path = walk(t, start, q, {partner} if partner is not None else set())
                if path is None:
                    continue
                gap = abs(q[0] - partner[0]) + abs(q[1] - partner[1]) if partner else 0
                near_cost = abs(q[0] - near[0]) + abs(q[1] - near[1]) if near else 0
                candidates.append((near_cost, -gap, len(path), q))
        if not candidates:
            return start if start not in gates and start != partner else None
        candidates.sort()
        return candidates[0][-1]

    def make_task(self, kind, target=None, item="", tag="", **extra):
        target = point(target)
        return {"kind": kind, "target": target, "item": item, "tag": tag,
                "id": "%s:%s:%s:%s" % (kind, item, target, tag), **extra}

    def nearest(self, obs, start, targets, partner=None):
        choices = []
        for target in targets:
            pose, path = route(terrain(obs), start, target, {partner} if partner is not None else set())
            if path is not None:
                choices.append((len(path), target, pose))
        if not choices:
            return None, None
        choices.sort(key=lambda z: (z[0], z[1]))
        return choices[0][1], choices[0][2]

    def cooking_route_cost(self, obs, start, pot, partner):
        best = 999
        t = terrain(obs)
        for dish in cells(t, "D"):
            _, first = route(t, start, dish, {partner} if partner is not None else set())
            if first is None:
                continue
            for pp in poses(t, pot):
                second = walk(t, first[-1], pp, {partner} if partner is not None else set())
                if second is not None:
                    best = min(best, (len(first) - 1) + 1 + (len(second) - 1) + 1)
        return best

    def should_fetch_dish(self, obs, pot, current, partner):
        if not pot or not pot.get("cooking"):
            return False
        if not cells(terrain(obs), "D"):
            return False
        # The runner owns the dish token as soon as cooking begins.  This is
        # intentionally earlier than the parent's just-in-time threshold: it
        # removes the runner's post-delivery ingredient detour without claiming
        # the pot approach while the supplier is still filling it.
        if self.role == "runner" and self.dish_mode == "early":
            return True
        remaining = pot.get("remaining")
        if not isinstance(remaining, (int, float)):
            return True
        return remaining <= max(5, self.cooking_route_cost(obs, current, pot["pos"], partner) + 2)

    def plan(self, obs):
        self.initial_role(obs)
        me, partner, _, _ = actors(obs)
        current, partner_pos = point(me.get("position")), point(partner.get("position"))
        myheld, pheld = held(me), held(partner)
        pots = pot_states(obs)
        t = terrain(obs)

        # Object ownership is an invariant stronger than any role hint.
        if myheld == "soup":
            target, _ = self.nearest(obs, current, cells(t, "S"), partner_pos)
            return self.make_task("deliver", target, tag="carried") if target is not None else self.make_task("hold")
        if myheld == "dish":
            ready = [x for x in pots if x["ready"]]
            if ready:
                ready.sort(key=lambda x: (self._route_length(obs, current, x["pos"], partner_pos), x["pos"]))
                return self.make_task("pickup", ready[0]["pos"], tag="ready")
            cooking = [x for x in pots if x["cooking"]]
            if cooking:
                pot = min(cooking, key=lambda x: (x.get("remaining", 999) if isinstance(x.get("remaining"), (int, float)) else 999, x["pos"]))
                park = self.safe_floor(obs, current, partner_pos, pot["pos"])
                return self.make_task("wait_dish", park, tag=str(pot["pos"]))
            park = self.safe_floor(obs, current, partner_pos)
            return self.make_task("wait_dish", park, tag="no_pot")
        if myheld in INGREDIENTS:
            compatible = [x for x in pots if x["phase"] in {"empty", "partial", "full_idle"} and myheld in x["missing"]]
            compatible.sort(key=lambda x: (self._route_length(obs, current, x["pos"], partner_pos), x["pos"]))
            if compatible:
                return self.make_task("fill", compatible[0]["pos"], item=myheld)
            free = counters(obs)
            target, _ = self.nearest(obs, current, free, partner_pos)
            return self.make_task("put_counter", target, item=myheld) if target is not None else self.make_task("hold")

        ready = [x for x in pots if x["ready"]]
        if ready:
            pot = min(ready, key=lambda x: (self._route_length(obs, current, x["pos"], partner_pos), x["pos"]))
            cover = self.partner_owns_handoff(obs, pot)
            hinted_yield = self.handoff_hint == "yield" and tick(obs) <= self.handoff_until
            if not cover and not hinted_yield:
                target, _ = self.nearest(obs, current, cells(t, "D"), partner_pos)
                if target is not None:
                    return self.make_task("fetch_dish", target, tag="backup" if pheld == "dish" else "ready")
            # A supplier should continue an independent build while the
            # partner owns the ready-soup transaction.  A runner keeps its
            # current delivery-side cell; moving to the geometrically nearest
            # floor cell can claim the one-cell refill lane.
            if self.role == "supplier":
                build = self._build_task(obs, current, partner_pos, pots)
                if build is not None:
                    return build
            park = current if self.role == "runner" else self.safe_floor(obs, current, partner_pos, pot["pos"])
            return self.make_task("yield_ready", park, tag=str(pot["pos"]))

        # Explicit start is a phase boundary and outranks all supply work.
        idle = [x for x in pots if x["phase"] == "full_idle"]
        if idle:
            idle.sort(key=lambda x: (self._route_length(obs, current, x["pos"], partner_pos), x["pos"]))
            return self.make_task("start", idle[0]["pos"])

        if self.role == "supplier":
            built = self._build_task(obs, current, partner_pos, pots)
            if built is not None:
                return built

        cooking = [x for x in pots if x["cooking"]]
        if cooking:
            cooking.sort(key=lambda x: (x.get("remaining", 999) if isinstance(x.get("remaining"), (int, float)) else 999, x["pos"]))
            pot = cooking[0]
            if self.role == "runner" and not self.partner_owns_handoff(obs, pot) and self.should_fetch_dish(obs, pot, current, partner_pos):
                target, _ = self.nearest(obs, current, cells(t, "D"), partner_pos)
                if target is not None:
                    return self.make_task("fetch_dish", target, tag="timed")
            if self.role == "supplier":
                built = self._build_task(obs, current, partner_pos, pots)
                if built is not None:
                    return built
            park = current if self.role == "runner" else self.safe_floor(obs, current, partner_pos, pot["pos"])
            return self.make_task("park", park, tag=str(pot["pos"]))

        # Before the first delivery the opening chef may help build the first
        # pot.  Once a runner has earned delivery credit, ingredient work is a
        # deliberate supplier responsibility; do not silently drift back into
        # the old greedy alternation.
        if self.role == "supplier" or not self.role_locked or (self.role == "runner" and not self.assist_used):
            built = self._build_task(obs, current, partner_pos, pots)
            if built is not None:
                return built
        # A confirmed runner waits where the previous transaction left it.
        # This is intentional: it gives the supplier the central aisle instead
        # of making the runner's wait task a second traffic plan.
        park = current if self.role == "runner" else self.safe_floor(obs, current, partner_pos)
        return self.make_task("park", park, tag="open")

    def _route_length(self, obs, start, target, partner=None):
        _, path = route(terrain(obs), start, target, {partner} if partner is not None else set())
        return len(path) if path is not None else 999

    def _build_task(self, obs, current, partner, pots):
        buildable = [x for x in pots if x["phase"] in {"empty", "partial"} and x["missing"]]
        if not buildable:
            return None
        buildable.sort(key=lambda x: (len(x["missing"]), self._route_length(obs, current, x["pos"], partner), x["pos"]))
        pot = buildable[0]
        wanted = pot["missing"][0]
        staged = [(name, p) for name, p in counter_objects(obs) if name == wanted]
        if staged:
            staged.sort(key=lambda z: self._route_length(obs, current, z[1], partner))
            return self.make_task("take_counter", staged[0][1], item=wanted)
        symbol = "O" if wanted == "onion" else "T"
        target, _ = self.nearest(obs, current, cells(terrain(obs), symbol), partner)
        if target is not None:
            return self.make_task("fetch_ingredient", target, item=wanted)
        return None

    def task_same(self, a, b):
        if not a or not b:
            return False
        return (a.get("kind"), point(a.get("target")), a.get("item"), a.get("tag")) == \
               (b.get("kind"), point(b.get("target")), b.get("item"), b.get("tag"))

    def task_active(self, obs):
        """Keep an in-flight transaction; only phase facts may cancel it."""
        if not self.task:
            return False
        kind = self.task.get("kind")
        if kind in {"park", "wait_dish", "yield_ready", "hold"}:
            # Waiting is deliberately receding-horizon: ready/cooking and the
            # one-assist budget must be reconsidered every observation.
            return False
        me, _, _, _ = actors(obs)
        item, target = held(me), point(self.task.get("target"))
        t = terrain(obs)
        if target is None:
            return False
        if kind == "fetch_dish":
            return not item and tile(t, target) == "D"
        if kind == "fetch_ingredient":
            return not item and tile(t, target) in {"O", "T"}
        if kind == "take_counter":
            return not item and any(name == self.task.get("item") and p == target for name, p in counter_objects(obs))
        if kind == "put_counter":
            return item in INGREDIENTS and target not in occupied(obs)
        if kind == "fill":
            return bool(item in INGREDIENTS and any(x["pos"] == target and
                        x["phase"] in {"empty", "partial", "full_idle"} and item in x["missing"]
                        for x in pot_states(obs)))
        if kind == "start":
            return not item and any(x["pos"] == target and x["phase"] == "full_idle" for x in pot_states(obs))
        if kind == "pickup":
            return item == "dish" and any(x["pos"] == target and x["ready"] for x in pot_states(obs))
        if kind == "deliver":
            return item == "soup" and tile(t, target) == "S"
        return False

    def choose_task(self, obs):
        if self.task_active(obs):
            return
        wanted = self.plan(obs)
        if self.task_same(self.task, wanted):
            return
        if (self.role == "runner" and self.role_locked and
                wanted.get("kind") in {"fetch_ingredient", "take_counter"}):
            # Reserving one ingredient is the whole assist transaction; do not
            # allow the next planning pass to fetch a second one.
            self.assist_used = True
        self.task = wanted
        self.task_since = tick(obs)
        self.task_epoch += 1
        self.blocked = 0
        self.escape = 0
        self.log("role_task", tick=self.task_since, role=self.role, kind=wanted.get("kind"),
                 target=wanted.get("target"), epoch=self.task_epoch)

    def legal_interaction(self, obs, job):
        me, _, _, _ = actors(obs)
        item = held(me)
        kind, target = job.get("kind"), point(job.get("target"))
        t = terrain(obs)
        pots = pot_states(obs)
        pot = next((x for x in pots if x["pos"] == target), None)
        if target is None:
            return False
        if kind == "deliver":
            return item == "soup" and tile(t, target) == "S"
        if kind == "fetch_dish":
            return not item and tile(t, target) == "D"
        if kind == "fetch_ingredient":
            return not item and tile(t, target) in {"O", "T"}
        if kind == "take_counter":
            return not item and any(name == job.get("item") and p == target for name, p in counter_objects(obs))
        if kind == "put_counter":
            return item in INGREDIENTS and target not in occupied(obs)
        if kind == "fill":
            return bool(pot and item in INGREDIENTS and pot["phase"] in {"empty", "partial", "full_idle"} and item in pot["missing"])
        if kind == "start":
            return bool(pot and not item and pot["phase"] == "full_idle")
        if kind == "pickup":
            return bool(pot and item == "dish" and pot["ready"])
        return False

    def escape_step(self, obs, goal):
        me, partner, _, _ = actors(obs)
        cur, other = point(me.get("position")), point(partner.get("position"))
        if cur is None:
            return self.act_name(self.noop)
        predicted = partner_next(obs)
        choices = []
        for nxt, delta in around(cur):
            if not open_cell(terrain(obs), nxt) or nxt in {other, predicted}:
                continue
            if nxt in interaction_cells(obs) and not held(me):
                continue
            goal_gap = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) if goal else 0
            partner_gap = abs(nxt[0] - other[0]) + abs(nxt[1] - other[1]) if other else 0
            reverse = 1 if nxt == self.previous_pos else 0
            choices.append((goal_gap, reverse, -partner_gap, nxt, delta))
        if not choices:
            return self.act_name(self.noop)
        choices.sort(key=lambda z: z[:4])
        return self.act_name(STEP_NAME.get(choices[0][4], self.noop))

    def blocked_update(self, obs):
        me, _, _, _ = actors(obs)
        cur = point(me.get("position"))
        if self.last_pos is not None and cur == self.last_pos and self.last_action in STEP:
            self.blocked += 1
        elif cur is not None and cur != self.last_pos:
            self.blocked = max(0, self.blocked - 1)
        self.previous_pos = self.last_pos
        self.last_pos = cur

    def motor(self, obs):
        me, partner, _, _ = actors(obs)
        cur, other = point(me.get("position")), point(partner.get("position"))
        if self.task is None:
            return self.act_name(self.noop)
        if self.blocked >= 6 and self.escape <= 0 and held(me) != "soup":
            self.escape = 3
            self.escape_goal = point(self.task.get("target"))
            self.log("route_escape", tick=tick(obs), role=self.role, blocked=self.blocked)
        if self.escape > 0 and held(me) != "soup":
            self.escape -= 1
            return self.escape_step(obs, self.escape_goal)

        kind, target = self.task.get("kind"), point(self.task.get("target"))
        if kind in {"hold", "park", "wait_dish", "yield_ready"}:
            if target is None or cur is None:
                return self.act_name(self.noop)
            path = walk(terrain(obs), cur, target, {other} if other is not None and other != target else set())
            if path is not None and len(path) > 1:
                nxt = path[1]
                predicted = partner_next(obs)
                if nxt == other or nxt == predicted:
                    self.blocked += 1
                    return self.act_name(self.noop if self.blocked <= 2 else self.escape_step(obs, target))
                return self.act_name(STEP_NAME.get((nxt[0] - cur[0], nxt[1] - cur[1]), self.noop))
            return self.act_name(self.noop)
        if target is None or cur is None:
            return self.act_name(self.noop)
        pose, path = route(terrain(obs), cur, target, {other} if other is not None and other != target else set())
        if path is None or pose is None:
            self.blocked += 1
            return self.escape_step(obs, target)
        if len(path) > 1:
            nxt = path[1]
            predicted = partner_next(obs)
            if nxt == other or nxt == predicted or (predicted == cur and nxt == other):
                self.blocked += 1
                return self.act_name(self.noop if self.blocked <= 2 else self.escape_step(obs, pose))
            self.blocked = max(0, self.blocked - 1)
            return self.act_name(STEP_NAME.get((nxt[0] - cur[0], nxt[1] - cur[1]), self.noop))
        desired = (target[0] - cur[0], target[1] - cur[1])
        facing = point(me.get("orientation"))
        if desired in STEP_NAME and facing != desired:
            return self.act_name(STEP_NAME[desired])
        if self.legal_interaction(obs, self.task):
            return self.act_name("interact")
        return self.act_name(self.noop)

    def compact(self, obs):
        me, partner, _, _ = actors(obs)
        pots = pot_states(obs)
        return json_value({
            "tick": tick(obs), "layout": obs.get("layout_name", ""),
            "role": {"code_prior": self.role, "source": self.role_source, "locked": self.role_locked},
            "self": {"position": point(me.get("position")), "held": held(me), "facing": point(me.get("orientation"))},
            "partner": {"position": point(partner.get("position")), "held": held(partner),
                        "last_action": obs.get("partner_action_last", ""),
                        "still": self.partner_still, "route_not_improving": self.partner_bad},
            "pots": [{"position": x["pos"], "phase": x["phase"], "missing": x["missing"],
                      "remaining": x["remaining"]} for x in pots],
            "orders": recipes(obs),
            "current_task": {"kind": self.task.get("kind", "") if self.task else "",
                             "age": max(0, tick(obs) - self.task_since) if self.task else 0},
            "role_options": ["runner", "supplier", "keep_code_role"],
        })

    def launch_astra(self, obs):
        if self.astra_started or getattr(self.ctx, "system2", object()) is None:
            self.astra_started = True
            return
        self.astra_started = True
        prompt = (
            "You are Astra, a slow strategy architect for a realtime symbolic "
            "Overcooked-AI grid kitchen. The executor owns exact recipe arithmetic, "
            "BFS, interaction legality, collision recovery, and all primitive actions. "
            "Write a short opening role contract, not directions. A supplier feeds "
            "compatible pots and starts full idle pots; a runner owns dish pickup, "
            "ready-soup pickup, and serving. Roles may be revised only on a delivery "
            "boundary or a verified stalled handoff. Modern dynamics require an empty "
            "handed interaction to start a full pot. Return ONLY JSON with keys "
            "role (runner|supplier|keep), dish (early|timed), and trigger (credit|handoff).\n" +
            json.dumps(self.compact(obs), separators=(",", ":"))
        )

        def ask():
            if self.ctx.done:
                return ""
            try:
                result = self.ctx.think(prompt, timeout_s=90.0)
                return getattr(result, "text", "")
            except Exception:
                return ""
        try:
            self.astra_future = self.ctx.spawn(ask)
        except Exception:
            self.astra_future = None

    def poll_astra(self, obs):
        f = self.astra_future
        if f is None or not f.done():
            return
        self.astra_future = None
        try:
            raw = f.result()
            text = raw if isinstance(raw, str) else getattr(raw, "text", "")
            match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
            if not isinstance(parsed, dict):
                return
            if parsed.get("role") in {"runner", "supplier"}:
                self.astra_role = parsed["role"]
            if parsed.get("dish") in {"early", "timed"}:
                self.astra_mode = parsed["dish"]
            # Astra can alter the prior, never a role established by sparse
            # reward. This is a genuine division of labour, not raw steering.
            if not self.role_locked and tick(obs) < 100 and self.astra_role:
                self.role = self.astra_role
                self.role_source = "astra_contract"
            self.dish_mode = self.astra_mode or self.dish_mode
            self.log("astra_contract", tick=tick(obs), role=self.astra_role, dish=self.dish_mode)
        except Exception:
            return

    def launch_jev(self, obs, reason):
        if self.system1_off or self.jev_future is not None:
            return
        now = tick(obs)
        if now - self.last_jev < 24:
            return
        self.last_jev = now
        self.jev_epoch = self.task_epoch
        state_in = {"game": "Overcooked-AI symbolic kitchen", "observation": self.compact(obs)}
        questions = {
            "role": Choice(
                "Classify the controlled chef's medium-level role from the derived facts. "
                "Choose runner only when the partner is visibly doing supply work or the "
                "controlled chef has delivery credit; choose supplier when the partner owns "
                "dish/soup delivery or the controlled chef owns ingredient work. Never give "
                "directions and never invent a role.",
                {
                    "runner": "Own dish pickup, ready-soup pickup, and serving; do not fetch speculative ingredients.",
                    "supplier": "Own compatible ingredient delivery and explicit pot starts; yield the delivery lane.",
                    "keep_code_role": "The code role is supported or the evidence is insufficient to switch.",
                },
            ),
            "handoff": Noul(
                "Is the partner's current dish or soup handoff genuinely active? Use only "
                "partner.held, partner.still, partner.route_not_improving, and the pot phases. "
                "A partner facing the pot or improving its route is active; a stalled partner "
                "with a ready/nearly-ready pot is not active.",
                {
                    "true": "The partner is progressing toward or already facing the urgent pot/serve target.",
                    "false": "The partner handoff is absent or visibly stalled at an urgent pot.",
                },
            ),
        }

        def ask():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(state_in, questions, timeout_s=1.0)
            except Exception:
                return {}
        try:
            self.jev_future = self.ctx.spawn(ask)
            self.log("jev_contract_request", tick=now, reason=reason, epoch=self.task_epoch)
        except Exception:
            self.jev_future = None

    def poll_jev(self, obs):
        f = self.jev_future
        if f is None or not f.done():
            return
        self.jev_future = None
        try:
            answers = f.result()
        except Exception:
            return
        if not isinstance(answers, dict) or self.jev_epoch != self.task_epoch:
            return
        role_answer = answers.get("role")
        value = getattr(role_answer, "value", None)
        probs = getattr(role_answer, "probabilities", {}) or {}
        try:
            confidence = float(probs.get(value, 0.0)) if isinstance(probs, dict) else 0.0
        except Exception:
            confidence = 0.0
        # Jev can revise only an unconfirmed prior and only with a closed-world
        # answer. Exact delivery credit always wins.
        if not self.role_locked and value in {"runner", "supplier"} and confidence >= 0.68:
            self.role = value
            self.role_source = "jev_role"
            self.log("jev_role", tick=tick(obs), role=value, probability=confidence)
        hand = answers.get("handoff")
        try:
            p = float(getattr(hand, "value", 0.0))
        except Exception:
            p = 0.0
        if p >= 0.75:
            self.handoff_hint, self.handoff_until = "yield", tick(obs) + 10
            self.log("jev_handoff", tick=tick(obs), value="yield", probability=p)
        elif p <= 0.25:
            self.handoff_hint, self.handoff_until = "rescue", tick(obs) + 8
            self.log("jev_handoff", tick=tick(obs), value="rescue", probability=p)

    def run(self):
        while not self.ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self.observe_reward(obs)
            # Completing a fill interaction consumes the runner's one assist
            # slot.  This is checked from the new observation, so an ineffective
            # interaction cannot spend the slot.
            me_seen, _, _, _ = actors(obs)
            if self.role == "runner" and self.task and self.task.get("kind") == "fill" and held(me_seen) not in INGREDIENTS:
                self.assist_used = True
            self.update_partner(obs)
            self.blocked_update(obs)
            self.poll_astra(obs)
            self.poll_jev(obs)
            if not self.astra_started:
                self.launch_astra(obs)
            old_epoch = self.task_epoch
            self.choose_task(obs)
            # Ask Jev only at contract creation, delivery credit, or a real
            # partner anomaly; it is not in the primitive control loop.
            if self.task_epoch != old_epoch:
                self.launch_jev(obs, "role_task_boundary")
            elif self.partner_bad >= 5:
                self.launch_jev(obs, "handoff_anomaly")
            elif tick(obs) - self.last_jev >= 55:
                self.launch_jev(obs, "role_heartbeat")

            action = self.motor(obs)
            me_now, partner_now, _, _ = actors(obs)
            self.log("motor_state", tick=tick(obs), position=point(me_now.get("position")),
                     orientation=point(me_now.get("orientation")), held=held(me_now),
                     kind=self.task.get("kind") if self.task else "",
                     target=self.task.get("target") if self.task else None, action=action)
            try:
                if self.ctx.done:
                    return
                self.ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                try:
                    if not self.ctx.done:
                        self.ctx.act(self.act_name(self.noop))
                except Exception:
                    return
            self.last_action = action
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                return


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        if not getattr(game, "tick_s", None):
            return
        RolePipeline(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
