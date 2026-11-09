"""Candidate 0019: a confirmed-start conveyor with progress-gated takeover.

This controller is intentionally split into three layers.  Observation code derives
pot phases, recipe-compatible work, and interaction geometry.  A conveyor scheduler
turns that into short transactions, and a one-tick motor executes only revalidated
transactions.  Jev can choose between code-generated transactions; Astra can suggest
whether the next safe interval is better spent staging or protecting a handoff.  No
model is allowed to emit a primitive action.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


VECTORS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
ACTION_BY_VECTOR = {v: k for k, v in VECTORS.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}
CARRYABLE = INGREDIENTS | {"dish", "soup"}


def pair(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            if isinstance(value[0], bool) or isinstance(value[1], bool):
                return None
            return (int(value[0]), int(value[1]))
        except Exception:
            return None
    return None


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def state_of(obs):
    return obs.get("state", {}) if isinstance(obs, dict) and isinstance(obs.get("state"), dict) else {}


def terrain_of(obs):
    value = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return value if isinstance(value, list) else []


def tick_of(obs):
    try:
        return int(obs.get("tick", state_of(obs).get("timestep", 0)))
    except Exception:
        return 0


def cell(terrain, point):
    point = pair(point)
    if point is None:
        return "X"
    x, y = point
    if y < 0 or y >= len(terrain):
        return "X"
    try:
        row = terrain[y]
        if isinstance(row, (str, list, tuple)) and 0 <= x < len(row):
            return row[x]
    except Exception:
        pass
    return "X"


def walkable(terrain, point):
    return cell(terrain, point) not in FEATURES and cell(terrain, point) != "X"


def features(terrain, symbol):
    found = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                found.append((x, y))
    return found


def neighbours(point):
    x, y = point
    for vector in VECTORS.values():
        yield (x + vector[0], y + vector[1]), vector


def bfs(terrain, start, goal, forbidden=()):
    start, goal = pair(start), pair(goal)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    blocked = set(forbidden)
    blocked.discard(start)
    queue = deque([start])
    parent = {start: None}
    while queue:
        here = queue.popleft()
        for nxt, _ in neighbours(here):
            if nxt in parent or nxt in blocked or not walkable(terrain, nxt):
                continue
            parent[nxt] = here
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return None


def interaction_poses(terrain, target):
    target = pair(target)
    if target is None:
        return []
    answer = []
    for vector in VECTORS.values():
        pose = (target[0] - vector[0], target[1] - vector[1])
        if walkable(terrain, pose) and pose not in answer:
            answer.append(pose)
    return answer


def path_to_interaction(terrain, start, target, forbidden=()):
    choices = []
    for pose in interaction_poses(terrain, target):
        path = bfs(terrain, start, pose, forbidden)
        if path is not None:
            choices.append((len(path), pose, path))
    if not choices:
        # A partner may be on the only approach.  Keep a route queued, but do not
        # fabricate a path through the partner in the normal motor.
        for pose in interaction_poses(terrain, target):
            path = bfs(terrain, start, pose)
            if path is not None:
                choices.append((len(path), pose, path))
    if not choices:
        return None, None
    choices.sort(key=lambda item: (item[0], item[1]))
    return choices[0][1], choices[0][2]


def held_name(player):
    obj = player.get("held_object") if isinstance(player, dict) else None
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("name", "") or "").lower()


def players_of(obs):
    value = state_of(obs).get("players", [])
    return value if isinstance(value, list) else []


def actors(obs):
    players = players_of(obs)
    try:
        index = int(obs.get("controlled_agent_index", 0))
    except Exception:
        index = 0
    if index not in (0, 1) or index >= len(players):
        index = 0
    other = 1 - index if len(players) > 1 else index
    me = players[index] if index < len(players) and isinstance(players[index], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, partner


def objects_of(obs):
    value = state_of(obs).get("objects", [])
    return value if isinstance(value, list) else []


def object_name(obj):
    return str(obj.get("name", "") or "").lower() if isinstance(obj, dict) else ""


def object_pos(obj):
    return pair(obj.get("position")) if isinstance(obj, dict) else None


def ingredients_of(obj):
    if not isinstance(obj, dict):
        return []
    raw = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("name", "")
        item = str(item or "").lower()
        if item in INGREDIENTS:
            result.append(item)
    return result


def recipes_of(obs):
    raw = state_of(obs).get("all_orders", [])
    answer = []
    if isinstance(raw, list):
        for order in raw:
            values = order.get("ingredients") if isinstance(order, dict) else order
            if not isinstance(values, list):
                continue
            recipe = []
            for item in values:
                if isinstance(item, dict):
                    item = item.get("name", "")
                item = str(item or "").lower()
                if item in INGREDIENTS:
                    recipe.append(item)
            if recipe:
                answer.append(recipe)
    return answer or [["onion", "onion", "onion"]]


def recipe_missing(have, recipes):
    """Return the compatible active recipe and ordered missing ingredients."""
    have_count = Counter(have)
    best_recipe, best_missing = None, None
    for recipe in recipes:
        want = Counter(recipe)
        if any(have_count[name] > want[name] for name in have_count):
            continue
        current = Counter(have)
        missing = []
        for name in recipe:
            if current[name] < want[name]:
                missing.append(name)
                current[name] += 1
        if best_missing is None or (len(missing), tuple(recipe)) < (len(best_missing), tuple(best_recipe)):
            best_recipe, best_missing = list(recipe), missing
    return best_recipe or [], best_missing if best_missing is not None else []


def pot_facts(obs):
    terrain = terrain_of(obs)
    soups = {}
    for obj in objects_of(obs):
        point = object_pos(obj)
        if point is not None and object_name(obj) == "soup":
            soups[point] = obj
    recipes = recipes_of(obs)
    answer = []
    for point in features(terrain, "P"):
        soup = soups.get(point)
        have = ingredients_of(soup)
        recipe, missing = recipe_missing(have, recipes)
        ready = bool(soup and soup.get("is_ready"))
        raw_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        raw_time = soup.get("cook_time", -1) if soup else -1
        cooking = bool(soup and soup.get("is_cooking")) and not ready
        if not ready and isinstance(raw_tick, (int, float)) and isinstance(raw_time, (int, float)):
            if raw_tick >= 0 and raw_time >= 0 and soup and not soup.get("is_idle", False):
                cooking = True
        full = bool(soup and not missing and recipe)
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
        answer.append({
            "pos": point, "ingredients": have, "recipe": recipe, "missing": missing,
            "phase": phase, "ready": ready, "cooking": cooking, "full": full,
            "remaining": remaining,
        })
    return answer


def counter_items(obs):
    terrain = terrain_of(obs)
    result = []
    for obj in objects_of(obs):
        point, name = object_pos(obj), object_name(obj)
        if point is not None and name in CARRYABLE and cell(terrain, point) == "X":
            result.append((name, point))
    return result


def counter_sites(obs):
    terrain = terrain_of(obs)
    occupied = {object_pos(obj) for obj in objects_of(obs) if object_pos(obj) is not None}
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            point = (x, y)
            if value != "X" or point in occupied:
                continue
            if any(walkable(terrain, nxt) for nxt, _ in neighbours(point)):
                result.append(point)
    return result


def all_gate_cells(obs):
    terrain = terrain_of(obs)
    gates = set()
    for symbol in ("O", "T", "D", "P", "S"):
        for target in features(terrain, symbol):
            gates.update(interaction_poses(terrain, target))
    return gates


def nearest_route(terrain, start, goals, forbidden=()):
    choices = []
    for goal in goals:
        path = bfs(terrain, start, goal, forbidden)
        if path is not None:
            choices.append((len(path), goal, path))
    if not choices:
        return None, None
    choices.sort(key=lambda row: (row[0], row[1]))
    return choices[0][1], choices[0][2]


class ConveyorPilot:
    """A receding-horizon executor with explicit start and bounded handoffs."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = getattr(game, "action_spec", None)
        self.valid_actions = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.valid_actions:
            self.noop = "stay" if "stay" in self.valid_actions else (next(iter(self.valid_actions), "stay"))

        self.plan = None
        self.plan_started = -1
        self.plan_epoch = 0
        self.plan_hint = None
        self.hint_tick = -999
        self.blocked = 0
        self.escape = 0
        self.previous_pos = None
        self.last_pos = None
        self.last_action = self.noop
        self.previous_partner = None
        self.partner_distance = None
        self.partner_still = 0
        self.partner_stall = 0
        self.partner_goal = None
        self.last_jev_tick = -999
        self.jev_future = None
        self.jev_epoch = -1
        self.astra_future = None
        self.astra_started = False
        self.astra_hint = {}
        self.system1_off = getattr(ctx, "system1", object()) is None
        self.system2_off = getattr(ctx, "system2", object()) is None

    # ---------- defensive plumbing -------------------------------------
    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def action(self, name):
        if name in self.valid_actions:
            return name
        if self.noop in self.valid_actions:
            return self.noop
        return next(iter(self.valid_actions), "stay")

    def _partner_next(self, obs):
        _, partner = actors(obs)
        point = pair(partner.get("position"))
        last = obs.get("partner_action_last", "") if isinstance(obs, dict) else ""
        delta = VECTORS.get(last, (0, 0))
        return (point[0] + delta[0], point[1] + delta[1]) if point is not None else None

    # ---------- partner progress ---------------------------------------
    def _partner_goal(self, obs, facts):
        _, partner = actors(obs)
        held = held_name(partner)
        terrain = facts["terrain"]
        if held == "soup":
            return ("serve", tuple(features(terrain, "S")))
        if held == "dish":
            ready = tuple(p["pos"] for p in facts["pots"] if p["ready"])
            if ready:
                return ("ready", ready)
            cooking = tuple(p["pos"] for p in facts["pots"] if p["cooking"])
            return ("cooking", cooking)
        return ("none", ())

    def update_partner(self, obs, facts):
        _, partner = actors(obs)
        current = pair(partner.get("position"))
        if current == self.previous_partner:
            self.partner_still += 1
        else:
            self.partner_still = 0
        goal = self._partner_goal(obs, facts)
        if goal != self.partner_goal:
            self.partner_goal = goal
            self.partner_distance = None
            self.partner_stall = 0
        goals = [pose for target in goal[1] for pose in interaction_poses(facts["terrain"], target)]
        if current is not None and goals:
            _, path = nearest_route(facts["terrain"], current, goals)
            distance = len(path) - 1 if path is not None else 999
            if self.partner_distance is not None:
                if distance < self.partner_distance:
                    self.partner_stall = 0
                elif distance > self.partner_distance + 1 or self.partner_still >= 2:
                    self.partner_stall += 1
            self.partner_distance = distance
        else:
            self.partner_distance = None
            self.partner_stall = 0
        self.previous_partner = current

    def partner_at_target(self, obs, target):
        _, partner = actors(obs)
        current, facing, target = pair(partner.get("position")), pair(partner.get("orientation")), pair(target)
        if current is None or facing is None or target is None:
            return False
        return current in interaction_poses(terrain_of(obs), target) and facing == (target[0] - current[0], target[1] - current[1])

    def takeover_needed(self, obs, facts, pot):
        if not pot or not (pot["ready"] or pot["cooking"]):
            return False
        _, partner = actors(obs)
        if held_name(partner) != "dish":
            return True
        if self.partner_at_target(obs, pot["pos"]):
            return False
        # Do not duplicate a dish merely because a cook is nearly finished:
        # the scripted partner often acquires its dish several ticks early.
        # A backup is reserved for a ready pot, or for the final two cook ticks
        # when the partner is not already adjacent to its approach.
        remaining = pot.get("remaining")
        urgent = pot["ready"] or (isinstance(remaining, (int, float)) and remaining <= 2)
        if not urgent:
            return False
        if isinstance(self.partner_distance, (int, float)) and self.partner_distance <= 1:
            return False
        return self.partner_stall >= 8 or self.partner_still >= 10

    # ---------- facts and transaction construction ---------------------
    def facts(self, obs):
        me, partner = actors(obs)
        return {
            "terrain": terrain_of(obs),
            "me": me,
            "partner": partner,
            "me_pos": pair(me.get("position")),
            "partner_pos": pair(partner.get("position")),
            "held": held_name(me),
            "partner_held": held_name(partner),
            "pots": pot_facts(obs),
            "recipes": recipes_of(obs),
            "counters": counter_items(obs),
            "free_counters": counter_sites(obs),
        }

    def task(self, kind, target=None, item="", priority=10, wait=False, tag="", description=""):
        target = pair(target)
        suffix = "_".join(map(str, target)) if target is not None else "none"
        ident = "%s_%s_%s%s" % (kind, item or "none", suffix, ("_" + tag) if tag else "")
        return {"id": ident, "kind": kind, "target": target, "item": item,
                "priority": int(priority), "wait": bool(wait), "tag": tag,
                "description": description or kind}

    def near_end(self, obs, margin=36):
        try:
            horizon = int(obs.get("horizon", state_of(obs).get("horizon", 400)))
            return tick_of(obs) >= horizon - margin
        except Exception:
            return tick_of(obs) >= 364

    def cooking_buffer(self, facts):
        """Return one or two next ingredients, but only after a real cook exists."""
        if self.near_end(self.last_obs):
            return None
        if not any(p["cooking"] for p in facts["pots"]):
            return None
        # Keep one counter as a shared handoff/traffic reserve.  The number of
        # already staged ingredients is derived, not inferred from a layout name.
        staged = [name for name, _ in facts["counters"] if name in INGREDIENTS]
        free = len(facts["free_counters"])
        capacity = min(2, max(0, free - 1))
        if len(staged) >= capacity:
            return None
        # Prefer the missing item of the shortest compatible future recipe.
        counts = Counter(staged)
        candidates = []
        for recipe in facts["recipes"]:
            need = Counter(recipe)
            if any(counts[name] > need[name] for name in counts):
                continue
            for item in recipe:
                if counts[item] < need[item]:
                    candidates.append(item)
                    break
        if not candidates:
            candidates = [facts["recipes"][0][0]] if facts["recipes"] and facts["recipes"][0] else []
        if not candidates:
            return None
        item = candidates[0]
        source = "O" if item == "onion" else "T"
        sources = features(facts["terrain"], source)
        return (item, sources) if sources else None

    def dish_due(self, facts, pot):
        remaining = pot.get("remaining")
        if not pot or not pot.get("cooking"):
            return False
        if not isinstance(remaining, (int, float)):
            return True
        terrain = facts["terrain"]
        current, partner = facts["me_pos"], facts["partner_pos"]
        estimate = 999
        for dispenser in features(terrain, "D"):
            pose, first = path_to_interaction(terrain, current, dispenser, {partner} if partner else set())
            if first is None or pose is None:
                continue
            for pot_pose in interaction_poses(terrain, pot["pos"]):
                second = bfs(terrain, pose, pot_pose)
                if second is not None:
                    estimate = min(estimate, len(first) + 1 + len(second) + 1)
        return remaining <= max(5, estimate + 2)

    def parking_pose(self, facts, target):
        terrain = facts["terrain"]
        current, partner = facts["me_pos"], facts["partner_pos"]
        if current is None:
            return None
        gates = set(interaction_poses(terrain, target))
        candidates = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                point = (x, y)
                if point in gates or point == partner or not walkable(terrain, point):
                    continue
                path = bfs(terrain, current, point, {partner} if partner else set())
                if path is None:
                    continue
                back = any(bfs(terrain, point, pose, {partner} if partner else set()) is not None for pose in gates)
                if back:
                    # Stay off every interaction approach, but remain close to
                    # the pot and on the side not occupied by the partner.
                    gap = abs(point[0] - target[0]) + abs(point[1] - target[1])
                    partner_gap = abs(point[0] - partner[0]) + abs(point[1] - partner[1]) if partner else 0
                    candidates.append((len(path), gap, -partner_gap, point))
        if not candidates:
            return current if current not in gates and current != partner else None
        candidates.sort()
        return candidates[0][3]

    def yield_target(self, facts):
        """Choose a floor cell outside every interaction approach for a soup runner."""
        terrain = facts["terrain"]
        current, partner = facts["me_pos"], facts["partner_pos"]
        gates = all_gate_cells(self.last_obs if isinstance(getattr(self, "last_obs", None), dict) else {})
        choices = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                point = (x, y)
                if not walkable(terrain, point) or point in gates or point == partner:
                    continue
                path = bfs(terrain, current, point, {partner} if partner else set())
                if path is None:
                    continue
                gap = abs(point[0] - partner[0]) + abs(point[1] - partner[1]) if partner else 0
                # Prefer a distant non-gate cell, but keep the displacement short
                # so yielding does not become a second long-term role.
                choices.append((-gap, len(path), point))
        if not choices:
            return current
        choices.sort()
        return choices[0][2]

    def soup_route_conflict(self, facts):
        """Whether the controlled chef is actually in the partner's serving path."""
        if facts["partner_held"] != "soup":
            return False
        current, partner = facts["me_pos"], facts["partner_pos"]
        if current is None or partner is None:
            return False
        terrain = facts["terrain"]
        goals = [pose for target in features(terrain, "S") for pose in interaction_poses(terrain, target)]
        _, open_path = nearest_route(terrain, partner, goals, {current})
        _, route = nearest_route(terrain, partner, goals)
        if open_path is None:
            return True
        if route is None:
            return False
        # Treat the first few route cells as a soft reservation, but do not
        # reserve the whole kitchen for a partner who can go around us.
        return current in route[:4] or abs(current[0] - partner[0]) + abs(current[1] - partner[1]) <= 1

    def candidates(self, obs, facts):
        terrain = facts["terrain"]
        current, partner = facts["me_pos"], facts["partner_pos"]
        held, partner_held = facts["held"], facts["partner_held"]
        pots = facts["pots"]
        result = []

        # Carrying an object fixes the next legal operation; models cannot
        # redirect it to a speculative job.
        if held == "soup":
            return [self.task("deliver", target, priority=0, description="deliver the carried soup").copy()
                    for target in features(terrain, "S")] or [self.task("idle", wait=True)]

        # A partner carrying finished soup has the only truly urgent traffic
        # claim.  Move the controlled chef, even when carrying a staged onion or
        # dish, to a non-gate cell instead of blocking the serving corridor.
        if partner_held == "soup" and self.soup_route_conflict(facts):
            target = self.yield_target(facts)
            kind = "yield_carry" if held else "yield"
            return [self.task(kind, target, priority=0, wait=True,
                              description="yield outside the partner soup route")]

        if held == "dish":
            ready = sorted((p for p in pots if p["ready"]), key=lambda p: p["pos"])
            if ready:
                return [self.task("pickup", p["pos"], priority=1, description="pick up ready soup with the carried dish") for p in ready]
            cooking = sorted((p for p in pots if p["cooking"]), key=lambda p: (p["remaining"] if isinstance(p["remaining"], (int, float)) else 999, p["pos"]))
            if cooking:
                park = self.parking_pose(facts, cooking[0]["pos"])
                q = self.task("hold_dish", park, priority=3, wait=True, tag="staged",
                              description="hold the dish in a non-gate parking cell until the pot is ready")
                q["pot"] = cooking[0]["pos"]
                return [q]
            return [self.task("idle", wait=True, description="hold the dish safely")]
        if held in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial", "full_idle"} and held in p["missing"]]
            compatible.sort(key=lambda p: (len(p["missing"]), p["pos"]))
            if compatible:
                return [self.task("fill", p["pos"], held, priority=2,
                                   description="put the carried ingredient into a compatible pot") for p in compatible]
            # Deposit only on a genuinely free counter.  Return all nearby
            # sites so the route ranker can choose the short side of a kitchen;
            # terrain enumeration otherwise tends to select a remote wall cell
            # before the useful counter beside the current lane.
            sites = sorted(
                facts["free_counters"],
                key=lambda point: (self.route_cost(facts, self.task("drop_counter", point, held)), point),
            )
            if sites:
                return [self.task("drop_counter", point, held, priority=6,
                                  description="stage the carried ingredient on a free counter") for point in sites[:5]]
            return [self.task("idle", wait=True, description="wait with the ingredient for a legal pot")]

        ready = [p for p in pots if p["ready"]]
        if ready:
            ready.sort(key=lambda p: p["pos"])
            urgent = ready[0]
            if partner_held != "dish" or self.takeover_needed(obs, facts, urgent):
                return [self.task("fetch_dish", target, priority=1,
                                  tag="backup" if partner_held == "dish" else "urgent",
                                  description="fetch a dish for the ready soup; partner handoff is absent or stalled")
                        for target in features(terrain, "D")]

        # Explicit start always outranks a new supply run.  This is the
        # confirmed-start guard that prevents pre-start staging from delaying the
        # first soup (the failure mode observed in 0015).
        full_idle = [p for p in pots if p["phase"] == "full_idle"]
        if full_idle:
            full_idle.sort(key=lambda p: p["pos"])
            return [self.task("start", p["pos"], priority=2,
                              description="explicitly start the full idle pot") for p in full_idle]

        # In the last cook-and-travel window, a fresh dispenser run usually
        # creates an undeliverable onion and can obstruct the partner's finished
        # soup.  Finish carried work and ready/cooking work, but do not launch a
        # new empty-handed build that cannot clear the horizon.
        if self.near_end(obs) and not any(p["cooking"] for p in pots):
            return [self.task("idle", self.yield_target(facts), priority=18, wait=True,
                              description="protect the final delivery window")]

        buildable = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]]
        if buildable:
            buildable.sort(key=lambda p: (len(p["missing"]), p["pos"]))
            pot = buildable[0]
            wanted = pot["missing"][0]
            for name, point in facts["counters"]:
                if name == wanted:
                    result.append(self.task("take_counter", point, wanted, priority=3,
                                            description="retrieve the matching staged ingredient from the counter"))
            symbol = "O" if wanted == "onion" else "T"
            for source in features(terrain, symbol):
                result.append(self.task("fetch_ingredient", source, wanted, priority=5,
                                        description="fetch the next recipe ingredient from its dispenser"))
            return result or [self.task("idle", wait=True)]

        cooking = sorted((p for p in pots if p["cooking"]), key=lambda p: (p["remaining"] if isinstance(p["remaining"], (int, float)) else 999, p["pos"]))
        if cooking:
            pot = cooking[0]
            # A dish is requested just in time, unless the partner's dish route
            # is active.  A stale active handoff gets a bounded backup, never an
            # indefinite claim/lease.
            if partner_held != "dish" and self.dish_due(facts, pot):
                return [self.task("fetch_dish", target, priority=1, tag="timed",
                                  description="fetch a dish now to meet the cooking boundary") for target in features(terrain, "D")]
            if partner_held == "dish" and self.takeover_needed(obs, facts, pot):
                return [self.task("fetch_dish", target, priority=1, tag="backup",
                                  description="take a second dish because the partner handoff stopped progressing") for target in features(terrain, "D")]

            # Staging is productive only once the observation proves that a pot
            # is cooking.  Never stage during the initial fill/full-idle phase.
            buffer_item = self.cooking_buffer(facts)
            if buffer_item:
                item, sources = buffer_item
                for source in sources:
                    result.append(self.task("fetch_ingredient", source, item, priority=7, tag="stage",
                                            description="pre-stage one next-recipe ingredient during the confirmed cook"))
                if result:
                    return result
            park = self.parking_pose(facts, pot["pos"])
            q = self.task("park", park, priority=10, wait=True,
                          description="clear the pot approach while the soup cooks")
            q["pot"] = pot["pos"]
            return [q]

        # With no pot transaction available, keep the chef off feature gates.
        return [self.task("idle", self.parking_pose(facts, features(terrain, "P")[0]) if features(terrain, "P") else current,
                          priority=20, wait=True, description="wait in a safe floor cell")]

    # ---------- model-facing interface ---------------------------------
    def compact(self, obs, facts, options):
        plan = self.plan or {}
        return json_safe({
            "tick": tick_of(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {"position": facts["me_pos"], "facing": pair(facts["me"].get("orientation")), "held": facts["held"]},
            "partner": {"position": facts["partner_pos"], "facing": pair(facts["partner"].get("orientation")),
                        "held": facts["partner_held"], "still": self.partner_still,
                        "route_not_improving": self.partner_stall,
                        "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else ""},
            "pots": [{"position": p["pos"], "phase": p["phase"], "missing": p["missing"], "remaining": p["remaining"]} for p in facts["pots"]],
            "buffer": {"staged": facts["counters"], "free_sites": len(facts["free_counters"])},
            "handoff": {"takeover_needed": any(self.takeover_needed(obs, facts, p) for p in facts["pots"]),
                        "partner_goal": self.partner_goal},
            "plan": {"kind": plan.get("kind", ""), "target": plan.get("target"), "age": max(0, tick_of(obs) - self.plan_started), "blocked": self.blocked},
            "candidates": [{"id": q.get("id"), "kind": q.get("kind"), "priority": q.get("priority"), "description": q.get("description")} for q in options[:10]],
            "astra": self.astra_hint,
        })

    def launch_astra(self, obs, facts, options):
        if self.astra_started or self.system2_off:
            self.astra_started = True
            return
        self.astra_started = True
        state = self.compact(obs, facts, options)
        prompt = (
            "You are a slow strategy reviewer for a real-time symbolic Overcooked-AI kitchen. "
            "The executor owns exact recipe arithmetic, pot phase, BFS geometry, collision safety, "
            "and primitive actions. Review this snapshot and return ONLY JSON with fields "
            "tempo='stage_after_start'|'protect_dish', lane='nearest'|'left'|'right', "
            "and takeover='bounded'|'conservative'. Never output directions or a new task id. "
            "Staging before the first explicit start is forbidden; a ready soup has precedence.\n" +
            json.dumps(state, separators=(",", ":"))
        )

        def ask():
            if self.ctx.done:
                return ""
            try:
                answer = self.ctx.think(prompt, timeout_s=90.0)
                return getattr(answer, "text", "")
            except Exception:
                return ""

        try:
            self.astra_future = self.ctx.spawn(ask)
        except Exception:
            self.astra_future = None

    def poll_astra(self, obs):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = result if isinstance(result, str) else getattr(result, "text", "")
            match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
            value = json.loads(match.group(0)) if match else {}
            if isinstance(value, dict):
                self.astra_hint = {key: value[key] for key in ("tempo", "lane", "takeover")
                                   if value.get(key) in {"stage_after_start", "protect_dish", "nearest", "left", "right", "bounded", "conservative"}}
                self.log("astra_policy", tick=tick_of(obs), policy=self.astra_hint)
        except Exception:
            self.astra_hint = {}

    def launch_jev(self, obs, facts, options, reason):
        if self.system1_off or self.jev_future is not None or not options:
            return
        now = tick_of(obs)
        if now - self.last_jev_tick < 18:
            return
        criteria = {str(q["id"]): str(q.get("description", q.get("kind", "candidate"))) for q in options[:10] if q.get("id")}
        if not criteria:
            return
        self.last_jev_tick = now
        self.jev_epoch = self.plan_epoch
        state = {"game": "Overcooked-AI symbolic real-time kitchen", "observation": self.compact(obs, facts, options)}
        questions = {
            "transaction": Choice(
                "Choose one code-generated transaction in observation.candidates. Prefer the lower-priority-number useful job, but never invent an id, direction, object, or arithmetic. The executor will recheck legality.",
                criteria,
            ),
            "takeover": Noul(
                "Should the controlled chef fetch a backup dish now? Say true only when observation.handoff.takeover_needed is true or the partner dish route has visibly stopped improving near a ready/nearly-ready pot. Say false for a progressing partner or no urgent pot.",
                {"true": "The partner handoff is stale and an urgent pot needs a dish.",
                 "false": "The partner is progressing or the urgent handoff condition is absent."},
            ),
        }

        def ask():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(state, questions, timeout_s=1.05)
            except Exception:
                return {}

        try:
            self.jev_future = self.ctx.spawn(ask)
            self.log("jev_request", tick=now, reason=reason, epoch=self.plan_epoch)
        except Exception:
            self.jev_future = None

    def poll_jev(self, obs, facts, options):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict) or self.jev_epoch != self.plan_epoch:
            return
        valid = {q.get("id"): q for q in options}
        answer = answers.get("transaction")
        selected = getattr(answer, "value", None)
        probabilities = getattr(answer, "probabilities", {}) or {}
        try:
            confidence = float(probabilities.get(selected, 0.0)) if isinstance(probabilities, dict) else 0.0
        except Exception:
            confidence = 0.0
        if selected in valid and (confidence >= 0.56 or confidence == 0.0):
            best_priority = min(q.get("priority", 20) for q in options)
            if valid[selected].get("priority", 20) <= best_priority + 1:
                self.plan_hint = selected
                self.hint_tick = tick_of(obs)
                self.log("jev_transaction", tick=tick_of(obs), value=selected, probability=confidence)
        takeover = answers.get("takeover")
        try:
            probability = float(getattr(takeover, "value", 0.0))
        except Exception:
            probability = 0.0
        if probability >= 0.80:
            for q in options:
                if q.get("kind") == "fetch_dish" and q.get("tag") == "backup":
                    self.plan_hint = q["id"]
                    self.hint_tick = tick_of(obs)
                    self.log("jev_takeover", tick=tick_of(obs), value=q["id"], probability=probability)
                    break

    # ---------- plan validity and motor ---------------------------------
    def same_task(self, left, right):
        return bool(left and right and left.get("kind") == right.get("kind") and
                    pair(left.get("target")) == pair(right.get("target")) and
                    left.get("item", "") == right.get("item", "") and
                    left.get("tag", "") == right.get("tag", ""))

    def age_limit(self, kind):
        return {"deliver": 48, "fetch_dish": 42, "fetch_ingredient": 36,
                "take_counter": 30, "drop_counter": 20, "fill": 20, "start": 18,
                "pickup": 22, "park": 24, "hold_dish": 50, "idle": 18}.get(kind, 28)

    def precondition(self, obs, facts, plan):
        kind, target, held = plan.get("kind"), pair(plan.get("target")), facts["held"]
        terrain = facts["terrain"]
        pot = next((p for p in facts["pots"] if p["pos"] == target), None)
        if kind == "deliver": return held == "soup" and cell(terrain, target) == "S"
        if kind == "fetch_dish": return not held and cell(terrain, target) == "D"
        if kind == "fetch_ingredient": return not held and cell(terrain, target) in {"O", "T"}
        if kind == "take_counter": return not held and any(n == plan.get("item") and p == target for n, p in facts["counters"])
        if kind == "drop_counter": return held in INGREDIENTS and target in facts["free_counters"]
        if kind == "fill": return held in INGREDIENTS and pot is not None and held in pot["missing"] and pot["phase"] in {"empty", "partial", "full_idle"}
        if kind == "start": return not held and pot is not None and pot["phase"] == "full_idle"
        if kind == "pickup": return held == "dish" and pot is not None and pot["ready"]
        if kind in {"park", "hold_dish", "idle", "yield", "yield_carry"}: return True
        return False

    def choose_plan(self, obs, facts, options):
        if not options:
            return self.task("idle", wait=True)
        chosen = None
        if self.plan_hint and tick_of(obs) - self.hint_tick <= 5:
            for q in options:
                if q.get("id") == self.plan_hint:
                    chosen = q.copy()
                    break
        if chosen is None:
            chosen = min(options, key=lambda q: (q.get("priority", 20), self.route_cost(facts, q), q.get("id", ""))).copy()
        if chosen.get("kind") in {"park", "hold_dish"} and chosen.get("target") is None:
            chosen["target"] = facts["me_pos"]
        chosen["pose"] = chosen.get("target")
        if chosen.get("kind") not in {"park", "hold_dish", "idle"}:
            pose, _ = path_to_interaction(facts["terrain"], facts["me_pos"], chosen.get("target"),
                                          {facts["partner_pos"]} if facts["partner_pos"] else set())
            chosen["pose"] = pose
        return chosen

    def route_cost(self, facts, plan):
        target = pair(plan.get("target"))
        if target is None or facts["me_pos"] is None:
            return 999
        if plan.get("kind") in {"park", "hold_dish", "idle"}:
            path = bfs(facts["terrain"], facts["me_pos"], target, {facts["partner_pos"]} if facts["partner_pos"] else set())
        else:
            _, path = path_to_interaction(facts["terrain"], facts["me_pos"], target,
                                          {facts["partner_pos"]} if facts["partner_pos"] else set())
        return len(path) if path is not None else 999

    def refresh(self, obs, facts, options):
        if self.plan is not None:
            age = tick_of(obs) - self.plan_started
            if age <= self.age_limit(self.plan.get("kind", "")) and self.precondition(obs, facts, self.plan):
                for option in options:
                    if self.same_task(self.plan, option):
                        return False
                # Waiting jobs are dynamically represented in options, so a
                # missing matching option means their phase changed.
        chosen = self.choose_plan(obs, facts, options)
        self.plan = chosen
        self.plan_started = tick_of(obs)
        self.plan_epoch += 1
        self.blocked = 0
        self.escape = 0
        self.plan_hint = None
        self.log("conveyor_commit", tick=self.plan_started, kind=chosen.get("kind"),
                 target=chosen.get("target"), tag=chosen.get("tag", ""), epoch=self.plan_epoch)
        return True

    def escape_action(self, facts, goal=None):
        current, partner = facts["me_pos"], facts["partner_pos"]
        if current is None:
            return self.action(self.noop)
        predicted = self._partner_next(self.last_obs) if isinstance(getattr(self, "last_obs", None), dict) else None
        choices = []
        for nxt, vector in neighbours(current):
            if not walkable(facts["terrain"], nxt) or nxt == partner or nxt == predicted:
                continue
            gate_penalty = 1 if nxt in all_gate_cells(self.last_obs) and not facts["held"] else 0
            toward = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) if goal else 0
            away = -(abs(nxt[0] - partner[0]) + abs(nxt[1] - partner[1])) if partner else 0
            reverse = 1 if nxt == self.previous_pos else 0
            choices.append((gate_penalty, reverse, toward, away, nxt, vector))
        if not choices:
            return self.action(self.noop)
        choices.sort(key=lambda q: q[:4])
        return self.action(ACTION_BY_VECTOR.get(choices[0][5], self.noop))

    def can_interact_now(self, obs, facts, plan):
        target = pair(plan.get("target"))
        current = facts["me_pos"]
        facing = pair(facts["me"].get("orientation"))
        if current is None or target is None or (target[0] - current[0], target[1] - current[1]) not in ACTION_BY_VECTOR:
            return False
        desired = (target[0] - current[0], target[1] - current[1])
        if facing != desired:
            return False
        return self.precondition(obs, facts, plan)

    def motor(self, obs, facts):
        plan = self.plan or self.task("idle", wait=True)
        kind, target = plan.get("kind"), pair(plan.get("target"))
        if kind == "idle" or target is None:
            return self.action(self.noop)
        if self.escape > 0 and facts["held"] != "soup":
            self.escape -= 1
            return self.escape_action(facts, target)

        if kind in {"park", "hold_dish", "yield", "yield_carry"}:
            current, other = facts["me_pos"], facts["partner_pos"]
            path = bfs(facts["terrain"], current, target, {other} if other and other != target else set())
            if path is not None and len(path) > 1:
                nxt = path[1]
                predicted = self._partner_next(obs)
                if nxt == other or nxt == predicted:
                    self.blocked += 1
                    if self.blocked >= 3:
                        self.escape = 2
                        return self.escape_action(facts, target)
                    return self.action(self.noop)
                delta = (nxt[0] - current[0], nxt[1] - current[1])
                return self.action(ACTION_BY_VECTOR.get(delta, self.noop))
            return self.action(self.noop)

        current, other = facts["me_pos"], facts["partner_pos"]
        pose, path = path_to_interaction(facts["terrain"], current, target, {other} if other else set())
        if pose is None or path is None:
            self.blocked += 1
            if self.blocked >= 3:
                self.escape = 2
                return self.escape_action(facts, pose or target)
            return self.action(self.noop)
        if len(path) > 1:
            nxt = path[1]
            predicted = self._partner_next(obs)
            if nxt == other or nxt == predicted or (predicted == current and nxt == other):
                self.blocked += 1
                if self.blocked >= 3:
                    self.escape = 2
                    return self.escape_action(facts, pose)
                return self.action(self.noop)
            self.blocked = 0
            delta = (nxt[0] - current[0], nxt[1] - current[1])
            return self.action(ACTION_BY_VECTOR.get(delta, self.noop))
        desired = (target[0] - current[0], target[1] - current[1])
        facing = pair(facts["me"].get("orientation"))
        if facing != desired:
            return self.action(ACTION_BY_VECTOR.get(desired, self.noop))
        if plan.get("wait"):
            return self.action(self.noop)
        if self.precondition(obs, facts, plan):
            return self.action("interact")
        return self.action(self.noop)

    # ---------- realtime loop -------------------------------------------
    def run(self):
        ctx = self.ctx
        self.last_obs = {}
        while not ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self.last_obs = obs
            facts = self.facts(obs)
            self.update_partner(obs, facts)
            self.poll_astra(obs)
            options = self.candidates(obs, facts)
            if not self.astra_started:
                self.launch_astra(obs, facts, options)

            current = facts["me_pos"]
            if self.last_pos is not None and current == self.last_pos and self.last_action in VECTORS:
                self.blocked += 1
            elif current is not None and current != self.last_pos:
                self.blocked = max(0, self.blocked - 1)
            self.previous_pos = self.last_pos
            self.last_pos = current

            old_epoch = self.plan_epoch
            changed = self.refresh(obs, facts, options)
            self.poll_jev(obs, facts, options)
            if changed:
                self.launch_jev(obs, facts, options, "transaction_boundary")
            elif self.blocked >= 4:
                self.launch_jev(obs, facts, options, "blocked_route")
            elif tick_of(obs) - self.last_jev_tick >= 42:
                self.launch_jev(obs, facts, options, "heartbeat")

            if self.blocked >= 6 and facts["held"] != "soup":
                self.escape = max(self.escape, 2)
                self.log("bounded_escape", tick=tick_of(obs), blocked=self.blocked, epoch=self.plan_epoch)
                self.blocked = 0
            action = self.motor(obs, facts)
            try:
                if ctx.done:
                    return
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                try:
                    if not ctx.done:
                        ctx.act(self.action(self.noop))
                except Exception:
                    return
            self.last_action = action
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                return


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        if not getattr(game, "tick_s", None):
            return
        ConveyorPilot(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
