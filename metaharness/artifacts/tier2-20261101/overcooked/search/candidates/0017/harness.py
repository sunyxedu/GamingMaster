"""Candidate 0017: a progress-bounded pipeline with dish-backup recovery.

The kitchen is driven by code-owned symbolic production and a one-tick motor.
A medium-level transaction may survive movement, but it is not allowed to
survive a long period without physical progress.  In particular, a partner
holding a dish is treated as an active handoff only while the partner is
actually approaching a pot.  A stalled handoff creates a second, legal dish
run instead of leaving a ready soup unattended.

Astra and Jev are advisory and asynchronous.  Neither model emits a primitive
movement action.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


DIRECTIONS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
ACTION_FOR = {delta: name for name, delta in DIRECTIONS.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}
CARRYABLE = INGREDIENTS | {"dish", "soup"}


def pos(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            if isinstance(value[0], bool) or isinstance(value[1], bool):
                return None
            return int(value[0]), int(value[1])
        except Exception:
            return None
    return None


def finite_json(value):
    """Convert observations and model-facing summaries to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def state_of(obs):
    return obs.get("state", {}) if isinstance(obs, dict) and isinstance(obs.get("state"), dict) else {}


def terrain_of(obs):
    terrain = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return terrain if isinstance(terrain, list) else []


def tick_of(obs):
    try:
        return int(obs.get("tick", state_of(obs).get("timestep", 0)))
    except Exception:
        return 0


def cell(terrain, point):
    point = pos(point)
    if point is None:
        return "X"
    x, y = point
    if y < 0 or y >= len(terrain):
        return "X"
    try:
        row = terrain[y]
        if isinstance(row, (list, tuple, str)) and 0 <= x < len(row):
            return row[x]
    except Exception:
        pass
    return "X"


def walkable(terrain, point):
    value = cell(terrain, point)
    return value not in FEATURES and value != "X"


def feature_cells(terrain, symbol):
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (list, tuple, str)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                result.append((x, y))
    return result


def players_of(obs):
    value = state_of(obs).get("players", [])
    return value if isinstance(value, list) else []


def actors(obs):
    players = players_of(obs)
    try:
        mine = int(obs.get("controlled_agent_index", 0))
    except Exception:
        mine = 0
    if mine < 0 or mine >= len(players):
        mine = 0
    other = 1 - mine if len(players) >= 2 and mine in (0, 1) else mine
    me = players[mine] if mine < len(players) and isinstance(players[mine], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, partner


def held_name(player):
    obj = player.get("held_object") if isinstance(player, dict) else None
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("name", "") or "").lower()


def objects_of(obs):
    value = state_of(obs).get("objects", [])
    return value if isinstance(value, list) else []


def object_name(obj):
    return str(obj.get("name", "") or "").lower() if isinstance(obj, dict) else ""


def object_pos(obj):
    return pos(obj.get("position")) if isinstance(obj, dict) else None


def ingredient_names(obj):
    if not isinstance(obj, dict):
        return []
    values = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(values, list):
        return []
    names = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name", "")
        value = str(value or "").lower()
        if value in INGREDIENTS:
            names.append(value)
    return names


def recipes_of(obs):
    raw = state_of(obs).get("all_orders", [])
    recipes = []
    if isinstance(raw, list):
        for order in raw:
            values = order.get("ingredients") if isinstance(order, dict) else order
            if not isinstance(values, list):
                continue
            recipe = []
            for value in values:
                if isinstance(value, dict):
                    value = value.get("name", "")
                value = str(value or "").lower()
                if value in INGREDIENTS:
                    recipe.append(value)
            if recipe:
                recipes.append(recipe)
    return recipes or [["onion", "onion", "onion"]]


def missing_for(have, recipes):
    """Pick a compatible active order and return its missing ordered items."""
    counts = Counter(have)
    best_recipe = None
    best_missing = None
    for recipe in recipes:
        wanted = Counter(recipe)
        if any(counts[name] > wanted[name] for name in counts):
            continue
        remaining = Counter(have)
        missing = []
        for item in recipe:
            if remaining[item] < wanted[item]:
                missing.append(item)
                remaining[item] += 1
        if best_missing is None or len(missing) < len(best_missing):
            best_recipe = recipe
            best_missing = missing
    return best_recipe or [], best_missing if best_missing is not None else []


def neighbours(point):
    x, y = point
    for delta in DIRECTIONS.values():
        yield (x + delta[0], y + delta[1]), delta


def bfs(terrain, start, goal, forbidden=()):
    start, goal = pos(start), pos(goal)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    forbidden = set(forbidden)
    forbidden.discard(start)
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        for nxt, _ in neighbours(current):
            if nxt in previous or nxt in forbidden or not walkable(terrain, nxt):
                continue
            previous[nxt] = current
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return None


def interaction_poses(terrain, target):
    target = pos(target)
    if target is None:
        return []
    result = []
    for delta in DIRECTIONS.values():
        candidate = (target[0] - delta[0], target[1] - delta[1])
        if walkable(terrain, candidate) and candidate not in result:
            result.append(candidate)
    return result


def route_to_feature(terrain, start, target, blocked=()):
    """Return (interaction pose, path), preferring a route away from actors."""
    start = pos(start)
    target = pos(target)
    if start is None or target is None:
        return None, None
    blocked = set(blocked)
    choices = []
    for pose_target in interaction_poses(terrain, target):
        forbidden = set(blocked)
        if pose_target in forbidden:
            forbidden.discard(pose_target)
        path = bfs(terrain, start, pose_target, forbidden)
        if path is not None:
            choices.append((len(path), pose_target, path))
    if choices:
        choices.sort(key=lambda item: (item[0], item[1]))
        return choices[0][1], choices[0][2]
    # Keep a queued task if the partner currently owns the only pose.
    fallback = []
    for pose_target in interaction_poses(terrain, target):
        path = bfs(terrain, start, pose_target)
        if path is not None:
            fallback.append((len(path), pose_target, path))
    if fallback:
        fallback.sort(key=lambda item: (item[0], item[1]))
        return fallback[0][1], fallback[0][2]
    return None, None


def shortest_to_any(terrain, start, goals, forbidden=()):
    choices = []
    for goal in goals:
        path = bfs(terrain, start, goal, forbidden)
        if path is not None:
            choices.append((len(path), goal, path))
    if not choices:
        return None, None
    choices.sort(key=lambda item: (item[0], item[1]))
    return choices[0][1], choices[0][2]


def all_interaction_cells(obs):
    terrain = terrain_of(obs)
    cells = set()
    for symbol in FEATURES:
        for target in feature_cells(terrain, symbol):
            cells.update(interaction_poses(terrain, target))
    return cells


def counter_cells(obs, occupied=True):
    terrain = terrain_of(obs)
    used = {object_pos(obj) for obj in objects_of(obs) if object_pos(obj) is not None} if occupied else set()
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (list, tuple, str)):
            continue
        for x, value in enumerate(row):
            point = (x, y)
            if value != "X" or point in used:
                continue
            if any(walkable(terrain, nxt) for nxt, _ in neighbours(point)):
                result.append(point)
    return result


def counter_items(obs):
    terrain = terrain_of(obs)
    result = []
    for obj in objects_of(obs):
        point = object_pos(obj)
        name = object_name(obj)
        if point is not None and name in CARRYABLE and cell(terrain, point) == "X":
            result.append((name, point))
    return result


def pot_info(obs):
    terrain = terrain_of(obs)
    recipes = recipes_of(obs)
    soups = {}
    for obj in objects_of(obs):
        point = object_pos(obj)
        if point is not None and object_name(obj) == "soup":
            soups[point] = obj
    result = []
    for target in feature_cells(terrain, "P"):
        soup = soups.get(target)
        ingredients = ingredient_names(soup)
        recipe, missing = missing_for(ingredients, recipes)
        ready = bool(soup and soup.get("is_ready"))
        cooking = bool(soup and soup.get("is_cooking")) and not ready
        cook_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        cook_time = soup.get("cook_time", -1) if soup else -1
        if not ready and not cooking and isinstance(cook_tick, (int, float)) and isinstance(cook_time, (int, float)):
            cooking = cook_tick >= 0 and cook_time >= 0 and not bool(soup and soup.get("is_idle"))
        full = bool(soup and recipe and not missing)
        if ready:
            phase = "ready"
        elif cooking:
            phase = "cooking"
        elif full:
            phase = "full_idle"
        elif ingredients:
            phase = "partial"
        else:
            phase = "empty"
        remaining = None
        if cooking and isinstance(cook_tick, (int, float)) and isinstance(cook_time, (int, float)):
            remaining = max(0, int(cook_time - cook_tick))
        result.append({
            "pos": target,
            "ingredients": ingredients,
            "recipe": recipe,
            "missing": missing,
            "phase": phase,
            "ready": ready,
            "cooking": cooking,
            "full": full,
            "remaining": remaining,
        })
    return result


class AdaptivePipeline:
    """A bounded transaction executor with a code-checked model bridge."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = getattr(game, "action_spec", None)
        self.actions = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.actions:
            self.noop = "stay" if "stay" in self.actions else (next(iter(self.actions), "stay"))

        self.task = None
        self.task_epoch = 0
        self.task_started = -1
        self.task_hint = None
        self.hint_tick = -999
        self.last_task_key = None

        self.last_action = self.noop
        self.last_me_pos = None
        self.last_partner_pos = None
        self.blocked = 0
        self.recovery = 0
        self.recovery_goal = None
        self.waited_at_park = 0

        self.partner_held = ""
        self.partner_still = 0
        self.partner_bad_progress = 0
        self.partner_goal_key = None
        self.partner_distance = None

        self.lane = None
        self.astra_mode = None
        self.astra_lane = None
        self.astra_backup = None
        self.astra_future = None
        self.astra_started = False

        self.jev_future = None
        self.jev_epoch = -1
        self.last_jev_tick = -999
        self.system1_off = getattr(ctx, "system1", object()) is None
        self.last_obs = {}

    # ----- safe infrastructure ----------------------------------------

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def action(self, value):
        if value in self.actions:
            return value
        if self.noop in self.actions:
            return self.noop
        return next(iter(self.actions), "stay")

    def _partner_next(self, obs):
        _, partner = actors(obs)
        p = pos(partner.get("position"))
        move = DIRECTIONS.get(obs.get("partner_action_last", ""), (0, 0)) if isinstance(obs, dict) else (0, 0)
        return (p[0] + move[0], p[1] + move[1]) if p is not None else None

    def _initial_lane(self, obs):
        if self.lane in {"left", "right"}:
            return
        me, partner = actors(obs)
        mp, pp = pos(me.get("position")), pos(partner.get("position"))
        if mp is not None and pp is not None and mp[0] != pp[0]:
            self.lane = "left" if mp[0] < pp[0] else "right"
        else:
            self.lane = "nearest"
        self.log("pipeline_lane", tick=tick_of(obs), lane=self.lane)

    def _update_motion(self, obs):
        me, partner = actors(obs)
        current = pos(me.get("position"))
        partner_pos = pos(partner.get("position"))
        if self.last_me_pos is not None and current == self.last_me_pos and self.last_action in DIRECTIONS:
            self.blocked += 1
        elif current is not None and current != self.last_me_pos:
            self.blocked = max(0, self.blocked - 1)
        if partner_pos is not None and partner_pos == self.last_partner_pos:
            self.partner_still += 1
        else:
            self.partner_still = 0
        held = held_name(partner)
        if held != self.partner_held:
            self.partner_held = held
            self.partner_bad_progress = 0
            self.partner_goal_key = None
            self.partner_distance = None
        self.last_me_pos = current
        self.last_partner_pos = partner_pos

    # ----- partner handoff recognition ---------------------------------

    def _partner_goal(self, obs):
        _, partner = actors(obs)
        pheld = held_name(partner)
        if pheld == "soup":
            goals = feature_cells(terrain_of(obs), "S")
            return "serve", goals
        if pheld == "dish":
            pots = pot_info(obs)
            ready = [p["pos"] for p in pots if p["ready"]]
            if ready:
                return "ready", ready
            cooking = [p["pos"] for p in pots if p["cooking"]]
            return "cooking", cooking
        return "none", []

    def _update_partner_progress(self, obs):
        me, partner = actors(obs)
        ppos = pos(partner.get("position"))
        kind, goals = self._partner_goal(obs)
        key = (kind, tuple(goals))
        if key != self.partner_goal_key:
            self.partner_goal_key = key
            self.partner_bad_progress = 0
            self.partner_distance = None
        if ppos is not None and goals:
            terrain = terrain_of(obs)
            _, path = shortest_to_any(
                terrain,
                ppos,
                [pose for target in goals for pose in interaction_poses(terrain, target)],
            )
            distance = (len(path) - 1) if path is not None else 999
            if self.partner_distance is not None:
                if distance < self.partner_distance:
                    self.partner_bad_progress = 0
                elif self.partner_still >= 2:
                    self.partner_bad_progress += 1
                elif distance > self.partner_distance + 1:
                    self.partner_bad_progress += 1
            self.partner_distance = distance
        elif not goals:
            self.partner_bad_progress = 0
            self.partner_distance = None

    def _partner_at_interaction(self, obs, target):
        _, partner = actors(obs)
        ppos = pos(partner.get("position"))
        facing = pos(partner.get("orientation"))
        target = pos(target)
        if ppos is None or facing is None or target is None:
            return False
        desired = (target[0] - ppos[0], target[1] - ppos[1])
        return ppos in interaction_poses(terrain_of(obs), target) and facing == desired

    def _dish_backup_needed(self, obs, pot):
        """A dish in the partner's hand is not enough; verify its progress."""
        _, partner = actors(obs)
        pheld = held_name(partner)
        if not pot or not (pot.get("ready") or pot.get("cooking")):
            return False
        if pheld != "dish":
            return True
        if pot.get("ready") and self._partner_at_interaction(obs, pot["pos"]):
            return False
        # A moving partner or a short route gets a grace period.  Once the
        # route is not improving, a second dish transaction is safer than a
        # ready pot waiting for a stale handoff.
        remaining = pot.get("remaining")
        near_deadline = pot.get("ready") or (isinstance(remaining, (int, float)) and remaining <= 7)
        if near_deadline and (self.partner_bad_progress >= 5 or self.partner_still >= 8):
            return True
        return False

    # ----- exact task construction -------------------------------------

    def _task(self, kind, target=None, item="", priority=10, description="", wait=False, tag=""):
        target = pos(target)
        suffix = "_".join(str(v) for v in target) if target is not None else "none"
        ident = "%s_%s_%s%s" % (kind, item or "none", suffix, ("_" + tag) if tag else "")
        return {
            "id": ident,
            "kind": kind,
            "target": target,
            "item": item,
            "priority": int(priority),
            "description": description or kind,
            "wait": bool(wait),
            "tag": str(tag or ""),
        }

    def _safe_parking(self, obs, current, partner, exclude_target=None):
        terrain = terrain_of(obs)
        if current is None:
            return None
        gates = all_interaction_cells(obs)
        if exclude_target is not None:
            gates.update(interaction_poses(terrain, exclude_target))
        candidates = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (list, tuple, str)):
                continue
            for x in range(len(row)):
                point = (x, y)
                if not walkable(terrain, point) or point in gates or point == partner:
                    continue
                path = bfs(terrain, current, point, {partner} if partner is not None else set())
                if path is None:
                    continue
                partner_gap = abs(point[0] - partner[0]) + abs(point[1] - partner[1]) if partner else 0
                # Prefer a nearby non-gate cell, but use distance from the
                # partner to avoid making a second one-cell choke point.
                lane_penalty = 0
                if self.lane in {"left", "right"}:
                    width = len(terrain[0]) if terrain and isinstance(terrain[0], (list, tuple, str)) else 0
                    on_left = point[0] < width / 2.0
                    lane_penalty = 0 if (on_left == (self.lane == "left")) else 1
                candidates.append((lane_penalty, -partner_gap, len(path), point))
        if not candidates:
            return current if current not in gates and current != partner else None
        candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        return candidates[0][3]

    def _monitor_spot(self, obs, current, partner, pot_target):
        """Do not walk to a parking cell if the current cell is already safe."""
        if current is not None and current != partner and current not in all_interaction_cells(obs):
            return current
        return self._safe_parking(obs, current, partner, pot_target)

    def _cook_due(self, obs, pot, current, partner):
        if not pot or not pot.get("cooking"):
            return False
        remaining = pot.get("remaining")
        if not isinstance(remaining, (int, float)):
            return True
        terrain = terrain_of(obs)
        dishes = feature_cells(terrain, "D")
        if not dishes:
            return True
        cost = 999
        for dish in dishes:
            _, first = route_to_feature(terrain, current, dish, {partner} if partner else set())
            if first is None:
                continue
            dish_pose = first[-1]
            for pot_pose in interaction_poses(terrain, pot["pos"]):
                second = bfs(terrain, dish_pose, pot_pose)
                if second is not None:
                    cost = min(cost, (len(first) - 1) + 1 + (len(second) - 1) + 1)
        return remaining <= max(5, cost + 2)

    def _build_options(self, obs):
        terrain = terrain_of(obs)
        me, partner = actors(obs)
        current, partner_pos = pos(me.get("position")), pos(partner.get("position"))
        held, pheld = held_name(me), held_name(partner)
        pots = pot_info(obs)
        options = []

        # These branches are mutually exclusive and enforce object safety.
        if held == "soup":
            for target in feature_cells(terrain, "S"):
                options.append(self._task("deliver", target, priority=0,
                    description="deliver held soup at the serving window"))
            return options or [self._task("hold", priority=20, wait=True, description="hold soup safely")]

        if held == "dish":
            ready = [p for p in pots if p["ready"]]
            for pot in ready:
                options.append(self._task("pickup_ready", pot["pos"], priority=1,
                    description="pick up ready soup with the held dish"))
            if not ready:
                cooking = [p for p in pots if p["cooking"]]
                if cooking:
                    pot = min(cooking, key=lambda p: (p.get("remaining", 999) if isinstance(p.get("remaining"), (int, float)) else 999, p["pos"]))
                    parking = self._safe_parking(obs, current, partner_pos, pot["pos"])
                    options.append(self._task("wait_dish", parking, priority=4, wait=True,
                        description="keep the dish off the pot gate until cooking finishes",
                        tag="%s_%s" % pot["pos"]))
            return options or [self._task("hold", priority=20, wait=True, description="hold dish safely")]

        if held in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial", "full_idle"} and held in p["missing"]]
            compatible.sort(key=lambda p: (len(p["missing"]), p["pos"]))
            for pot in compatible:
                options.append(self._task("fill", pot["pos"], held, priority=2,
                    description="put held %s into a compatible pot" % held))
            if not options:
                for target in counter_cells(obs):
                    options.append(self._task("put_counter", target, held, priority=7,
                        description="stage held %s on a free counter" % held))
            return options or [self._task("hold", priority=20, wait=True, description="hold ingredient safely")]

        # Empty hands: a ready soup is an invariant, but a partner-owned
        # handoff is monitored only when there is no independent pot work.  On
        # a two-pot layout, yielding here would throw away the useful cooking
        # overlap that made the parent controller fast.
        ready = [p for p in pots if p["ready"]]
        if ready:
            target_pot = min(ready, key=lambda p: p["pos"])
            backup = self._dish_backup_needed(obs, target_pot)
            if backup:
                for target in feature_cells(terrain, "D"):
                    options.append(self._task("fetch_dish", target, priority=1,
                        description="fetch a backup dish for a ready or stalled soup",
                        tag="backup"))
            elif pheld in {"dish", "soup"}:
                independent_work = any(
                    p["pos"] != target_pot["pos"] and
                    (p["phase"] == "full_idle" or (p["phase"] in {"empty", "partial"} and p["missing"]))
                    for p in pots
                )
                if not independent_work:
                    parking = self._monitor_spot(obs, current, partner_pos, target_pot["pos"])
                    options.append(self._task("monitor_ready", parking, priority=3, wait=True,
                        description="yield the pot approach while partner's dish transaction progresses",
                        tag="monitor"))
            else:
                for target in feature_cells(terrain, "D"):
                    options.append(self._task("fetch_dish", target, priority=1,
                        description="fetch a dish for a ready soup"))
            if options:
                return options

        # Start is before speculative supply.  It is an explicit interaction
        # in modern dynamics and remains valid for mixed recipes.
        for pot in sorted((p for p in pots if p["phase"] == "full_idle"), key=lambda p: p["pos"]):
            options.append(self._task("start", pot["pos"], priority=2,
                description="explicitly start the full idle pot"))
        if options:
            return options

        # Build the most complete compatible pot.  Counter items are a short
        # handoff path and are considered before dispensers.
        buildable = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]]
        buildable.sort(key=lambda p: (len(p["missing"]), p["pos"]))
        if buildable:
            target_pot = buildable[0]
            wanted = target_pot["missing"][0]
            for name, target in counter_items(obs):
                if name == wanted:
                    options.append(self._task("take_counter", target, wanted, priority=4,
                        description="retrieve the staged %s for the next pot ingredient" % wanted))
            symbol = "O" if wanted == "onion" else "T"
            sources = feature_cells(terrain, symbol)
            for target in sources:
                options.append(self._task("fetch_ingredient", target, wanted, priority=5,
                    description="fetch one %s for the next compatible pot" % wanted))
            if options:
                return options

        cooking = [p for p in pots if p["cooking"]]
        if cooking:
            pot = min(cooking, key=lambda p: (p.get("remaining", 999) if isinstance(p.get("remaining"), (int, float)) else 999, p["pos"]))
            due = self._cook_due(obs, pot, current, partner_pos)
            # A dish is not staged early solely because a pot exists.  If the
            # partner already owns an improving dish transaction, let that
            # transaction supply the timed dish.  Only a stalled handoff earns
            # the backup exception.
            partner_active = pheld in {"dish", "soup"} and not self._dish_backup_needed(obs, pot)
            if (due and not partner_active) or self._dish_backup_needed(obs, pot):
                for target in feature_cells(terrain, "D"):
                    options.append(self._task("fetch_dish", target, priority=3,
                        description="fetch a dish timed to the cooking pot"))
                if options:
                    return options
            parking = self._safe_parking(obs, current, partner_pos, pot["pos"])
            options.append(self._task("park_cooking", parking, priority=12, wait=True,
                description="clear the cooking-pot gate while the dish is not yet due",
                tag="cook"))
            return options

        # No urgent work: a dish can seed the pipeline, otherwise remain in a
        # safe floor cell.  This branch is useful on sparse/adaptive layouts.
        if pheld not in {"dish", "soup"}:
            for target in feature_cells(terrain, "D"):
                options.append(self._task("fetch_dish", target, priority=9,
                    description="fetch a future dish for the production pipeline"))
        if options:
            return options
        parking = self._safe_parking(obs, current, partner_pos)
        return [self._task("idle", parking, priority=20, wait=True, description="wait in a non-gate floor cell")]

    # ----- task selection and model bridge -----------------------------

    def _task_key(self, task):
        if not task:
            return None
        return (task.get("kind"), pos(task.get("target")), task.get("item", ""), task.get("tag", ""))

    def _same_task(self, left, right):
        return self._task_key(left) == self._task_key(right)

    def _task_limits(self, kind):
        return {
            "deliver": 44,
            "fetch_dish": 42,
            "fetch_ingredient": 36,
            "take_counter": 32,
            "put_counter": 18,
            "fill": 16,
            "start": 16,
            "pickup_ready": 20,
            "wait_dish": 14,
            "monitor_ready": 14,
            "park_cooking": 18,
            "idle": 16,
            "hold": 16,
        }.get(kind, 24)

    def _valid_task(self, obs, options):
        if not self.task:
            return False
        if self.task.get("kind") in {"monitor_ready", "wait_dish", "park_cooking", "idle", "hold"}:
            # Waiting tasks must be rechecked against the current phase; the
            # same target is okay, but a due/ready transition must be visible.
            pass
        if any(self._same_task(self.task, option) for option in options):
            age = tick_of(obs) - self.task_started
            if age <= self._task_limits(self.task.get("kind", "")):
                return True
        return False

    def _route_cost(self, obs, task):
        me, partner = actors(obs)
        start = pos(me.get("position"))
        other = pos(partner.get("position"))
        target = pos(task.get("target"))
        if target is None:
            return 0
        if task.get("kind") in {"park_cooking", "wait_dish", "monitor_ready", "idle"}:
            path = bfs(terrain_of(obs), start, target, {other} if other else set())
            return len(path) if path is not None else 999
        _, path = route_to_feature(terrain_of(obs), start, target, {other} if other else set())
        return len(path) if path is not None else 999

    def _lane_cost(self, obs, task):
        if self.astra_lane not in {"left", "right"} or task.get("target") is None:
            return 0
        terrain = terrain_of(obs)
        width = len(terrain[0]) if terrain and isinstance(terrain[0], (list, tuple, str)) else 0
        target = task["target"]
        left = target[0] < width / 2.0
        return 0 if (left == (self.astra_lane == "left")) else 1

    def _choose_code(self, obs, options):
        if not options:
            return self._task("idle", priority=20, wait=True, description="safe idle")
        ranked = []
        for option in options:
            # Route is only a tie breaker after the production priority.
            ranked.append((option.get("priority", 20), self._lane_cost(obs, option),
                           self._route_cost(obs, option), option.get("id", ""), option))
        ranked.sort(key=lambda row: row[:4])
        return ranked[0][4].copy()

    def _compact(self, obs, options):
        me, partner = actors(obs)
        pots = pot_info(obs)
        return finite_json({
            "tick": tick_of(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {"position": pos(me.get("position")), "facing": pos(me.get("orientation")), "held": held_name(me)},
            "partner": {"position": pos(partner.get("position")), "facing": pos(partner.get("orientation")),
                        "held": held_name(partner), "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else "",
                        "progress": {"still": self.partner_still, "route_not_improving": self.partner_bad_progress}},
            "pots": [{"position": p["pos"], "phase": p["phase"], "missing": p["missing"],
                      "cook_remaining": p["remaining"]} for p in pots],
            "orders": recipes_of(obs),
            "current_transaction": {"kind": self.task.get("kind", "") if self.task else "",
                                     "target": self.task.get("target") if self.task else None,
                                     "age": max(0, tick_of(obs) - self.task_started) if self.task else 0,
                                     "blocked": self.blocked},
            "recovery": {"active": self.recovery > 0, "reason": "no_progress" if self.blocked >= 4 else "none"},
            "astra": {"mode": self.astra_mode, "lane": self.astra_lane, "backup": self.astra_backup},
            "candidates": [{"id": option.get("id"), "kind": option.get("kind"),
                            "priority": option.get("priority"), "description": option.get("description")} for option in options[:10]],
        })

    def _launch_astra(self, obs, options):
        if self.astra_started or getattr(self.ctx, "system2", object()) is None:
            self.astra_started = True
            return
        self.astra_started = True
        snapshot = self._compact(obs, options)
        prompt = (
            "You are a slow strategy reviewer for a realtime symbolic Overcooked-AI kitchen. "
            "The executor owns exact recipes, legality, BFS, collision recovery, and primitive "
            "actions. Review the first snapshot only. Choose a broad tempo, a soft ingredient "
            "side preference, and whether a stalled partner handoff should be rescued sooner. "
            "Return ONLY JSON {\"tempo\":\"timed\"|\"steady\","
            "\"lane\":\"left\"|\"right\"|\"nearest\","
            "\"backup\":\"conservative\"|\"aggressive\"}. Never output directions.\n" +
            json.dumps(snapshot, separators=(",", ":"))
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

    def _poll_astra(self, obs):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            value = future.result()
            text = value if isinstance(value, str) else getattr(value, "text", "")
            match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
            if not isinstance(parsed, dict):
                return
            if parsed.get("tempo") in {"timed", "steady"}:
                self.astra_mode = parsed["tempo"]
            if parsed.get("lane") in {"left", "right", "nearest"}:
                self.astra_lane = parsed["lane"]
            if parsed.get("backup") in {"conservative", "aggressive"}:
                self.astra_backup = parsed["backup"]
            self.log("astra_policy", tick=tick_of(obs), mode=self.astra_mode, lane=self.astra_lane,
                     backup=self.astra_backup)
        except Exception:
            return

    def _launch_jev(self, obs, options, reason):
        if self.system1_off or self.jev_future is not None or not options:
            return
        now = tick_of(obs)
        if now - self.last_jev_tick < 16:
            return
        self.last_jev_tick = now
        self.jev_epoch = self.task_epoch
        state = {"game": "Overcooked-AI symbolic kitchen", "observation": self._compact(obs, options)}
        criteria = {str(option.get("id")): str(option.get("description", option.get("kind", "candidate")))
                    for option in options[:10] if option.get("id")}
        if not criteria:
            return
        questions = {
            "transaction": Choice(
                "Choose one code-generated medium-level transaction in `observation.candidates`. "
                "Prefer the listed delivery, ready-soup pickup, explicit start, or compatible fill "
                "according to their already-computed priorities. Never invent an id, direction, "
                "object, or arithmetic result.",
                criteria,
            ),
            "rescue": Noul(
                "Should the controlled chef rescue a partner's dish handoff now? Say true only "
                "when a ready/nearly-ready pot is present and `observation.partner.progress` "
                "shows the partner has stopped improving its route; a moving partner or a partner "
                "already facing the pot is not stalled.",
                {"true": "The partner's dish route is visibly stalled near an urgent pot.",
                 "false": "The partner is progressing or the ready-pot invariant is absent."},
            ),
        }

        def ask():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(state, questions, timeout_s=1.1)
            except Exception:
                return {}

        try:
            self.jev_future = self.ctx.spawn(ask)
            self.log("jev_request", tick=now, reason=reason, epoch=self.task_epoch)
        except Exception:
            self.jev_future = None

    def _poll_jev(self, obs, options):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict) or self.jev_epoch != self.task_epoch:
            return
        valid = {option.get("id"): option for option in options}
        answer = answers.get("transaction")
        selected = getattr(answer, "value", None)
        probabilities = getattr(answer, "probabilities", {}) or {}
        try:
            probability = float(probabilities.get(selected, 0.0)) if isinstance(probabilities, dict) else 0.0
        except Exception:
            probability = 0.0
        if selected in valid and (probability >= 0.55 or probability == 0.0):
            candidate = valid[selected]
            best = self._choose_code(obs, options)
            # Jev can choose among the same urgency band, but cannot demote a
            # delivery or start into a speculative trip.
            if candidate.get("priority", 20) <= best.get("priority", 20) + 1:
                self.task_hint = selected
                self.hint_tick = tick_of(obs)
                self.log("jev_transaction", tick=tick_of(obs), value=selected, probability=probability)
        rescue = answers.get("rescue")
        try:
            rescue_probability = float(getattr(rescue, "value", 0.0))
        except Exception:
            rescue_probability = 0.0
        if rescue_probability >= 0.78:
            for option in options:
                if option.get("kind") == "fetch_dish" and option.get("tag") == "backup":
                    self.task_hint = option.get("id")
                    self.hint_tick = tick_of(obs)
                    self.log("jev_rescue", tick=tick_of(obs), value=self.task_hint)
                    break

    def _select(self, obs, options, force=False):
        if not force and self._valid_task(obs, options):
            return
        chosen = None
        if self.task_hint and tick_of(obs) - self.hint_tick <= 5:
            for option in options:
                if option.get("id") == self.task_hint:
                    best = self._choose_code(obs, options)
                    if option.get("priority", 20) <= best.get("priority", 20) + 1:
                        chosen = option.copy()
                    break
        if chosen is None:
            chosen = self._choose_code(obs, options)
        self.task = chosen
        self.task_started = tick_of(obs)
        self.task_epoch += 1
        self.blocked = 0
        self.waited_at_park = 0
        self.task_hint = None
        self.last_task_key = self._task_key(chosen)
        self.log("transaction_commit", tick=self.task_started, kind=chosen.get("kind"),
                 target=chosen.get("target"), epoch=self.task_epoch, priority=chosen.get("priority"))

    # ----- motor and recovery ------------------------------------------

    def _can_interact(self, obs, task):
        me, _ = actors(obs)
        held = held_name(me)
        kind = task.get("kind")
        target = pos(task.get("target"))
        pots = pot_info(obs)
        pot = next((p for p in pots if p["pos"] == target), None)
        terrain = terrain_of(obs)
        if kind == "deliver":
            return held == "soup"
        if kind == "fetch_ingredient":
            return not held and cell(terrain, target) in {"O", "T"}
        if kind == "fetch_dish":
            return not held and cell(terrain, target) == "D"
        if kind == "take_counter":
            return not held and any(name == task.get("item") and point == target for name, point in counter_items(obs))
        if kind == "put_counter":
            return held in INGREDIENTS and target not in {object_pos(obj) for obj in objects_of(obs)}
        if kind == "fill":
            return bool(pot and held in INGREDIENTS and pot["phase"] in {"empty", "partial", "full_idle"}
                        and held in pot["missing"])
        if kind == "start":
            return bool(pot and not held and pot["phase"] == "full_idle")
        if kind == "pickup_ready":
            return bool(pot and held == "dish" and pot["ready"])
        return False

    def _escape_step(self, obs, goal=None):
        me, partner = actors(obs)
        current, other = pos(me.get("position")), pos(partner.get("position"))
        if current is None:
            return self.action(self.noop)
        terrain = terrain_of(obs)
        predicted = self._partner_next(obs)
        choices = []
        for nxt, delta in neighbours(current):
            if not walkable(terrain, nxt) or nxt == other or nxt == predicted:
                continue
            if nxt in all_interaction_cells(obs) and not held_name(me):
                # Escape should clear a gate, not claim another one.
                continue
            partner_gap = abs(nxt[0] - other[0]) + abs(nxt[1] - other[1]) if other else 0
            goal_gap = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) if goal else 0
            reverse = 1 if nxt == self.last_me_pos else 0
            choices.append((-partner_gap, goal_gap, reverse, nxt, delta))
        if not choices:
            return self.action(self.noop)
        choices.sort(key=lambda row: row[:4])
        return self.action(ACTION_FOR.get(choices[0][4], self.noop))

    def _recover_if_needed(self, obs):
        if self.recovery <= 0:
            return None
        target = self.recovery_goal or (self.task.get("target") if self.task else None)
        action = self._escape_step(obs, target)
        self.recovery -= 1
        return action

    def _motor(self, obs):
        me, partner = actors(obs)
        current, other = pos(me.get("position")), pos(partner.get("position"))
        if self.task is None:
            return self.action(self.noop)

        # Once a path has demonstrably stopped, move away from the contested
        # cell for a few ticks.  This is the bounded escape inherited in spirit
        # from the robust parent, but it never abandons a carried soup.
        if self.blocked >= 5 and self.recovery <= 0:
            self.recovery = 3
            self.recovery_goal = pos(self.task.get("target"))
            self.task_hint = None
            self.log("bounded_recovery", tick=tick_of(obs), task=self.task.get("kind"), blocked=self.blocked)
        recovery_action = self._recover_if_needed(obs)
        if recovery_action is not None and held_name(me) != "soup":
            return recovery_action

        kind = self.task.get("kind")
        target = pos(self.task.get("target"))
        if target is None or kind in {"hold"}:
            return self.action(self.noop)
        if kind in {"wait_dish", "monitor_ready", "park_cooking", "idle"}:
            # A parking transaction is deliberately a floor transaction, not a
            # pot reservation.  Replanning at the next phase change handles
            # ready soup or a stalled partner.
            path = bfs(terrain_of(obs), current, target, {other} if other is not None and other != target else set())
            if path is not None and len(path) > 1:
                nxt = path[1]
                predicted = self._partner_next(obs)
                if nxt == other or nxt == predicted:
                    self.blocked += 1
                    return self.action(self.noop)
                delta = (nxt[0] - current[0], nxt[1] - current[1])
                return self.action(ACTION_FOR.get(delta, self.noop))
            self.waited_at_park += 1
            return self.action(self.noop)

        pose, path = route_to_feature(terrain_of(obs), current, target,
                                      {other} if other is not None and other != target else set())
        if pose is None or path is None:
            self.blocked += 1
            return self._escape_step(obs, target)
        if len(path) > 1:
            nxt = path[1]
            predicted = self._partner_next(obs)
            if nxt == other or nxt == predicted or (predicted == current and nxt == other):
                self.blocked += 1
                if self.blocked <= 2:
                    return self.action(self.noop)
                return self._escape_step(obs, pose)
            self.blocked = max(0, self.blocked - 1)
            delta = (nxt[0] - current[0], nxt[1] - current[1])
            return self.action(ACTION_FOR.get(delta, self.noop))

        desired_delta = (target[0] - current[0], target[1] - current[1])
        facing = pos(me.get("orientation"))
        desired_action = ACTION_FOR.get(desired_delta, self.noop)
        if desired_delta in ACTION_FOR and facing != desired_delta:
            return self.action(desired_action)
        if self._can_interact(obs, self.task):
            return self.action("interact")
        # A waiting task reached its floor target; an interaction task whose
        # precondition vanished will be invalidated at the next observation.
        return self.action(self.noop)

    # ----- main realtime loop ------------------------------------------

    def run(self):
        ctx = self.ctx
        while not ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self.last_obs = obs
            self._initial_lane(obs)
            self._update_motion(obs)
            self._update_partner_progress(obs)
            self._poll_astra(obs)
            options = self._build_options(obs)
            if not self.astra_started:
                self._launch_astra(obs, options)

            # A hard replan is a recovery from a physical stall, not a new
            # answer from Jev.  It keeps the deterministic motor alive.
            stale = bool(self.task and tick_of(obs) - self.task_started > self._task_limits(self.task.get("kind", "")))
            if stale or (self.blocked >= 8 and held_name(actors(obs)[0]) != "soup"):
                self.task = None
                self.recovery = max(self.recovery, 2)
                self.log("transaction_expired", tick=tick_of(obs), blocked=self.blocked)
            self._poll_jev(obs, options)
            old_epoch = self.task_epoch
            self._select(obs, options, force=False)
            if self.task_epoch != old_epoch:
                self._launch_jev(obs, options, "transaction_boundary")
            elif self.blocked >= 4:
                self._launch_jev(obs, options, "blocked_route")
            elif tick_of(obs) - self.last_jev_tick >= 40:
                self._launch_jev(obs, options, "heartbeat")

            action = self._motor(obs)
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
        AdaptivePipeline(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
