"""Candidate 0021: bounded phase leases for a symbolic Overcooked kitchen.

The executor is deliberately deterministic at the primitive level.  The two
models can classify a small set of already legal transactions, but neither can
produce a direction, recipe arithmetic, or an interaction that the current
observation does not support.

The distinctive part of this controller is a *scoped* role lease.  A chef may
own the delivery lane for the current soup, or the supply lane for one missing
ingredient, but the ownership expires at a phase boundary, after a short
silence, or when the physical transaction stops progressing.  This avoids both
unbounded yielding (the failure mode of a locked runner) and unbounded
ingredient detours (the failure mode of a free-running supplier).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


STEP = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
STEP_NAME = {v: k for k, v in STEP.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}
CARRYABLE = INGREDIENTS | {"dish", "soup"}


def xy(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            if isinstance(value[0], bool) or isinstance(value[1], bool):
                return None
            return int(value[0]), int(value[1])
        except Exception:
            return None
    return None


def safe_json(value):
    """Make an observation-derived value acceptable to ctx.decide."""
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
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


def tile(terrain, point):
    point = xy(point)
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


def open_cell(terrain, point):
    value = tile(terrain, point)
    return value not in FEATURES and value != "X"


def feature_cells(terrain, symbol):
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                result.append((x, y))
    return result


def neighbours(point):
    x, y = point
    # This order is stable, which matters when several equal routes exist.
    for delta in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        yield (x + delta[0], y + delta[1]), delta


def bfs(terrain, start, goal, blocked=()):
    start, goal = xy(start), xy(goal)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    blocked = {xy(p) for p in blocked if xy(p) is not None}
    blocked.discard(start)
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        for nxt, _ in neighbours(current):
            if nxt in previous or nxt in blocked or not open_cell(terrain, nxt):
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
    target = xy(target)
    if target is None:
        return []
    result = []
    for delta in STEP.values():
        pose = (target[0] - delta[0], target[1] - delta[1])
        if open_cell(terrain, pose) and pose not in result:
            result.append(pose)
    return result


def feature_route(terrain, start, target, blocked=()):
    """Shortest path to a legal interaction pose, with a blocked fallback."""
    start, target = xy(start), xy(target)
    if start is None or target is None:
        return None, None
    blocked = set(blocked)
    candidates = []
    for pose in interaction_poses(terrain, target):
        route_blocked = set(blocked)
        route_blocked.discard(pose)
        path = bfs(terrain, start, pose, route_blocked)
        if path is not None:
            candidates.append((len(path), pose, path))
    if not candidates:
        # Do not discard a useful transaction just because the partner owns its
        # only approach for this one snapshot.  The motor yields briefly and
        # then chooses an escape step.
        for pose in interaction_poses(terrain, target):
            path = bfs(terrain, start, pose)
            if path is not None:
                candidates.append((len(path), pose, path))
    if not candidates:
        return None, None
    candidates.sort(key=lambda value: (value[0], value[1]))
    _, pose, path = candidates[0]
    return pose, path


def players_of(obs):
    players = state_of(obs).get("players", [])
    return players if isinstance(players, list) else []


def actors(obs):
    players = players_of(obs)
    try:
        mine = int(obs.get("controlled_agent_index", 0))
    except Exception:
        mine = 0
    if mine not in (0, 1) or mine >= len(players):
        mine = 0
    other = 1 - mine if len(players) >= 2 else mine
    me = players[mine] if mine < len(players) and isinstance(players[mine], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, partner


def held_of(player):
    obj = player.get("held_object") if isinstance(player, dict) else None
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("name", "") or "").lower()


def objects_of(obs):
    objects = state_of(obs).get("objects", [])
    return objects if isinstance(objects, list) else []


def object_name(obj):
    return str(obj.get("name", "") or "").lower() if isinstance(obj, dict) else ""


def object_position(obj):
    return xy(obj.get("position")) if isinstance(obj, dict) else None


def object_ingredients(obj):
    if not isinstance(obj, dict):
        return []
    values = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name", "")
        value = str(value or "").lower()
        if value in INGREDIENTS:
            result.append(value)
    return result


def recipes_of(obs):
    raw = state_of(obs).get("all_orders", [])
    result = []
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
                result.append(recipe)
    return result or [["onion", "onion", "onion"]]


def recipe_gap(have, available):
    """Return the compatible active recipe and its ordered missing items."""
    present = Counter(have)
    selected = None
    missing_selected = None
    for recipe in available:
        wanted = Counter(recipe)
        if any(present[item] > wanted[item] for item in present):
            continue
        left = Counter(have)
        missing = []
        for item in recipe:
            if left[item] < wanted[item]:
                missing.append(item)
                left[item] += 1
        if missing_selected is None or len(missing) < len(missing_selected):
            selected, missing_selected = recipe, missing
    return selected or [], missing_selected if missing_selected is not None else []


def pot_states(obs):
    terrain = terrain_of(obs)
    soups = {}
    for obj in objects_of(obs):
        point = object_position(obj)
        if point is not None and object_name(obj) == "soup":
            soups[point] = obj
    result = []
    active = recipes_of(obs)
    for point in feature_cells(terrain, "P"):
        soup = soups.get(point)
        have = object_ingredients(soup)
        recipe, missing = recipe_gap(have, active)
        ready = bool(soup and soup.get("is_ready"))
        cooking = bool(soup and soup.get("is_cooking")) and not ready
        raw_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        raw_time = soup.get("cook_time", -1) if soup else -1
        if not ready and not cooking and isinstance(raw_tick, (int, float)) and isinstance(raw_time, (int, float)):
            cooking = raw_tick >= 0 and raw_time >= 0 and not bool(soup and soup.get("is_idle"))
        full = bool(soup and recipe and not missing)
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
            "pos": point,
            "phase": phase,
            "ingredients": have,
            "recipe": recipe,
            "missing": missing,
            "ready": ready,
            "cooking": cooking,
            "full": full,
            "remaining": remaining,
        })
    return result


def occupied_cells(obs):
    return {object_position(obj) for obj in objects_of(obs) if object_position(obj) is not None}


def counter_items(obs):
    terrain = terrain_of(obs)
    result = []
    for obj in objects_of(obs):
        point = object_position(obj)
        name = object_name(obj)
        if point is not None and name in CARRYABLE and tile(terrain, point) == "X":
            result.append((name, point))
    return result


def free_counters(obs):
    terrain = terrain_of(obs)
    occupied = occupied_cells(obs)
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            point = (x, y)
            if value == "X" and point not in occupied and any(open_cell(terrain, q) for q, _ in neighbours(point)):
                result.append(point)
    return result


def interaction_cells(obs):
    terrain = terrain_of(obs)
    result = set()
    for symbol in FEATURES:
        for target in feature_cells(terrain, symbol):
            result.update(interaction_poses(terrain, target))
    return result


def partner_next(obs):
    _, partner = actors(obs)
    point = xy(partner.get("position"))
    action = obs.get("partner_action_last", "") if isinstance(obs, dict) else ""
    delta = STEP.get(action, (0, 0))
    if point is None:
        return None
    return point[0] + delta[0], point[1] + delta[1]


class BoundedKitchen:
    """Code-owned symbolic planner with short-lived phase ownership."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = getattr(game, "action_spec", None)
        self.legal = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.legal:
            self.noop = "stay" if "stay" in self.legal else (next(iter(self.legal), "stay"))

        self.task = None
        self.task_since = -1
        self.epoch = 0
        self.last_action = self.noop
        self.last_position = None
        self.last_partner_position = None
        self.blocked = 0
        self.escape_left = 0
        self.escape_goal = None

        # Partner progress is evidence for a handoff, not a permanent claim.
        self.partner_held = ""
        self.partner_still = 0
        self.partner_bad = 0
        self.partner_distance = None
        self.partner_goal_key = None
        self.last_pot_signature = None
        self.supply_silence = 0

        # Phase lease: mode is a tie-breaker only and has a hard expiry.
        self.lease_mode = "supply"
        self.lease_until = 18
        self.delivery_seen = 0.0
        self.last_score = None
        self.assist_used = False
        self.lane = "nearest"

        self.hint_id = None
        self.hint_until = -1
        self.jev_future = None
        self.jev_epoch = -1
        self.last_jev_tick = -999
        self.astra_future = None
        self.astra_started = False
        self.astra_lane = None
        self.astra_rescue = None
        self.system1_off = getattr(ctx, "system1", object()) is None

    # ---------- harmless wrappers and observation bookkeeping ----------

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def action(self, value):
        if value in self.legal:
            return value
        if self.noop in self.legal:
            return self.noop
        return next(iter(self.legal), "stay")

    def update_motion(self, obs):
        me, partner = actors(obs)
        current = xy(me.get("position"))
        other = xy(partner.get("position"))
        if self.last_position is not None and current == self.last_position and self.last_action in STEP:
            self.blocked += 1
        elif current is not None and current != self.last_position:
            self.blocked = max(0, self.blocked - 1)
        if other is not None and other == self.last_partner_position:
            self.partner_still += 1
        else:
            self.partner_still = 0
        held = held_of(partner)
        if held != self.partner_held:
            self.partner_held = held
            self.partner_bad = 0
            self.partner_distance = None
            self.partner_goal_key = None
        self.last_position = current
        self.last_partner_position = other

    def observe_phase(self, obs):
        score = obs.get("sparse_score", obs.get("score", 0.0)) if isinstance(obs, dict) else 0.0
        try:
            score = float(score)
        except Exception:
            score = 0.0
        now = tick_of(obs)
        if self.last_score is not None and score > self.last_score:
            self.delivery_seen += score - self.last_score
            # A delivery opens a short courier lease.  It cannot suppress
            # supply forever: the expiry is deliberately smaller than a soup
            # cycle on every supplied layout.
            self.lease_mode = "courier"
            self.lease_until = now + 11
            self.assist_used = False
            self.task = None
            self.log("phase_boundary", tick=now, reason="team_delivery", lease=self.lease_mode)
        if score != self.last_score:
            self.last_score = score

        pots = pot_states(obs)
        signature = tuple((p["pos"], p["phase"], tuple(p["ingredients"]), p["remaining"]) for p in pots)
        if signature == self.last_pot_signature:
            self.supply_silence += 1
        else:
            self.supply_silence = 0
        self.last_pot_signature = signature
        if now >= self.lease_until:
            self.lease_mode = "supply"

    # ---------- partner progress and handoff evidence -----------------

    def partner_target(self, obs):
        _, partner = actors(obs)
        hand = held_of(partner)
        pots = pot_states(obs)
        if hand == "soup":
            return "serve", feature_cells(terrain_of(obs), "S")
        if hand == "dish":
            ready = [p["pos"] for p in pots if p["ready"]]
            if ready:
                return "ready", ready
            return "cooking", [p["pos"] for p in pots if p["cooking"]]
        if hand in INGREDIENTS:
            compatible = [p["pos"] for p in pots if hand in p["missing"] and p["phase"] in {"empty", "partial"}]
            return "fill", compatible
        return "none", []

    def update_partner_progress(self, obs):
        _, partner = actors(obs)
        point = xy(partner.get("position"))
        kind, goals = self.partner_target(obs)
        key = (kind, tuple(goals))
        if key != self.partner_goal_key:
            self.partner_goal_key = key
            self.partner_distance = None
            self.partner_bad = 0
        if point is not None and goals:
            distances = []
            for target in goals:
                for pose in interaction_poses(terrain_of(obs), target):
                    path = bfs(terrain_of(obs), point, pose)
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
            self.partner_distance = None
            self.partner_bad = 0

    def partner_at(self, obs, target):
        _, partner = actors(obs)
        point = xy(partner.get("position"))
        facing = xy(partner.get("orientation"))
        target = xy(target)
        if point is None or facing is None or target is None:
            return False
        return point in interaction_poses(terrain_of(obs), target) and facing == (target[0] - point[0], target[1] - point[1])

    def partner_handoff_active(self, obs, pot):
        if not pot:
            return False
        _, partner = actors(obs)
        hand = held_of(partner)
        if hand == "soup":
            return True
        if hand != "dish":
            return False
        if self.partner_at(obs, pot["pos"]):
            return True
        remaining = pot.get("remaining")
        urgent = pot.get("ready") or (isinstance(remaining, (int, float)) and remaining <= 7)
        if urgent:
            return self.partner_bad < 4 and self.partner_still < 7
        return self.partner_bad < 7 and self.partner_still < 10

    def backup_needed(self, obs, pot):
        if not pot or not (pot.get("ready") or pot.get("cooking")):
            return False
        _, partner = actors(obs)
        hand = held_of(partner)
        if hand != "dish":
            return True
        if self.partner_handoff_active(obs, pot):
            return False
        remaining = pot.get("remaining")
        return bool(pot.get("ready") or (isinstance(remaining, (int, float)) and remaining <= 6))

    def partner_supply_claim(self, obs, desired):
        """Recognize a partner who is visibly doing this exact supply leg."""
        _, partner = actors(obs)
        hand = held_of(partner)
        point = xy(partner.get("position"))
        facing = xy(partner.get("orientation"))
        terrain = terrain_of(obs)
        symbol = "O" if desired == "onion" else "T"
        for source in feature_cells(terrain, symbol):
            if point in interaction_poses(terrain, source) and facing == (source[0] - point[0], source[1] - point[1]):
                return True
        if hand == desired:
            return True
        if point is None:
            return False
        # A partner already at a compatible pot with the ingredient is also a
        # live supply claim; this matters during a one-cell handoff.
        for pot in pot_states(obs):
            if desired in pot["missing"] and point in interaction_poses(terrain, pot["pos"]):
                return True
        return False

    # ---------- geometry-derived transactions -------------------------

    def make_task(self, kind, target=None, item="", priority=10, tag="", wait=False, description=""):
        target = xy(target)
        return {
            "id": "%s|%s|%s|%s" % (kind, item, target, tag),
            "kind": kind,
            "target": target,
            "item": item,
            "priority": int(priority),
            "tag": str(tag or ""),
            "wait": bool(wait),
            "description": description or kind,
        }

    def route_length(self, obs, start, target, partner=None):
        _, path = feature_route(terrain_of(obs), start, target, {partner} if partner is not None else set())
        return len(path) if path is not None else 999

    def cooking_cost(self, obs, start, pot, partner):
        terrain = terrain_of(obs)
        best = 999
        for dish in feature_cells(terrain, "D"):
            _, first = feature_route(terrain, start, dish, {partner} if partner is not None else set())
            if first is None:
                continue
            for pose in interaction_poses(terrain, pot["pos"]):
                second = bfs(terrain, first[-1], pose, {partner} if partner is not None else set())
                if second is not None:
                    # one interaction at D and one at P
                    best = min(best, len(first) - 1 + 1 + len(second) - 1 + 1)
        return best

    def safe_floor(self, obs, start, partner, near=None):
        terrain = terrain_of(obs)
        if start is None:
            return None
        forbidden = interaction_cells(obs)
        if near is not None:
            forbidden.update(interaction_poses(terrain, near))
        candidates = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                point = (x, y)
                if not open_cell(terrain, point) or point in forbidden or point == partner:
                    continue
                path = bfs(terrain, start, point, {partner} if partner is not None else set())
                if path is None:
                    continue
                partner_gap = abs(point[0] - partner[0]) + abs(point[1] - partner[1]) if partner else 0
                near_gap = abs(point[0] - near[0]) + abs(point[1] - near[1]) if near else 0
                lane_penalty = 0
                if self.astra_lane in {"left", "right"}:
                    width = len(terrain[0]) if terrain and isinstance(terrain[0], (str, list, tuple)) else 0
                    is_left = point[0] < width / 2.0
                    lane_penalty = 0 if is_left == (self.astra_lane == "left") else 1
                candidates.append((lane_penalty, near_gap, -partner_gap, len(path), point))
        if not candidates:
            return start if start not in forbidden and start != partner else None
        candidates.sort()
        return candidates[0][-1]

    def build_tasks(self, obs, current, partner):
        """Return the legal supply candidates for the most complete pot."""
        pots = pot_states(obs)
        buildable = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]]
        if not buildable:
            return []
        buildable.sort(key=lambda p: (len(p["missing"]), self.route_length(obs, current, p["pos"], partner), p["pos"]))
        pot = buildable[0]
        desired = pot["missing"][0]
        result = []
        for name, point in counter_items(obs):
            if name == desired:
                result.append(self.make_task("take_counter", point, desired, 4, "staged",
                                        description="retrieve the staged %s for the next pot" % desired))
        symbol = "O" if desired == "onion" else "T"
        for source in feature_cells(terrain_of(obs), symbol):
            tag = "rescue" if self.supply_silence >= 6 and not self.partner_supply_claim(obs, desired) else "source"
            result.append(self.make_task("fetch_ingredient", source, desired, 5, tag,
                                    description="fetch one %s for a compatible pot" % desired))
        return result

    def options(self, obs):
        terrain = terrain_of(obs)
        me, partner_player = actors(obs)
        current, partner = xy(me.get("position")), xy(partner_player.get("position"))
        hand, partner_hand = held_of(me), held_of(partner_player)
        pots = pot_states(obs)
        result = []

        # Object ownership is a hard invariant and is evaluated before roles.
        if hand == "soup":
            for station in feature_cells(terrain, "S"):
                result.append(self.make_task("deliver", station, priority=0,
                                        description="serve the soup at the window"))
            return result or [self.make_task("hold", priority=20, wait=True, description="hold soup")]

        if hand == "dish":
            ready = [p for p in pots if p["ready"]]
            if ready:
                ready.sort(key=lambda p: (self.route_length(obs, current, p["pos"], partner), p["pos"]))
                for pot in ready:
                    result.append(self.make_task("pickup", pot["pos"], priority=0,
                                            description="take the ready soup with the held dish"))
                return result
            cooking = [p for p in pots if p["cooking"]]
            if cooking:
                cooking.sort(key=lambda p: (p.get("remaining", 999) if isinstance(p.get("remaining"), (int, float)) else 999, p["pos"]))
                target = cooking[0]["pos"]
                park = self.safe_floor(obs, current, partner, target)
                return [self.make_task("wait_dish", park, priority=3, tag=str(target), wait=True,
                                   description="keep the dish off the cooking pot approach")]
            return [self.make_task("wait_dish", self.safe_floor(obs, current, partner), priority=8,
                               wait=True, description="hold the dish safely")]

        if hand in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial", "full_idle"} and hand in p["missing"]]
            compatible.sort(key=lambda p: (self.route_length(obs, current, p["pos"], partner), p["pos"]))
            for pot in compatible:
                result.append(self.make_task("fill", pot["pos"], hand, 1,
                                        description="put the carried ingredient into the compatible pot"))
            if result:
                return result
            for point in free_counters(obs):
                result.append(self.make_task("put_counter", point, hand, 6,
                                        description="stage the carried ingredient on a free counter"))
            return result or [self.make_task("hold", priority=20, wait=True, description="hold ingredient")]

        # Ready soup protection.  A partner-held dish is a claim only while
        # physical progress supports it.  A stale claim creates a backup dish
        # candidate; Jev may confirm it but cannot create it.
        ready = [p for p in pots if p["ready"]]
        if ready:
            ready.sort(key=lambda p: (self.route_length(obs, current, p["pos"], partner), p["pos"]))
            urgent = ready[0]
            if self.backup_needed(obs, urgent):
                for station in feature_cells(terrain, "D"):
                    result.append(self.make_task("fetch_dish", station, priority=1, tag="backup",
                                            description="fetch a backup dish for a ready or stalled soup"))
                if result:
                    return result
            # While a healthy partner handoff is active, use an available
            # partial pot rather than standing in the gate.  This is the first
            # half of the scoped lease: ownership of the ready transaction
            # does not cancel independent production.
            if partner_hand in {"dish", "soup"}:
                supply = self.build_tasks(obs, current, partner)
                if supply:
                    return supply
                park = self.safe_floor(obs, current, partner, urgent["pos"])
                return [self.make_task("yield_ready", park, priority=4, tag="ready", wait=True,
                                   description="yield the ready-pot approach to the active partner")]
            for station in feature_cells(terrain, "D"):
                result.append(self.make_task("fetch_dish", station, priority=1, tag="ready",
                                        description="fetch a dish for the ready soup"))
            if result:
                return result

        # Explicit modern start is never delegated to a role or model.
        idle_full = [p for p in pots if p["phase"] == "full_idle"]
        if idle_full:
            idle_full.sort(key=lambda p: (self.route_length(obs, current, p["pos"], partner), p["pos"]))
            for pot in idle_full:
                result.append(self.make_task("start", pot["pos"], 2, "start",
                                        description="start the full idle pot with an empty-handed interaction"))
            return result

        supply = self.build_tasks(obs, current, partner)
        if supply:
            # A bounded courier lease can suppress only speculative supply,
            # never a partial pot that has been silent for six ticks and never
            # a counter ingredient already in hand.  The latter condition is
            # the supply-silence rescue that replaces 0018's permanent park.
            partial_exists = any(p["phase"] in {"empty", "partial"} and p["missing"] for p in pots)
            if self.lease_mode == "courier" and tick_of(obs) < self.lease_until and not partial_exists:
                pass
            else:
                return supply

        cooking = [p for p in pots if p["cooking"]]
        if cooking:
            cooking.sort(key=lambda p: (p.get("remaining", 999) if isinstance(p.get("remaining"), (int, float)) else 999, p["pos"]))
            pot = cooking[0]
            remaining = pot.get("remaining")
            cost = self.cooking_cost(obs, current, pot, partner)
            due = not isinstance(remaining, (int, float)) or remaining <= max(5, cost + 2)
            active = self.partner_handoff_active(obs, pot)
            if due and not active:
                for station in feature_cells(terrain, "D"):
                    result.append(self.make_task("fetch_dish", station, 3, "timed",
                                            description="fetch a dish timed to the cooking pot"))
                if result:
                    return result
            park = self.safe_floor(obs, current, partner, pot["pos"])
            return [self.make_task("park_cooking", park, 8, "cook", wait=True,
                               description="wait away from the pot gate while it cooks")]

        # No soup is urgent and no pot can currently accept an ingredient.  A
        # short dish seeding trip is useful; a short floor wait is safer than
        # repeatedly claiming a dispenser when the partner is supplying.
        if not partner_hand:
            stations = feature_cells(terrain, "D")
            if stations and self.lease_mode != "courier":
                for station in stations:
                    result.append(self.make_task("fetch_dish", station, 9, "seed",
                                            description="stage a dish for the next cooking phase"))
        if result:
            return result
        park = self.safe_floor(obs, current, partner)
        return [self.make_task("idle", park, 15, "idle", wait=True, description="wait in a safe floor cell")]

    # ---------- phase lease, task validity, and Jev interface ----------

    def task_key(self, task):
        if not task:
            return None
        return (task.get("kind"), xy(task.get("target")), task.get("item", ""), task.get("tag", ""))

    def task_limit(self, kind):
        return {
            "deliver": 46,
            "pickup": 24,
            "fetch_dish": 42,
            "fetch_ingredient": 34,
            "take_counter": 28,
            "put_counter": 18,
            "fill": 16,
            "start": 16,
            "wait_dish": 7,
            "yield_ready": 7,
            "park_cooking": 8,
            "idle": 7,
            "hold": 7,
        }.get(kind, 24)

    def current_is_valid(self, obs, options):
        if not self.task:
            return False
        key = self.task_key(self.task)
        if not any(self.task_key(option) == key for option in options):
            return False
        age = tick_of(obs) - self.task_since
        if age > self.task_limit(self.task.get("kind", "")):
            return False
        if self.task.get("wait") and age >= 7:
            return False
        if self.blocked >= 8 and held_of(actors(obs)[0]) != "soup":
            return False
        return True

    def route_rank(self, obs, candidate):
        me, partner = actors(obs)
        start, other = xy(me.get("position")), xy(partner.get("position"))
        target = candidate.get("target")
        if target is None:
            return 0
        if candidate.get("kind") in {"wait_dish", "yield_ready", "park_cooking", "idle"}:
            path = bfs(terrain_of(obs), start, target, {other} if other else set())
            return len(path) if path is not None else 999
        return self.route_length(obs, start, target, other)

    def code_choice(self, obs, options):
        if not options:
            return self.make_task("idle", priority=20, wait=True, description="safe wait")
        scored = []
        for option in options:
            lane_penalty = 0
            target = option.get("target")
            terrain = terrain_of(obs)
            if target is not None and self.astra_lane in {"left", "right"}:
                width = len(terrain[0]) if terrain and isinstance(terrain[0], (str, list, tuple)) else 0
                is_left = target[0] < width / 2.0
                lane_penalty = 0 if is_left == (self.astra_lane == "left") else 1
            # Priority dominates.  Lane and route only resolve equal urgency.
            scored.append((option.get("priority", 20), lane_penalty, self.route_rank(obs, option), option.get("id", ""), option))
        scored.sort(key=lambda row: row[:4])
        return scored[0][-1].copy()

    def compact_state(self, obs, options):
        me, partner = actors(obs)
        pots = pot_states(obs)
        return safe_json({
            "game": "Overcooked-AI symbolic grid kitchen",
            "tick": tick_of(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "phase_lease": {"mode": self.lease_mode, "expires": self.lease_until,
                            "supply_silence": self.supply_silence, "assist_used": self.assist_used},
            "self": {"position": xy(me.get("position")), "facing": xy(me.get("orientation")),
                     "held": held_of(me)},
            "partner": {"position": xy(partner.get("position")), "facing": xy(partner.get("orientation")),
                        "held": held_of(partner), "last_action": obs.get("partner_action_last", ""),
                        "still": self.partner_still, "route_not_improving": self.partner_bad},
            "orders": recipes_of(obs),
            "pots": [{"position": p["pos"], "phase": p["phase"], "ingredients": p["ingredients"],
                      "missing": p["missing"], "remaining": p["remaining"]} for p in pots],
            "transaction": {"kind": self.task.get("kind", "") if self.task else "",
                            "age": max(0, tick_of(obs) - self.task_since) if self.task else 0,
                            "blocked": self.blocked},
            "candidates": [{"id": x.get("id"), "kind": x.get("kind"), "priority": x.get("priority"),
                            "description": x.get("description")} for x in options[:10]],
        })

    def launch_astra(self, obs, options):
        if self.astra_started or getattr(self.ctx, "system2", object()) is None:
            self.astra_started = True
            return
        self.astra_started = True
        state = self.compact_state(obs, options)
        prompt = (
            "You are a slow strategic reviewer for a realtime symbolic Overcooked-AI kitchen. "
            "The executor owns all recipe arithmetic, BFS, legal interaction checks, and primitive "
            "actions. Design only a short policy hint for a bounded phase lease. A courier lease "
            "protects a ready-soup handoff for a few ticks, but must yield to a partial pot that is "
            "silent or to an actually urgent ready soup. Return ONLY JSON with lane left/right/nearest "
            "and rescue conservative/aggressive. Do not output directions or a plan.\n" +
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
            value = future.result()
            text = value if isinstance(value, str) else getattr(value, "text", "")
            match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
            if not isinstance(parsed, dict):
                return
            if parsed.get("lane") in {"left", "right", "nearest"}:
                self.astra_lane = parsed["lane"]
            if parsed.get("rescue") in {"conservative", "aggressive"}:
                self.astra_rescue = parsed["rescue"]
            self.log("astra_hint", tick=tick_of(obs), lane=self.astra_lane, rescue=self.astra_rescue)
        except Exception:
            return

    def launch_jev(self, obs, options, reason):
        if self.system1_off or self.jev_future is not None or not options:
            return
        now = tick_of(obs)
        if now - self.last_jev_tick < 18:
            return
        self.last_jev_tick = now
        self.jev_epoch = self.epoch
        state = self.compact_state(obs, options)
        criteria = {str(option.get("id")): str(option.get("description", option.get("kind", "candidate")))
                    for option in options[:10] if option.get("id")}
        if not criteria:
            return
        questions = {
            "event": Choice(
                "Choose one id from observation.candidates. Prefer a lower-priority-number "
                "delivery, ready pickup, explicit start, or compatible fill. Never invent an id, "
                "direction, target, count, or arithmetic result.",
                criteria,
            ),
            "handoff": Noul(
                "Is the partner's current dish/soup handoff physically progressing toward an urgent "
                "pot or serving window? Use only partner.held, partner.still, partner.route_not_improving, "
                "partner.facing, and pots. A stalled partner with a ready or nearly-ready pot is false; "
                "a partner already facing or improving toward the target is true.",
                {"true": "The partner is making a live handoff and should not be contested.",
                 "false": "There is no live handoff, or the urgent handoff is stalled."},
            ),
        }

        def ask():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(state, questions, timeout_s=1.2)
            except Exception:
                return {}

        try:
            self.jev_future = self.ctx.spawn(ask)
            self.log("jev_request", tick=now, reason=reason, epoch=self.epoch)
        except Exception:
            self.jev_future = None

    def poll_jev(self, obs, options):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict) or self.jev_epoch != self.epoch:
            return
        valid = {x.get("id"): x for x in options}
        event = answers.get("event")
        selected = getattr(event, "value", None)
        probabilities = getattr(event, "probabilities", {}) or {}
        try:
            probability = float(probabilities.get(selected, 0.0)) if isinstance(probabilities, dict) else 0.0
        except Exception:
            probability = 0.0
        # Fake/preflight answers often have no distribution.  The closed-world
        # value is still safe because it is checked against current options.
        if selected in valid and (probability >= 0.52 or not probabilities):
            best = self.code_choice(obs, options)
            chosen = valid[selected]
            if chosen.get("priority", 20) <= best.get("priority", 20) + 1:
                self.hint_id = selected
                self.hint_until = tick_of(obs) + 5
                self.log("jev_event", tick=tick_of(obs), value=selected, probability=probability)
        handoff = answers.get("handoff")
        try:
            handoff_probability = float(getattr(handoff, "value", 0.0))
        except Exception:
            handoff_probability = 0.0
        if handoff_probability >= 0.78:
            # A positive answer only extends a short yield, never the task
            # itself.  The next phase observation can revoke it.
            self.lease_mode = "courier"
            self.lease_until = max(self.lease_until, tick_of(obs) + 6)
            self.log("jev_handoff", tick=tick_of(obs), value="active", probability=handoff_probability)
        elif handoff_probability <= 0.22:
            self.lease_mode = "supply"
            self.lease_until = tick_of(obs)
            self.log("jev_handoff", tick=tick_of(obs), value="stalled", probability=handoff_probability)

    def choose_task(self, obs, options):
        if self.current_is_valid(obs, options):
            return False
        chosen = None
        if self.hint_id and tick_of(obs) <= self.hint_until:
            for option in options:
                if option.get("id") == self.hint_id:
                    best = self.code_choice(obs, options)
                    if option.get("priority", 20) <= best.get("priority", 20) + 1:
                        chosen = option.copy()
                    break
        if chosen is None:
            chosen = self.code_choice(obs, options)
        self.task = chosen
        self.task_since = tick_of(obs)
        self.epoch += 1
        self.blocked = 0
        self.escape_left = 0
        self.hint_id = None
        if chosen.get("kind") in {"fetch_ingredient", "take_counter"}:
            self.assist_used = True
        self.log("lease_task", tick=self.task_since, kind=chosen.get("kind"),
                 target=chosen.get("target"), priority=chosen.get("priority"),
                 mode=self.lease_mode, epoch=self.epoch)
        return True

    # ---------- legality and one-tick motor ----------------------------

    def can_interact(self, obs, task):
        me, _ = actors(obs)
        hand = held_of(me)
        kind = task.get("kind")
        target = xy(task.get("target"))
        terrain = terrain_of(obs)
        pots = pot_states(obs)
        pot = next((p for p in pots if p["pos"] == target), None)
        if target is None:
            return False
        if kind == "deliver":
            return hand == "soup" and tile(terrain, target) == "S"
        if kind == "fetch_dish":
            return not hand and tile(terrain, target) == "D"
        if kind == "fetch_ingredient":
            return not hand and tile(terrain, target) in {"O", "T"}
        if kind == "take_counter":
            return not hand and any(name == task.get("item") and point == target for name, point in counter_items(obs))
        if kind == "put_counter":
            return hand in INGREDIENTS and target not in occupied_cells(obs)
        if kind == "fill":
            return bool(pot and hand in INGREDIENTS and pot["phase"] in {"empty", "partial", "full_idle"}
                        and hand in pot["missing"])
        if kind == "start":
            return bool(pot and not hand and pot["phase"] == "full_idle")
        if kind == "pickup":
            return bool(pot and hand == "dish" and pot["ready"])
        return False

    def escape_action(self, obs, goal=None):
        me, partner = actors(obs)
        current, other = xy(me.get("position")), xy(partner.get("position"))
        if current is None:
            return self.action(self.noop)
        terrain = terrain_of(obs)
        predicted = partner_next(obs)
        choices = []
        for nxt, delta in neighbours(current):
            if not open_cell(terrain, nxt) or nxt in {other, predicted}:
                continue
            # Empty-handed escapes should leave a feature approach rather than
            # claim a second gate.  A carried object may use any safe floor.
            if not held_of(me) and nxt in interaction_cells(obs):
                continue
            goal_gap = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) if goal else 0
            partner_gap = abs(nxt[0] - other[0]) + abs(nxt[1] - other[1]) if other else 0
            reverse = 1 if nxt == self.last_position else 0
            choices.append((goal_gap, reverse, -partner_gap, nxt, delta))
        if not choices:
            return self.action(self.noop)
        choices.sort(key=lambda value: value[:4])
        return self.action(STEP_NAME.get(choices[0][4], self.noop))

    def motor(self, obs):
        me, partner = actors(obs)
        current, other = xy(me.get("position")), xy(partner.get("position"))
        if not self.task:
            return self.action(self.noop)

        if self.blocked >= 5 and self.escape_left <= 0:
            self.escape_left = 3
            self.escape_goal = xy(self.task.get("target"))
            self.log("escape", tick=tick_of(obs), blocked=self.blocked, kind=self.task.get("kind"))
        if self.escape_left > 0 and held_of(me) != "soup":
            self.escape_left -= 1
            return self.escape_action(obs, self.escape_goal)

        kind = self.task.get("kind")
        target = xy(self.task.get("target"))
        if target is None or kind in {"hold"}:
            return self.action(self.noop)
        terrain = terrain_of(obs)
        partner_next_point = partner_next(obs)

        if kind in {"wait_dish", "yield_ready", "park_cooking", "idle"}:
            path = bfs(terrain, current, target, {other} if other is not None and other != target else set())
            if path is not None and len(path) > 1:
                nxt = path[1]
                if nxt in {other, partner_next_point}:
                    self.blocked += 1
                    return self.action(self.noop if self.blocked <= 2 else self.escape_action(obs, target))
                return self.action(STEP_NAME.get((nxt[0] - current[0], nxt[1] - current[1]), self.noop))
            return self.action(self.noop)

        pose, path = feature_route(terrain, current, target, {other} if other is not None and other != target else set())
        if pose is None or path is None:
            self.blocked += 1
            return self.escape_action(obs, target)
        if len(path) > 1:
            nxt = path[1]
            # Prevent both moving through the same one-cell gate or swapping.
            if nxt == other or nxt == partner_next_point or (partner_next_point == current and nxt == other):
                self.blocked += 1
                return self.action(self.noop if self.blocked <= 2 else self.escape_action(obs, pose))
            self.blocked = max(0, self.blocked - 1)
            return self.action(STEP_NAME.get((nxt[0] - current[0], nxt[1] - current[1]), self.noop))

        desired = (target[0] - current[0], target[1] - current[1])
        facing = xy(me.get("orientation"))
        if desired in STEP_NAME and facing != desired:
            # Directional actions both turn and attempt one tile; the feature
            # is blocked, so this is the canonical orientation action.
            return self.action(STEP_NAME[desired])
        if self.can_interact(obs, self.task):
            return self.action("interact")
        return self.action(self.noop)

    # ---------- realtime loop ------------------------------------------

    def run(self):
        while not self.ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self.update_motion(obs)
            self.observe_phase(obs)
            self.update_partner_progress(obs)
            self.poll_astra(obs)

            options = self.options(obs)
            if not self.astra_started:
                self.launch_astra(obs, options)
            self.poll_jev(obs, options)

            changed = self.choose_task(obs, options)
            if changed:
                self.launch_jev(obs, options, "phase_or_transaction_boundary")
            elif self.blocked >= 3:
                self.launch_jev(obs, options, "traffic_stall")
            elif tick_of(obs) - self.last_jev_tick >= 48:
                self.launch_jev(obs, options, "heartbeat")

            action = self.motor(obs)
            try:
                if self.ctx.done:
                    return
                self.ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                # A malformed adaptor action should never be emitted by this
                # module, but retain the contract's safe no-op recovery.
                try:
                    if not self.ctx.done:
                        self.ctx.act(self.action(self.noop))
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
        BoundedKitchen(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
