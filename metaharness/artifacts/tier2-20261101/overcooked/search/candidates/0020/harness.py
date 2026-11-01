"""Candidate 0020: an event-calendar controller with a joint action lattice.

The controller does not treat the partner as a static obstacle or as an
unbounded claim on a gate.  It infers the partner's current medium-level
intent from the symbolic state, forecasts a short route to that intent, and
chooses the first primitive of a small joint occupancy lattice.  Production
is scheduled as a sequence of interaction events (fill/start/dish/pickup/
serve), with a committed event baton surviving ordinary movement observations.
Astra writes one broad protocol; Jev only breaks ties between code-generated
next events.
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
ACTION_FOR = {delta: action for action, delta in DIRECTIONS.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}
OBJECTS = INGREDIENTS | {"dish", "soup"}


def point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        if isinstance(value[0], bool) or isinstance(value[1], bool):
            return None
        return int(value[0]), int(value[1])
    except Exception:
        return None


def clean(value):
    """Make a strict JSON-safe value for Jev, including hostile stand-ins."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def state_of(obs):
    if isinstance(obs, dict) and isinstance(obs.get("state"), dict):
        return obs["state"]
    return {}


def terrain_of(obs):
    value = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return value if isinstance(value, list) else []


def tick_of(obs):
    try:
        return int(obs.get("tick", state_of(obs).get("timestep", 0)))
    except Exception:
        return 0


def cell(terrain, p):
    p = point(p)
    if p is None:
        return "X"
    x, y = p
    if y < 0 or y >= len(terrain):
        return "X"
    try:
        row = terrain[y]
        if isinstance(row, (str, list, tuple)) and 0 <= x < len(row):
            return row[x]
    except Exception:
        pass
    return "X"


def walkable(terrain, p):
    return cell(terrain, p) not in FEATURES and cell(terrain, p) != "X"


def feature_cells(terrain, symbol):
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                result.append((x, y))
    return result


def neighbours(p):
    x, y = p
    for delta in DIRECTIONS.values():
        yield (x + delta[0], y + delta[1]), delta


def bfs(terrain, start, goal, forbidden=()):
    start, goal = point(start), point(goal)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    blocked = {point(v) for v in forbidden if point(v) is not None}
    blocked.discard(start)
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        for nxt, _ in neighbours(current):
            if nxt in previous or nxt in blocked or not walkable(terrain, nxt):
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
    target = point(target)
    if target is None:
        return []
    result = []
    for delta in DIRECTIONS.values():
        pose = (target[0] - delta[0], target[1] - delta[1])
        if pose not in result and walkable(terrain, pose):
            result.append(pose)
    return result


def route_to(terrain, start, target, forbidden=()):
    """Shortest route to a feature/counter interaction pose."""
    start, target = point(start), point(target)
    if start is None or target is None:
        return None, None
    blocked = set(forbidden)
    choices = []
    for pose in interaction_poses(terrain, target):
        # Keep an occupied interaction pose blocked in the first pass.  The
        # alternate approach is often the whole solution in a ring; allowing
        # the partner's cell as a goal here makes the lattice choose waiting
        # forever rather than routing to the other legal side.
        local = set(blocked)
        path = bfs(terrain, start, pose, local)
        if path is not None:
            choices.append((len(path), pose, path))
    if choices:
        choices.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
        return choices[0][1], choices[0][2]
    # A moving partner is not a permanent wall.  This fallback queues the
    # interaction while the joint lattice handles the immediate conflict.
    choices = []
    for pose in interaction_poses(terrain, target):
        path = bfs(terrain, start, pose)
        if path is not None:
            choices.append((len(path), pose, path))
    if choices:
        choices.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
        return choices[0][1], choices[0][2]
    return None, None


def object_name(obj):
    return str(obj.get("name", "") or "").lower() if isinstance(obj, dict) else ""


def object_position(obj):
    return point(obj.get("position")) if isinstance(obj, dict) else None


def held_name(player):
    held = player.get("held_object") if isinstance(player, dict) else None
    return object_name(held) if isinstance(held, dict) else ""


def players_of(obs):
    players = state_of(obs).get("players", [])
    return players if isinstance(players, list) else []


def actors(obs):
    players = players_of(obs)
    try:
        index = int(obs.get("controlled_agent_index", 0))
    except Exception:
        index = 0
    if index < 0 or index >= len(players):
        index = 0
    other = 1 - index if len(players) >= 2 and index in (0, 1) else index
    me = players[index] if index < len(players) and isinstance(players[index], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, partner, index


def objects_of(obs):
    objects = state_of(obs).get("objects", [])
    return objects if isinstance(objects, list) else []


def ingredient_names(obj):
    if not isinstance(obj, dict):
        return []
    raw = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(raw, list):
        return []
    result = []
    for value in raw:
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


def missing_for(have, recipes):
    counts = Counter(have)
    best_recipe, best_missing = [], None
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
            best_recipe, best_missing = list(recipe), missing
    return best_recipe or [], best_missing if best_missing is not None else []


def pot_info(obs):
    terrain = terrain_of(obs)
    soups = {}
    for obj in objects_of(obs):
        p = object_position(obj)
        if p is not None and object_name(obj) == "soup":
            soups[p] = obj
    result = []
    for target in feature_cells(terrain, "P"):
        soup = soups.get(target)
        ingredients = ingredient_names(soup)
        recipe, missing = missing_for(ingredients, recipes_of(obs))
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


def counter_cells(obs):
    terrain = terrain_of(obs)
    occupied = {object_position(obj) for obj in objects_of(obs) if object_position(obj) is not None}
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            p = (x, y)
            if value == "X" and p not in occupied and any(walkable(terrain, q) for q, _ in neighbours(p)):
                result.append(p)
    return result


def counter_items(obs):
    terrain = terrain_of(obs)
    result = []
    for obj in objects_of(obs):
        p, name = object_position(obj), object_name(obj)
        if p is not None and name in OBJECTS and cell(terrain, p) == "X":
            result.append((name, p))
    return result


def parse_json_object(text):
    if not isinstance(text, str):
        return {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


class EventCalendar:
    """Event-baton scheduler and short-horizon joint motor."""

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
        self.task_age_limit = 50
        self.task_hint = None
        self.hint_tick = -999
        self.blocked = 0
        self.last_position = None
        self.last_action = self.noop
        self.previous_partner = None
        self.partner_still = 0
        self.partner_bad_route = 0
        self.partner_distance = None
        self.partner_goal_key = None
        self.partner_route = []
        self.yield_ticks = 0
        self.route_flip = 0

        self.astra_future = None
        self.astra_started = False
        self.protocol = {"focus": "balanced", "traffic": "joint", "dish": "timed"}

        self.jev_future = None
        self.jev_epoch = -1
        self.last_jev_tick = -999
        self.system1_off = getattr(ctx, "system1", object()) is None

    # ---- resilient plumbing -------------------------------------------
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
        return sorted(self.actions)[0] if self.actions else "stay"

    def _partner_plan(self, obs, pots=None):
        """Infer a partner target, but never assume that possession means progress."""
        terrain = terrain_of(obs)
        me, partner, _ = actors(obs)
        ppos = point(partner.get("position"))
        held = held_name(partner)
        pots = pots if pots is not None else pot_info(obs)
        target = None
        kind = "none"
        if held == "soup":
            targets = feature_cells(terrain, "S")
            target = targets[0] if targets else None
            kind = "serve"
        elif held == "dish":
            ready = [p["pos"] for p in pots if p["ready"]]
            cooking = [p for p in pots if p["cooking"]]
            if ready:
                target = sorted(ready)[0]
                kind = "pickup"
            elif cooking:
                target = min(cooking, key=lambda p: (p["remaining"] if isinstance(p["remaining"], (int, float)) else 999, p["pos"]))["pos"]
                kind = "pickup"
        elif held in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial"} and held in p["missing"]]
            if compatible:
                target = min(compatible, key=lambda p: (len(p["missing"]), p["pos"]))["pos"]
                kind = "fill"
        else:
            ready = [p for p in pots if p["ready"]]
            if ready:
                target = feature_cells(terrain, "D")[0] if feature_cells(terrain, "D") else None
                kind = "get_dish"
            else:
                build = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]]
                if build:
                    wanted = min(build, key=lambda p: (len(p["missing"]), p["pos"]))["missing"][0]
                    sources = feature_cells(terrain, "O" if wanted == "onion" else "T")
                    if sources:
                        target = min(sources, key=lambda s: abs(s[0] - ppos[0]) + abs(s[1] - ppos[1]) if ppos else 999)
                        kind = "get_ingredient"
        pose, path = route_to(terrain, ppos, target) if ppos is not None and target is not None else (None, None)
        goal_key = (kind, target)
        distance = len(path) - 1 if path else (0 if target is None else 999)
        return {
            "kind": kind,
            "held": held,
            "position": ppos,
            "target": target,
            "pose": pose,
            "path": path or [],
            "distance": distance,
            "goal_key": goal_key,
        }

    def update_partner(self, obs, pots):
        plan = self._partner_plan(obs, pots)
        pos_now = plan["position"]
        signature = (pos_now, plan["held"], plan["goal_key"])
        if self.previous_partner is None:
            self.previous_partner = signature
            self.partner_still = 0
            self.partner_bad_route = 0
            self.partner_distance = plan["distance"]
            self.partner_goal_key = plan["goal_key"]
        else:
            old_pos, old_held, old_goal = self.previous_partner
            if signature != self.previous_partner:
                self.partner_still = 0 if pos_now != old_pos else self.partner_still + 1
                if plan["goal_key"] != self.partner_goal_key:
                    self.partner_bad_route = 0
                    self.partner_distance = plan["distance"]
                    self.partner_goal_key = plan["goal_key"]
                elif self.partner_distance is not None and plan["distance"] < self.partner_distance:
                    self.partner_bad_route = 0
                    self.partner_distance = plan["distance"]
                else:
                    self.partner_bad_route += 1
                    self.partner_distance = plan["distance"]
                self.previous_partner = signature
            else:
                self.partner_still += 1
                self.partner_bad_route += 1
        self.partner_route = plan["path"]
        return plan

    # ---- Astra protocol author ----------------------------------------
    def _astra_prompt(self, obs, candidates, partner):
        me, other, controlled = actors(obs)
        return (
            "You are Astra, a slow protocol author for a realtime symbolic "
            "Overcooked-AI kitchen. The executor owns recipes, exact interaction "
            "legality, shortest routes, and primitive actions. Review this initial "
            "snapshot and return ONLY JSON with focus=produce|deliver|balanced, "
            "traffic=joint|partner_first|self_first, and dish=timed|early. "
            "Do not output directions or task ids. Choose a protocol that keeps "
            "the two chefs complementary rather than making them share a gate.\n" +
            json.dumps(clean({
                "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
                "controlled_index": controlled,
                "terrain": terrain_of(obs),
                "self": {"position": point(me.get("position")), "held": held_name(me)},
                "partner": {"position": point(other.get("position")), "held": held_name(other),
                            "intent": partner.get("partner_task", "") if isinstance(partner, dict) else ""},
                "pots": [{"position": p["pos"], "phase": p["phase"], "missing": p["missing"],
                          "remaining": p["remaining"]} for p in pot_info(obs)],
                "candidates": [{"kind": c.get("kind"), "target": c.get("target"),
                                "priority": c.get("priority")} for c in candidates[:8]],
            }), separators=(",", ":"))
        )

    def start_astra(self, obs, candidates, partner):
        if self.astra_started:
            return
        self.astra_started = True
        if getattr(self.ctx, "system2", object()) is None or self.ctx.done:
            return
        prompt = self._astra_prompt(obs, candidates, partner)

        def work():
            if self.ctx.done:
                return ""
            try:
                result = self.ctx.think(prompt, timeout_s=90.0)
                return getattr(result, "text", "")
            except Exception:
                return ""

        try:
            self.astra_future = self.ctx.spawn(work)
        except Exception:
            self.astra_future = None

    def poll_astra(self, obs):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            text = future.result()
            text = text if isinstance(text, str) else getattr(text, "text", "")
            parsed = parse_json_object(text)
            if parsed.get("focus") in {"produce", "deliver", "balanced"}:
                self.protocol["focus"] = parsed["focus"]
            if parsed.get("traffic") in {"joint", "partner_first", "self_first"}:
                self.protocol["traffic"] = parsed["traffic"]
            if parsed.get("dish") in {"timed", "early"}:
                self.protocol["dish"] = parsed["dish"]
            self.log("astra_protocol", tick=tick_of(obs), protocol=dict(self.protocol))
        except Exception:
            return

    # ---- event candidates ---------------------------------------------
    def make_task(self, kind, target=None, item="", priority=10, description="", wait=False, tag=""):
        target = point(target)
        suffix = "_".join(str(v) for v in target) if target is not None else "none"
        return {
            "id": "%s_%s_%s_%s" % (kind, item or "none", suffix, tag or "x"),
            "kind": kind,
            "target": target,
            "item": item,
            "priority": int(priority),
            "description": description or kind,
            "wait": bool(wait),
            "tag": str(tag or ""),
        }

    def safe_floor(self, obs, current, partner, avoid=()):
        terrain = terrain_of(obs)
        if current is None:
            return None
        gates = set()
        for symbol in FEATURES:
            for target in feature_cells(terrain, symbol):
                gates.update(interaction_poses(terrain, target))
        gates.update(point(v) for v in avoid if point(v) is not None)
        candidates = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                p = (x, y)
                if not walkable(terrain, p) or p in gates or p == partner:
                    continue
                path = bfs(terrain, current, p, {partner} if partner else set())
                if path is None:
                    continue
                gap = abs(p[0] - partner[0]) + abs(p[1] - partner[1]) if partner else 0
                candidates.append((-gap, len(path), p))
        if not candidates:
            return current if current not in gates and current != partner else None
        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        return candidates[0][2]

    def dish_due(self, obs, pot, current, partner_pos):
        if not pot or not pot.get("cooking"):
            return False
        remaining = pot.get("remaining")
        if not isinstance(remaining, (int, float)):
            return True
        dishes = feature_cells(terrain_of(obs), "D")
        best = 999
        for dish in dishes:
            _, first = route_to(terrain_of(obs), current, dish, {partner_pos} if partner_pos else set())
            if first is None:
                continue
            for pose in interaction_poses(terrain_of(obs), pot["pos"]):
                second = bfs(terrain_of(obs), first[-1], pose)
                if second is not None:
                    best = min(best, len(first) - 1 + 1 + len(second) - 1 + 1)
        return remaining <= max(5, best + 2)

    def candidates(self, obs, partner_plan):
        terrain = terrain_of(obs)
        me, partner, _ = actors(obs)
        current, ppos = point(me.get("position")), point(partner.get("position"))
        held, pheld = held_name(me), held_name(partner)
        pots = pot_info(obs)
        result = []

        if held == "soup":
            for serving in feature_cells(terrain, "S"):
                result.append(self.make_task("serve", serving, priority=0, description="serve carried soup"))
            return result or [self.make_task("hold", priority=20, wait=True, description="hold soup safely")]

        if held == "dish":
            ready = [p for p in pots if p["ready"]]
            if ready:
                for pot in ready:
                    result.append(self.make_task("pickup", pot["pos"], priority=0, description="pick up ready soup with dish"))
                return result
            cooking = [p for p in pots if p["cooking"]]
            if cooking:
                pot = min(cooking, key=lambda p: (p["remaining"] if isinstance(p["remaining"], (int, float)) else 999, p["pos"]))
                floor = self.safe_floor(obs, current, ppos, interaction_poses(terrain, pot["pos"]))
                result.append(self.make_task("wait_dish", floor, priority=3, wait=True,
                                        description="stage dish off the pot approach", tag=str(pot["pos"])))
            return result or [self.make_task("hold", priority=20, wait=True, description="hold dish safely")]

        if held in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial"} and held in p["missing"]]
            compatible.sort(key=lambda p: (len(p["missing"]), p["pos"]))
            for pot in compatible:
                result.append(self.make_task("fill", pot["pos"], held, priority=1,
                                        description="put held ingredient in compatible pot"))
            if result:
                return result
            for counter in counter_cells(obs):
                result.append(self.make_task("put_counter", counter, held, priority=7,
                                        description="stage held ingredient on a free counter"))
            return result or [self.make_task("hold", priority=20, wait=True, description="hold ingredient safely")]

        # Empty hand: first honor a ready soup unless the partner is clearly
        # carrying soup to the window or has an improving dish route to it.
        ready = [p for p in pots if p["ready"]]
        if ready:
            urgent = min(ready, key=lambda p: p["pos"])
            partner_owns_ready = (
                pheld == "soup" and partner_plan.get("kind") == "serve"
            ) or (
                pheld == "dish" and partner_plan.get("target") == urgent["pos"]
                and self.partner_bad_route < 5 and self.partner_still < 8
            )
            if not partner_owns_ready:
                for dish in feature_cells(terrain, "D"):
                    result.append(self.make_task("get_dish", dish, priority=0,
                                            description="fetch dish for ready soup", tag="ready"))
                if result:
                    return result

        # Explicit start is an event boundary, never a side effect of filling.
        for pot in sorted((p for p in pots if p["phase"] == "full_idle"), key=lambda p: p["pos"]):
            result.append(self.make_task("start", pot["pos"], priority=1,
                                    description="start full idle pot"))
        if result:
            return result

        buildable = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]]
        buildable.sort(key=lambda p: (len(p["missing"]), p["pos"]))
        if buildable:
            target_pot = buildable[0]
            wanted = target_pot["missing"][0]
            for name, counter in counter_items(obs):
                if name == wanted:
                    result.append(self.make_task("get_counter", counter, wanted, priority=3,
                                            description="retrieve matching staged ingredient"))
            symbol = "O" if wanted == "onion" else "T"
            for source in feature_cells(terrain, symbol):
                result.append(self.make_task("get_ingredient", source, wanted, priority=4,
                                        description="fetch next recipe ingredient"))
            if result:
                return result

        cooking = [p for p in pots if p["cooking"]]
        if cooking:
            pot = min(cooking, key=lambda p: (p["remaining"] if isinstance(p["remaining"], (int, float)) else 999, p["pos"]))
            partner_has_dish = pheld == "dish" and partner_plan.get("target") == pot["pos"]
            if (not partner_has_dish or self.partner_bad_route >= 5 or self.partner_still >= 8) and self.dish_due(obs, pot, current, ppos):
                for dish in feature_cells(terrain, "D"):
                    result.append(self.make_task("get_dish", dish, priority=2,
                                            description="fetch dish timed to cooking completion", tag="timed"))
                if result:
                    return result
            floor = self.safe_floor(obs, current, ppos, interaction_poses(terrain, pot["pos"]))
            result.append(self.make_task("park", floor, priority=12, wait=True,
                                    description="clear feature approaches while soup cooks", tag="cook"))
            return result

        floor = self.safe_floor(obs, current, ppos)
        return [self.make_task("idle", floor, priority=20, wait=True, description="wait in safe floor cell")]

    # ---- joint event selection ----------------------------------------
    def task_key(self, task):
        if not task:
            return None
        return (task.get("kind"), point(task.get("target")), task.get("item", ""), task.get("tag", ""))

    def same_task(self, left, right):
        return self.task_key(left) == self.task_key(right)

    def target_route(self, obs, task, block_partner=True):
        me, partner, _ = actors(obs)
        start, ppos = point(me.get("position")), point(partner.get("position"))
        target = point(task.get("target"))
        if target is None:
            return start, [start] if start is not None else None
        if task.get("kind") in {"wait_dish", "park", "idle"}:
            path = bfs(terrain_of(obs), start, target, {ppos} if block_partner and ppos else set())
            return target, path
        return route_to(terrain_of(obs), start, target, {ppos} if block_partner and ppos else set())

    def partner_conflict_cost(self, obs, task, partner_plan):
        _, path = self.target_route(obs, task, True)
        if not path:
            return 999
        corridor = set(path[: min(7, len(path))])
        ppath = set(partner_plan.get("path", [])[:7])
        return len(corridor & ppath)

    def code_choice(self, obs, options, partner_plan):
        if not options:
            return self.make_task("idle", priority=20, wait=True, description="safe idle")
        ranked = []
        for option in options:
            _, path = self.target_route(obs, option, True)
            length = len(path) if path is not None else 999
            conflict = self.partner_conflict_cost(obs, option, partner_plan)
            kind = option.get("kind")
            # Exact production ordering is hard.  The lattice cost only breaks
            # ties and chooses among equal source/counter routes.
            conflict_weight = 5 if kind in {"serve", "pickup", "start", "fill"} else 9
            if self.protocol.get("traffic") == "partner_first":
                conflict_weight += 3
            if self.protocol.get("traffic") == "self_first":
                conflict_weight = max(1, conflict_weight - 2)
            ranked.append((int(option.get("priority", 20)), conflict * conflict_weight,
                           length, self.route_flip if option.get("target") else 0,
                           str(option.get("id", "")), option))
        ranked.sort(key=lambda item: item[:5])
        return ranked[0][5].copy()

    def task_valid(self, obs, options):
        if not self.task:
            return False
        if not any(self.same_task(self.task, option) for option in options):
            return False
        age = max(0, tick_of(obs) - self.task_started)
        return age <= self.task_age_limit

    def commit(self, obs, options, partner_plan, reason):
        chosen = None
        if self.task_hint and tick_of(obs) - self.hint_tick <= 8:
            for option in options:
                if option.get("id") == self.task_hint:
                    best = self.code_choice(obs, options, partner_plan)
                    if int(option.get("priority", 20)) <= int(best.get("priority", 20)) + 1:
                        chosen = option.copy()
                    break
        if chosen is None:
            chosen = self.code_choice(obs, options, partner_plan)
        me, _, _ = actors(obs)
        pose, path = self.target_route(obs, chosen, True)
        chosen["pose"] = pose
        chosen["initial_route"] = len(path) if path else 999
        self.task = chosen
        self.task_started = tick_of(obs)
        self.task_age_limit = 18 if chosen.get("wait") else 55
        self.task_epoch += 1
        self.blocked = 0
        self.task_hint = None
        self.log("event_commit", tick=self.task_started, reason=reason,
                 kind=chosen.get("kind"), target=chosen.get("target"),
                 pose=pose, epoch=self.task_epoch)

    # ---- Jev interface -------------------------------------------------
    def compact_state(self, obs, options, partner_plan, reason):
        me, partner, controlled = actors(obs)
        pots = pot_info(obs)
        return clean({
            "game": "Overcooked-AI symbolic kitchen",
            "tick": tick_of(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "controlled_index": controlled,
            "self": {"position": point(me.get("position")), "facing": point(me.get("orientation")),
                     "held": held_name(me)},
            "partner": {"position": point(partner.get("position")), "facing": point(partner.get("orientation")),
                        "held": held_name(partner), "intent": partner_plan.get("kind"),
                        "target": partner_plan.get("target"),
                        "route_distance": partner_plan.get("distance"),
                        "still_ticks": self.partner_still,
                        "route_not_improving": self.partner_bad_route,
                        "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else ""},
            "pots": [{"position": p["pos"], "phase": p["phase"], "missing": p["missing"],
                      "cook_remaining": p["remaining"]} for p in pots],
            "event_baton": {"kind": self.task.get("kind", "") if self.task else "",
                            "target": self.task.get("target") if self.task else None,
                            "age": max(0, tick_of(obs) - self.task_started) if self.task else 0,
                            "blocked": self.blocked},
            "protocol": dict(self.protocol),
            "reason": reason,
            "candidates": [{"id": c.get("id"), "kind": c.get("kind"),
                            "priority": c.get("priority"), "target": c.get("target"),
                            "description": c.get("description")} for c in options[:10]],
        })

    def launch_jev(self, obs, options, partner_plan, reason):
        if self.system1_off or self.jev_future is not None or not options or self.ctx.done:
            return
        now = tick_of(obs)
        if now - self.last_jev_tick < 18:
            return
        criteria = {str(c.get("id")): str(c.get("description", c.get("kind", "candidate"))) for c in options[:10] if c.get("id")}
        if not criteria:
            return
        self.last_jev_tick = now
        self.jev_epoch = self.task_epoch
        state = {"observation": self.compact_state(obs, options, partner_plan, reason)}
        forward = dict(criteria)
        reverse = dict(reversed(list(criteria.items())))
        questions = {
            "event_a": Choice(
                "Choose one listed code-generated next event. Use the already computed priority and partner intent; never invent an id, direction, arithmetic, or object.",
                forward,
            ),
            "event_b": Choice(
                "Choose one listed code-generated next event. Use the already computed priority and partner intent; never invent an id, direction, arithmetic, or object.",
                reverse,
            ),
            "yield": Noul(
                "Should the controlled chef yield briefly to the partner? True only when the partner has an urgent carried soup/dish, its listed route overlaps the current event, and the route is not improving. Otherwise false.",
                {"true": "An urgent partner transaction is stalled on the shared route.",
                 "false": "The partner is progressing, non-urgent, or not on this route."},
            ),
        }

        def work():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(state, questions, timeout_s=1.0)
            except Exception:
                return {}

        try:
            self.jev_future = self.ctx.spawn(work)
            self.log("jev_request", tick=now, reason=reason, epoch=self.task_epoch)
        except Exception:
            self.jev_future = None

    def poll_jev(self, obs, options, partner_plan):
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
        valid = {c.get("id"): c for c in options}
        a = answers.get("event_a")
        b = answers.get("event_b")
        av, bv = getattr(a, "value", None), getattr(b, "value", None)
        pa = getattr(a, "probabilities", {}) or {}
        pb = getattr(b, "probabilities", {}) or {}
        try:
            p_a = float(pa.get(av, 0.0)) if isinstance(pa, dict) else 0.0
        except Exception:
            p_a = 0.0
        try:
            p_b = float(pb.get(bv, 0.0)) if isinstance(pb, dict) else 0.0
        except Exception:
            p_b = 0.0
        if av == bv and av in valid and (p_a == 0.0 or p_b == 0.0 or min(p_a, p_b) >= 0.55):
            best = self.code_choice(obs, options, partner_plan)
            if int(valid[av].get("priority", 20)) <= int(best.get("priority", 20)) + 1:
                self.task_hint = av
                self.hint_tick = tick_of(obs)
                self.log("jev_event", tick=tick_of(obs), value=av, probability=min(p_a, p_b))
        yield_answer = answers.get("yield")
        try:
            probability = float(getattr(yield_answer, "value", 0.0))
        except Exception:
            probability = 0.0
        me, partner, _ = actors(obs)
        if probability >= 0.75 and not held_name(me) and held_name(partner) in {"dish", "soup"} and partner_plan.get("kind") in {"pickup", "serve"}:
            self.yield_ticks = max(self.yield_ticks, 2)
            self.log("jev_yield", tick=tick_of(obs), ticks=self.yield_ticks)

    # ---- interaction and joint primitive lattice ----------------------
    def can_interact(self, obs, task):
        me, _, _ = actors(obs)
        held = held_name(me)
        kind = task.get("kind")
        target = point(task.get("target"))
        terrain = terrain_of(obs)
        pots = pot_info(obs)
        pot = next((p for p in pots if p["pos"] == target), None)
        if kind == "serve":
            return held == "soup" and cell(terrain, target) == "S"
        if kind == "get_ingredient":
            return not held and cell(terrain, target) in {"O", "T"}
        if kind == "get_dish":
            return not held and cell(terrain, target) == "D"
        if kind == "get_counter":
            return not held and any(name == task.get("item") and p == target for name, p in counter_items(obs))
        if kind == "put_counter":
            return held in INGREDIENTS and target not in {object_position(obj) for obj in objects_of(obs)}
        if kind == "fill":
            return bool(pot and held in INGREDIENTS and pot["phase"] in {"empty", "partial"} and held in pot["missing"])
        if kind == "start":
            return bool(pot and not held and pot["phase"] == "full_idle")
        if kind == "pickup":
            return bool(pot and held == "dish" and pot["ready"])
        return False

    def partner_forecast(self, obs, partner_plan, horizon=5):
        p0 = partner_plan.get("position")
        path = partner_plan.get("path") or []
        if p0 is None:
            return [None] * (horizon + 1)
        result = [p0]
        for i in range(1, horizon + 1):
            if i < len(path):
                result.append(path[i])
            else:
                result.append(result[-1])
        return result

    def lattice_first_action(self, obs, goal, partner_plan, task):
        me, partner, _ = actors(obs)
        start, p0 = point(me.get("position")), point(partner.get("position"))
        if start is None or goal is None:
            return self.action(self.noop)
        terrain = terrain_of(obs)
        forecast = self.partner_forecast(obs, partner_plan, 5)
        # Beam search over self positions.  The partner route is a forecast,
        # not a lease: conflict penalties are finite and expire as the route
        # stops improving.
        beam = [(0.0, start, self.noop)]
        first_actions = [self.noop] + [a for a in DIRECTIONS if a in self.actions]
        if self.blocked >= 4:
            self.route_flip = 1 - self.route_flip
        for depth in range(1, 6):
            expanded = []
            candidates = first_actions if depth == 1 else [self.noop] + [a for a in DIRECTIONS if a in self.actions]
            partner_now = forecast[depth - 1] if depth - 1 < len(forecast) else None
            partner_next = forecast[depth] if depth < len(forecast) else partner_now
            for cost, current, first in beam:
                for action in candidates:
                    nxt = current
                    if action in DIRECTIONS:
                        delta = DIRECTIONS[action]
                        candidate = (current[0] + delta[0], current[1] + delta[1])
                        if walkable(terrain, candidate):
                            nxt = candidate
                    local = cost + 1.0
                    if nxt == partner_next:
                        local += 160.0
                    if partner_now == nxt and partner_next == current:
                        local += 120.0
                    if nxt in forecast[1:depth]:
                        local += 9.0 if self.partner_bad_route < 5 else 3.0
                    dist_path = bfs(terrain, nxt, goal)
                    dist = len(dist_path) - 1 if dist_path else 80
                    local += float(dist) * (1.0 + 0.08 * depth)
                    if action == self.noop and nxt != goal:
                        local += 2.0 + (self.blocked * 0.6)
                    if self.route_flip and action in {"north", "south"}:
                        local += 0.15
                    expanded.append((local, nxt, first))
            expanded.sort(key=lambda row: (row[0], row[1], row[2]))
            # Keep diverse end cells so the fallback is not a single greedy BFS.
            beam = expanded[:18]
        if not beam:
            return self.action(self.noop)
        beam.sort(key=lambda row: row[0])
        return self.action(beam[0][2])

    def motor(self, obs, partner_plan):
        me, partner, controlled = actors(obs)
        current = point(me.get("position"))
        facing = point(me.get("orientation"))
        ppos = point(partner.get("position"))
        task = self.task
        if not task:
            return self.action(self.noop)
        kind = task.get("kind")
        target = point(task.get("target"))
        if kind in {"hold"} or target is None:
            return self.action(self.noop)

        # Once at an interaction pose, orientation and interaction are separate
        # primitives.  The action is checked against the current state every tick.
        pose = point(task.get("pose"))
        if pose is None or current is None:
            pose, _ = self.target_route(obs, task, True)
            task["pose"] = pose
        # The partner can enter a pose after the baton was committed.  Rebind
        # the baton to another legal approach before the lattice is asked to
        # choose a step; otherwise its finite collision penalty can make an
        # occupied goal look like a reason to wait forever.
        if kind not in {"wait_dish", "park", "idle"} and pose is not None and pose == ppos:
            alternate, _ = route_to(terrain_of(obs), current, target, {ppos} if ppos else set())
            if alternate is not None and alternate != pose:
                pose = alternate
                task["pose"] = alternate
            if current == target:
                return self.action(self.noop)
            return self.lattice_first_action(obs, target, partner_plan, task)
        if current == pose and pose is not None:
            delta = (target[0] - current[0], target[1] - current[1])
            turn = ACTION_FOR.get(delta)
            if turn and facing != delta:
                return self.action(turn)
            if self.can_interact(obs, task):
                return self.action("interact")
            return self.action(self.noop)

        # Index-order arbitration applies only to a same-target interaction.
        # Agent 0 resolves its interaction first; agent 1 yields one tick when
        # both are already facing the same counter/pot.
        if pose is not None and ppos == pose and partner_plan.get("target") == target:
            partner_facing = point(partner.get("orientation"))
            desired = (target[0] - ppos[0], target[1] - ppos[1])
            partner_ready = partner_facing == desired and held_name(partner) in OBJECTS | {""}
            if partner_ready and controlled == 1:
                return self.action(self.noop)

        action = self.lattice_first_action(obs, pose, partner_plan, task)
        return self.action(action)

    # ---- realtime loop -------------------------------------------------
    def run(self):
        ctx = self.ctx
        while not ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            pots = pot_info(obs)
            partner_plan = self.update_partner(obs, pots)
            self.poll_astra(obs)
            options = self.candidates(obs, partner_plan)
            self.start_astra(obs, options, partner_plan)

            me, _, _ = actors(obs)
            current = point(me.get("position"))
            if self.last_position is not None and current == self.last_position and self.last_action in DIRECTIONS:
                self.blocked += 1
            elif current is not None and current != self.last_position:
                self.blocked = max(0, self.blocked - 1)
            self.last_position = current

            # A phase transition or completion invalidates an event baton.  A
            # blocked baton is rerouted, not converted into a permanent wait.
            if not self.task_valid(obs, options) or (self.blocked >= 8 and not held_name(me) == "soup"):
                if self.blocked >= 8:
                    self.route_flip = 1 - self.route_flip
                self.commit(obs, options, partner_plan, "event_boundary" if self.blocked < 8 else "blocked_route")
            self.poll_jev(obs, options, partner_plan)
            if self.task_epoch != self.jev_epoch and self.task is not None and self.task_started == tick_of(obs):
                self.launch_jev(obs, options, partner_plan, "event_boundary")
            elif self.blocked >= 4:
                self.launch_jev(obs, options, partner_plan, "route_conflict")
            elif tick_of(obs) - self.last_jev_tick >= 42:
                self.launch_jev(obs, options, partner_plan, "heartbeat")

            action = self.motor(obs, partner_plan)
            if self.yield_ticks > 0 and not held_name(me) and self.task.get("priority", 20) > 1:
                self.yield_ticks -= 1
                action = self.action(self.noop)
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
        EventCalendar(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
