"""Candidate 0026: deadline-checked physical transactions.

Code owns production, conservative arrival estimates, and legal primitives.
Astra supplies a bounded counter/handheld protocol; Jev supplies short-lived
operating-mode and partner-progress judgments. No model emits directions.

The initial unconditional dish-buffer hypothesis was refuted in live tries.
A dish is now put down only to free a hand for real production, not merely
because a pot is cooking. Dish acquisition and the final approach are timed
from observed readiness, every wait is phase-preemptible, and partner claims
expire without new-best-distance progress. See proposal.md for the evidence.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


MOVES = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
MOVE_NAME = {delta: name for name, delta in MOVES.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}
HELD_NAMES = INGREDIENTS | {"dish", "soup"}


# ---------------------------------------------------------------------------
# Defensive observation decoding.  The adaptor's JSON is authoritative, but
# pre-flight deliberately supplies malformed model text and occasionally
# sparse observations.  These helpers never let that become a harness error.


def point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        return None
    if not isinstance(value[0], (int, float)) or not isinstance(value[1], (int, float)):
        return None
    try:
        return int(value[0]), int(value[1])
    except Exception:
        return None


def number(value, default=None):
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return default
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value.strip()):
        try:
            return int(float(value.strip()))
        except Exception:
            return default
    return default


def flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ready", "cooking"}
    return False


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
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
    if not isinstance(obs, dict):
        return 0
    value = obs.get("tick", state_of(obs).get("timestep", 0))
    return number(value, 0) or 0


def cell(grid, value):
    p = point(value)
    if p is None:
        return "X"
    x, y = p
    if y < 0 or y >= len(grid):
        return "X"
    try:
        row = grid[y]
        if isinstance(row, (str, list, tuple)) and 0 <= x < len(row):
            return row[x]
    except Exception:
        pass
    return "X"


def open_cell(grid, value):
    return cell(grid, value) == " "


def feature_positions(grid, symbol):
    found = []
    for y, row in enumerate(grid):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, item in enumerate(row):
            if item == symbol:
                found.append((x, y))
    return found


def players_of(obs):
    raw = state_of(obs).get("players", [])
    return raw if isinstance(raw, list) else []


def actors(obs):
    players = players_of(obs)
    index = number(obs.get("controlled_agent_index", 0), 0) if isinstance(obs, dict) else 0
    if index not in (0, 1) or index >= len(players):
        index = 0
    other = 1 - index if len(players) > 1 else index
    mine = players[index] if index < len(players) and isinstance(players[index], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return mine, partner


def held_name(player):
    raw = player.get("held_object") if isinstance(player, dict) else None
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("name", "") or "").lower()


def objects_of(obs):
    raw = state_of(obs).get("objects", [])
    return raw if isinstance(raw, list) else []


def object_name(obj):
    return str(obj.get("name", "") or "").lower() if isinstance(obj, dict) else ""


def object_position(obj):
    return point(obj.get("position")) if isinstance(obj, dict) else None


def object_at(obs, target):
    target = point(target)
    if target is None:
        return None
    for obj in objects_of(obs):
        if object_position(obj) == target:
            return obj
    return None


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
    result = []
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
                result.append(recipe)
    return result or [["onion", "onion", "onion"]]


def missing_items(have, recipes):
    counts = Counter(have)
    chosen = None
    missing = None
    for recipe in recipes:
        wanted = Counter(recipe)
        if any(counts[item] > wanted[item] for item in counts):
            continue
        current = Counter(have)
        needed = []
        for item in recipe:
            if current[item] < wanted[item]:
                needed.append(item)
                current[item] += 1
        if missing is None or len(needed) < len(missing):
            chosen = list(recipe)
            missing = needed
    return chosen or [], missing if missing is not None else []


def neighbours(p):
    x, y = p
    # This order is deterministic but is not exposed to either model.
    for name in ("north", "south", "west", "east"):
        delta = MOVES[name]
        yield (x + delta[0], y + delta[1]), delta


def bfs(grid, start, goals, blocked=()):
    start = point(start)
    if start is None:
        return None
    if isinstance(goals, tuple) and len(goals) == 2 and all(isinstance(x, int) for x in goals):
        goals = [goals]
    elif isinstance(goals, list) and len(goals) == 2 and all(isinstance(x, int) for x in goals):
        goals = [tuple(goals)]
    goal_set = {point(g) for g in (goals or []) if point(g) is not None}
    if not goal_set:
        return None
    blocked = {point(x) for x in blocked if point(x) is not None}
    blocked.discard(start)
    queue = deque([start])
    previous = {start: None}
    if start in goal_set:
        return [start]
    while queue:
        current = queue.popleft()
        for nxt, _ in neighbours(current):
            if nxt in previous or nxt in blocked or not open_cell(grid, nxt):
                continue
            previous[nxt] = current
            if nxt in goal_set:
                path = [nxt]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


def interaction_poses(grid, target):
    target = point(target)
    if target is None:
        return []
    result = []
    for delta in MOVES.values():
        pose = (target[0] - delta[0], target[1] - delta[1])
        if open_cell(grid, pose) and pose not in result:
            result.append(pose)
    return result


def route_to(grid, start, target, blocked=()):
    """Shortest route to an interaction pose, with a static fallback."""
    start, target = point(start), point(target)
    if start is None or target is None:
        return None, None
    blocked = set(blocked)
    options = []
    for pose in interaction_poses(grid, target):
        path = bfs(grid, start, [pose], blocked)
        if path is not None:
            options.append((len(path), pose, path))
    if not options:
        for pose in interaction_poses(grid, target):
            path = bfs(grid, start, [pose])
            if path is not None:
                options.append((len(path), pose, path))
    if not options:
        return None, None
    options.sort(key=lambda row: (row[0], row[1]))
    return options[0][1], options[0][2]


def route_to_cell(grid, start, target, blocked=()):
    path = bfs(grid, start, [target], blocked)
    return point(target), path


def counter_sites(obs):
    grid = terrain_of(obs)
    occupied = {object_position(obj) for obj in objects_of(obs) if object_position(obj) is not None}
    result = []
    for y, row in enumerate(grid):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, item in enumerate(row):
            p = (x, y)
            if item == "X" and p not in occupied and any(open_cell(grid, q) for q, _ in neighbours(p)):
                result.append(p)
    return result


def counter_objects(obs):
    grid = terrain_of(obs)
    result = []
    for obj in objects_of(obs):
        p = object_position(obj)
        name = object_name(obj)
        if p is not None and name in HELD_NAMES and cell(grid, p) == "X":
            result.append((name, p))
    return result


def all_interaction_cells(obs):
    grid = terrain_of(obs)
    cells = set()
    for symbol in FEATURES:
        for target in feature_positions(grid, symbol):
            cells.update(interaction_poses(grid, target))
    return cells


def pot_facts(obs):
    grid = terrain_of(obs)
    recipes = recipes_of(obs)
    soup_by_pos = {}
    for obj in objects_of(obs):
        p = object_position(obj)
        if p is not None and object_name(obj) == "soup":
            soup_by_pos[p] = obj
    result = []
    for target in feature_positions(grid, "P"):
        soup = soup_by_pos.get(target)
        have = ingredients_of(soup)
        recipe, missing = missing_items(have, recipes)
        ready = bool(soup and flag(soup.get("is_ready")))
        cooking = bool(soup and flag(soup.get("is_cooking"))) and not ready
        cook_tick = number(soup.get("_cooking_tick", soup.get("cooking_tick", -1)), -1) if soup else -1
        cook_time = number(soup.get("cook_time", -1), -1) if soup else -1
        if not ready and not cooking and cook_tick is not None and cook_time is not None:
            # A nonnegative cooking tick is useful evidence, but a string such
            # as "start" or "cook" is not numeric evidence.
            cooking = cook_tick >= 0 and cook_time >= 0 and not flag(soup.get("is_idle"))
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
        if cooking and cook_tick is not None and cook_time is not None and cook_time >= 0:
            remaining = max(0, cook_time - cook_tick)
        result.append({
            "pos": target,
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


class BufferedKitchen:
    """A rolling producer whose handoff medium is an observed counter object."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = getattr(game, "action_spec", None)
        self.legal = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.legal:
            self.noop = "stay" if "stay" in self.legal else (next(iter(self.legal), "stay"))

        # Physical executor state.
        self.job = None
        self.job_since = -1
        self.job_epoch = 0
        self.last_action = self.noop
        self.last_self_pos = None
        self.last_partner_pos = None
        self.blocked = 0
        self.partner_still = 0
        self.partner_stall = 0
        self.partner_distance = None
        self.partner_best_tick = 0
        self.last_hand = ""
        self.partner_intent_key = None
        self.dish_slot = None
        self.dish_slot_pot = None
        self.lookahead_key = None
        self.previous_cooking_key = None
        self.mode = "finish"
        self.mode_until = -1
        self.jevis_handoff = None
        self.jevis_until = -1

        # Counter use is conditional on real work and observed slack; a
        # model can disable optional buffering but cannot force an unsafe trip.
        self.astra_buffer_id = None
        self.astra_lookahead = "one"
        self.astra_protocol = "counter"
        self.astra_future = None
        self.astra_started = False
        self.protocol_reviewed = False
        self.last_handoff_override = None

        # Jev observes a semantic phase, not a primitive candidate list.
        self.jev_future = None
        self.jev_tick = -999
        self.jev_signature = None
        self.system1_off = getattr(ctx, "system1", object()) is None

    # ----- safe integration -------------------------------------------------

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

    # ----- partner and motion facts ----------------------------------------

    def partner_next(self, obs):
        _, partner = actors(obs)
        p = point(partner.get("position"))
        name = obs.get("partner_action_last", "") if isinstance(obs, dict) else ""
        delta = MOVES.get(name)
        if p is None or delta is None:
            return p
        return p[0] + delta[0], p[1] + delta[1]

    def partner_intent(self, obs):
        """Infer one concrete target from the partner's hand and pot facts."""
        _, partner = actors(obs)
        start = point(partner.get("position"))
        held = held_name(partner)
        grid = terrain_of(obs)
        pots = pot_facts(obs)
        target = None
        kind = "none"
        urgent = False
        if held == "soup":
            goals = feature_positions(grid, "S")
            target = goals[0] if goals else None
            kind, urgent = "deliver", True
        elif held == "dish":
            ready = [p["pos"] for p in pots if p["ready"]]
            cooking = [p["pos"] for p in pots if p["cooking"]]
            goals = ready or cooking
            target = sorted(goals)[0] if goals else None
            kind, urgent = ("pickup" if ready else "dish_stage"), bool(goals)
        elif held in INGREDIENTS:
            matches = [p for p in pots if p["phase"] in {"empty", "partial", "full_idle"} and held in p["missing"]]
            if matches:
                target = min(matches, key=lambda p: (len(p["missing"]), p["pos"]))["pos"]
                kind = "fill"
        if start is None or target is None:
            return {"kind": kind, "target": target, "distance": None, "path": [], "urgent": urgent, "held": held}
        _, path = route_to(grid, start, target)
        distance = len(path) - 1 if path else None
        return {"kind": kind, "target": target, "distance": distance, "path": path or [], "urgent": urgent, "held": held}

    def partner_corridor(self, obs, horizon=8):
        fact = self.partner_intent(obs)
        cells = set((fact.get("path") or [])[1:horizon + 1])
        target = fact.get("target")
        if target is not None:
            cells.update(interaction_poses(terrain_of(obs), target))
        nxt = self.partner_next(obs)
        if nxt is not None:
            cells.add(nxt)
        return cells, fact

    def update_motion(self, obs):
        me, partner = actors(obs)
        current = point(me.get("position"))
        other = point(partner.get("position"))
        if self.last_self_pos is not None and current == self.last_self_pos and self.last_action in MOVES:
            self.blocked += 1
        elif current is not None and current != self.last_self_pos:
            self.blocked = max(0, self.blocked - 1)
        if self.last_partner_pos is not None and other == self.last_partner_pos:
            self.partner_still += 1
        else:
            self.partner_still = 0

        fact = self.partner_intent(obs)
        key = (fact.get("kind"), fact.get("target"), fact.get("held"))
        distance = fact.get("distance")
        if key != self.partner_intent_key:
            self.partner_intent_key = key
            self.partner_distance = distance
            self.partner_best_tick = tick_of(obs)
        elif distance is not None and (self.partner_distance is None or distance < self.partner_distance):
            # Progress means a new best distance, not merely returning to a
            # previously visited cell after an avoidance oscillation.
            self.partner_distance = distance
            self.partner_best_tick = tick_of(obs)
        self.partner_stall = max(0, tick_of(obs) - self.partner_best_tick) if fact.get("target") else 0
        mine = held_name(me)
        if self.last_hand == "" and mine in INGREDIENTS and self.job and self.job.get("kind") == "fetch_lookahead":
            self.lookahead_key = self.active_cooking_key(pot_facts(obs))
        self.last_hand = mine

        self.last_self_pos = current
        self.last_partner_pos = other

    # ----- counter-buffer bookkeeping --------------------------------------

    def refresh_dish_slot(self, obs):
        me, _ = actors(obs)
        mine = held_name(me)
        if self.dish_slot is not None:
            obj = object_at(obs, self.dish_slot)
            if object_name(obj) == "dish":
                return
            # Keep the remembered slot while the controlled chef is carrying
            # the dish to it or away from it.  Once empty-handed and absent,
            # the physical reservation is gone.
            if mine != "dish":
                self.dish_slot = None
                self.dish_slot_pot = None
        if self.dish_slot is None and mine != "dish":
            for name, p in counter_objects(obs):
                if name == "dish":
                    self.dish_slot = p
                    if self.dish_slot_pot is None:
                        active = [x for x in pot_facts(obs) if x["ready"] or x["cooking"]]
                        self.dish_slot_pot = active[0]["pos"] if active else None
                    return

    def dish_is_staged(self, obs):
        self.refresh_dish_slot(obs)
        return self.dish_slot is not None and object_name(object_at(obs, self.dish_slot)) == "dish"

    def free_counters(self, obs, exclude_dish=True):
        occupied = {p for _, p in counter_objects(obs)}
        result = []
        for p in counter_sites(obs):
            if p in occupied:
                continue
            if exclude_dish and self.dish_slot is not None and p == self.dish_slot:
                continue
            result.append(p)
        return result

    def buffer_options(self, obs):
        # IDs are stable within an episode and are the only counter labels
        # shown to Astra.  They do not encode an unverified layout name.
        result = []
        grid = terrain_of(obs)
        for p in sorted(feature_positions(grid, "X")):
            if interaction_poses(grid, p):
                result.append({"id": "buffer_%d_%d" % p, "position": p,
                               "free": object_at(obs, p) is None})
        return result

    def choose_dish_counter(self, obs, current=None):
        me, partner = actors(obs)
        current = point(current) if current is not None else point(me.get("position"))
        partner_pos = point(partner.get("position"))
        grid = terrain_of(obs)
        corridor, fact = self.partner_corridor(obs)
        free = self.free_counters(obs, exclude_dish=False)
        if current is None:
            return None
        preferred = None
        if self.astra_buffer_id:
            for option in self.buffer_options(obs):
                if option["id"] == self.astra_buffer_id:
                    preferred = option["position"]
                    break
        scored = []
        for target in free:
            _, path = route_to(grid, current, target, {partner_pos} if partner_pos else set())
            if path is None:
                continue
            overlap = sum(1 for p in path[1:] if p in corridor)
            # A preferred buffer is a soft preference only.  A buffer in the
            # partner's current corridor is never forced by Astra.
            pref_penalty = 0 if preferred == target else 1
            if preferred == target and (target in corridor or target == partner_pos):
                pref_penalty = 5
            partner_gap = abs(target[0] - partner_pos[0]) + abs(target[1] - partner_pos[1]) if partner_pos else 0
            # Staging beside the current chef is safer than carrying a dish
            # across the partner's lane. The model breaks close ties only.
            scored.append((len(path), overlap, pref_penalty, -partner_gap, target))
        if not scored:
            return None
        scored.sort(key=lambda row: row)
        return scored[0][-1]

    def safe_floor(self, obs, current=None, avoid=()):
        me, partner = actors(obs)
        current = point(current) if current is not None else point(me.get("position"))
        partner_pos = point(partner.get("position"))
        if current is None:
            return None
        corridor, _ = self.partner_corridor(obs, horizon=7)
        gates = all_interaction_cells(obs)
        avoid = set(avoid)
        candidates = []
        grid = terrain_of(obs)
        for y, row in enumerate(grid):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                target = (x, y)
                if not open_cell(grid, target) or target in avoid or target == partner_pos:
                    continue
                path = bfs(grid, current, [target], {partner_pos} if partner_pos else set())
                if path is None:
                    continue
                overlap = sum(1 for p in path[1:] if p in corridor)
                gate_penalty = 1 if target in gates else 0
                gap = abs(target[0] - partner_pos[0]) + abs(target[1] - partner_pos[1]) if partner_pos else 0
                # A zero-length path must NOT make a blocked corridor cell
                # look safe. Test the destination itself before route overlap.
                destination_conflict = int(target in corridor)
                candidates.append((destination_conflict, overlap, gate_penalty, len(path), -gap, target))
        if not candidates:
            return current
        candidates.sort(key=lambda row: row)
        return candidates[0][-1]

    # ----- production facts and jobs ---------------------------------------

    def active_cooking_key(self, pots):
        cooking = [p for p in pots if p["cooking"]]
        if not cooking:
            return None
        pot = min(cooking, key=lambda p: (p.get("remaining") if isinstance(p.get("remaining"), int) else 999, p["pos"]))
        return pot["pos"], tuple(pot.get("ingredients", []))

    def next_item(self, pots):
        cooking = [p for p in pots if p["cooking"]]
        if cooking:
            pot = min(cooking, key=lambda p: (p.get("remaining") if isinstance(p.get("remaining"), int) else 999, p["pos"]))
            if pot.get("recipe"):
                return pot["recipe"][0]
        for recipe in recipes_of(self.last_obs):
            if recipe:
                return recipe[0]
        return "onion"

    def make_job(self, kind, target=None, item="", priority=10, tag=""):
        target = point(target)
        suffix = "none" if target is None else "%d_%d" % target
        return {
            "id": "%s:%s:%s:%s" % (kind, item or "none", suffix, tag or "x"),
            "kind": kind,
            "target": target,
            "item": item,
            "priority": priority,
            "tag": tag,
        }

    def partner_handoff_active(self, obs, target=None):
        _, partner = actors(obs)
        ph = held_name(partner)
        fact = self.partner_intent(obs)
        if ph == "soup":
            return True
        if ph != "dish":
            return False
        if target is not None and fact.get("target") not in {target, None}:
            return False
        distance = fact.get("distance")
        baseline = self.partner_stall < 6 and (distance is None or distance <= 12)
        chosen = baseline
        if self.jevis_handoff is True and self.jevis_until >= tick_of(obs):
            chosen = self.partner_stall < 7 and (distance is None or distance <= 14)
        if self.jevis_handoff is False and self.jevis_until >= tick_of(obs) and self.partner_stall >= 3:
            chosen = False
        if chosen != baseline:
            event_key = (self.jevis_until, chosen, fact.get("target"))
            if event_key != self.last_handoff_override:
                self.last_handoff_override = event_key
                self.log("jev_handoff_apply", tick=tick_of(obs), baseline_active=baseline,
                         selected_active=chosen, target=fact.get("target"), stall=self.partner_stall)
        return chosen

    def lookahead_allowed(self, obs, pots):
        if self.astra_lookahead != "one":
            return False
        key = self.active_cooking_key(pots)
        if key is None or self.lookahead_key == key:
            return False
        if not self.dish_is_staged(obs):
            return False
        _, partner = actors(obs)
        if held_name(partner) in INGREDIENTS:
            return False
        if any(name in INGREDIENTS for name, _ in counter_objects(obs)):
            return False
        return bool(self.free_counters(obs, exclude_dish=True))

    def source_jobs(self, obs, item, priority=5, lookahead=False):
        result = []
        if not lookahead:
            for name, target in counter_objects(obs):
                if name == item:
                    result.append(self.make_job("take_counter", target, item, priority=priority, tag="matching"))
        for target in feature_positions(terrain_of(obs), "O" if item == "onion" else "T"):
            result.append(self.make_job("fetch_lookahead" if lookahead else "fetch_ingredient", target,
                                        item, priority=priority, tag="one_buffer" if lookahead else "source"))
        return result

    def dish_jobs(self, obs, priority=3):
        # An obstructed counter is never an exclusive obligation. Both the
        # counter and unlimited dispensers are legal alternatives every tick.
        result = [self.make_job("retrieve_dish", target, priority=priority, tag="physical")
                  for name, target in counter_objects(obs) if name == "dish"]
        result.extend(self.make_job("fetch_dish", target, priority=priority, tag="due")
                      for target in feature_positions(terrain_of(obs), "D"))
        return result

    def feature_cost(self, obs, current, target):
        _, path = route_to(terrain_of(obs), current, target)
        # Include a possible facing turn and the interaction. No arithmetic
        # is delegated to Jev, and this is conservative rather than exact
        # when the initial facing happens already to match.
        return len(path) + 1 if path is not None else 999

    def dish_due(self, obs, pot):
        me, _ = actors(obs)
        current = point(me.get("position"))
        remaining = pot.get("remaining")
        if not isinstance(remaining, int):
            return True
        cost = 999
        for job in self.dish_jobs(obs):
            pose, path = route_to(terrain_of(obs), current, job["target"])
            if path is not None:
                cost = min(cost, len(path) + 1 + self.feature_cost(obs, pose, pot["pos"]))
        return remaining <= cost + 2

    def jobs(self, obs):
        self.last_obs = obs
        self.refresh_dish_slot(obs)
        grid = terrain_of(obs)
        me, partner = actors(obs)
        current = point(me.get("position"))
        mine, phold = held_name(me), held_name(partner)
        pots = pot_facts(obs)
        cooking_key = self.active_cooking_key(pots)
        if cooking_key != self.previous_cooking_key:
            self.lookahead_key = None
        self.previous_cooking_key = cooking_key
        ready = sorted((p for p in pots if p["ready"]), key=lambda p: p["pos"])
        full = sorted((p for p in pots if p["phase"] == "full_idle"), key=lambda p: p["pos"])
        buildable = sorted((p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]),
                           key=lambda p: (len(p["missing"]), p["pos"]))
        cooking = sorted((p for p in pots if p["cooking"]),
                         key=lambda p: (p["remaining"] if isinstance(p["remaining"], int) else 999, p["pos"]))

        if mine == "soup":
            return [self.make_job("deliver", target, priority=0, tag="soup")
                    for target in feature_positions(grid, "S")] or [self.make_job("hold", priority=0)]

        if mine == "dish":
            if ready:
                return [self.make_job("pickup_ready", p["pos"], priority=1, tag="ready") for p in ready]
            # Hands are released only for an existing useful operation, not
            # merely because a cooking timer is running. This prevents the
            # unconditional fetch/stash/retrieve overhead of the initial test.
            if full:
                slot = self.choose_dish_counter(obs, current)
                if slot is not None:
                    self.dish_slot, self.dish_slot_pot = slot, full[0]["pos"]
                    return [self.make_job("stash_dish", slot, priority=2, tag="free_hand_to_start")]
            if cooking:
                pot = cooking[0]
                remaining = pot["remaining"]
                distance = self.feature_cost(obs, current, pot["pos"])
                if not isinstance(remaining, int) or remaining <= distance + 2:
                    return [self.make_job("approach_ready", pot["pos"], priority=2, tag="timed")]
                # Existing partial pot work can justify a counter release if
                # there is enough observed cooking slack for the detour.
                if buildable and self.astra_protocol == "counter" and self.astra_lookahead == "one":
                    wanted = buildable[0]["missing"][0]
                    slot = self.choose_dish_counter(obs, current)
                    if slot is not None:
                        slot_pose, slot_path = route_to(grid, current, slot)
                        sources = self.source_jobs(obs, wanted)
                        trip = 999
                        for job in sources:
                            source_pose, source_path = route_to(grid, slot_pose, job["target"])
                            if source_path is not None:
                                trip = min(trip, len(slot_path) + len(source_path) +
                                           self.feature_cost(obs, source_pose, buildable[0]["pos"]) + 8)
                        if remaining > trip:
                            self.dish_slot, self.dish_slot_pot = slot, pot["pos"]
                            return [self.make_job("stash_dish", slot, priority=3, tag="free_hand_to_fill")]
                return [self.make_job("wait_dish", self.safe_floor(obs, current), priority=8, tag="dish_wait")]
            # A premature dish must not prevent production when no soup is
            # cooking. Putting it down frees the hand to fill/start a pot.
            slot = self.choose_dish_counter(obs, current)
            if slot is not None:
                self.dish_slot = slot
                return [self.make_job("stash_dish", slot, priority=4, tag="free_hand")]
            return [self.make_job("wait_dish", self.safe_floor(obs, current), priority=10, tag="dish_wait")]

        if mine in INGREDIENTS:
            matches = [p for p in buildable if mine in p["missing"]]
            if matches:
                return [self.make_job("fill_pot", p["pos"], mine, priority=2, tag="fill") for p in matches]
            return [self.make_job("put_counter", target, mine, priority=5, tag="surplus")
                    for target in self.free_counters(obs)] or [self.make_job("hold", priority=15)]

        # One held dish can claim at most one soup. A second ready pot is an
        # independent delivery job, not another reason to yield indefinitely.
        partner_active = phold == "dish" and self.partner_handoff_active(obs)
        if ready and (len(ready) > int(partner_active) or not partner_active):
            dishes = self.dish_jobs(obs, priority=1)
            if dishes:
                return dishes
        if full:
            return [self.make_job("start_cooking", p["pos"], priority=2, tag="start") for p in full]
        if buildable:
            supplies = self.source_jobs(obs, buildable[0]["missing"][0])
            if supplies:
                return supplies
        if ready:
            return [self.make_job("yield_ready", self.safe_floor(obs, current), priority=4, tag="handoff")]
        if cooking:
            pot = cooking[0]
            # Start the dish trip by its arrival deadline, not immediately
            # after every start. Already-held dishes approach shortly before
            # readiness, and readiness interrupts every wait at once.
            if self.dish_due(obs, pot) and not partner_active:
                dishes = self.dish_jobs(obs)
                if dishes:
                    return dishes
            if self.lookahead_allowed(obs, pots):
                # Only acquire one item if a complete source->counter trip
                # fits before the dish retrieval deadline. A staged ingredient
                # is a physical buffer; merely fetching does not complete it.
                wanted = self.next_item(pots)
                sources = self.source_jobs(obs, wanted, priority=6, lookahead=True)
                feasible = []
                for job in sources:
                    pose, path = route_to(grid, current, job["target"])
                    if path is None:
                        continue
                    drop_cost = min((self.feature_cost(obs, pose, x) for x in self.free_counters(obs)), default=999)
                    remaining = pot.get("remaining")
                    if isinstance(remaining, int) and remaining > len(path) + drop_cost + 9:
                        feasible.append(job)
                if feasible:
                    return feasible
            return [self.make_job("park_cooking", self.safe_floor(obs, current), priority=10, tag="cook_wait")]
        return [self.make_job("park", self.safe_floor(obs, current), priority=20, tag="idle")]

    # ----- task validity and deterministic choice --------------------------

    def same_job(self, a, b):
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        return (a.get("kind"), point(a.get("target")), a.get("item"), a.get("tag")) == (
            b.get("kind"), point(b.get("target")), b.get("item"), b.get("tag"))

    def job_limit(self, kind):
        return {
            "deliver": 48,
            "pickup_ready": 22,
            "approach_ready": 16,
            "retrieve_dish": 18,
            "fetch_dish": 34,
            "stash_dish": 18,
            "fetch_ingredient": 34,
            "fetch_lookahead": 34,
            "take_counter": 24,
            "put_counter": 18,
            "fill_pot": 18,
            "start_cooking": 16,
            "wait_dish": 8,
            "yield_ready": 8,
            "yield_cooking": 8,
            "park_cooking": 8,
            "park": 8,
            "hold": 8,
        }.get(kind, 24)

    def job_legal(self, obs, job):
        if not isinstance(job, dict):
            return False
        kind = job.get("kind")
        target = point(job.get("target"))
        me, _ = actors(obs)
        mine = held_name(me)
        grid = terrain_of(obs)
        pots = {p["pos"]: p for p in pot_facts(obs)}
        if kind in {"park", "wait_dish", "yield_ready", "yield_cooking", "park_cooking", "hold"}:
            return True
        if target is None:
            return False
        if kind == "deliver":
            return mine == "soup" and cell(grid, target) == "S"
        if kind == "fetch_dish":
            return not mine and cell(grid, target) == "D"
        if kind in {"fetch_ingredient"}:
            return not mine and cell(grid, target) in {"O", "T"}
        if kind == "fetch_lookahead":
            return not mine and cell(grid, target) in {"O", "T"}
        if kind == "take_counter":
            return not mine and any(name == job.get("item") and p == target for name, p in counter_objects(obs))
        if kind == "put_counter":
            return mine in INGREDIENTS and target not in {p for _, p in counter_objects(obs)}
        if kind == "stash_dish":
            return mine == "dish" and cell(grid, target) == "X" and target not in {p for _, p in counter_objects(obs)}
        if kind == "retrieve_dish":
            return not mine and object_name(object_at(obs, target)) == "dish"
        pot = pots.get(target)
        if kind == "fill_pot":
            return bool(pot and mine in INGREDIENTS and pot["phase"] in {"empty", "partial", "full_idle"} and mine in pot["missing"])
        if kind == "start_cooking":
            return bool(pot and not mine and pot["phase"] == "full_idle")
        if kind == "pickup_ready":
            return bool(pot and mine == "dish" and pot["ready"])
        if kind == "approach_ready":
            return bool(pot and mine == "dish" and (pot["cooking"] or pot["ready"]))
        return False

    def route_length(self, obs, job):
        me, partner = actors(obs)
        current = point(me.get("position"))
        other = point(partner.get("position"))
        target = point(job.get("target"))
        if current is None or target is None:
            return 999
        blocked = {other} if other is not None and other != target else set()
        kind = job.get("kind")
        if kind in {"park", "wait_dish", "yield_ready", "yield_cooking", "park_cooking", "hold"}:
            path = bfs(terrain_of(obs), current, [target], blocked)
        else:
            _, path = route_to(terrain_of(obs), current, target, blocked)
            # A stored dish is replaceable. Do not repeatedly recommit its
            # blocked static-fallback route when an unlimited dispenser has
            # a clear path; that recreated an ownership deadlock in a try.
            if kind == "retrieve_dish" and path is not None and other in path[1:]:
                return 999
        return len(path) if path is not None else 999

    def choose_job(self, obs, options):
        if not options:
            return self.make_job("park", priority=20, tag="empty")
        _, partner = actors(obs)
        phold = held_name(partner)
        fact = self.partner_intent(obs)
        corridor, _ = self.partner_corridor(obs)
        ranked = []
        for job in options:
            kind = job.get("kind")
            rank = job.get("priority", 20)
            if self.mode == "finish" and kind in {"retrieve_dish", "pickup_ready", "fetch_dish", "deliver"}:
                rank -= 1
            if self.mode == "buffer" and kind in {"stash_dish", "fetch_lookahead", "put_counter"}:
                rank -= 1
            if self.mode == "yield" and kind in {"park", "park_cooking", "yield_ready", "yield_cooking"} and phold:
                rank -= 2
            target = point(job.get("target"))
            conflict = 1 if target in corridor else 0
            if phold == "soup" and kind not in {"deliver", "retrieve_dish", "pickup_ready"}:
                conflict += 1
            ranked.append((rank, conflict, self.route_length(obs, job), str(job.get("id", "")), job))
        ranked.sort(key=lambda row: row[:4])
        return dict(ranked[0][4])

    def current_valid(self, obs, options):
        if not self.job:
            return False
        now = tick_of(obs)
        if now - self.job_since > self.job_limit(self.job.get("kind", "")):
            return False
        if self.job.get("kind") not in {"park", "wait_dish", "yield_ready", "yield_cooking", "park_cooking", "hold"} and not self.job_legal(obs, self.job):
            return False
        if any(self.same_job(self.job, candidate) for candidate in options):
            if self.job.get("kind") in {"park", "wait_dish", "yield_ready", "yield_cooking", "park_cooking"}:
                corridor, _ = self.partner_corridor(obs)
                if point(self.job.get("target")) in corridor:
                    return False
            if self.job.get("kind") == "retrieve_dish" and self.blocked >= 2:
                return False
            return True
        # Dynamic parking can retain its floor target only while the same
        # logical waiting phase is still offered. Ready food, a new empty pot,
        # and explicit starts interrupt waits immediately, not at age expiry.
        if self.job.get("kind") in {"park", "wait_dish", "yield_ready", "yield_cooking", "park_cooking"}:
            if not any(x.get("kind") == self.job.get("kind") and
                       x.get("tag") == self.job.get("tag") for x in options):
                return False
            target = point(self.job.get("target"))
            _, partner = actors(obs)
            pp = point(partner.get("position"))
            corridor, _ = self.partner_corridor(obs)
            return target is not None and open_cell(terrain_of(obs), target) and target != pp and target not in corridor
        return False

    def commit(self, obs, options):
        chosen = self.choose_job(obs, options)
        self.job = chosen
        self.job_since = tick_of(obs)
        self.job_epoch += 1
        self.blocked = 0
        me, partner = actors(obs)
        self.log("buffer_job", tick=self.job_since, kind=chosen.get("kind"), target=chosen.get("target"),
                 item=chosen.get("item"), epoch=self.job_epoch, mode=self.mode,
                 self_position=point(me.get("position")), self_held=held_name(me),
                 partner_position=point(partner.get("position")), partner_held=held_name(partner),
                 pots=[{"position": q["pos"], "phase": q["phase"], "remaining": q["remaining"]} for q in pot_facts(obs)])

    # ----- Astra protocol author -------------------------------------------

    def astra_snapshot(self, obs, options):
        me, partner = actors(obs)
        pots = pot_facts(obs)
        return jsonable({
            "game": "Overcooked-AI symbolic realtime kitchen",
            "tick": tick_of(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "terrain": terrain_of(obs),
            "self": {"position": point(me.get("position")), "held": held_name(me)},
            "partner": {"position": point(partner.get("position")), "held": held_name(partner),
                        "intent": self.partner_intent(obs), "stall_ticks": self.partner_stall},
            "pots": [{"position": p["pos"], "phase": p["phase"], "ingredients": p["ingredients"],
                      "missing": p["missing"], "remaining": p["remaining"]} for p in pots],
            "counter_options": self.buffer_options(obs),
            "free_counters": self.free_counters(obs, exclude_dish=False),
            "current_jobs": [{"kind": x.get("kind"), "target": x.get("target"), "priority": x.get("priority")}
                             for x in options[:8]],
        })

    def start_astra(self, obs, options):
        if self.astra_started or getattr(self.ctx, "system2", object()) is None:
            self.astra_started = True
            return
        self.astra_started = True
        snapshot = self.astra_snapshot(obs, options)
        choices = [x["id"] for x in self.buffer_options(obs)]
        prompt = (
            "You are Astra, a slow strategic scheduler for a symbolic Overcooked-AI realtime kitchen. "
            "The executor owns recipe matching, pot phases, reachability, collision safety, and every "
            "primitive action. Design only a physical counter protocol for the controlled chef. "
            "Choose one legal counter id from counter_options for temporarily staging a dish; choose "
            "'none' only if no counter is safe. Choose whether to stage at most one next-cycle "
            "ingredient while a dish is buffered. Return ONLY JSON with keys "
            "dish_buffer (one of %s or 'none'), lookahead ('one'|'none'), "
            "protocol ('counter'|'handheld'). Never output directions, coordinates, or arithmetic.\n"
            % (choices or ["none"]) + json.dumps(snapshot, separators=(",", ":"))
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

    def parse_object(self, text):
        if not isinstance(text, str):
            return {}
        # Use the first decodable object rather than a greedy parse of a
        # prose answer containing several brace pairs.
        for start, char in enumerate(text):
            if char != "{":
                continue
            depth = 0
            for end in range(start, min(len(text), start + 12000)):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = json.loads(text[start:end + 1])
                            return value if isinstance(value, dict) else {}
                        except Exception:
                            break
        return {}

    def poll_astra(self, obs):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            raw = future.result()
            text = raw if isinstance(raw, str) else getattr(raw, "text", "")
            value = self.parse_object(text)
            allowed = {x["id"] for x in self.buffer_options(obs)}
            if value.get("dish_buffer") in allowed:
                self.astra_buffer_id = value["dish_buffer"]
            elif value.get("dish_buffer") == "none":
                self.astra_buffer_id = None
            if value.get("lookahead") in {"one", "none"}:
                self.astra_lookahead = value["lookahead"]
            if value.get("protocol") in {"counter", "handheld"}:
                self.astra_protocol = value["protocol"]
            self.protocol_reviewed = True
            self.log("astra_buffer_protocol", tick=tick_of(obs), buffer=self.astra_buffer_id,
                     lookahead=self.astra_lookahead, protocol=self.astra_protocol)
        except Exception:
            return

    # ----- Jev phase classifier --------------------------------------------

    def phase_signature(self, obs):
        me, partner = actors(obs)
        pots = pot_facts(obs)
        return (
            held_name(me), held_name(partner),
            tuple((p["pos"], p["phase"], tuple(p["ingredients"])) for p in pots),
            bool(self.dish_is_staged(obs)),
            (self.astra_protocol, self.astra_lookahead, self.astra_buffer_id),
        )

    def jev_state(self, obs, options, reason):
        me, partner = actors(obs)
        pots = pot_facts(obs)
        corridor, intent = self.partner_corridor(obs)
        return jsonable({
            "game": "Overcooked-AI symbolic realtime gridworld",
            "tick": tick_of(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {"position": point(me.get("position")), "held": held_name(me)},
            "partner": {"position": point(partner.get("position")), "held": held_name(partner),
                        "intent": {"kind": intent.get("kind"), "target": intent.get("target"),
                                   "distance_bucket": "near" if isinstance(intent.get("distance"), int) and intent.get("distance") <= 4 else "far" if intent.get("distance") is not None else "unknown"},
                        "stationary_ticks": self.partner_still, "route_stall_ticks": self.partner_stall,
                        "progress": "unknown" if intent.get("target") is None else
                                    "at_goal" if intent.get("distance") == 0 else
                                    "stalled" if self.partner_stall >= 5 else "advancing"},
            "pots": [{"position": p["pos"], "phase": p["phase"], "missing": p["missing"],
                      "remaining_bucket": "ready" if p["ready"] else "soon" if isinstance(p["remaining"], int) and p["remaining"] <= 5 else "later" if p["cooking"] else "none"}
                     for p in pots],
            "buffer": {"dish_counter": self.dish_slot, "dish_staged": self.dish_is_staged(obs),
                       "free_counters": self.free_counters(obs), "lookahead": self.astra_lookahead},
            "current": {"kind": self.job.get("kind", "") if self.job else "", "age": max(0, tick_of(obs) - self.job_since) if self.job else 0,
                        "blocked": self.blocked},
            "work": {"direct_available": any(x.get("kind") in {"deliver", "pickup_ready", "start_cooking", "fill_pot", "fetch_ingredient", "fetch_dish", "retrieve_dish", "take_counter", "approach_ready"} for x in options),
                     "buffer_available": any(x.get("kind") in {"stash_dish", "put_counter", "fetch_lookahead"} for x in options),
                     "route_stalled": self.blocked >= 3,
                     "offered_jobs": [x.get("kind") for x in options[:8]]},
            "astra_protocol": {"protocol": self.astra_protocol, "lookahead": self.astra_lookahead,
                               "preferred_buffer": self.astra_buffer_id},
            "corridor_cells": sorted(corridor),
            "candidate_modes": ["finish", "buffer", "yield", "rebuild", "unknown"],
            "reason": reason,
        })

    def start_jev(self, obs, options, reason):
        if self.system1_off or self.jev_future is not None or not options:
            return
        now = tick_of(obs)
        if now - self.jev_tick < 18:
            return
        self.jev_tick = now
        self.jev_signature = self.phase_signature(obs)
        state = self.jev_state(obs, options, reason)
        modes = {
            "finish": "Continue useful production: supply/fill/start a compatible pot, obtain a due dish, pick up soup, or deliver.",
            "buffer": "Release the occupied hand for a listed productive counter operation when work.buffer_available is true.",
            "yield": "Allow an advancing partner handoff while no direct production job is offered; do not reserve a pot indefinitely.",
            "rebuild": "The current job is stalled or contradicted by live facts; recompute it instead of repeating the commitment.",
            "unknown": "The supplied facts do not distinguish a mode; keep the deterministic controller unchanged.",
        }
        order = list(modes)
        if (now // 18) % 2:
            order.reverse()
        first = {key: modes[key] for key in order}
        second = {key: modes[key] for key in reversed(order)}
        instructions = (
            "Classify the operating mode using work, current, partner.progress, pots, and buffer. "
            "Prefer finish for useful direct production; buffer only for an offered physical counter job; "
            "yield only for an advancing handoff with no direct work; rebuild only for an actual stall. "
            "Use unknown if unsupported. Never invent a target/direction or compute arithmetic."
        )
        questions = {
            "mode_a": Choice(instructions, first),
            "mode_b": Choice(instructions, second),
            "handoff": Noul(
                "Does the evidence support trusting this concrete partner handoff for a short interval? "
                "Use partner.intent, partner.progress, its held object, and the pot phases together. "
                "A dish carrier waiting at its cooking target can be active; moving back and forth "
                "without a new best route distance is not progress. Unknown/absent intent is unsupported.",
                {"true": "Current carried-object, target, and progress evidence supports the handoff.",
                 "false": "The handoff is absent, stalled, wrong-target, or unsupported."}),
        }

        def ask():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(state, questions, timeout_s=1.0)
            except Exception:
                return {}

        try:
            self.jev_future = self.ctx.spawn(ask)
            self.log("phase_jev_request", tick=now, reason=reason)
        except Exception:
            self.jev_future = None

    def probability(self, answer, selected):
        probs = getattr(answer, "probabilities", {}) or {}
        if not isinstance(probs, dict) or not isinstance(selected, str):
            return 0.0
        try:
            return float(probs.get(selected, 0.0))
        except Exception:
            return 0.0

    def poll_jev(self, obs):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict):
            return
        now = tick_of(obs)
        # Semantic answers survive ordinary movement, but never a phase or
        # buffer transition and never more than twelve logical ticks.
        if now - self.jev_tick > 12 or self.jev_signature != self.phase_signature(obs):
            return
        a, b = answers.get("mode_a"), answers.get("mode_b")
        av, bv = getattr(a, "value", None), getattr(b, "value", None)
        allowed = {"finish", "buffer", "yield", "rebuild"}
        pa, pb = self.probability(a, av), self.probability(b, bv)
        if isinstance(av, str) and isinstance(bv, str) and av in allowed and av == bv and min(pa, pb) >= 0.52:
            self.mode = av
            self.mode_until = now + 10
            self.log("phase_jev_mode", tick=now, mode=av, probability=min(pa, pb))
        handoff = answers.get("handoff")
        try:
            hp = float(getattr(handoff, "value", float("nan")))
        except Exception:
            hp = float("nan")
        if not math.isfinite(hp) or not 0.0 <= hp <= 1.0:
            return
        if hp >= 0.75:
            self.jevis_handoff = True
            self.jevis_until = now + 8
            self.log("phase_jev_handoff", tick=now, progressing=True, probability=hp)
        elif hp <= 0.25:
            self.jevis_handoff = False
            self.jevis_until = now + 8
            self.log("phase_jev_handoff", tick=now, progressing=False, probability=hp)

    # ----- primitive motor --------------------------------------------------

    def escape_step(self, obs, goal=None):
        me, partner = actors(obs)
        current = point(me.get("position"))
        other = point(partner.get("position"))
        if current is None:
            return self.action(self.noop)
        predicted = self.partner_next(obs)
        corridor, _ = self.partner_corridor(obs, horizon=6)
        options = []
        for nxt, delta in neighbours(current):
            if not open_cell(terrain_of(obs), nxt) or nxt in {other, predicted}:
                continue
            goal_gap = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) if point(goal) else 0
            away = abs(nxt[0] - other[0]) + abs(nxt[1] - other[1]) if other else 0
            overlap = 1 if nxt in corridor else 0
            reverse = 1 if nxt == self.last_self_pos else 0
            options.append((overlap, -away, goal_gap, reverse, nxt, delta))
        if not options:
            return self.action(self.noop)
        options.sort(key=lambda row: row[:5])
        return self.action(MOVE_NAME.get(options[0][5], self.noop))

    def interaction_ready(self, obs, job):
        kind = job.get("kind")
        target = point(job.get("target"))
        me, _ = actors(obs)
        mine = held_name(me)
        grid = terrain_of(obs)
        if kind == "deliver":
            return mine == "soup" and cell(grid, target) == "S"
        if kind == "fetch_dish":
            return not mine and cell(grid, target) == "D"
        if kind in {"fetch_ingredient", "fetch_lookahead"}:
            return not mine and cell(grid, target) in {"O", "T"}
        if kind == "take_counter":
            return not mine and any(name == job.get("item") and p == target for name, p in counter_objects(obs))
        if kind == "put_counter":
            return mine in INGREDIENTS and target not in {p for _, p in counter_objects(obs)}
        if kind == "stash_dish":
            return mine == "dish" and cell(grid, target) == "X" and target not in {p for _, p in counter_objects(obs)}
        if kind == "retrieve_dish":
            return not mine and object_name(object_at(obs, target)) == "dish"
        pots = {p["pos"]: p for p in pot_facts(obs)}
        pot = pots.get(target)
        if kind == "fill_pot":
            return bool(pot and mine in INGREDIENTS and pot["phase"] in {"empty", "partial", "full_idle"} and mine in pot["missing"])
        if kind == "start_cooking":
            return bool(pot and not mine and pot["phase"] == "full_idle")
        if kind in {"pickup_ready", "approach_ready"}:
            return bool(pot and mine == "dish" and pot["ready"])
        return False

    def motor(self, obs):
        me, partner = actors(obs)
        current = point(me.get("position"))
        partner_pos = point(partner.get("position"))
        if current is None or not self.job:
            return self.action(self.noop)

        # A long physical stall is handled by a short detour, then the job is
        # rebuilt by the next observation.  Carried soup is never abandoned.

        kind = self.job.get("kind")
        target = point(self.job.get("target"))
        if kind == "hold" or target is None:
            return self.action(self.noop)

        grid = terrain_of(obs)
        if kind in {"park", "wait_dish", "yield_ready", "yield_cooking", "park_cooking"}:
            path = bfs(grid, current, [target], {partner_pos} if partner_pos and partner_pos != target else set())
            if path is not None and len(path) > 1:
                nxt = path[1]
                predicted = self.partner_next(obs)
                if nxt in {partner_pos, predicted}:
                    self.blocked += 1
                    return self.action(self.noop if self.blocked < 3 else self.escape_step(obs, target))
                delta = nxt[0] - current[0], nxt[1] - current[1]
                return self.action(MOVE_NAME.get(delta, self.noop))
            return self.action(self.noop)

        pose, path = route_to(grid, current, target, {partner_pos} if partner_pos and partner_pos != target else set())
        if pose is None or path is None:
            self.blocked += 1
            return self.escape_step(obs, target)
        if len(path) > 1:
            nxt = path[1]
            predicted = self.partner_next(obs)
            swap = predicted == current and nxt == partner_pos
            if nxt in {partner_pos, predicted} or swap:
                self.blocked += 1
                if self.blocked <= 2:
                    return self.action(self.noop)
                return self.escape_step(obs, pose)
            self.blocked = max(0, self.blocked - 1)
            delta = nxt[0] - current[0], nxt[1] - current[1]
            return self.action(MOVE_NAME.get(delta, self.noop))

        desired = target[0] - current[0], target[1] - current[1]
        facing = point(me.get("orientation"))
        if desired in MOVE_NAME and facing != desired:
            return self.action(MOVE_NAME[desired])
        if self.interaction_ready(obs, self.job):
            return self.action("interact")
        return self.action(self.noop)

    # ----- realtime loop ----------------------------------------------------

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
            self.refresh_dish_slot(obs)
            self.update_motion(obs)
            self.poll_astra(obs)
            self.poll_jev(obs)
            if self.mode_until < tick_of(obs):
                self.mode = "finish"

            options = self.jobs(obs)
            if not self.astra_started:
                self.start_astra(obs, options)

            old_epoch = self.job_epoch
            expired = bool(self.job and tick_of(obs) - self.job_since > self.job_limit(self.job.get("kind", "")))
            physically_stalled = bool(self.job and self.blocked >= 7 and held_name(actors(obs)[0]) != "soup")
            if self.mode == "rebuild" and self.mode_until >= tick_of(obs) and self.blocked >= 3:
                if not physically_stalled and self.job:
                    self.log("jev_rebuild_apply", tick=tick_of(obs), kind=self.job.get("kind"), blocked=self.blocked)
                physically_stalled = True
            if expired or physically_stalled or not self.current_valid(obs, options):
                if self.job is not None:
                    self.log("buffer_replan", tick=tick_of(obs), old_kind=self.job.get("kind"),
                             blocked=self.blocked, expired=expired)
                self.job = None
            if self.job is None:
                self.commit(obs, options)

            # Ask Jev at semantic boundaries and occasional traffic stalls.  A
            # late answer cannot delay this tick or mutate a target directly.
            if self.job_epoch != old_epoch:
                self.start_jev(obs, options, "phase_boundary")
            elif self.blocked >= 3:
                self.start_jev(obs, options, "physical_stall")
            elif tick_of(obs) - self.jev_tick >= 52:
                self.start_jev(obs, options, "heartbeat")

            action = self.motor(obs)
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
        BufferedKitchen(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
