"""Partner-lease controller for the symbolic Overcooked real-time adaptor.

The controller is deliberately split into a small symbolic scheduler and a
primitive motor.  Its distinctive policy is an ingredient lease: once the
scripted chef is visibly in the middle of supplying a pot, the controlled chef
stops fetching the same ingredient and waits for that transaction to commit.
This prevents the extra-onion race that otherwise leaves the partner parked in
the only pot approach with a useless ingredient in hand.
"""
from __future__ import annotations

import json
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
VECTOR_ACTION = {value: key for key, value in DIRECTIONS.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}


def pair(value, default=(0, 0)):
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (int(value[0]), int(value[1]))
    except Exception:
        pass
    return default


def state_of(obs):
    if isinstance(obs, dict) and isinstance(obs.get("state"), dict):
        return obs["state"]
    return {}


def terrain_of(obs):
    if isinstance(obs, dict) and isinstance(obs.get("terrain_mtx"), list):
        return obs["terrain_mtx"]
    return []


def cell(terrain, pos):
    x, y = pair(pos)
    try:
        row = terrain[y]
        if isinstance(row, (str, list, tuple)):
            return row[x]
    except Exception:
        pass
    return "X"


def feature_positions(terrain, symbol=None):
    answer = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            if symbol is None:
                if value in FEATURES:
                    answer.append((value, (x, y)))
            elif value == symbol:
                answer.append((value, (x, y)))
    return answer


def walkable(terrain, pos):
    value = cell(terrain, pos)
    return value not in FEATURES and value != "X"


def objects_of(obs):
    value = state_of(obs).get("objects", [])
    return value if isinstance(value, list) else []


def object_position(obj):
    return pair(obj.get("position")) if isinstance(obj, dict) else None


def object_name(obj):
    return str(obj.get("name", "")) if isinstance(obj, dict) else ""


def held_object(player):
    if not isinstance(player, dict):
        return None
    value = player.get("held_object")
    return value if isinstance(value, dict) else None


def held_name(player):
    value = held_object(player)
    return object_name(value) if value else ""


def soup_ingredients(obj):
    if not isinstance(obj, dict):
        return []
    values = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(values, list):
        return []
    answer = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name")
        if value is not None:
            answer.append(str(value))
    return answer


def orders_of(state):
    values = state.get("all_orders", []) if isinstance(state, dict) else []
    if not isinstance(values, list) or not values:
        values = state.get("bonus_orders", []) if isinstance(state, dict) else []
    answer = []
    for value in values:
        ingredients = value.get("ingredients") if isinstance(value, dict) else value
        if isinstance(ingredients, list) and ingredients:
            answer.append([str(item) for item in ingredients])
    return answer or [["onion", "onion", "onion"]]


def missing_for(ingredients, recipes):
    """Return the shortest legal recipe completion, preserving recipe order."""
    best_recipe = None
    best_missing = None
    for recipe in recipes:
        have = Counter(str(item) for item in ingredients)
        wanted = Counter(str(item) for item in recipe)
        if any(have[item] > wanted[item] for item in have):
            continue
        missing = []
        for item in recipe:
            if have[item] < wanted[item]:
                missing.append(item)
                have[item] += 1
        if best_missing is None or len(missing) < len(best_missing):
            best_recipe = list(recipe)
            best_missing = missing
    if best_missing is None:
        return None, []
    return best_recipe, best_missing


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class LeaseKitchen:
    """A symbolic scheduler with partner transaction leases and a safe motor."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = getattr(game, "action_spec", None)
        self.action_names = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.action_names and "stay" in self.action_names:
            self.noop = "stay"

        self.system1_off = getattr(ctx, "system1", object()) is None
        self.system2_off = getattr(ctx, "system2", object()) is None
        self.jev_tick = -999
        self.jev_signature = None
        self.jev_choice = None
        self.jev_choice_tick = -999
        self.jev_yield = False
        self.astra_future = None
        self.astra_started = False
        self.astra_hint = {}

        # Each entry is a partner-owned transaction, not a permanent role.
        self.claims = {}
        self.current_tick = 0

        self.previous_position = None
        self.previous_task = None
        self.blocked_ticks = 0
        self.recovery_goal = None
        self.recovery_ticks = 0

    # ---------- observation and geometry ----------

    def players(self, obs):
        value = state_of(obs).get("players", [])
        return value if isinstance(value, list) else []

    def indices(self, obs):
        players = self.players(obs)
        try:
            mine = int(obs.get("controlled_agent_index", 0))
        except Exception:
            mine = 0
        if mine < 0 or mine >= len(players):
            mine = 0
        other = 1 if mine == 0 else 0
        if other >= len(players):
            other = mine
        return mine, other

    def actors(self, obs):
        players = self.players(obs)
        mine, other = self.indices(obs)
        me = players[mine] if mine < len(players) and isinstance(players[mine], dict) else {}
        partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
        return me, partner

    def path(self, terrain, start, goal, blocked=()):
        start = pair(start)
        goal = pair(goal)
        blocked = set(pair(item) for item in blocked)
        blocked.discard(start)
        if start == goal:
            return [start]
        queue = deque([start])
        previous = {start: None}
        # Fixed order makes routes stable across ticks and avoids motor dithering.
        for_current = tuple(DIRECTIONS.values())
        while queue:
            current = queue.popleft()
            for dx, dy in for_current:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in previous or nxt in blocked or not walkable(terrain, nxt):
                    continue
                previous[nxt] = current
                if nxt == goal:
                    result = [nxt]
                    while result[-1] != start:
                        result.append(previous[result[-1]])
                    result.reverse()
                    return result
                queue.append(nxt)
        return None

    def poses(self, terrain, target):
        tx, ty = pair(target)
        result = []
        for dx, dy in DIRECTIONS.values():
            pose = (tx - dx, ty - dy)
            if pose not in result and walkable(terrain, pose):
                result.append(pose)
        return result

    def route_to_feature(self, obs, target, preferred_pose=None, ignore_partner=False):
        terrain = terrain_of(obs)
        me, partner = self.actors(obs)
        start = pair(me.get("position"))
        partner_position = pair(partner.get("position"))
        choices = self.poses(terrain, target)
        if preferred_pose in choices:
            choices.remove(preferred_pose)
            choices.insert(0, preferred_pose)
        blocked = () if ignore_partner else (partner_position,)
        routes = []
        for pose in choices:
            route = self.path(terrain, start, pose, blocked=blocked)
            if route is not None:
                routes.append((len(route), pose, route))
        if not routes:
            return None, choices
        routes.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
        _, pose, route = routes[0]
        return {"route": route, "pose": pose}, choices

    def shortest_partner_route(self, obs, target):
        terrain = terrain_of(obs)
        _, partner = self.actors(obs)
        routes = []
        for pose in self.poses(terrain, target):
            route = self.path(terrain, pair(partner.get("position")), pose)
            if route is not None:
                routes.append(route)
        if not routes:
            return None
        routes.sort(key=lambda route: (len(route), route[-1]))
        return routes[0]

    # ---------- exact symbolic ledger ----------

    def pots(self, obs):
        terrain = terrain_of(obs)
        state = state_of(obs)
        recipes = orders_of(state)
        at_position = {}
        for obj in objects_of(obs):
            position = object_position(obj)
            if position is not None:
                at_position.setdefault(position, []).append(obj)

        result = []
        for _, position in feature_positions(terrain, "P"):
            soup = None
            for obj in at_position.get(position, []):
                if object_name(obj) == "soup":
                    soup = obj
                    break
            ingredients = soup_ingredients(soup)
            recipe, missing = missing_for(ingredients, recipes)
            ready = bool(soup and soup.get("is_ready"))
            cooking = bool(soup and soup.get("is_cooking"))
            if soup and not ready and not cooking:
                cook_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1))
                cook_time = soup.get("cook_time", -1)
                if isinstance(cook_tick, (int, float)) and cook_tick >= 0 and cook_time not in (-1, None):
                    cooking = not bool(soup.get("is_idle", False))
            full = recipe is not None and not missing
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
            result.append({
                "id": "pot_%d_%d" % position,
                "pos": position,
                "ingredients": ingredients,
                "recipe": recipe or [],
                "missing": missing,
                "phase": phase,
                "ready": ready,
                "cooking": cooking,
                "full": full,
                "object": soup,
            })
        return result

    def counter_objects(self, obs):
        terrain = terrain_of(obs)
        result = []
        for obj in objects_of(obs):
            name = object_name(obj)
            position = object_position(obj)
            if name in INGREDIENTS | {"dish", "soup"} and position is not None and cell(terrain, position) == "X":
                result.append({"name": name, "pos": position, "object": obj})
        return result

    def free_counters(self, obs):
        terrain = terrain_of(obs)
        occupied = {object_position(obj) for obj in objects_of(obs) if object_position(obj) is not None}
        result = []
        for _, position in feature_positions(terrain, "X"):
            if position in occupied:
                continue
            if any(walkable(terrain, (position[0] + dx, position[1] + dy)) for dx, dy in DIRECTIONS.values()):
                result.append(position)
        return result

    def tick(self, obs):
        value = obs.get("tick", state_of(obs).get("timestep", 0)) if isinstance(obs, dict) else 0
        try:
            return int(value)
        except Exception:
            return 0

    # ---------- partner transaction lease ----------

    def partner_claim_base(self, obs, pot):
        """Whether observations support the partner owning the next pot fill."""
        if pot["phase"] not in {"empty", "partial"} or not pot["missing"]:
            return False
        me, partner = self.actors(obs)
        partner_position = pair(partner.get("position"))
        partner_held = held_name(partner)
        target = pot["pos"]
        poses = self.poses(terrain_of(obs), target)

        # A carrying partner with a matching ingredient has an explicit lease.
        # With multiple pots, assign that lease only to the partner's nearest
        # compatible pot rather than suppressing every parallel pipeline.
        if partner_held in INGREDIENTS:
            eligible = []
            for other in self.pots(obs):
                if other["phase"] not in {"empty", "partial"} or partner_held not in other["missing"]:
                    continue
                route = self.shortest_partner_route(obs, other["pos"])
                if route is None:
                    # An unreachable pot can still be served by a counter
                    # handoff; Manhattan distance is only a tie-breaker.
                    rank = 1000 + abs(partner_position[0] - other["pos"][0]) + abs(partner_position[1] - other["pos"][1])
                else:
                    rank = len(route)
                eligible.append((rank, other["pos"], other["id"], route))
            eligible.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
            if not eligible or eligible[0][2] != pot["id"]:
                return False
            # Any matching held ingredient is an ownership signal, even when
            # the partner must first put it on a counter.  The lease timeout,
            # rather than a map-specific distance constant, handles a partner
            # that is wandering or blocked.
            return True

        # After an interaction at a partial pot, GreedyHumanModel normally
        # leaves empty-handed to fetch the next ingredient.  Treating this as a
        # one-transaction reservation prevents the controlled chef from racing
        # the same fetch on the following tick.  Again choose only one pot if
        # several approach poses happen to overlap.
        if not partner_held and pot["phase"] == "partial" and partner_position in poses:
            eligible = [
                other for other in self.pots(obs)
                if other["phase"] == "partial" and partner_position in self.poses(terrain_of(obs), other["pos"])
            ]
            eligible.sort(key=lambda other: (len(other["missing"]), other["pos"]))
            return bool(eligible and eligible[0]["id"] == pot["id"])
        return False

    def update_claims(self, obs, pots):
        now = self.tick(obs)
        for pot in pots:
            key = pot["id"]
            base = self.partner_claim_base(obs, pot)
            pot_signature = (pot["phase"], tuple(pot["ingredients"]))
            record = self.claims.get(key)

            if pot["phase"] not in {"empty", "partial"} or not pot["missing"]:
                self.claims.pop(key, None)
                continue

            if record is None:
                if base:
                    record = {"pot_signature": pot_signature, "since": now, "expired": False}
                    self.claims[key] = record
            elif record.get("pot_signature") != pot_signature:
                # If a previously leased partner just added an ingredient,
                # transfer the lease to its next fetch even after it has left
                # the approach.  Otherwise a one-tick departure would reopen
                # the duplicate-fetch race.
                old_signature = record.get("pot_signature", ("", ()))
                old_ingredients = tuple(old_signature[1]) if isinstance(old_signature, tuple) and len(old_signature) > 1 else ()
                grew = len(pot["ingredients"]) > len(old_ingredients)
                if base or (not record.get("expired") and grew):
                    record = {"pot_signature": pot_signature, "since": now, "expired": False}
                    self.claims[key] = record
                else:
                    self.claims.pop(key, None)
                    record = None
            # Once a lease exists, partner movement alone does not cancel it.
            # The pot signature or the bounded timeout is the invalidation.
            if record is not None and not record.get("expired") and now - int(record.get("since", now)) > 18:
                record["expired"] = True

    def claim_active(self, pot):
        record = self.claims.get(pot["id"])
        return bool(record and not record.get("expired"))

    # ---------- task construction ----------

    def task(self, task_id, kind, target=None, ingredient=None, description="", pot_id=None, wait=False, pose=None):
        return {
            "id": str(task_id),
            "kind": str(kind),
            "target": list(target) if target is not None else None,
            "ingredient": ingredient,
            "description": description or str(kind),
            "pot_id": pot_id,
            "wait": bool(wait),
            "pose": list(pose) if pose is not None else None,
        }

    def task_candidates(self, obs, pots=None):
        if pots is None:
            pots = self.pots(obs)
        terrain = terrain_of(obs)
        me, partner = self.actors(obs)
        mine = held_name(me)
        partner_item = held_name(partner)
        tasks = []

        # Carrying soup is a committed delivery transaction.
        if mine == "soup":
            for _, position in feature_positions(terrain, "S"):
                tasks.append(self.task(
                    "deliver_%d_%d" % position, "deliver", position,
                    description="deliver the held soup at serving window %s" % (position,),
                ))
            return tasks or [self.task("wait", "wait", description="wait safely")]

        # A dish holder either takes a ready soup or reserves its interaction
        # pose while cooking.  It never fetches another object.
        if mine == "dish":
            ready = [pot for pot in pots if pot["ready"]]
            if ready:
                for pot in ready:
                    x, y = pot["pos"]
                    tasks.append(self.task(
                        "pickup_%s" % pot["id"], "pickup_ready", pot["pos"],
                        description="pick up the ready soup in pot %s" % (pot["pos"],),
                        pot_id=pot["id"],
                    ))
                return tasks
            cooking = [pot for pot in pots if pot["cooking"]]
            for pot in cooking:
                x, y = pot["pos"]
                tasks.append(self.task(
                    "reserve_%s" % pot["id"], "stage_dish", pot["pos"],
                    description="hold the dish at the cooking pot approach %s until ready" % (pot["pos"],),
                    pot_id=pot["id"], wait=True,
                ))
            return tasks or [self.task("hold_dish", "wait", description="hold the dish without blocking")]

        # A held ingredient has a legal destination or goes to a counter.  Do
        # not ask Jev to decide whether an interaction is legal.
        if mine in INGREDIENTS:
            compatible = [
                pot for pot in pots
                if mine in pot["missing"] and pot["phase"] in {"empty", "partial"}
            ]
            compatible.sort(key=lambda pot: (len(pot["missing"]), pot["pos"]))
            for pot in compatible:
                x, y = pot["pos"]
                tasks.append(self.task(
                    "fill_%s" % pot["id"], "fill_pot", pot["pos"], mine,
                    description="put the held %s into pot %s" % (mine, pot["pos"]),
                    pot_id=pot["id"],
                ))
            if not tasks:
                for position in self.free_counters(obs)[:4]:
                    tasks.append(self.task(
                        "drop_%s_%d_%d" % (mine, position[0], position[1]),
                        "deposit_counter", position, mine,
                        description="put the surplus %s on counter %s" % (mine, position),
                    ))
            return tasks or [self.task("hold_ingredient", "wait", description="hold the ingredient safely")]

        ready = [pot for pot in pots if pot["ready"]]
        if ready and partner_item not in {"dish", "soup"}:
            for _, position in feature_positions(terrain, "D"):
                tasks.append(self.task(
                    "dish_for_ready_%d_%d" % position, "fetch_dish", position,
                    description="fetch a dish for a ready soup from %s" % (position,),
                ))
            if tasks:
                return tasks

        # Launching is deliberately before new ingredient fetching.  If the
        # pot is full, an extra partner ingredient is no longer useful; start
        # now so the pipeline can move to dish staging.
        for pot in pots:
            if pot["phase"] == "full_idle":
                tasks.append(self.task(
                    "start_%s" % pot["id"], "start_cooking", pot["pos"],
                    description="start the full idle pot at %s" % (pot["pos"],),
                    pot_id=pot["id"],
                ))
        if tasks:
            return tasks

        # The central structural change: a partner-owned partial pot is not a
        # reason to fetch another copy.  Wait in place until the pot signature
        # changes, then continue normally.  The lease expires after 18 ticks.
        staged_names = {item["name"] for item in self.counter_objects(obs) if item["name"] in INGREDIENTS}
        leased = [
            pot for pot in pots
            if pot["phase"] in {"empty", "partial"}
            and self.claim_active(pot)
            and not any(item in staged_names for item in pot["missing"])
        ]
        if leased:
            for pot in leased:
                tasks.append(self.task(
                    "yield_%s" % pot["id"], "yield_partner", None,
                    description="let the partner finish the reserved fill for pot %s" % (pot["pos"],),
                    pot_id=pot["id"], wait=True,
                ))
            return tasks

        # Build the nearest unfinished pot.  Counter ingredients are preferred
        # because they are already in the shared workspace.
        buildable = [pot for pot in pots if pot["missing"] and pot["phase"] in {"empty", "partial"}]
        buildable.sort(key=lambda pot: (len(pot["missing"]), pot["pos"]))
        if buildable:
            pot = buildable[0]
            ingredient = pot["missing"][0]
            staged = [item for item in self.counter_objects(obs) if item["name"] == ingredient]
            for item in staged[:3]:
                x, y = item["pos"]
                tasks.append(self.task(
                    "take_%s_%d_%d" % (ingredient, x, y), "fetch_counter", item["pos"], ingredient,
                    description="take the staged %s from counter %s" % (ingredient, item["pos"]),
                    pot_id=pot["id"],
                ))
            dispenser = "O" if ingredient == "onion" else "T"
            for _, position in feature_positions(terrain, dispenser):
                tasks.append(self.task(
                    "fetch_%s_%d_%d" % (ingredient, position[0], position[1]),
                    "fetch_ingredient", position, ingredient,
                    description="fetch one %s from dispenser %s" % (ingredient, position),
                    pot_id=pot["id"],
                ))
            if tasks:
                return tasks

        # During cooking, stage a dish unless the partner already owns the
        # dish/soup transaction.  A dish is never duplicated unnecessarily.
        if any(pot["cooking"] for pot in pots) and partner_item not in {"dish", "soup"}:
            for _, position in feature_positions(terrain, "D"):
                tasks.append(self.task(
                    "dish_during_cook_%d_%d" % position, "fetch_dish", position,
                    description="fetch a dish while the pot cooks from %s" % (position,),
                ))
            if tasks:
                return tasks

        if partner_item not in {"dish", "soup"}:
            for _, position in feature_positions(terrain, "D"):
                tasks.append(self.task(
                    "pipeline_dish_%d_%d" % position, "fetch_dish", position,
                    description="fetch a spare dish for the next soup from %s" % (position,),
                ))
        return tasks or [self.task("wait", "wait", description="wait safely for the kitchen")]

    # ---------- model interfaces ----------

    def compact(self, obs, tasks, pots):
        me, partner = self.actors(obs)
        claim_view = []
        for pot in pots:
            record = self.claims.get(pot["id"])
            claim_view.append({
                "pot": pot["pos"],
                "phase": pot["phase"],
                "ingredients": pot["ingredients"],
                "missing": pot["missing"],
                "partner_lease": bool(record and not record.get("expired")),
            })
        return jsonable({
            "tick": self.tick(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {
                "position": pair(me.get("position")),
                "facing": pair(me.get("orientation")),
                "held": held_name(me),
            },
            "partner": {
                "position": pair(partner.get("position")),
                "facing": pair(partner.get("orientation")),
                "held": held_name(partner),
                "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else "",
                "task": obs.get("partner_task", "") if isinstance(obs, dict) else "",
            },
            "pots": claim_view,
            "orders": orders_of(state_of(obs)),
            "recent_events": (obs.get("last_events", []) if isinstance(obs, dict) else [])[-6:],
            "candidates": [{"id": t["id"], "kind": t["kind"], "description": t["description"]} for t in tasks[:8]],
        })

    def decision_signature(self, obs, tasks, pots):
        me, partner = self.actors(obs)
        return (
            held_name(me), held_name(partner),
            tuple((p["id"], p["phase"], tuple(p["ingredients"]), self.claim_active(p)) for p in pots),
            tuple((task["id"], task["kind"]) for task in tasks[:4]),
        )

    def ask_jev(self, obs, tasks, pots):
        if self.system1_off or not tasks or self.ctx.done:
            return
        now = self.tick(obs)
        criteria = {task["id"]: task["description"] for task in tasks[:8]}
        if not criteria:
            return
        questions = {
            "task": Choice(
                "Choose only one listed candidate for the controlled chef. The candidate list was generated from the current symbolic state and all candidates are legal. Prefer the first candidate unless a later candidate is a clearly equivalent route. Never invent an id or a primitive direction.",
                criteria,
            ),
            "lease": Noul(
                "Does the observation show that the partner should be allowed to complete its current ingredient transaction before the controlled chef starts another ingredient run? Use partner.lease and the pot phase; answer true for a visible partner lease, otherwise false.",
                {
                    "true": "A partial or empty pot has a short-lived partner-owned fill lease.",
                    "false": "No partner-owned fill lease is established.",
                },
            ),
        }
        try:
            answers = self.ctx.decide(
                {"game": "Overcooked-AI", "observation": self.compact(obs, tasks, pots)},
                questions,
                timeout_s=3.0,
            )
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            self.system1_off = True
            return
        except Exception:
            # A failed or malformed recognition result never owns the motor.
            self.jev_tick = now
            self.jev_signature = self.decision_signature(obs, tasks, pots)
            return

        self.jev_tick = now
        self.jev_signature = self.decision_signature(obs, tasks, pots)
        try:
            answer = answers.get("task")
            choice = getattr(answer, "value", None)
            probs = getattr(answer, "probabilities", {}) or {}
            if isinstance(probs, dict):
                probability = float(probs.get(choice, 0.0))
            else:
                probability = 0.0
            valid = {task["id"]: task for task in tasks}
            safe = tasks[0]
            selected = valid.get(choice)
            # Jev may break ties within the deterministic phase, but cannot
            # bypass a lease or turn a delivery into speculative work.
            if selected is not None and (probability >= 0.40 or probability == 0.0):
                if selected["kind"] == safe["kind"] or (safe["kind"] == "fetch_ingredient" and selected["kind"] == "fetch_counter"):
                    self.jev_choice = selected["id"]
                    self.jev_choice_tick = now
            lease = answers.get("lease")
            lease_value = getattr(lease, "value", 0.0)
            if isinstance(lease_value, (int, float)) and lease_value >= 0.70:
                if any(self.claim_active(pot) for pot in pots):
                    self.jev_yield = True
        except Exception:
            pass

    def start_astra(self, obs, tasks, pots):
        if self.astra_started or self.system2_off or self.ctx.done:
            return
        self.astra_started = True
        compact = self.compact(obs, tasks, pots)
        prompt = (
            "You are a slow strategic reviewer for symbolic Overcooked-AI. The executable "
            "controller owns geometry and legality. Diagnose the partner transaction leases "
            "and choose only a short horizon role policy. Return one compact JSON object with "
            "preferred_kind (one of yield_partner, fetch_ingredient, fetch_dish, stage_dish, "
            "start_cooking, pickup_ready, deliver, wait) and a brief replan_trigger. Never "
            "output directions, arithmetic, or invented task ids. Preserve a ready soup and "
            "never race a partner-owned partial pot.\n" + json.dumps(compact, separators=(",", ":"))
        )

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

    def poll_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = getattr(result, "text", result if isinstance(result, str) else "")
            if not isinstance(text, str):
                return
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                self.astra_hint = {
                    key: value[key]
                    for key in ("preferred_kind", "replan_trigger")
                    if isinstance(value.get(key), str)
                }
        except Exception:
            self.astra_hint = {}

    # ---------- task choice and primitive motor ----------

    def choose_task(self, tasks, now):
        if not tasks:
            return self.task("wait", "wait", description="wait safely")
        safe = tasks[0]
        valid = {task["id"]: task for task in tasks}
        selected = valid.get(self.jev_choice)
        if selected is not None and now - self.jev_choice_tick <= 10 and selected["kind"] == safe["kind"]:
            return selected
        preferred_kind = self.astra_hint.get("preferred_kind")
        if preferred_kind and preferred_kind == safe["kind"]:
            for task in tasks:
                if task["kind"] == preferred_kind:
                    return task
        return safe

    def recovery_action(self, obs, task):
        terrain = terrain_of(obs)
        me, partner = self.actors(obs)
        start = pair(me.get("position"))
        partner_pos = pair(partner.get("position"))
        target = pair(task.get("target")) if task.get("target") is not None else None
        forbidden = set(self.poses(terrain, target)) if target is not None else set()
        candidates = []
        # Select a real parking cell by global route, not a local left/right
        # oscillation.  It should be away from both the partner and the target.
        for y, row in enumerate(terrain):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                position = (x, y)
                if not walkable(terrain, position) or position in {partner_pos} or position in forbidden:
                    continue
                distance_partner = abs(x - partner_pos[0]) + abs(y - partner_pos[1])
                distance_target = abs(x - target[0]) + abs(y - target[1]) if target is not None else 0
                if distance_partner < 2:
                    continue
                route = self.path(terrain, start, position, blocked={partner_pos})
                if route is not None:
                    candidates.append((-(distance_partner + distance_target), len(route), position, route))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            route = candidates[0][3]
            if len(route) > 1:
                dx = route[1][0] - start[0]
                dy = route[1][1] - start[1]
                return self.action(VECTOR_ACTION.get((dx, dy), self.noop))
        return self.action(self.noop)

    def motor(self, obs, task):
        me, partner = self.actors(obs)
        mine = held_name(me)
        if task.get("kind") in {"wait", "yield_partner"} or task.get("target") is None:
            return self.action(self.noop)

        target = pair(task.get("target"))
        preferred_pose = pair(task.get("pose")) if task.get("pose") is not None else None
        selected, poses = self.route_to_feature(obs, target, preferred_pose=preferred_pose)
        current = pair(me.get("position"))
        partner_pos = pair(partner.get("position"))
        if selected is None:
            self.recovery_ticks += 1
            # For a high-value ready soup, a short wait lets a moving partner
            # clear the approach; for ordinary work, use a global parking path.
            if task.get("kind") in {"pickup_ready", "deliver"} and self.recovery_ticks <= 5:
                return self.action(self.noop)
            return self.recovery_action(obs, task)

        self.recovery_ticks = 0
        route = selected["route"]
        pose = selected["pose"]
        if len(route) > 1:
            nxt = route[1]
            if nxt == partner_pos:
                self.recovery_ticks += 1
                return self.action(self.noop)
            delta = (nxt[0] - current[0], nxt[1] - current[1])
            return self.action(VECTOR_ACTION.get(delta, self.noop))

        desired_delta = (target[0] - pose[0], target[1] - pose[1])
        desired_action = VECTOR_ACTION.get(desired_delta, self.noop)
        if pair(me.get("orientation")) != desired_delta:
            return self.action(desired_action)
        if task.get("wait"):
            return self.action(self.noop)
        return self.action("interact")

    def action(self, name):
        if name in self.action_names:
            return name
        if self.noop in self.action_names:
            return self.noop
        if self.action_names:
            return sorted(self.action_names)[0]
        return name

    def should_ask(self, obs, tasks, pots):
        if self.system1_off or not tasks:
            return False
        now = self.tick(obs)
        signature = self.decision_signature(obs, tasks, pots)
        if signature == self.jev_signature:
            return False
        urgent = tasks[0]["kind"] in {"pickup_ready", "start_cooking", "yield_partner"}
        return now - self.jev_tick >= (6 if urgent else 18)

    # ---------- realtime loop ----------

    def run(self):
        while not self.ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self.current_tick = self.tick(obs)
            pots = self.pots(obs)
            self.update_claims(obs, pots)
            tasks = self.task_candidates(obs, pots)
            if not self.astra_started:
                self.start_astra(obs, tasks, pots)
            self.poll_astra()
            task = self.choose_task(tasks, self.current_tick)

            me, _ = self.actors(obs)
            position = pair(me.get("position"))
            if position == self.previous_position and task.get("id") == self.previous_task and task.get("kind") not in {"wait", "yield_partner"}:
                self.blocked_ticks += 1
            else:
                self.blocked_ticks = 0
            self.previous_position = position
            self.previous_task = task.get("id")
            if self.blocked_ticks >= 8:
                self.recovery_ticks = max(self.recovery_ticks, 1)
                self.blocked_ticks = 0

            action = self.motor(obs, task)
            try:
                self.ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                try:
                    self.ctx.act(self.action(self.noop))
                except Exception:
                    return

            if self.should_ask(obs, tasks, pots):
                self.ask_jev(obs, tasks, pots)
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                return


def build(ctx):
    class Harness:
        def __init__(self, context):
            self.ctx = context

        def run_episode(self, game):
            if not getattr(game, "tick_s", None):
                return
            LeaseKitchen(self.ctx, game).run()

    return Harness(ctx)
