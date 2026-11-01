"""Candidate 0016: a role-and-handoff controller for realtime Overcooked-AI.

The executable part is deliberately a small event-driven finite-state machine.
It owns geometry, recipe matching, interaction legality, and primitive motion.
Astra writes a one-shot kitchen protocol; Jev adjudicates only a disputed role
boundary (feed the pot, take a delivery, or yield briefly).  Neither model is
on the realtime tick path and neither can emit a primitive action.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
ACTION_FOR = {v: k for k, v in DIRS.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}


def pos(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        if isinstance(value[0], bool) or isinstance(value[1], bool):
            return None
        return int(value[0]), int(value[1])
    except Exception:
        return None


def safe(value):
    """Convert observations and stand-in values to strict JSON."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
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


def cell(terrain, p):
    p = pos(p)
    if p is None:
        return "X"
    x, y = p
    try:
        if y < 0 or y >= len(terrain):
            return "X"
        row = terrain[y]
        if isinstance(row, (str, list, tuple)) and 0 <= x < len(row):
            return row[x]
    except Exception:
        pass
    return "X"


def walkable(terrain, p):
    return cell(terrain, p) not in FEATURES and cell(terrain, p) != "X"


def features(terrain, symbol):
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
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
    if mine not in (0, 1) or mine >= len(players):
        mine = 0
    other = 1 - mine if len(players) > 1 else mine
    me = players[mine] if mine < len(players) and isinstance(players[mine], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, partner


def held(player):
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


def ingredients_of(obj):
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


def orders_of(obs):
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


def missing_for(ingredients, recipes):
    best_recipe, best_missing = [], []
    for recipe in recipes:
        have = Counter(ingredients)
        wanted = Counter(recipe)
        if any(have[k] > wanted[k] for k in have):
            continue
        missing = []
        for item in recipe:
            if have[item] < wanted[item]:
                missing.append(item)
                have[item] += 1
        if not best_missing or len(missing) < len(best_missing):
            best_recipe, best_missing = list(recipe), missing
    return best_recipe, best_missing


def pot_view(obs):
    terrain = terrain_of(obs)
    soups = {}
    for obj in objects_of(obs):
        p = object_pos(obj)
        if p is not None and object_name(obj) == "soup":
            soups[p] = obj
    recipes = orders_of(obs)
    result = []
    for p in features(terrain, "P"):
        soup = soups.get(p)
        ingredients = ingredients_of(soup)
        recipe, missing = missing_for(ingredients, recipes)
        ready = bool(soup and soup.get("is_ready"))
        cooking = bool(soup and soup.get("is_cooking")) and not ready
        raw_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        cook_time = soup.get("cook_time", -1) if soup else -1
        if not ready and not cooking and isinstance(raw_tick, (int, float)) and isinstance(cook_time, (int, float)):
            cooking = raw_tick >= 0 and cook_time >= 0
        full = bool(soup and recipe and not missing)
        phase = "ready" if ready else "cooking" if cooking else "full_idle" if full else "partial" if ingredients else "empty"
        remaining = None
        if cooking and isinstance(raw_tick, (int, float)) and isinstance(cook_time, (int, float)):
            remaining = max(0, int(cook_time - raw_tick))
        result.append({
            "id": "pot_%d_%d" % p,
            "pos": p,
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


def neighbors(p):
    x, y = p
    for delta in DIRS.values():
        yield (x + delta[0], y + delta[1]), delta


def bfs(terrain, start, goal, blocked=()):
    start, goal = pos(start), pos(goal)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    blocked = {pos(x) for x in blocked if pos(x) is not None}
    blocked.discard(start)
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        for nxt, _ in neighbors(current):
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
    target = pos(target)
    if target is None:
        return []
    result = []
    for delta in DIRS.values():
        approach = (target[0] - delta[0], target[1] - delta[1])
        if approach not in result and walkable(terrain, approach):
            result.append(approach)
    return result


def shortest_pose(terrain, start, target, blocked=()):
    choices = []
    for approach in interaction_poses(terrain, target):
        route = bfs(terrain, start, approach, blocked)
        if route is not None:
            choices.append((len(route), approach, route))
    if choices:
        choices.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
        return choices[0][1], choices[0][2]
    # The other chef is dynamic, not a wall.  A route without it is still useful
    # for a bounded traffic detour.
    for approach in interaction_poses(terrain, target):
        route = bfs(terrain, start, approach)
        if route is not None:
            choices.append((len(route), approach, route))
    if choices:
        choices.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
        return choices[0][1], choices[0][2]
    return None, None


def counter_sites(obs):
    terrain = terrain_of(obs)
    occupied = {object_pos(o) for o in objects_of(obs) if object_pos(o) is not None}
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            p = (x, y)
            if value == "X" and p not in occupied and any(walkable(terrain, q) for q, _ in neighbors(p)):
                result.append(p)
    return result


def counter_items(obs):
    terrain = terrain_of(obs)
    result = []
    for obj in objects_of(obs):
        p, name = object_pos(obj), object_name(obj)
        if p is not None and name in INGREDIENTS and cell(terrain, p) == "X":
            result.append((name, p))
    return result


def parse_object(text):
    if not isinstance(text, str):
        return {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        result = json.loads(match.group(0))
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


class RoleKitchen:
    """A stable role machine with code-owned physical transactions."""

    def __init__(self, ctx, game):
        self.ctx, self.game = ctx, game
        spec = getattr(game, "action_spec", None)
        self.names = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.names:
            self.noop = "stay" if "stay" in self.names else (next(iter(self.names), "stay"))

        self.job = None
        self.job_epoch = 0
        self.blocked = 0
        self.previous_position = None
        self.previous_partner_signature = None
        self.partner_idle = 0
        self.partner_progress_tick = 0
        self.last_action = self.noop

        # Astra authors a protocol, not an action plan.
        self.astra_future = None
        self.astra_started = False
        self.protocol = {"role": "adaptive", "circulation": "nearest", "handoff": "takeover"}

        # Jev only arbitrates the current role boundary.
        self.s1_off = getattr(ctx, "system1", object()) is None
        self.jev_future = None
        self.jev_epoch = -1
        self.jev_request_tick = -999
        self.role_hint = None
        self.role_hint_until = -1
        self.takeover_hint = False

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def action(self, value):
        if value in self.names:
            return value
        if self.noop in self.names:
            return self.noop
        return sorted(self.names)[0] if self.names else "stay"

    def partner_state(self, obs):
        _, partner = actors(obs)
        signature = (pos(partner.get("position")), held(partner))
        if self.previous_partner_signature is None:
            self.previous_partner_signature = signature
            self.partner_progress_tick = tick_of(obs)
        elif signature != self.previous_partner_signature:
            self.partner_idle = 0
            self.partner_progress_tick = tick_of(obs)
            self.previous_partner_signature = signature
        else:
            self.partner_idle += 1
        return signature

    def protocol_prompt(self, obs):
        terrain = terrain_of(obs)
        pots = pot_view(obs)
        me, partner = actors(obs)
        return (
            "You are Astra, a slow strategy reviewer for a realtime symbolic "
            "Overcooked-AI kitchen. The executable controller already owns exact "
            "recipes, BFS, interaction legality, and primitive actions. Author only "
            "a short protocol for one controlled chef paired with a scripted partner. "
            "Choose role adaptive|supplier|delivery, circulation nearest|left|right, "
            "and handoff yield|takeover. Supplier means fill/start pots and avoid "
            "stealing a partner's active dish transaction. Delivery means take a ready "
            "soup when the partner visibly stalls. Return ONLY JSON with those keys; "
            "never output directions or actions.\n" + json.dumps(safe({
                "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
                "terrain": terrain,
                "self": {"position": pos(me.get("position")), "held": held(me)},
                "partner": {"position": pos(partner.get("position")), "held": held(partner),
                            "task": obs.get("partner_task", "") if isinstance(obs, dict) else ""},
                "pots": [{"position": p["pos"], "phase": p["phase"],
                          "missing": p["missing"], "remaining": p["remaining"]} for p in pots],
                "orders": orders_of(obs),
            }), separators=(",", ":"))
        )

    def start_astra(self, obs):
        if self.astra_started or getattr(self.ctx, "system2", object()) is None or self.ctx.done:
            return
        self.astra_started = True
        prompt = self.protocol_prompt(obs)

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
            value = future.result()
            data = parse_object(value if isinstance(value, str) else getattr(value, "text", ""))
            role = data.get("role")
            circ = data.get("circulation")
            handoff = data.get("handoff")
            if role in {"adaptive", "supplier", "delivery"}:
                self.protocol["role"] = role
            if circ in {"nearest", "left", "right"}:
                self.protocol["circulation"] = circ
            if handoff in {"yield", "takeover"}:
                self.protocol["handoff"] = handoff
            self.log("astra_protocol", tick=tick_of(obs), protocol=dict(self.protocol))
        except Exception:
            return

    def pot_for(self, pots, p):
        p = pos(p)
        return next((pot for pot in pots if pot.get("pos") == p), None)

    def partner_near_pot(self, obs, pot):
        if not pot:
            return False
        _, partner = actors(obs)
        pp = pos(partner.get("position"))
        return pp in set(interaction_poses(terrain_of(obs), pot.get("pos"))) if pp is not None else False

    def ready_transaction(self, obs, pots):
        """Return whether delivery should be left to the scripted partner."""
        _, partner = actors(obs)
        partner_held = held(partner)
        ready = [p for p in pots if p.get("ready")]
        if not ready:
            return False
        if partner_held == "soup":
            return True
        if partner_held == "dish":
            # A dish is a claim only while the partner is visibly progressing.
            if self.partner_idle < 8:
                return True
            if any(self.partner_near_pot(obs, p) for p in ready):
                return True
        return False

    def choose_focus(self, obs, pots):
        terrain = terrain_of(obs)
        me, partner = actors(obs)
        start = pos(me.get("position"))
        pp = pos(partner.get("position"))
        candidates = []
        for pot in pots:
            if pot["phase"] in {"empty", "partial"} and pot["missing"]:
                pose, route = shortest_pose(terrain, start, pot["pos"], {pp} if pp else ())
                candidates.append((len(route) if route else 999, len(pot["missing"]), pot["pos"], pot))
        for pot in pots:
            if pot["phase"] == "full_idle":
                pose, route = shortest_pose(terrain, start, pot["pos"], {pp} if pp else ())
                candidates.append((-100, 0, pot["pos"], pot))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return candidates[0][3] if candidates else None

    def task(self, kind, target=None, item="", pot=None, priority=10, wait=False):
        target = pos(target)
        suffix = "_".join(map(str, target)) if target is not None else "none"
        pid = pot.get("id", "") if isinstance(pot, dict) else ""
        return {"id": "%s_%s_%s" % (kind, suffix, pid), "kind": kind,
                "target": target, "item": item, "pot_id": pid,
                "priority": priority, "wait": bool(wait)}

    def tasks(self, obs, pots):
        terrain = terrain_of(obs)
        me, partner = actors(obs)
        mine, theirs = held(me), held(partner)
        result = []

        # Held-object branches are never delegated to a model.
        if mine == "soup":
            return [self.task("deliver", p, priority=0) for p in features(terrain, "S")] or [self.task("wait", wait=True)]
        if mine == "dish":
            ready = [p for p in pots if p["ready"]]
            if ready:
                return [self.task("pickup_ready", p["pos"], pot=p, priority=0) for p in ready]
            cooking = [p for p in pots if p["cooking"]]
            if cooking:
                cooking.sort(key=lambda p: (p["remaining"] if isinstance(p["remaining"], (int, float)) else 999, p["pos"]))
                return [self.task("wait_ready", cooking[0]["pos"], pot=cooking[0], priority=1, wait=True)]
            return [self.task("wait", wait=True)]
        if mine in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial"} and mine in p["missing"]]
            compatible.sort(key=lambda p: (len(p["missing"]), p["pos"]))
            if compatible:
                return [self.task("fill_pot", p["pos"], mine, p, priority=1) for p in compatible]
            free = counter_sites(obs)
            return [self.task("deposit_counter", p, mine, priority=6) for p in free[:4]] or [self.task("wait", wait=True)]

        # Ready delivery is the only branch that can supersede production.  The
        # partner gets a short, observable claim, never an indefinite exclusion.
        ready = [p for p in pots if p["ready"]]
        partner_claim = self.ready_transaction(obs, pots)
        takeover = bool(ready and not partner_claim)
        if ready and takeover:
            for d in features(terrain, "D"):
                result.append(self.task("fetch_dish", d, priority=0))
            if result:
                return result

        # A full idle pot is an immediate transition; this fixes the old pattern
        # of buffering an extra ingredient before starting the only available pot.
        full = [p for p in pots if p["phase"] == "full_idle"]
        if full:
            full.sort(key=lambda p: p["pos"])
            return [self.task("start_cooking", p["pos"], pot=p, priority=1) for p in full]

        # The normal role is supplier.  Only ask for a dish during cooking when
        # the scripted partner has not claimed it, or after a bounded stall.
        unfinished = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]]
        if unfinished:
            # Avoid duplicating a carried partner ingredient when another pot is
            # available.  This is a soft observation, not a lease.
            if theirs in INGREDIENTS:
                alternate = [p for p in unfinished if theirs not in p["missing"]]
                if alternate:
                    unfinished = alternate
            unfinished.sort(key=lambda p: (len(p["missing"]), p["pos"]))
            target_pot = unfinished[0]
            wanted = target_pot["missing"][0]
            for name, p in counter_items(obs):
                if name == wanted:
                    result.append(self.task("fetch_counter", p, wanted, target_pot, priority=3))
            symbol = "O" if wanted == "onion" else "T"
            for source in features(terrain, symbol):
                result.append(self.task("fetch_ingredient", source, wanted, target_pot, priority=4))
            if result:
                return result

        cooking = [p for p in pots if p["cooking"]]
        cooking.sort(key=lambda p: (p["remaining"] if isinstance(p["remaining"], (int, float)) else 999, p["pos"]))
        if cooking and not partner_claim:
            remaining = cooking[0].get("remaining")
            # Dish takeover is driven by the observed partner transaction rather
            # than a permanently staged gate.  With no timer, fetch conservatively.
            due = remaining is None or remaining <= 8
            if due or self.protocol.get("role") == "delivery":
                for d in features(terrain, "D"):
                    result.append(self.task("fetch_dish", d, priority=3))
                if result:
                    return result

        # Park away from all pot approaches.  This is a productive role boundary:
        # the next observation can immediately select a ready/partial transition.
        if cooking:
            parking = self.parking_cell(obs, cooking[0]["pos"])
            return [self.task("park", parking, priority=10, wait=parking is None)]
        return [self.task("wait", wait=True)]

    def parking_cell(self, obs, pot_target):
        terrain = terrain_of(obs)
        me, partner = actors(obs)
        start, other = pos(me.get("position")), pos(partner.get("position"))
        if start is None:
            return None
        forbidden = set(interaction_poses(terrain, pot_target))
        if other is not None:
            forbidden.add(other)
        choices = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                p = (x, y)
                if p in forbidden or not walkable(terrain, p):
                    continue
                route = bfs(terrain, start, p, {other} if other else ())
                if route is None:
                    continue
                # Prefer a cell with at least one route back to a pot approach.
                return_route = any(bfs(terrain, p, q, {other} if other else ()) for q in interaction_poses(terrain, pot_target))
                if not return_route:
                    continue
                choices.append((len(route), abs(p[0] - pot_target[0]) + abs(p[1] - pot_target[1]), p))
        choices.sort(key=lambda x: (x[0], x[1], x[2]))
        return choices[0][2] if choices else None

    def task_key(self, tasks, pots, obs):
        me, partner = actors(obs)
        return (held(me), held(partner),
                tuple((p["id"], p["phase"], tuple(p["ingredients"])) for p in pots),
                tuple((t["kind"], t.get("target"), t.get("item")) for t in tasks[:5]))

    def compact(self, obs, tasks, pots):
        me, partner = actors(obs)
        return safe({
            "tick": tick_of(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {"position": pos(me.get("position")), "facing": pos(me.get("orientation")), "held": held(me)},
            "partner": {"position": pos(partner.get("position")), "facing": pos(partner.get("orientation")),
                        "held": held(partner), "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else "",
                        "idle_ticks": self.partner_idle},
            "pots": [{"position": p["pos"], "phase": p["phase"], "missing": p["missing"],
                      "remaining": p["remaining"]} for p in pots],
            "protocol": dict(self.protocol),
            "job": {"kind": self.job.get("kind", "") if self.job else "", "target": self.job.get("target") if self.job else None,
                    "blocked": self.blocked},
            "candidates": [{"id": t["id"], "kind": t["kind"], "description": t["kind"]} for t in tasks[:8]],
        })

    def request_jev(self, obs, tasks, pots, reason):
        if self.s1_off or self.jev_future is not None or not tasks or self.ctx.done:
            return
        now = tick_of(obs)
        if now - self.jev_request_tick < 18:
            return
        # Jev is a boundary reviewer, not a per-tick policy.
        self.jev_request_tick = now
        self.jev_epoch = self.job_epoch
        state = {"observation": self.compact(obs, tasks, pots), "reason": reason,
                 "rules": {"ready_delivery_precedes_supply": True,
                           "answers_choose_roles_not_directions": True}}
        questions = {
            "role": Choice(
                "Choose one current role from observation.candidates and the partner evidence. "
                "Use take_delivery only if a ready pot exists and the partner is not making "
                "visible progress; use supply for missing ingredients or a full idle pot; "
                "use yield only to preserve a partner's active dish transaction. Never invent ids.",
                {"supply": "Fill missing ingredients or start a full idle pot; do not fetch a dish.",
                 "take_delivery": "Fetch a dish and take a ready/cooking delivery transaction after a partner stall.",
                 "yield": "Move away or wait briefly while the partner visibly carries the urgent dish/soup."},
            ),
            "takeover": Noul(
                "Is it justified for the controlled chef to take over delivery now? "
                "Look directly at observation.pots, observation.partner.held, and "
                "observation.partner.idle_ticks. True requires a ready pot and no visible "
                "partner progress; it must not cancel a held soup delivery.",
                {"true": "A ready soup is stranded or the partner's dish transaction is stalled.",
                 "false": "The partner is moving with a dish/soup or no ready soup is waiting."},
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
            self.log("jev_request", tick=now, reason=reason, epoch=self.job_epoch)
        except Exception:
            self.jev_future = None

    def poll_jev(self, obs, pots):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict) or self.jev_epoch != self.job_epoch:
            return
        now = tick_of(obs)
        role_answer = answers.get("role")
        role = getattr(role_answer, "value", None)
        probs = getattr(role_answer, "probabilities", {}) or {}
        try:
            probability = float(probs.get(role, 0.0)) if isinstance(probs, dict) else 0.0
        except Exception:
            probability = 0.0
        if role in {"supply", "take_delivery", "yield"} and (probability >= 0.55 or probability == 0.0):
            self.role_hint = role
            self.role_hint_until = now + 24
            self.log("jev_role", tick=now, role=role, probability=probability)
        take = answers.get("takeover")
        try:
            if float(getattr(take, "value", 0.0)) >= 0.70 and any(p["ready"] for p in pots):
                self.takeover_hint = True
        except Exception:
            pass

    def same_task(self, left, right):
        return bool(left and right and left.get("kind") == right.get("kind") and
                    left.get("target") == right.get("target") and left.get("item", "") == right.get("item", ""))

    def legal(self, obs, job):
        me, _ = actors(obs)
        mine = held(me)
        target = pos(job.get("target"))
        if target is None:
            return False
        kind = job.get("kind")
        terrain = terrain_of(obs)
        pots = pot_view(obs)
        pot = self.pot_for(pots, target)
        if kind == "deliver":
            return mine == "soup" and cell(terrain, target) == "S"
        if kind == "fetch_dish":
            return not mine and cell(terrain, target) == "D"
        if kind == "fetch_ingredient":
            return not mine and cell(terrain, target) in {"O", "T"}
        if kind == "fetch_counter":
            return not mine and any(n == job.get("item") and p == target for n, p in counter_items(obs))
        if kind == "deposit_counter":
            return mine in INGREDIENTS and target not in {object_pos(o) for o in objects_of(obs)}
        if kind == "fill_pot":
            return bool(pot and mine in INGREDIENTS and pot["phase"] in {"empty", "partial"} and mine in pot["missing"])
        if kind == "start_cooking":
            return bool(pot and not mine and pot["phase"] == "full_idle")
        if kind == "pickup_ready":
            return bool(pot and mine == "dish" and pot["ready"])
        return False

    def valid_job(self, obs, tasks):
        if not self.job:
            return False
        me, _ = actors(obs)
        mine = held(me)
        kind = self.job.get("kind")
        target = pos(self.job.get("target"))
        if kind in {"park", "wait", "wait_ready"}:
            # A passive mission is valid only while the scheduler still emits
            # the same passive option.  This is the exit condition missing from
            # the old parking controller: a newly ready/partial pot must reopen
            # the production graph on the next observation.
            return any(self.same_task(self.job, task) for task in tasks)
        for task in tasks:
            if self.same_task(self.job, task):
                return True
        if kind in {"fetch_ingredient", "fetch_counter"} and mine in INGREDIENTS:
            return False
        if kind == "fetch_dish" and mine == "dish":
            return False
        if kind == "pickup_ready" and mine == "soup":
            return False
        if kind == "deliver" and mine != "soup":
            return False
        return False

    def choose(self, obs, tasks):
        if not tasks:
            return self.task("wait", wait=True)
        # Hard priority is code-owned.  The role hint can only select an option
        # inside the current safe band, never suppress delivery/start/fill.
        rank = min(int(t.get("priority", 10)) for t in tasks)
        band = [t for t in tasks if int(t.get("priority", 10)) == rank]
        if self.role_hint and tick_of(obs) <= self.role_hint_until:
            if self.role_hint == "take_delivery":
                picks = [t for t in band if t.get("kind") in {"fetch_dish", "pickup_ready", "deliver"}]
                if picks:
                    band = picks
            elif self.role_hint == "yield":
                picks = [t for t in band if t.get("kind") in {"park", "wait", "wait_ready"}]
                if picks:
                    band = picks
            elif self.role_hint == "supply":
                picks = [t for t in band if t.get("kind") in {"fetch_ingredient", "fetch_counter", "fill_pot", "start_cooking"}]
                if picks:
                    band = picks
        me, partner = actors(obs)
        start, other = pos(me.get("position")), pos(partner.get("position"))
        terrain = terrain_of(obs)
        scored = []
        for task in band:
            target = task.get("target")
            if task.get("kind") in {"wait", "wait_ready"}:
                route = [start] if start is not None else None
                approach = start
            elif task.get("kind") == "park":
                route = bfs(terrain, start, target, {other} if other else ()) if target else None
                approach = target
            else:
                approach, route = shortest_pose(terrain, start, target, {other} if other else ())
            scored.append((len(route) if route else 999, str(task.get("id", "")), task.copy(), approach))
        scored.sort(key=lambda row: (row[0], row[1]))
        chosen = scored[0][2]
        chosen["pose"] = scored[0][3]
        return chosen

    def commit(self, obs, tasks, reason="boundary"):
        chosen = self.choose(obs, tasks)
        self.job = chosen
        self.job_epoch += 1
        self.blocked = 0
        self.log("mission_commit", tick=tick_of(obs), reason=reason, kind=chosen.get("kind"),
                 target=chosen.get("target"), pose=chosen.get("pose"), epoch=self.job_epoch)

    def detour(self, obs, goal):
        me, partner = actors(obs)
        start, other = pos(me.get("position")), pos(partner.get("position"))
        if start is None:
            return self.action(self.noop)
        predicted = pos(partner.get("position"))
        candidates = []
        for nxt, delta in neighbors(start):
            if not walkable(terrain_of(obs), nxt) or nxt == other:
                continue
            toward = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) if goal else 0
            away = -(abs(nxt[0] - other[0]) + abs(nxt[1] - other[1])) if other else 0
            candidates.append((toward, away, nxt, delta))
        if not candidates:
            return self.action(self.noop)
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return self.action(ACTION_FOR.get(candidates[0][3], self.noop))

    def motor(self, obs):
        if not self.job:
            return self.action(self.noop)
        me, partner = actors(obs)
        current = pos(me.get("position"))
        facing = pos(me.get("orientation"))
        other = pos(partner.get("position"))
        kind = self.job.get("kind")
        target = pos(self.job.get("target"))
        approach = pos(self.job.get("pose"))
        if current is None or kind in {"wait", "wait_ready"} or target is None:
            return self.action(self.noop)
        if kind == "park":
            if current == target:
                return self.action(self.noop)
            route = bfs(terrain_of(obs), current, target, {other} if other else ())
            if route and len(route) > 1:
                delta = (route[1][0] - current[0], route[1][1] - current[1])
                return self.action(ACTION_FOR.get(delta, self.noop))
            return self.action(self.noop)

        if current == approach:
            delta = (target[0] - current[0], target[1] - current[1])
            turn = ACTION_FOR.get(delta)
            if turn and facing != delta:
                return self.action(turn)
            if self.legal(obs, self.job):
                return self.action("interact")
            return self.action(self.noop)

        route = bfs(terrain_of(obs), current, approach, {other} if other is not None and other != approach else ()) if approach else None
        if route is None and approach is not None:
            route = bfs(terrain_of(obs), current, approach)
        if route is None or len(route) <= 1:
            self.blocked += 1
            if self.blocked >= 5 and kind not in {"deliver", "pickup_ready", "start_cooking"} and target is not None:
                return self.detour(obs, approach)
            return self.action(self.noop)
        nxt = route[1]
        partner_action = str(obs.get("partner_action_last", "")) if isinstance(obs, dict) else ""
        delta_partner = DIRS.get(partner_action, (0, 0))
        predicted = (other[0] + delta_partner[0], other[1] + delta_partner[1]) if other else None
        if nxt == other or nxt == predicted or (predicted == current and nxt == other):
            self.blocked += 1
            if self.blocked >= 5 and kind not in {"deliver", "pickup_ready", "start_cooking"}:
                return self.detour(obs, approach)
            return self.action(self.noop)
        self.blocked = 0
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        return self.action(ACTION_FOR.get(delta, self.noop))

    def run(self):
        while not self.ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            now = tick_of(obs)
            self.partner_state(obs)
            self.poll_astra(obs)
            self.start_astra(obs)
            pots = pot_view(obs)
            tasks = self.tasks(obs, pots)
            self.poll_jev(obs, pots)

            me, _ = actors(obs)
            current = pos(me.get("position"))
            if self.previous_position is not None and current == self.previous_position and self.last_action in DIRS:
                self.blocked += 1
            elif current is not None and current != self.previous_position:
                self.blocked = max(0, self.blocked - 1)
            self.previous_position = current

            old_epoch = self.job_epoch
            if not self.valid_job(obs, tasks):
                self.commit(obs, tasks, "state_boundary")
            if self.job_epoch != old_epoch or self.blocked >= 5:
                self.request_jev(obs, tasks, pots, "mission_boundary" if self.job_epoch != old_epoch else "traffic_block")

            # A strong, code-verified takeover is allowed at the next boundary;
            # it cannot interrupt a chef carrying soup or a valid interaction.
            if self.takeover_hint and any(p["ready"] for p in pots) and held(me) == "":
                self.role_hint = "take_delivery"
                self.role_hint_until = now + 8
                self.takeover_hint = False

            action = self.motor(obs)
            try:
                if self.ctx.done:
                    return
                self.ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
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
        RoleKitchen(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
