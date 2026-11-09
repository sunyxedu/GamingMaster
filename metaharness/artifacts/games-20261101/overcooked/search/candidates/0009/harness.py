"""Candidate 0009: a transaction-aware committed pipeline.

This controller uses a different boundary from the earlier schedulers.  A job
is a physical transaction (walk to one interaction pose, orient, interact),
while a short partner claim is an ownership token for the *next* ingredient.
The claim can suppress duplicate fetching, but it cannot suppress ready-soup
work and it expires on lack of observable progress.  All motor decisions are
local and every model request is fire-and-forget from the realtime thread.
"""
from __future__ import annotations

import json
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
ACT_FOR = {v: k for k, v in DIRS.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}


def point(value, default=None):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return default
    try:
        if isinstance(value[0], bool) or isinstance(value[1], bool):
            return default
        return int(value[0]), int(value[1])
    except Exception:
        return default


def state_of(obs):
    return obs.get("state", {}) if isinstance(obs, dict) and isinstance(obs.get("state"), dict) else {}


def terrain_of(obs):
    value = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return value if isinstance(value, list) else []


def at(terrain, p):
    p = point(p)
    if p is None:
        return "X"
    x, y = p
    try:
        row = terrain[y]
        if x >= 0 and isinstance(row, (str, list, tuple)) and x < len(row):
            return row[x]
    except Exception:
        pass
    return "X"


def walkable(terrain, p):
    return at(terrain, p) not in FEATURES and at(terrain, p) != "X"


def features(terrain, symbol):
    out = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                out.append((x, y))
    return out


def actors(obs):
    players = state_of(obs).get("players", [])
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
    them = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, them


def held(player):
    obj = player.get("held_object") if isinstance(player, dict) else None
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("name", "")).lower()


def objects_of(obs):
    value = state_of(obs).get("objects", [])
    return value if isinstance(value, list) else []


def obj_name(obj):
    return str(obj.get("name", "")).lower() if isinstance(obj, dict) else ""


def obj_pos(obj):
    return point(obj.get("position")) if isinstance(obj, dict) else None


def ingredients_of(obj):
    if not isinstance(obj, dict):
        return []
    values = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name", "")
        value = str(value).lower() if value is not None else ""
        if value in INGREDIENTS:
            out.append(value)
    return out


def orders_of(obs):
    raw = state_of(obs).get("all_orders", [])
    if not isinstance(raw, list):
        raw = []
    out = []
    for order in raw:
        values = order.get("ingredients") if isinstance(order, dict) else order
        if isinstance(values, list):
            recipe = []
            for value in values:
                value = value.get("name", "") if isinstance(value, dict) else value
                value = str(value).lower()
                if value in INGREDIENTS:
                    recipe.append(value)
            if recipe:
                out.append(recipe)
    return out or [["onion", "onion", "onion"]]


def missing_for(ingredients, recipes):
    best_recipe, best_missing = None, None
    for recipe in recipes:
        have = Counter(ingredients)
        want = Counter(recipe)
        if any(have[k] > want[k] for k in have):
            continue
        missing = []
        for item in recipe:
            if have[item] < want[item]:
                missing.append(item)
                have[item] += 1
        if best_missing is None or len(missing) < len(best_missing):
            best_recipe, best_missing = list(recipe), missing
    return best_recipe or [], best_missing or []


def tick_of(obs):
    value = obs.get("tick", state_of(obs).get("timestep", 0)) if isinstance(obs, dict) else 0
    try:
        return int(value)
    except Exception:
        return 0


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def neighbors(p):
    x, y = p
    for delta in DIRS.values():
        yield (x + delta[0], y + delta[1]), delta


def bfs(terrain, start, goal, blocked=()):
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
        for nxt, _ in neighbors(cur):
            if nxt in prev or nxt in blocked or not walkable(terrain, nxt):
                continue
            prev[nxt] = cur
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            q.append(nxt)
    return None


def interaction_poses(terrain, target):
    target = point(target)
    if target is None:
        return []
    out = []
    for dx, dy in DIRS.values():
        pose = (target[0] - dx, target[1] - dy)
        if pose not in out and walkable(terrain, pose):
            out.append(pose)
    return out


def shortest_pose(terrain, start, target, blocked=(), preferred=None):
    choices = interaction_poses(terrain, target)
    if preferred in choices:
        choices.remove(preferred)
        choices.insert(0, preferred)
    found = []
    for pose in choices:
        route = bfs(terrain, start, pose, blocked)
        if route is not None:
            found.append((len(route), pose, route))
    if found:
        found.sort(key=lambda x: (x[0], x[1][1], x[1][0]))
        return found[0][1], found[0][2]
    # The partner is a dynamic obstacle, not a wall.  Retain a route so the
    # committed job can wait or use the traffic recovery policy.
    for pose in choices:
        route = bfs(terrain, start, pose)
        if route is not None:
            found.append((len(route), pose, route))
    if found:
        found.sort(key=lambda x: (x[0], x[1][1], x[1][0]))
        return found[0][1], found[0][2]
    return None, None


def counter_sites(obs):
    terrain = terrain_of(obs)
    occupied = {obj_pos(o) for o in objects_of(obs) if obj_pos(o) is not None}
    out = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            p = (x, y)
            if value == "X" and p not in occupied and any(walkable(terrain, q) for q, _ in neighbors(p)):
                out.append(p)
    return out


def counter_items(obs):
    terrain = terrain_of(obs)
    out = []
    for obj in objects_of(obs):
        p, name = obj_pos(obj), obj_name(obj)
        if p is not None and name and at(terrain, p) == "X":
            out.append((name, p))
    return out


def pot_view(obs):
    terrain = terrain_of(obs)
    soups = {obj_pos(o): o for o in objects_of(obs) if obj_name(o) == "soup" and obj_pos(o) is not None}
    recipes = orders_of(obs)
    out = []
    for p in features(terrain, "P"):
        soup = soups.get(p)
        ing = ingredients_of(soup)
        recipe, missing = missing_for(ing, recipes)
        ready = bool(soup and soup.get("is_ready"))
        cooking = bool(soup and soup.get("is_cooking"))
        ct = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        total = soup.get("cook_time", -1) if soup else -1
        if not ready and not cooking and isinstance(ct, (int, float)) and isinstance(total, (int, float)) and ct >= 0 and total >= 0:
            cooking = True
        full = bool(soup and recipe and not missing)
        phase = "ready" if ready else "cooking" if cooking else "full_idle" if full else "partial" if ing else "empty"
        rem = None
        if cooking and isinstance(ct, (int, float)) and isinstance(total, (int, float)):
            rem = max(0, int(total - ct))
        out.append({"id": "pot_%d_%d" % p, "pos": p, "ingredients": ing, "missing": missing,
                    "phase": phase, "ready": ready, "cooking": cooking, "full": full,
                    "remaining": rem})
    return out


class PipelineController:
    """Code owns legality; model outputs can only choose among safe branches."""

    def __init__(self, ctx, game):
        self.ctx, self.game = ctx, game
        spec = getattr(game, "action_spec", None)
        self.names = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.names:
            self.noop = "stay" if "stay" in self.names else (next(iter(self.names), "stay"))

        self.job = None
        self.epoch = 0
        self.blocked = 0
        self.previous_pos = None
        self.previous_signature = None
        self.last_observation = {}
        self.last_partner = None
        self.partner_history = deque(maxlen=5)

        # A claim is deliberately short-lived and keyed to a pot signature.
        self.claims = {}
        self.claim_change = 0

        self.s1_off = getattr(ctx, "system1", object()) is None
        self.s1_future = None
        self.s1_epoch = -1
        self.s1_tick = -999
        self.s1_key = None
        self.jev_choice = None
        self.jev_choice_tick = -999
        self.jev_yield = False

        self.s2_off = getattr(ctx, "system2", object()) is None
        self.s2_future = None
        self.astra_started = False
        self.astra_kind = None
        self.astra_traffic = None

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def safe_action(self, action):
        if action in self.names:
            return action
        if self.noop in self.names:
            return self.noop
        return sorted(self.names)[0] if self.names else "stay"

    # -------------------- partner ownership --------------------

    def partner_route_len(self, obs, pot):
        _, partner = actors(obs)
        start = point(partner.get("position"))
        if start is None:
            return 9999
        routes = [bfs(terrain_of(obs), start, pose) for pose in interaction_poses(terrain_of(obs), pot["pos"])]
        routes = [r for r in routes if r is not None]
        return min((len(r) for r in routes), default=9999)

    def likely_partner_pot(self, obs, pots, item):
        choices = [(self.partner_route_len(obs, pot), pot["pos"], pot) for pot in pots
                   if pot["phase"] in {"empty", "partial"} and item in pot["missing"]]
        choices.sort(key=lambda x: (x[0], x[1][1], x[1][0]))
        return choices[0][2] if choices else None

    def actor_route_len(self, obs, player, pot):
        start = point(player.get("position")) if isinstance(player, dict) else None
        if start is None:
            return 9999
        routes = [bfs(terrain_of(obs), start, pose) for pose in interaction_poses(terrain_of(obs), pot["pos"])]
        routes = [r for r in routes if r is not None]
        return min((len(r) for r in routes), default=9999)

    def partner_dish_pot(self, obs, pots):
        """Return the ready/cooking pot for which a partner-held dish is closest."""
        _, partner = actors(obs)
        candidates = [p for p in pots if p["phase"] in {"ready", "cooking"}]
        # Ready soup wins over a merely cooking soup; distance breaks ties.
        candidates.sort(key=lambda p: (0 if p["ready"] else 1, self.actor_route_len(obs, partner, p), p["pos"]))
        return candidates[0] if candidates and self.actor_route_len(obs, partner, candidates[0]) < 9999 else None

    def partner_owns_dish_transaction(self, obs, pot):
        me, partner = actors(obs)
        if held(partner) != "dish":
            return False
        partner_distance = self.actor_route_len(obs, partner, pot)
        self_distance = self.actor_route_len(obs, me, pot)
        partner_position = point(partner.get("position"))
        at_approach = partner_position in set(interaction_poses(terrain_of(obs), pot["pos"]))
        return at_approach or partner_distance <= self_distance + 1

    def update_claims(self, obs, pots):
        now = tick_of(obs)
        _, partner = actors(obs)
        partner_pos = point(partner.get("position"))
        partner_item = held(partner)
        partner_action = str(obs.get("partner_action_last", "")) if isinstance(obs, dict) else ""
        previous = self.last_partner or {}
        previous_item = previous.get("held", "")
        previous_pos = previous.get("pos")
        self.last_partner = {"pos": partner_pos, "held": partner_item, "action": partner_action}
        self.partner_history.append(self.last_partner.copy())

        active_ids = set()
        for pot in pots:
            pid, signature = pot["id"], (pot["phase"], tuple(pot["ingredients"]), tuple(pot["missing"]))
            if pot["phase"] not in {"empty", "partial"} or not pot["missing"]:
                self.claims.pop(pid, None)
                continue

            intended = self.likely_partner_pot(obs, pots, partner_item) if partner_item in INGREDIENTS else None
            carrying = bool(partner_item in pot["missing"] and intended is pot and self.partner_route_len(obs, pot) < 9999)
            approach = set(interaction_poses(terrain_of(obs), pot["pos"]))
            just_filled = (not partner_item and pot["phase"] == "partial" and partner_pos in approach and
                           (previous_item in INGREDIENTS or partner_action == "interact"))
            evidence = carrying or just_filled
            record = self.claims.get(pid)
            if evidence:
                if record is None or record.get("signature") != signature:
                    # Keep a claim through the visible ingredient increment,
                    # but restart its bounded clock.  Movement alone does not.
                    self.claims[pid] = {"signature": signature, "since": now,
                                        "last_progress": now, "reason": "carrying" if carrying else "fill",
                                        "expired": False}
                    self.claim_change += 1
                    record = self.claims[pid]
                elif record.get("expired"):
                    # Do not recreate an expired lease merely because a stuck
                    # partner keeps holding the same object.  A pot signature
                    # change or a genuine release/reacquire starts a new one.
                    pass
                else:
                    record["last_seen"] = now
                # Carrying is evidence of intent, not proof of progress.  A
                # partner that remains stuck must not reserve the pot forever.
                if record is not None and not record.get("expired") and now - int(record.get("since", now)) <= 12:
                    active_ids.add(pid)
                elif record is not None and not record.get("expired"):
                    record["expired"] = True
                    self.claim_change += 1
            elif record is not None:
                # A partner may have just put the object down; give that
                # transition a few ticks, then reopen the pot to us.
                if record.get("expired"):
                    self.claims.pop(pid, None)
                    self.claim_change += 1
                    continue
                age = now - int(record.get("since", now))
                grace = 4 if not partner_item else 0
                if age <= 12 + grace and signature == record.get("signature"):
                    active_ids.add(pid)
                else:
                    self.claims.pop(pid, None)
                    self.claim_change += 1
        # Keep an expired marker while the same carried object remains visible;
        # otherwise the next tick would recreate the very lease that expired.
        for pid in list(self.claims):
            record = self.claims[pid]
            if (pid not in active_ids and not record.get("expired") and
                    now - int(record.get("since", now)) > 12):
                record["expired"] = True
                self.claim_change += 1

    def claim_active(self, pot):
        record = self.claims.get(pot["id"])
        return bool(record and not record.get("expired"))

    # -------------------- tasks and compact model state --------------------

    def task(self, kind, target=None, item="", pot=None, desc="", lock=False, hold=False):
        target = point(target)
        suffix = "_".join(map(str, target)) if target is not None else "none"
        pid = pot.get("id", "") if isinstance(pot, dict) else ""
        ident = "%s_%s_%s" % (kind, suffix, pid.replace("pot_", ""))
        return {"id": ident, "kind": kind, "target": target, "item": item,
                "pot_id": pid, "description": desc or kind, "lock": bool(lock), "hold": bool(hold)}

    def tasks(self, obs, pots):
        terrain = terrain_of(obs)
        me, partner = actors(obs)
        mine, theirs = held(me), held(partner)
        out = []

        if mine == "soup":
            for target in features(terrain, "S"):
                out.append(self.task("deliver", target, desc="deliver held soup", lock=True))
            return out or [self.task("wait", desc="hold soup safely", hold=True)]

        if mine == "dish":
            ready = [p for p in pots if p["ready"]]
            if ready:
                owned = self.partner_dish_pot(obs, ready) if theirs == "dish" else None
                if owned is not None and self.partner_owns_dish_transaction(obs, owned):
                    return [self.task("yield_partner", self.parking_cell(obs, owned["pos"]), pot=owned,
                                      desc="let the partner deliver its dish", hold=True)]
                return [self.task("pickup_ready", p["pos"], pot=p,
                                  desc="pick up ready soup with dish", lock=True) for p in ready]
            cooking = [p for p in pots if p["cooking"]]
            if cooking:
                owned = self.partner_dish_pot(obs, cooking) if theirs == "dish" else None
                if owned is not None and self.partner_owns_dish_transaction(obs, owned):
                    return [self.task("yield_partner", self.parking_cell(obs, owned["pos"]), pot=owned,
                                      desc="let the partner stage its dish", hold=True)]
                return [self.task("stage_dish", p["pos"], pot=p,
                                  desc="reserve the cooking-pot approach", lock=True, hold=True) for p in cooking]
            return [self.task("wait", desc="hold dish without wasting it", hold=True)]

        if mine in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial"} and mine in p["missing"]]
            compatible.sort(key=lambda p: (len(p["missing"]), p["pos"]))
            if compatible:
                return [self.task("fill_pot", p["pos"], item=mine, pot=p,
                                  desc="put held %s in a compatible pot" % mine, lock=True) for p in compatible]
            return [self.task("deposit_counter", p, item=mine,
                              desc="stage surplus %s on a free counter" % mine, lock=True)
                    for p in counter_sites(obs)[:5]] or [self.task("wait", desc="hold ingredient safely", hold=True)]

        # These branches are hard priorities.  They are intentionally before
        # any partner lease or Jev suggestion.
        ready = [p for p in pots if p["ready"]]
        if ready and theirs not in {"dish", "soup"}:
            dishes = [self.task("fetch_dish", p, desc="fetch a dish for ready soup", lock=True)
                      for p in features(terrain, "D")]
            if dishes:
                return dishes

        starts = [self.task("start_cooking", p["pos"], pot=p,
                            desc="explicitly start a full idle pot", lock=True)
                  for p in pots if p["phase"] == "full_idle"]
        if starts:
            return starts

        # A lease is an exclusion on the duplicate *ingredient* branch, not a
        # global pause.  On multi-pot layouts another unclaimed pot wins.
        unfinished = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]]
        available = [p for p in unfinished if not self.claim_active(p)]
        available.sort(key=lambda p: (len(p["missing"]), p["pos"]))
        if available:
            pot = available[0]
            item = pot["missing"][0]
            staged = [q for name, q in counter_items(obs) if name == item]
            choices = [self.task("fetch_counter", q, item=item, pot=pot,
                                 desc="retrieve staged %s" % item, lock=True) for q in staged[:4]]
            dispenser = "O" if item == "onion" else "T"
            choices.extend(self.task("fetch_ingredient", q, item=item, pot=pot,
                                     desc="fetch one %s from dispenser" % item, lock=True)
                           for q in features(terrain, dispenser))
            if choices:
                return choices

        leased = [p for p in unfinished if self.claim_active(p)]
        if leased:
            # A physical parking target keeps the partner's route open instead
            # of turning a lease into an immobile one-cell blockade.
            pot = sorted(leased, key=lambda p: (len(p["missing"]), p["pos"]))[0]
            parking = self.parking_cell(obs, pot["pos"])
            return [self.task("yield_partner", parking, pot=pot,
                              desc="yield to the partner's active fill transaction", hold=True)]

        if any(p["cooking"] for p in pots) and theirs not in {"dish", "soup"}:
            dishes = [self.task("fetch_dish", p, desc="fetch dish during cooking", lock=True)
                      for p in features(terrain, "D")]
            if dishes:
                return dishes
        if theirs not in {"dish", "soup"}:
            out.extend(self.task("fetch_dish", p, desc="stage a future dish", lock=True)
                       for p in features(terrain, "D"))
        return out or [self.task("wait", desc="wait for the next production transition", hold=True)]

    def parking_cell(self, obs, target):
        terrain = terrain_of(obs)
        me, partner = actors(obs)
        start, other = point(me.get("position")), point(partner.get("position"))
        forbidden = set(interaction_poses(terrain, target)) | ({other} if other else set())
        if start is None:
            return None
        candidates = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                p = (x, y)
                if not walkable(terrain, p) or p in forbidden:
                    continue
                route = bfs(terrain, start, p, {other} if other else ())
                if route is None:
                    continue
                dp = abs(p[0] - target[0]) + abs(p[1] - target[1])
                do = abs(p[0] - other[0]) + abs(p[1] - other[1]) if other else 0
                candidates.append((-dp - do, len(route), p))
        if not candidates:
            return start
        candidates.sort()
        return candidates[0][2]

    def task_key(self, tasks, pots, obs):
        me, partner = actors(obs)
        return (held(me), held(partner),
                tuple((p["id"], p["phase"], tuple(p["ingredients"]), self.claim_active(p)) for p in pots),
                tuple((t["kind"], t["pot_id"], t["item"]) for t in tasks[:5]))

    def compact(self, obs, tasks, pots):
        me, partner = actors(obs)
        return json_safe({
            "tick": tick_of(obs), "layout": obs.get("layout_name", ""),
            "self": {"position": point(me.get("position")), "facing": point(me.get("orientation")), "held": held(me)},
            "partner": {"position": point(partner.get("position")), "facing": point(partner.get("orientation")),
                        "held": held(partner), "last_action": obs.get("partner_action_last", ""),
                        "task": obs.get("partner_task", "")},
            "pots": [{"position": p["pos"], "phase": p["phase"], "ingredients": p["ingredients"],
                      "missing": p["missing"], "remaining": p["remaining"],
                      "partner_claim": self.claim_active(p)} for p in pots],
            "orders": orders_of(obs),
            "job": {"kind": self.job.get("kind", "") if self.job else "",
                    "target": self.job.get("target") if self.job else None,
                    "blocked": self.blocked},
            "candidates": [{"id": t["id"], "kind": t["kind"], "description": t["description"]} for t in tasks[:10]],
        })

    # -------------------- asynchronous model bridge --------------------

    def start_astra(self, obs, tasks, pots):
        if self.astra_started or self.s2_off or self.ctx.done:
            return
        self.astra_started = True
        body = self.compact(obs, tasks, pots)
        prompt = (
            "Act as a slow strategic reviewer for a realtime symbolic Overcooked-AI kitchen. "
            "The executable controller owns exact recipes, legality, routes, interaction poses "
            "and primitive actions.  Choose a role bias only: delivery, supply, handoff, or "
            "balanced; also choose traffic claim, yield, or none.  A partner claim is temporary. "
            "Return ONLY JSON {\"preferred_kind\":\"...\",\"traffic\":\"claim|yield|none\","
            "\"replan_trigger\":\"...\"}; never output directions or task ids.\n" +
            json.dumps(body, separators=(",", ":"))
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
            self.s2_future = self.ctx.spawn(work)
        except Exception:
            self.s2_future = None

    def poll_astra(self, obs):
        future = self.s2_future
        if future is None or not future.done():
            return
        self.s2_future = None
        try:
            result = future.result()
            text = result if isinstance(result, str) else getattr(result, "text", "")
            match = re.search(r"\{.*\}", text, re.DOTALL) if isinstance(text, str) else None
            data = json.loads(match.group(0)) if match else {}
            if isinstance(data, dict):
                kind = data.get("preferred_kind")
                if kind in {"deliver", "pickup_ready", "start_cooking", "fill_pot", "fetch_ingredient",
                            "fetch_counter", "fetch_dish", "stage_dish", "yield_partner", "wait"}:
                    self.astra_kind = kind
                traffic = data.get("traffic")
                self.astra_traffic = traffic if traffic in {"claim", "yield", "none"} else None
        except Exception:
            self.astra_kind, self.astra_traffic = None, None

    def request_jev(self, obs, tasks, pots, reason):
        if self.s1_off or self.s1_future is not None or not tasks or self.ctx.done:
            return
        now, key = tick_of(obs), self.task_key(tasks, pots, obs)
        urgent = tasks[0]["kind"] in {"deliver", "pickup_ready", "start_cooking", "yield_partner"}
        # A blocked motor can persist for many ticks.  Keep the fast model a
        # diagnostic adjudicator, not a per-tick request generator.
        if reason == "blocked" and now - self.s1_tick < 8:
            return
        if key == self.s1_key and now - self.s1_tick < (8 if urgent else 20):
            return
        self.s1_key, self.s1_tick, self.s1_epoch = key, now, self.epoch
        criteria = {t["id"]: t["description"] for t in tasks[:10]}
        state = {"game": "Overcooked-AI", "observation": self.compact(obs, tasks, pots)}
        questions = {
            "task": Choice(
                "Choose one id from observation.candidates. They were generated from the current "
                "symbolic state. Prefer the candidate's safe production branch; never invent a "
                "primitive action or id. A yield candidate means only a short partner handoff.",
                criteria,
            ),
            "yield": Noul(
                "Should the controlled chef respect an already visible partner_claim for a few "
                "ticks? True only when partner_claim is true and the partner is carrying the "
                "matching ingredient or has just filled that pot; false cannot cancel urgent "
                "delivery or pickup work.",
                {"true": "The listed partner claim is credible and temporary.",
                 "false": "There is no credible active partner claim."},
            ),
        }

        def ask():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(state, questions, timeout_s=1.0)
            except Exception:
                return {}

        try:
            self.s1_future = self.ctx.spawn(ask)
            self.log("jev_request", tick=now, epoch=self.epoch, reason=reason)
        except Exception:
            self.s1_future = None

    def poll_jev(self, obs, tasks, pots):
        future = self.s1_future
        if future is None or not future.done():
            return
        self.s1_future = None
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict) or self.s1_epoch != self.epoch:
            return
        now = tick_of(obs)
        valid = {t["id"]: t for t in tasks}
        try:
            answer = answers.get("task")
            choice = getattr(answer, "value", None)
            probs = getattr(answer, "probabilities", {}) or {}
            probability = float(probs.get(choice, 0.0)) if isinstance(probs, dict) else 0.0
            if choice in valid and (probability >= 0.42 or probability == 0.0):
                self.jev_choice, self.jev_choice_tick = choice, now
                self.log("jev_choice", tick=now, choice=choice, probability=probability)
            yanswer = answers.get("yield")
            yvalue = getattr(yanswer, "value", 0.0)
            if isinstance(yvalue, (int, float)) and yvalue >= 0.72 and any(self.claim_active(p) for p in pots):
                self.jev_yield = True
        except Exception:
            pass

    # -------------------- committed job and motor --------------------

    def legal_tasks(self, obs, pots):
        return self.tasks(obs, pots)

    def priority(self, task):
        return {"deliver": 0, "pickup_ready": 1, "start_cooking": 2, "fill_pot": 3,
                "fetch_counter": 4, "fetch_ingredient": 5, "deposit_counter": 6,
                "fetch_dish": 7, "stage_dish": 8, "yield_partner": 9, "wait": 20}.get(task.get("kind"), 15)

    def choose_task(self, obs, tasks):
        me, partner = actors(obs)
        start, other = point(me.get("position")), point(partner.get("position"))
        terrain = terrain_of(obs)
        if not tasks:
            return self.task("wait", desc="safe wait", hold=True)
        safe_rank = min(self.priority(t) for t in tasks)
        candidates = []
        for t in tasks:
            pose, route = shortest_pose(terrain, start, t.get("target"), {other} if other else (), t.get("pose")) if t.get("target") is not None else (None, [start] if start else None)
            distance = len(route) if route is not None else 9999
            # Jev and Astra can choose only within the code's highest-priority
            # band, never make a lease lose to speculative work.
            candidates.append((self.priority(t), distance, t["id"], t, pose))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        safe = candidates[0][3].copy()
        safe["pose"] = candidates[0][4]
        safe_kind = safe.get("kind")
        selected = None
        if self.jev_choice and tick_of(obs) - self.jev_choice_tick <= 10:
            selected = next((x for x in candidates if x[3]["id"] == self.jev_choice), None)
        if selected is None and self.astra_kind:
            selected = next((x for x in candidates if x[3]["kind"] == self.astra_kind and x[0] == safe_rank), None)
        if selected is not None and selected[0] == safe_rank:
            safe = selected[3].copy()
            safe["pose"] = selected[4]
        # Never let an advisory answer turn a high-priority branch into wait.
        if self.priority(safe) != safe_rank:
            safe = candidates[0][3].copy()
            safe["pose"] = candidates[0][4]
        return safe

    def same_task(self, a, b):
        return bool(a and b and a.get("kind") == b.get("kind") and
                    a.get("target") == b.get("target") and a.get("item", "") == b.get("item", "") and
                    a.get("pot_id", "") == b.get("pot_id", ""))

    def job_valid(self, obs, tasks):
        if not self.job:
            return False
        kind = self.job.get("kind")
        me, _ = actors(obs)
        mine = held(me)
        # The cooking-to-ready transition is the one intentional task change
        # that preserves the same physical pose.
        if kind in {"stage_dish", "pickup_ready"} and mine == "dish":
            for t in tasks:
                if t.get("kind") in {"stage_dish", "pickup_ready"} and t.get("target") == self.job.get("target"):
                    self.job.update({"kind": t["kind"], "description": t["description"], "lock": t["lock"], "hold": t["hold"]})
                    return True
        # A parking target is recomputed from the current position.  Keep the
        # lease job across those harmless target changes; otherwise each step
        # would select a new far-away parking cell and oscillate.
        if kind == "yield_partner":
            if any(t.get("kind") == "yield_partner" and t.get("pot_id") == self.job.get("pot_id") for t in tasks):
                return True
        for t in tasks:
            if self.same_task(self.job, t):
                return True
        # Completion is determined from the actual held object / pot state.
        if kind in {"fetch_ingredient", "fetch_counter"} and mine in INGREDIENTS:
            return False
        if kind == "fetch_dish" and mine == "dish":
            return False
        if kind == "pickup_ready" and mine == "soup":
            return False
        if kind == "deliver" and mine != "soup":
            return False
        if kind in {"fill_pot", "deposit_counter"} and not mine:
            return False
        return False

    def commit(self, obs, tasks):
        selected = self.choose_task(obs, tasks)
        me, partner = actors(obs)
        start, other = point(me.get("position")), point(partner.get("position"))
        target = selected.get("target")
        pose, route = (shortest_pose(terrain_of(obs), start, target, {other} if other else (), selected.get("pose"))
                       if target is not None else (None, None))
        selected["pose"] = pose
        selected["initial_route"] = len(route) if route is not None else 9999
        self.job = selected
        self.epoch += 1
        self.blocked = 0
        self.log("job_commit", tick=tick_of(obs), kind=selected.get("kind"), target=selected.get("target"),
                 pose=selected.get("pose"), epoch=self.epoch)

    def valid_interaction(self, obs, job):
        me, _ = actors(obs)
        mine = held(me)
        target = point(job.get("target"))
        if target is None:
            return False
        kind = job.get("kind")
        pots = pot_view(obs)
        pot = next((p for p in pots if p["pos"] == target), None)
        terrain = terrain_of(obs)
        if kind == "deliver":
            return mine == "soup" and at(terrain, target) == "S"
        if kind == "fetch_dish":
            return not mine and at(terrain, target) == "D"
        if kind == "fetch_ingredient":
            return not mine and at(terrain, target) in {"O", "T"}
        if kind == "fetch_counter":
            return not mine and any(n in INGREDIENTS and p == target for n, p in counter_items(obs))
        if kind == "deposit_counter":
            occupied = {obj_pos(o) for o in objects_of(obs)}
            return mine in INGREDIENTS and target not in occupied
        if kind == "fill_pot":
            return bool(pot and mine in INGREDIENTS and pot["phase"] in {"empty", "partial"} and mine in pot["missing"])
        if kind == "start_cooking":
            return bool(pot and not mine and pot["phase"] == "full_idle")
        if kind == "pickup_ready":
            return bool(pot and mine == "dish" and pot["ready"])
        return False

    def predicted_partner(self, obs):
        _, partner = actors(obs)
        p = point(partner.get("position"))
        action = str(obs.get("partner_action_last", "")) if isinstance(obs, dict) else ""
        d = DIRS.get(action, (0, 0))
        return (p[0] + d[0], p[1] + d[1]) if p is not None else None

    def recovery_route(self, obs, target):
        # A real parking route, rather than an alternating north/south escape.
        p = self.parking_cell(obs, target)
        me, partner = actors(obs)
        start, other = point(me.get("position")), point(partner.get("position"))
        return bfs(terrain_of(obs), start, p, {other} if other else ()) if p is not None else None

    def motor(self, obs):
        job = self.job
        if not job:
            return self.safe_action(self.noop)
        me, partner = actors(obs)
        current, facing, other = point(me.get("position")), point(me.get("orientation")), point(partner.get("position"))
        kind, target, pose = job.get("kind"), point(job.get("target")), point(job.get("pose"))
        if current is None:
            return self.safe_action(self.noop)
        if kind == "wait" or target is None:
            return self.safe_action(self.noop)

        if kind == "yield_partner":
            route = bfs(terrain_of(obs), current, target, {other} if other else ())
            if route is None:
                return self.safe_action(self.noop)
            if len(route) <= 1:
                return self.safe_action(self.noop)
            delta = (route[1][0] - current[0], route[1][1] - current[1])
            return self.safe_action(ACT_FOR.get(delta, self.noop))

        # Keep the committed interaction pose.  It is a reservation during the
        # cooking window, but no move is posted into an occupied/swap cell.
        if current == pose:
            delta = (target[0] - current[0], target[1] - current[1])
            turn = ACT_FOR.get(delta)
            if turn and facing != delta:
                return self.safe_action(turn)
            if job.get("hold"):
                return self.safe_action(self.noop)
            if self.valid_interaction(obs, job):
                return self.safe_action("interact")
            if job.get("lock"):
                return self.safe_action(self.noop)

        route = bfs(terrain_of(obs), current, pose, {other} if other is not None and other != pose else ()) if pose is not None else None
        if route is None:
            route = bfs(terrain_of(obs), current, pose) if pose is not None else None
        if route is None or len(route) <= 1:
            self.blocked += 1
            if self.blocked >= 6 and kind not in {"pickup_ready", "deliver", "start_cooking"}:
                spare = self.recovery_route(obs, target)
                if spare is not None and len(spare) > 1:
                    d = (spare[1][0] - current[0], spare[1][1] - current[1])
                    return self.safe_action(ACT_FOR.get(d, self.noop))
            return self.safe_action(self.noop)

        nxt = route[1]
        predicted = self.predicted_partner(obs)
        # Avoid actual collision and direct swap.  A short wait is cheaper than
        # allowing the MDP to retain both positions and lose the whole route.
        if nxt == other or nxt == predicted or (predicted == current and nxt == other):
            self.blocked += 1
            if self.blocked >= 7 and kind not in {"pickup_ready", "deliver", "start_cooking"}:
                spare = self.recovery_route(obs, target)
                if spare is not None and len(spare) > 1:
                    d = (spare[1][0] - current[0], spare[1][1] - current[1])
                    return self.safe_action(ACT_FOR.get(d, self.noop))
            return self.safe_action(self.noop)
        self.blocked = 0
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        return self.safe_action(ACT_FOR.get(delta, self.noop))

    def run(self):
        while not self.ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self.last_observation = obs
            pots = pot_view(obs)
            self.update_claims(obs, pots)
            tasks = self.legal_tasks(obs, pots)
            self.poll_astra(obs)
            self.start_astra(obs, tasks, pots)
            self.poll_jev(obs, tasks, pots)

            if not self.job_valid(obs, tasks):
                self.commit(obs, tasks)
                self.request_jev(obs, tasks, pots, "new_job")
                self.claim_change = 0
            elif self.blocked >= 5:
                self.request_jev(obs, tasks, pots, "blocked")
                self.claim_change = 0
            elif self.claim_change:
                self.request_jev(obs, tasks, pots, "claim_change")
                self.claim_change = 0

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
                        self.ctx.act(self.safe_action(self.noop))
                except Exception:
                    return
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                return


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        if not getattr(game, "tick_s", None):
            return
        PipelineController(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
