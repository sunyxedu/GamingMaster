"""Candidate 0014: an event-driven kitchen director with receding-horizon motion.

The controller treats a pot assignment as a soft production intention, not as a
permanent reservation of one interaction square.  Exact state interpretation,
recipe compatibility, interaction legality, and movement are local code.  Astra
writes a small layout protocol in the background; Jev judges partner intent and
which already-generated event should be pursued.  Neither model emits a
primitive action.
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
ACTION_FOR_VECTOR = {v: k for k, v in DIRECTIONS.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}
URGENT_KINDS = {"deliver", "pickup_ready", "start_cooking"}


def _point(value, fallback=None):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return fallback
    try:
        if isinstance(value[0], bool) or isinstance(value[1], bool):
            return fallback
        return int(value[0]), int(value[1])
    except Exception:
        return fallback


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _state(obs):
    if isinstance(obs, dict) and isinstance(obs.get("state"), dict):
        return obs["state"]
    return {}


def _terrain(obs):
    value = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return value if isinstance(value, list) else []


def _cell(terrain, value):
    p = _point(value)
    if p is None:
        return "X"
    x, y = p
    if x < 0 or y < 0 or y >= len(terrain):
        return "X"
    try:
        row = terrain[y]
        if isinstance(row, (str, list, tuple)) and x < len(row):
            return row[x]
    except Exception:
        pass
    return "X"


def _feature_cells(terrain, symbol):
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                result.append((x, y))
    return result


def _walkable(terrain, value):
    symbol = _cell(terrain, value)
    return symbol not in FEATURES and symbol != "X"


def _all_floor(terrain):
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x in range(len(row)):
            p = (x, y)
            if _walkable(terrain, p):
                result.append(p)
    return result


def _neighbors(p):
    p = _point(p)
    if p is None:
        return
    x, y = p
    # This order is stable across episodes and makes equal routes reproducible.
    for delta in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        yield (x + delta[0], y + delta[1]), delta


def _bfs(terrain, start, goal, blocked=()):
    start, goal = _point(start), _point(goal)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    forbidden = {_point(p) for p in blocked if _point(p) is not None}
    forbidden.discard(start)
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        for nxt, _ in _neighbors(current):
            if nxt in previous or nxt in forbidden or not _walkable(terrain, nxt):
                continue
            previous[nxt] = current
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


def _interaction_poses(terrain, target):
    target = _point(target)
    if target is None:
        return []
    result = []
    for delta in DIRECTIONS.values():
        pose = (target[0] - delta[0], target[1] - delta[1])
        if pose not in result and _walkable(terrain, pose):
            result.append(pose)
    return result


def _tick(obs):
    if not isinstance(obs, dict):
        return 0
    value = obs.get("tick", _state(obs).get("timestep", 0))
    try:
        return int(value)
    except Exception:
        return 0


def _players(obs):
    value = _state(obs).get("players", [])
    return value if isinstance(value, list) else []


def _actors(obs):
    players = _players(obs)
    try:
        index = int(obs.get("controlled_agent_index", 0)) if isinstance(obs, dict) else 0
    except Exception:
        index = 0
    if index not in (0, 1) or index >= len(players):
        index = 0
    other = 1 - index if len(players) > 1 else index
    me = players[index] if index < len(players) and isinstance(players[index], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, partner


def _held_name(player):
    value = player.get("held_object") if isinstance(player, dict) else None
    if not isinstance(value, dict):
        return ""
    return str(value.get("name", "") or "").lower()


def _objects(obs):
    value = _state(obs).get("objects", [])
    return value if isinstance(value, list) else []


def _object_name(obj):
    return str(obj.get("name", "") or "").lower() if isinstance(obj, dict) else ""


def _object_pos(obj):
    return _point(obj.get("position")) if isinstance(obj, dict) else None


def _ingredients(obj):
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


def _recipes(obs):
    raw = _state(obs).get("all_orders", [])
    if not isinstance(raw, list):
        raw = []
    result = []
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


def _missing(ingredients, recipes):
    current = Counter(ingredients)
    best_recipe = None
    best_missing = None
    for recipe in recipes:
        wanted = Counter(recipe)
        if any(current[name] > wanted[name] for name in current):
            continue
        have = Counter(ingredients)
        missing = []
        for name in recipe:
            if have[name] < wanted[name]:
                missing.append(name)
                have[name] += 1
        if best_missing is None or (len(missing), tuple(recipe)) < (len(best_missing), tuple(best_recipe)):
            best_recipe = list(recipe)
            best_missing = missing
    return best_recipe or [], best_missing or []


def _pot_view(obs):
    soup_by_pos = {}
    for obj in _objects(obs):
        p = _object_pos(obj)
        if p is not None and _object_name(obj) == "soup":
            soup_by_pos[p] = obj
    recipes = _recipes(obs)
    result = []
    for p in _feature_cells(_terrain(obs), "P"):
        soup = soup_by_pos.get(p)
        ingredients = _ingredients(soup)
        recipe, missing = _missing(ingredients, recipes)
        ready = bool(soup and soup.get("is_ready"))
        cooking = bool(soup and soup.get("is_cooking"))
        raw_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        raw_time = soup.get("cook_time", -1) if soup else -1
        try:
            cook_tick = int(raw_tick)
        except Exception:
            cook_tick = -1
        try:
            cook_time = int(raw_time)
        except Exception:
            cook_time = -1
        if not ready and not cooking and cook_tick >= 0 and cook_time >= 0:
            cooking = True
        full = bool(soup and recipe and not missing)
        phase = "ready" if ready else "cooking" if cooking else "full_idle" if full else "partial" if ingredients else "empty"
        remaining = max(0, cook_time - cook_tick) if cooking and cook_time >= 0 and cook_tick >= 0 else None
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


def _counter_objects(obs):
    terrain = _terrain(obs)
    result = []
    for obj in _objects(obs):
        p, name = _object_pos(obj), _object_name(obj)
        if p is not None and name and _cell(terrain, p) == "X":
            result.append((name, p))
    return result


def _counter_spaces(obs):
    terrain = _terrain(obs)
    occupied = {_object_pos(obj) for obj in _objects(obs) if _object_pos(obj) is not None}
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, symbol in enumerate(row):
            p = (x, y)
            if symbol != "X" or p in occupied:
                continue
            if any(_walkable(terrain, q) for q, _ in _neighbors(p)):
                result.append(p)
    return result


def _best_pose_path(terrain, start, target, forbidden=(), preferred=None):
    poses = _interaction_poses(terrain, target)
    if preferred in poses:
        poses.remove(preferred)
        poses.insert(0, preferred)
    routes = []
    blocked = set(forbidden)
    for pose in poses:
        path = _bfs(terrain, start, pose, blocked)
        if path is not None:
            routes.append((len(path), 0 if pose == preferred else 1, pose, path))
    if routes:
        routes.sort(key=lambda x: (x[0], x[1], x[2][1], x[2][0]))
        return routes[0][2], routes[0][3]
    # A partner can occupy the only approach.  Keep a route so the controller
    # can use a bounded escape rather than declaring the physical task done.
    routes = []
    for pose in poses:
        path = _bfs(terrain, start, pose)
        if path is not None:
            routes.append((len(path), pose, path))
    if routes:
        routes.sort(key=lambda x: (x[0], x[1][1], x[1][0]))
        return routes[0][1], routes[0][2]
    return None, None


def _predicted_partner(obs):
    _, partner = _actors(obs)
    p = _point(partner.get("position"))
    action = str(obs.get("partner_action_last", "") or "") if isinstance(obs, dict) else ""
    delta = DIRECTIONS.get(action, (0, 0))
    return (p[0] + delta[0], p[1] + delta[1]) if p is not None else None


def _route_len_to_feature(terrain, start, target):
    _, path = _best_pose_path(terrain, start, target)
    return len(path) if path is not None else 9999


class AdaptiveKitchen:
    """A soft-assignment event machine with a continuously rebuilt motor route."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = getattr(game, "action_spec", None)
        self.allowed = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.allowed:
            self.noop = "stay" if "stay" in self.allowed else (sorted(self.allowed)[0] if self.allowed else "stay")

        self.job = None
        self.job_epoch = 0
        self.blocked = 0
        self.stall = 0
        self.last_pos = None
        self.last_world = None
        self.last_tick = -1
        self.last_action = self.noop
        self.avoid = {}
        self.last_commit_key = None

        self.protocol = {"circulation": "shortest", "delivery": "balanced", "handoff": "counter"}
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.astra_revision = False
        self.last_snapshot = {}

        self.s1_disabled = getattr(ctx, "system1", object()) is None
        self.jev_future = None
        self.jev_tick = -999
        self.jev_key = None
        self.jev_job_epoch = -1
        self.jev_task = None
        self.jev_partner_hint = None
        self.jev_release = False

    # ---------- defensive model and trace bridge ----------

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _safe_action(self, action):
        if action in self.allowed:
            return action
        if self.noop in self.allowed:
            return self.noop
        return sorted(self.allowed)[0] if self.allowed else "stay"

    def _closed(self):
        try:
            return bool(self.ctx.done)
        except Exception:
            return True

    def _protocol_prompt(self, obs, reason="initial"):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        pots = _pot_view(obs)
        return (
            "You are Astra, the slow strategic director for a realtime symbolic "
            "Overcooked-AI kitchen. This is a pinned grid MDP: code executes every "
            "single-tile action and owns recipes, legality, routes, and timing. Do "
            "not output directions or actions. Choose a robust role protocol that "
            "will not reserve a one-cell gate forever. A protocol is advisory and "
            "must be safe on layouts not seen in training. Return only JSON with "
            "circulation equal to shortest|clockwise|counterclockwise, delivery "
            "equal to self|partner|balanced, handoff equal to counter|direct|none, "
            "and replan equal to event|blocked. Prefer shortest for ordinary open "
            "rooms, delivery protection when soup is ready, and counter handoff "
            "only when the map actually exposes a usable counter. If a partner is "
            "carrying an ingredient, let code choose another legal job rather than "
            "telling it to wait indefinitely.\n"
            + json.dumps(_jsonable({
                "reason": reason,
                "tick": _tick(obs),
                "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
                "terrain": terrain,
                "self": {"position": _point(me.get("position")), "held": _held_name(me)},
                "partner": {"position": _point(partner.get("position")), "held": _held_name(partner),
                            "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else ""},
                "pots": [{"id": p["id"], "phase": p["phase"], "ingredients": p["ingredients"],
                          "missing": p["missing"], "remaining": p["remaining"]} for p in pots],
                "counters": len(_counter_spaces(obs)),
            }), separators=(",", ":"))
        )

    def _launch_astra(self, obs, reason="initial"):
        if self.astra_future is not None or self._closed():
            return
        if getattr(self.ctx, "system2", object()) is None:
            self.astra_revision = True
            return
        if self.astra_calls >= 2:
            return
        self.astra_calls += 1
        prompt = self._protocol_prompt(obs, reason)
        session = self.astra_session if self.astra_revision else None
        self.astra_revision = False

        def ask():
            if self._closed():
                return None
            try:
                result = self.ctx.think(prompt, session_id=session, timeout_s=90.0)
                return {"text": getattr(result, "text", ""),
                        "session_id": getattr(result, "session_id", None)}
            except Exception:
                return None

        try:
            self.astra_future = self.ctx.spawn(ask)
            self._log("astra_request", tick=_tick(obs), reason=reason, call=self.astra_calls)
        except Exception:
            self.astra_future = None

    def _poll_astra(self, obs):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = result.get("text", "") if isinstance(result, dict) else getattr(result, "text", "")
            session = result.get("session_id") if isinstance(result, dict) else getattr(result, "session_id", None)
            if session:
                self.astra_session = session
            match = re.search(r"\{.*\}", text, flags=re.DOTALL) if isinstance(text, str) else None
            data = json.loads(match.group(0)) if match else {}
            if not isinstance(data, dict):
                return
            if data.get("circulation") in {"shortest", "clockwise", "counterclockwise"}:
                self.protocol["circulation"] = data["circulation"]
            if data.get("delivery") in {"self", "partner", "balanced"}:
                self.protocol["delivery"] = data["delivery"]
            if data.get("handoff") in {"counter", "direct", "none"}:
                self.protocol["handoff"] = data["handoff"]
            if data.get("replan") in {"event", "blocked"}:
                self.protocol["replan"] = data["replan"]
            self._log("astra_protocol", tick=_tick(obs), protocol=dict(self.protocol))
        except Exception:
            return

    # ---------- observation-derived ledger ----------

    def _partner_intent(self, obs, pots=None):
        if pots is None:
            pots = _pot_view(obs)
        _, partner = _actors(obs)
        ppos = _point(partner.get("position"))
        held = _held_name(partner)
        if ppos is None:
            return {"kind": "none", "pot_id": "", "item": ""}
        if held in INGREDIENTS:
            choices = []
            for pot in pots:
                if pot["phase"] in {"empty", "partial"} and held in pot["missing"]:
                    distance = _route_len_to_feature(_terrain(obs), ppos, pot["pos"])
                    choices.append((distance, pot["id"], pot))
            if choices:
                choices.sort(key=lambda x: (x[0], x[1]))
                return {"kind": "ingredient", "pot_id": choices[0][1], "item": held}
            return {"kind": "ingredient", "pot_id": "", "item": held}
        if held == "dish":
            choices = [p for p in pots if p["phase"] in {"ready", "cooking"}]
            choices.sort(key=lambda p: (0 if p["ready"] else 1,
                                        _route_len_to_feature(_terrain(obs), ppos, p["pos"]), p["id"]))
            return {"kind": "dish", "pot_id": choices[0]["id"] if choices else "", "item": "dish"}
        if held == "soup":
            return {"kind": "delivery", "pot_id": "", "item": "soup"}
        return {"kind": "none", "pot_id": "", "item": ""}

    def _world_signature(self, obs):
        me, partner = _actors(obs)
        pots = _pot_view(obs)
        pot_sig = tuple((p["id"], p["phase"], tuple(p["ingredients"])) for p in pots)
        score = obs.get("sparse_score", 0) if isinstance(obs, dict) else 0
        try:
            score = float(score)
        except Exception:
            score = 0.0
        return (
            _point(me.get("position")), _held_name(me),
            _point(partner.get("position")), _held_name(partner),
            pot_sig, score,
        )

    def _record_progress(self, obs):
        current = _point(_actors(obs)[0].get("position"))
        world = self._world_signature(obs)
        if self.last_world is None:
            self.last_world = world
            self.last_pos = current
            self.last_tick = _tick(obs)
            return
        moved = current is not None and current != self.last_pos
        changed = world != self.last_world
        if moved or changed:
            self.blocked = 0
            self.stall = 0
        elif self.job and self.job.get("kind") not in {"stage_dish", "park", "hold"}:
            self.blocked += 1
            self.stall += 1
        self.last_world = world
        self.last_pos = current
        self.last_tick = _tick(obs)

    # ---------- task graph ----------

    def _task(self, kind, target=None, item="", pot=None, description="", rank=10, goal=None):
        target = _point(target)
        goal = _point(goal)
        pot_id = pot.get("id", "") if isinstance(pot, dict) else ""
        point_tag = "none" if target is None else "%d_%d" % target
        goal_tag = "" if goal is None else "_g%d_%d" % goal
        ident = "%s_%s_%s%s" % (kind, item or "x", point_tag, goal_tag)
        if pot_id:
            ident += "_" + pot_id
        return {
            "id": ident,
            "kind": kind,
            "target": target,
            "goal": goal,
            "item": item,
            "pot_id": pot_id,
            "description": description or kind,
            "rank": int(rank),
        }

    def _occupied_positions(self, obs):
        return {_object_pos(obj) for obj in _objects(obs) if _object_pos(obj) is not None}

    def _partner_pot_id(self, obs, pots):
        return self._partner_intent(obs, pots).get("pot_id", "")

    def _safe_cell(self, obs, near=None, forbidden=()):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        start = _point(me.get("position"))
        ppos = _point(partner.get("position"))
        forbidden = set(_point(p) for p in forbidden if _point(p) is not None)
        if ppos is not None:
            forbidden.add(ppos)
        if start is None:
            return None
        target = _point(near)
        poses = set()
        if target is not None:
            poses.update(_interaction_poses(terrain, target))
        candidates = []
        for cell in _all_floor(terrain):
            if cell in forbidden or cell in poses:
                continue
            path = _bfs(terrain, start, cell, forbidden)
            if path is None:
                continue
            to_partner = abs(cell[0] - ppos[0]) + abs(cell[1] - ppos[1]) if ppos else 0
            to_target = abs(cell[0] - target[0]) + abs(cell[1] - target[1]) if target else 0
            # Prefer an accessible, non-gate cell which remains reasonably close
            # to the next event, while separating from the partner.
            candidates.append((to_target, -to_partner, len(path), cell))
        if candidates:
            candidates.sort()
            return candidates[0][3]
        return start

    def _staging_goal(self, obs, pot):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        start = _point(me.get("position"))
        ppos = _point(partner.get("position"))
        target = pot.get("pos") if isinstance(pot, dict) else None
        if start is None or target is None:
            return start
        poses = set(_interaction_poses(terrain, target))
        candidates = []
        for cell in _all_floor(terrain):
            if cell in poses or cell == ppos:
                continue
            distance = abs(cell[0] - target[0]) + abs(cell[1] - target[1])
            if distance > 4:
                continue
            path = _bfs(terrain, start, cell, {ppos} if ppos else ())
            if path is None:
                continue
            away = abs(cell[0] - ppos[0]) + abs(cell[1] - ppos[1]) if ppos else 0
            candidates.append((distance, -away, len(path), cell))
        if candidates:
            candidates.sort()
            return candidates[0][3]
        return self._safe_cell(obs, near=target, forbidden=poses)

    def _dish_job_for_pot(self, obs, pot):
        me, _ = _actors(obs)
        start = _point(me.get("position"))
        if start is None:
            return self._task("park", goal=start, description="hold dish safely", rank=10)
        remaining = pot.get("remaining")
        _, route = _best_pose_path(_terrain(obs), start, pot.get("pos"))
        travel = len(route) if route is not None else 9999
        if pot.get("phase") == "ready" or (isinstance(remaining, int) and remaining <= travel + 3):
            return self._task("stage_dish", pot.get("pos"), pot=pot,
                              description="approach the pot just before pickup", rank=2,
                              goal=_interaction_poses(_terrain(obs), pot.get("pos"))[0]
                              if _interaction_poses(_terrain(obs), pot.get("pos")) else None)
        return self._task("park", pot.get("pos"), pot=pot,
                          description="park with dish away from the pot gate", rank=7,
                          goal=self._staging_goal(obs, pot))

    def _tasks(self, obs):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        held = _held_name(me)
        partner_held = _held_name(partner)
        pots = _pot_view(obs)
        result = []

        if held == "soup":
            for target in _feature_cells(terrain, "S"):
                result.append(self._task("deliver", target, description="deliver held soup", rank=0))
            return result or [self._task("hold", description="hold soup safely", rank=0)]

        if held == "dish":
            ready = [p for p in pots if p["ready"]]
            if ready:
                for pot in ready:
                    result.append(self._task("pickup_ready", pot["pos"], pot=pot,
                                             description="pick up ready soup with dish", rank=1))
                return result
            active = [p for p in pots if p["cooking"]]
            if active:
                return [self._dish_job_for_pot(obs, p) for p in active]
            return [self._task("park", goal=self._safe_cell(obs),
                               description="park while holding a dish", rank=8)]

        if held in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial"} and held in p["missing"]]
            compatible.sort(key=lambda p: (0 if p["id"] != self._partner_pot_id(obs, pots) else 1,
                                           len(p["missing"]), p["id"]))
            for pot in compatible:
                result.append(self._task("fill_pot", pot["pos"], item=held, pot=pot,
                                         description="put held %s into a compatible pot" % held, rank=3))
            for site in _counter_spaces(obs)[:8]:
                result.append(self._task("deposit_counter", site, item=held,
                                         description="stage held %s on a counter" % held, rank=6))
            return result or [self._task("hold", description="hold ingredient safely", rank=6)]

        # Empty hands: preserve hard event priorities, but retain alternatives
        # instead of returning a single globally locked job.
        ready = [p for p in pots if p["ready"]]
        if ready and partner_held not in {"dish", "soup"}:
            for target in _feature_cells(terrain, "D"):
                nearest = min(ready, key=lambda p: (_route_len_to_feature(terrain, _point(me.get("position")), p["pos"]), p["id"]))
                result.append(self._task("fetch_dish", target, pot=nearest,
                                         description="fetch a dish for ready soup", rank=1))

        for pot in pots:
            if pot["phase"] == "full_idle":
                result.append(self._task("start_cooking", pot["pos"], pot=pot,
                                         description="explicitly start a full idle pot", rank=2))

        partner_pot = self._partner_pot_id(obs, pots)
        for pot in pots:
            if pot["phase"] not in {"empty", "partial"} or not pot["missing"]:
                continue
            item = pot["missing"][0]
            for name, site in _counter_objects(obs):
                if name == item:
                    result.append(self._task("fetch_counter", site, item=item, pot=pot,
                                             description="retrieve staged %s" % item, rank=4))
            dispenser = "O" if item == "onion" else "T"
            for target in _feature_cells(terrain, dispenser):
                rank = 5 if pot["id"] != partner_pot else 6
                result.append(self._task("fetch_ingredient", target, item=item, pot=pot,
                                         description="fetch one %s for %s" % (item, pot["id"]), rank=rank))

        if any(p["cooking"] for p in pots) and partner_held not in {"dish", "soup"}:
            for target in _feature_cells(terrain, "D"):
                active = min((p for p in pots if p["cooking"]),
                             key=lambda p: (_route_len_to_feature(terrain, _point(me.get("position")), p["pos"]), p["id"]))
                result.append(self._task("fetch_dish", target, pot=active,
                                         description="fetch a dish during cooking", rank=7))
        elif partner_held not in {"dish", "soup"}:
            for target in _feature_cells(terrain, "D"):
                result.append(self._task("fetch_dish", target, description="stage a future dish", rank=9))

        if not result:
            result.append(self._task("park", goal=self._safe_cell(obs),
                                     description="move to a safe non-gate cell", rank=10))
        return result

    def _task_key(self, task):
        if not isinstance(task, dict):
            return None
        return (task.get("kind"), task.get("target"), task.get("pot_id"), task.get("item"))

    def _equivalent(self, old, new):
        if not old or not new:
            return False
        old_kind, new_kind = old.get("kind"), new.get("kind")
        if old_kind == "stage_dish" and new_kind in {"stage_dish", "pickup_ready", "park"}:
            return old.get("pot_id") == new.get("pot_id")
        if old_kind == "park" and new_kind in {"park", "stage_dish"}:
            if old.get("pot_id") or new.get("pot_id"):
                return bool(old.get("pot_id") and old.get("pot_id") == new.get("pot_id"))
            return new_kind == "park" and self._task_key(old) == self._task_key(new)
        return self._task_key(old) == self._task_key(new)

    def _rank_with_partner(self, task, partner_intent):
        rank = int(task.get("rank", 10))
        if partner_intent.get("pot_id") and task.get("pot_id") == partner_intent.get("pot_id"):
            if task.get("kind") in {"fetch_ingredient", "fetch_counter", "fill_pot"}:
                rank += 12
        if self.protocol.get("delivery") == "self" and task.get("kind") in {"fetch_dish", "stage_dish", "pickup_ready"}:
            rank -= 1
        if self.protocol.get("delivery") == "partner" and task.get("kind") in {"fetch_dish", "stage_dish"}:
            rank += 1
        if self.protocol.get("handoff") == "counter" and task.get("kind") in {"fetch_counter", "deposit_counter"}:
            rank -= 1
        return rank

    def _choose_task(self, obs, tasks):
        if not tasks:
            return self._task("hold", description="safe hold", rank=20)
        me, partner = _actors(obs)
        start = _point(me.get("position"))
        ppos = _point(partner.get("position"))
        predicted = _predicted_partner(obs)
        intent = self._partner_intent(obs)
        now = _tick(obs)
        usable = [t for t in tasks if self.avoid.get(self._task_key(t), -1) <= now]
        if not usable:
            usable = list(tasks)
        scored = []
        for task in usable:
            target = task.get("target")
            goal = task.get("goal")
            if task.get("kind") in {"park", "hold"}:
                path = _bfs(_terrain(obs), start, goal, {ppos} if ppos is not None else ()) if goal is not None else [start]
            else:
                forbidden = {p for p in (ppos, predicted) if p is not None}
                _, path = _best_pose_path(_terrain(obs), start, target, forbidden)
            distance = len(path) if path is not None else 9999
            rank = self._rank_with_partner(task, intent)
            # A parked dish is a useful fallback, but never beats a legal ready
            # delivery or explicit start.
            if task.get("kind") in {"park", "hold"} and any(t.get("kind") in URGENT_KINDS for t in tasks):
                rank += 8
            side = 0
            if target is not None and self.protocol.get("circulation") != "shortest":
                width = len(_terrain(obs)[0]) if _terrain(obs) and isinstance(_terrain(obs)[0], (str, list, tuple)) else 0
                on_left = target[0] < width / 2 if width else False
                wanted_left = self.protocol.get("circulation") == "counterclockwise"
                side = 0 if on_left == wanted_left else 1
            contention = 1 if ppos is not None and target is not None and abs(target[0] - ppos[0]) + abs(target[1] - ppos[1]) <= 1 else 0
            scored.append((rank, contention, side, distance, str(task.get("id", "")), task))
        scored.sort(key=lambda row: row[:-1])
        best_rank = scored[0][0]

        # Jev is a tie-breaker over an already safe event band.  A stale or
        # low-confidence answer cannot turn delivery into speculative work.
        suggested = None
        if self.jev_task and _tick(obs) - self.jev_tick <= 8:
            suggested = next((row for row in scored if row[5].get("id") == self.jev_task), None)
        if suggested is not None and suggested[0] <= best_rank:
            chosen = suggested[5].copy()
        else:
            chosen = scored[0][5].copy()
        return chosen

    def _refresh_job(self, obs, tasks):
        valid = False
        if self.job:
            for task in tasks:
                if self._equivalent(self.job, task):
                    # Recompute a staging/parking destination only when the old
                    # destination is no longer usable.  Otherwise a moving
                    # partner would make a dish holder chase a new safe cell on
                    # every tick.
                    old_goal = _point(self.job.get("goal"))
                    keep_goal = False
                    if old_goal is not None and self.job.get("kind") == task.get("kind") and self.job.get("kind") in {"park", "stage_dish"}:
                        partner_pos = _point(_actors(obs)[1].get("position"))
                        keep_goal = (_walkable(_terrain(obs), old_goal) and old_goal != partner_pos)
                    self.job.update({"kind": task.get("kind"), "target": task.get("target"),
                                     "goal": old_goal if keep_goal else task.get("goal"),
                                     "description": task.get("description"),
                                     "rank": task.get("rank"), "pot_id": task.get("pot_id"),
                                     "item": task.get("item", "")})
                    valid = True
                    break
        if valid and self.blocked >= 6 and self.job.get("kind") not in {"deliver", "pickup_ready", "start_cooking", "stage_dish", "park"}:
            self.avoid[self._task_key(self.job)] = _tick(obs) + 5
            valid = False
        # A partner's newly acquired ingredient is a live reassignment event,
        # not merely a score penalty.  Abandon only an empty-hand fetch when a
        # lower-ranked alternative exists; once carrying an item, complete or
        # stage it rather than dropping the transaction mid-route.
        if valid and self.job.get("kind") in {"fetch_ingredient", "fetch_counter"}:
            intent = self._partner_intent(obs)
            if intent.get("pot_id") and intent.get("pot_id") == self.job.get("pot_id"):
                current_rank = self._rank_with_partner(self.job, intent)
                alternatives = [t for t in tasks
                                if self._task_key(t) != self._task_key(self.job)
                                and self._rank_with_partner(t, intent) < current_rank]
                if alternatives:
                    self.avoid[self._task_key(self.job)] = _tick(obs) + 5
                    valid = False
        if valid:
            return
        chosen = self._choose_task(obs, tasks)
        self.job = chosen
        self.job_epoch += 1
        self.blocked = 0
        self.stall = 0
        self.last_commit_key = self._task_key(chosen)
        self._log("job_commit", tick=_tick(obs), epoch=self.job_epoch,
                  kind=chosen.get("kind"), target=chosen.get("target"), goal=chosen.get("goal"),
                  pot=chosen.get("pot_id"))

    # ---------- Jev bridge ----------

    def _compact(self, obs, tasks):
        me, partner = _actors(obs)
        pots = _pot_view(obs)
        intent = self._partner_intent(obs, pots)
        return _jsonable({
            "game": "Overcooked-AI symbolic grid kitchen",
            "tick": _tick(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {"position": _point(me.get("position")), "held": _held_name(me),
                     "last_action": self.last_action},
            "partner": {"position": _point(partner.get("position")), "held": _held_name(partner),
                        "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else "",
                        "predicted_next": _predicted_partner(obs), "intent": intent},
            "pots": [{"id": p["id"], "position": p["pos"], "phase": p["phase"],
                      "ingredients": p["ingredients"], "missing": p["missing"],
                      "cook_ticks_remaining": p["remaining"]} for p in pots],
            "current_job": {"kind": self.job.get("kind", "") if self.job else "",
                            "target": self.job.get("target") if self.job else None,
                            "pot_id": self.job.get("pot_id", "") if self.job else "",
                            "blocked_ticks": self.blocked, "stall_ticks": self.stall},
            "protocol": dict(self.protocol),
            "candidates": [{"id": t.get("id"), "kind": t.get("kind"), "pot_id": t.get("pot_id", ""),
                            "description": t.get("description")} for t in tasks[:12]],
        })

    def _request_jev(self, obs, tasks, reason):
        if self.s1_disabled or self.jev_future is not None or not tasks or self._closed():
            return
        now = _tick(obs)
        state = self._compact(obs, tasks)
        key = (self.job_epoch, json.dumps(state, sort_keys=True, separators=(",", ":"))[:5000])
        if self.jev_key == key and now - self.jev_tick < 8:
            return
        if now - self.jev_tick < 3 and reason != "commit":
            return
        self.jev_key = key
        self.jev_tick = now
        self.jev_job_epoch = self.job_epoch
        criteria = {str(t.get("id")): str(t.get("description", t.get("kind", "candidate"))) for t in tasks[:12]}
        if not criteria:
            return
        questions = {
            "event": Choice(
                "Choose one id from `candidates` for the next event. The candidates were generated "
                "from the current symbolic state. Prefer delivery, ready-soup pickup, explicit start, "
                "or filling a compatible pot; never invent a direction, object, or id.", criteria),
            "partner": Choice(
                "Classify the partner's immediate physical intent from `partner.held`, "
                "`partner.intent`, `partner.position`, and `partner.last_action`.", {
                    "ingredient": "The partner is carrying an ingredient toward a compatible pot.",
                    "dish": "The partner is carrying a dish toward a cooking or ready pot.",
                    "delivery": "The partner is carrying soup toward serving.",
                    "none": "No carried-object transaction is supported by the listed facts.",
                }),
            "release": Noul(
                "Should the controlled chef abandon its current non-urgent approach and recompute "
                "another legal event because the partner owns the gate? True only when the current "
                "job is non-urgent, `current_job.blocked_ticks` is at least 3, and the partner facts "
                "show a carried ingredient or dish transaction. Never release a ready pickup, delivery, "
                "or explicit start solely because of uncertainty.", {
                    "true": "A short replan is safer than contesting a demonstrably occupied gate.",
                    "false": "Continue the current event; there is no sufficiently evidenced gate conflict.",
                }),
        }

        def ask():
            if self._closed():
                return {}
            try:
                return self.ctx.decide(state, questions, timeout_s=1.0)
            except Exception:
                return {}

        try:
            self.jev_future = self.ctx.spawn(ask)
            self._log("jev_request", tick=now, epoch=self.job_epoch, reason=reason)
        except Exception:
            self.jev_future = None

    def _poll_jev(self, obs, tasks):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict) or self.jev_job_epoch != self.job_epoch:
            return
        current = {t.get("id"): t for t in tasks}
        try:
            answer = answers.get("event")
            value = getattr(answer, "value", None)
            probabilities = getattr(answer, "probabilities", {}) or {}
            probability = float(probabilities.get(value, 0.0)) if isinstance(probabilities, dict) else 0.0
            if value in current:
                task = current[value]
                safe_rank = min(self._rank_with_partner(t, self._partner_intent(obs)) for t in tasks)
                task_rank = self._rank_with_partner(task, self._partner_intent(obs))
                if task_rank <= safe_rank and (probability >= 0.5 or probability == 0.0):
                    self.jev_task = value
                    self._log("jev_event", tick=_tick(obs), value=value, probability=probability)
            panswer = answers.get("partner")
            pvalue = getattr(panswer, "value", None)
            if pvalue in {"ingredient", "dish", "delivery", "none"}:
                self.jev_partner_hint = pvalue
            ranswer = answers.get("release")
            rvalue = getattr(ranswer, "value", 0.0)
            if isinstance(rvalue, (int, float)) and rvalue >= 0.75 and self.blocked >= 3:
                if self.job and self.job.get("kind") not in URGENT_KINDS:
                    self.jev_release = True
        except Exception:
            return

    # ---------- physical legality and receding-horizon motor ----------

    def _valid_interaction(self, obs, job):
        me, _ = _actors(obs)
        held = _held_name(me)
        target = _point(job.get("target"))
        if target is None:
            return False
        terrain = _terrain(obs)
        kind = job.get("kind")
        if kind == "deliver":
            return held == "soup" and _cell(terrain, target) == "S"
        if kind == "fetch_dish":
            return not held and _cell(terrain, target) == "D"
        if kind == "fetch_ingredient":
            return not held and _cell(terrain, target) in {"O", "T"}
        if kind == "fetch_counter":
            return not held and any(name == job.get("item") and p == target for name, p in _counter_objects(obs))
        if kind == "deposit_counter":
            return held in INGREDIENTS and _cell(terrain, target) == "X" and target not in self._occupied_positions(obs)
        if kind == "fill_pot":
            pot = next((p for p in _pot_view(obs) if p["id"] == job.get("pot_id") or p["pos"] == target), None)
            return bool(pot and held in INGREDIENTS and pot["phase"] in {"empty", "partial"} and held in pot["missing"])
        if kind == "start_cooking":
            pot = next((p for p in _pot_view(obs) if p["id"] == job.get("pot_id") or p["pos"] == target), None)
            return bool(pot and not held and pot["phase"] == "full_idle")
        if kind == "pickup_ready":
            pot = next((p for p in _pot_view(obs) if p["id"] == job.get("pot_id") or p["pos"] == target), None)
            return bool(pot and held == "dish" and pot["ready"])
        return False

    def _dynamic_path(self, obs, job):
        me, partner = _actors(obs)
        current = _point(me.get("position"))
        ppos = _point(partner.get("position"))
        predicted = _predicted_partner(obs)
        kind = job.get("kind") if job else ""
        goal = _point(job.get("goal")) if job else None
        if kind in {"park", "stage_dish"} and goal is not None:
            blocked = {p for p in (ppos, predicted) if p is not None and p != goal}
            path = _bfs(_terrain(obs), current, goal, blocked)
            if path is not None:
                return goal, path
            return goal, _bfs(_terrain(obs), current, goal)
        target = _point(job.get("target")) if job else None
        blocked = {p for p in (ppos, predicted) if p is not None}
        return _best_pose_path(_terrain(obs), current, target, blocked, job.get("pose") if job else None)

    def _escape_action(self, obs, job):
        me, partner = _actors(obs)
        current = _point(me.get("position"))
        ppos = _point(partner.get("position"))
        predicted = _predicted_partner(obs)
        if current is None:
            return self._safe_action(self.noop)
        target = _point(job.get("target")) if job else None
        choices = []
        for nxt, delta in _neighbors(current):
            if not _walkable(_terrain(obs), nxt) or nxt in {ppos, predicted}:
                continue
            path = _bfs(_terrain(obs), nxt, target) if target is not None else None
            distance = len(path) if path is not None else 999
            away = abs(nxt[0] - ppos[0]) + abs(nxt[1] - ppos[1]) if ppos else 0
            reverse = 1 if nxt == self.last_pos else 0
            choices.append((distance, reverse, -away, nxt, ACTION_FOR_VECTOR.get(delta, self.noop)))
        if choices:
            choices.sort()
            return self._safe_action(choices[0][4])
        return self._safe_action(self.noop)

    def _motor(self, obs):
        job = self.job
        if not job:
            return self._safe_action(self.noop)
        me, partner = _actors(obs)
        current = _point(me.get("position"))
        facing = _point(me.get("orientation"))
        other = _point(partner.get("position"))
        if current is None:
            return self._safe_action(self.noop)
        kind = job.get("kind")
        if kind in {"hold", "park", "stage_dish"} and kind == "hold":
            return self._safe_action(self.noop)
        goal, path = self._dynamic_path(obs, job)
        if goal is None:
            return self._safe_action(self.noop)
        if path is None:
            self.blocked += 1
            return self._escape_action(obs, job)
        if len(path) <= 1:
            if kind in {"park", "stage_dish"}:
                return self._safe_action(self.noop)
            target = _point(job.get("target"))
            if target is None:
                return self._safe_action(self.noop)
            delta = (target[0] - current[0], target[1] - current[1])
            turn = ACTION_FOR_VECTOR.get(delta)
            if turn and facing != delta:
                return self._safe_action(turn)
            if self._valid_interaction(obs, job):
                return self._safe_action("interact")
            self.blocked += 1
            return self._escape_action(obs, job)

        nxt = path[1]
        predicted = _predicted_partner(obs)
        if nxt == other or nxt == predicted or (predicted == current and nxt == other):
            self.blocked += 1
            if self.blocked <= 2 and kind in URGENT_KINDS:
                # A single tick of right-of-way is useful for a ready pot, but
                # never turns into the long logical waits of earlier candidates.
                return self._safe_action(self.noop)
            return self._escape_action(obs, job)
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        return self._safe_action(ACTION_FOR_VECTOR.get(delta, self.noop))

    # ---------- realtime episode ----------

    def run(self):
        while not self._closed() and not bool(getattr(self.game, "done", False)):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self.last_snapshot = obs
            self._poll_astra(obs)
            self._record_progress(obs)
            pots = _pot_view(obs)
            tasks = self._tasks(obs)
            self._poll_jev(obs, tasks)

            if self.astra_future is None and self.astra_calls == 0:
                self._launch_astra(obs, "initial")
            elif self.astra_future is None and self.stall >= 14 and self.astra_calls < 2:
                self.astra_revision = True
                self._launch_astra(obs, "blocked_recovery")

            self._refresh_job(obs, tasks)
            if self.jev_release and self.job and self.job.get("kind") not in URGENT_KINDS:
                self.avoid[self._task_key(self.job)] = _tick(obs) + 4
                self.job = None
                self.jev_release = False
                self._refresh_job(obs, tasks)
                self._log("jev_replan", tick=_tick(obs), reason="partner_gate")

            if self.job and (self.job_epoch != self.jev_job_epoch or self.blocked >= 2 or self.stall >= 8):
                self._request_jev(obs, tasks, "commit" if self.job_epoch != self.jev_job_epoch else "blocked")

            action = self._motor(obs)
            try:
                if self._closed():
                    return
                self.ctx.act(action)
                self.last_action = action
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                try:
                    if not self._closed():
                        self.ctx.act(self._safe_action(self.noop))
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
        AdaptiveKitchen(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
