"""Candidate 0022: route-reserved production.

This controller treats a waiting chef as a moving traffic participant, not as a
piece of furniture.  It keeps a production macro (fetch/fill/start/dish/serve)
but continuously reserves the partner's *current route* from the symbolic
observation.  A dish carrier waiting for a cooking pot is moved to a floor cell
outside the partner's predicted ingredient/dish corridor.  The reservation is
recomputed every tick, so a blocked approach does not become a fixed parking
lease.

The models can choose among code-generated equal-band transactions and can
label a live traffic conflict.  They never provide directions or state facts
that the executor has not checked.
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
HELD = INGREDIENTS | {"dish", "soup"}


def point(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            if isinstance(value[0], bool) or isinstance(value[1], bool):
                return None
            return int(value[0]), int(value[1])
        except Exception:
            return None
    return None


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def state(obs):
    return obs.get("state", {}) if isinstance(obs, dict) and isinstance(obs.get("state"), dict) else {}


def terrain(obs):
    value = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return value if isinstance(value, list) else []


def tick(obs):
    try:
        return int(obs.get("tick", state(obs).get("timestep", 0)))
    except Exception:
        return 0


def cell(grid, p):
    p = point(p)
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


def open_cell(grid, p):
    return cell(grid, p) not in FEATURES and cell(grid, p) != "X"


def feature_cells(grid, symbol):
    found = []
    for y, row in enumerate(grid):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                found.append((x, y))
    return found


def players(obs):
    raw = state(obs).get("players", [])
    return raw if isinstance(raw, list) else []


def actors(obs):
    ps = players(obs)
    try:
        mine = int(obs.get("controlled_agent_index", 0))
    except Exception:
        mine = 0
    if mine not in (0, 1) or mine >= len(ps):
        mine = 0
    other = 1 - mine if len(ps) > 1 else mine
    me = ps[mine] if mine < len(ps) and isinstance(ps[mine], dict) else {}
    partner = ps[other] if other < len(ps) and isinstance(ps[other], dict) else {}
    return me, partner


def held(player):
    raw = player.get("held_object") if isinstance(player, dict) else None
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("name", "") or "").lower()


def objects(obs):
    raw = state(obs).get("objects", [])
    return raw if isinstance(raw, list) else []


def obj_name(value):
    return str(value.get("name", "") or "").lower() if isinstance(value, dict) else ""


def obj_pos(value):
    return point(value.get("position")) if isinstance(value, dict) else None


def ingredients(value):
    if not isinstance(value, dict):
        return []
    raw = value.get("_ingredients", value.get("ingredients", []))
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


def recipes(obs):
    raw = state(obs).get("all_orders", [])
    answer = []
    if isinstance(raw, list):
        for order in raw:
            vals = order.get("ingredients") if isinstance(order, dict) else order
            if not isinstance(vals, list):
                continue
            row = []
            for item in vals:
                if isinstance(item, dict):
                    item = item.get("name", "")
                item = str(item or "").lower()
                if item in INGREDIENTS:
                    row.append(item)
            if row:
                answer.append(row)
    return answer or [["onion", "onion", "onion"]]


def missing_items(have, orders):
    counts = Counter(have)
    chosen = None
    missing = None
    for recipe in orders:
        wanted = Counter(recipe)
        if any(counts[k] > wanted[k] for k in counts):
            continue
        left = Counter(have)
        need = []
        for item in recipe:
            if left[item] < wanted[item]:
                need.append(item)
                left[item] += 1
        if missing is None or len(need) < len(missing):
            chosen, missing = list(recipe), need
    return chosen or [], missing if missing is not None else []


def neighbours(p):
    x, y = p
    for delta in STEP.values():
        yield (x + delta[0], y + delta[1]), delta


def bfs(grid, start, goals, forbidden=()):
    start = point(start)
    if start is None:
        return None
    if isinstance(goals, (list, tuple, set)) and goals and isinstance(next(iter(goals)), int):
        goals = [goals]
    goals = {point(g) for g in (goals or []) if point(g) is not None}
    if not goals:
        return None
    blocked = {point(p) for p in forbidden if point(p) is not None}
    blocked.discard(start)
    queue = deque([start])
    previous = {start: None}
    if start in goals:
        return [start]
    while queue:
        cur = queue.popleft()
        for nxt, _ in neighbours(cur):
            if nxt in previous or nxt in blocked or not open_cell(grid, nxt):
                continue
            previous[nxt] = cur
            if nxt in goals:
                path = [nxt]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return None


def poses(grid, target):
    target = point(target)
    if target is None:
        return []
    ans = []
    for delta in STEP.values():
        candidate = (target[0] - delta[0], target[1] - delta[1])
        if open_cell(grid, candidate) and candidate not in ans:
            ans.append(candidate)
    return ans


def route(grid, start, target, forbidden=()):
    choices = []
    for pose in poses(grid, target):
        path = bfs(grid, start, [pose], forbidden)
        if path is not None:
            choices.append((len(path), pose, path))
    if not choices:
        # A task remains meaningful if the partner currently occupies its only
        # approach.  The motor will wait/escape rather than discard the task.
        for pose in poses(grid, target):
            path = bfs(grid, start, [pose])
            if path is not None:
                choices.append((len(path), pose, path))
    if not choices:
        return None, None
    choices.sort(key=lambda x: (x[0], x[1]))
    return choices[0][1], choices[0][2]


def all_approaches(obs):
    grid = terrain(obs)
    ans = set()
    for symbol in FEATURES:
        for target in feature_cells(grid, symbol):
            ans.update(poses(grid, target))
    return ans


def counter_sites(obs):
    grid = terrain(obs)
    occupied = {obj_pos(o) for o in objects(obs) if obj_pos(o) is not None}
    answer = []
    for y, row in enumerate(grid):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            p = (x, y)
            if value == "X" and p not in occupied and any(open_cell(grid, q) for q, _ in neighbours(p)):
                answer.append(p)
    return answer


def counter_objects(obs):
    grid = terrain(obs)
    answer = []
    for obj in objects(obs):
        p = obj_pos(obj)
        name = obj_name(obj)
        if p is not None and name in HELD and cell(grid, p) == "X":
            answer.append((name, p))
    return answer


def pot_facts(obs):
    grid = terrain(obs)
    order_list = recipes(obs)
    by_pos = {obj_pos(o): o for o in objects(obs) if obj_pos(o) is not None and obj_name(o) == "soup"}
    result = []
    for p in feature_cells(grid, "P"):
        soup = by_pos.get(p)
        have = ingredients(soup)
        recipe, missing = missing_items(have, order_list)
        ready = bool(soup and soup.get("is_ready"))
        cooking = bool(soup and soup.get("is_cooking")) and not ready
        c_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        c_time = soup.get("cook_time", -1) if soup else -1
        if soup and not ready and not cooking and isinstance(c_tick, (int, float)) and isinstance(c_time, (int, float)):
            cooking = c_tick >= 0 and c_time >= 0 and not bool(soup.get("is_idle"))
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
        if cooking and isinstance(c_tick, (int, float)) and isinstance(c_time, (int, float)):
            remaining = max(0, int(c_time - c_tick))
        result.append({"pos": p, "phase": phase, "ingredients": have,
                       "recipe": recipe, "missing": missing, "ready": ready,
                       "cooking": cooking, "full": full, "remaining": remaining})
    return result


def partner_next(obs):
    _, partner = actors(obs)
    p = point(partner.get("position"))
    raw = obs.get("partner_action_last", "") if isinstance(obs, dict) else ""
    delta = STEP.get(raw, (0, 0))
    return (p[0] + delta[0], p[1] + delta[1]) if p is not None else None


class RouteReservedPilot:
    """A production state machine with a dynamic partner-route reservation."""

    def __init__(self, ctx, game):
        self.ctx, self.game = ctx, game
        spec = getattr(game, "action_spec", None)
        self.legal = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.legal:
            self.noop = "stay" if "stay" in self.legal else (next(iter(self.legal), "stay"))
        self.task = None
        self.task_since = -1
        self.epoch = 0
        self.last_action = self.noop
        self.last_me = None
        self.last_partner = None
        self.blocked = 0
        self.no_progress = 0
        self.yield_until = -1
        self.hint = None
        self.hint_tick = -999
        self.jev_future = None
        self.jev_epoch = -1
        self.last_jev = -999
        self.astra_future = None
        self.astra_started = False
        self.policy = {"parking": "route_aware", "handoff": "predictive", "tempo": "pipeline"}
        self.system1_off = getattr(ctx, "system1", object()) is None
        self.last_obs = {}

    def log(self, name, **fields):
        try:
            self.ctx.log(name, **fields)
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
        mp, pp = point(me.get("position")), point(partner.get("position"))
        if self.last_me is not None and mp == self.last_me and self.last_action in STEP:
            self.blocked += 1
        elif mp is not None and mp != self.last_me:
            self.blocked = max(0, self.blocked - 1)
        if self.last_partner is not None and pp == self.last_partner:
            self.no_progress += 1
        else:
            self.no_progress = 0
        self.last_me, self.last_partner = mp, pp

    # ---------- partner route reservation ---------------------------------

    def partner_intent(self, obs):
        _, p = actors(obs)
        ph = held(p)
        facts = pot_facts(obs)
        grid = terrain(obs)
        start = point(p.get("position"))
        if start is None:
            return {"kind": "none", "target": None, "path": [], "urgent": False}
        targets = []
        kind = "none"
        urgent = False
        if ph == "soup":
            kind, targets, urgent = "serve", feature_cells(grid, "S"), True
        elif ph == "dish":
            ready = [f["pos"] for f in facts if f["ready"]]
            cooking = [f["pos"] for f in facts if f["cooking"]]
            if ready:
                kind, targets, urgent = "ready", ready, True
            elif cooking:
                kind, targets, urgent = "cooking", cooking, True
        elif ph in INGREDIENTS:
            matching = [f for f in facts if f["phase"] in {"empty", "partial", "full_idle"}
                        and ph in f["missing"]]
            if matching:
                kind, targets, urgent = "fill", [min(matching, key=lambda f: f["pos"])["pos" ]], False
        if not targets:
            return {"kind": kind, "target": None, "path": [], "urgent": urgent,
                    "held": ph, "stationary": self.no_progress}
        paths = []
        for target in targets:
            _, path = route(grid, start, target, ())
            if path is not None:
                paths.append((len(path), target, path))
        if not paths:
            return {"kind": kind, "target": targets[0], "path": [], "urgent": urgent,
                    "held": ph, "stationary": self.no_progress}
        paths.sort(key=lambda row: (row[0], row[1]))
        _, target, path = paths[0]
        return {"kind": kind, "target": target, "path": path, "urgent": urgent,
                "held": ph, "stationary": self.no_progress}

    def reservation(self, obs, horizon=7):
        """Cells the partner is likely to occupy before its next operation."""
        intent = self.partner_intent(obs)
        path = intent.get("path", [])
        cells = set(path[1:horizon + 1])
        if path:
            cells.add(path[-1])
            # The interaction pose is the real contested resource; include
            # nearby poses even when the route planner selected another one.
            for q in poses(terrain(obs), intent.get("target")):
                cells.add(q)
        nxt = partner_next(obs)
        if nxt is not None:
            cells.add(nxt)
        return cells, intent

    def route_conflict(self, obs, task):
        if not task:
            return False
        reserved, intent = self.reservation(obs, horizon=4)
        if not reserved or intent.get("kind") == "none":
            return False
        me, _ = actors(obs)
        cur = point(me.get("position"))
        _, path = route(terrain(obs), cur, task.get("target"), {point(actors(obs)[1].get("position"))})
        if not path:
            return True
        return any(q in reserved for q in path[1:4])

    def free_parking(self, obs, current=None, prefer=None):
        """Find a floor cell outside the partner's predicted corridor.

        Static interaction cells are only a soft penalty: a left dispenser pose
        may be the only safe refuge while the partner traverses the centre.
        """
        grid = terrain(obs)
        me, partner = actors(obs)
        current = point(current) if current is not None else point(me.get("position"))
        pp = point(partner.get("position"))
        reserved, intent = self.reservation(obs, horizon=9)
        if current is None:
            return None
        candidates = []
        for y, row in enumerate(grid):
            if not isinstance(row, (str, list, tuple)):
                continue
            for x in range(len(row)):
                q = (x, y)
                if not open_cell(grid, q) or q == pp or q in reserved:
                    continue
                forbidden = set(reserved)
                forbidden.discard(current)
                if pp is not None:
                    forbidden.add(pp)
                path = bfs(grid, current, [q], forbidden)
                if path is None:
                    continue
                gap = abs(q[0] - pp[0]) + abs(q[1] - pp[1]) if pp else 0
                corridor_gap = min((abs(q[0] - z[0]) + abs(q[1] - z[1]) for z in reserved), default=4)
                target_gap = abs(q[0] - prefer[0]) + abs(q[1] - prefer[1]) if prefer else 0
                approach_penalty = 2 if q in all_approaches(obs) else 0
                # Stay close enough to the active pot to resume, but strongly
                # prefer a cell away from the partner's route tube.
                candidates.append((approach_penalty, -corridor_gap, -gap,
                                   target_gap, len(path), q))
        if not candidates:
            return current if current != pp and current not in reserved else None
        candidates.sort()
        return candidates[0][-1]

    # ---------- facts and task creation -----------------------------------

    def make_task(self, kind, target=None, item="", priority=10, dynamic=False, tag=""):
        target = point(target)
        suffix = "none" if target is None else "%s_%s" % target
        return {"id": "%s_%s_%s_%s" % (kind, item or "none", suffix, tag or "x"),
                "kind": kind, "target": target, "item": item,
                "priority": int(priority), "dynamic": bool(dynamic), "tag": tag or ""}

    def closest(self, obs, start, targets, forbidden=()):
        choices = []
        for target in targets:
            pose, path = route(terrain(obs), start, target, forbidden)
            if path is not None:
                choices.append((len(path), target, pose))
        if not choices:
            return None, None
        choices.sort(key=lambda x: (x[0], x[1]))
        return choices[0][1], choices[0][2]

    def build_task(self, obs, current, partner, facts):
        grid = terrain(obs)
        buildable = [f for f in facts if f["phase"] in {"empty", "partial"} and f["missing"]]
        if not buildable:
            return None
        buildable.sort(key=lambda f: (len(f["missing"]), f["pos"]))
        pot = buildable[0]
        wanted = pot["missing"][0]
        staged = [q for name, q in counter_objects(obs) if name == wanted]
        if staged:
            target, _ = self.closest(obs, current, staged, {partner} if partner else ())
            if target is not None:
                return self.make_task("take_counter", target, wanted, 3, tag="matching")
        symbol = "O" if wanted == "onion" else "T"
        target, _ = self.closest(obs, current, feature_cells(grid, symbol), {partner} if partner else ())
        if target is not None:
            return self.make_task("fetch_ingredient", target, wanted, 5, tag="source")
        return None

    def cooking_due(self, obs, pot, current, partner):
        remaining = pot.get("remaining")
        if not isinstance(remaining, (int, float)):
            return True
        dishes = feature_cells(terrain(obs), "D")
        best = 999
        for dish in dishes:
            _, first = route(terrain(obs), current, dish, {partner} if partner else ())
            if first is None:
                continue
            for pose in poses(terrain(obs), pot["pos"]):
                second = bfs(terrain(obs), first[-1], [pose], {partner} if partner else ())
                if second is not None:
                    best = min(best, len(first) - 1 + len(second) - 1 + 2)
        return remaining <= max(5, best + 2)

    def plan(self, obs):
        grid = terrain(obs)
        me, partner = actors(obs)
        cur, pp = point(me.get("position")), point(partner.get("position"))
        mine, theirs = held(me), held(partner)
        facts = pot_facts(obs)

        # Object ownership is an absolute invariant.
        if mine == "soup":
            target, _ = self.closest(obs, cur, feature_cells(grid, "S"), {pp} if pp else ())
            return self.make_task("deliver", target, priority=0, tag="carry") if target is not None else self.make_task("hold", priority=0, dynamic=True)
        if mine == "dish":
            ready = [f for f in facts if f["ready"]]
            if ready:
                target, _ = self.closest(obs, cur, [f["pos"] for f in ready], {pp} if pp else ())
                return self.make_task("pickup", target, priority=1, tag="ready") if target is not None else self.make_task("wait", dynamic=True, priority=1)
            # A dish is a reservation token, not a reason to occupy the pot pose.
            return self.make_task("wait", dynamic=True, priority=4, tag="dish_wait")
        if mine in INGREDIENTS:
            compatible = [f for f in facts if f["phase"] in {"empty", "partial", "full_idle"}
                          and mine in f["missing"]]
            compatible.sort(key=lambda f: (len(f["missing"]), f["pos"]))
            if compatible:
                return self.make_task("fill", compatible[0]["pos"], mine, 2, tag="pot")
            free = counter_sites(obs)
            target, _ = self.closest(obs, cur, free, {pp} if pp else ())
            return self.make_task("put_counter", target, mine, 7, tag="buffer") if target is not None else self.make_task("hold", dynamic=True, priority=8)

        # Ready soup takes precedence unless the partner has a demonstrably
        # improving dish/soup route.  The controlled chef then does independent
        # build work while waiting, rather than freezing the whole kitchen.
        ready = [f for f in facts if f["ready"]]
        if ready:
            intent = self.partner_intent(obs)
            active_handoff = intent.get("kind") in {"ready", "cooking", "serve"} and bool(intent.get("path"))
            if not active_handoff:
                target, _ = self.closest(obs, cur, feature_cells(grid, "D"), {pp} if pp else ())
                if target is not None:
                    return self.make_task("fetch_dish", target, priority=1, tag="ready")
            independent = self.build_task(obs, cur, pp, facts)
            if independent is not None:
                return independent
            return self.make_task("wait", dynamic=True, priority=3, tag="ready_yield")

        # Modern dynamics: full idle means explicit start before any supply.
        full_idle = [f for f in facts if f["phase"] == "full_idle"]
        if full_idle:
            target, _ = self.closest(obs, cur, [f["pos"] for f in full_idle], {pp} if pp else ())
            return self.make_task("start", target, priority=2, tag="explicit") if target is not None else self.make_task("wait", dynamic=True)

        # Feed another pot while a different pot cooks, or fetch a timed dish.
        built = self.build_task(obs, cur, pp, facts)
        if built is not None:
            return built
        cooking = [f for f in facts if f["cooking"]]
        if cooking:
            cooking.sort(key=lambda f: (f.get("remaining", 999) if isinstance(f.get("remaining"), (int, float)) else 999, f["pos"]))
            pot = cooking[0]
            intent = self.partner_intent(obs)
            partner_dish_active = intent.get("kind") in {"ready", "cooking"} and bool(intent.get("path"))
            if self.cooking_due(obs, pot, cur, pp) and not partner_dish_active:
                target, _ = self.closest(obs, cur, feature_cells(grid, "D"), {pp} if pp else ())
                if target is not None:
                    return self.make_task("fetch_dish", target, priority=3, tag="timed")
            return self.make_task("wait", dynamic=True, priority=12, tag="cook_yield")

        # Seed a dish only when there is an actual pot transaction to serve;
        # otherwise build_task will resume on the next observation.
        built = self.build_task(obs, cur, pp, facts)
        if built is not None:
            return built
        return self.make_task("wait", dynamic=True, priority=20, tag="idle")

    # ---------- task validity and model-facing candidates -----------------

    def same_task(self, a, b):
        if not a or not b:
            return False
        return (a.get("kind"), point(a.get("target")), a.get("item"), a.get("tag")) == \
               (b.get("kind"), point(b.get("target")), b.get("item"), b.get("tag"))

    def legal_now(self, obs, task):
        if not task:
            return False
        kind, target = task.get("kind"), point(task.get("target"))
        me, _ = actors(obs)
        item = held(me)
        grid = terrain(obs)
        facts = pot_facts(obs)
        if kind in {"wait", "hold"}:
            if item == "dish":
                return not any(f["ready"] for f in facts)
            return True
        if target is None:
            return False
        if kind == "deliver":
            return item == "soup" and cell(grid, target) == "S"
        if kind == "fetch_dish":
            return not item and cell(grid, target) == "D"
        if kind == "fetch_ingredient":
            return not item and cell(grid, target) in {"O", "T"}
        if kind == "take_counter":
            return not item and any(n == task.get("item") and p == target for n, p in counter_objects(obs))
        if kind == "put_counter":
            return item in INGREDIENTS and target not in {obj_pos(o) for o in objects(obs)}
        pot = next((f for f in facts if f["pos"] == target), None)
        if kind == "fill":
            return bool(pot and item in INGREDIENTS and pot["phase"] in {"empty", "partial", "full_idle"}
                        and item in pot["missing"])
        if kind == "start":
            return bool(pot and item == "" and pot["phase"] == "full_idle")
        if kind == "pickup":
            return bool(pot and item == "dish" and pot["ready"])
        return False

    def candidate_tasks(self, obs):
        chosen = self.plan(obs)
        result = [chosen]
        # Equal-band alternatives make Jev useful without allowing it to alter
        # the production invariant.  In particular, dispenser ties are safe.
        if chosen.get("kind") == "fetch_ingredient":
            item = chosen.get("item")
            symbol = "O" if item == "onion" else "T"
            me, partner = actors(obs)
            cur, pp = point(me.get("position")), point(partner.get("position"))
            _, best_path = route(terrain(obs), cur, chosen.get("target"), {pp} if pp else ())
            best_len = len(best_path) if best_path is not None else 999
            for source in feature_cells(terrain(obs), symbol):
                alt = self.make_task("fetch_ingredient", source, item, chosen.get("priority", 5), tag="alternate")
                _, alt_path = route(terrain(obs), cur, source, {pp} if pp else ())
                # Never let a semantic tie-breaker select an unreachable or
                # materially longer source in a side-specialized kitchen.
                if alt_path is None or len(alt_path) > best_len + 1:
                    continue
                if self.legal_now(obs, alt) and not any(self.same_task(alt, x) for x in result):
                    result.append(alt)
        elif chosen.get("kind") == "fetch_dish":
            for source in feature_cells(terrain(obs), "D"):
                alt = self.make_task("fetch_dish", source, priority=chosen.get("priority", 3), tag="alternate")
                if self.legal_now(obs, alt) and not any(self.same_task(alt, x) for x in result):
                    result.append(alt)
        return result[:6]

    def compact(self, obs, options, reason=""):
        me, partner = actors(obs)
        reserved, intent = self.reservation(obs, horizon=7)
        facts = pot_facts(obs)
        return safe_json({
            "game": "Overcooked-AI symbolic realtime gridworld",
            "tick": tick(obs), "layout": obs.get("layout_name", ""),
            "self": {"position": point(me.get("position")), "facing": point(me.get("orientation")), "held": held(me)},
            "partner": {"position": point(partner.get("position")), "held": held(partner),
                        "last_action": obs.get("partner_action_last", ""),
                        "intent": {"kind": intent.get("kind"), "target": intent.get("target"),
                                   "urgent": intent.get("urgent"), "route": intent.get("path", [])[:8]},
                        "stationary_ticks": self.no_progress},
            "reservation": {"cells": sorted(reserved), "route_conflict": self.route_conflict(obs, self.task)},
            "pots": [{"position": f["pos"], "phase": f["phase"], "missing": f["missing"],
                      "remaining": f["remaining"]} for f in facts],
            "orders": recipes(obs),
            "current": {"kind": self.task.get("kind", "") if self.task else "",
                        "target": self.task.get("target") if self.task else None,
                        "age": max(0, tick(obs) - self.task_since) if self.task else 0,
                        "blocked": self.blocked},
            "reason": reason,
            "candidates": [{"id": x.get("id"), "kind": x.get("kind"), "priority": x.get("priority"),
                            "description": x.get("kind", "task")} for x in options],
        })

    # ---------- advisory calls --------------------------------------------

    def start_astra(self, obs, options):
        if self.astra_started or getattr(self.ctx, "system2", object()) is None:
            self.astra_started = True
            return
        self.astra_started = True
        prompt = (
            "You are a slow protocol reviewer for a realtime Overcooked-AI symbolic kitchen. "
            "The code executor owns recipes, BFS, legal interactions, and primitive actions. "
            "The novel protocol is route reservation: a waiting chef must move outside the "
            "partner's predicted route corridor, while urgent carried soup keeps priority. "
            "Return ONLY JSON with parking='route_aware'|'simple', handoff='predictive'|'basic', "
            "tempo='pipeline'|'conservative'. Never output directions or task ids.\n" +
            json.dumps(self.compact(obs, options, "initial"), separators=(",", ":"))
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
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            raw = future.result()
            text = raw if isinstance(raw, str) else getattr(raw, "text", "")
            match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
            value = json.loads(match.group(0)) if match else {}
            if not isinstance(value, dict):
                return
            if value.get("parking") in {"route_aware", "simple"}:
                # Simple advice never disables the safety invariant; it only
                # records Astra's disagreement for later traces.
                self.policy["parking"] = value["parking"]
            if value.get("handoff") in {"predictive", "basic"}:
                self.policy["handoff"] = value["handoff"]
            if value.get("tempo") in {"pipeline", "conservative"}:
                self.policy["tempo"] = value["tempo"]
            self.log("astra_route_policy", tick=tick(obs), policy=dict(self.policy))
        except Exception:
            return

    def start_jev(self, obs, options, reason):
        if self.system1_off or self.jev_future is not None or not options:
            return
        now = tick(obs)
        if now - self.last_jev < 18:
            return
        self.last_jev = now
        self.jev_epoch = self.epoch
        state_in = self.compact(obs, options, reason)
        criteria = {str(x.get("id")): str(x.get("kind", "candidate")) for x in options if x.get("id")}
        if not criteria:
            return
        reverse = list(criteria.items())
        reverse.reverse()
        questions = {
            "task": Choice(
                "Choose one listed code-generated transaction from `candidates`. Use the "
                "smallest priority number unless an equal-priority route is clearly safer. "
                "Never invent an id, direction, object, or arithmetic.", criteria),
            "yield": Noul(
                "Should the controlled chef yield for a few ticks before pursuing its current "
                "task? Say true only when `reservation.route_conflict` is true and the partner "
                "has a non-none urgent or ingredient intent. A carried soup delivery is not "
                "something to abandon; the executor will apply the safety gate.",
                {"true": "The partner's predicted corridor overlaps the current route and has priority.",
                 "false": "There is no verified corridor conflict or the partner is not urgent."}),
        }
        # A second ordering question is intentionally independent: it reduces
        # option-order bias, but its result is accepted only if it agrees.
        questions["task_reversed"] = Choice(
            "Choose the same kind of listed transaction as `candidates`, with no invented id. "
            "Prefer urgent delivery/start/fill over speculative work.", dict(reverse))

        def ask():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(state_in, questions, timeout_s=1.0)
            except Exception:
                return {}
        try:
            self.jev_future = self.ctx.spawn(ask)
            self.log("route_jev_request", tick=now, reason=reason, epoch=self.epoch)
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
        a = answers.get("task")
        b = answers.get("task_reversed")
        av, bv = getattr(a, "value", None), getattr(b, "value", None)
        pa = getattr(a, "probabilities", {}) or {}
        pb = getattr(b, "probabilities", {}) or {}
        try:
            ca = float(pa.get(av, 0.0)) if isinstance(pa, dict) else 0.0
            cb = float(pb.get(bv, 0.0)) if isinstance(pb, dict) else 0.0
        except Exception:
            ca = cb = 0.0
        if av in valid and av == bv and (ca >= 0.52 or (ca == 0.0 and cb == 0.0)):
            self.hint = av
            self.hint_tick = tick(obs)
            self.log("route_jev_choice", tick=tick(obs), value=av, confidence=ca)
        y = answers.get("yield")
        try:
            yp = float(getattr(y, "value", 0.0))
        except Exception:
            yp = 0.0
        if yp >= 0.78 and self.route_conflict(obs, self.task):
            self.yield_until = max(self.yield_until, tick(obs) + 5)
            self.log("route_jev_yield", tick=tick(obs), probability=yp)

    # ---------- motor -----------------------------------------------------

    def escape(self, obs, goal=None):
        me, partner = actors(obs)
        cur, pp = point(me.get("position")), point(partner.get("position"))
        if cur is None:
            return self.action(self.noop)
        reserved, _ = self.reservation(obs, horizon=4)
        predicted = partner_next(obs)
        choices = []
        for nxt, delta in neighbours(cur):
            if not open_cell(terrain(obs), nxt) or nxt in {pp, predicted}:
                continue
            goal_gap = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) if goal else 0
            route_gap = min((abs(nxt[0] - q[0]) + abs(nxt[1] - q[1]) for q in reserved), default=5)
            partner_gap = abs(nxt[0] - pp[0]) + abs(nxt[1] - pp[1]) if pp else 0
            choices.append((-route_gap, -partner_gap, goal_gap, nxt, delta))
        if not choices:
            return self.action(self.noop)
        choices.sort(key=lambda z: z[:4])
        return self.action(STEP_NAME.get(choices[0][4], self.noop))

    def parking_action(self, obs):
        me, partner = actors(obs)
        cur, pp = point(me.get("position")), point(partner.get("position"))
        reserved, _ = self.reservation(obs, horizon=9)
        if cur is None:
            return self.action(self.noop)
        # A current cell is safe only if it is not in the partner route tube.
        if cur != pp and cur not in reserved:
            return self.action(self.noop)
        target = self.free_parking(obs, cur)
        if target is None:
            return self.escape(obs)
        forbidden = set(reserved)
        forbidden.discard(cur)
        if pp is not None:
            forbidden.add(pp)
        path = bfs(terrain(obs), cur, [target], forbidden)
        if path is None or len(path) < 2:
            return self.escape(obs, target)
        nxt = path[1]
        delta = (nxt[0] - cur[0], nxt[1] - cur[1])
        return self.action(STEP_NAME.get(delta, self.noop))

    def can_interact(self, obs, task):
        kind, target = task.get("kind"), point(task.get("target"))
        me, _ = actors(obs)
        item = held(me)
        grid = terrain(obs)
        facts = pot_facts(obs)
        pot = next((f for f in facts if f["pos"] == target), None)
        if kind == "deliver":
            return item == "soup" and cell(grid, target) == "S"
        if kind == "fetch_dish":
            return not item and cell(grid, target) == "D"
        if kind == "fetch_ingredient":
            return not item and cell(grid, target) in {"O", "T"}
        if kind == "take_counter":
            return not item and any(n == task.get("item") and p == target for n, p in counter_objects(obs))
        if kind == "put_counter":
            return item in INGREDIENTS and target not in {obj_pos(o) for o in objects(obs)}
        if kind == "fill":
            return bool(pot and item in INGREDIENTS and pot["phase"] in {"empty", "partial", "full_idle"}
                        and item in pot["missing"])
        if kind == "start":
            return bool(pot and not item and pot["phase"] == "full_idle")
        if kind == "pickup":
            return bool(pot and item == "dish" and pot["ready"])
        return False

    def motor(self, obs):
        if not self.task:
            return self.action(self.noop)
        me, partner = actors(obs)
        cur, pp = point(me.get("position")), point(partner.get("position"))
        kind, target = self.task.get("kind"), point(self.task.get("target"))
        if kind in {"wait", "hold"}:
            return self.parking_action(obs)
        if cur is None or target is None:
            return self.action(self.noop)
        reserved, intent = self.reservation(obs, horizon=4)
        predicted = partner_next(obs)
        forbidden = {pp} if pp is not None and pp != target else set()
        pose, path = route(terrain(obs), cur, target, forbidden)
        if path is None or pose is None:
            self.blocked += 1
            return self.escape(obs, pose or target)
        if len(path) > 1:
            nxt = path[1]
            conflict = nxt == pp or nxt == predicted or (predicted == cur and nxt == pp)
            # The full route tube is a parking reservation, not a hard wall
            # for productive work.  Only the partner's actual/next cell is a
            # hard collision check; otherwise ring layouts can deadlock when
            # both chefs need the same long corridor.
            if tick(obs) < self.yield_until and kind in {"wait", "hold"}:
                conflict = True
            if conflict:
                self.blocked += 1
                if self.blocked <= 2:
                    return self.action(self.noop)
                return self.escape(obs, pose)
            self.blocked = max(0, self.blocked - 1)
            delta = (nxt[0] - cur[0], nxt[1] - cur[1])
            return self.action(STEP_NAME.get(delta, self.noop))
        desired = (target[0] - cur[0], target[1] - cur[1])
        facing = point(me.get("orientation"))
        if desired in STEP_NAME and facing != desired:
            return self.action(STEP_NAME[desired])
        if self.can_interact(obs, self.task):
            return self.action("interact")
        return self.action(self.noop)

    def task_limit(self, kind):
        return {"deliver": 48, "fetch_dish": 45, "fetch_ingredient": 38,
                "take_counter": 34, "put_counter": 20, "fill": 20,
                "start": 18, "pickup": 22, "wait": 16, "hold": 16}.get(kind, 30)

    def choose(self, obs, options):
        current = self.task
        if current and self.legal_now(obs, current):
            age = tick(obs) - self.task_since
            if age <= self.task_limit(current.get("kind", "")):
                # Dynamic waits have a moving floor target, but their logical
                # phase must still be refreshed.  A cooking wait therefore
                # changes to pickup/dish work on the very observation where a
                # pot becomes ready instead of waiting for its age limit.
                if current.get("dynamic"):
                    desired = options[0] if options else None
                    if desired is not None and self.same_task(current, desired):
                        return
                    current = None
                if current is not None:
                    if self.hint and tick(obs) - self.hint_tick <= 4:
                        for candidate in options:
                            if candidate.get("id") == self.hint:
                                self.task = candidate.copy()
                                self.hint = None
                                self.log("route_hint_apply", tick=tick(obs), kind=candidate.get("kind"))
                                return
                    return
        old = current
        selected = None
        if self.hint and tick(obs) - self.hint_tick <= 5:
            for candidate in options:
                if candidate.get("id") == self.hint and self.legal_now(obs, candidate):
                    selected = candidate.copy()
                    break
        if selected is None:
            selected = min(options, key=lambda x: (x.get("priority", 99), x.get("id", ""))) if options else self.make_task("wait", dynamic=True)
        self.task = selected
        self.task_since = tick(obs)
        self.epoch += 1
        self.blocked = 0
        self.hint = None
        self.log("reservation_task", tick=self.task_since, kind=selected.get("kind"),
                 target=selected.get("target"), epoch=self.epoch, priority=selected.get("priority"),
                 replaced=old.get("kind") if old else "")

    def run(self):
        while not self.ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self.last_obs = obs
            self.update_motion(obs)
            self.poll_astra(obs)
            options = self.candidate_tasks(obs)
            if not self.astra_started:
                self.start_astra(obs, options)
            self.poll_jev(obs, options)
            old_epoch = self.epoch
            if self.task and (not self.legal_now(obs, self.task) or
                              tick(obs) - self.task_since > self.task_limit(self.task.get("kind", "")) or
                              (self.blocked >= 9 and held(actors(obs)[0]) != "soup")):
                self.log("reservation_replan", tick=tick(obs), blocked=self.blocked,
                         old_kind=self.task.get("kind"))
                self.task = None
            self.choose(obs, options)
            if self.epoch != old_epoch:
                self.start_jev(obs, options, "task_boundary")
            elif self.blocked >= 4:
                self.start_jev(obs, options, "traffic_conflict")
            elif tick(obs) - self.last_jev >= 55:
                self.start_jev(obs, options, "heartbeat")
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
        RouteReservedPilot(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
