"""Candidate 0011: a committed production controller with a bounded gate release.

The controller is deliberately split into three layers.  The symbolic layer
computes recipes, pot phases, interaction poses and routes.  A small traffic
scheduler commits to one physical job until an observable transition completes;
it never asks a model for a primitive direction.  Astra supplies an occasional
role/mode hint and Jev adjudicates only phase/traffic exceptions in the
background, so inference cannot stop the 6 Hz motor.  The isolated change from
0006 is a short, code-guarded release when a dish-carrying partner persistently
contests a still-cooking pot gate.
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
VEC_TO_ACTION = {v: k for k, v in DIRECTIONS.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}


# ---------- defensive observation adapters ----------


def _pos(value, default=None):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x, y = value[0], value[1]
            if isinstance(x, bool) or isinstance(y, bool):
                return default
            return int(x), int(y)
        except Exception:
            pass
    return default


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _state(obs):
    return obs.get("state", {}) if isinstance(obs, dict) and isinstance(obs.get("state"), dict) else {}


def _terrain(obs):
    value = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return value if isinstance(value, list) else []


def _cell(terrain, point):
    point = _pos(point)
    if point is None:
        return "X"
    x, y = point
    if x < 0 or y < 0 or y >= len(terrain):
        return "X"
    try:
        row = terrain[y]
        if isinstance(row, (list, tuple, str)) and x < len(row):
            return row[x]
    except Exception:
        pass
    return "X"


def _features(terrain, symbol):
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (list, tuple, str)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                result.append((x, y))
    return result


def _walkable(terrain, point):
    # In the public representation all feature cells and X are surfaces, not
    # player cells.  Spawn digits are intentionally treated as floor.
    return _cell(terrain, point) not in FEATURES and _cell(terrain, point) != "X"


def _players(obs):
    value = _state(obs).get("players", [])
    return value if isinstance(value, list) else []


def _actors(obs):
    players = _players(obs)
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


def _held_object(player):
    value = player.get("held_object") if isinstance(player, dict) else None
    return value if isinstance(value, dict) else None


def _held_name(player):
    value = _held_object(player)
    if not value:
        return ""
    name = value.get("name", "")
    return str(name).lower() if name is not None else ""


def _objects(obs):
    value = _state(obs).get("objects", [])
    return value if isinstance(value, list) else []


def _object_name(obj):
    if not isinstance(obj, dict):
        return ""
    value = obj.get("name", "")
    return str(value).lower() if value is not None else ""


def _object_pos(obj):
    return _pos(obj.get("position")) if isinstance(obj, dict) else None


def _ingredient_list(obj):
    if not isinstance(obj, dict):
        return []
    value = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("name", "")
        if item is not None and str(item).lower() in INGREDIENTS:
            result.append(str(item).lower())
    return result


def _recipes(obs):
    raw = _state(obs).get("all_orders", [])
    if not isinstance(raw, list):
        raw = []
    recipes = []
    for item in raw:
        values = item.get("ingredients") if isinstance(item, dict) else item
        if isinstance(values, list):
            recipe = [str(v.get("name", "") if isinstance(v, dict) else v).lower()
                      for v in values]
            recipe = [v for v in recipe if v in INGREDIENTS]
            if recipe:
                recipes.append(recipe)
    if recipes:
        return recipes
    # This is only a defensive fallback for a malformed snapshot; normal
    # observations always expose all_orders.
    return [["onion", "onion", "onion"]]


def _missing_for(ingredients, recipes):
    have = Counter(ingredients)
    best = None
    best_missing = None
    for recipe in recipes:
        wanted = Counter(recipe)
        if any(have[k] > wanted[k] for k in have):
            continue
        current = Counter(ingredients)
        missing = []
        for item in recipe:
            if current[item] < wanted[item]:
                missing.append(item)
                current[item] += 1
        if best is None or len(missing) < len(best_missing):
            best, best_missing = recipe, missing
    return best or [], best_missing if best_missing is not None else []


def _counter_sites(obs):
    terrain = _terrain(obs)
    occupied = {_object_pos(o) for o in _objects(obs) if _object_pos(o) is not None}
    result = []
    # X is the public counter/wall symbol.  Only X cells with a floor neighbor
    # can be approached and therefore can be used for a handoff.
    for y, row in enumerate(terrain):
        if not isinstance(row, (list, tuple, str)):
            continue
        for x, value in enumerate(row):
            point = (x, y)
            if value != "X" or point in occupied:
                continue
            if any(_walkable(terrain, (x + dx, y + dy)) for dx, dy in DIRECTIONS.values()):
                result.append(point)
    return result


def _counter_items(obs):
    terrain = _terrain(obs)
    result = []
    for obj in _objects(obs):
        point = _object_pos(obj)
        name = _object_name(obj)
        if point is not None and name and _cell(terrain, point) == "X":
            result.append((name, point))
    return result


def _pot_infos(obs):
    state = _state(obs)
    objects = _objects(obs)
    recipes = _recipes(obs)
    soup_by_pos = {}
    for obj in objects:
        point = _object_pos(obj)
        if point is not None and _object_name(obj) == "soup":
            soup_by_pos[point] = obj
    infos = []
    for point in _features(_terrain(obs), "P"):
        soup = soup_by_pos.get(point)
        ingredients = _ingredient_list(soup)
        recipe, missing = _missing_for(ingredients, recipes)
        ready = bool(soup and soup.get("is_ready"))
        cooking = bool(soup and soup.get("is_cooking"))
        # Some adaptor snapshots expose only the cooking counter.  Do not call
        # a ready soup cooking merely because its counter is present.
        cook_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        cook_time = soup.get("cook_time", -1) if soup else -1
        if not ready and not cooking and isinstance(cook_tick, (int, float)):
            if cook_tick >= 0 and isinstance(cook_time, (int, float)) and cook_time >= 0:
                cooking = True
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
        infos.append({
            "pos": point,
            "ingredients": ingredients,
            "recipe": recipe,
            "missing": missing,
            "phase": phase,
            "ready": ready,
            "cooking": cooking,
            "full": full,
            "remaining": remaining,
        })
    return infos


# ---------- geometry ----------


def _neighbors(point):
    x, y = point
    for delta in DIRECTIONS.values():
        yield (x + delta[0], y + delta[1]), delta


def _bfs(terrain, start, goal, forbidden=()):
    start, goal = _pos(start), _pos(goal)
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


def _poses(terrain, target):
    target = _pos(target)
    result = []
    if target is None:
        return result
    for delta in DIRECTIONS.values():
        pose = (target[0] - delta[0], target[1] - delta[1])
        if _walkable(terrain, pose) and pose not in result:
            result.append(pose)
    return result


def _distance_path(terrain, start, target, partner=None):
    paths = []
    for pose in _poses(terrain, target):
        forbidden = {partner} if partner is not None and partner != pose else set()
        path = _bfs(terrain, start, pose, forbidden)
        if path is not None:
            paths.append((len(path), pose, path))
    if paths:
        paths.sort(key=lambda item: (item[0], item[1]))
        return paths[0][1], paths[0][2]
    # If the partner occupies the only approach, retain a deterministic queue
    # pose instead of declaring the task complete or emitting a random action.
    fallback = []
    for pose in _poses(terrain, target):
        path = _bfs(terrain, start, pose)
        if path is not None:
            fallback.append((len(path), pose, path))
    if fallback:
        fallback.sort(key=lambda item: (item[0], item[1]))
        return fallback[0][1], fallback[0][2]
    return None, None


def _tick(obs):
    if not isinstance(obs, dict):
        return 0
    value = obs.get("tick", _state(obs).get("timestep", 0))
    try:
        return int(value)
    except Exception:
        return 0


class CommittedKitchen:
    """A stateful job executor.  Jobs survive ordinary observations."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = game.action_spec
        self.names = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.names:
            self.noop = "stay" if "stay" in self.names else (next(iter(self.names), "stay"))

        self.job = None
        self.job_started_tick = -1
        self.job_epoch = 0
        self.last_snapshot = None
        self.last_pos = None
        self.last_action = self.noop
        self.last_progress_tick = 0
        self.blocked_ticks = 0
        self.last_traffic_pos = None
        self.traffic_target = None
        self.traffic_ticks = 0
        # A dish at a cooking pot normally reserves the interaction gate.  Keep
        # one bounded escape hatch for the specific case where the partner is
        # also carrying a dish and is repeatedly trying to use that gate.
        self.stage_gate_pressure = 0
        self.stage_release = None
        self.stage_release_done = False
        self.phase_hint = None
        self.phase_hint_until = -1
        self.phase_hint_epoch = -1
        self.yield_ticks = 0

        self.system1_disabled = getattr(ctx, "system1", object()) is None
        self.jev_future = None
        self.jev_request_epoch = -1
        self.last_jev_epoch = -1
        self.last_jev_tick = -999

        self.astra_future = None
        self.astra_done = False
        self.astra_mode = None
        self.astra_side = None

    # ---- model bridge -------------------------------------------------

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _compact(self, obs, tasks=None):
        me, partner = _actors(obs)
        pots = _pot_infos(obs)
        job = self.job or {}
        task_list = tasks if tasks is not None else []
        partner_pos = _pos(partner.get("position"))
        target = _pos(job.get("target"))
        return _jsonable({
            "game": "Overcooked-AI symbolic kitchen",
            "tick": _tick(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {
                "position": _pos(me.get("position")),
                "facing": _pos(me.get("orientation")),
                "held": _held_name(me),
            },
            "partner": {
                "position": partner_pos,
                "facing": _pos(partner.get("orientation")),
                "held": _held_name(partner),
                "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else "",
                "near_job_target": bool(target and partner_pos and
                                        abs(target[0] - partner_pos[0]) + abs(target[1] - partner_pos[1]) <= 2),
            },
            "pots": [{
                "position": p["pos"], "phase": p["phase"],
                "ingredients": p["ingredients"], "missing": p["missing"],
                "cook_ticks_remaining": p["remaining"],
            } for p in pots],
            "orders": _recipes(obs),
            "job": {"kind": job.get("kind", ""), "target": target,
                    "locked": bool(job.get("lock")),
                    "blocked_ticks": self.blocked_ticks},
            "traffic": {"yield_ticks": self.yield_ticks,
                        "mode_hint": self.phase_hint if self.phase_hint_until >= _tick(obs) else None,
                        "astra_mode": self.astra_mode},
            "legal_medium_options": [
                {"id": str(t.get("kind", "")), "description": str(t.get("description", ""))}
                for t in task_list[:8]
            ],
        })

    def _launch_astra(self, obs, tasks):
        if self.astra_done or self.astra_future is not None:
            return
        if getattr(self.ctx, "system2", object()) is None:
            self.astra_done = True
            return
        self.astra_done = True
        compact = self._compact(obs, tasks)
        prompt = (
            "You are the slow strategic allocator for a realtime symbolic "
            "Overcooked-AI kitchen. The code executor owns recipes, exact legality, "
            "shortest paths and primitive actions. Choose only a high-level mode "
            "for the controlled chef from: delivery, supply, handoff, or balanced. "
            "Protect ready soup and do not make both chefs contest a one-cell gate. "
            "Return ONLY JSON: {\"mode\":\"...\",\"side\":\"left|right|none\","
            "\"traffic\":\"claim|yield|none\"}. Never output directions.\n"
            + json.dumps(compact, separators=(",", ":"))
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

    def _poll_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = result if isinstance(result, str) else getattr(result, "text", "")
            if not isinstance(text, str):
                return
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return
            value = json.loads(match.group(0))
            if not isinstance(value, dict):
                return
            if value.get("mode") in {"delivery", "supply", "handoff", "balanced"}:
                self.astra_mode = value["mode"]
            if value.get("side") in {"left", "right", "none"}:
                self.astra_side = value["side"]
            if value.get("traffic") == "claim":
                self.phase_hint = "claim_gate"
                self.phase_hint_until = _tick(self.last_snapshot or {}) + 24
            elif value.get("traffic") == "yield":
                self.phase_hint = "yield"
                self.phase_hint_until = _tick(self.last_snapshot or {}) + 8
        except Exception:
            return

    def _launch_jev(self, obs, tasks, reason):
        if self.system1_disabled or self.jev_future is not None or not tasks:
            return
        now = _tick(obs)
        if self.last_jev_epoch == self.job_epoch and now - self.last_jev_tick < 12:
            return
        self.last_jev_epoch = self.job_epoch
        self.last_jev_tick = now
        self.jev_request_epoch = self.job_epoch
        state = self._compact(obs, tasks)
        criteria = {
            "continue": "Keep the code-selected physical job and do not change roles.",
            "delivery": "Protect a ready or nearly ready soup: keep or obtain a dish and claim its pot approach.",
            "supply": "Prioritize the next missing recipe ingredient or a legal pot start.",
            "handoff": "Use an available counter to stage the needed ingredient instead of contesting the partner's route.",
            "yield": "Give the partner a brief right of way because the partner has the urgent carried object or the only gate.",
        }
        questions = {
            "phase": Choice(
                "Classify the current medium-level phase using only `pots`, `self.held`, `partner.held`, and `traffic` in the state. This is advice to code; do not invent directions or objects.",
                criteria,
            ),
            "yield": Noul(
                "Should the controlled chef yield for at most two ticks? Say true only if `partner.held` is soup or dish, or if the partner is visibly occupying the only approach for the current job; otherwise false.",
                {"true": "The partner has the urgent carried object or is at the sole approach.",
                 "false": "The controlled chef should continue its current job."},
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
            self.jev_future = self.ctx.spawn(ask)
            self._log("jev_request", tick=now, reason=reason, epoch=self.job_epoch)
        except Exception:
            self.jev_future = None

    def _poll_jev(self, obs):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict) or self.jev_request_epoch != self.job_epoch:
            return
        phase_answer = answers.get("phase")
        value = getattr(phase_answer, "value", None)
        probabilities = getattr(phase_answer, "probabilities", {}) or {}
        try:
            probability = float(probabilities.get(value, 0.0)) if isinstance(probabilities, dict) else 0.0
        except Exception:
            probability = 0.0
        if value in {"delivery", "supply", "handoff", "continue", "yield"} and (probability >= 0.55 or probability == 0.0):
            self.phase_hint = value
            self.phase_hint_until = _tick(obs) + 14
            self.phase_hint_epoch = self.job_epoch
            self._log("jev_phase", tick=_tick(obs), value=value, probability=probability)
        yield_answer = answers.get("yield")
        yvalue = getattr(yield_answer, "value", 0.0)
        try:
            if isinstance(yvalue, (int, float)) and yvalue >= 0.72:
                me, partner = _actors(obs)
                if not _held_name(me) and _held_name(partner) in {"soup", "dish"}:
                    self.yield_ticks = max(self.yield_ticks, 2)
        except Exception:
            pass

    # ---- task construction -------------------------------------------

    def _task(self, kind, target=None, item="", description="", lock=False, wait=False):
        target = _pos(target)
        suffix = "_".join(str(v) for v in target) if target is not None else "none"
        ident = "%s_%s%s" % (kind, item, ("_" + suffix) if suffix else "")
        return {
            "id": ident,
            "kind": kind,
            "target": target,
            "item": item,
            "description": description or kind,
            "lock": bool(lock),
            "wait": bool(wait),
        }

    def _dispenser_for(self, ingredient):
        return "O" if ingredient == "onion" else "T"

    def _tasks(self, obs):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        held = _held_name(me)
        partner_held = _held_name(partner)
        pots = _pot_infos(obs)
        tasks = []

        if held == "soup":
            for target in _features(terrain, "S"):
                tasks.append(self._task("deliver", target, description="deliver held soup at serving window", lock=True))
            return tasks or [self._task("wait", wait=True, description="hold soup safely")]

        if held == "dish":
            ready = [p for p in pots if p["ready"]]
            cooking = [p for p in pots if p["cooking"]]
            for pot in ready:
                tasks.append(self._task("pickup_ready", pot["pos"], description="pick up ready soup with held dish", lock=True))
            for pot in cooking:
                tasks.append(self._task("stage_dish", pot["pos"], description="stage held dish at cooking pot", lock=True, wait=True))
            if tasks:
                return tasks
            return [self._task("wait", wait=True, description="hold dish safely")]

        if held in INGREDIENTS:
            compatible = [p for p in pots
                          if p["phase"] in {"empty", "partial", "full_idle"}
                          and held in p["missing"]]
            compatible.sort(key=lambda p: (len(p["missing"]), p["pos"]))
            for pot in compatible:
                tasks.append(self._task("fill_pot", pot["pos"], held,
                                        "put held %s into compatible pot" % held, lock=True))
            if tasks:
                return tasks
            for point in _counter_sites(obs):
                tasks.append(self._task("deposit_counter", point, held,
                                        "stage held %s on a free counter" % held, lock=True))
            return tasks or [self._task("wait", wait=True, description="hold ingredient safely")]

        # Empty hands: urgent work is entirely code-owned.  The model may only
        # bias the next non-urgent branch after this list has been built.
        ready = [p for p in pots if p["ready"]]
        if ready and partner_held not in {"dish", "soup"}:
            for target in _features(terrain, "D"):
                tasks.append(self._task("fetch_dish", target, description="fetch a dish for ready soup", lock=True))
            if tasks:
                return tasks

        full_idle = [p for p in pots if p["phase"] == "full_idle"]
        for pot in full_idle:
            tasks.append(self._task("start_cooking", pot["pos"], description="start full idle pot", lock=True))
        if tasks:
            return tasks

        # A partially built pot is the next production reservation.  Prefer a
        # staged ingredient, then a dispenser.  The recipe/order itself is
        # computed from the observation rather than assumed to be onion-only.
        buildable = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]]
        buildable.sort(key=lambda p: (len(p["missing"]), p["pos"]))
        if buildable:
            pot = buildable[0]
            ingredient = pot["missing"][0]
            for name, point in _counter_items(obs):
                if name == ingredient:
                    tasks.append(self._task("fetch_counter", point, ingredient,
                                            "pick up staged %s from counter" % ingredient, lock=True))
            for target in _features(terrain, self._dispenser_for(ingredient)):
                tasks.append(self._task("fetch_ingredient", target, ingredient,
                                        "fetch one %s for pot" % ingredient, lock=True))
            if tasks:
                return tasks

        if any(p["phase"] == "cooking" for p in pots):
            for target in _features(terrain, "D"):
                tasks.append(self._task("fetch_dish", target, description="fetch dish while pot cooks", lock=True))
            if tasks:
                return tasks

        if partner_held not in {"dish", "soup"}:
            for target in _features(terrain, "D"):
                tasks.append(self._task("fetch_dish", target, description="stage a future dish", lock=True))
        return tasks or [self._task("wait", wait=True, description="wait for a useful production event")]

    def _priority(self, task, obs):
        kind = task.get("kind")
        phase = self.phase_hint if self.phase_hint_until >= _tick(obs) and self.phase_hint_epoch in {-1, self.job_epoch} else None
        # These ranks are safety/production invariants and cannot be overridden.
        base = {
            "deliver": 0, "pickup_ready": 1, "start_cooking": 2,
            "fill_pot": 3, "fetch_counter": 4, "fetch_ingredient": 5,
            "deposit_counter": 6, "fetch_dish": 7, "stage_dish": 8,
            "wait": 20,
        }.get(kind, 10)
        if phase == "delivery" and kind in {"fetch_dish", "stage_dish", "pickup_ready"}:
            base -= 2
        elif phase == "supply" and kind in {"fill_pot", "fetch_counter", "fetch_ingredient", "start_cooking"}:
            base -= 1
        elif phase == "handoff" and kind in {"fetch_counter", "deposit_counter"}:
            base -= 1
        elif phase == "yield":
            base += 3
        # Astra's role is deliberately a weak tie-breaker.  It cannot steal a
        # ready soup from the hard priority ordering.
        if self.astra_mode == "delivery" and kind in {"fetch_dish", "stage_dish"}:
            base -= 1
        if self.astra_mode == "supply" and kind in {"fetch_ingredient", "fetch_counter", "fill_pot"}:
            base -= 1
        if self.astra_mode == "handoff" and kind in {"deposit_counter", "fetch_counter"}:
            base -= 1
        return base

    def _choose(self, obs, tasks):
        if not tasks:
            return self._task("wait", wait=True)
        me, partner = _actors(obs)
        me_pos = _pos(me.get("position"))
        partner_pos = _pos(partner.get("position"))
        terrain = _terrain(obs)
        ranked = []
        for task in tasks:
            pose, path = _distance_path(terrain, me_pos, task.get("target"), partner_pos)
            length = len(path) if path is not None else 999
            # Avoid picking the same dispenser the partner is already carrying
            # toward, but retain all alternatives when no route exists.
            target = task.get("target")
            partner_distance = abs(target[0] - partner_pos[0]) + abs(target[1] - partner_pos[1]) if target and partner_pos else 99
            contention = 1 if task.get("kind") in {"fetch_ingredient", "fetch_dish"} and partner_distance <= 1 else 0
            side_bias = 0
            if self.astra_side in {"left", "right"} and target is not None:
                width = 0
                if terrain and isinstance(terrain[0], (list, tuple, str)):
                    width = len(terrain[0])
                side_bias = 0 if ((target[0] < width / 2) == (self.astra_side == "left")) else 1
            ranked.append((self._priority(task, obs), contention, side_bias, length, task.get("id", ""), task, pose))
        ranked.sort(key=lambda row: row[:-2])
        chosen = ranked[0][5].copy()
        chosen["pose"] = ranked[0][6]
        return chosen

    def _same_target_kind(self, a, b):
        if not a or not b:
            return False
        return (a.get("kind") == b.get("kind") and
                _pos(a.get("target")) == _pos(b.get("target")) and
                a.get("item", "") == b.get("item", ""))

    def _job_still_possible(self, obs, tasks):
        if not self.job:
            return False
        me, _ = _actors(obs)
        held = _held_name(me)
        kind = self.job.get("kind")
        target = _pos(self.job.get("target"))
        # Preserve the physical approach across the cooking -> ready boundary.
        if kind == "stage_dish" and held == "dish":
            for task in tasks:
                if task.get("kind") in {"stage_dish", "pickup_ready"} and _pos(task.get("target")) == target:
                    self.job.update({k: task[k] for k in ("kind", "item", "description", "lock", "wait") if k in task})
                    return True
        for task in tasks:
            if self._same_target_kind(self.job, task):
                return True
        # A job can complete between sparse observations.  The held-object
        # transitions below are safer than trusting a model's stale answer.
        if kind in {"fetch_ingredient", "fetch_counter"} and held in INGREDIENTS:
            return False
        if kind == "fetch_dish" and held == "dish":
            return False
        if kind == "pickup_ready" and held == "soup":
            return False
        if kind == "deliver" and held != "soup":
            return False
        if kind in {"fill_pot", "start_cooking", "deposit_counter"} and not held:
            return False
        return False

    def _new_job(self, obs, tasks):
        # A Jev phase answer is about the next event boundary as well as the
        # old job.  Carry it across exactly one physical transition, but give
        # it the new epoch so it cannot leak into a later unrelated cycle.
        old_hint = self.phase_hint
        old_hint_until = self.phase_hint_until
        now = _tick(obs)
        chosen = self._choose(obs, tasks)
        me, partner = _actors(obs)
        start = _pos(me.get("position"))
        partner_pos = _pos(partner.get("position"))
        pose = chosen.get("pose")
        path = None
        if pose is not None:
            path = _bfs(_terrain(obs), start, pose,
                        {partner_pos} if partner_pos is not None and partner_pos != pose else set())
            if path is None:
                path = _bfs(_terrain(obs), start, pose)
        chosen["pose"] = pose
        chosen["initial_path_len"] = len(path) if path is not None else 999
        chosen["lock"] = bool(chosen.get("lock"))
        self.job = chosen
        self.job_started_tick = now
        self.job_epoch += 1
        self.blocked_ticks = 0
        self.last_traffic_pos = None
        self.traffic_target = None
        self.traffic_ticks = 0
        self.stage_gate_pressure = 0
        self.stage_release = None
        self.stage_release_done = False
        if old_hint and old_hint_until >= now:
            self.phase_hint = old_hint
            self.phase_hint_until = old_hint_until
            self.phase_hint_epoch = self.job_epoch
        else:
            self.phase_hint = None
            self.phase_hint_until = -1
            self.phase_hint_epoch = -1
        self._log("job_commit", tick=now, kind=chosen.get("kind"), target=chosen.get("target"), pose=pose)

    def _refresh_job(self, obs, tasks):
        if self.job and self._job_still_possible(obs, tasks):
            return
        self.job = None
        self._new_job(obs, tasks)

    # ---- traffic and motor -------------------------------------------

    def _safe_action(self, action):
        if action in self.names:
            return action
        if self.noop in self.names:
            return self.noop
        return next(iter(self.names), "stay")

    def _partner_prediction(self, obs):
        _, partner = _actors(obs)
        point = _pos(partner.get("position"))
        action = obs.get("partner_action_last", "") if isinstance(obs, dict) else ""
        delta = DIRECTIONS.get(action, (0, 0))
        return ((point[0] + delta[0], point[1] + delta[1]) if point else None)

    def _traffic_step(self, obs, goal):
        me, partner = _actors(obs)
        current = _pos(me.get("position"))
        other = _pos(partner.get("position"))
        terrain = _terrain(obs)
        if current is None:
            return self._safe_action(self.noop)
        candidates = []
        previous = self.last_traffic_pos
        for nxt, delta in _neighbors(current):
            if not _walkable(terrain, nxt) or nxt == other:
                continue
            if previous is not None and nxt == previous and self.traffic_ticks < 4:
                continue
            distance_to_partner = abs(nxt[0] - other[0]) + abs(nxt[1] - other[1]) if other else 0
            distance_to_goal = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) if goal else 0
            candidates.append((-
                distance_to_partner, distance_to_goal, nxt, VEC_TO_ACTION.get(delta, self.noop)))
        if not candidates:
            return self._safe_action(self.noop)
        candidates.sort()
        _, _, chosen, action = candidates[0]
        self.last_traffic_pos = current
        self.traffic_target = goal
        self.traffic_ticks += 1
        return self._safe_action(action)

    def _stage_gate_pressure_update(self, obs, pose):
        """Count only persistent partner-dish contention at our stage pose."""
        _, partner = _actors(obs)
        partner_pos = _pos(partner.get("position"))
        if _held_name(partner) != "dish" or partner_pos is None or pose is None:
            self.stage_gate_pressure = 0
            return
        distance = abs(partner_pos[0] - pose[0]) + abs(partner_pos[1] - pose[1])
        partner_next = self._partner_prediction(obs)
        if distance <= 1 or partner_next == pose:
            self.stage_gate_pressure += 1
        else:
            self.stage_gate_pressure = 0

    def _stage_parking_pose(self, obs, current, pose, target, partner_pos):
        """Find a nearby floor cell that frees the gate and is returnable."""
        terrain = _terrain(obs)
        if current is None or pose is None:
            return None
        candidates = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (list, tuple, str)):
                continue
            for x in range(len(row)):
                point = (x, y)
                if point == pose or point == target or point == partner_pos:
                    continue
                if not _walkable(terrain, point):
                    continue
                near_pose = abs(point[0] - pose[0]) + abs(point[1] - pose[1])
                if near_pose < 1 or near_pose > 3:
                    continue
                out = _bfs(terrain, current, point,
                           {partner_pos} if partner_pos is not None else set())
                back = _bfs(terrain, point, pose,
                            {partner_pos} if partner_pos is not None else set())
                if out is None or back is None:
                    continue
                away = (abs(point[0] - partner_pos[0]) +
                        abs(point[1] - partner_pos[1])) if partner_pos is not None else 0
                candidates.append((-away, len(out) + len(back), len(out), point))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][3]

    def _stage_release_action(self, obs):
        """Execute or finish a short gate release; None returns to normal motor."""
        release = self.stage_release
        if not isinstance(release, dict):
            return None
        me, partner = _actors(obs)
        current = _pos(me.get("position"))
        other = _pos(partner.get("position"))
        pose = _pos(release.get("pose"))
        park = _pos(release.get("park"))
        if current is None or pose is None or park is None:
            self.stage_release = None
            return None
        target = _pos(self.job.get("target")) if self.job else None
        pot = next((p for p in _pot_infos(obs) if p["pos"] == target), None)
        remaining = pot.get("remaining") if isinstance(pot, dict) else None
        if not pot or not pot.get("cooking"):
            release["phase"] = "return"
        if release.get("phase") == "out":
            if current == park:
                release["phase"] = "wait"
            else:
                path = _bfs(_terrain(obs), current, park,
                            {other} if other is not None else set())
                if path is None or len(path) <= 1:
                    release["phase"] = "return"
                else:
                    nxt = path[1]
                    if nxt == other or nxt == self._partner_prediction(obs):
                        return self._safe_action(self.noop)
                    delta = (nxt[0] - current[0], nxt[1] - current[1])
                    return self._safe_action(VEC_TO_ACTION.get(delta, self.noop))
        if release.get("phase") == "wait":
            now = _tick(obs)
            if (now >= int(release.get("return_at", now)) or
                    (isinstance(remaining, (int, float)) and remaining <= 5)):
                release["phase"] = "return"
            else:
                return self._safe_action(self.noop)
        if release.get("phase") == "return" and current == pose:
            self.stage_release = None
        return None

    def _interaction_ready(self, obs, job):
        me, _ = _actors(obs)
        held = _held_name(me)
        kind = job.get("kind")
        target = _pos(job.get("target"))
        pots = _pot_infos(obs)
        pot = next((p for p in pots if p["pos"] == target), None)
        if kind == "deliver":
            return held == "soup"
        if kind == "fetch_ingredient":
            return not held and _cell(_terrain(obs), target) in {"O", "T"}
        if kind == "fetch_dish":
            return not held and _cell(_terrain(obs), target) == "D"
        if kind == "fetch_counter":
            return not held and any(name in INGREDIENTS and point == target for name, point in _counter_items(obs))
        if kind == "deposit_counter":
            return held in INGREDIENTS and target not in {_object_pos(o) for o in _objects(obs)}
        if kind == "fill_pot":
            return bool(pot and held in INGREDIENTS and not pot["ready"] and not pot["cooking"] and held in pot["missing"])
        if kind == "start_cooking":
            return bool(pot and not held and pot["phase"] == "full_idle")
        if kind == "pickup_ready":
            return bool(pot and held == "dish" and pot["ready"])
        return False

    def _motor(self, obs):
        me, partner = _actors(obs)
        current = _pos(me.get("position"))
        facing = _pos(me.get("orientation"))
        other = _pos(partner.get("position"))
        if not self.job:
            return self._safe_action(self.noop)
        job = self.job
        if job.get("wait") and job.get("target") is None:
            return self._safe_action(self.noop)
        target = _pos(job.get("target"))
        pose = _pos(job.get("pose"))
        if current is None or target is None or pose is None:
            return self._safe_action(self.noop)

        # A locked approach is normally a reservation, not a continuously
        # replanned waypoint.  The one exception is persistent, observable
        # contention: if both chefs have dishes and the pot is still cooking,
        # briefly free the gate so the partner can pass, then reclaim it before
        # the soup is ready.  This is bounded and never applies to a ready pot.
        if current == pose:
            desired_delta = (target[0] - current[0], target[1] - current[1])
            desired = VEC_TO_ACTION.get(desired_delta, self.noop)
            if facing != desired_delta and desired_delta in VEC_TO_ACTION:
                return self._safe_action(desired)
            if job.get("kind") == "stage_dish" and not self._interaction_ready(obs, job):
                pots = _pot_infos(obs)
                pot = next((p for p in pots if p["pos"] == target), None)
                self._stage_gate_pressure_update(obs, pose)
                partner_held = _held_name(partner)
                remaining = pot.get("remaining") if isinstance(pot, dict) else None
                if (pot and pot.get("cooking") and partner_held == "dish" and
                        not self.stage_release_done and self.blocked_ticks >= 3 and
                        self.stage_gate_pressure >= 2 and
                        isinstance(remaining, (int, float)) and remaining >= 8):
                    park = self._stage_parking_pose(
                        obs, current, pose, target, other)
                    if park is not None:
                        hold = max(3, min(6, int(remaining) - 3))
                        self.stage_release = {
                            "phase": "out", "park": park, "pose": pose,
                            "return_at": _tick(obs) + hold,
                        }
                        self.stage_release_done = True
                        self._log("stage_gate_release", tick=_tick(obs),
                                  target=target, pose=pose, park=park,
                                  return_at=_tick(obs) + hold,
                                  remaining=remaining)
                        release_action = self._stage_release_action(obs)
                        if release_action is not None:
                            return release_action
                release_action = self._stage_release_action(obs)
                if release_action is not None:
                    return release_action
                return self._safe_action(self.noop)
            if self._interaction_ready(obs, job):
                return self._safe_action("interact")
            # The precondition changed while the job was committed.  Holding a
            # critical pose is safer than wandering into the partner.
            if job.get("lock"):
                return self._safe_action(self.noop)

        partner_next = self._partner_prediction(obs)
        path = _bfs(_terrain(obs), current, pose,
                    {other} if other is not None and other != pose else set())
        if path is None:
            self.blocked_ticks += 1
            return self._traffic_step(obs, pose)
        if len(path) <= 1:
            self.blocked_ticks = 0
            return self._safe_action(self.noop)

        nxt = path[1]
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        action = VEC_TO_ACTION.get(delta, self.noop)
        # If the delivery/start job is claiming the exact gate and the partner
        # is predicted to enter it, contest the cell rather than yield it.  If
        # both move, the MDP keeps both old positions; the reservation survives.
        if nxt == partner_next and nxt == pose and job.get("lock"):
            self.blocked_ticks += 1
            return self._safe_action(action)
        # Do not enter a cell the partner is predicted to enter either.  This
        # catches same-cell moves through a corridor as well as direct swaps.
        if nxt == partner_next:
            self.blocked_ticks += 1
            if job.get("lock") and self.blocked_ticks < 3:
                return self._safe_action(self.noop)
            return self._traffic_step(obs, pose)
        # Never walk into the partner or swap cells.  For low-priority work a
        # deterministic side step avoids the north/south oscillation seen in
        # the previous candidate; a locked job waits if there is no side step.
        if nxt == other or (partner_next == current and nxt == other):
            self.blocked_ticks += 1
            if job.get("lock") and self.blocked_ticks < 3:
                return self._safe_action(self.noop)
            return self._traffic_step(obs, pose)
        self.blocked_ticks = 0
        return self._safe_action(action)

    # ---- episode loop -------------------------------------------------

    def run(self):
        ctx = self.ctx
        while not ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self.last_snapshot = obs
            self._poll_astra()
            self._poll_jev(obs)
            tasks = self._tasks(obs)
            self._launch_astra(obs, tasks)

            # Job commitment is invalidated only by observable state/task
            # transitions, not by every movement tick.
            self._refresh_job(obs, tasks)
            if self.job is not None and self.blocked_ticks >= 3:
                self._launch_jev(obs, tasks, "traffic_or_progress")
            elif self.job is not None and self.job_started_tick == _tick(obs):
                self._launch_jev(obs, tasks, "new_job")
            elif self.job is None:
                self._launch_jev(obs, tasks, "no_job")

            if self.yield_ticks > 0:
                me, _ = _actors(obs)
                if not _held_name(me):
                    self.yield_ticks -= 1
                    action = self._safe_action(self.noop)
                else:
                    self.yield_ticks = 0
                    action = self._motor(obs)
            else:
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
                        ctx.act(self._safe_action(self.noop))
                except Exception:
                    return
            self.last_action = action
            self.last_pos = _pos(_actors(obs)[0].get("position"))
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                return


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        if not getattr(game, "tick_s", None):
            return
        CommittedKitchen(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
