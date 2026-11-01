"""Candidate 0012: a just-in-time dish controller with soft ingredient lanes.

The controller treats a dish as a timed resource rather than a reservation of
an interaction square.  While a pot cooks, a chef prepares the next legal
piece (including a counter handoff when one is available), and only fetches a
dish when the shortest safe route can reach the pot near readiness.  A dish
already in hand waits in a non-gate parking square.  This is deliberately a
real-time motor: all model calls are advisory background work and every tick
has a deterministic action.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


DIRS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
VEC_ACTION = {v: k for k, v in DIRS.items()}
INGREDIENTS = {"onion", "tomato"}
FEATURES = {"O", "T", "D", "P", "S"}


def _pos(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            if isinstance(value[0], bool) or isinstance(value[1], bool):
                return None
            return int(value[0]), int(value[1])
        except Exception:
            return None
    return None


def _safe(value):
    """Make a strict JSON value, including for hostile stand-in observations."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
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


def _tick(obs):
    try:
        return int(obs.get("tick", _state(obs).get("timestep", 0)))
    except Exception:
        return 0


def _cell(terrain, point):
    point = _pos(point)
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


def _features(terrain, symbol):
    out = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (list, tuple, str)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                out.append((x, y))
    return out


def _walkable(terrain, point):
    value = _cell(terrain, point)
    return value not in FEATURES and value != "X"


def _players(obs):
    value = _state(obs).get("players", [])
    return value if isinstance(value, list) else []


def _actors(obs):
    players = _players(obs)
    try:
        index = int(obs.get("controlled_agent_index", 0))
    except Exception:
        index = 0
    if index not in (0, 1) or index >= len(players):
        index = 0
    other = 1 - index if len(players) > 1 else index
    me = players[index] if index < len(players) and isinstance(players[index], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, partner


def _held(player):
    obj = player.get("held_object") if isinstance(player, dict) else None
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("name", "") or "").lower()


def _objects(obs):
    value = _state(obs).get("objects", [])
    return value if isinstance(value, list) else []


def _obj_name(obj):
    return str(obj.get("name", "") or "").lower() if isinstance(obj, dict) else ""


def _obj_pos(obj):
    return _pos(obj.get("position")) if isinstance(obj, dict) else None


def _ingredients(obj):
    if not isinstance(obj, dict):
        return []
    values = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name", "")
        value = str(value or "").lower()
        if value in INGREDIENTS:
            out.append(value)
    return out


def _recipes(obs):
    raw = _state(obs).get("all_orders", [])
    out = []
    if isinstance(raw, list):
        for order in raw:
            values = order.get("ingredients") if isinstance(order, dict) else order
            if isinstance(values, list):
                recipe = []
                for value in values:
                    if isinstance(value, dict):
                        value = value.get("name", "")
                    value = str(value or "").lower()
                    if value in INGREDIENTS:
                        recipe.append(value)
                if recipe:
                    out.append(recipe)
    return out or [["onion", "onion", "onion"]]


def _missing(ingredients, recipes):
    current = Counter(ingredients)
    chosen = None
    missing = None
    for recipe in recipes:
        wanted = Counter(recipe)
        if any(current[name] > wanted[name] for name in current):
            continue
        left = Counter(ingredients)
        need = []
        for name in recipe:
            if left[name] < wanted[name]:
                need.append(name)
                left[name] += 1
        if missing is None or len(need) < len(missing):
            chosen, missing = recipe, need
    return chosen or [], missing if missing is not None else []


def _pots(obs):
    objects = _objects(obs)
    soup_at = {}
    for obj in objects:
        point = _obj_pos(obj)
        if point is not None and _obj_name(obj) == "soup":
            soup_at[point] = obj
    recipes = _recipes(obs)
    result = []
    for point in _features(_terrain(obs), "P"):
        soup = soup_at.get(point)
        ing = _ingredients(soup)
        recipe, missing = _missing(ing, recipes)
        ready = bool(soup and soup.get("is_ready"))
        cooking = bool(soup and soup.get("is_cooking")) and not ready
        raw_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        raw_time = soup.get("cook_time", -1) if soup else -1
        if not ready and isinstance(raw_tick, (int, float)) and isinstance(raw_time, (int, float)):
            cooking = cooking or (raw_tick >= 0 and raw_time >= 0)
        full = bool(soup and ing and not missing)
        if ready:
            phase = "ready"
        elif cooking:
            phase = "cooking"
        elif full:
            phase = "full_idle"
        elif ing:
            phase = "partial"
        else:
            phase = "empty"
        remaining = None
        if cooking and isinstance(raw_tick, (int, float)) and isinstance(raw_time, (int, float)):
            remaining = max(0, int(raw_time - raw_tick))
        result.append({
            "pos": point,
            "ingredients": ing,
            "missing": missing,
            "recipe": recipe,
            "phase": phase,
            "ready": ready,
            "cooking": cooking,
            "full": full,
            "remaining": remaining,
        })
    return result


def _neighbors(point):
    x, y = point
    for delta in DIRS.values():
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
                return list(reversed(path))
            queue.append(nxt)
    return None


def _poses(terrain, target):
    target = _pos(target)
    if target is None:
        return []
    out = []
    for delta in DIRS.values():
        pose = (target[0] - delta[0], target[1] - delta[1])
        if _walkable(terrain, pose) and pose not in out:
            out.append(pose)
    return out


def _best_pose(terrain, start, target, partner=None):
    choices = []
    for pose in _poses(terrain, target):
        blocked = {partner} if partner is not None and partner != pose else set()
        path = _bfs(terrain, start, pose, blocked)
        if path is not None:
            choices.append((len(path), pose, path))
    if choices:
        choices.sort(key=lambda item: (item[0], item[1]))
        return choices[0][1], choices[0][2]
    # A partner can occupy a unique interaction approach.  Keep the task alive
    # and let the traffic motor queue rather than abandoning the job.
    for pose in _poses(terrain, target):
        path = _bfs(terrain, start, pose)
        if path is not None:
            choices.append((len(path), pose, path))
    if choices:
        choices.sort(key=lambda item: (item[0], item[1]))
        return choices[0][1], choices[0][2]
    return None, None


def _distance(terrain, start, target, partner=None):
    _, path = _best_pose(terrain, start, target, partner)
    return len(path) if path is not None else 999


def _counter_sites(obs):
    terrain = _terrain(obs)
    occupied = {_obj_pos(obj) for obj in _objects(obs) if _obj_pos(obj) is not None}
    out = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (list, tuple, str)):
            continue
        for x, value in enumerate(row):
            point = (x, y)
            if value != "X" or point in occupied:
                continue
            if any(_walkable(terrain, q) for q, _ in _neighbors(point)):
                out.append(point)
    return out


def _counter_items(obs):
    terrain = _terrain(obs)
    out = []
    for obj in _objects(obs):
        point = _obj_pos(obj)
        name = _obj_name(obj)
        if point is not None and name in INGREDIENTS and _cell(terrain, point) == "X":
            out.append((name, point))
    return out


class TimedKitchen:
    """Code owns legality; the active job is a timed macro, never a model action."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = game.action_spec
        self.names = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.names:
            self.noop = "stay" if "stay" in self.names else (next(iter(self.names), "stay"))

        self.lane = None
        self.alt_lane_until = -1
        self.job = None
        self.epoch = 0
        self.job_tick = -1
        self.blocked = 0
        self.last_position = None
        self.previous_cell = None
        self.last_action = self.noop
        self.last_jev_tick = -999
        self.jev_future = None
        self.jev_epoch = -1
        self.timing_hint = None
        self.timing_until = -1
        self.yield_ticks = 0
        self.astra_future = None
        self.astra_started = False
        self.astra_tempo = None
        self.last_obs = None

        self.no_system1 = getattr(ctx, "system1", object()) is None

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _init_lane(self, obs):
        if self.lane in {"left", "right"}:
            return
        me, partner = _actors(obs)
        mp, pp = _pos(me.get("position")), _pos(partner.get("position"))
        if mp is not None and pp is not None and mp[0] != pp[0]:
            self.lane = "left" if mp[0] < pp[0] else "right"
        else:
            sources = _features(_terrain(obs), "O") + _features(_terrain(obs), "T")
            if mp is not None and sources:
                left = min(abs(p[0] - mp[0]) for p in sources if p[0] <= mp[0]) if any(p[0] <= mp[0] for p in sources) else 999
                right = min(abs(p[0] - mp[0]) for p in sources if p[0] > mp[0]) if any(p[0] > mp[0] for p in sources) else 999
                self.lane = "left" if left <= right else "right"
            else:
                self.lane = "left"
        self._log("lane_assign", tick=_tick(obs), lane=self.lane)

    def _width(self, obs):
        terrain = _terrain(obs)
        for row in terrain:
            if isinstance(row, (list, tuple, str)):
                return len(row)
        return 0

    def _lane_penalty(self, obs, point, lane=None):
        point = _pos(point)
        if point is None:
            return 0
        lane = lane or self.lane
        if lane not in {"left", "right"}:
            return 0
        midpoint = self._width(obs) / 2.0
        is_left = point[0] < midpoint
        return 0 if (lane == "left") == is_left else 1

    def _partner_next(self, obs):
        _, partner = _actors(obs)
        point = _pos(partner.get("position"))
        action = obs.get("partner_action_last", "") if isinstance(obs, dict) else ""
        delta = DIRS.get(action, (0, 0))
        return (point[0] + delta[0], point[1] + delta[1]) if point is not None else None

    def _summary(self, obs, candidates):
        me, partner = _actors(obs)
        job = self.job or {}
        return _safe({
            "tick": _tick(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {"position": _pos(me.get("position")), "held": _held(me),
                     "facing": _pos(me.get("orientation"))},
            "partner": {"position": _pos(partner.get("position")), "held": _held(partner),
                        "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else ""},
            "lane": {"assigned": self.lane, "alternate_until": self.alt_lane_until},
            "pots": [{"position": p["pos"], "phase": p["phase"],
                      "remaining": p["remaining"], "missing": p["missing"]} for p in _pots(obs)],
            "job": {"kind": job.get("kind", ""), "target": job.get("target"),
                    "waiting": bool(job.get("wait_ready")), "blocked": self.blocked},
            "timing": {"hint": self.timing_hint if self.timing_until >= _tick(obs) else None,
                       "candidates": [c.get("kind", "") for c in candidates[:6]]},
        })

    def _start_astra(self, obs, candidates):
        if self.astra_started:
            return
        self.astra_started = True
        if getattr(self.ctx, "system2", object()) is None:
            return
        state = self._summary(obs, candidates)
        prompt = (
            "Act as a slow strategy reviewer for a realtime symbolic Overcooked-AI "
            "kitchen. The executor already owns exact recipes, legality, BFS and "
            "primitive actions. Decide only whether this layout benefits from "
            "just-in-time dish fetching or continuous staging, and whether the "
            "controlled chef should keep its derived left/right ingredient lane. "
            "Return ONLY JSON: {\"tempo\":\"just_in_time|continuous\","
            "\"lane\":\"keep|alternate\"}. Never give directions.\n" +
            json.dumps(state, separators=(",", ":"))
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
            if parsed.get("tempo") in {"just_in_time", "continuous"}:
                self.astra_tempo = parsed["tempo"]
            if parsed.get("lane") == "alternate" and self.blocked >= 3:
                self.alt_lane_until = _tick(obs) + 80
                self._log("astra_lane_advice", tick=_tick(obs), advice="alternate")
        except Exception:
            return

    def _launch_jev(self, obs, candidates, reason):
        if self.no_system1 or self.jev_future is not None or not candidates:
            return
        now = _tick(obs)
        if now - self.last_jev_tick < 18:
            return
        self.last_jev_tick = now
        self.jev_epoch = self.epoch
        state = self._summary(obs, candidates)
        questions = {
            "timing": Choice(
                "Choose the safest timing mode from the code-generated options. "
                "Use `pots` and `timing.candidates`; do not invent a task or count.",
                {
                    "prepare": "A legal ingredient or counter preparation is available and no dish pickup is urgent.",
                    "fetch_dish": "A ready or nearly-ready pot needs a dish soon; favor fetching it over speculative prep.",
                    "wait": "No useful safe transition is available; preserve the current object and position.",
                },
            ),
            "traffic": Noul(
                "Should the controlled chef yield briefly to the partner? Say true only when the partner's held object is soup or dish and the partner is at or approaching the current job's route; otherwise say false.",
                {
                    "true": "The partner visibly has the urgent carried object and the route is shared.",
                    "false": "The partner is empty-handed or is not on this route.",
                },
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
            self._log("jev_request", tick=now, reason=reason, epoch=self.epoch)
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
        if not isinstance(answers, dict) or self.jev_epoch != self.epoch:
            return
        timing = answers.get("timing")
        value = getattr(timing, "value", None)
        probs = getattr(timing, "probabilities", {}) or {}
        try:
            p = float(probs.get(value, 0.0)) if isinstance(probs, dict) else 0.0
        except Exception:
            p = 0.0
        if value in {"prepare", "fetch_dish", "wait"} and (p >= 0.60 or p == 0.0):
            self.timing_hint = value
            self.timing_until = _tick(obs) + 16
            self._log("jev_timing", tick=_tick(obs), value=value, probability=p)
        traffic = answers.get("traffic")
        tvalue = getattr(traffic, "value", None)
        try:
            # Noul.value is the probability of the true proposition, not a
            # string label.  Require a consequence-appropriate confidence.
            if float(tvalue) >= 0.72 and _held(_actors(obs)[1]) in {"dish", "soup"}:
                self.yield_ticks = max(self.yield_ticks, 2)
        except Exception:
            pass

    def _task(self, kind, target=None, item="", priority=10, wait_ready=False, description=""):
        target = _pos(target)
        suffix = "_".join(str(v) for v in target) if target is not None else "none"
        return {
            "id": "%s_%s_%s" % (kind, item or "none", suffix),
            "kind": kind,
            "target": target,
            "item": item,
            "priority": int(priority),
            "wait_ready": bool(wait_ready),
            "description": description or kind,
        }

    def _pot_choice(self, obs, pots, current, partner):
        terrain = _terrain(obs)
        ranked = []
        for pot in pots:
            length = _distance(terrain, current, pot["pos"], partner)
            ranked.append((length, self._lane_penalty(obs, pot["pos"]), pot["pos"], pot))
        ranked.sort(key=lambda x: x[:3])
        return ranked[0][3] if ranked else None

    def _dish_due(self, obs, pot, current, partner):
        if not pot or not pot.get("cooking"):
            return False
        remaining = pot.get("remaining")
        if not isinstance(remaining, (int, float)):
            return True
        terrain = _terrain(obs)
        dishes = _features(terrain, "D")
        poses = _poses(terrain, pot["pos"])
        if not dishes or not poses:
            return True
        estimate = 999
        for dish in dishes:
            dish_pose, first = _best_pose(terrain, current, dish, partner)
            if dish_pose is None or first is None:
                continue
            for pot_pose in poses:
                # After interacting, the chef remains on the dispenser's
                # interaction pose, not on the non-walkable feature cell.
                second = _bfs(terrain, dish_pose, pot_pose)
                if second is not None:
                    estimate = min(estimate, len(first) + 1 + len(second) + 1)
        return remaining <= max(5, estimate + 2)

    def _parking_pose(self, obs, pot, current, partner):
        terrain = _terrain(obs)
        gate_poses = set(_poses(terrain, pot))
        choices = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (list, tuple, str)):
                continue
            for x in range(len(row)):
                point = (x, y)
                if point in gate_poses or point == partner or not _walkable(terrain, point):
                    continue
                path = _bfs(terrain, current, point, {partner} if partner is not None else set())
                if path is None:
                    continue
                back = any(_bfs(terrain, point, gate, {partner} if partner is not None else set()) is not None
                           for gate in gate_poses)
                if not back:
                    continue
                # Prefer the assigned side, then a short route and a cell close
                # to the pot.  It is intentionally not an interaction pose.
                score = (self._lane_penalty(obs, point), len(path),
                         abs(point[0] - pot[0]) + abs(point[1] - pot[1]), point)
                choices.append((score, point))
        if choices:
            choices.sort(key=lambda x: x[0])
            return choices[0][1]
        return None

    def _candidates(self, obs):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        current, partner_pos = _pos(me.get("position")), _pos(partner.get("position"))
        held, partner_held = _held(me), _held(partner)
        pots = _pots(obs)
        tasks = []

        if held == "soup":
            for target in _features(terrain, "S"):
                tasks.append(self._task("deliver", target, priority=0, description="deliver held soup"))
            return tasks or [self._task("wait", priority=20)]

        if held == "dish":
            ready = [p for p in pots if p["ready"]]
            ready.sort(key=lambda p: (self._lane_penalty(obs, p["pos"]), p["pos"]))
            for pot in ready:
                tasks.append(self._task("pickup_ready", pot["pos"], priority=1,
                                        description="pick up ready soup with dish"))
            if not ready:
                cooking = [p for p in pots if p["cooking"]]
                cooking.sort(key=lambda p: (p["remaining"] if isinstance(p["remaining"], (int, float)) else 999, p["pos"]))
                if cooking:
                    pot = cooking[0]
                    parking = self._parking_pose(obs, pot["pos"], current, partner_pos)
                    task = self._task("await_dish", pot["pos"], priority=3,
                                      wait_ready=True, description="wait off the pot gate until ready")
                    task["parking"] = parking
                    tasks.append(task)
            return tasks or [self._task("wait", priority=20)]

        if held in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial", "full_idle"}
                          and held in p["missing"]]
            compatible.sort(key=lambda p: (_distance(terrain, current, p["pos"], partner_pos), p["pos"]))
            for pot in compatible:
                tasks.append(self._task("fill_pot", pot["pos"], held, priority=2,
                                        description="put held ingredient in a compatible pot"))
            if not tasks:
                for counter in _counter_sites(obs):
                    tasks.append(self._task("deposit_counter", counter, held, priority=6,
                                            description="stage ingredient on a free counter"))
            return tasks or [self._task("wait", priority=20)]

        # A ready soup outranks every speculative trip.  If the partner visibly
        # carries the dish, the partner has the first claim and we prepare food.
        ready = [p for p in pots if p["ready"]]
        if ready and partner_held not in {"dish", "soup"}:
            for target in _features(terrain, "D"):
                tasks.append(self._task("fetch_dish", target, priority=1,
                                        description="fetch a dish for ready soup"))
            if tasks:
                return tasks

        full_idle = [p for p in pots if p["phase"] == "full_idle"]
        full_idle.sort(key=lambda p: (_distance(terrain, current, p["pos"], partner_pos), p["pos"]))
        for pot in full_idle:
            tasks.append(self._task("start_cooking", pot["pos"], priority=2,
                                    description="explicitly start a full idle pot"))
        if tasks:
            return tasks

        buildable = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"]]
        buildable.sort(key=lambda p: (_distance(terrain, current, p["pos"], partner_pos), p["pos"]))
        if buildable:
            pot = buildable[0]
            wanted = pot["missing"][0]
            # Prefer a staged matching item.  This is important when cooking a
            # different pot and lets the controller use a counter as a buffer.
            for name, point in _counter_items(obs):
                if name == wanted:
                    tasks.append(self._task("fetch_counter", point, wanted, priority=4,
                                            description="retrieve staged ingredient"))
            target_symbol = "O" if wanted == "onion" else "T"
            sources = _features(terrain, target_symbol)
            # A lane is a soft ownership rule, not a hard resource assumption.
            sources.sort(key=lambda p: (self._lane_penalty(obs, p,
                                      "right" if self.alt_lane_until >= _tick(obs) and self.lane == "left" else
                                      "left" if self.alt_lane_until >= _tick(obs) and self.lane == "right" else self.lane),
                                        _distance(terrain, current, p, partner_pos), p))
            for target in sources:
                tasks.append(self._task("fetch_ingredient", target, wanted, priority=5,
                                        description="fetch next recipe ingredient"))
            return tasks

        # Do not occupy a pot approach for the whole cook interval.  Fetch the
        # dish only when its exact route can meet the readiness boundary.
        cooking = [p for p in pots if p["cooking"]]
        cooking.sort(key=lambda p: (p["remaining"] if isinstance(p["remaining"], (int, float)) else 999, p["pos"]))
        if cooking and partner_held not in {"dish", "soup"}:
            pot = cooking[0]
            due = self._dish_due(obs, pot, current, partner_pos)
            hint = self.timing_hint if self.timing_until >= _tick(obs) else None
            if due or hint == "fetch_dish":
                for target in _features(terrain, "D"):
                    tasks.append(self._task("fetch_dish", target, priority=3,
                                            description="fetch dish just in time for cooking pot"))
                if tasks:
                    return tasks

        if cooking:
            # Starting a pot leaves the chef on its interaction pose.  A bare
            # wait there would recreate the very gate reservation this
            # controller is designed to remove, so explicitly clear the gate.
            pot = cooking[0]
            parking = self._parking_pose(obs, pot["pos"], current, partner_pos)
            task = self._task("park_cook", pot["pos"], priority=15,
                              description="clear the cooking-pot interaction gate")
            task["parking"] = parking
            return [task]

        return [self._task("wait", priority=20, description="wait without blocking a feature")]

    def _rank_tasks(self, obs, tasks):
        me, partner = _actors(obs)
        current, partner_pos = _pos(me.get("position")), _pos(partner.get("position"))
        terrain = _terrain(obs)
        ranked = []
        active_lane = self.lane
        if self.alt_lane_until >= _tick(obs) and active_lane in {"left", "right"}:
            active_lane = "right" if active_lane == "left" else "left"
        for task in tasks:
            if task.get("kind") in {"await_dish", "park_cook"}:
                pose = task.get("parking")
                path = _bfs(terrain, current, pose, {partner_pos} if partner_pos is not None else set()) if pose else None
            elif task.get("kind") == "wait":
                pose, path = None, [current] if current is not None else None
            else:
                pose, path = _best_pose(terrain, current, task.get("target"), partner_pos)
            length = len(path) if path is not None else 999
            kind = task.get("kind", "")
            lane_cost = 0
            if kind in {"fetch_ingredient", "fetch_counter", "deposit_counter"}:
                lane_cost = self._lane_penalty(obs, task.get("target"), active_lane)
            partner_near = 1 if (task.get("target") is not None and partner_pos is not None and
                                  abs(task["target"][0] - partner_pos[0]) + abs(task["target"][1] - partner_pos[1]) <= 1) else 0
            # Lane ownership is deliberately soft: it pays a small route
            # surcharge, but a substantially shorter legal source still wins.
            lane_score = length + 2 * lane_cost
            ranked.append((task.get("priority", 10), partner_near, lane_score, length,
                           str(task.get("id", "")), task, pose))
        ranked.sort(key=lambda row: row[:5])
        if not ranked:
            return self._task("wait", priority=20)
        chosen = ranked[0][5].copy()
        chosen["pose"] = ranked[0][6]
        return chosen

    def _same_task(self, old, new):
        return bool(old and new and old.get("kind") == new.get("kind") and
                    _pos(old.get("target")) == _pos(new.get("target")) and
                    old.get("item", "") == new.get("item", ""))

    def _job_valid(self, obs, candidates):
        if not self.job:
            return False
        me, _ = _actors(obs)
        held = _held(me)
        kind = self.job.get("kind")
        target = _pos(self.job.get("target"))
        if kind == "await_dish" and held == "dish":
            pot = next((p for p in _pots(obs) if p["pos"] == target), None)
            if pot and pot["ready"]:
                pose, _ = _best_pose(_terrain(obs), _pos(me.get("position")), target,
                                     _pos(_actors(obs)[1].get("position")))
                self.job = self._task("pickup_ready", target, priority=1,
                                      description="pick up soup whose cook boundary arrived")
                self.job["pose"] = pose
                self.job_tick = _tick(obs)
                self.epoch += 1
                self._log("dish_ready", tick=_tick(obs), target=target)
                return True
            if pot and pot["cooking"]:
                return True
        for candidate in candidates:
            if self._same_task(self.job, candidate):
                return True
        if kind in {"fetch_ingredient", "fetch_counter"} and held in INGREDIENTS:
            return False
        if kind == "fetch_dish" and held == "dish":
            return False
        if kind == "pickup_ready" and held == "soup":
            return False
        if kind == "deliver" and held != "soup":
            return False
        if kind == "fill_pot" and held not in INGREDIENTS:
            return False
        if kind == "deposit_counter" and held not in INGREDIENTS:
            return False
        if kind == "start_cooking":
            pot = next((p for p in _pots(obs) if p["pos"] == target), None)
            if not pot or pot["phase"] != "full_idle":
                return False
        return False

    def _refresh(self, obs, candidates):
        if self._job_valid(obs, candidates):
            return
        chosen = self._rank_tasks(obs, candidates)
        self.job = chosen
        self.job_tick = _tick(obs)
        self.epoch += 1
        self.blocked = 0
        self._log("job_commit", tick=self.job_tick, kind=chosen.get("kind"),
                  target=chosen.get("target"), pose=chosen.get("pose"),
                  wait_ready=chosen.get("wait_ready", False))

    def _can_interact(self, obs, job):
        me, _ = _actors(obs)
        held = _held(me)
        kind = job.get("kind")
        target = _pos(job.get("target"))
        pots = _pots(obs)
        pot = next((p for p in pots if p["pos"] == target), None)
        terrain = _terrain(obs)
        if kind == "deliver":
            return held == "soup"
        if kind == "fetch_ingredient":
            return not held and _cell(terrain, target) in {"O", "T"}
        if kind == "fetch_dish":
            return not held and _cell(terrain, target) == "D"
        if kind == "fetch_counter":
            return not held and any(name == job.get("item") and point == target
                                    for name, point in _counter_items(obs))
        if kind == "deposit_counter":
            occupied = {_obj_pos(obj) for obj in _objects(obs)}
            return held in INGREDIENTS and target not in occupied
        if kind == "fill_pot":
            return bool(pot and held in INGREDIENTS and pot["phase"] in {"empty", "partial", "full_idle"}
                        and held in pot["missing"])
        if kind == "start_cooking":
            return bool(pot and not held and pot["phase"] == "full_idle")
        if kind == "pickup_ready":
            return bool(pot and held == "dish" and pot["ready"])
        return False

    def _safe_action(self, action):
        if action in self.names:
            return action
        if self.noop in self.names:
            return self.noop
        return next(iter(self.names), "stay")

    def _detour(self, obs, goal):
        me, partner = _actors(obs)
        current, other = _pos(me.get("position")), _pos(partner.get("position"))
        if current is None:
            return self._safe_action(self.noop)
        predicted = self._partner_next(obs)
        candidates = []
        for nxt, delta in _neighbors(current):
            if not _walkable(_terrain(obs), nxt) or nxt == other or nxt == predicted:
                continue
            away = -(abs(nxt[0] - other[0]) + abs(nxt[1] - other[1])) if other else 0
            toward = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) if goal else 0
            # Do not immediately reverse into the cell just left unless there
            # is no other legal detour; this removes the old north/south pingpong.
            reverse = 1 if nxt == self.previous_cell else 0
            candidates.append((reverse, toward, away, self._lane_penalty(obs, nxt), nxt, delta))
        if not candidates:
            return self._safe_action(self.noop)
        candidates.sort(key=lambda row: row[:4])
        chosen = candidates[0]
        return self._safe_action(VEC_ACTION.get(chosen[5], self.noop))

    def _motor(self, obs):
        me, partner = _actors(obs)
        current = _pos(me.get("position"))
        other = _pos(partner.get("position"))
        if not self.job:
            return self._safe_action(self.noop)
        job = self.job
        if job.get("kind") == "wait" or (job.get("kind") in {"await_dish", "park_cook"} and job.get("parking") is None):
            return self._safe_action(self.noop)
        pose = _pos(job.get("pose"))
        target = _pos(job.get("target"))
        if current is None or pose is None or target is None:
            return self._safe_action(self.noop)

        if current == pose:
            if job.get("kind") in {"await_dish", "park_cook"}:
                return self._safe_action(self.noop)
            desired = (target[0] - current[0], target[1] - current[1])
            facing = _pos(me.get("orientation"))
            if desired in VEC_ACTION and facing != desired:
                return self._safe_action(VEC_ACTION[desired])
            if self._can_interact(obs, job):
                return self._safe_action("interact")
            # A critical pose is useful only while its precondition is true.
            # Keeping it for one or two ticks is safer than entering traffic.
            if job.get("kind") in {"pickup_ready", "deliver", "start_cooking", "fill_pot"}:
                return self._safe_action(self.noop)

        path = _bfs(_terrain(obs), current, pose, {other} if other is not None and other != pose else set())
        if path is None:
            self.blocked += 1
            return self._detour(obs, pose)
        if len(path) <= 1:
            self.blocked = 0
            return self._safe_action(self.noop)
        nxt = path[1]
        predicted = self._partner_next(obs)
        if nxt == other or nxt == predicted or (predicted == current and nxt == other):
            self.blocked += 1
            if self.blocked <= 2:
                return self._safe_action(self.noop)
            return self._detour(obs, pose)
        self.blocked = 0
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        return self._safe_action(VEC_ACTION.get(delta, self.noop))

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
            self._init_lane(obs)
            self._poll_astra(obs)
            self._poll_jev(obs)

            me, _ = _actors(obs)
            now_pos = _pos(me.get("position"))
            if (self.last_position is not None and now_pos == self.last_position and
                    self.last_action in DIRS and self.job and self.job.get("kind") != "await_dish"):
                self.blocked += 1
            elif now_pos is not None and now_pos != self.last_position:
                self.blocked = max(0, self.blocked - 1)
            self.previous_cell = self.last_position
            self.last_position = now_pos

            candidates = self._candidates(obs)
            self._start_astra(obs, candidates)
            old_epoch = self.epoch
            self._refresh(obs, candidates)
            reason = None
            if self.epoch != old_epoch:
                reason = "job_boundary"
            elif self.blocked >= 4:
                reason = "route_blocked"
            if reason:
                self._launch_jev(obs, candidates, reason)

            action = self._motor(obs)
            if self.yield_ticks > 0 and _held(me) == "":
                self.yield_ticks -= 1
                action = self._safe_action(self.noop)
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
        TimedKitchen(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
