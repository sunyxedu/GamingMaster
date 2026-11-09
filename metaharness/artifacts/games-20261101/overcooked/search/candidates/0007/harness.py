"""Tier-2 Overcooked controller with a leased right-of-way motor.

The slow model is a non-blocking strategic hint and Jev chooses among small,
code-generated jobs.  All movement, facing, interaction checks, and traffic
yielding are deterministic.  In particular, yielding is a lease: once the
partner owns an interaction approach, the controller walks to one parking tile
and waits there instead of repeatedly choosing a new escape direction.
"""
from __future__ import annotations

import json
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
MOVE_BY_VEC = {value: key for key, value in MOVES.items()}
FEATURES = {"O", "T", "D", "P", "S"}


def _point(value, default=(0, 0)):
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (int(value[0]), int(value[1]))
    except Exception:
        pass
    return default


def _state(obs):
    if isinstance(obs, dict) and isinstance(obs.get("state"), dict):
        return obs["state"]
    return {}


def _terrain(obs):
    if not isinstance(obs, dict):
        return []
    value = obs.get("terrain_mtx", [])
    return value if isinstance(value, list) else []


def _cell(terrain, pos):
    x, y = _point(pos)
    try:
        row = terrain[y]
        if isinstance(row, str):
            return row[x]
        if isinstance(row, (list, tuple)):
            return row[x]
    except Exception:
        pass
    return "X"


def _walkable(terrain, pos):
    cell = _cell(terrain, pos)
    # Spawn digits are floor in observations; all named fixtures and X are not.
    return cell not in FEATURES and cell != "X" and cell not in (None, "")


def _features(terrain, wanted=None):
    found = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (str, list, tuple)):
            continue
        for x, value in enumerate(row):
            if wanted is None:
                if value in FEATURES:
                    found.append((value, (x, y)))
            elif value == wanted:
                found.append((value, (x, y)))
    return found


def _feature_positions(terrain, wanted):
    return [position for _, position in _features(terrain, wanted)]


def _players(obs):
    value = _state(obs).get("players", [])
    return value if isinstance(value, list) else []


def _indices(obs):
    players = _players(obs)
    try:
        index = int(obs.get("controlled_agent_index", 0))
    except Exception:
        index = 0
    if index < 0 or index >= len(players):
        index = 0
    other = 1 if index == 0 else 0
    if other >= len(players):
        other = index
    return index, other


def _actors(obs):
    players = _players(obs)
    mine, other = _indices(obs)
    me = players[mine] if mine < len(players) and isinstance(players[mine], dict) else {}
    partner = players[other] if other < len(players) and isinstance(players[other], dict) else {}
    return me, partner


def _objects(obs):
    value = _state(obs).get("objects", [])
    return value if isinstance(value, list) else []


def _object_position(obj):
    return _point(obj.get("position")) if isinstance(obj, dict) else None


def _object_name(obj):
    return str(obj.get("name", "")) if isinstance(obj, dict) else ""


def _held_object(player):
    if not isinstance(player, dict):
        return None
    value = player.get("held_object")
    return value if isinstance(value, dict) else None


def _held_name(player):
    obj = _held_object(player)
    return _object_name(obj) if obj else ""


def _ingredient_names(obj):
    if not isinstance(obj, dict):
        return []
    values = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name")
        if value is not None:
            result.append(str(value))
    return result


def _recipes(obs):
    state = _state(obs)
    values = state.get("all_orders", [])
    if not isinstance(values, list) or not values:
        values = state.get("bonus_orders", [])
    result = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, dict):
            ingredients = value.get("ingredients", value.get("_ingredients", []))
        else:
            ingredients = value
        if isinstance(ingredients, list):
            names = []
            for ingredient in ingredients:
                if isinstance(ingredient, dict):
                    ingredient = ingredient.get("name")
                if ingredient is not None:
                    names.append(str(ingredient))
            if names:
                result.append(names)
    return result or [["onion", "onion", "onion"]]


def _missing_for(ingredients, recipes):
    have = Counter(str(value) for value in ingredients)
    best = None
    best_missing = None
    for recipe in recipes:
        wanted = Counter(recipe)
        if any(have[name] > wanted[name] for name in have):
            continue
        remaining = have.copy()
        missing = []
        for name in recipe:
            if remaining[name]:
                remaining[name] -= 1
            else:
                missing.append(name)
        if best is None or len(missing) < len(best_missing):
            best = list(recipe)
            best_missing = missing
    return best, (best_missing if best_missing is not None else [])


def _jsonable(value):
    """Keep every model request composed only of strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tick(obs):
    value = obs.get("tick", _state(obs).get("timestep", 0)) if isinstance(obs, dict) else 0
    try:
        return int(value)
    except Exception:
        return 0


class Kitchen:
    """A deterministic job executor surrounded by two small model bridges."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = getattr(game, "action_spec", None)
        names = getattr(spec, "names", []) if spec is not None else []
        self.names = list(names) if isinstance(names, (list, tuple, set)) else []
        advertised_noop = getattr(spec, "noop", None) if spec is not None else None
        self.noop = advertised_noop if advertised_noop in self.names else (
            "stay" if "stay" in self.names else (self.names[0] if self.names else "stay")
        )

        self.system1_off = getattr(ctx, "system1", object()) is None
        self.system2_off = getattr(ctx, "system2", object()) is None
        self.jev_future = None
        self.jev_request_sig = None
        self.jev_request_tick = -999
        self.jev_last_attempt = -999
        self.jev_failures = 0
        self.jev_advice = None
        self.jev_advice_sig = None
        self.active_task = None

        self.astra_future = None
        self.astra_started = False
        self.astra_hint = {}

        # Traffic is deliberately stateful.  A single lease prevents a blocked
        # approach from turning into the north/south oscillation seen in 0005.
        self.traffic_anchor = None
        self.traffic_task = None
        self.traffic_goal = None
        self.traffic_until = -1
        self.traffic_reason = ""

        self.last_position = None
        self.last_task_id = None
        self.stuck_ticks = 0

    def action(self, name):
        return name if name in self.names else self.noop

    def _path(self, terrain, start, goal, blocked=()):
        start = _point(start)
        goal = _point(goal)
        blocked = set(blocked)
        blocked.discard(start)
        if start == goal:
            return [start]
        queue = deque([start])
        previous = {start: None}
        while queue:
            here = queue.popleft()
            for dx, dy in MOVES.values():
                there = (here[0] + dx, here[1] + dy)
                if there in previous or there in blocked or not _walkable(terrain, there):
                    continue
                previous[there] = here
                if there == goal:
                    result = [there]
                    while result[-1] != start:
                        result.append(previous[result[-1]])
                    result.reverse()
                    return result
                queue.append(there)
        return None

    def _poses(self, terrain, target):
        tx, ty = _point(target)
        result = []
        for dx, dy in MOVES.values():
            pose = (tx - dx, ty - dy)
            if pose not in result and _walkable(terrain, pose):
                result.append(pose)
        return result

    def _path_to_target(self, obs, target, blocked_partner=True):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        start = _point(me.get("position"))
        partner_pos = _point(partner.get("position"))
        blocked = {partner_pos} if blocked_partner else set()
        paths = []
        for pose in self._poses(terrain, target):
            path = self._path(terrain, start, pose, blocked=blocked)
            if path is not None:
                paths.append(path)
        paths.sort(key=lambda path: (len(path), path[-1]))
        return (paths[0] if paths else None), self._poses(terrain, target)

    def _pot_info(self, obs):
        terrain = _terrain(obs)
        recipes = _recipes(obs)
        by_position = {}
        for obj in _objects(obs):
            position = _object_position(obj)
            if position is not None:
                by_position.setdefault(position, []).append(obj)
        result = []
        for position in _feature_positions(terrain, "P"):
            soup = next(
                (obj for obj in by_position.get(position, []) if _object_name(obj) == "soup"),
                None,
            )
            ingredients = _ingredient_names(soup)
            recipe, missing = _missing_for(ingredients, recipes)
            ready = bool(soup and soup.get("is_ready"))
            cooking = bool(soup and soup.get("is_cooking"))
            if soup and not ready and not cooking:
                cooking_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1))
                if isinstance(cooking_tick, (int, float)) and cooking_tick >= 0:
                    cooking = not bool(soup.get("is_idle", False))
            full = bool(soup and recipe and not missing and not cooking and not ready)
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
                "pos": position,
                "ingredients": ingredients,
                "recipe": recipe or [],
                "missing": missing,
                "ready": ready,
                "cooking": cooking,
                "full": full,
                "phase": phase,
                "object": soup,
            })
        return result

    def _counter_objects(self, obs):
        terrain = _terrain(obs)
        result = []
        for obj in _objects(obs):
            name = _object_name(obj)
            position = _object_position(obj)
            if name in {"onion", "tomato", "dish", "soup"} and position is not None:
                if _cell(terrain, position) == "X":
                    result.append((name, position, obj))
        return result

    def _free_counters(self, obs):
        terrain = _terrain(obs)
        occupied = {
            _object_position(obj)
            for obj in _objects(obs)
            if _object_position(obj) is not None
        }
        result = []
        for _, position in _features(terrain, "X"):
            if position in occupied:
                continue
            if any(_walkable(terrain, (position[0] + dx, position[1] + dy)) for dx, dy in MOVES.values()):
                result.append(position)
        return result

    @staticmethod
    def _job(job_id, kind, target=None, ingredient=None, text="", wait=False, priority=50):
        return {
            "id": str(job_id),
            "kind": str(kind),
            "target": list(target) if target is not None else None,
            "ingredient": ingredient,
            "description": text or str(kind),
            "wait_until_ready": bool(wait),
            "priority": int(priority),
        }

    def _jobs(self, obs):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        held = _held_name(me)
        partner_held = _held_name(partner)
        pots = self._pot_info(obs)
        jobs = []

        if held == "soup":
            for pos in _feature_positions(terrain, "S"):
                jobs.append(self._job(
                    "serve_%d_%d" % pos, "deliver", pos,
                    text="carry the held soup to serving window %s and interact" % (pos,),
                    priority=0,
                ))
            return jobs or [self._job("wait", "wait", text="wait safely", priority=99)]

        if held == "dish":
            for pot in [item for item in pots if item["ready"]]:
                x, y = pot["pos"]
                jobs.append(self._job(
                    "take_ready_%d_%d" % (x, y), "pickup_ready", pot["pos"],
                    text="face ready soup pot %s and pick it up" % (pot["pos"],), priority=1,
                ))
            for pot in [item for item in pots if item["cooking"]]:
                x, y = pot["pos"]
                jobs.append(self._job(
                    "wait_pot_%d_%d" % (x, y), "stage_dish", pot["pos"],
                    text="stage with the dish by cooking pot %s and wait" % (pot["pos"],),
                    wait=True, priority=5,
                ))
            return jobs or [self._job("wait_dish", "wait", text="hold the dish safely", priority=99)]

        if held in {"onion", "tomato"}:
            compatible = [
                pot for pot in pots
                if held in pot["missing"] and not pot["cooking"] and not pot["ready"]
            ]
            compatible.sort(key=lambda pot: (len(pot["missing"]), pot["pos"]))
            for pot in compatible[:3]:
                x, y = pot["pos"]
                jobs.append(self._job(
                    "fill_%s_%d_%d" % (held, x, y), "fill_pot", pot["pos"], held,
                    text="put held %s into compatible pot %s" % (held, pot["pos"]), priority=2,
                ))
            if not jobs:
                for x, y in self._free_counters(obs)[:3]:
                    jobs.append(self._job(
                        "park_%s_%d_%d" % (held, x, y), "deposit_counter", (x, y), held,
                        text="put held %s on free counter (%d, %d)" % (held, x, y), priority=30,
                    ))
            return jobs or [self._job("wait_held", "wait", text="wait for a compatible pot", priority=99)]

        # Empty hands: protect a ready soup, then launch a full pot, then feed
        # the most complete idle recipe.  This ordering is code-owned so a bad
        # Jev choice cannot turn a ready soup into a speculative ingredient run.
        ready = [pot for pot in pots if pot["ready"]]
        if ready and partner_held not in {"dish", "soup"}:
            for _, (x, y) in _features(terrain, "D"):
                jobs.append(self._job(
                    "dish_%d_%d" % (x, y), "fetch_dish", (x, y),
                    text="fetch a dish from dispenser (%d, %d) for ready soup" % (x, y),
                    priority=0,
                ))
            if jobs:
                return jobs

        full_idle = [pot for pot in pots if pot["full"]]
        for pot in full_idle:
            x, y = pot["pos"]
            jobs.append(self._job(
                "start_%d_%d" % (x, y), "start_cooking", pot["pos"],
                text="start cooking the full idle pot at %s with empty hands" % (pot["pos"],),
                priority=1,
            ))
        if jobs:
            return jobs

        buildable = [pot for pot in pots if pot["missing"] and not pot["cooking"] and not pot["ready"]]
        buildable.sort(key=lambda pot: (len(pot["missing"]), pot["pos"]))
        if buildable:
            pot = buildable[0]
            ingredient = pot["missing"][0]
            staged = [item for item in self._counter_objects(obs) if item[0] == ingredient]
            for name, position, _ in staged[:3]:
                x, y = position
                jobs.append(self._job(
                    "counter_%s_%d_%d" % (name, x, y), "fetch_counter", position, name,
                    text="pick up staged %s from counter (%d, %d)" % (name, x, y), priority=3,
                ))
            dispenser = "O" if ingredient == "onion" else "T"
            for x, y in _feature_positions(terrain, dispenser)[:3]:
                jobs.append(self._job(
                    "fetch_%s_%d_%d" % (ingredient, x, y), "fetch_ingredient", (x, y), ingredient,
                    text="fetch one %s from dispenser (%d, %d) for pot %s" % (ingredient, x, y, pot["pos"]),
                    priority=4,
                ))
            if jobs:
                return jobs

        if any(pot["cooking"] for pot in pots) and partner_held not in {"dish", "soup"}:
            for _, (x, y) in _features(terrain, "D"):
                jobs.append(self._job(
                    "dish_cook_%d_%d" % (x, y), "fetch_dish", (x, y),
                    text="fetch a dish while soup cooks at (%d, %d)" % (x, y), priority=6,
                ))
            if jobs:
                return jobs

        if partner_held not in {"dish", "soup"}:
            for _, (x, y) in _features(terrain, "D"):
                jobs.append(self._job(
                    "dish_pipeline_%d_%d" % (x, y), "fetch_dish", (x, y),
                    text="fetch a dish to keep the pipeline moving from (%d, %d)" % (x, y), priority=20,
                ))
        return jobs or [self._job("wait", "wait", text="wait safely for the partner", priority=99)]

    def _job_signature(self, obs, jobs):
        me, partner = _actors(obs)
        pot_sig = []
        for pot in self._pot_info(obs):
            pot_sig.append((
                pot["pos"], pot["phase"], tuple(pot["ingredients"]), tuple(pot["missing"]),
            ))
        return (
            _held_name(me), _held_name(partner), tuple(pot_sig),
            tuple(job["id"] for job in jobs),
        )

    def _compact(self, obs, jobs):
        me, partner = _actors(obs)
        pots = self._pot_info(obs)
        return _jsonable({
            "tick": _tick(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {
                "position": _point(me.get("position")),
                "facing": _point(me.get("orientation")),
                "held": _held_name(me),
            },
            "partner": {
                "position": _point(partner.get("position")),
                "facing": _point(partner.get("orientation")),
                "held": _held_name(partner),
                "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else "",
                "task": obs.get("partner_task", "") if isinstance(obs, dict) else "",
            },
            "pots": [
                {"position": pot["pos"], "phase": pot["phase"],
                 "ingredients": pot["ingredients"], "missing": pot["missing"]}
                for pot in pots
            ],
            "orders": _recipes(obs),
            "events": (obs.get("last_events", []) if isinstance(obs, dict) else [])[-6:],
            "candidates": [
                {"id": job["id"], "kind": job["kind"], "description": job["description"]}
                for job in jobs[:8]
            ],
            "traffic": {
                "lease": self.traffic_reason,
                "stuck_ticks": self.stuck_ticks,
            },
        })

    def _astra_start(self, obs, jobs):
        if self.astra_started or self.system2_off:
            return
        self.astra_started = True
        snapshot = self._compact(obs, jobs)
        prompt = (
            "Act as the slow strategic coordinator for this symbolic Overcooked-AI episode. "
            "The executor owns exact geometry, recipes, facing, and primitive actions. Choose "
            "only a rolling medium-level preference for the controlled chef while respecting a "
            "scripted partner and one-cell traffic. Protect ready soup, start full idle pots, "
            "and do not make the chef contest the partner's interaction approach. Return ONLY "
            "compact JSON with optional string fields preferred_kind (one of deliver, "
            "pickup_ready, start_cooking, fill_pot, fetch_ingredient, fetch_dish, "
            "stage_dish, fetch_counter, deposit_counter, wait), traffic_rule, and rationale. "
            "Never return directions or an action name.\n" + json.dumps(snapshot, separators=(",", ":"))
        )

        def ask():
            if self.ctx.done:
                return ""
            try:
                result = self.ctx.think(prompt, timeout_s=100.0)
                return getattr(result, "text", "")
            except Exception:
                return ""

        try:
            self.astra_future = self.ctx.spawn(ask)
        except Exception:
            self.astra_future = None

    def _astra_poll(self):
        future = self.astra_future
        if future is None:
            return
        try:
            if not future.done():
                return
            self.astra_future = None
            result = future.result()
            text = getattr(result, "text", result if isinstance(result, str) else "")
            if not isinstance(text, str):
                return
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return
            value = json.loads(match.group(0))
            if not isinstance(value, dict):
                return
            self.astra_hint = {
                key: value[key] for key in ("preferred_kind", "traffic_rule")
                if isinstance(value.get(key), str)
            }
        except Exception:
            self.astra_future = None

    def _jev_state(self, obs, jobs):
        return {
            "game": "Overcooked-AI symbolic real-time kitchen",
            "observation": self._compact(obs, jobs),
            "rules": {
                "one_primitive_per_tick": True,
                "interact_requires_facing_the_listed_target": True,
                "directional_candidates_are_not_actions": True,
                "right_of_way": "yield briefly when the partner owns the only approach",
            },
        }

    def _launch_jev(self, obs, jobs):
        if self.system1_off or self.jev_future is not None or not jobs:
            return
        now = _tick(obs)
        signature = self._job_signature(obs, jobs)
        changed = signature != self.jev_request_sig
        heartbeat = now - self.jev_last_attempt >= 20
        if not changed and not heartbeat:
            return
        if now - self.jev_last_attempt < 2:
            return
        criteria = {
            job["id"]: {
                "what": job["description"],
                "not_for": "inventing a direction, changing a recipe, or ignoring a ready-soup emergency",
            }
            for job in jobs[:8]
        }
        questions = {
            "job": Choice(
                "Choose the best current medium-level job for the controlled chef from `observation.candidates`. "
                "The executor will compute the shortest safe path. Prefer a listed ready-soup pickup or "
                "delivery, then start/fill work that is already legal; never output a raw direction or an "
                "unlisted id.",
                criteria,
            ),
            "yield": Noul(
                "Should the controlled chef voluntarily give the partner one short right-of-way interval? "
                "Answer true only when `observation.partner` is carrying soup/dish or its facing/position "
                "shows an immediate approach conflict; otherwise false.",
                {
                    "true": "A brief stay or evacuation protects the partner's high-priority route.",
                    "false": "No immediate partner traffic conflict is established.",
                },
            ),
        }
        request_state = self._jev_state(obs, jobs)
        self.jev_request_sig = signature
        self.jev_request_tick = now
        self.jev_last_attempt = now

        def ask():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(request_state, questions, timeout_s=3.0)
            except Exception as error:
                return {"_error": type(error).__name__}

        try:
            self.jev_future = self.ctx.spawn(ask)
        except Exception:
            self.jev_future = None
            self.jev_failures += 1

    def _jev_poll(self, obs, jobs):
        future = self.jev_future
        if future is None:
            return
        try:
            if not future.done():
                return
            self.jev_future = None
            answers = future.result()
            if not isinstance(answers, dict):
                self.jev_failures += 1
                return
            if "_error" in answers:
                self.jev_failures += 1
                if answers.get("_error") in {"BudgetExceeded", "EpisodeOver", "ModelUnavailable"}:
                    self.system1_off = True
                return
            current_sig = self._job_signature(obs, jobs)
            if current_sig != self.jev_request_sig:
                return
            valid = {job["id"] for job in jobs}
            answer = answers.get("job")
            chosen = getattr(answer, "value", None)
            probabilities = getattr(answer, "probabilities", {}) or {}
            try:
                probability = float(probabilities.get(chosen, 0.0)) if isinstance(probabilities, dict) else 0.0
            except Exception:
                probability = 0.0
            # No distribution is common in pre-flight stand-ins.  The exact-id
            # check remains the hard safety gate in that case.
            if chosen in valid and (probability == 0.0 or probability >= 0.40):
                self.jev_advice = chosen
                self.jev_advice_sig = current_sig
            yield_answer = answers.get("yield")
            yield_value = getattr(yield_answer, "value", 0.0)
            if isinstance(yield_value, (int, float)) and yield_value >= 0.72:
                # The motor still verifies geometry.  This flag only adds a
                # short lease when the partner is already near the controlled chef.
                self.traffic_reason = "jev_right_of_way"
        except Exception:
            self.jev_failures += 1
        if self.jev_failures >= 4:
            self.system1_off = True

    def _select_job(self, obs, jobs):
        if not jobs:
            return self._job("wait", "wait", text="wait safely", priority=99)
        current_sig = self._job_signature(obs, jobs)
        ids = {job["id"] for job in jobs}

        # Preserve a route through several movement ticks.  Replanning is
        # forced by held-object/pot phase changes because then the old target
        # may be mechanically invalid.
        if self.active_task and self.active_task.get("id") in ids:
            active = self.active_task
        else:
            active = None

        urgent = min(job["priority"] for job in jobs)
        advised = None
        if self.jev_advice in ids and self.jev_advice_sig == current_sig:
            advised = next(job for job in jobs if job["id"] == self.jev_advice)
        # A fresh Jev answer may select an equal-cost dispenser/route, but it
        # cannot demote an already active job below a more urgent code-ranked
        # candidate.  Consume the answer once so it does not thrash the route.
        if advised is not None and (
            active is None or advised["priority"] <= active.get("priority", 50)
        ):
            chosen = advised
            if active is None or active.get("id") != advised.get("id"):
                self.jev_advice = None
        elif active is not None and active.get("priority", 50) <= urgent:
            chosen = active
        else:
            preferred_kind = self.astra_hint.get("preferred_kind")
            choices = [job for job in jobs if job["kind"] == preferred_kind]
            chosen = choices[0] if choices else jobs[0]

        # A blocked dispenser is not a reason to park if code already exposed
        # an equally urgent alternative route.  This matters on asymmetric and
        # mixed layouts; the sole-pot-approach case still takes the lease.
        if chosen.get("target") is not None and self._target_blocked_by_partner(obs, chosen["target"]):
            alternatives = [
                job for job in jobs
                if job.get("id") != chosen.get("id")
                and job.get("priority", 50) <= chosen.get("priority", 50)
                and job.get("target") is not None
            ]
            for alternative in alternatives:
                route, _ = self._path_to_target(obs, alternative["target"], blocked_partner=True)
                if route is not None:
                    chosen = alternative
                    self.jev_advice = None
                    break
        self.active_task = chosen
        return chosen

    def _partner_next(self, obs):
        _, partner = _actors(obs)
        position = _point(partner.get("position"))
        action = obs.get("partner_action_last", "") if isinstance(obs, dict) else ""
        delta = MOVES.get(action)
        if delta is None:
            return None
        return (position[0] + delta[0], position[1] + delta[1])

    def _partner_faces_me(self, obs):
        me, partner = _actors(obs)
        me_pos = _point(me.get("position"))
        partner_pos = _point(partner.get("position"))
        facing = _point(partner.get("orientation"), default=(99, 99))
        return (partner_pos[0] + facing[0], partner_pos[1] + facing[1]) == me_pos

    def _target_blocked_by_partner(self, obs, target):
        # Occupying one interaction pose is harmless when another pose remains
        # reachable.  Only declare a lease when every reachable route needs
        # the partner's current cell.
        path_with, poses = self._path_to_target(obs, target, blocked_partner=True)
        if path_with is not None:
            return False
        _, partner = _actors(obs)
        partner_pos = _point(partner.get("position"))
        if partner_pos not in poses:
            return False
        path_without, _ = self._path_to_target(obs, target, blocked_partner=False)
        return path_without is not None

    def _clear_traffic(self):
        self.traffic_anchor = None
        self.traffic_task = None
        self.traffic_goal = None
        self.traffic_until = -1
        self.traffic_reason = ""

    def _parking_goal(self, obs, task):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        start = _point(me.get("position"))
        partner_pos = _point(partner.get("position"))
        target_poses = set(self._poses(terrain, task.get("target"))) if task.get("target") else set()
        # A bounded BFS tree gives a stable, nearby parking tile.  Prefer a
        # tile at least two cells from the partner and not the target approach.
        queue = deque([start])
        paths = {start: [start]}
        while queue:
            here = queue.popleft()
            for dx, dy in MOVES.values():
                there = (here[0] + dx, here[1] + dy)
                if there in paths or not _walkable(terrain, there) or there == partner_pos:
                    continue
                paths[there] = paths[here] + [there]
                queue.append(there)
        candidates = []
        predicted = self._partner_next(obs)
        for position, path in paths.items():
            if position in target_poses or position == partner_pos:
                continue
            if predicted is not None and position == predicted:
                continue
            distance = abs(position[0] - partner_pos[0]) + abs(position[1] - partner_pos[1])
            exits = sum(
                _walkable(terrain, (position[0] + dx, position[1] + dy))
                for dx, dy in MOVES.values()
            )
            candidates.append((distance < 2, len(path), -distance, -int(exits), position, path))
        if not candidates:
            return start
        candidates.sort()
        return candidates[0][-2]

    def _traffic_action(self, obs, task, reason):
        """Return a safe primitive, or None when the lease has been released."""
        _, partner = _actors(obs)
        partner_pos = _point(partner.get("position"))
        me, _ = _actors(obs)
        current = _point(me.get("position"))
        now = _tick(obs)
        task_id = task.get("id") if isinstance(task, dict) else ""

        if self.traffic_anchor != partner_pos or self.traffic_task != task_id:
            self.traffic_anchor = partner_pos
            self.traffic_task = task_id
            self.traffic_goal = None
            self.traffic_until = now + 3
            self.traffic_reason = reason
        elif not self.traffic_reason:
            self.traffic_reason = reason

        if self.traffic_goal is None:
            self.traffic_goal = self._parking_goal(obs, task)

        # A changed partner position grants the lane back immediately.
        if partner_pos != self.traffic_anchor:
            self._clear_traffic()
            return None

        if current != self.traffic_goal:
            path = self._path(_terrain(obs), current, self.traffic_goal, blocked={partner_pos})
            if path is not None and len(path) > 1:
                next_position = path[1]
                predicted = self._partner_next(obs)
                if next_position == predicted or next_position == partner_pos:
                    return self.action("stay")
                delta = (next_position[0] - current[0], next_position[1] - current[1])
                return self.action(MOVE_BY_VEC.get(delta, "stay"))
            return self.action("stay")

        # At the stable parking tile, wait while the old conflict persists. If
        # it was only a partner-facing collision, a short lease is enough; a
        # partner occupying the requested approach keeps the lease until it
        # actually moves.
        target_blocked = bool(task.get("target") is not None and self._target_blocked_by_partner(obs, task["target"]))
        if target_blocked or now < self.traffic_until:
            return self.action("stay")
        self._clear_traffic()
        return None

    def _motor(self, obs, task):
        if task.get("kind") == "wait" or task.get("target") is None:
            return self.action("stay")

        # Parent 0003's right-of-way rule is applied before path following.
        # It is not enough to merely choose a different target: the partner
        # can be facing into our current tile even when our own target is free.
        if self._partner_faces_me(obs):
            yielded = self._traffic_action(obs, task, "partner_facing_into_self")
            if yielded is not None:
                return yielded

        target = _point(task.get("target"))
        path, poses = self._path_to_target(obs, target, blocked_partner=True)
        if path is None:
            # Distinguish an actual disconnected target from a temporary
            # partner obstruction.  Only the latter gets a parking lease.
            ignoring_partner, _ = self._path_to_target(obs, target, blocked_partner=False)
            if ignoring_partner is not None:
                yielded = self._traffic_action(obs, task, "approach_occupied")
                if yielded is not None:
                    return yielded
                path, poses = self._path_to_target(obs, target, blocked_partner=True)
            if path is None:
                return self.action("stay")

        me, partner = _actors(obs)
        current = _point(me.get("position"))
        partner_pos = _point(partner.get("position"))
        predicted = self._partner_next(obs)
        if len(path) > 1:
            next_position = path[1]
            # Avoid both direct collisions and a predicted simultaneous swap.
            if next_position == partner_pos or next_position == predicted:
                yielded = self._traffic_action(obs, task, "next_step_conflict")
                if yielded is not None:
                    return yielded
                return self.action("stay")
            # A successful route clears an obsolete traffic lease.
            if self.traffic_anchor is not None and partner_pos != self.traffic_anchor:
                self._clear_traffic()
            delta = (next_position[0] - current[0], next_position[1] - current[1])
            return self.action(MOVE_BY_VEC.get(delta, "stay"))

        # We are in an interaction pose.  Movement and facing are separate
        # actions; never spend an interaction tick with the wrong orientation.
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        desired = MOVE_BY_VEC.get((dx, dy))
        if desired is None:
            return self.action("stay")
        if _point(me.get("orientation")) != (dx, dy):
            return self.action(desired)
        if task.get("wait_until_ready"):
            return self.action("stay")
        return self.action("interact")

    def _update_stuck(self, obs, task, action):
        me, _ = _actors(obs)
        position = _point(me.get("position"))
        if position == self.last_position and action in MOVES and task.get("kind") not in {"stage_dish", "wait"}:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        self.last_position = position
        self.last_task_id = task.get("id")
        if self.stuck_ticks >= 7:
            # Invalidate only the semantic lease; the next motor pass will
            # either find a detour or establish one stable traffic parking tile.
            self.active_task = None
            self.jev_advice = None
            self.stuck_ticks = 0

    def _observe(self):
        try:
            return self.game.observe()
        except Exception:
            try:
                return self.ctx.observe()
            except Exception:
                return {}

    def run(self):
        ctx = self.ctx
        while not ctx.done and not bool(getattr(self.game, "done", False)):
            obs = self._observe()
            if not isinstance(obs, dict):
                obs = {}
            jobs = self._jobs(obs)
            if not self.astra_started:
                self._astra_start(obs, jobs)
            self._astra_poll()
            self._jev_poll(obs, jobs)
            task = self._select_job(obs, jobs)
            action = self._motor(obs, task)
            self._update_stuck(obs, task, action)

            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                try:
                    ctx.act(self.action("stay"))
                except Exception:
                    return

            # Model calls start only after a safe action has been posted. They
            # cannot hold the game tick, and their answers are checked next loop.
            self._launch_jev(obs, jobs)
            if ctx.done or bool(getattr(self.game, "done", False)):
                break
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                return


def build(ctx):
    class Harness:
        def __init__(self, context):
            self.ctx = context

        def run_episode(self, game):
            if not getattr(game, "tick_s", None):
                return
            Kitchen(self.ctx, game).run()

    return Harness(ctx)
