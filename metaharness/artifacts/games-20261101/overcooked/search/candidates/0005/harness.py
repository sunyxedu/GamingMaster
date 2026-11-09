"""A real-time symbolic kitchen controller.

The controller keeps the motor layer deterministic: it derives interaction poses,
shortest paths, and legal facing actions from the observation.  Jev only chooses
between already legal medium-level jobs, and a background Astra call may change
that job preference.  Thus a slow or malformed model answer cannot turn into a
raw direction or stop the clock.
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
VEC_ACTION = {v: k for k, v in DIRECTIONS.items()}
FEATURES = {"O", "T", "D", "P", "S"}


def _pair(value, default=(0, 0)):
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
    terrain = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return terrain if isinstance(terrain, list) else []


def _cell(terrain, pos):
    x, y = pos
    try:
        row = terrain[y]
        return row[x] if isinstance(row, (list, tuple, str)) else "X"
    except Exception:
        return "X"


def _features(terrain, symbol=None):
    found = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (list, tuple, str)):
            continue
        for x, value in enumerate(row):
            if symbol is None:
                if value in FEATURES:
                    found.append((value, (x, y)))
            elif value == symbol:
                found.append((value, (x, y)))
    return found


def _walkable(terrain, pos):
    # Digits used for spawn markers are floor.  Features and X are not floor.
    return _cell(terrain, pos) not in FEATURES and _cell(terrain, pos) != "X"


def _objects(state):
    value = state.get("objects", []) if isinstance(state, dict) else []
    return value if isinstance(value, list) else []


def _obj_pos(obj):
    return _pair(obj.get("position")) if isinstance(obj, dict) else None


def _obj_name(obj):
    return str(obj.get("name", "")) if isinstance(obj, dict) else ""


def _held(player):
    if not isinstance(player, dict):
        return None
    value = player.get("held_object")
    return value if isinstance(value, dict) else None


def _ingredient_names(obj):
    if not isinstance(obj, dict):
        return []
    values = obj.get("_ingredients", obj.get("ingredients", []))
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if isinstance(value, dict):
            name = value.get("name")
        else:
            name = value
        if name is not None:
            result.append(str(name))
    return result


def _orders(state):
    values = state.get("all_orders", []) if isinstance(state, dict) else []
    if not isinstance(values, list) or not values:
        values = state.get("bonus_orders", []) if isinstance(state, dict) else []
    result = []
    for value in values:
        ingredients = value.get("ingredients") if isinstance(value, dict) else value
        if isinstance(ingredients, list) and ingredients:
            result.append([str(x) for x in ingredients])
    return result or [["onion", "onion", "onion"]]


def _recipe_missing(ingredients, recipes):
    have = Counter(ingredients)
    best = None
    best_missing = None
    for recipe in recipes:
        wanted = Counter(recipe)
        if any(have[key] > wanted[key] for key in have):
            continue
        missing = []
        for key in recipe:
            if have[key] < wanted[key]:
                missing.append(key)
                have[key] += 1
        # Recompute from the original counts; the temporary increment above is
        # only used to preserve the recipe's ingredient order.
        current = Counter(ingredients)
        missing = []
        for key in recipe:
            if current[key] < wanted[key]:
                missing.append(key)
                current[key] += 1
        if best is None or len(missing) < len(best_missing):
            best = recipe
            best_missing = missing
    return best, (best_missing if best_missing is not None else [])


def _feature_positions(terrain, symbol):
    return [pos for _, pos in _features(terrain, symbol)]


def _held_name(player):
    obj = _held(player)
    return _obj_name(obj) if obj else ""


def _jsonable(value):
    """Make model state safe even if an adaptor returns a tuple-like value."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class KitchenController:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.names = set(getattr(game.action_spec, "names", []) or [])
        self.noop = getattr(game.action_spec, "noop", None) or "stay"
        if self.noop not in self.names and "stay" in self.names:
            self.noop = "stay"
        self.preference_id = None
        self.preference_tick = -999
        self.astra_hint = {}
        self.astra_future = None
        self.last_jev_tick = -999
        self.last_signature = None
        self.system1_disabled = getattr(ctx, "system1", object()) is None
        self.last_pos = None
        self.last_task_id = None
        self.stuck = 0
        self.escape_ticks = 0
        self.yield_ticks = 0
        self.started_astra = False

    def action(self, name):
        if name in self.names:
            return name
        if self.noop in self.names:
            return self.noop
        return next(iter(self.names), name)

    def _players(self, obs):
        players = _state(obs).get("players", [])
        return players if isinstance(players, list) else []

    def _indices(self, obs):
        players = self._players(obs)
        try:
            me_index = int(obs.get("controlled_agent_index", 0))
        except Exception:
            me_index = 0
        if me_index < 0 or me_index >= len(players):
            me_index = 0
        partner_index = 1 if me_index == 0 else 0
        if partner_index >= len(players):
            partner_index = me_index
        return me_index, partner_index

    def _me_partner(self, obs):
        players = self._players(obs)
        mi, pi = self._indices(obs)
        me = players[mi] if mi < len(players) and isinstance(players[mi], dict) else {}
        partner = players[pi] if pi < len(players) and isinstance(players[pi], dict) else {}
        return me, partner

    def _path(self, terrain, start, goal, blocked=()):
        start = _pair(start)
        goal = _pair(goal)
        blocked = set(blocked)
        blocked.discard(start)
        if start == goal:
            return [start]
        queue = deque([start])
        previous = {start: None}
        while queue:
            current = queue.popleft()
            for dx, dy in DIRECTIONS.values():
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in previous or nxt in blocked or not _walkable(terrain, nxt):
                    continue
                previous[nxt] = current
                if nxt == goal:
                    path = [nxt]
                    while path[-1] is not None and path[-1] != start:
                        path.append(previous[path[-1]])
                    path.reverse()
                    return path
                queue.append(nxt)
        return None

    def _interaction_path(self, obs, target):
        terrain = _terrain(obs)
        me, partner = self._me_partner(obs)
        start = _pair(me.get("position"))
        partner_pos = _pair(partner.get("position"))
        tx, ty = _pair(target)
        poses = []
        for dx, dy in DIRECTIONS.values():
            pose = (tx - dx, ty - dy)
            if _walkable(terrain, pose) and pose not in poses:
                poses.append(pose)
        paths = []
        for pose in poses:
            path = self._path(terrain, start, pose, blocked={partner_pos})
            if path is not None:
                paths.append(path)
        if not paths:
            # If the partner occupies the only pose, retain the target so the
            # caller can yield rather than inventing an illegal interaction.
            return None, poses
        paths.sort(key=lambda path: (len(path), path[-1]))
        return paths[0], poses

    def _pot_infos(self, obs):
        terrain = _terrain(obs)
        state = _state(obs)
        recipes = _orders(state)
        objects = _objects(state)
        by_position = {}
        for obj in objects:
            pos = _obj_pos(obj)
            if pos is not None:
                by_position.setdefault(pos, []).append(obj)
        infos = []
        for pos in _feature_positions(terrain, "P"):
            obj = None
            for candidate in by_position.get(pos, []):
                if _obj_name(candidate) == "soup":
                    obj = candidate
                    break
            ingredients = _ingredient_names(obj)
            recipe, missing = _recipe_missing(ingredients, recipes)
            ready = bool(obj and obj.get("is_ready"))
            cooking = bool(obj and obj.get("is_cooking"))
            if not ready and not cooking and obj:
                # Some snapshots expose only the cooking tick flags.
                tick = obj.get("_cooking_tick", obj.get("cooking_tick", -1))
                if isinstance(tick, (int, float)) and tick >= 0 and obj.get("cook_time", -1) not in (-1, None):
                    cooking = not bool(obj.get("is_idle", False))
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
            infos.append({
                "pos": pos,
                "ingredients": ingredients,
                "recipe": recipe or [],
                "missing": missing,
                "ready": ready,
                "cooking": cooking,
                "full": full,
                "phase": phase,
                "object": obj,
            })
        return infos

    def _counter_objects(self, obs):
        terrain = _terrain(obs)
        result = []
        for obj in _objects(_state(obs)):
            name = _obj_name(obj)
            pos = _obj_pos(obj)
            if name in {"onion", "tomato", "dish", "soup"} and pos is not None and _cell(terrain, pos) == "X":
                result.append((name, pos, obj))
        return result

    def _free_counters(self, obs):
        terrain = _terrain(obs)
        occupied = {_obj_pos(obj) for obj in _objects(_state(obs)) if _obj_pos(obj) is not None}
        result = []
        for _, pos in _features(terrain, "X"):
            if pos in occupied:
                continue
            if any(_walkable(terrain, (pos[0] + dx, pos[1] + dy)) for dx, dy in DIRECTIONS.values()):
                result.append(pos)
        return result

    def _task(self, task_id, kind, target=None, ingredient=None, description="", wait_until_ready=False):
        return {
            "id": str(task_id),
            "kind": str(kind),
            "target": list(target) if target is not None else None,
            "ingredient": ingredient,
            "description": description or str(kind),
            "wait_until_ready": bool(wait_until_ready),
        }

    def _task_candidates(self, obs):
        terrain = _terrain(obs)
        state = _state(obs)
        me, partner = self._me_partner(obs)
        held = _held_name(me)
        partner_held = _held_name(partner)
        pots = self._pot_infos(obs)
        tasks = []

        if held == "soup":
            for pos in _feature_positions(terrain, "S"):
                tasks.append(self._task(
                    "deliver_%d_%d" % pos, "deliver", pos,
                    description="carry the soup to the serving window at %s and interact" % (pos,),
                ))
            return tasks or [self._task("wait", "wait", description="wait safely")]

        if held == "dish":
            ready = [pot for pot in pots if pot["ready"]]
            cooking = [pot for pot in pots if pot["cooking"]]
            for pot in ready:
                x, y = pot["pos"]
                tasks.append(self._task(
                    "pickup_ready_%d_%d" % (x, y), "pickup_ready", pot["pos"],
                    description="face the ready soup pot at %s and pick it up" % (pot["pos"],),
                ))
            for pot in cooking:
                x, y = pot["pos"]
                tasks.append(self._task(
                    "stage_dish_%d_%d" % (x, y), "stage_dish", pot["pos"],
                    description="stage by the cooking pot at %s; do not interact until ready" % (pot["pos"],),
                    wait_until_ready=True,
                ))
            return tasks or [self._task("wait", "wait", description="hold the dish safely")]

        if held in {"onion", "tomato"}:
            compatible = [pot for pot in pots if pot["missing"] and held in pot["missing"] and not pot["cooking"] and not pot["ready"]]
            compatible.sort(key=lambda pot: (len(pot["missing"]), pot["pos"]))
            for pot in compatible[:3]:
                x, y = pot["pos"]
                tasks.append(self._task(
                    "fill_%s_%d_%d" % (held, x, y), "fill_pot", pot["pos"], held,
                    description="put the held %s into the compatible pot at %s" % (held, pot["pos"]),
                ))
            if not tasks:
                for pos in self._free_counters(obs)[:3]:
                    x, y = pos
                    tasks.append(self._task(
                        "deposit_%s_%d_%d" % (held, x, y), "deposit_counter", pos, held,
                        description="put the held %s on the free counter at %s" % (held, pos),
                    ))
            return tasks or [self._task("wait", "wait", description="wait for a compatible pot or counter")]

        # Empty hands: first protect a ready soup, then launch a full idle pot.
        ready = [pot for pot in pots if pot["ready"]]
        if ready and partner_held != "dish":
            for pos in _feature_positions(terrain, "D"):
                tasks.append(self._task(
                    "fetch_dish_%d_%d" % pos, "fetch_dish", pos,
                    description="fetch a dish from the dispenser at %s for a ready soup" % (pos,),
                ))
            if tasks:
                return tasks

        full_idle = [pot for pot in pots if pot["full"] and not pot["ready"] and not pot["cooking"]]
        for pot in full_idle:
            x, y = pot["pos"]
            tasks.append(self._task(
                "start_%d_%d" % (x, y), "start_cooking", pot["pos"],
                description="start cooking the full idle pot at %s with empty hands" % (pot["pos"],),
            ))
        if tasks:
            return tasks

        # Select the nearest-to-completion idle pot as the current build target.
        buildable = [pot for pot in pots if pot["missing"] and not pot["cooking"] and not pot["ready"]]
        buildable.sort(key=lambda pot: (len(pot["missing"]), pot["pos"]))
        if buildable:
            pot = buildable[0]
            ingredient = pot["missing"][0]
            # A staged counter ingredient is preferable to a long dispenser run.
            staged = [entry for entry in self._counter_objects(obs) if entry[0] == ingredient]
            for name, pos, _ in staged[:2]:
                x, y = pos
                tasks.append(self._task(
                    "counter_%s_%d_%d" % (name, x, y), "fetch_counter", pos, name,
                    description="pick up the staged %s from counter %s" % (name, pos),
                ))
            dispensers = _feature_positions(terrain, "O" if ingredient == "onion" else "T")
            for pos in dispensers[:2]:
                x, y = pos
                tasks.append(self._task(
                    "fetch_%s_%d_%d" % (ingredient, x, y), "fetch_ingredient", pos, ingredient,
                    description="fetch one %s from dispenser %s for pot %s" % (ingredient, pos, pot["pos"]),
                ))
            if tasks:
                return tasks

        cooking = [pot for pot in pots if pot["cooking"]]
        if cooking or ready:
            for pos in _feature_positions(terrain, "D"):
                tasks.append(self._task(
                    "fetch_dish_%d_%d" % pos, "fetch_dish", pos,
                    description="fetch a dish from dispenser %s while soup cooks" % (pos,),
                ))
            if tasks:
                return tasks

        # If the partner is already carrying a dish or soup, do not duplicate
        # that trip; otherwise a dish is a useful future pipeline action.
        if partner_held not in {"dish", "soup"}:
            for pos in _feature_positions(terrain, "D"):
                tasks.append(self._task(
                    "fetch_dish_%d_%d" % pos, "fetch_dish", pos,
                    description="fetch a dish to keep the production pipeline moving",
                ))
        return tasks or [self._task("wait", "wait", description="wait safely for the partner")]

    def _compact_state(self, obs, tasks):
        me, partner = self._me_partner(obs)
        pots = self._pot_infos(obs)
        return _jsonable({
            "tick": obs.get("tick", _state(obs).get("timestep", 0)) if isinstance(obs, dict) else 0,
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {"position": _pair(me.get("position")), "facing": _pair(me.get("orientation")), "held": _held_name(me)},
            "partner": {"position": _pair(partner.get("position")), "facing": _pair(partner.get("orientation")), "held": _held_name(partner), "last_action": obs.get("partner_action_last", "")},
            "pots": [{"position": p["pos"], "phase": p["phase"], "ingredients": p["ingredients"], "missing": p["missing"]} for p in pots],
            "orders": _orders(_state(obs)),
            "recent_events": (obs.get("last_events", []) if isinstance(obs, dict) else [])[-8:],
            "candidates": [{"id": t["id"], "kind": t["kind"], "description": t["description"]} for t in tasks],
            "execution": {"stuck_ticks": self.stuck, "preferred_candidate": self.preference_id},
        })

    def _signature(self, obs, tasks):
        me, partner = self._me_partner(obs)
        return (
            _held_name(me), _held_name(partner),
            tuple((p["pos"], p["phase"], tuple(p["ingredients"])) for p in self._pot_infos(obs)),
            tuple(t["id"] for t in tasks[:3]),
        )

    def _needs_jev(self, obs, tasks):
        if self.system1_disabled or not tasks:
            return False
        tick = obs.get("tick", _state(obs).get("timestep", 0)) if isinstance(obs, dict) else 0
        sig = self._signature(obs, tasks)
        changed = sig != self.last_signature
        periodic = tick - self.last_jev_tick >= 12
        return changed or periodic

    def _ask_jev(self, obs, tasks):
        if self.system1_disabled or not tasks:
            return
        tick = obs.get("tick", _state(obs).get("timestep", 0)) if isinstance(obs, dict) else 0
        criteria = {task["id"]: task["description"] for task in tasks[:8]}
        if not criteria:
            return
        questions = {
            "job": Choice(
                "Choose one candidate job for the controlled chef. The candidate descriptions are already checked by code for the current observation. Prefer completing a ready soup, then starting a full idle pot, then filling a compatible pot or fetching a needed item. Never invent a direction or an id not listed in observation.candidates.",
                criteria,
            ),
            "yield": Noul(
                "Is yielding for one short interval safer than continuing the current job? Answer true only when the observation shows the partner is carrying soup or is directly occupying the only interaction approach; otherwise answer false.",
                {"true": "The partner has a high-priority soup/approach conflict and the controlled chef should briefly wait.", "false": "There is no explicit high-priority partner conflict."},
            ),
        }
        try:
            answers = self.ctx.decide(
                {"game": "Overcooked-AI symbolic real-time kitchen", "observation": self._compact_state(obs, tasks)},
                questions,
                timeout_s=4.0,
            )
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            self.system1_disabled = True
            return
        except (TimeoutError, RuntimeError, TypeError, ValueError):
            # Keep the deterministic motor policy alive after a late/bad answer.
            self.last_jev_tick = tick
            self.last_signature = self._signature(obs, tasks)
            return
        self.last_jev_tick = tick
        self.last_signature = self._signature(obs, tasks)
        try:
            answer = answers.get("job")
            selected = getattr(answer, "value", None)
            probability = getattr(answer, "probabilities", {}) or {}
            selected_probability = float(probability.get(selected, 0.0)) if isinstance(probability, dict) else 0.0
            valid_ids = {task["id"] for task in tasks}
            if selected in valid_ids and (selected_probability >= 0.42 or selected_probability == 0.0):
                self.preference_id = selected
                self.preference_tick = tick
            yield_answer = answers.get("yield")
            yield_value = getattr(yield_answer, "value", 0.0)
            if isinstance(yield_value, (int, float)) and yield_value >= 0.72:
                me, partner = self._me_partner(obs)
                if not _held_name(me) and _held_name(partner) in {"soup", "dish"}:
                    self.yield_ticks = max(self.yield_ticks, 2)
        except Exception:
            # Stand-in models may return arbitrary answer objects in pre-flight.
            pass

    def _poll_astra(self):
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
                hint = {}
                for key in ("preferred_task", "preferred_kind", "priority", "handoff_protocol"):
                    if isinstance(value.get(key), str):
                        hint[key] = value[key]
                self.astra_hint = hint
        except Exception:
            self.astra_hint = {}

    def _start_astra(self, obs, tasks):
        if self.started_astra or getattr(self.ctx, "system2", object()) is None:
            return
        self.started_astra = True
        compact = self._compact_state(obs, tasks)
        prompt = (
            "You are the slow strategic planner for a symbolic Overcooked-AI kitchen. "
            "The motor controller already performs legal shortest-path movement. Review this "
            "snapshot and choose a short-horizon priority for the controlled chef. Protect ready "
            "soup, launch full pots, preserve counters for handoffs, and adapt to the scripted "
            "partner. Do not output directions. Return only JSON with optional string fields "
            "preferred_task (an exact candidate id), preferred_kind, priority, and handoff_protocol.\n"
            + json.dumps(compact, separators=(",", ":"))
        )

        def job():
            if self.ctx.done:
                return ""
            try:
                result = self.ctx.think(prompt, timeout_s=120.0)
                return getattr(result, "text", "")
            except Exception:
                return ""

        try:
            self.astra_future = self.ctx.spawn(job)
        except Exception:
            self.astra_future = None

    def _choose_task(self, obs, tasks):
        if not tasks:
            return self._task("wait", "wait", description="wait safely")
        tick = obs.get("tick", _state(obs).get("timestep", 0)) if isinstance(obs, dict) else 0
        ids = {task["id"] for task in tasks}
        if self.preference_id in ids and tick - self.preference_tick <= 8:
            return next(task for task in tasks if task["id"] == self.preference_id)
        preferred_task = self.astra_hint.get("preferred_task")
        if preferred_task in ids:
            return next(task for task in tasks if task["id"] == preferred_task)
        preferred_kind = self.astra_hint.get("preferred_kind")
        if preferred_kind:
            for task in tasks:
                if task["kind"] == preferred_kind:
                    return task
        return tasks[0]

    def _escape_action(self, obs):
        terrain = _terrain(obs)
        me, partner = self._me_partner(obs)
        current = _pair(me.get("position"))
        partner_pos = _pair(partner.get("position"))
        partner_action = obs.get("partner_action_last", "") if isinstance(obs, dict) else ""
        pdx, pdy = DIRECTIONS.get(partner_action, (0, 0))
        predicted_partner = (partner_pos[0] + pdx, partner_pos[1] + pdy)
        choices = []
        for name, (dx, dy) in DIRECTIONS.items():
            nxt = (current[0] + dx, current[1] + dy)
            if not _walkable(terrain, nxt) or nxt == partner_pos or (nxt == partner_pos and predicted_partner == current):
                continue
            distance = abs(nxt[0] - partner_pos[0]) + abs(nxt[1] - partner_pos[1])
            choices.append((distance, name))
        if not choices:
            return self.action("stay")
        choices.sort(reverse=True)
        return self.action(choices[0][1])

    def _motor_action(self, obs, task):
        if self.yield_ticks > 0 and not _held_name(self._me_partner(obs)[0]):
            self.yield_ticks -= 1
            return self.action("stay")
        if self.escape_ticks > 0:
            self.escape_ticks -= 1
            return self._escape_action(obs)
        if task.get("kind") == "wait" or task.get("target") is None:
            return self.action("stay")

        target = _pair(task.get("target"))
        path, poses = self._interaction_path(obs, target)
        me, partner = self._me_partner(obs)
        current = _pair(me.get("position"))
        partner_pos = _pair(partner.get("position"))
        if path is None:
            self.escape_ticks = 2
            return self._escape_action(obs)
        pose = path[-1]
        if len(path) > 1:
            nxt = path[1]
            if nxt == partner_pos:
                self.escape_ticks = 1
                return self.action("stay")
            dx, dy = nxt[0] - current[0], nxt[1] - current[1]
            return self.action(VEC_ACTION.get((dx, dy), "stay"))

        dx, dy = target[0] - pose[0], target[1] - pose[1]
        facing = _pair(me.get("orientation"))
        desired = VEC_ACTION.get((dx, dy), "stay")
        if facing != (dx, dy):
            return self.action(desired)
        if task.get("wait_until_ready"):
            # Re-evaluating the task on the next observation will convert this
            # to pickup_ready once the soup becomes ready.
            return self.action("stay")
        return self.action("interact")

    def run(self):
        ctx = self.ctx
        while not ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            tasks = self._task_candidates(obs)
            if not self.started_astra:
                self._start_astra(obs, tasks)
            self._poll_astra()
            task = self._choose_task(obs, tasks)

            me, _ = self._me_partner(obs)
            pos = _pair(me.get("position"))
            if self.last_pos == pos and self.last_task_id == task.get("id"):
                self.stuck += 1
            else:
                self.stuck = 0
            self.last_pos = pos
            self.last_task_id = task.get("id")
            if self.stuck >= 6:
                self.escape_ticks = max(self.escape_ticks, 3)
                self.preference_id = None
                self.stuck = 0

            action = self._motor_action(obs, task)
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                # A malformed action must never be emitted, but a defensive
                # retry with the advertised noop protects stand-in adaptors.
                try:
                    ctx.act(self.action("stay"))
                except Exception:
                    return

            if self._needs_jev(obs, tasks):
                self._ask_jev(obs, tasks)
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                return


def build(ctx):
    # Constructing the controller is deliberately side-effect free; the game
    # object is available only to run_episode under the harness contract.
    class Harness:
        def __init__(self, context):
            self.ctx = context

        def run_episode(self, game):
            if not getattr(game, "tick_s", None):
                return
            KitchenController(self.ctx, game).run()

    return Harness(ctx)
