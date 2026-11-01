"""Candidate 0010: ownership tokens and time-expanded traffic planning.

The controlled chef is a symbolic executor.  It never asks a model for a
primitive direction.  Instead it infers a short-lived *physical ownership*
claim from the partner's carried object and recent motion, then plans a route
through a small time-expanded occupancy graph.  The token is released on an
observable pot/hand transition (or after a short no-progress bound), rather
than being an 18-tick wait lease.  This is intended to stop both chefs from
contesting the same one-cell approach while retaining a bounded takeover.
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
VEC_ACTION = {value: key for key, value in DIRECTIONS.items()}
FEATURES = {"O", "T", "D", "P", "S"}
INGREDIENTS = {"onion", "tomato"}


def _pair(value, default=None):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            if isinstance(value[0], bool) or isinstance(value[1], bool):
                return default
            return (int(value[0]), int(value[1]))
        except Exception:
            return default
    return default


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _state(obs):
    return obs.get("state", {}) if isinstance(obs, dict) and isinstance(obs.get("state"), dict) else {}


def _terrain(obs):
    value = obs.get("terrain_mtx", []) if isinstance(obs, dict) else []
    return value if isinstance(value, list) else []


def _tick(obs):
    value = obs.get("tick", _state(obs).get("timestep", 0)) if isinstance(obs, dict) else 0
    try:
        return int(value)
    except Exception:
        return 0


def _cell(terrain, point):
    point = _pair(point)
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


def _walkable(terrain, point):
    return _cell(terrain, point) not in FEATURES and _cell(terrain, point) != "X"


def _feature_positions(terrain, symbol):
    result = []
    for y, row in enumerate(terrain):
        if not isinstance(row, (list, tuple, str)):
            continue
        for x, value in enumerate(row):
            if value == symbol:
                result.append((x, y))
    return result


def _neighbors(point):
    x, y = point
    # Stable order is useful for reproducible tie-breaking.  It is not used
    # as a model option order.
    for action in ("north", "south", "west", "east"):
        delta = DIRECTIONS[action]
        yield (x + delta[0], y + delta[1]), delta


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
    other = 1 - index if len(players) >= 2 else index
    me = players[index] if index < len(players) and isinstance(players[index], dict) else {}
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
    return str(obj.get("name", "") or "").lower()


def _object_pos(obj):
    return _pair(obj.get("position")) if isinstance(obj, dict) else None


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
        value = str(value).lower() if value is not None else ""
        if value in INGREDIENTS:
            result.append(value)
    return result


def _recipes(obs):
    values = _state(obs).get("all_orders", [])
    if not isinstance(values, list):
        values = []
    result = []
    for value in values:
        raw = value.get("ingredients") if isinstance(value, dict) else value
        if not isinstance(raw, list):
            continue
        recipe = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("name", "")
            item = str(item).lower() if item is not None else ""
            if item in INGREDIENTS:
                recipe.append(item)
        if recipe:
            result.append(recipe)
    return result or [["onion", "onion", "onion"]]


def _missing(ingredients, recipes):
    have = Counter(ingredients)
    best_recipe = None
    best_missing = None
    for recipe in recipes:
        wanted = Counter(recipe)
        if any(have[name] > wanted[name] for name in have):
            continue
        current = Counter(ingredients)
        missing = []
        for name in recipe:
            if current[name] < wanted[name]:
                missing.append(name)
                current[name] += 1
        if best_recipe is None or len(missing) < len(best_missing):
            best_recipe = list(recipe)
            best_missing = missing
    if best_recipe is None:
        return [], []
    return best_recipe, best_missing


def _counter_items(obs):
    terrain = _terrain(obs)
    result = []
    for obj in _objects(obs):
        point = _object_pos(obj)
        name = _object_name(obj)
        if point is not None and name and _cell(terrain, point) == "X":
            result.append((name, point))
    return result


def _counter_sites(obs):
    terrain = _terrain(obs)
    occupied = {_object_pos(obj) for obj in _objects(obs) if _object_pos(obj) is not None}
    result = []
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


def _pot_infos(obs):
    terrain = _terrain(obs)
    recipes = _recipes(obs)
    soup_by_position = {}
    for obj in _objects(obs):
        point = _object_pos(obj)
        if point is not None and _object_name(obj) == "soup":
            soup_by_position[point] = obj
    result = []
    for point in _feature_positions(terrain, "P"):
        soup = soup_by_position.get(point)
        ingredients = _ingredients(soup)
        recipe, missing = _missing(ingredients, recipes)
        ready = bool(soup and soup.get("is_ready"))
        cooking = bool(soup and soup.get("is_cooking"))
        cooking_tick = soup.get("_cooking_tick", soup.get("cooking_tick", -1)) if soup else -1
        cook_time = soup.get("cook_time", -1) if soup else -1
        if not ready and not cooking:
            try:
                cooking = int(cooking_tick) >= 0 and int(cook_time) >= 0
            except Exception:
                cooking = False
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
        if cooking:
            try:
                remaining = max(0, int(cook_time) - int(cooking_tick))
            except Exception:
                remaining = None
        result.append({
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
    return result


def _bfs(terrain, start, goal, blocked=()):
    start = _pair(start)
    goal = _pair(goal)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    blocked = set(blocked)
    blocked.discard(start)
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        for nxt, _ in _neighbors(current):
            if nxt in previous or nxt in blocked or not _walkable(terrain, nxt):
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
    target = _pair(target)
    if target is None:
        return []
    result = []
    for delta in DIRECTIONS.values():
        pose = (target[0] - delta[0], target[1] - delta[1])
        if _walkable(terrain, pose) and pose not in result:
            result.append(pose)
    return result


def _distance(a, b):
    a, b = _pair(a), _pair(b)
    if a is None or b is None:
        return 999
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pot_signature(pot):
    return (pot.get("pos"), pot.get("phase"), tuple(pot.get("ingredients", [])))


class OwnershipKitchen:
    """Realtime controller with an explicit partner token and route reservations."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        spec = getattr(game, "action_spec", None)
        self.names = set(getattr(spec, "names", []) or [])
        self.noop = getattr(spec, "noop", None) or "stay"
        if self.noop not in self.names:
            self.noop = "stay" if "stay" in self.names else (next(iter(self.names), "stay"))

        self.job = None
        self.route = []
        self.pose = None
        self.epoch = 0
        self.last_action = self.noop
        self.last_self_position = None
        self.last_job_key = None
        self.blocked_ticks = 0
        self.conflict_ticks = 0
        self.retreat_ticks = 0
        self.retreat_goal = None
        self.last_signature = None
        self.last_model_tick = -999
        self.model_epoch = -1
        self.jev_future = None
        self.jev_protocol = None
        self.jev_until = -1
        self.astra_future = None
        self.astra_started = False
        self.astra_style = "balanced"
        self.astra_lane = "none"

        # Recent physical observations are the evidence for partner intent.
        self.partner_history = deque(maxlen=8)
        # pos -> {signature, last_seen, evidence}.  These are ownership tokens,
        # not an unconditional timer: an unchanged token is discarded unless
        # current hand/motion evidence still supports it.
        self.partner_tokens = {}

        self.system1_disabled = getattr(ctx, "system1", object()) is None

    # ---------- defensive model bridge ---------------------------------

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _compact(self, obs, tasks):
        me, partner = _actors(obs)
        partner_pos = _pair(partner.get("position"))
        return _jsonable({
            "game": "Overcooked-AI symbolic realtime kitchen",
            "tick": _tick(obs),
            "layout": obs.get("layout_name", "") if isinstance(obs, dict) else "",
            "self": {
                "position": _pair(me.get("position")),
                "facing": _pair(me.get("orientation")),
                "held": _held_name(me),
            },
            "partner": {
                "position": partner_pos,
                "facing": _pair(partner.get("orientation")),
                "held": _held_name(partner),
                "last_action": obs.get("partner_action_last", "") if isinstance(obs, dict) else "",
                "recent_positions": [item[1] for item in list(self.partner_history)[-4:]],
            },
            "pots": [{
                "position": p["pos"],
                "phase": p["phase"],
                "ingredients": p["ingredients"],
                "missing": p["missing"],
                "cook_ticks_remaining": p["remaining"],
                "partner_token": self._token_active(p["pos"], obs),
            } for p in _pot_infos(obs)],
            "orders": _recipes(obs),
            "job": {
                "kind": (self.job or {}).get("kind", ""),
                "target": (self.job or {}).get("target"),
                "epoch": self.epoch,
                "blocked_ticks": self.blocked_ticks,
                "conflict_ticks": self.conflict_ticks,
            },
            "candidates": [{
                "id": t.get("id"), "kind": t.get("kind"),
                "target": t.get("target"), "description": t.get("description", ""),
            } for t in tasks[:8]],
            "protocol_hint": self.jev_protocol or self.astra_style,
        })

    def _start_astra(self, obs, tasks):
        if self.astra_started or getattr(self.ctx, "system2", object()) is None:
            self.astra_started = True
            return
        self.astra_started = True
        prompt = (
            "Act as the slow strategic reviewer for a symbolic Overcooked-AI "
            "realtime kitchen. Exact recipes, legality, ownership evidence, and "
            "primitive movement belong to code. Choose only a tie-break protocol "
            "for the controlled chef. `partner_first` means respect an observed "
            "partner handoff; `self_first` means keep a ready-soup reservation; "
            "`balanced` means no preference. Choose a circulation lane only when "
            "it is clearly useful. Return ONLY JSON: {\"style\":\"partner_first|self_first|balanced\","
            "\"lane\":\"left|right|none\"}. Never emit directions or new tasks.\n"
            + json.dumps(self._compact(obs, tasks), separators=(",", ":"))
        )

        def ask():
            if self.ctx.done:
                return ""
            try:
                result = self.ctx.think(prompt, timeout_s=75.0)
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
            if value.get("style") in {"partner_first", "self_first", "balanced"}:
                self.astra_style = value["style"]
            if value.get("lane") in {"left", "right", "none"}:
                self.astra_lane = value["lane"]
            self._log("astra_protocol", tick=_tick(obs), style=self.astra_style, lane=self.astra_lane)
        except Exception:
            return

    def _launch_jev(self, obs, tasks, reason):
        if self.system1_disabled or self.jev_future is not None or not tasks:
            return
        now = _tick(obs)
        if now - self.last_model_tick < 8 and reason != "hard_conflict":
            return
        self.last_model_tick = now
        self.model_epoch = self.epoch
        state = self._compact(obs, tasks)
        base = {
            "continue_job": "Keep the code-selected job and preserve its current route reservation.",
            "yield_partner": "The partner has a current physical claim or higher-priority carried object; move to a safe parking cell and let that transaction progress.",
            "detour": "Keep the same job but use a different legal approach rather than entering a predicted partner cell.",
            "take_ready": "A ready soup or held soup has priority over ordinary supply work.",
        }
        # Reorder the same closed world on alternate calls to counter first-item
        # bias.  The bridge requires agreement between two independent views.
        keys = list(base)
        if (now // 8) % 2:
            keys.reverse()
        criteria_a = {key: base[key] for key in keys}
        criteria_b = {key: base[key] for key in reversed(keys)}
        questions = {
            "protocol_a": Choice(
                "Using only the derived `partner`, `pots`, `job`, and `candidates` fields, select the safest traffic protocol. This is advisory: never invent a primitive action or candidate.",
                criteria_a,
            ),
            "protocol_b": Choice(
                "Select the safest protocol from the exact same closed-world options. Treat a partner carrying soup/dish or owning a pot as evidence for yielding; otherwise preserve a useful self job.",
                criteria_b,
            ),
            "partner_claim": Noul(
                "Does the current observation support respecting a partner ownership token for the current pot? True requires a matching carried ingredient/dish or recent approach/interact evidence; an empty distant partner is false.",
                {
                    "true": "The partner's current hand or recent motion identifies a concrete pot transaction.",
                    "false": "There is no concrete current partner transaction; code may take over after revalidation.",
                },
            ),
        }

        def ask():
            if self.ctx.done:
                return {}
            try:
                return self.ctx.decide(state, questions, timeout_s=1.1)
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
        if not isinstance(answers, dict) or self.model_epoch != self.epoch:
            return
        a = answers.get("protocol_a")
        b = answers.get("protocol_b")
        av = getattr(a, "value", None)
        bv = getattr(b, "value", None)
        allowed = {"continue_job", "yield_partner", "detour", "take_ready"}
        if av in allowed and av == bv:
            pa = getattr(a, "probabilities", {}) or {}
            pb = getattr(b, "probabilities", {}) or {}
            try:
                p = min(float(pa.get(av, 0.0)), float(pb.get(bv, 0.0)))
            except Exception:
                p = 0.0
            if p >= 0.58 or (not pa and not pb):
                self.jev_protocol = av
                self.jev_until = _tick(obs) + 8
                self._log("jev_protocol", tick=_tick(obs), value=av, probability=p)
        claim_answer = answers.get("partner_claim")
        claim_value = getattr(claim_answer, "value", 0.0)
        try:
            claim_probability = float(claim_value)
        except Exception:
            claim_probability = 0.0
        if claim_probability >= 0.72 and self._any_partner_token(obs):
            # This only affects low-priority movement.  A carried soup or a
            # legal ready pickup remains code-owned and cannot be yielded away.
            me, _ = _actors(obs)
            if _held_name(me) not in {"soup"} and self._job_priority(self.job) >= 3:
                self.jev_protocol = "yield_partner"
                self.jev_until = _tick(obs) + 4

    # ---------- partner ownership evidence -----------------------------

    def _partner_token_evidence(self, obs, pot):
        me, partner = _actors(obs)
        ppos = _pair(partner.get("position"))
        held = _held_name(partner)
        phase = pot["phase"]
        if held in INGREDIENTS and held in pot["missing"]:
            # A matching item is a concrete reservation.  Select only the pot
            # that is nearest to the partner among compatible candidates below.
            return "matching_ingredient"
        if held == "dish" and phase in {"cooking", "ready"}:
            return "dish_for_soup"
        if held == "soup":
            return "partner_delivery"
        if ppos is None:
            return ""
        near = _distance(ppos, pot["pos"]) <= 1
        last_action = str(obs.get("partner_action_last", "")) if isinstance(obs, dict) else ""
        if near and last_action == "interact" and phase in {"partial", "full_idle", "ready"}:
            return "recent_pot_interact"
        # If the last two recorded moves reduce the distance to this pot and
        # the partner is empty-handed, treat it as a tentative token.  It is
        # intentionally short-lived and cannot suppress a takeover forever.
        history = list(self.partner_history)
        if len(history) >= 3 and not held:
            d0 = _distance(history[-3][1], pot["pos"])
            d1 = _distance(history[-2][1], pot["pos"])
            d2 = _distance(history[-1][1], pot["pos"])
            if d2 <= d1 < d0 and phase in {"partial", "full_idle"}:
                return "approaching_pot"
        return ""

    def _partner_claim_pot(self, obs, pots):
        _, partner = _actors(obs)
        ppos = _pair(partner.get("position"))
        held = _held_name(partner)
        candidates = []
        for pot in pots:
            evidence = self._partner_token_evidence(obs, pot)
            if not evidence:
                continue
            if held in INGREDIENTS and held not in pot["missing"]:
                continue
            if held == "dish" and pot["phase"] not in {"cooking", "ready"}:
                continue
            if held == "soup":
                continue
            approach_lengths = []
            for approach in _interaction_poses(_terrain(obs), pot["pos"]):
                approach_path = _bfs(_terrain(obs), ppos, approach)
                if approach_path is not None:
                    approach_lengths.append(len(approach_path))
            distance = min(approach_lengths) if approach_lengths else 999
            # A direct near/interact observation wins over a speculative route.
            evidence_rank = 0 if evidence in {"matching_ingredient", "dish_for_soup", "recent_pot_interact"} else 1
            candidates.append((evidence_rank, distance, pot["pos"], evidence))
        if not candidates:
            return None, ""
        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        return candidates[0][2], candidates[0][3]

    def _token_active(self, pos, obs):
        token = self.partner_tokens.get(_pair(pos))
        if not token:
            return False
        pots = {p["pos"]: p for p in _pot_infos(obs)}
        if _pair(pos) not in pots:
            return False
        return True

    def _any_partner_token(self, obs):
        return any(self._token_active(pos, obs) for pos in self.partner_tokens)

    def _partner_delivery(self, obs):
        _, partner = _actors(obs)
        held = _held_name(partner)
        return held in {"dish", "soup"}

    # ---------- task construction ---------------------------------------

    def _task(self, kind, target=None, item="", description="", wait=False, owner=None):
        target = _pair(target)
        suffix = "_".join(str(v) for v in target) if target is not None else "none"
        ident = "%s_%s%s" % (kind, item, ("_" + suffix) if suffix else "")
        return {
            "id": ident,
            "kind": kind,
            "target": target,
            "item": item,
            "description": description or kind,
            "wait": bool(wait),
            "owner": owner,
        }

    def _claimed(self, pos, obs):
        return _pair(pos) in self.partner_tokens and self._token_active(pos, obs)

    def _parking_cell(self, obs, avoid=None):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        start = _pair(me.get("position"))
        ppos = _pair(partner.get("position"))
        avoid = set(avoid or ())
        pot_poses = set()
        for pot in _pot_infos(obs):
            pot_poses.update(_interaction_poses(terrain, pot["pos"]))
        cells = []
        for y, row in enumerate(terrain):
            if not isinstance(row, (list, tuple, str)):
                continue
            for x, _ in enumerate(row):
                point = (x, y)
                if not _walkable(terrain, point) or point in avoid or point in pot_poses:
                    continue
                if ppos is not None and point == ppos:
                    continue
                path = _bfs(terrain, start, point, {ppos} if ppos is not None else set())
                if path is None:
                    continue
                # Far from the partner and from all interaction gates is a
                # better parking cell.  Route length keeps it reachable.
                score = (-(100 * _distance(point, ppos)), len(path), point)
                cells.append((score, point))
        if cells:
            cells.sort(key=lambda row: row[0])
            return cells[0][1]
        return start

    def _tasks(self, obs):
        terrain = _terrain(obs)
        me, partner = _actors(obs)
        held = _held_name(me)
        partner_held = _held_name(partner)
        pots = _pot_infos(obs)
        tasks = []

        if held == "soup":
            for target in _feature_positions(terrain, "S"):
                tasks.append(self._task("deliver", target, description="deliver held soup at serving window"))
            return tasks or [self._task("wait", wait=True, description="hold soup safely")]

        if held == "dish":
            for pot in pots:
                token = self.partner_tokens.get(pot["pos"], {})
                token_evidence = token.get("evidence", "")
                # A ready soup is an invariant: only a partner that is
                # actually holding the dish for this pot may suppress pickup.
                if pot["ready"] and (not self._claimed(pot["pos"], obs) or token_evidence != "dish_for_soup"):
                    tasks.append(self._task("pickup_ready", pot["pos"], description="pick up ready soup with held dish", owner="self"))
            if tasks:
                return tasks
            for pot in pots:
                token = self.partner_tokens.get(pot["pos"], {})
                token_evidence = token.get("evidence", "")
                if pot["cooking"] and (not self._claimed(pot["pos"], obs) or token_evidence == "recent_pot_interact"):
                    tasks.append(self._task("stage_dish", pot["pos"], description="stage dish at cooking pot until it is ready", wait=True, owner="self"))
            if tasks:
                return tasks
            # A partner with the only active pot owns the approach.  Park the
            # dish away from that gate instead of becoming a blocker.
            target = self._parking_cell(obs, avoid={p["pos"] for p in pots})
            return [self._task("park", target, description="park while partner completes the soup transaction", wait=True)]

        if held in INGREDIENTS:
            compatible = [p for p in pots if p["phase"] in {"empty", "partial"} and held in p["missing"] and not self._claimed(p["pos"], obs)]
            compatible.sort(key=lambda p: (len(p["missing"]), p["pos"]))
            for pot in compatible:
                tasks.append(self._task("fill_pot", pot["pos"], held, "put held %s into compatible pot" % held, owner="self"))
            if tasks:
                return tasks
            # Never stand at a partner-owned pot with a redundant object.
            for point in _counter_sites(obs):
                if point not in {_object_pos(obj) for obj in _objects(obs)}:
                    tasks.append(self._task("deposit_counter", point, held, "stage held %s on a free counter" % held))
            return tasks or [self._task("park", self._parking_cell(obs), held, "park while holding a surplus ingredient", wait=True)]

        # Empty hands.  A partner carrying soup/dish is a concrete delivery
        # transaction; do not duplicate its dish trip unless a ready pot is
        # clearly unclaimed and self has a better legal route.
        ready = [p for p in pots if p["ready"] and not self._claimed(p["pos"], obs)]
        if ready and partner_held not in {"dish", "soup"}:
            for target in _feature_positions(terrain, "D"):
                tasks.append(self._task("fetch_dish", target, description="fetch a dish for an unclaimed ready soup"))
            if tasks:
                return tasks

        full_idle = [p for p in pots if p["phase"] == "full_idle" and not self._claimed(p["pos"], obs)]
        for pot in full_idle:
            tasks.append(self._task("start_cooking", pot["pos"], description="explicitly start a full idle pot", owner="self"))
        if tasks:
            return tasks

        buildable = [p for p in pots if p["phase"] in {"empty", "partial"} and p["missing"] and not self._claimed(p["pos"], obs)]
        buildable.sort(key=lambda p: (len(p["missing"]), p["pos"]))
        if buildable:
            pot = buildable[0]
            ingredient = pot["missing"][0]
            for name, point in _counter_items(obs):
                if name == ingredient:
                    tasks.append(self._task("fetch_counter", point, ingredient, "retrieve staged %s" % ingredient))
            symbol = "O" if ingredient == "onion" else "T"
            for target in _feature_positions(terrain, symbol):
                tasks.append(self._task("fetch_ingredient", target, ingredient, "fetch one %s for the next pot" % ingredient))
            if tasks:
                return tasks

        cooking = [p for p in pots if p["cooking"] and not self._claimed(p["pos"], obs)]
        if cooking and partner_held not in {"dish", "soup"}:
            for target in _feature_positions(terrain, "D"):
                tasks.append(self._task("fetch_dish", target, description="fetch a dish during the cooking window"))
            if tasks:
                return tasks

        # A free dish is still a useful pipeline object only when there is an
        # active pot.  Otherwise it would block the sole pot approach.
        if cooking and partner_held not in {"dish", "soup"}:
            for target in _feature_positions(terrain, "D"):
                tasks.append(self._task("fetch_dish", target, description="stage a future dish"))
        if tasks:
            return tasks
        return [self._task("park", self._parking_cell(obs), description="park in a non-gate floor cell", wait=True)]

    def _job_priority(self, job):
        if not isinstance(job, dict):
            return 30
        return {
            "deliver": 0,
            "pickup_ready": 1,
            "start_cooking": 2,
            "fill_pot": 3,
            "fetch_counter": 4,
            "fetch_ingredient": 5,
            "fetch_dish": 6,
            "stage_dish": 7,
            "deposit_counter": 8,
            "park": 20,
            "wait": 30,
        }.get(job.get("kind"), 15)

    def _task_key(self, task):
        return (task.get("kind"), _pair(task.get("target")), task.get("item", ""))

    def _choose_task(self, obs, tasks):
        if not tasks:
            return self._task("park", self._parking_cell(obs), description="park safely", wait=True)
        me, partner = _actors(obs)
        start = _pair(me.get("position"))
        partner_pos = _pair(partner.get("position"))
        terrain = _terrain(obs)
        ranked = []
        for task in tasks:
            kind = task.get("kind")
            target = _pair(task.get("target"))
            if kind == "park":
                path = _bfs(terrain, start, target, {partner_pos} if partner_pos is not None else set())
                length = len(path) if path is not None else 999
                pose = target
            else:
                poses = _interaction_poses(terrain, target)
                routes = []
                for candidate_pose in poses:
                    route = self._time_path(obs, start, candidate_pose, horizon=14)
                    if route is not None:
                        routes.append((len(route), candidate_pose, route))
                if routes:
                    routes.sort(key=lambda row: (row[0], row[1]))
                    length, pose, path = routes[0]
                else:
                    pose, path = None, None
                    length = 999
            # Code priority is invariant.  Astra/Jev only break close ties in
            # low-priority work and cannot demote delivery/start/fill.
            rank = self._job_priority(task)
            if self.astra_style == "partner_first" and rank >= 5 and self._any_partner_token(obs):
                rank += 1
            if self.astra_style == "self_first" and rank in {5, 6}:
                rank -= 1
            lane_bias = 0
            if target is not None and self.astra_lane in {"left", "right"}:
                width = len(terrain[0]) if terrain and isinstance(terrain[0], (list, tuple, str)) else 0
                is_left = target[0] < width / 2
                if (self.astra_lane == "left") != is_left:
                    lane_bias = 1
            # Do not use the model's protocol to choose an unsafe target.
            ranked.append((rank, lane_bias, length, str(task.get("id", "")), task, pose, path))
        ranked.sort(key=lambda row: row[:4])
        chosen = ranked[0]
        task = dict(chosen[4])
        task["pose"] = chosen[5]
        task["route"] = chosen[6] or []
        return task

    def _same_job(self, first, second):
        return isinstance(first, dict) and isinstance(second, dict) and self._task_key(first) == self._task_key(second)

    def _job_valid(self, obs, tasks):
        if not self.job:
            return False
        keys = {self._task_key(task) for task in tasks}
        key = self._task_key(self.job)
        # A cooking dish reservation naturally changes into ready pickup at
        # the same pot without changing its physical pose.
        if self.job.get("kind") == "stage_dish":
            for task in tasks:
                if task.get("kind") == "pickup_ready" and _pair(task.get("target")) == _pair(self.job.get("target")):
                    self.job.update({"kind": "pickup_ready", "description": task.get("description", "")})
                    return True
        if key not in keys:
            return False
        me, _ = _actors(obs)
        held = _held_name(me)
        kind = self.job.get("kind")
        if kind == "deliver" and held != "soup":
            return False
        if kind == "fetch_dish" and held == "dish":
            return False
        if kind in {"fetch_ingredient", "fetch_counter"} and held in INGREDIENTS:
            return False
        if kind == "pickup_ready" and held == "soup":
            return False
        if kind == "deposit_counter" and not held:
            return False
        if kind == "fill_pot" and held not in INGREDIENTS:
            return False
        if kind == "start_cooking" and held:
            return False
        return True

    def _commit(self, obs, tasks):
        chosen = self._choose_task(obs, tasks)
        self.job = chosen
        self.pose = _pair(chosen.get("pose"))
        self.route = list(chosen.get("route") or [])
        self.epoch += 1
        self.last_job_key = self._task_key(chosen)
        self.blocked_ticks = 0
        self.conflict_ticks = 0
        self.retreat_ticks = 0
        self.retreat_goal = None
        self._log("ownership_job", tick=_tick(obs), kind=chosen.get("kind"), target=chosen.get("target"), pose=self.pose, epoch=self.epoch)

    def _refresh(self, obs, tasks):
        if self._job_valid(obs, tasks):
            # Recompute a route only when the current route is no longer rooted
            # at the actual player cell.  Ordinary motion does not invalidate a
            # physical reservation.
            me, _ = _actors(obs)
            current = _pair(me.get("position"))
            if self.route and current not in self.route[:2] and self.pose is not None:
                self.route = self._time_path(obs, current, self.pose, horizon=14) or self.route
            return
        self._commit(obs, tasks)

    # ---------- time-expanded routing and traffic -----------------------

    def _partner_forecast(self, obs, horizon=14):
        _, partner = _actors(obs)
        current = _pair(partner.get("position"))
        terrain = _terrain(obs)
        if current is None:
            return [None] * (horizon + 1)
        history = list(self.partner_history)
        velocity = None
        if len(history) >= 2 and history[-1][1] is not None and history[-2][1] is not None:
            a, b = history[-2][1], history[-1][1]
            velocity = (b[0] - a[0], b[1] - a[1])
            if velocity not in VEC_ACTION:
                velocity = None
        last_action = str(obs.get("partner_action_last", "")) if isinstance(obs, dict) else ""
        action_delta = DIRECTIONS.get(last_action)
        if velocity is None and action_delta is not None:
            velocity = action_delta
        forecast = [current]
        point = current
        # Forecast only a short continuation.  After four cells, hold the
        # last estimate rather than hallucinating a long partner route.
        for step in range(1, horizon + 1):
            if step <= 4 and velocity is not None:
                candidate = (point[0] + velocity[0], point[1] + velocity[1])
                if _walkable(terrain, candidate):
                    point = candidate
                else:
                    point = point
            forecast.append(point)
        return forecast

    def _time_path(self, obs, start, goal, horizon=14):
        terrain = _terrain(obs)
        start, goal = _pair(start), _pair(goal)
        if start is None or goal is None:
            return None
        if start == goal:
            return [start]
        partner_path = self._partner_forecast(obs, horizon)
        partner_now = partner_path[0] if partner_path else None
        queue = deque([(start, 0)])
        previous = {(start, 0): None}
        # Stay is a route action as well as a legal fallback.  It allows a
        # chef to wait one tick for a traffic token instead of colliding.
        move_deltas = [(0, 0)] + [DIRECTIONS[name] for name in ("north", "south", "west", "east")]
        while queue:
            current, at = queue.popleft()
            if current == goal:
                path = [current]
                key = (current, at)
                while previous[key] is not None:
                    key = previous[key]
                    path.append(key[0])
                path.reverse()
                return path
            if at >= horizon:
                continue
            next_partner = partner_path[min(at + 1, len(partner_path) - 1)] if partner_path else None
            current_partner = partner_path[min(at, len(partner_path) - 1)] if partner_path else None
            for dx, dy in move_deltas:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt != goal and not _walkable(terrain, nxt):
                    continue
                if next_partner is not None and nxt == next_partner:
                    continue
                if current_partner is not None and nxt == current_partner and next_partner == current:
                    continue
                state = (nxt, at + 1)
                if state in previous:
                    continue
                previous[state] = (current, at)
                queue.append(state)
        # A static route is preferable to refusing all progress when a noisy
        # forecast fills the small map.  The motor still checks live conflict.
        blocked = {partner_now} if partner_now is not None else set()
        return _bfs(terrain, start, goal, blocked)

    def _safe_action(self, action):
        if action in self.names:
            return action
        if self.noop in self.names:
            return self.noop
        return next(iter(self.names), "stay")

    def _partner_next(self, obs):
        forecast = self._partner_forecast(obs, horizon=2)
        return forecast[1] if len(forecast) > 1 else forecast[0]

    def _partner_priority(self, obs):
        _, partner = _actors(obs)
        held = _held_name(partner)
        pots = _pot_infos(obs)
        if held == "soup":
            return 0
        if held == "dish" and any(p["ready"] or p["cooking"] for p in pots):
            return 1
        if held in INGREDIENTS:
            return 3
        if self._any_partner_token(obs):
            return 4
        return 8

    def _retreat_action(self, obs):
        me, partner = _actors(obs)
        current = _pair(me.get("position"))
        ppos = _pair(partner.get("position"))
        if current is None:
            return self._safe_action(self.noop)
        if self.retreat_goal is None:
            self.retreat_goal = self._parking_cell(obs, avoid={ppos} if ppos is not None else set())
        if current == self.retreat_goal:
            return self._safe_action(self.noop)
        path = _bfs(_terrain(obs), current, self.retreat_goal, {ppos} if ppos is not None else set())
        if not path:
            self.retreat_goal = self._parking_cell(obs, avoid={ppos} if ppos is not None else set())
            path = _bfs(_terrain(obs), current, self.retreat_goal, {ppos} if ppos is not None else set())
        if not path or len(path) <= 1:
            return self._safe_action(self.noop)
        nxt = path[1]
        if nxt == ppos or nxt == self._partner_next(obs):
            return self._safe_action(self.noop)
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        return self._safe_action(VEC_ACTION.get(delta, self.noop))

    def _ready_for_interaction(self, obs, job):
        me, _ = _actors(obs)
        held = _held_name(me)
        target = _pair(job.get("target"))
        kind = job.get("kind")
        if kind == "deliver":
            return held == "soup"
        if kind == "fetch_ingredient":
            return not held and _cell(_terrain(obs), target) in {"O", "T"}
        if kind == "fetch_dish":
            return not held and _cell(_terrain(obs), target) == "D"
        if kind == "fetch_counter":
            return not held and any(name in INGREDIENTS and point == target for name, point in _counter_items(obs))
        if kind == "deposit_counter":
            occupied = {_object_pos(obj) for obj in _objects(obs)}
            return held in INGREDIENTS and target not in occupied
        pots = {p["pos"]: p for p in _pot_infos(obs)}
        pot = pots.get(target)
        if pot is None:
            return False
        if kind == "fill_pot":
            return held in INGREDIENTS and not pot["ready"] and not pot["cooking"] and held in pot["missing"]
        if kind == "start_cooking":
            return not held and pot["phase"] == "full_idle"
        if kind == "pickup_ready":
            return held == "dish" and pot["ready"]
        return False

    def _motor(self, obs):
        me, partner = _actors(obs)
        current = _pair(me.get("position"))
        facing = _pair(me.get("orientation"))
        partner_pos = _pair(partner.get("position"))
        if current is None or not self.job:
            return self._safe_action(self.noop)

        # A model suggestion can only influence a low-priority conflict.  The
        # current object and pot facts remain the hard safety boundary.
        if self.jev_until >= _tick(obs) and self.jev_protocol == "yield_partner":
            if _held_name(me) not in {"soup"} and self._partner_priority(obs) < self._job_priority(self.job):
                self.retreat_ticks = max(self.retreat_ticks, 2)
        if self.retreat_ticks > 0:
            self.retreat_ticks -= 1
            return self._retreat_action(obs)

        target = _pair(self.job.get("target"))
        pose = _pair(self.job.get("pose"))
        if target is None or pose is None:
            return self._safe_action(self.noop)
        # Both park and interaction jobs consume a route below.  Initialize it
        # before the branch so a freshly committed park job is safe.
        path = self.route

        if self.job.get("kind") == "park":
            if current == target:
                return self._safe_action(self.noop)
            if path and path[0] == current and len(path) > 1 and path[1] != partner_pos:
                pass
            else:
                path = self._time_path(obs, current, target, horizon=10)
                if path:
                    self.route = path
        else:
            if current == pose:
                delta = (target[0] - current[0], target[1] - current[1])
                desired = VEC_ACTION.get(delta, self.noop)
                if facing != delta and delta in VEC_ACTION:
                    return self._safe_action(desired)
                if self._ready_for_interaction(obs, self.job):
                    self.conflict_ticks = 0
                    return self._safe_action("interact")
                if self.job.get("kind") == "stage_dish":
                    # Staying at the approach is safe only while the partner
                    # has no higher-priority physical token.
                    if self._partner_priority(obs) <= 1:
                        self.conflict_ticks += 1
                        return self._retreat_action(obs)
                    return self._safe_action(self.noop)
                # If the job became temporarily impossible, hold position for
                # one tick; refresh_job will invalidate it after an object or
                # pot transition.
                return self._safe_action(self.noop)
            path = self.route
            if not path or path[0] != current or (len(path) > 1 and path[1] == partner_pos):
                path = self._time_path(obs, current, pose, horizon=14)
                if path:
                    self.route = path

        if not path or len(path) <= 1:
            self.blocked_ticks += 1
            self.conflict_ticks += 1
            if self.conflict_ticks >= 5:
                # A bounded takeover/detour prevents a bad forecast from
                # turning into the parent's long north/south oscillation.
                self.retreat_ticks = 2
                self.retreat_goal = None
                return self._retreat_action(obs)
            return self._safe_action(self.noop)

        nxt = path[1]
        forecast_next = self._partner_next(obs)
        partner_priority = self._partner_priority(obs)
        self_priority = self._job_priority(self.job)
        conflict = nxt == partner_pos or nxt == forecast_next or (forecast_next == current and nxt == partner_pos)
        if conflict:
            self.conflict_ticks += 1
            # A carrying soup or a ready pickup is not abandoned, but it still
            # waits at the current cell rather than deliberately colliding.
            if partner_priority < self_priority and _held_name(me) not in {"soup"}:
                self.retreat_ticks = max(self.retreat_ticks, 2)
                return self._retreat_action(obs)
            if self.conflict_ticks >= 5:
                # A bounded detour also breaks equal-priority soup/soup
                # crossings; waiting forever would make both chefs stale.
                self.retreat_ticks = 2
                self.retreat_goal = None
                return self._retreat_action(obs)
            return self._safe_action(self.noop)

        self.blocked_ticks = 0
        self.conflict_ticks = 0
        self.route = path[1:]
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        return self._safe_action(VEC_ACTION.get(delta, self.noop))

    # ---------- realtime loop -------------------------------------------

    def run(self):
        ctx = self.ctx
        while not ctx.done and not getattr(self.game, "done", False):
            try:
                obs = self.game.observe()
            except Exception:
                return
            if not isinstance(obs, dict):
                obs = {}
            self._observe_partner(obs)
            self._poll_astra(obs)
            self._poll_jev(obs)
            tasks = self._tasks(obs)
            if not self.astra_started:
                self._start_astra(obs, tasks)
            self._refresh(obs, tasks)

            if self.job and (self.job.get("kind") == "park" or self.blocked_ticks >= 2 or self.conflict_ticks >= 2):
                reason = "hard_conflict" if self.conflict_ticks >= 2 else "job_boundary"
                self._launch_jev(obs, tasks, reason)
            elif self.job and self.last_job_key == self._task_key(self.job) and self.epoch == self.model_epoch:
                # No request is needed on ordinary route ticks.
                pass
            elif self.job:
                self._launch_jev(obs, tasks, "new_ownership_job")

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
            self.last_self_position = _pair(_actors(obs)[0].get("position"))
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                return

    def _observe_partner(self, obs):
        self._update_partner_history_and_tokens(obs)

    def _update_partner_history_and_tokens(self, obs):
        # Kept as a separate method so malformed observations simply produce a
        # conservative empty history instead of an exception in pre-flight.
        try:
            self._observe_partner_impl(obs)
        except Exception:
            try:
                _, partner = _actors(obs)
                self.partner_history.append((_tick(obs), _pair(partner.get("position")), _held_name(partner), ""))
            except Exception:
                pass

    def _observe_partner_impl(self, obs):
        # The body is intentionally explicit: state transitions, not model
        # answers, own token creation and release.
        _, partner = _actors(obs)
        ppos = _pair(partner.get("position"))
        held = _held_name(partner)
        action = str(obs.get("partner_action_last", "")) if isinstance(obs, dict) else ""
        self.partner_history.append((_tick(obs), ppos, held, action))
        pots = _pot_infos(obs)
        signatures = {p["pos"]: _pot_signature(p) for p in pots}
        for pos, token in list(self.partner_tokens.items()):
            if signatures.get(pos) != token.get("signature"):
                self.partner_tokens.pop(pos, None)
        chosen, evidence = self._partner_claim_pot(obs, pots)
        if chosen is not None:
            self.partner_tokens[chosen] = {
                "signature": signatures.get(chosen),
                "last_seen": _tick(obs),
                "evidence": evidence,
            }
        for pos, token in list(self.partner_tokens.items()):
            evidence = token.get("evidence")
            age = _tick(obs) - int(token.get("last_seen", 0))
            # Hand-backed evidence is renewed by _partner_claim_pot while the
            # object is still in the partner's hand.  Empty-handed approach or
            # interact evidence is only a brief transaction hint.
            if evidence in {"approaching_pot", "recent_pot_interact"} and age > 4:
                self.partner_tokens.pop(pos, None)
            elif evidence in {"matching_ingredient", "dish_for_soup"} and chosen != pos:
                self.partner_tokens.pop(pos, None)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        if not getattr(game, "tick_s", None):
            return
        OwnershipKitchen(self.ctx, game).run()


def build(ctx):
    return Harness(ctx)
