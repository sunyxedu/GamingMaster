"""A real-time Doom controller with heading frontiers and stall recovery.

Code owns a committed sweep/escape machine so ASYNC_PLAYER never waits on
reasoning.  The local heading frontier reduces repeated turns; an independent
position watchdog adds a finite diagonal departure when the player has made no
measurable progress, including while the combat controller is only turning.
Jev and Astra can adjust bounded macro hints but never own the motor clock.
"""
from __future__ import annotations

import collections
import json
import math
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


# Doom's public projection uses world units and degrees.  These are deliberately
# broad buckets; no map geometry is assumed.
MOVE_ACTIONS = {
    "forward", "backward", "strafe_left", "strafe_right",
    "forward_left", "forward_right", "backward_left", "backward_right",
    "fire_forward", "fire_strafe_left", "fire_strafe_right",
}
TURN_ACTIONS = {"turn_left", "turn_right"}
HITSCAN = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS",
           "SpiderMastermind"}
MAJOR = {"Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind",
         "Revenant"}
MELEE = {"Demon", "Spectre", "LostSoul", "Pinkie", "Pinky"}


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    try:
        return int(number(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def wrap(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_observation(raw):
    """Make malformed stand-in observations harmless and JSON-friendly."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    player = raw.get("player")
    if not isinstance(player, dict):
        player = {}
    # Copy only ordinary values.  The adaptor normally supplies the full dict;
    # the defaults also make the pre-flight stand-in safe.
    result = dict(raw)
    result["player"] = dict(player)
    for key in ("x", "y", "z", "angle"):
        result["player"][key] = number(player.get(key))
    labels = raw.get("visible")
    result["visible"] = labels if isinstance(labels, list) else []
    return result


def position(obs):
    p = obs.get("player") or {}
    return (number(p.get("x")), number(p.get("y")), number(p.get("z")))


def cell(point):
    # 64-unit cells are a memory of where the player has actually been, not a
    # claim that Doom's sectors have this size.
    return (math.floor(point[0] / 64.0), math.floor(point[1] / 64.0),
            math.floor(point[2] / 64.0))


def heading_bin(angle):
    return int(math.floor((angle % 360.0 + 22.5) / 45.0)) % 8


def label_objects(obs):
    """Translate visible labels into derived facts for code and Jev."""
    p = position(obs)
    angle = number((obs.get("player") or {}).get("angle")) % 360.0
    objects = []
    for label in obs.get("visible", []):
        if not isinstance(label, dict):
            continue
        world = label.get("world")
        if not isinstance(world, (list, tuple)) or len(world) < 3:
            continue
        q = (number(world[0]), number(world[1]), number(world[2]))
        name = str(label.get("name", label.get("object_name", "unknown")))[:80]
        category = str(label.get("category", label.get("object_category", "Unknown")))[:40]
        lower_category = category.lower()
        lower_name = name.lower()
        if lower_category == "self" or name == "DoomPlayer":
            continue
        d = dist(p, q)
        bearing = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 360.0
        obj = {
            "id": str(label.get("id", label.get("object_id", name))),
            "name": name,
            "category": category,
            "pos": q,
            "distance": d,
            "height": q[2] - p[2],
            "error": wrap(bearing - angle),
            "bearing": bearing,
        }
        obj["monster"] = lower_category == "monster" and not lower_name.startswith("dead")
        obj["projectile"] = ("projectile" in lower_category or
                             "missile" in lower_category or
                             lower_name in {"rocket", "plasmafire", "fireball"})
        obj["interact"] = (
            lower_category in {"exit", "switch", "door", "teleporter", "teleport"}
            or any(token in lower_name for token in ("exit", "switch", "door", "teleport"))
        )
        obj["key"] = lower_category == "key" or "key" in lower_name
        obj["pickup"] = lower_category in {
            "weapon", "health", "armor", "ammo", "key", "powerup", "power-up",
        }
        obj["hitscan"] = obj["monster"] and name in HITSCAN
        obj["major"] = obj["monster"] and name in MAJOR
        obj["melee"] = obj["monster"] and name in MELEE
        objects.append(obj)
    return objects


def answer_choice(answers, key, minimum=0.58):
    if not isinstance(answers, dict):
        return None
    answer = answers.get(key)
    value = getattr(answer, "value", None)
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(value, str):
        return None
    # Stand-in answers sometimes omit probabilities.  A legal typed value is
    # still safe here because it only changes a future macro boundary.
    if isinstance(probabilities, dict):
        if number(probabilities.get(value), 0.0) < minimum:
            return None
    confidence = getattr(answer, "confidence", None)
    if confidence is not None and number(confidence, 0.0) < 0.10:
        return None
    return value


def answer_yes(answers, key, minimum=0.72):
    if not isinstance(answers, dict):
        return False
    value = number(getattr(answers.get(key), "value", 0.0), 0.0)
    return value >= minimum


def json_object(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    # Astra is explicitly asked for JSON, but extraction protects the game loop
    # from prose, markdown fences, and arbitrary pre-flight text.
    for i, char in enumerate(text[:12000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            continue
    return {}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.noop = "noop"

        self.previous = None
        self.previous_position = None
        self.previous_tic = -1
        self.previous_action = "noop"
        self.turn_sign = 1.0       # positive angle change for turn_left
        self.turn_rate = 5.6
        self.blocked_ticks = 0
        self.damage_until = -100
        # Track near-zero displacement across successive observations: combat
        # alignment can otherwise issue alternating turn buttons indefinitely.
        self.stationary_tics = 0
        self.last_position_recovery = -10000
        self.position_recoveries = 0

        self.visits = collections.Counter()
        self.heading_visits = collections.Counter()
        self.trail = collections.deque(maxlen=160)
        self.last_new_tic = 0
        self.last_new_cell = None
        self.last_counts = (0, 0, 0)
        self.last_health = 100.0
        self.collision_count = 0
        self.sweep_count = 0
        self.sweep_left = 48
        self.sweep_period = 48
        self.turn_left = 0
        self.turn_side = 1           # +1 left, -1 right
        self.escape = collections.deque()
        self.preferred_turn = None
        self.novelty_break = False

        self.goal = None
        self.goal_cooldown = {}
        self.last_use_tic = -1000
        self.probed_faces = {}
        self.force_use = False
        self.last_weapon_tic = -1000
        self.fire_side = 1
        self.last_target_id = None
        self.last_fire_tic = -1000

        # Jev is a semantic adviser, never the motor clock.  Astra is a
        # deliberately bounded macro-policy author, not a per-tic oracle.
        self.jev_future = None
        self.jev_request = None
        self.next_jev_tic = 0
        self.jev_intent = "advance"
        self.jev_side = None
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.next_astra_tic = 35
        self.policy = {
            "hand": "left",
            "sweep": "medium",
            "combat": "clear_blockers",
        }
        self.history = collections.deque(maxlen=20)

    # ---------- compact facts and typed-model bridge ----------
    def legal(self, action):
        return action if action in self.allowed else self.noop

    def turn_action(self, side):
        # Doom's sign is learned from actual observations after the first turn;
        # the fallback is the ordinary angle convention.
        wants_positive_angle = side > 0
        left_makes_positive = self.turn_sign > 0
        if wants_positive_angle == left_makes_positive:
            return "turn_left"
        return "turn_right"

    def derived_objects(self, obs):
        return label_objects(obs)

    def threats(self, objects):
        result = []
        for item in objects:
            if not item["monster"]:
                continue
            score = 0.0
            score += 130.0 if item["hitscan"] else 0.0
            score += 120.0 if item["major"] else 0.0
            score += 80.0 if item["melee"] else 0.0
            score += 150.0 if item["distance"] < 115 else 60.0 if item["distance"] < 260 else 0.0
            score -= item["distance"] * 0.18
            score -= abs(item["error"]) * 0.10
            item["threat_score"] = score
            result.append(item)
        return sorted(result, key=lambda x: x["threat_score"], reverse=True)

    def goal_value(self, item, obs):
        category = item["category"].lower()
        name = item["name"].lower()
        health = number(obs.get("health"), 100)
        armor = number(obs.get("armor"), 0)
        ammo = number(obs.get("weapon_ammo"), 0)
        if item["interact"]:
            if category == "exit" or "exit" in name:
                return 10000.0
            if category in {"switch", "door", "teleporter", "teleport"} or any(
                    x in name for x in ("switch", "door", "teleport")):
                return 7000.0
            return 5000.0
        if item["key"]:
            return 3500.0
        if category == "weapon":
            return 1400.0 if number(obs.get("weapon_ammo"), 0) < 12 else 500.0
        if category == "health":
            return 1600.0 if health < 55 else 450.0 if health < 85 else 0.0
        if category == "armor":
            return 1300.0 if armor < 25 else 350.0 if armor < 75 else 0.0
        if category == "ammo":
            return 950.0 if ammo < 8 else 250.0
        return 0.0

    def current_goal(self, objects, obs, tic):
        candidates = []
        for item in objects:
            if item["monster"] or item["projectile"]:
                continue
            if abs(item["height"]) > 88 or item["distance"] > 900:
                continue
            if self.goal_cooldown.get(item["id"], -1) > tic:
                continue
            value = self.goal_value(item, obs)
            if value <= 0:
                continue
            value -= item["distance"] * (0.06 if item["interact"] else 0.11)
            candidates.append((value, item))
        if not candidates:
            # A briefly remembered interaction is useful while a label flickers,
            # but pickups are intentionally forgotten once not rendered.
            if self.goal and self.goal.get("kind") == "interact" and tic - self.goal["started"] < 25:
                return self.goal
            self.goal = None
            return None
        _, item = max(candidates, key=lambda pair: pair[0])
        old = self.goal
        if old and old.get("id") == item["id"]:
            d = item["distance"]
            if d < old["best"] - 8:
                old["best"] = d
                old["progress"] = tic
            old["pos"] = item["pos"]
            old["item"] = item
            if tic - old["started"] > 140 or tic - old["progress"] > 55:
                self.goal_cooldown[old["id"]] = tic + 280
                self.goal = None
                return None
            return old
        self.goal = {
            "id": item["id"], "kind": "interact" if item["interact"] else "pickup",
            "pos": item["pos"], "item": item, "started": tic,
            "progress": tic, "best": item["distance"],
        }
        self.ctx.log("macro_goal", kind=self.goal["kind"], name=item["name"], distance=round(item["distance"]))
        return self.goal

    def compact_object(self, item):
        d = item["distance"]
        return {
            "id": item["id"], "name": item["name"], "category": item["category"],
            "range": "contact" if d < 80 else "near" if d < 220 else "medium" if d < 550 else "far",
            "direction": "center" if abs(item["error"]) < 12 else "left" if item["error"] > 0 else "right",
            "level": "same" if abs(item["height"]) <= 40 else "different",
            "hitscan": bool(item["hitscan"]), "major": bool(item["major"]),
            "body_blocker": bool(item["melee"] and d < 150),
        }

    def jev_state(self, obs, objects, threats, tic):
        p = position(obs)
        c = cell(p)
        recent_damage = tic <= self.damage_until
        goal = self.goal
        return {
            "task": "Reach and activate the exit alive; optional kills/items are secondary.",
            "rules": "Only currently rendered labels are visible. Code handles bearings, collision, ammo legality, and the 35 Hz motor. Choose a high-level hint, not a button or a route.",
            "health": "critical" if number(obs.get("health"), 100) < 25 else "wounded" if number(obs.get("health"), 100) < 60 else "healthy",
            "armor": "none" if number(obs.get("armor"), 0) < 10 else "present",
            "damage_recent": recent_damage,
            "motion": {
                "phase": "escape" if self.escape else "turning" if self.turn_left else "sweep",
                "blocked": self.blocked_ticks >= 3,
                "novelty": "new_cell_recently" if tic - self.last_new_tic < 25 else "revisiting",
                "visited_cells": min(255, len(self.visits)),
                "same_cell_visits": min(12, self.visits.get(c, 0)),
            },
            "threats": [self.compact_object(x) for x in threats[:4]],
            "local_goals": [self.compact_object(x) for x in sorted(
                [o for o in objects if not o["monster"] and not o["projectile"]],
                key=lambda x: (0 if x["interact"] else 1, x["distance"]))[:4]],
            "goal": {"kind": goal.get("kind"), "range": self.compact_object(goal["item"])["range"]} if goal else None,
            "macro": {
                "hand": self.policy["hand"], "sweep": self.policy["sweep"],
                "turn_bias": "left" if self.turn_side > 0 else "right",
                "previous_action": self.previous_action,
            },
        }

    def make_questions(self, state, reverse=False):
        route = {
            "advance": "No immediate danger or obstruction is established: keep the committed sweep and seek new space.",
            "turn_left": "At the next safe macro boundary, turn left to expose a different view or leave a repeated cell.",
            "turn_right": "At the next safe macro boundary, turn right to expose a different view or leave a repeated cell.",
            "use": "A rendered door, switch, teleporter or exit, or a blocked face, makes one USE pulse worthwhile.",
            "retreat": "Recent damage or a close threat makes a short retreat/escape safer than advancing.",
            "unknown": "The facts do not support a semantic change; let the deterministic fallback continue.",
        }
        if reverse:
            route = dict(reversed(list(route.items())))
        return {
            "route": Choice(
                "Using only the named facts in state.motion, state.threats and state.local_goals, choose one high-level hint. Do not invent a door, key, map route, distance comparison, or action that is not described.",
                route,
            ),
            "dodge": Choice(
                "If a dodge is needed, choose a side that preserves motion around the listed threat. This answer only biases the next strafing direction.",
                {
                    "left": "Strafe/circle to the player's left while keeping a retreat lane.",
                    "right": "Strafe/circle to the player's right while keeping a retreat lane.",
                    "back": "Back away from a contact or body-blocking threat.",
                    "none": "No special dodge is justified by the listed facts.",
                },
            ),
            "open": Noul(
                "Do the current visible facts justify issuing one USE pulse soon, rather than waiting for a model plan?",
                {"true": "A visible interaction or a blocked face makes USE plausibly useful.",
                 "false": "No visible interaction or useful blocked-face evidence is present."},
            ),
        }

    def dispatch_jev(self, obs, objects, threats, tic):
        if self.ctx.system1 is None or self.jev_future is not None or tic < self.next_jev_tic:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if remaining.get("system1_calls", 0) <= 8 or remaining.get("wall_s", 0) < 3:
            return
        state = self.jev_state(obs, objects, threats, tic)
        questions = self.make_questions(state, bool(self.heading_visits.total() % 2))
        self.jev_request = {"tic": tic, "state": state}
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=1.35)
            )
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
            return
        # A fast semantic hint is useful at most about once per second.  The
        # motor remains fully code-driven while this future is pending.
        self.next_jev_tic = tic + (22 if threats else 52)

    def poll_jev(self, obs, tic):
        if self.jev_future is None or not self.jev_future.done():
            return
        future, request = self.jev_future, self.jev_request
        self.jev_future = None
        self.jev_request = None
        try:
            answers = future.result()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, EpisodeOver, CancelledError):
            return
        except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
            self.ctx.log("jev_late", error=type(exc).__name__)
            return
        if not request:
            return
        age = tic - integer(request.get("tic"), tic)
        if age > 55:
            self.ctx.log("jev_stale", age_tics=age)
            return
        route = answer_choice(answers, "route", 0.57)
        dodge = answer_choice(answers, "dodge", 0.54)
        if route in {"turn_left", "turn_right"}:
            self.preferred_turn = route
        elif route == "use":
            self.force_use = True
        elif route == "retreat" and not self.escape and not self.threats(label_objects(obs)):
            self.start_escape(tic, "jev_retreat", obs)
        if dodge in {"left", "right"}:
            self.fire_side = 1 if dodge == "left" else -1
        if answer_yes(answers, "open"):
            self.force_use = True
        self.jev_intent = route or self.jev_intent
        self.ctx.log("jev_macro_hint", route=route, dodge=dodge, use=self.force_use, age_tics=age)

    def launch_astra(self, obs, tic):
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if remaining.get("system2_calls", 0) <= 0 or remaining.get("wall_s", 0) < 120:
            return
        state = self.jev_state(obs, label_objects(obs), self.threats(label_objects(obs)), tic)
        payload = {
            "state": state,
            "measured_history": list(self.history),
            "collision_count": self.collision_count,
            "visited_cells": len(self.visits),
            "current_policy": self.policy,
        }
        prompt = (
            "You are Astra, the slow strategic layer of a real-time Classic Doom "
            "controller. The controller below already owns aiming, collision tests, "
            "USE timing, weapon legality, and a deterministic forward-sweep/turn/escape "
            "machine. Do not output buttons, coordinates, hidden geometry, a map-specific "
            "route, or code. Revise only the bounded macro policy from measured public "
            "facts. Exit success is primary and survival is required. A late answer is "
            "still safe because it is applied only at a future macro boundary. Return ONLY "
            "a JSON object: {\"hand\":\"left|right\", "
            "\"sweep\":\"short|medium|long\", "
            "\"combat\":\"clear_blockers|avoid_optional\", \"reason\":\"brief evidence\"}. "
            "If evidence is insufficient, repeat the current policy.\n" +
            json.dumps(payload, separators=(",", ":"), allow_nan=False)
        )
        session = self.astra_session
        self.astra_calls += 1
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, session_id=session, timeout_s=180.0)
            )
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None

    def poll_astra(self, tic):
        if self.astra_future is None or not self.astra_future.done():
            return
        future = self.astra_future
        self.astra_future = None
        try:
            result = future.result()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, EpisodeOver, CancelledError):
            return
        except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
            self.ctx.log("astra_late", error=type(exc).__name__)
            return
        self.astra_session = getattr(result, "session_id", self.astra_session)
        proposal = json_object(getattr(result, "text", ""))
        updated = dict(self.policy)
        changed = False
        if proposal.get("hand") in {"left", "right"}:
            updated["hand"] = proposal["hand"]
            changed = True
        if proposal.get("sweep") in {"short", "medium", "long"}:
            updated["sweep"] = proposal["sweep"]
            changed = True
        if proposal.get("combat") in {"clear_blockers", "avoid_optional"}:
            updated["combat"] = proposal["combat"]
            changed = True
        if changed:
            self.policy = updated
            self.turn_side = 1 if self.policy["hand"] == "left" else -1
            self.sweep_period = {"short": 32, "medium": 48, "long": 72}[self.policy["sweep"]]
            self.sweep_left = min(self.sweep_left, self.sweep_period)
            self.ctx.log("astra_macro_policy", policy=dict(self.policy), reason=str(proposal.get("reason", ""))[:240])

    # ---------- deterministic controller ----------
    def update(self, obs):
        tic = integer(obs.get("tic"), self.previous_tic + 1)
        if tic <= self.previous_tic:
            return tic, False
        p = position(obs)
        angle = number((obs.get("player") or {}).get("angle"))
        if self.previous is not None and self.previous_position is not None:
            dt = tic - self.previous_tic
            moved = dist(p, self.previous_position)
            old_angle = number((self.previous.get("player") or {}).get("angle"))
            delta = wrap(angle - old_angle)
            if self.previous_action in TURN_ACTIONS and 0 < dt <= 10 and abs(delta) > 0.15:
                rate = abs(delta) / max(1, dt)
                if 0.2 < rate < 25:
                    self.turn_rate = 0.75 * self.turn_rate + 0.25 * rate
                    if self.previous_action == "turn_left":
                        self.turn_sign = 1.0 if delta > 0 else -1.0
                    else:
                        self.turn_sign = -1.0 if delta > 0 else 1.0
            if self.previous_action in MOVE_ACTIONS and dt <= 14:
                if moved < max(2.5, dt * 0.55):
                    self.blocked_ticks += dt
                else:
                    self.blocked_ticks = 0
            else:
                self.blocked_ticks = 0
            # The short parent detector above only sees a failed translation.
            # Track continuous near-zero displacement too: combat alignment can
            # otherwise issue alternating turn buttons indefinitely at one spot.
            if moved < 0.75:
                self.stationary_tics += dt
            else:
                self.stationary_tics = 0
            old_health = number(self.previous.get("health"), self.last_health)
            health = number(obs.get("health"), old_health)
            if health < old_health - 0.1:
                self.damage_until = tic + 35
            self.last_health = health
        c = cell(p)
        was_unseen = c not in self.visits
        self.visits[c] += 1
        self.heading_visits[(c, heading_bin(angle))] += 1
        # A back-and-forth across an already known cell is not exploration.
        # Only a genuinely new measured cell resets the novelty watchdog.
        if was_unseen:
            self.last_new_cell = c
            self.last_new_tic = tic
        if not self.trail or dist(self.trail[-1][1], p) > 20:
            self.trail.append((tic, p))
        counts = tuple(integer(obs.get(k), 0) for k in ("kills", "items", "secrets"))
        if counts != self.last_counts:
            self.last_counts = counts
            self.last_new_tic = tic
        self.history.append({
            "tic": tic, "cell": list(c), "health": round(number(obs.get("health"), 100)),
            "kills": counts[0], "items": counts[1], "action": self.previous_action,
        })
        self.previous = obs
        self.previous_position = p
        self.previous_tic = tic
        return tic, True

    def frontier_turn_plan(self, obs, preferred=None):
        """Choose a locally underused outgoing heading for the next macro.

        The parent counted visited headings but used that counter only to order
        Jev's questions.  Here it becomes an execution-side frontier signal:
        at a repeated cell, prefer the turn whose resulting heading has been
        observed least often at that cell.  This is still purely odometric and
        does not claim that a heading is open; a collision can override it on
        the next observation.
        """
        if not isinstance(obs, dict):
            obs = self.previous if isinstance(self.previous, dict) else {}
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        angle = number(player.get("angle"))
        here = cell(position(obs))
        preferred = preferred if preferred in (-1, 1) else self.turn_side
        # Try cardinal quarter turns first, then diagonals and a reversal.
        # Counts, rather than a fixed left/right alternation, are decisive.
        options = []
        for magnitude in (2, 1, 3, 4):
            for side in (preferred, -preferred):
                # `side` is the desired mathematical angle sign; turn_action
                # maps it to the physical left/right button using turn_sign.
                delta = side * 45.0 * magnitude
                target = heading_bin(angle + delta)
                seen = self.heading_visits.get((here, target), 0)
                options.append((seen, 0 if side == preferred else 1,
                                0 if magnitude == 2 else 1, magnitude,
                                side, target))
        seen, _, _, magnitude, side, target = min(options)
        steps = max(8, int(round((45.0 * magnitude) / max(2.0, self.turn_rate))))
        return side, steps, target, seen

    def start_escape(self, tic, reason, obs=None):
        if self.escape:
            return
        side, turn_steps, target_bin, heading_seen = self.frontier_turn_plan(
            obs, self.turn_side
        )
        self.turn_side = side
        turn = self.turn_action(side)
        # The frontier selects the heading, while the diagonal motor makes the
        # departure genuinely translational.  This is important when the
        # preceding failure was an alternating turn/aim lock rather than a wall
        # push.  Fall back gracefully for reduced action specifications.
        lateral = "left" if turn == "turn_left" else "right"
        retreat = "backward_" + lateral
        depart = "forward_" + lateral
        if retreat not in self.allowed:
            retreat = "backward" if "backward" in self.allowed else turn
        if depart not in self.allowed:
            depart = "forward" if "forward" in self.allowed else turn
        seq = []
        seq.extend([retreat] * 5)
        seq.extend([turn] * turn_steps)
        seq.extend([depart] * 14)
        self.escape.extend(x for x in seq if x in self.allowed)
        self.blocked_ticks = 0
        self.stationary_tics = 0
        self.last_position_recovery = tic if reason == "position_stall" else self.last_position_recovery
        if reason == "position_stall":
            self.position_recoveries += 1
        self.collision_count += 1
        self.goal = None
        self.ctx.log(
            "macro_escape", tic=tic, reason=reason, turn=turn,
            target_heading=target_bin, heading_seen=heading_seen,
            turn_steps=turn_steps, collisions=self.collision_count,
            departure=depart, position_recoveries=self.position_recoveries,
        )

    def weapon_switch(self, obs, tic, threats):
        if tic - self.last_weapon_tic < 35:
            return None
        weapons = obs.get("weapons") if isinstance(obs.get("weapons"), dict) else {}
        ammo = obs.get("ammo") if isinstance(obs.get("ammo"), dict) else {}
        current = integer(obs.get("weapon"), 2)
        current_ammo = number(obs.get("weapon_ammo"), 0)
        desired = None
        # Workhorse weapons first.  Rockets/BFG are deliberately omitted because
        # the symbolic interface cannot certify splash clearance.
        for slot in (6, 4, 3, 2):
            if not bool(weapons.get(str(slot), False)):
                continue
            available = number(ammo.get(str(slot)), 0)
            if slot == current:
                available = max(available, current_ammo)
            if available > 0:
                desired = slot
                break
        if desired is None or desired == current:
            return None
        if current_ammo > 0 and current in (3, 4, 6):
            return None
        action = "select_weapon_" + str(desired)
        if action in self.allowed:
            self.last_weapon_tic = tic
            return action
        return None

    def combat_action(self, obs, threat, tic):
        selected = self.weapon_switch(obs, tic, [threat])
        if selected:
            return selected
        error = threat["error"]
        tolerance = max(6.0, self.turn_rate * 1.35)
        if abs(error) > tolerance:
            return self.turn_action(1 if error > 0 else -1)
        health = number(obs.get("health"), 100)
        distance = threat["distance"]
        if distance < 105 and (threat["melee"] or tic <= self.damage_until):
            return self.legal("backward_left" if self.fire_side > 0 else "backward_right")
        if number(obs.get("weapon_ammo"), 0) <= 0:
            return self.legal("backward_left" if self.fire_side > 0 else "backward_right")
        # Even routine monsters are fired at once they are close enough to be a
        # blocker; distant optional monsters are bypassed by the caller.
        self.last_fire_tic = tic
        if self.fire_side > 0:
            action = "fire_strafe_left"
        else:
            action = "fire_strafe_right"
        if action not in self.allowed:
            action = "fire_forward" if "fire_forward" in self.allowed else "fire"
        if health < 28 and distance < 180:
            return self.legal("backward_left" if self.fire_side > 0 else "backward_right")
        return self.legal(action)

    def steer_to_goal(self, goal, tic):
        item = goal.get("item") or {}
        error = number(item.get("error"), 0.0)
        if abs(error) > max(7.0, self.turn_rate * 1.25):
            return self.turn_action(1 if error > 0 else -1)
        if item.get("distance", 9999) < 48:
            if goal.get("kind") == "interact" and tic - self.last_use_tic >= 12:
                self.last_use_tic = tic
                self.force_use = False
                return self.legal("use")
            if goal.get("kind") == "interact":
                return self.legal("forward")
            self.goal = None
            return self.legal("forward")
        return self.legal("forward")

    def periodic_use(self, tic):
        # USE is a one-tic pulse.  A pulse roughly every 0.9 seconds gives doors,
        # lifts and switches a chance without holding USE in combat.
        if tic - self.last_use_tic >= 31:
            self.last_use_tic = tic
            return self.legal("use")
        return None

    def macro_action(self, tic, obs):
        if self.escape:
            return self.escape.popleft()
        if self.turn_left > 0:
            self.turn_left -= 1
            return self.turn_action(self.turn_side)
        # Force a deterministic novelty break when the player has been in the
        # same measured region too long.  This is not just a turn-in-place vote:
        # it backs away, rotates, and commits to a new translation.
        if tic - self.last_new_tic > 115 and not self.novelty_break:
            self.novelty_break = True
            self.start_escape(tic, "no_new_cell", obs)
            return self.escape.popleft() if self.escape else self.legal("backward")
        if tic - self.last_new_tic < 80:
            self.novelty_break = False
        if self.sweep_left <= 0:
            self.sweep_count += 1
            self.sweep_left = self.sweep_period
            preferred = self.turn_side
            if self.preferred_turn in {"turn_left", "turn_right"}:
                preferred = 1 if self.preferred_turn == "turn_left" else -1
            side, turn_steps, target_bin, heading_seen = self.frontier_turn_plan(
                obs, preferred
            )
            self.turn_side = side
            self.preferred_turn = None
            self.turn_left = turn_steps
            self.ctx.log(
                "macro_turn", tic=tic,
                side="left" if side > 0 else "right", count=self.sweep_count,
                target_heading=target_bin, heading_seen=heading_seen,
                turn_steps=turn_steps,
            )
            self.turn_left -= 1
            return self.turn_action(side)
        self.sweep_left -= 1
        return self.legal("forward")

    def action(self, obs, objects, threats, tic):
        # A projectile emergency is the one thing that may interrupt a lease:
        # lateral motion is cheaper than preserving a planned heading here.
        close_projectile = any(x["distance"] < 190 for x in objects if x["projectile"])
        if close_projectile:
            self.fire_side *= -1
            return self.legal("strafe_left" if self.fire_side > 0 else "strafe_right")

        # Finish a committed retreat/turn/departure before asking the semantic
        # layer or target aligner to take control again.  The old leader tested
        # this after immediate combat, which is how alternating turn actions
        # could starve an already-started escape.
        if self.escape:
            return self.escape.popleft()

        # 0003's watchdog covered a translational wall push.  This additional
        # guard covers the leader's turn-only combat lock.  Do not pull away
        # from a close rendered door/exit/switch: the normal goal path gets a
        # chance to pulse USE first.
        close_interaction = any(
            x["interact"] and x["distance"] < 150 for x in objects
        )
        if (
            self.stationary_tics >= 32
            and tic - self.last_position_recovery >= 90
            and not close_interaction
        ):
            stalled_for = self.stationary_tics
            self.start_escape(tic, "position_stall", obs)
            if self.escape:
                self.ctx.log(
                    "position_stall_recovery", tic=tic,
                    stationary_tics=stalled_for,
                    position=list(position(obs)),
                    previous_action=self.previous_action,
                )
                return self.escape.popleft()

        target = threats[0] if threats else None
        routine_limit = 180 if self.policy.get("combat") == "avoid_optional" else 260
        immediate = target is not None and (
            target["distance"] < routine_limit or (target["hitscan"] and target["distance"] < 650)
            or target["major"] or tic <= self.damage_until
        )
        if immediate:
            self.goal = None
            return self.combat_action(obs, target, tic)

        # A distant ordinary monster is not allowed to lure the route away from
        # an exit.  If it has become a blocker, the distance rule above engages it.
        goal = self.current_goal(objects, obs, tic)
        if goal is not None:
            if self.force_use and goal.get("kind") == "interact" and tic - self.last_use_tic >= 12:
                self.force_use = False
                self.last_use_tic = tic
                return self.legal("use")
            return self.steer_to_goal(goal, tic)
        if self.force_use:
            self.force_use = False
            pulse = self.periodic_use(tic)
            if pulse:
                return pulse
        if self.blocked_ticks >= 4:
            face = (cell(position(obs)), heading_bin(number((obs.get("player") or {}).get("angle"))))
            if self.probed_faces.get(face, -1) < tic and tic - self.last_use_tic > 12:
                self.probed_faces[face] = tic + 140
                self.last_use_tic = tic
                return self.legal("use")
            self.start_escape(tic, "collision", obs)
            if self.escape:
                return self.escape.popleft()
        if tic - self.last_use_tic >= 31 and not threats:
            pulse = self.periodic_use(tic)
            if pulse:
                return pulse
        return self.macro_action(tic, obs)

    # ---------- lifecycle ----------
    def run_episode(self, game):
        ctx = self.ctx
        names = getattr(game.action_spec, "names", ()) or ()
        self.allowed = set(names)
        if not self.allowed:
            return
        self.noop = getattr(game.action_spec, "noop", "noop")
        if self.noop not in self.allowed:
            self.noop = next(iter(self.allowed))
        try:
            while not ctx.done and not getattr(game, "done", False):
                obs = parse_observation(ctx.observe())
                if ctx.done or getattr(game, "done", False):
                    break
                if obs.get("terminal") or obs.get("dead"):
                    break
                tic, fresh = self.update(obs)
                if not fresh:
                    ctx.wait_for_tick()
                    continue
                objects = self.derived_objects(obs)
                threats = self.threats(objects)
                self.poll_jev(obs, tic)
                self.poll_astra(tic)
                self.dispatch_jev(obs, objects, threats, tic)
                if self.astra_future is None and tic >= self.next_astra_tic:
                    self.launch_astra(obs, tic)
                    self.next_astra_tic = tic + 2800
                action = self.action(obs, objects, threats, tic)
                action = self.legal(action)
                # A post is made on every processed engine state.  Thus a late
                # model leaves a fresh deterministic action rather than a stale
                # held action or noop.
                ctx.act(action)
                self.previous_action = action
                ctx.wait_for_tick()
        except (EpisodeOver, BudgetExceeded):
            return
        finally:
            # Inference calls are the only background work.  Do not leave a
            # controller loop behind after terminal/deadline; evaluator-owned
            # model futures may be cancelled safely.
            for future_name in ("jev_future", "astra_future"):
                future = getattr(self, future_name, None)
                if future is not None and not future.done():
                    try:
                        future.cancel()
                    except Exception:
                        pass
            try:
                ctx.memory["macro_controller_summary"] = {
                    "visited_cells": len(self.visits),
                    "collisions": self.collision_count,
                    "position_recoveries": self.position_recoveries,
                    "sweeps": self.sweep_count,
                    "policy": dict(self.policy),
                }
            except Exception:
                pass


def build(ctx):
    return Harness(ctx)
