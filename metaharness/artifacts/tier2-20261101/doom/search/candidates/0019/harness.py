"""Adaptive real-time controller for Doom.

The motor is deliberately code-owned.  It keeps 0011's local heading frontier,
but treats each turn/escape as an experiment whose measured result is remembered.
The moving combat lease inherited from 0015 is guarded by the same effect sensor,
so a blocked backward action cannot become a long-lived death sentence.
"""
from __future__ import annotations

import collections
import json
import math
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


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
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    try:
        return int(number(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def wrap(degrees):
    return (number(degrees) + 180.0) % 360.0 - 180.0


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_observation(raw):
    """Accept both adaptor JSON and arbitrary pre-flight stand-in values."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    out = dict(raw)
    player = raw.get("player")
    out["player"] = dict(player) if isinstance(player, dict) else {}
    for key in ("x", "y", "z", "angle"):
        out["player"][key] = number(out["player"].get(key))
    out["visible"] = raw.get("visible") if isinstance(raw.get("visible"), list) else []
    return out


def position(obs):
    p = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    return (number(p.get("x")), number(p.get("y")), number(p.get("z")))


def cell(point):
    return (math.floor(point[0] / 64.0), math.floor(point[1] / 64.0),
            math.floor(point[2] / 64.0))


def heading_bin(angle):
    return int(math.floor((number(angle) % 360.0 + 22.5) / 45.0)) % 8


def label_objects(obs):
    """Derive only view-visible, name/category/bookkeeping facts."""
    p = position(obs)
    angle = number((obs.get("player") or {}).get("angle")) % 360.0
    result = []
    for raw in obs.get("visible", []):
        if not isinstance(raw, dict):
            continue
        world = raw.get("world")
        if not isinstance(world, (list, tuple)) or len(world) < 3:
            continue
        q = (number(world[0]), number(world[1]), number(world[2]))
        name = str(raw.get("name", raw.get("object_name", "unknown")))[:80]
        category = str(raw.get("category", raw.get("object_category", "Unknown")))[:40]
        lower_name, lower_category = name.lower(), category.lower()
        if lower_category == "self" or name == "DoomPlayer":
            continue
        bearing = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 360.0
        item = {
            "id": str(raw.get("id", raw.get("object_id", name))),
            "name": name, "category": category, "pos": q,
            "distance": distance(p, q), "height": q[2] - p[2],
            "error": wrap(bearing - angle), "bearing": bearing,
        }
        item["monster"] = lower_category == "monster" and not lower_name.startswith("dead")
        item["projectile"] = (
            "projectile" in lower_category or "missile" in lower_category or
            lower_name in {"rocket", "plasmafire", "fireball", "doomimpball"}
        )
        item["interact"] = (
            lower_category in {"exit", "switch", "door", "teleporter", "teleport"}
            or any(s in lower_name for s in ("exit", "switch", "door", "teleport"))
        )
        item["key"] = lower_category == "key" or "key" in lower_name
        item["pickup"] = lower_category in {
            "weapon", "health", "armor", "ammo", "key", "powerup", "power-up"
        }
        item["hitscan"] = item["monster"] and name in HITSCAN
        item["major"] = item["monster"] and name in MAJOR
        item["melee"] = item["monster"] and name in MELEE
        result.append(item)
    return result


def answer_choice(answers, key, minimum):
    if not isinstance(answers, dict):
        return None
    answer = answers.get(key)
    value = getattr(answer, "value", None)
    if not isinstance(value, str):
        return None
    probs = getattr(answer, "probabilities", None)
    # A real typed answer without a distribution is safe only for a hint that
    # cannot post an arbitrary button.  Stand-ins are deliberately tolerated.
    if isinstance(probs, dict) and number(probs.get(value), 0.0) < minimum:
        return None
    confidence = getattr(answer, "confidence", None)
    if confidence is not None and number(confidence, 0.0) < 0.10:
        return None
    return value


def answer_yes(answers, key, minimum):
    if not isinstance(answers, dict):
        return False
    return number(getattr(answers.get(key), "value", 0.0), 0.0) >= minimum


def json_object(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for i, char in enumerate(text[:16000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            pass
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
        self.turn_sign = 1.0
        self.turn_rate = 5.6
        self.blocked_ticks = 0
        self.motion_bad_ticks = 0
        self.damage_until = -100
        self.last_health = 100.0
        self.last_counts = (0, 0, 0)
        self.contact_break_pending = False

        # The 0011 heading frontier is retained, while these counters record
        # whether a heading was actually productive rather than merely seen.
        self.visits = collections.Counter()
        self.heading_visits = collections.Counter()
        self.edge_stats = {}
        self.cell_failures = collections.Counter()
        self.trail = collections.deque(maxlen=180)
        self.history = collections.deque(maxlen=36)
        self.last_new_tic = 0
        self.last_new_cell = None
        self.same_cell_streak = 0
        self.collision_count = 0
        self.sweep_count = 0

        # Finite motor state.
        self.turn_side = 1
        self.turn_left = 0
        self.sweep_period = 48
        self.sweep_left = 48
        self.escape = collections.deque()
        self.escape_reason = ""
        self.route_mode = "frontier"
        self.trace_side = 1
        self.trace_remaining = 0
        self.trace_phase = 0
        self.trace_cooldown = 0
        self.epoch = None
        self.epoch_number = 0
        self.preferred_turn = None

        # Visible object/interaction memory.
        self.goal = None
        self.goal_cooldown = {}
        self.goal_turns = 0
        self.last_use_tic = -1000
        self.probed_faces = {}
        self.force_use = False
        self.last_weapon_tic = -1000
        self.fire_side = 1
        self.projectile_ticks = 0

        # 0015's moving aim, guarded by observed displacement.
        self.aim_turn_ticks = 0
        self.aim_move_left = 0
        self.aim_move_action = None
        self.aim_bad_ticks = 0

        # Jev is a fast semantic hint; it never owns the game clock.
        self.jev_future = None
        self.jev_request = None
        self.next_jev_tic = 0
        self.jev_intent = "frontier"
        self.requested_mode = None

        # Astra may author a bounded macro mode/sequence.  It is intentionally
        # more actionable than the old policy-only response, but still filtered.
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.next_astra_tic = 140
        self.astra_request_tic = -1
        self.astra_mode = None
        self.astra_plan = collections.deque()
        self.policy = {"hand": "left", "sweep": "medium", "combat": "clear_blockers"}

    # ---------- execution-safe helpers ----------
    def legal(self, action):
        if action in self.allowed:
            return action
        for fallback in ("forward", "forward_left", "strafe_left", "backward", self.noop):
            if fallback in self.allowed:
                return fallback
        return next(iter(self.allowed), self.noop)

    def turn_action(self, side):
        positive = side > 0
        if positive == (self.turn_sign > 0):
            return self.legal("turn_left")
        return self.legal("turn_right")

    def movement_choice(self, side, retreat=False, fire=False):
        if side >= 0:
            if fire:
                names = ("fire_strafe_left", "forward_left", "backward_left", "strafe_left")
            elif retreat:
                names = ("backward_left", "strafe_left", "forward_left", "forward")
            else:
                names = ("forward_left", "strafe_left", "forward", "backward_left")
        else:
            if fire:
                names = ("fire_strafe_right", "forward_right", "backward_right", "strafe_right")
            elif retreat:
                names = ("backward_right", "strafe_right", "forward_right", "forward")
            else:
                names = ("forward_right", "strafe_right", "forward", "backward_right")
        # If the last direction was measured blocked, do not immediately repeat
        # it.  This is the key distinction from the unguarded 0015 lease.
        if self.blocked_ticks >= 3 and not retreat:
            names = tuple(x for x in names if x not in {"backward_left", "backward_right"}) + ("forward",)
        return next((x for x in names if x in self.allowed), self.legal("forward"))

    # ---------- public-state derivation ----------
    def threats(self, objects):
        out = []
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
            out.append(item)
        return sorted(out, key=lambda x: x["threat_score"], reverse=True)

    def goal_value(self, item, obs):
        category = item["category"].lower()
        name = item["name"].lower()
        health = number(obs.get("health"), 100)
        armor = number(obs.get("armor"), 0)
        ammo = number(obs.get("weapon_ammo"), 0)
        if item["interact"]:
            return 10000.0 if (category == "exit" or "exit" in name) else 8000.0
        if item["key"]:
            return 5500.0
        if category == "weapon":
            weapons = obs.get("weapons") if isinstance(obs.get("weapons"), dict) else {}
            return 1800.0 if not any(bool(v) for v in weapons.values()) else 950.0
        # Unlike the parents, ordinary bonuses cannot constantly steal the route.
        if category == "health":
            return 1800.0 if health < 45 else 0.0
        if category == "armor":
            return 1500.0 if armor < 25 else 0.0
        if category == "ammo":
            return 1200.0 if ammo < 6 else 0.0
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
            # During a trace only a real lock/capability may interrupt the wall
            # experiment; the exit/key/weapon cases remain visible priorities.
            if self.route_mode != "frontier" and not (item["interact"] or item["key"] or item["category"].lower() == "weapon"):
                continue
            value -= item["distance"] * (0.06 if item["interact"] else 0.12)
            candidates.append((value, item))
        if not candidates:
            if self.goal and self.goal.get("kind") == "interact" and tic - self.goal.get("started", tic) < 25:
                return self.goal
            self.goal = None
            self.goal_turns = 0
            return None
        _, item = max(candidates, key=lambda pair: pair[0])
        old = self.goal
        if old and old.get("id") == item["id"]:
            d = item["distance"]
            if d < old.get("best", d) - 8:
                old["best"] = d
                old["progress"] = tic
            old["item"] = item
            old["pos"] = item["pos"]
            if tic - old.get("started", tic) > 140 or tic - old.get("progress", tic) > 55:
                self.goal_cooldown[old["id"]] = tic + 280
                self.goal = None
                return None
            return old
        self.goal = {"id": item["id"], "kind": "interact" if item["interact"] else "pickup",
                     "pos": item["pos"], "item": item, "started": tic,
                     "progress": tic, "best": item["distance"]}
        self.goal_turns = 0
        self.ctx.log("macro_goal", kind=self.goal["kind"], name=item["name"],
                     distance=round(item["distance"]))
        return self.goal

    def compact_object(self, item):
        d = item["distance"]
        direction = "center" if abs(item["error"]) < 12 else "left" if item["error"] > 0 else "right"
        return {
            "id": item["id"], "name": item["name"], "category": item["category"],
            "range": "contact" if d < 80 else "near" if d < 220 else "medium" if d < 550 else "far",
            "direction": direction, "level": "same" if abs(item["height"]) <= 40 else "different",
            "hitscan": bool(item["hitscan"]), "major": bool(item["major"]),
            "body_blocker": bool(item["melee"] and d < 150),
        }

    def edge_key(self, start_cell, target_heading, mode="frontier"):
        return (tuple(start_cell), int(target_heading), str(mode))

    def edge_summary(self, start_cell=None):
        if start_cell is None:
            start_cell = cell(position(self.previous or {}))
        rows = []
        for key, value in self.edge_stats.items():
            if tuple(key[0]) == tuple(start_cell):
                rows.append({"heading": key[1], "mode": key[2], "tries": value.get("tries", 0),
                             "failed": value.get("failed", 0), "best": round(value.get("best", -99), 2)})
        return sorted(rows, key=lambda x: (x["failed"], -x["best"]))[:8]

    def route_epoch_state(self):
        if not self.epoch:
            return "none"
        stats = self.edge_stats.get(self.epoch["key"], {})
        if stats.get("tries", 0) == 0:
            return "untried"
        return "productive" if stats.get("best", -99) > 0 else "failed"

    def jev_state(self, obs, objects, threats, tic):
        here = cell(position(obs))
        visible_specials = [o for o in objects if not o["monster"] and not o["projectile"]]
        last_result = self.route_epoch_state()
        phase = "escape" if self.escape else "combat" if threats and threats[0]["distance"] < 260 else "trace" if self.trace_remaining else "frontier"
        return {
            "task": "Reach and activate the exit alive; optional kills/items are secondary.",
            "rules": "Only currently rendered labels are visible. Code computes bearings, ranges, legality, motion effects and the 35 Hz motor. Give a bounded hint, never a button stream or hidden route.",
            "health": "critical" if number(obs.get("health"), 100) < 25 else "wounded" if number(obs.get("health"), 100) < 60 else "healthy",
            "armor": "none" if number(obs.get("armor"), 0) < 10 else "present",
            "damage_recent": tic <= self.damage_until,
            "threats": [self.compact_object(x) for x in threats[:4]],
            "specials": [self.compact_object(x) for x in sorted(visible_specials, key=lambda x: x["distance"])[:5]],
            "motion": {
                "phase": phase, "blocked": self.blocked_ticks >= 3,
                "novelty": "new" if tic - self.last_new_tic < 25 else "old",
                "cell_revisits": "high" if self.visits.get(here, 0) >= 8 else "low",
                "epoch": last_result,
            },
            "route": {
                "hand": self.policy["hand"], "failed_recently": self.cell_failures.get(here, 0) > 0,
                "trace_side": "left" if self.trace_side > 0 else "right" if self.trace_remaining else "none",
                "previous": self.previous_action,
            },
        }

    # ---------- Jev bridge ----------
    def make_questions(self, reverse=False):
        route = {
            "frontier": "At the next safe macro boundary, keep the measured heading frontier and seek a new cell.",
            "trace_left": "The current cell is being revisited or a branch failed; briefly trace a left-hand moving boundary, not a stationary turn.",
            "trace_right": "The current cell is being revisited or a branch failed; briefly trace a right-hand moving boundary, not a stationary turn.",
            "use": "A currently rendered interaction or a blocked face makes one USE pulse justified.",
            "retreat": "Recent damage or a close threat makes a finite escape safer than route progress.",
            "unknown": "The facts do not support a semantic change; leave the deterministic fallback in charge.",
        }
        if reverse:
            route = dict(reversed(list(route.items())))
        return {
            "route": Choice(
                "Using only state.motion, state.route, state.threats and state.specials, choose one bounded route hint. Do not infer a hidden exit, compare raw numbers, or emit a button.",
                route,
            ),
            "dodge": Choice(
                "For a visible close threat, choose the movement side most likely to preserve a retreat lane. This only sets a future lease preference.",
                {
                    "left": "Move/circle to the player's left.",
                    "right": "Move/circle to the player's right.",
                    "forward": "Break contact through a forward diagonal when the rear is blocked.",
                    "back": "Back away only when the rear is not the blocked direction.",
                    "none": "No special side is justified.",
                },
            ),
            "engage": Choice(
                "Given state.health, state.damage_recent, state.threats and state.motion.blocked, decide whether to fight the named threat or break contact. Do not select a weapon.",
                {
                    "fight": "The visible threat can be engaged while a movement lane remains usable.",
                    "break_contact": "Immediate damage/contact or a blocked movement lease makes movement away the safer local intention.",
                    "bypass": "The threat is not an immediate blocker; keep the route instead of chasing it.",
                    "unknown": "Evidence is insufficient; code safety rules should decide.",
                },
            ),
            "open": Noul(
                "Do the currently visible `specials` or `motion.blocked` facts justify one USE pulse soon?",
                {"true": "A visible door/switch/exit/teleporter or a tested blocked face makes USE plausible.",
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
        self.jev_request = {"tic": tic, "state": state}
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, self.make_questions(bool(self.heading_visits.total() % 2)), timeout_s=1.35)
            )
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
            return
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
        route = answer_choice(answers, "route", 0.58)
        dodge = answer_choice(answers, "dodge", 0.54)
        engage = answer_choice(answers, "engage", 0.58)
        request_state = request.get("state") if isinstance(request, dict) else {}
        motion = request_state.get("motion") if isinstance(request_state, dict) else {}
        threat_facts = request_state.get("threats") if isinstance(request_state, dict) else []
        if not isinstance(threat_facts, list):
            threat_facts = []
        revisiting = (
            motion.get("cell_revisits") == "high" or
            motion.get("novelty") == "old" or
            motion.get("epoch") == "failed"
        ) if isinstance(motion, dict) else False
        close_fact = any(
            isinstance(x, dict) and x.get("range") in {"contact", "near"}
            for x in threat_facts
        )
        # Trace is an exploration intervention, not a free-form model button:
        # require code-derived recurrence evidence before accepting it.
        if route in {"trace_left", "trace_right"} and revisiting:
            self.requested_mode = route
        elif route == "frontier":
            self.requested_mode = route
        elif route == "use":
            self.force_use = True
        elif route == "retreat" and (bool(request_state.get("damage_recent")) or close_fact):
            self.contact_break_pending = True
        if dodge == "left":
            self.fire_side = 1
        elif dodge == "right":
            self.fire_side = -1
        elif dodge == "forward":
            self.fire_side = 1 if self.fire_side >= 0 else -1
            self.aim_move_action = self.movement_choice(self.fire_side, retreat=False)
        elif dodge == "back":
            self.aim_move_action = self.movement_choice(self.fire_side, retreat=True)
        if engage == "break_contact" and (bool(request_state.get("damage_recent")) or close_fact):
            self.contact_break_pending = True
        if answer_yes(answers, "open", 0.72):
            self.force_use = True
        self.jev_intent = route or self.jev_intent
        self.ctx.log("jev_route_hint", route=route, dodge=dodge, engage=engage,
                     use=self.force_use, age_tics=age)

    # ---------- Astra bounded policy bridge ----------
    def launch_astra(self, obs, tic):
        if self.astra_future is not None or self.astra_calls >= 4:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if remaining.get("system2_calls", 0) <= 0 or remaining.get("wall_s", 0) < 120:
            return
        objects = label_objects(obs)
        threats = self.threats(objects)
        payload = {
            "map": str(obs.get("map", "unknown")), "iwad": str(obs.get("iwad", "unknown")),
            "skill": integer(obs.get("skill"), 0), "seed": integer(obs.get("seed"), 0),
            "tic": tic, "health": number(obs.get("health"), 100),
            "counts": {"kills": integer(obs.get("kills")), "items": integer(obs.get("items")),
                       "secrets": integer(obs.get("secrets"))},
            "visible": [self.compact_object(x) for x in objects[:10]],
            "recent": list(self.history), "route_epochs": self.edge_summary(),
            "visited_cells": len(self.visits), "collisions": self.collision_count,
            "current_policy": self.policy, "current_mode": self.route_mode,
        }
        prompt = (
            "You are Astra, the slow planning layer of a real-time Classic Doom "
            "controller. Exit success is primary and survival is required. The public "
            "state has only currently rendered labels and an observed trajectory; do not "
            "invent a hidden door, coordinate waypoint, map route, or omniscient object. "
            "Code owns target selection, legality, collision effects, exact arithmetic, "
            "weapon safety and the 35 Hz clock. Revise a bounded macro policy after the "
            "route-epoch evidence. Return ONLY one JSON object with exactly these optional "
            "fields: {\"mode\":\"frontier|left_trace|right_trace|safe\", "
            "\"hand\":\"left|right\", \"sweep\":\"short|medium|long\", "
            "\"combat\":\"clear_blockers|avoid_optional\", "
            "\"phases\":[{\"kind\":\"forward|forward_left|forward_right|turn_left|turn_right|use\",\"tics\":1}], "
            "\"reason\":\"brief\"}. At most four phases and at most 48 tics per phase. "
            "If evidence is weak, use mode frontier and no phases.\n" +
            json.dumps(payload, separators=(",", ":"), allow_nan=False)
        )
        self.astra_calls += 1
        self.astra_request_tic = tic
        session = self.astra_session
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
        if not isinstance(proposal, dict):
            return
        if proposal.get("mode") in {"frontier", "left_trace", "right_trace", "safe"}:
            self.astra_mode = proposal["mode"]
        updated = dict(self.policy)
        if proposal.get("hand") in {"left", "right"}:
            updated["hand"] = proposal["hand"]
        if proposal.get("sweep") in {"short", "medium", "long"}:
            updated["sweep"] = proposal["sweep"]
        if proposal.get("combat") in {"clear_blockers", "avoid_optional"}:
            updated["combat"] = proposal["combat"]
        if updated != self.policy:
            self.policy = updated
            self.turn_side = 1 if updated["hand"] == "left" else -1
            self.sweep_period = {"short": 32, "medium": 48, "long": 72}[updated["sweep"]]
            self.sweep_left = min(self.sweep_left, self.sweep_period)
        # A delayed plan is useful only as a short quiet hint.  No arbitrary
        # action or duration can enter the motor, and stale plans are dropped.
        plan = []
        if 0 <= tic - self.astra_request_tic <= 1800:
            raw_phases = proposal.get("phases")
            if isinstance(raw_phases, list):
                for phase in raw_phases[:4]:
                    if not isinstance(phase, dict):
                        continue
                    kind = phase.get("kind")
                    n = integer(phase.get("tics"), 0)
                    if kind in {"forward", "forward_left", "forward_right", "turn_left", "turn_right", "use"} and 1 <= n <= 48:
                        plan.append([kind, n])
        self.astra_plan = collections.deque(plan)
        self.ctx.log("astra_route_policy", mode=self.astra_mode, policy=dict(self.policy),
                     phases=len(plan), reason=str(proposal.get("reason", ""))[:220])

    # ---------- route-effect ledger ----------
    def finish_epoch(self, obs, tic, reason):
        epoch = self.epoch
        if not epoch:
            return
        start = epoch["position"]
        now = position(obs)
        moved = distance(start, now)
        count_now = tuple(integer(obs.get(k), 0) for k in ("kills", "items", "secrets"))
        count_gain = sum(max(0, count_now[i] - epoch["counts"][i]) for i in range(3))
        health_loss = max(0.0, epoch["health"] - number(obs.get("health"), epoch["health"]))
        new_cells = max(0, len(self.visits) - epoch["visited"])
        same = cell(now) == epoch["cell"]
        score = new_cells * 3.0 + min(2.0, moved / 160.0) + count_gain * 0.8
        score -= min(8.0, health_loss / 6.0)
        if same:
            score -= 1.5
        if moved < 24:
            score -= 1.0
        key = epoch["key"]
        stats = self.edge_stats.setdefault(key, {"tries": 0, "failed": 0, "best": -99.0})
        stats["tries"] += 1
        stats["best"] = max(number(stats.get("best"), -99.0), score)
        failed = (score <= 0.0 or (moved < 18 and new_cells == 0) or
                  health_loss >= 18.0 or number(obs.get("health"), 100) < 25)
        if failed:
            stats["failed"] += 1
            self.cell_failures[epoch["cell"]] += 1
        else:
            self.cell_failures[epoch["cell"]] = max(0, self.cell_failures[epoch["cell"]] - 1)
        self.ctx.log("route_epoch", number=epoch["number"], mode=epoch["mode"],
                     reason=reason, cell=list(epoch["cell"]), heading=epoch["heading"],
                     moved=round(moved, 1), new_cells=new_cells, health_loss=round(health_loss, 1),
                     score=round(score, 2), failed=failed)
        self.epoch = None

    def begin_epoch(self, obs, tic, side, target_heading, mode, duration):
        if self.epoch:
            return
        p = position(obs)
        c = cell(p)
        key = self.edge_key(c, target_heading, mode)
        self.epoch_number += 1
        self.epoch = {
            "number": self.epoch_number, "cell": c, "heading": target_heading,
            "key": key, "mode": mode, "position": p,
            "health": number(obs.get("health"), 100),
            "counts": tuple(integer(obs.get(k), 0) for k in ("kills", "items", "secrets")),
            "visited": len(self.visits), "until": tic + max(8, duration), "side": side,
        }

    def frontier_turn_plan(self, obs, preferred=None):
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        angle = number(player.get("angle"))
        here = cell(position(obs))
        preferred = preferred if preferred in (-1, 1) else self.turn_side
        choices = []
        for magnitude in (2, 1, 3, 4):
            for side in (preferred, -preferred):
                target = heading_bin(angle + side * 45.0 * magnitude)
                stats = self.edge_stats.get(self.edge_key(here, target, "frontier"), {})
                seen = self.heading_visits.get((here, target), 0)
                failed = integer(stats.get("failed"), 0)
                best = number(stats.get("best"), 0.0)
                # A failed heading is not banned: a new view can turn a dead
                # branch into an exit after an interaction, but it is deprioritized.
                cost = seen + failed * 5 - min(2.0, max(0.0, best))
                choices.append((cost, 0 if side == preferred else 1,
                                0 if magnitude == 2 else 1, magnitude, side, target, failed))
        _, _, _, magnitude, side, target, failed = min(choices)
        steps = max(8, int(round((45.0 * magnitude) / max(2.0, self.turn_rate))))
        return side, steps, target, failed

    def start_escape(self, tic, reason, obs, trace=False, side=None):
        if self.escape:
            return
        if self.epoch and tic >= self.epoch.get("until", tic):
            self.finish_epoch(obs, tic, "expired")
        preferred = side if side in (-1, 1) else self.turn_side
        chosen, turn_steps, target, failed = self.frontier_turn_plan(obs, preferred)
        self.turn_side = chosen
        turn = self.turn_action(chosen)
        # Avoid repeating an empirically blocked retreat direction.
        first = self.legal("backward")
        if self.blocked_ticks >= 3 and self.previous_action in {"backward", "backward_left", "backward_right"}:
            first = self.movement_choice(-chosen, retreat=False)
        if trace:
            travel = self.movement_choice(chosen, retreat=False)
            seq = [first] * 4 + [turn] * turn_steps + [travel] * 22
            mode = "trace_escape"
        else:
            seq = [first] * 5 + [turn] * turn_steps + [self.legal("forward")] * 12
            mode = "escape"
        self.escape = collections.deque(seq)
        self.escape_reason = reason
        self.blocked_ticks = 0
        self.motion_bad_ticks = 0
        self.contact_break_pending = False
        self.collision_count += 1
        self.goal = None
        self.begin_epoch(obs, tic, chosen, target, mode, len(seq) + 8)
        self.ctx.log("adaptive_escape", tic=tic, reason=reason, turn=turn,
                     target_heading=target, prior_failures=failed,
                     turn_steps=turn_steps, trace=trace, collisions=self.collision_count)

    def start_trace(self, obs, tic, side):
        if self.trace_remaining > 0 or self.escape:
            return
        self.trace_side = 1 if side >= 0 else -1
        self.trace_remaining = 190
        self.trace_phase = 0
        self.route_mode = "trace"
        # Use the current heading as the effect ledger's trace key.  The trace
        # itself supplies translation; a collision can replace it with escape.
        self.begin_epoch(obs, tic, self.trace_side,
                         heading_bin((obs.get("player") or {}).get("angle")),
                         "trace_left" if self.trace_side > 0 else "trace_right", 190)
        self.ctx.log("trace_start", tic=tic, side="left" if self.trace_side > 0 else "right",
                     failures=self.cell_failures.get(cell(position(obs)), 0))

    # ---------- deterministic combat and movement ----------
    def weapon_switch(self, obs, tic):
        if tic - self.last_weapon_tic < 35:
            return None
        weapons = obs.get("weapons") if isinstance(obs.get("weapons"), dict) else {}
        ammo = obs.get("ammo") if isinstance(obs.get("ammo"), dict) else {}
        current = integer(obs.get("weapon"), 2)
        current_ammo = number(obs.get("weapon_ammo"), 0)
        desired = None
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

    def choose_aim_lease(self, threat, tic):
        # Preserve the parent's stable opening side for a distant target; only
        # alternate the orbit after contact or fresh damage makes the current
        # side diagnostically unsafe.
        if threat["distance"] < 230 or tic <= self.damage_until:
            self.fire_side *= -1
        close = threat["distance"] < 230 or tic <= self.damage_until
        if self.blocked_ticks >= 3:
            # The observed blocked direction is deliberately not selected.
            candidates = ("forward_left", "forward_right", "strafe_left", "strafe_right") if self.fire_side > 0 else ("forward_right", "forward_left", "strafe_right", "strafe_left")
        elif close:
            candidates = (("backward_left", "forward_left", "strafe_left") if self.fire_side > 0 else ("backward_right", "forward_right", "strafe_right"))
        else:
            candidates = (("strafe_left", "forward_left", "strafe_right") if self.fire_side > 0 else ("strafe_right", "forward_right", "strafe_left"))
        self.aim_move_action = next((x for x in candidates if x in self.allowed), self.legal("forward"))
        self.aim_move_left = 4
        self.aim_bad_ticks = 0
        return self.aim_move_action

    def combat_action(self, obs, threat, tic):
        # A movement lease is an experiment, not an unconditional button hold.
        # A distant hitscanner can be legally visible while the player is
        # turning against a spawn wall; treating three blocked lease tics as a
        # contact emergency there caused the early 0019 smoke-test death.
        close_contact = threat["distance"] < 300 or tic <= self.damage_until
        if ((self.contact_break_pending and close_contact) or
                (self.blocked_ticks >= 4 and close_contact) or
                (self.aim_bad_ticks >= 3 and threat["distance"] < 220)):
            self.aim_move_left = 0
            self.aim_move_action = None
            self.start_escape(tic, "contact_blocked", obs, trace=True, side=self.fire_side)
            if self.escape:
                return self.escape.popleft()
        if self.aim_move_left > 0:
            self.aim_move_left -= 1
            action = self.legal(self.aim_move_action or self.movement_choice(self.fire_side))
            return action
        selected = self.weapon_switch(obs, tic)
        if selected:
            return selected
        health = number(obs.get("health"), 100)
        distance_now = threat["distance"]
        error = threat["error"]
        tolerance = max(6.0, self.turn_rate * 1.35)
        # At critical health, don't spend a fresh observation turning in place.
        if health < 25 and distance_now < 300:
            return self.choose_aim_lease(threat, tic)
        if abs(error) > tolerance:
            self.aim_turn_ticks += 1
            if self.aim_turn_ticks >= 5:
                self.aim_turn_ticks = 0
                # A far hitscanner is not a reason to abandon the parent's
                # stable turning/fire rhythm.  Use the moving lease for the
                # close/damaged cases it was meant to solve; otherwise spend
                # another short turn correction.
                if threat["distance"] < 360 or tic <= self.damage_until:
                    return self.choose_aim_lease(threat, tic)
            return self.turn_action(1 if error > 0 else -1)
        self.aim_turn_ticks = 0
        if distance_now < 105 and (threat["melee"] or tic <= self.damage_until):
            return self.movement_choice(self.fire_side, retreat=True)
        if number(obs.get("weapon_ammo"), 0) <= 0:
            return self.movement_choice(self.fire_side, retreat=True)
        if health < 28 and distance_now < 190:
            return self.movement_choice(self.fire_side, retreat=True)
        action = self.movement_choice(self.fire_side, fire=True)
        self.fire_side = 1 if action.endswith("left") else -1 if action.endswith("right") else self.fire_side
        return self.legal(action)

    def steer_to_goal(self, goal, tic):
        item = goal.get("item") or {}
        error = number(item.get("error"), 0.0)
        if abs(error) > max(7.0, self.turn_rate * 1.25):
            self.goal_turns += 1
            if self.goal_turns >= 8:
                self.goal_turns = 0
                return self.movement_choice(1 if error > 0 else -1)
            return self.turn_action(1 if error > 0 else -1)
        self.goal_turns = 0
        if number(item.get("distance"), 9999) < 48:
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
        if tic - self.last_use_tic >= 31:
            self.last_use_tic = tic
            return self.legal("use")
        return None

    def trace_action(self, obs, tic):
        if self.trace_remaining <= 0:
            self.route_mode = "frontier"
            return None
        self.trace_remaining -= 1
        if self.blocked_ticks >= 3:
            face = (cell(position(obs)), heading_bin((obs.get("player") or {}).get("angle")))
            if self.probed_faces.get(face, -1) < tic and tic - self.last_use_tic > 12:
                self.probed_faces[face] = tic + 140
                self.last_use_tic = tic
                return self.legal("use")
            self.start_escape(tic, "trace_collision", obs, trace=True, side=self.trace_side)
            return self.escape.popleft() if self.escape else self.movement_choice(self.trace_side)
        # A short curved translation probes a boundary; a straight interval
        # prevents the curve from becoming an in-place spiral.
        self.trace_phase = (self.trace_phase + 1) % 36
        if self.trace_phase < 22:
            return self.movement_choice(self.trace_side, retreat=False)
        return self.legal("forward")

    def planned_action(self, tic):
        if not self.astra_plan:
            return None
        kind, left = self.astra_plan[0]
        if left <= 0:
            self.astra_plan.popleft()
            return self.planned_action(tic)
        self.astra_plan[0][1] = left - 1
        if left == 1:
            self.astra_plan.popleft()
        if kind == "use":
            if tic - self.last_use_tic < 10:
                return None
            self.last_use_tic = tic
        return self.legal(kind)

    def macro_action(self, tic, obs):
        if self.escape:
            return self.escape.popleft()
        if self.turn_left > 0:
            self.turn_left -= 1
            return self.turn_action(self.turn_side)
        if self.trace_remaining > 0:
            action = self.trace_action(obs, tic)
            if action:
                return action
        if tic - self.last_new_tic > 115 and self.route_mode == "frontier":
            here = cell(position(obs))
            if self.cell_failures.get(here, 0) >= 2 and tic >= self.trace_cooldown:
                self.trace_cooldown = tic + 700
                self.start_trace(obs, tic, -self.turn_side)
                return self.trace_action(obs, tic) or self.legal("forward")
            self.start_escape(tic, "no_new_cell", obs)
            return self.escape.popleft() if self.escape else self.legal("forward")
        if tic - self.last_new_tic < 80:
            self.trace_cooldown = min(self.trace_cooldown, tic + 100)
        if self.route_mode == "frontier" and self.sweep_left <= 0:
            self.sweep_count += 1
            self.sweep_left = self.sweep_period
            preferred = self.turn_side
            if self.preferred_turn in {"turn_left", "turn_right"}:
                preferred = 1 if self.preferred_turn == "turn_left" else -1
            side, steps, target, failed = self.frontier_turn_plan(obs, preferred)
            self.turn_side = side
            self.preferred_turn = None
            self.turn_left = steps
            self.begin_epoch(obs, tic, side, target, "frontier", steps + self.sweep_period)
            self.ctx.log("frontier_turn", tic=tic, side="left" if side > 0 else "right",
                         target_heading=target, prior_failures=failed, turn_steps=steps,
                         count=self.sweep_count)
            self.turn_left -= 1
            return self.turn_action(side)
        if self.route_mode == "frontier":
            self.sweep_left -= 1
        return self.legal("forward")

    # ---------- priority policy ----------
    def apply_model_mode(self, obs, tic, threats):
        if threats or self.escape or self.goal is not None:
            return
        mode = self.astra_mode or self.requested_mode
        here = cell(position(obs))
        recurrent = (
            self.cell_failures.get(here, 0) >= 2 or
            tic - self.last_new_tic > 115 or
            self.route_epoch_state() == "failed"
        )
        if mode in {"left_trace", "right_trace"} and not recurrent:
            # A semantic trace suggestion without code evidence is ignored.
            self.requested_mode = None
            self.astra_mode = None
            return
        if mode == "left_trace":
            self.route_mode = "trace"
            self.start_trace(obs, tic, 1)
            self.requested_mode = None
            self.astra_mode = None
        elif mode == "right_trace":
            self.route_mode = "trace"
            self.start_trace(obs, tic, -1)
            self.requested_mode = None
            self.astra_mode = None
        elif mode == "frontier":
            self.route_mode = "frontier"
            self.requested_mode = None
            self.astra_mode = None
        elif mode == "safe":
            self.policy["combat"] = "avoid_optional"
            self.requested_mode = None
            self.astra_mode = None

    def action(self, obs, objects, threats, tic):
        close_projectile = any(x["distance"] < 190 for x in objects if x["projectile"])
        if close_projectile:
            # Hold one dodge side long enough to create lateral velocity.  If
            # the measured side is blocked, switch to a moving diagonal rather
            # than alternating two ineffective strafe pulses at a wall.
            self.projectile_ticks += 1
            self.contact_break_pending = False
            if self.blocked_ticks >= 3:
                self.fire_side *= -1
                self.projectile_ticks = 0
                return self.movement_choice(self.fire_side, retreat=False)
            if self.projectile_ticks % 18 == 0:
                self.fire_side *= -1
            return self.legal("strafe_left" if self.fire_side > 0 else "strafe_right")
        self.projectile_ticks = 0
        # Once a contact or collision escape is committed, finish its measured
        # turn/translation experiment even if the same actor remains visible.
        # Otherwise the immediate-threat branch can interrupt it on the very
        # next observation, which is exactly the blocked lease failure seen in
        # the 0015 worst trace.
        if self.escape:
            return self.escape.popleft()
        target = threats[0] if threats else None
        routine_limit = 180 if self.policy.get("combat") == "avoid_optional" else 260
        immediate = target is not None and (
            target["distance"] < routine_limit or
            (target["hitscan"] and target["distance"] < 650) or target["major"] or tic <= self.damage_until
        )
        if immediate:
            self.goal = None
            return self.combat_action(obs, target, tic)
        self.apply_model_mode(obs, tic, threats)
        if self.escape:
            return self.escape.popleft()
        planned = self.planned_action(tic)
        if planned and self.route_mode == "frontier":
            return planned
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
            face = (cell(position(obs)), heading_bin((obs.get("player") or {}).get("angle")))
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

    # ---------- observation update and lifecycle ----------
    def update(self, obs):
        tic = integer(obs.get("tic"), self.previous_tic + 1)
        if tic <= self.previous_tic:
            return tic, False
        p = position(obs)
        angle = number((obs.get("player") or {}).get("angle"))
        if self.previous is not None and self.previous_position is not None:
            dt = tic - self.previous_tic
            moved = distance(p, self.previous_position)
            delta = wrap(angle - number((self.previous.get("player") or {}).get("angle")))
            if self.previous_action in TURN_ACTIONS and 0 < dt <= 10 and abs(delta) > 0.15:
                rate = abs(delta) / max(1, dt)
                if 0.2 < rate < 25:
                    self.turn_rate = 0.75 * self.turn_rate + 0.25 * rate
                    self.turn_sign = (1.0 if delta > 0 else -1.0) if self.previous_action == "turn_left" else (-1.0 if delta > 0 else 1.0)
            if self.previous_action in MOVE_ACTIONS and dt <= 14:
                if moved < max(2.5, dt * 0.55):
                    self.blocked_ticks += dt
                else:
                    self.blocked_ticks = 0
            else:
                self.blocked_ticks = 0
            if self.aim_move_left > 0 and self.previous_action == self.aim_move_action:
                if moved < max(2.0, dt * 0.35):
                    self.aim_bad_ticks += dt
                else:
                    self.aim_bad_ticks = 0
            old_health = number(self.previous.get("health"), self.last_health)
            health = number(obs.get("health"), old_health)
            if health < old_health - 0.1:
                self.damage_until = tic + 35
                if self.aim_move_left > 0:
                    self.aim_bad_ticks += max(1, dt)
            self.last_health = health
        if self.epoch and tic >= self.epoch.get("until", tic):
            self.finish_epoch(obs, tic, "time")
        c = cell(p)
        was_unseen = c not in self.visits
        self.visits[c] += 1
        self.heading_visits[(c, heading_bin(angle))] += 1
        if was_unseen:
            self.last_new_cell = c
            self.last_new_tic = tic
            self.same_cell_streak = 0
        else:
            self.same_cell_streak += 1
        counts = tuple(integer(obs.get(k), 0) for k in ("kills", "items", "secrets"))
        if counts != self.last_counts:
            self.last_counts = counts
            self.last_new_tic = tic
        if not self.trail or distance(self.trail[-1][1], p) > 20:
            self.trail.append((tic, p))
        self.history.append({
            "tic": tic, "cell": list(c), "x": round(p[0], 1), "y": round(p[1], 1),
            "health": round(number(obs.get("health"), 100)), "kills": counts[0],
            "items": counts[1], "action": self.previous_action,
        })
        self.previous = obs
        self.previous_position = p
        self.previous_tic = tic
        return tic, True

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
                objects = label_objects(obs)
                threats = self.threats(objects)
                self.poll_jev(obs, tic)
                self.poll_astra(tic)
                self.dispatch_jev(obs, objects, threats, tic)
                if self.astra_future is None and tic >= self.next_astra_tic:
                    self.launch_astra(obs, tic)
                    self.next_astra_tic = tic + 4200
                action = self.legal(self.action(obs, objects, threats, tic))
                ctx.act(action)
                self.previous_action = action
                ctx.wait_for_tick()
        except (EpisodeOver, BudgetExceeded):
            return
        finally:
            for name in ("jev_future", "astra_future"):
                future = getattr(self, name, None)
                if future is not None and not future.done():
                    try:
                        future.cancel()
                    except Exception:
                        pass
            try:
                ctx.memory["adaptive_route_summary"] = {
                    "visited_cells": len(self.visits), "collisions": self.collision_count,
                    "epochs": self.epoch_number, "failed_cells": sum(self.cell_failures.values()),
                    "mode": self.route_mode, "policy": dict(self.policy),
                }
            except Exception:
                pass


def build(ctx):
    return Harness(ctx)
