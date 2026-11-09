"""Candidate 0014: contact-following navigation with committed moving combat.

The controller treats the level as an unknown planar route rather than as a
square sweep.  It follows one side of the currently encountered geometry,
opens a door/switch when it reaches a contact, and only changes the side after
measured stagnation.  Jev selects among finite route/combat *macros* and Astra
selects the navigator policy; neither model is allowed to post a raw button.
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

# These are only used after the visible label category says Monster.  Unknown
# monster names remain ordinary visible threats instead of being discarded.
HITSCAN = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS",
           "SpiderMastermind"}
MAJOR = {"Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind",
         "Revenant"}
MELEE = {"Demon", "Spectre", "LostSoul", "Pinkie", "Pinky"}


def finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def integer(value, default=0):
    try:
        return int(finite(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def wrap(angle):
    return (finite(angle) + 180.0) % 360.0 - 180.0


def as_dict(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def player_position(obs):
    player = obs.get("player") if isinstance(obs, dict) else None
    if not isinstance(player, dict):
        return (0.0, 0.0, 0.0)
    return tuple(finite(player.get(k), 0.0) for k in ("x", "y", "z"))


def distance(a, b):
    if not a or not b:
        return float("inf")
    return math.sqrt(sum((finite(x) - finite(y)) ** 2 for x, y in zip(a, b)))


def xy_distance(a, b):
    if not a or not b:
        return float("inf")
    return math.hypot(finite(a[0]) - finite(b[0]),
                      finite(a[1]) - finite(b[1]))


def heading_bin(angle):
    return int(math.floor((finite(angle) % 360.0 + 22.5) / 45.0)) % 8


def position_cell(point):
    # A memory of observed space, not a map assumption.  The coarseness keeps
    # collisions and wall contacts stable when the native coordinates jitter.
    return (math.floor(finite(point[0]) / 64.0),
            math.floor(finite(point[1]) / 64.0),
            math.floor(finite(point[2]) / 64.0))


def json_object(text):
    """Extract a first JSON object from arbitrary Astra text."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for i, char in enumerate(text[:16000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        return value if isinstance(value, dict) else {}
    return {}


def choice_value(answers, key, allowed, threshold=0.57):
    if not isinstance(answers, dict):
        return None
    answer = answers.get(key)
    value = getattr(answer, "value", None)
    if not isinstance(value, str) or value not in allowed:
        return None
    probabilities = getattr(answer, "probabilities", None)
    if isinstance(probabilities, dict) and value in probabilities:
        if finite(probabilities.get(value), 0.0) < threshold:
            return None
    confidence = getattr(answer, "confidence", None)
    if confidence is not None and finite(confidence, 0.0) < 0.07:
        return None
    return value


def yes_value(answers, key, threshold=0.72):
    if not isinstance(answers, dict):
        return False
    value = getattr(answers.get(key), "value", 0.0)
    if isinstance(value, bool):
        return value
    return finite(value, 0.0) >= threshold


class Harness:
    """One episode of a finite macro navigator."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.noop = "noop"

        # Observation bookkeeping and calibrated motor geometry.
        self.previous_obs = {}
        self.previous_position = None
        self.previous_tic = -1
        self.previous_action = "noop"
        self.last_health = 100.0
        self.last_damage_counter = 0.0
        self.damage_amount = 0.0
        self.damage_until = -1
        self.turn_sign = 1.0
        self.turn_rate = 5.6
        self.blocked_tics = 0
        self.contact_latched = False
        self.contact_pending = False
        self.contact_count = 0
        self.last_new_space_tic = 0
        self.last_cell = None
        self.last_angle = 0.0
        self.last_counts = (0, 0, 0)

        # Observed route evidence.  This is used to change a route hypothesis,
        # not to pretend that a coarse cell is a real Doom sector.
        self.cells = collections.Counter()
        self.heading_visits = collections.Counter()
        self.contact_signatures = collections.Counter()
        self.transitions = collections.Counter()
        self.route_history = collections.deque(maxlen=56)
        self.unique_cells = 0
        self.recoveries = 0
        self.current_cell = (0, 0, 0)

        # Navigator policy.  wall_side is a desired mathematical turn sign:
        # +1 follows the left boundary and -1 follows the right boundary.
        self.navigator = "wall_right"
        self.wall_side = -1
        self.combat_policy = "protect_health"
        self.door_policy = "use_on_contact"
        self.policy_version = 0
        self.nav_script = collections.deque()
        self.nav_script_kind = ""
        self.sweep_left = 0
        self.last_recovery_tic = -10000
        self.stagnation_switches = 0

        # Label-backed goals are deliberately narrow: exit-like interactions,
        # keys, capability weapons, and nearby survival items only.
        self.goal = None
        self.goal_defer = {}
        self.goal_turn_streak = 0
        self.last_use_tic = -10000
        self.force_use = False
        self.force_use_until = -1
        self.last_weapon_tic = -10000

        # Combat is a moving lease.  A target may cause at most two aiming tics
        # before the controller must translate/fire or retreat.
        self.fire_side = 1
        self.aim_turn_streak = 0
        self.combat_move_until = -1
        self.retreat_until = -1
        self.jev_target = None
        self.jev_recover = False

        self.objects = []
        self.threat_list = []
        self.projectiles = []
        self.current_map = ""
        self.current_skill = 0

        # Asynchronous Jev and Astra work.
        self.jev_future = None
        self.jev_meta = None
        self.next_jev_tic = 0
        self.jev_request_number = 0
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.next_astra_tic = 240
        self.history = collections.deque(maxlen=28)

    # ------------------------------------------------------------------
    # Legal actions and observation-derived facts
    def legal(self, action):
        if action in self.allowed:
            return action
        if self.noop in self.allowed:
            return self.noop
        return next(iter(self.allowed), "noop")

    def turn_action(self, desired_sign):
        # The sign is calibrated from observed ANGLE changes, so this does not
        # depend on a particular IWAD/config button ordering.
        positive_is_left = self.turn_sign > 0.0
        want_left = desired_sign > 0.0
        return self.legal("turn_left" if want_left == positive_is_left else "turn_right")

    def turn_steps(self, degrees=90.0):
        return max(5, int(round(abs(degrees) / max(2.0, self.turn_rate))))

    def moving_fallback(self):
        for action in ("forward", "forward_left", "strafe_left", "backward"):
            if action in self.allowed:
                return action
        return self.legal(self.noop)

    def backward_dodge(self):
        action = "backward_left" if self.fire_side > 0 else "backward_right"
        return self.legal(action)

    def parse_objects(self, obs):
        point = player_position(obs)
        angle = finite((obs.get("player") or {}).get("angle"), 0.0)
        raw_visible = obs.get("visible")
        if not isinstance(raw_visible, list):
            raw_visible = []
        objects = []
        for raw in raw_visible:
            if not isinstance(raw, dict):
                continue
            world = raw.get("world")
            if not isinstance(world, (list, tuple)) or len(world) < 3:
                world = raw.get("position")
            if not isinstance(world, (list, tuple)) or len(world) < 3:
                continue
            q = tuple(finite(x, 0.0) for x in world[:3])
            name = str(raw.get("name", raw.get("object_name", "unknown")))[:80]
            category = str(raw.get("category", raw.get("object_category", "Unknown")))[:40]
            low_name = name.lower()
            low_cat = category.lower()
            if low_cat == "self" or name == "DoomPlayer":
                continue
            d = distance(point, q)
            bearing = math.degrees(math.atan2(q[1] - point[1], q[0] - point[0])) % 360.0
            velocity = raw.get("velocity")
            if not isinstance(velocity, (list, tuple)) or len(velocity) < 3:
                velocity = (0.0, 0.0, 0.0)
            speed = math.sqrt(sum(finite(x) ** 2 for x in velocity[:3]))
            monster = low_cat == "monster" and not low_name.startswith("dead")
            projectile = (
                "projectile" in low_cat or "missile" in low_cat or
                low_name in {"doomimpball", "cacodemonball", "plasmaball",
                             "rocket", "fireball", "revenantmissile"}
            )
            interaction = (
                low_cat in {"exit", "switch", "door", "teleporter", "teleport"}
                or any(word in low_name for word in
                       ("exit", "switch", "door", "teleport"))
            )
            key = low_cat == "key" or "key" in low_name or low_name.endswith("card")
            pickup = low_cat in {
                "key", "weapon", "health", "armor", "ammo", "powerup", "power-up"
            }
            item = {
                "id": str(raw.get("id", raw.get("object_id", name))),
                "name": name,
                "category": category,
                "pos": q,
                "distance": d,
                "height": q[2] - point[2],
                "error": wrap(bearing - angle),
                "speed": speed,
                "monster": monster,
                "projectile": projectile,
                "interaction": interaction,
                "key": key,
                "pickup": pickup,
            }
            if monster:
                item["hitscan"] = name in HITSCAN
                item["major"] = name in MAJOR
                item["melee"] = name in MELEE
                active = speed > 0.12 or d < 300.0 or self.damage_until >= self.previous_tic
                item["active"] = active
                item["threat_score"] = (
                    (220.0 if item["major"] else 0.0) +
                    (125.0 if item["hitscan"] else 0.0) +
                    (110.0 if item["melee"] and d < 220.0 else 0.0) +
                    (170.0 if d < 100.0 else 80.0 if d < 250.0 else 0.0) +
                    (55.0 if active else 0.0) - min(180.0, d * 0.13)
                )
            objects.append(item)
        self.objects = objects
        self.threat_list = sorted(
            [x for x in objects if x.get("monster")],
            key=lambda x: finite(x.get("threat_score"), -1e9), reverse=True,
        )
        self.projectiles = sorted(
            [x for x in objects if x.get("projectile")],
            key=lambda x: finite(x.get("distance"), 1e9),
        )
        return objects

    def update(self, obs):
        """Consume one new engine observation and update exact local facts."""
        obs = as_dict(obs)
        if not obs:
            return False
        tic = integer(obs.get("tic"), integer(obs.get("elapsed_tics"), 0))
        if tic <= self.previous_tic:
            return False
        dt = max(1, tic - self.previous_tic) if self.previous_tic >= 0 else 1
        point = player_position(obs)
        angle = finite((obs.get("player") or {}).get("angle"), 0.0)
        moved = 0.0 if self.previous_position is None else distance(point, self.previous_position)

        if self.previous_position is not None:
            old_angle = finite((self.previous_obs.get("player") or {}).get("angle"), self.last_angle)
            delta = wrap(angle - old_angle)
            if self.previous_action in TURN_ACTIONS and 0 < dt <= 12 and abs(delta) > 0.15:
                rate = abs(delta) / max(1, dt)
                if 0.2 < rate < 25.0:
                    self.turn_rate = 0.78 * self.turn_rate + 0.22 * rate
                    if self.previous_action == "turn_left":
                        self.turn_sign = 1.0 if delta > 0 else -1.0
                    else:
                        self.turn_sign = -1.0 if delta > 0 else 1.0
            if self.previous_action in MOVE_ACTIONS:
                if moved < max(1.8, dt * 0.38):
                    self.blocked_tics += dt
                else:
                    self.blocked_tics = 0
            else:
                self.blocked_tics = 0

        health = finite(obs.get("health"), self.last_health)
        damage_counter = finite(obs.get("damage_taken"), self.last_damage_counter)
        health_loss = max(0.0, self.last_health - health)
        counter_loss = max(0.0, damage_counter - self.last_damage_counter)
        self.damage_amount = max(health_loss, counter_loss)
        if self.damage_amount >= 1.0:
            self.damage_until = tic + 38
            self.fire_side *= -1
            if self.damage_amount >= 7.0:
                self.retreat_until = max(self.retreat_until, tic + 55)

        here = position_cell(point)
        self.current_cell = here
        if self.last_cell is None:
            self.unique_cells = 1
            self.last_new_space_tic = tic
        elif here != self.last_cell:
            self.unique_cells += 1
            self.last_new_space_tic = tic
            self.contact_latched = False
            self.contact_pending = False
            self.blocked_tics = 0
            self.transitions[(self.last_cell, here)] += 1
        if moved >= 2.0:
            self.contact_latched = False
            self.blocked_tics = 0
        if self.blocked_tics >= 3 and not self.contact_latched:
            self.contact_latched = True
            self.contact_pending = True
            self.contact_count += 1
            signature = (here, heading_bin(angle), self.wall_side)
            self.contact_signatures[signature] += 1

        self.cells[here] += 1
        self.heading_visits[(here, heading_bin(angle))] += 1
        counts = tuple(integer(obs.get(k), 0) for k in ("kills", "items", "secrets"))
        if counts != self.last_counts:
            self.last_counts = counts
        self.current_map = str(obs.get("map", self.current_map))[:40]
        self.current_skill = integer(obs.get("skill"), self.current_skill)
        self.history.append({
            "tic": tic,
            "cell": list(here),
            "heading": heading_bin(angle),
            "moved": "yes" if moved >= 3.0 else "no",
            "health": round(health),
            "kills": counts[0],
            "items": counts[1],
            "action": self.previous_action,
        })
        self.route_history.append({
            "t": tic, "cell": list(here), "h": heading_bin(angle),
            "move": round(min(999.0, moved)),
        })
        self.previous_obs = obs
        self.previous_position = point
        self.previous_tic = tic
        self.last_cell = here
        self.last_angle = angle
        self.last_health = health
        self.last_damage_counter = damage_counter
        self.parse_objects(obs)
        return True

    # ------------------------------------------------------------------
    # Goals, weapons, and moving combat
    def compact_object(self, item):
        d = finite(item.get("distance"), 9999.0)
        err = finite(item.get("error"), 0.0)
        return {
            "id": str(item.get("id", "")),
            "name": str(item.get("name", ""))[:40],
            "category": str(item.get("category", ""))[:24],
            "range": "contact" if d < 70 else "near" if d < 220 else "medium" if d < 520 else "far",
            "side": "center" if abs(err) < 14 else "left" if err > 0 else "right",
            "level": "same" if abs(finite(item.get("height"), 0.0)) < 48 else "different",
            "hitscan": bool(item.get("hitscan")),
            "major": bool(item.get("major")),
            "body_blocker": bool(item.get("melee")) and d < 150,
        }

    def choose_goal(self, obs, tic):
        current = self.goal
        if current is not None:
            ident = current.get("id")
            latest = next((x for x in self.objects if x.get("id") == ident), None)
            if latest is None:
                if current.get("kind") == "interaction" and tic - current.get("started", tic) < 20:
                    return current
                self.goal = None
            else:
                current["item"] = latest
                d = finite(latest.get("distance"), 9999.0)
                if d < current.get("best", 99999.0) - 7.0:
                    current["best"] = d
                    current["progress"] = tic
                elif tic - current.get("progress", tic) > 80:
                    self.goal_defer[ident] = tic + 420
                    self.goal = None
                else:
                    return current

        health = finite(obs.get("health"), 100.0)
        armor = finite(obs.get("armor"), 0.0)
        weapon_ammo = finite(obs.get("weapon_ammo"), 0.0)
        weapons = obs.get("weapons") if isinstance(obs.get("weapons"), dict) else {}
        candidates = []
        for item in self.objects:
            if item.get("monster") or item.get("projectile"):
                continue
            if self.goal_defer.get(item.get("id"), -1) > tic:
                continue
            if abs(finite(item.get("height"), 9999.0)) > 82.0:
                continue
            d = finite(item.get("distance"), 9999.0)
            cat = str(item.get("category", "")).lower()
            name = str(item.get("name", "")).lower()
            if item.get("interaction"):
                if d > 600.0:
                    continue
                value = 10000.0 if cat == "exit" or "exit" in name else 8200.0
                kind = "interaction"
            elif item.get("key"):
                if d > 420.0:
                    continue
                value, kind = 5000.0, "key"
            elif cat == "weapon":
                if d > 260.0 or weapon_ammo > 8.0:
                    continue
                # Unknown weapon labels are still harmless; code only chases
                # them when the current weapon is nearly empty.
                value, kind = 1800.0, "capability"
            elif cat == "health" and health < 42.0:
                if d > 150.0:
                    continue
                value, kind = 2100.0, "survival"
            elif cat == "armor" and armor < 18.0:
                if d > 150.0:
                    continue
                value, kind = 1600.0, "survival"
            elif cat == "ammo" and weapon_ammo < 3.0:
                if d > 180.0:
                    continue
                value, kind = 1100.0, "capability"
            else:
                continue
            value -= d * (0.04 if kind == "interaction" else 0.22)
            candidates.append((value, item, kind))
        if not candidates:
            self.goal = None
            return None
        _, item, kind = max(candidates, key=lambda x: x[0])
        self.goal = {
            "id": item.get("id"), "kind": kind, "item": item,
            "started": tic, "progress": tic,
            "best": finite(item.get("distance"), 9999.0),
        }
        self.ctx.log("goal_acquired", kind=kind, name=item.get("name", ""),
                     range=self.compact_object(item).get("range"))
        return self.goal

    def goal_action(self, goal, tic):
        item = goal.get("item") or {}
        d = finite(item.get("distance"), 9999.0)
        err = finite(item.get("error"), 0.0)
        if goal.get("kind") == "interaction" and d < 72.0:
            if tic - self.last_use_tic >= 12:
                self.last_use_tic = tic
                self.force_use = False
                self.goal_turn_streak = 0
                return self.legal("use")
        if d < 46.0:
            self.goal = None
            return self.legal("forward")
        if abs(err) > max(11.0, self.turn_rate * 1.5):
            # A target bearing must not create the old indefinite in-place
            # turning loop: two aim/approach tics are followed by translation.
            if self.goal_turn_streak < 2:
                self.goal_turn_streak += 1
                return self.turn_action(1 if err > 0 else -1)
            self.goal_turn_streak = 0
            return self.moving_fallback()
        self.goal_turn_streak = 0
        return self.legal("forward")

    def weapon_switch(self, obs, tic):
        if tic - self.last_weapon_tic < 32:
            return None
        current = integer(obs.get("weapon"), 2)
        current_ammo = finite(obs.get("weapon_ammo"), 0.0)
        if current_ammo > 0.0:
            return None
        weapons = obs.get("weapons") if isinstance(obs.get("weapons"), dict) else {}
        ammo = obs.get("ammo") if isinstance(obs.get("ammo"), dict) else {}
        # Do not choose splash weapons merely because they are owned.
        for slot in (6, 4, 3, 2, 1):
            if not (bool(weapons.get(str(slot))) or bool(weapons.get(str(slot - 1)))):
                continue
            available = max(finite(ammo.get(str(slot)), 0.0),
                            finite(ammo.get(str(slot - 1)), 0.0))
            if available <= 0.0 or slot == current:
                continue
            action = "select_weapon_" + str(slot)
            if action in self.allowed:
                self.last_weapon_tic = tic
                return action
        return None

    def current_target(self, tic):
        if self.jev_target is not None:
            selected = next((x for x in self.threat_list
                             if str(x.get("id")) == self.jev_target), None)
            if selected is not None:
                # Jev may choose among visible threats, but code still requires
                # the target to meet a local urgency condition.
                d = finite(selected.get("distance"), 9999.0)
                if d < 650.0 or selected.get("major"):
                    return selected
        if not self.threat_list:
            return None
        return self.threat_list[0]

    def urgent_target(self, tic):
        target = self.current_target(tic)
        if target is None:
            return None
        d = finite(target.get("distance"), 9999.0)
        if target.get("major") or target.get("hitscan") and d < 620.0:
            return target
        if target.get("melee") and d < 300.0:
            return target
        if target.get("active") and d < 270.0:
            return target
        if d < 125.0:
            return target
        if tic <= self.damage_until and d < 430.0:
            return target
        return None

    def combat_action(self, obs, target, tic):
        health = finite(obs.get("health"), 100.0)
        d = finite(target.get("distance"), 9999.0)
        err = finite(target.get("error"), 0.0)

        if self.retreat_until >= tic:
            if tic % 3 == 0 and finite(obs.get("weapon_ammo"), 0.0) > 0:
                action = "fire_strafe_left" if self.fire_side > 0 else "fire_strafe_right"
                return self.legal(action)
            return self.backward_dodge()
        if health < 32.0 or (health < 48.0 and self.damage_amount >= 4.0):
            self.retreat_until = max(self.retreat_until, tic + 42)
            self.nav_script.clear()
            return self.backward_dodge()

        switch = self.weapon_switch(obs, tic)
        if switch:
            return switch

        # Force a translation burst after at most two turning tics.  This is the
        # key combat difference from the parent's long turn-in-place failure.
        if self.combat_move_until >= tic:
            action = "fire_strafe_left" if self.fire_side > 0 else "fire_strafe_right"
            if d < 115.0 and (target.get("melee") or health < 55.0):
                return self.backward_dodge()
            return self.legal(action if finite(obs.get("weapon_ammo"), 0.0) > 0 else
                              ("strafe_left" if self.fire_side > 0 else "strafe_right"))
        tolerance = max(8.0, self.turn_rate * 1.25)
        if abs(err) > tolerance and self.aim_turn_streak < 2:
            self.aim_turn_streak += 1
            return self.turn_action(1 if err > 0 else -1)
        self.aim_turn_streak = 0
        self.combat_move_until = tic + 4
        if finite(obs.get("weapon_ammo"), 0.0) <= 0.0:
            return self.backward_dodge()
        if d < 125.0 and (target.get("melee") or tic <= self.damage_until):
            return self.backward_dodge()
        action = "fire_strafe_left" if self.fire_side > 0 else "fire_strafe_right"
        if action not in self.allowed:
            action = "fire_forward" if "fire_forward" in self.allowed else "fire"
        return self.legal(action)

    # ------------------------------------------------------------------
    # Contact-following route automaton
    def schedule_corner(self, tic, reason, force_half=False):
        """Turn at a physical contact and resume translation.

        A repeated contact signature gets a 180-degree escape.  Otherwise the
        chosen wall side gets a 90-degree corner.  This is a finite reactive
        wall-following rule, not an unbounded turn macro.
        """
        if self.nav_script:
            return
        here = self.current_cell
        heading = heading_bin(self.last_angle)
        signature = (here, heading, self.wall_side)
        repeats = self.contact_signatures.get(signature, 0)
        if repeats >= 3 or force_half:
            angle = 180.0
            # A repeated boundary is evidence against the current wall-side
            # hypothesis.  Flip only at a measured repeat, never on a timer.
            self.wall_side *= -1
            self.stagnation_switches += 1
            self.navigator = "wall_left" if self.wall_side > 0 else "wall_right"
            self.ctx.log("route_side_switch", tic=tic, repeats=repeats,
                         side="left" if self.wall_side > 0 else "right",
                         reason=reason)
        else:
            angle = 90.0
        turn = self.turn_action(self.wall_side)
        sequence = [self.legal("backward")] * (8 if angle > 90 else 4)
        sequence += [turn] * self.turn_steps(angle)
        sequence += [self.legal("forward")] * (24 if angle > 90 else 18)
        self.nav_script = collections.deque(sequence)
        self.nav_script_kind = "reverse_contact" if angle > 90 else "corner"
        self.contact_pending = False
        self.last_recovery_tic = tic if angle > 90 else self.last_recovery_tic
        self.ctx.log("contact_macro", tic=tic, kind=self.nav_script_kind,
                     side="left" if self.wall_side > 0 else "right",
                     repeats=repeats)

    def schedule_recovery(self, tic, reason):
        if tic - self.last_recovery_tic < 90 or self.nav_script:
            return
        self.recoveries += 1
        self.last_recovery_tic = tic
        self.wall_side *= -1
        self.navigator = "wall_left" if self.wall_side > 0 else "wall_right"
        sequence = [self.legal("backward")] * 10
        sequence += [self.turn_action(self.wall_side)] * self.turn_steps(180.0)
        sequence += [self.legal("forward")] * 30
        self.nav_script = collections.deque(sequence)
        self.nav_script_kind = "stagnation_recovery"
        self.contact_pending = False
        self.ctx.log("stagnation_recovery", tic=tic, reason=reason,
                     unique_cells=self.unique_cells,
                     side="left" if self.wall_side > 0 else "right")

    def open_sweep_action(self, tic):
        # Astra may select this as a deliberate escape from a wall-following
        # loop.  It is still finite and returns to contact following.
        if self.sweep_left <= 0:
            self.sweep_left = 62
            self.nav_script = collections.deque(
                [self.turn_action(self.wall_side)] * self.turn_steps(90.0)
            )
            self.nav_script_kind = "open_scan"
        else:
            self.sweep_left -= 1
        return self.legal("forward")

    def navigation_action(self, tic):
        if self.nav_script:
            action = self.nav_script.popleft()
            if not self.nav_script:
                self.nav_script_kind = ""
            return self.legal(action)
        if self.navigator == "open_sweep":
            return self.open_sweep_action(tic)
        return self.legal("forward")

    def goal_supported_use(self, tic):
        if self.goal and self.goal.get("kind") == "interaction":
            item = self.goal.get("item") or {}
            return finite(item.get("distance"), 9999.0) < 115.0
        return self.contact_pending or self.blocked_tics >= 3

    def choose_action(self, obs, tic):
        if self.force_use and self.force_use_until >= 0 and tic > self.force_use_until:
            self.force_use = False
        # 1. Projectiles and immediate threats always outrank route hints.
        if self.projectiles:
            projectile = self.projectiles[0]
            if finite(projectile.get("distance"), 9999.0) < 205.0:
                self.fire_side *= -1
                return self.legal("strafe_left" if self.fire_side > 0 else "strafe_right")
        target = self.urgent_target(tic)
        if target is not None:
            self.goal = None
            return self.combat_action(obs, target, tic)

        # 2. A rendered interaction/key gets a bounded approach, but optional
        # health/ammo labels never steal the route when the player is safe.
        goal = self.choose_goal(obs, tic)
        if goal is not None and goal.get("kind") in {"interaction", "key", "capability", "survival"}:
            if self.force_use and self.goal_supported_use(tic) and tic - self.last_use_tic >= 12:
                self.force_use = False
                self.last_use_tic = tic
                return self.legal("use")
            return self.goal_action(goal, tic)

        # 3. Jev can request a route recovery, but only in a safe state and only
        # as the same finite code-owned macro.
        if self.jev_recover and self.blocked_tics < 3 and not self.nav_script:
            self.jev_recover = False
            self.schedule_recovery(tic, "jev_macro")
            if self.nav_script:
                return self.legal(self.nav_script.popleft())

        # 4. Contact is handled before an old script so a newly encountered wall
        # cannot leave the player repeating a stale corner.
        if self.contact_pending and not self.nav_script:
            if (self.door_policy != "sparse" and
                    tic - self.last_use_tic >= 12):
                self.last_use_tic = tic
                return self.legal("use")
            self.schedule_corner(tic, "contact")
            if self.nav_script:
                return self.legal(self.nav_script.popleft())

        if self.nav_script:
            return self.navigation_action(tic)

        # 5. A Jev USE vote is constrained by actual contact or an actual
        # rendered interaction.  It cannot invent a door in open space.
        if self.force_use and self.goal_supported_use(tic) and tic - self.last_use_tic >= 12:
            self.force_use = False
            self.last_use_tic = tic
            return self.legal("use")

        # 6. Stagnation changes the route hypothesis instead of adding another
        # short turn.  A long no-new-space interval is the trigger.
        no_space = tic - self.last_new_space_tic
        if no_space > 260 and tic - self.last_recovery_tic >= 100:
            self.schedule_recovery(tic, "no_new_space")
            if self.nav_script:
                return self.legal(self.nav_script.popleft())

        # 7. Sparse USE is a safety net for unlabeled switches/doors.  Contact
        # USE remains the principal interaction mechanism.
        period = 48 if self.door_policy == "often" else 120 if self.door_policy == "sparse" else 82
        if tic - self.last_use_tic >= period and not self.threat_list:
            self.last_use_tic = tic
            return self.legal("use")
        return self.navigation_action(tic)

    # ------------------------------------------------------------------
    # Jev: local macro arbitration, never raw action generation
    def jev_state(self, obs, tic):
        health = finite(obs.get("health"), 100.0)
        armor = finite(obs.get("armor"), 0.0)
        ammo = finite(obs.get("weapon_ammo"), 0.0)
        goal_items = [x for x in self.objects if not x.get("monster") and not x.get("projectile")]
        goal_items.sort(key=lambda x: (0 if x.get("interaction") or x.get("key") else 1,
                                       finite(x.get("distance"), 9999.0)))
        no_space = self.previous_tic - self.last_new_space_tic
        return {
            "task": "Reach and activate the map exit alive; optional completion is secondary.",
            "visibility": "Only rendered labels are supplied; do not infer hidden geometry.",
            "map": self.current_map,
            "skill": self.current_skill,
            "health": "critical" if health < 25 else "wounded" if health < 60 else "healthy",
            "armor": "none" if armor < 10 else "present",
            "ammo": "empty" if ammo <= 0 else "low" if ammo < 5 else "available",
            "damage_recent": tic <= self.damage_until,
            "navigator": {
                "policy": self.navigator,
                "wall_side": "left" if self.wall_side > 0 else "right",
                "phase": self.nav_script_kind or "forward_contact_search",
                "contact": self.contact_pending,
                "blocked": self.blocked_tics >= 3,
                "stagnating": no_space > 150,
                "unique_cells": min(500, self.unique_cells),
                "contacts": min(500, self.contact_count),
                "side_switches": min(40, self.stagnation_switches),
            },
            "threats": [self.compact_object(x) for x in self.threat_list[:5]],
            "visible_goals": [self.compact_object(x) for x in goal_items[:5]],
            "recent": list(self.history)[-6:],
            "policy": {
                "combat": self.combat_policy,
                "doors": self.door_policy,
            },
        }

    def jev_questions(self, state):
        macros = {
            "continue": "Keep the current finite wall/contact navigator; no local evidence requires a route change.",
            "left_wall": "At a safe boundary, follow the left side of encountered geometry; this changes only the finite navigator policy.",
            "right_wall": "At a safe boundary, follow the right side of encountered geometry; this changes only the finite navigator policy.",
            "use_contact": "A listed contact, blocked face, door/switch/exit, or key approach supports one bounded USE pulse.",
            "reverse": "The measured navigator is stagnating or unsafe; request one finite backward/turn/forward recovery.",
            "retreat": "Recent damage or a close threat justifies a short moving retreat; do not stand and turn.",
            "uncertain": "Evidence is insufficient; do not change the current code-owned macro.",
        }
        first = dict(macros)
        second = dict(reversed(list(macros.items())))
        target_criteria = {"none": "No currently rendered monster should override the code threat ranking."}
        for item in self.threat_list[:5]:
            target_criteria[str(item.get("id"))] = {
                "what": str(item.get("name", "visible monster")),
                "facts": "Select only if its supplied range, side and threat flags support attention.",
            }
        return {
            "macro_a": Choice(
                "Choose one finite local macro from state.navigator, state.threats and state.visible_goals. Do not output a button, route, coordinate, arithmetic result, or hidden object.",
                first,
            ),
            "macro_b": Choice(
                "Independently choose the safest useful macro from the exact same named JSON fields. Code owns movement, bearings, collision detection, and legality.",
                second,
            ),
            "target": Choice(
                "Which currently rendered monster is locally most urgent, or none? Never choose an absent ID.",
                target_criteria,
            ),
            "open": Noul(
                "Do the supplied facts justify one USE pulse soon? Yes only for a rendered interaction/key approach or an observed blocked contact, never for an imagined door."
            ),
        }

    def dispatch_jev(self, obs, tic):
        if getattr(self.ctx, "system1", True) is None:
            return
        if self.jev_future is not None or tic < self.next_jev_tic:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if not isinstance(remaining, dict):
            return
        if finite(remaining.get("system1_calls"), 0.0) <= 10 or finite(remaining.get("wall_s"), 0.0) < 2.0:
            return
        state = self.jev_state(obs, tic)
        questions = self.jev_questions(state)
        self.jev_request_number += 1
        self.jev_meta = {"tic": tic, "policy": self.policy_version,
                         "number": self.jev_request_number}
        danger = bool(self.threat_list or self.projectiles or tic <= self.damage_until)
        self.next_jev_tic = tic + (14 if danger else 48)
        try:
            self.jev_future = self.ctx.spawn(
                lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=0.72)
            )
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError):
            self.jev_future = None

    def poll_jev(self, tic):
        if self.jev_future is None or not self.jev_future.done():
            return
        future = self.jev_future
        meta = self.jev_meta or {}
        self.jev_future = None
        self.jev_meta = None
        try:
            answers = future.result()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, EpisodeOver, CancelledError):
            return
        except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
            self.ctx.log("jev_failure", error=type(exc).__name__)
            return
        age = tic - integer(meta.get("tic"), tic)
        if age < 0 or age > (34 if self.threat_list else 92):
            self.ctx.log("jev_discard", age=age, reason="old_observation")
            return
        if integer(meta.get("policy"), -1) != self.policy_version:
            self.ctx.log("jev_discard", age=age, reason="policy_changed")
            return
        macros = {"continue", "left_wall", "right_wall", "use_contact",
                  "reverse", "retreat", "uncertain"}
        a = choice_value(answers, "macro_a", macros)
        b = choice_value(answers, "macro_b", macros)
        macro = a if a is not None and a == b else None
        if macro == "left_wall":
            self.wall_side = 1
            self.navigator = "wall_left"
        elif macro == "right_wall":
            self.wall_side = -1
            self.navigator = "wall_right"
        elif macro == "use_contact":
            self.force_use = True
            self.force_use_until = tic + 70
        elif macro == "reverse":
            self.jev_recover = True
        elif macro == "retreat":
            health = finite((self.previous_obs or {}).get("health"), 100.0)
            if tic <= self.damage_until or health < 58.0:
                self.retreat_until = max(self.retreat_until, tic + 38)
        target_allowed = {str(x.get("id")) for x in self.threat_list[:5]} | {"none"}
        target = choice_value(answers, "target", target_allowed, 0.54)
        if target is not None:
            self.jev_target = None if target == "none" else target
        if yes_value(answers, "open"):
            self.force_use = True
            self.force_use_until = tic + 70
        if self.force_use and tic > self.force_use_until:
            self.force_use = False
        self.ctx.log("jev_macro", macro=macro, target=self.jev_target,
                     use=bool(self.force_use), age=age)

    # ------------------------------------------------------------------
    # Astra: route supervisor, not a per-tic controller
    def astra_payload(self, obs, tic):
        state = self.jev_state(obs, tic)
        return {
            "state": state,
            "route_evidence": {
                "unique_cells": self.unique_cells,
                "contact_count": self.contact_count,
                "repeated_contact_signatures": sum(1 for n in self.contact_signatures.values() if n >= 2),
                "recoveries": self.recoveries,
                "recent_route": list(self.route_history)[-24:],
            },
            "current_policy": {
                "navigator": self.navigator,
                "wall_side": "left" if self.wall_side > 0 else "right",
                "combat": self.combat_policy,
                "doors": self.door_policy,
            },
        }

    def launch_astra(self, obs, tic):
        if self.astra_future is not None or self.astra_calls >= 2 or tic < self.next_astra_tic:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if not isinstance(remaining, dict):
            return
        if finite(remaining.get("system2_calls"), 0.0) <= 0 or finite(remaining.get("wall_s"), 0.0) < 150.0:
            return
        payload = self.astra_payload(obs, tic)
        prompt = (
            "You are the slow route supervisor for a real-time Classic Doom controller. "
            "The public labels are view-visible only. Code already owns exact arithmetic, "
            "collision/contact detection, legal buttons, weapon safety, moving combat, and "
            "the finite macro executor. Choose a robust exploration policy from measured facts, "
            "not a hidden map. Exit success dominates optional kills/items. Return ONLY one short "
            "JSON object with these bounded fields: navigator (wall_left|wall_right|alternate|open_sweep), "
            "combat (protect_health|clear_blockers|avoid_optional), doors (use_on_contact|often|sparse), "
            "reason (at most 100 characters). Do not output buttons, coordinates, a route, code, "
            "or prose outside the JSON. Keep the current policy when evidence is insufficient. "
            "A late result is advisory and will be applied only by code at a safe boundary.\n" +
            json.dumps(payload, separators=(",", ":"), allow_nan=False)
        )
        session = self.astra_session
        self.astra_calls += 1
        try:
            self.astra_future = self.ctx.spawn(
                lambda p=prompt, s=session: self.ctx.think(
                    p, session_id=s, timeout_s=190.0
                )
            )
            # Give the first review time to observe a meaningful route.  A
            # stagnation-triggered second review is allowed earlier below.
            self.next_astra_tic = tic + (5200 if self.astra_calls == 1 else 1000000)
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError):
            self.astra_future = None

    def maybe_launch_second_astra(self, obs, tic):
        if self.astra_calls == 1 and self.astra_future is None:
            if tic - self.last_new_space_tic > 420 and tic - self.next_astra_tic > -4500:
                self.next_astra_tic = tic
        self.launch_astra(obs, tic)

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
        except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
            self.ctx.log("astra_failure", error=type(exc).__name__)
            return
        proposal = json_object(getattr(result, "text", ""))
        accepted = {}
        if proposal.get("navigator") in {"wall_left", "wall_right", "alternate", "open_sweep"}:
            accepted["navigator"] = proposal.get("navigator")
        if proposal.get("combat") in {"protect_health", "clear_blockers", "avoid_optional"}:
            accepted["combat"] = proposal.get("combat")
        if proposal.get("doors") in {"use_on_contact", "often", "sparse"}:
            accepted["doors"] = proposal.get("doors")
        self.astra_session = getattr(result, "session_id", self.astra_session)
        if not accepted:
            self.ctx.log("astra_invalid", fallback="contact_navigator")
            return
        if "navigator" in accepted:
            navigator = accepted["navigator"]
            self.navigator = navigator
            if navigator == "wall_left":
                self.wall_side = 1
            elif navigator == "wall_right":
                self.wall_side = -1
            elif navigator == "alternate":
                # Keep the currently successful side until stagnation, then
                # the code's measured repeat rule changes it.
                self.navigator = "alternate"
        if "combat" in accepted:
            self.combat_policy = accepted["combat"]
        if "doors" in accepted:
            self.door_policy = accepted["doors"]
        self.policy_version += 1
        self.jev_target = None
        self.jev_recover = False
        self.ctx.log("astra_route_policy", policy={
            "navigator": self.navigator,
            "wall_side": "left" if self.wall_side > 0 else "right",
            "combat": self.combat_policy,
            "doors": self.door_policy,
        }, reason=str(proposal.get("reason", ""))[:100])

    # ------------------------------------------------------------------
    # Lifecycle
    def run_episode(self, game):
        ctx = self.ctx
        names = getattr(game.action_spec, "names", ()) or ()
        self.allowed = set(names)
        if not self.allowed:
            return
        candidate_noop = getattr(game.action_spec, "noop", "noop")
        self.noop = candidate_noop if candidate_noop in self.allowed else next(iter(self.allowed))
        try:
            # Establish a safe held action before any observation/model work.
            ctx.act(self.noop)
            while not ctx.done and not bool(getattr(game, "done", False)):
                obs = as_dict(ctx.observe())
                if not obs:
                    ctx.wait_for_tick()
                    continue
                if bool(obs.get("terminal")) or bool(obs.get("dead")):
                    break
                tic = integer(obs.get("tic"), integer(obs.get("elapsed_tics"), 0))
                if tic <= self.previous_tic:
                    ctx.wait_for_tick()
                    continue
                if not self.update(obs):
                    ctx.wait_for_tick()
                    continue
                if ctx.done or bool(getattr(game, "done", False)):
                    break
                self.poll_jev(tic)
                self.poll_astra(tic)
                self.dispatch_jev(obs, tic)
                self.maybe_launch_second_astra(obs, tic)
                action = self.choose_action(obs, tic)
                action = self.legal(action)
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
                self.ctx.memory["contact_navigator_summary"] = {
                    "map": self.current_map,
                    "unique_cells": self.unique_cells,
                    "contacts": self.contact_count,
                    "recoveries": self.recoveries,
                    "side_switches": self.stagnation_switches,
                    "navigator": self.navigator,
                    "policy": self.policy_version,
                }
            except Exception:
                pass


def build(ctx):
    return Harness(ctx)
