"""Candidate 0006: a moving-macro executor with semantic model supervision.

The controller deliberately does not let a model own a button for an unbounded
period.  Code turns observations into short, measurable motor macros, records
which headings actually moved, and pulses USE at measured contacts.  Jev
chooses a local intent and a currently visible target; Astra changes only a
small durable phase policy.  Neither model can emit an arbitrary button.
"""
from __future__ import annotations

import collections
import json
import math
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


# The adaptor's public action names.  The controller checks each one against
# action_spec, so reduced action sets used by pre-flight remain legal.
MOVE_ACTIONS = {
    "forward", "backward", "strafe_left", "strafe_right",
    "forward_left", "forward_right", "backward_left", "backward_right",
    "fire_forward", "fire_strafe_left", "fire_strafe_right",
}
TURN_ACTIONS = {"turn_left", "turn_right"}
FIRE_ACTIONS = {"fire", "fire_forward", "fire_strafe_left", "fire_strafe_right"}

# Relative movement bearings follow the compact observation convention used by
# the public adaptor: 0 is the player's forward bearing and positive is left.
ACTION_OFFSET = {
    "forward": 0.0,
    "backward": 180.0,
    "strafe_left": 90.0,
    "strafe_right": -90.0,
    "forward_left": 45.0,
    "forward_right": -45.0,
    "backward_left": 135.0,
    "backward_right": -135.0,
    "fire_forward": 0.0,
    "fire_strafe_left": 90.0,
    "fire_strafe_right": -90.0,
}

INTERACT_CATEGORIES = {
    "exit", "switch", "door", "teleporter", "teleport", "exitdoor",
}
INTERACT_WORDS = ("exit", "switch", "door", "teleport")

HITSCANNERS = {
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS",
    "SpiderMastermind",
}
MAJOR_THREATS = {
    "Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind",
    "Revenant",
}
BODY_BLOCKERS = {"Demon", "Spectre", "LostSoul"}

# These are action slots, not a claim about a particular map's inventory.
WEAPON_SLOT = {
    "Pistol": 2,
    "Shotgun": 3,
    "SuperShotgun": 3,
    "Chaingun": 4,
    "RocketLauncher": 5,
    "PlasmaRifle": 6,
    "BFG9000": 7,
    "Chainsaw": 1,
}


def finite(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    value = finite(value, default)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def json_dict(value):
    """Decode an adaptor value without allowing bad model/test data to escape."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def wrap(angle):
    return (finite(angle) + 180.0) % 360.0 - 180.0


def distance(a, b):
    try:
        return math.hypot(finite(a[0]) - finite(b[0]), finite(a[1]) - finite(b[1]))
    except (IndexError, TypeError):
        return 0.0


def bearing(a, b):
    try:
        return math.degrees(math.atan2(finite(b[1]) - finite(a[1]),
                                       finite(b[0]) - finite(a[0]))) % 360.0
    except (IndexError, TypeError):
        return 0.0


def point_cell(point, size=64.0):
    return (
        int(math.floor(finite(point[0]) / size)),
        int(math.floor(finite(point[1]) / size)),
        int(math.floor((finite(point[2]) + 16.0) / 48.0)),
    )


def angle_octant(angle):
    return int(math.floor((finite(angle) % 360.0 + 22.5) / 45.0)) % 8


def range_bucket(value):
    value = finite(value, 99999.0)
    if value < 70:
        return "contact"
    if value < 180:
        return "near"
    if value < 450:
        return "medium"
    return "far"


def side_bucket(error):
    error = wrap(error)
    if abs(error) < 12:
        return "ahead"
    return "left" if error > 0 else "right"


def category_lower(value):
    return str(value or "").strip().lower()


def is_interactable(obj):
    category = category_lower(obj.get("category"))
    name = str(obj.get("name", "")).lower()
    return category in INTERACT_CATEGORIES or any(word in name for word in INTERACT_WORDS)


def is_live_monster(obj):
    return (category_lower(obj.get("category")) == "monster"
            and not str(obj.get("name", "")).lower().startswith("dead"))


def choice_answer(answers, key, legal, threshold=0.55):
    """Return a high-enough probability Choice value, never raise on garbage."""
    if not isinstance(answers, dict):
        return None
    try:
        answer = answers.get(key)
        value = getattr(answer, "value", None)
        probabilities = getattr(answer, "probabilities", None)
        if not isinstance(value, str) or value not in legal or not isinstance(probabilities, dict):
            return None
        probability = finite(probabilities.get(value), 0.0)
        if probability < threshold:
            return None
        confidence = getattr(answer, "confidence", None)
        if confidence is not None and finite(confidence, 0.0) < 0.10:
            return None
        return value
    except Exception:
        return None


def noul_yes(answers, key, threshold=0.76):
    if not isinstance(answers, dict):
        return False
    try:
        answer = answers.get(key)
        return finite(getattr(answer, "value", 0.0), 0.0) >= threshold
    except Exception:
        return False


def parse_json_object(text):
    """Extract the first usable JSON object from bounded Astra text."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    # Astra is asked for a tiny object.  Limiting scanning also protects the
    # controller if a stand-in returns a very large arbitrary string.
    for i, char in enumerate(text[:24000]):
        if char != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text[i:])
            if isinstance(result, dict):
                return result
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return {}


class MovingMacroController:
    """Observation-driven controller whose commitments expire or prove useful."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.legal = set()
        self.noop = "noop"
        self.started = 0.0

        self.previous = None
        self.previous_action = "noop"
        self.last_tic = -1
        self.last_action_tic = -1
        self.current_action = "noop"
        self.action_until = -1
        self.translate_until = -1
        self.turn_streak = 0
        self.motion_action = None
        self.motion_origin = None
        self.motion_started = -1
        self.blocked_event = None
        self.last_progress = -1

        self.cells = {}
        self.edges = collections.defaultdict(set)
        self.attempts = collections.Counter()
        self.blocked = {}
        self.trail = collections.deque(maxlen=180)
        self.sweep_index = 0
        self.goal = None
        self.goal_serial = 0
        self.landmarks = {}
        self.last_counts = (0, 0, 0)
        self.last_damage_tic = -10000
        self.damage_window = collections.deque(maxlen=10)
        self.visible_threats = []
        self.visible_objects = []
        self.projectiles = []

        # Doom's right turn normally decreases ANGLE.  It is calibrated from
        # observations whenever a turn macro has an observable angle delta.
        self.right_angle_sign = -1.0
        self.turn_rate = 5.6
        self.last_side_change = -10000
        self.dodge_side = 1
        self.last_use_tic = -10000
        self.use_faces = {}
        self.after_use = None
        self.last_weapon_switch = -10000
        self.last_target_id = None
        self.last_shot_tic = -10000
        self.last_kill = 0

        self.intent = "advance"
        self.intent_expires = -1.0
        self.intent_target = None
        self.intent_side = "right"
        self.intent_version = 0

        self.plan = {
            "phase": "sweep",
            "priority": "exit",
            "combat": "clear_near",
            "side": "right",
            "focus_id": None,
            "interaction": "use_on_contact",
        }
        self.plan_version = 0
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.next_astra = 0.8
        self.last_astra = -10000.0

        self.jev_future = None
        self.jev_meta = None
        self.jev_request = 0
        self.next_jev = 0.25
        self.last_jev_failure = -10000.0

    # ----- observation and bookkeeping ---------------------------------

    def normalize(self, raw, now):
        raw = json_dict(raw)
        player = raw.get("player")
        if not isinstance(player, dict):
            player = {}
        p = [finite(player.get("x")), finite(player.get("y")), finite(player.get("z"))]
        angle = finite(player.get("angle")) % 360.0
        tic = integer(raw.get("tic"), integer(now * 35.0))
        result = {
            "raw": raw,
            "tic": tic,
            "now": max(0.0, finite(now)),
            "p": p,
            "angle": angle,
            "health": finite(raw.get("health"), 100.0),
            "armor": finite(raw.get("armor")),
            "dead": bool(raw.get("dead") or raw.get("terminal")),
            "attack_ready": bool(raw.get("attack_ready")),
            "weapon": integer(raw.get("weapon"), 2),
            "weapon_ammo": finite(raw.get("weapon_ammo")),
            "ammo": raw.get("ammo") if isinstance(raw.get("ammo"), dict) else {},
            "weapons": raw.get("weapons") if isinstance(raw.get("weapons"), dict) else {},
            "kills": integer(raw.get("kills")),
            "items": integer(raw.get("items")),
            "secrets": integer(raw.get("secrets")),
            "damage": finite(raw.get("damage_taken")),
            "terminal": bool(raw.get("terminal") or raw.get("dead")),
            "labels": [],
            "threats": [],
            "objects": [],
            "projectiles": [],
        }
        labels = raw.get("visible")
        if not isinstance(labels, list):
            labels = []
        for index, label in enumerate(labels[:256]):
            if not isinstance(label, dict):
                continue
            world = label.get("world")
            if not isinstance(world, (list, tuple)) or len(world) < 3:
                continue
            q = [finite(world[0]), finite(world[1]), finite(world[2])]
            name = str(label.get("name", "unknown"))[:80]
            category = str(label.get("category", "Unknown"))[:32]
            ident = str(label.get("id", "%s:%s" % (name, index)))[:100]
            box = label.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                bbox = [finite(v) for v in box[:4]]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            d = distance(p, q)
            err = wrap(bearing(p, q) - angle)
            obj = {
                "id": ident,
                "name": name,
                "category": category,
                "p": q,
                "distance": d,
                "error": err,
                "dz": q[2] - p[2],
                "range": range_bucket(d),
                "side": side_bucket(err),
                "bbox": bbox,
                "seen_tic": tic,
                "active": False,
                "hitscan": name in HITSCANNERS,
                "major": name in MAJOR_THREATS,
                "body_blocker": name in BODY_BLOCKERS,
            }
            old = self.landmarks.get(ident)
            if old is not None and distance(old.get("p", q), q) > 2.0:
                obj["moving"] = True
            else:
                obj["moving"] = False
            obj["active"] = obj["moving"] or d < 180.0 or (tic - self.last_damage_tic < 45)
            result["labels"].append(obj)
            result["objects"].append(obj)
            if is_live_monster(obj):
                result["threats"].append(obj)
            elif category_lower(category) in ("projectile", "missile"):
                result["projectiles"].append(obj)
            elif (category_lower(category) in INTERACT_CATEGORIES
                  or category_lower(category) in {
                      "key", "weapon", "health", "armor", "ammo", "powerup", "power-up"
                  }
                  or is_interactable(obj)):
                self.landmarks[ident] = obj
        result["threats"].sort(key=lambda x: self.threat_score(x), reverse=True)
        result["projectiles"].sort(key=lambda x: x["distance"])
        self.visible_threats = result["threats"]
        self.visible_objects = result["objects"]
        self.projectiles = result["projectiles"]
        return result

    def threat_score(self, obj):
        d = finite(obj.get("distance"), 9999.0)
        score = 0.0
        if obj.get("major"):
            score += 240.0
        if obj.get("hitscan"):
            score += 150.0
        if obj.get("body_blocker"):
            score += 110.0
        if obj.get("active"):
            score += 50.0
        if d < 100:
            score += 210.0
        elif d < 240:
            score += 100.0
        elif d < 600:
            score += 35.0
        score -= d * 0.16
        score -= abs(finite(obj.get("error"))) * 0.08
        return score

    def update(self, obs):
        tic = obs["tic"]
        p = obs["p"]
        previous = self.previous
        self.blocked_event = None
        if previous is not None:
            dt = tic - previous["tic"]
            if 0 < dt <= 10 and self.previous_action in TURN_ACTIONS:
                delta = wrap(obs["angle"] - previous["angle"])
                if abs(delta) > 0.25:
                    observed_rate = abs(delta) / max(1, dt)
                    self.turn_rate = max(1.0, min(16.0, 0.75 * self.turn_rate + 0.25 * observed_rate))
                    if self.previous_action == "turn_right":
                        self.right_angle_sign = 1.0 if delta > 0 else -1.0
            loss = max(0.0, previous["health"] - obs["health"])
            delta_damage = max(loss, obs["damage"] - previous["damage"])
            if delta_damage > 0.0:
                self.last_damage_tic = tic
                self.damage_window.append((tic, delta_damage))
            counts = (obs["kills"], obs["items"], obs["secrets"])
            if counts != self.last_counts:
                self.last_progress = tic
            if obs["kills"] > previous["kills"]:
                self.last_kill = tic
            self.last_counts = counts

            old_cell = point_cell(previous["p"])
            new_cell = point_cell(p)
            moved = distance(previous["p"], p)
            if old_cell != new_cell or moved > 12.0:
                self.last_progress = tic
                self.trail.append((tic, list(p), old_cell, new_cell))
                if old_cell != new_cell:
                    self.edges[old_cell].add(new_cell)
                    if abs(p[2] - previous["p"][2]) < 36:
                        self.edges[new_cell].add(old_cell)

            if self.previous_action in MOVE_ACTIONS:
                if self.motion_action != self.previous_action or self.motion_origin is None:
                    self.motion_action = self.previous_action
                    self.motion_origin = list(previous["p"])
                    self.motion_started = previous["tic"]
                elif moved > 9.0:
                    # Measure each successful segment separately.  A later
                    # collision is therefore not hidden by an old displacement.
                    self.motion_origin = list(p)
                    self.motion_started = tic
                elif tic - self.motion_started >= 10:
                    heading = (previous["angle"] + ACTION_OFFSET.get(self.previous_action, 0.0)) % 360.0
                    face = (point_cell(p), angle_octant(heading))
                    self.blocked[face] = tic + 175
                    self.attempts[face] += 1
                    self.blocked_event = {
                        "face": face,
                        "heading": heading,
                        "action": self.previous_action,
                        "p": list(p),
                    }
                    # Re-arm only after the action changes or the player moves.
                    self.motion_origin = list(p)
                    self.motion_started = tic
            else:
                self.motion_action = None
                self.motion_origin = None
                self.motion_started = -1
        else:
            self.last_progress = tic
            self.trail.append((tic, list(p), None, point_cell(p)))
            self.last_counts = (obs["kills"], obs["items"], obs["secrets"])

        cell = point_cell(p)
        node = self.cells.setdefault(cell, {"p": list(p), "visits": 0})
        node["visits"] += 1
        node["p"] = list(p)
        if previous is None or point_cell(previous["p"]) != cell:
            self.last_progress = tic

        # Keep a short memory of rendered objects.  The object remains a
        # candidate only while its observation is recent; this avoids treating
        # a remembered pickup as omniscient map data.
        visible_ids = {obj["id"] for obj in obs["objects"]}
        for ident, obj in list(self.landmarks.items()):
            if ident not in visible_ids:
                obj["visible"] = False
            else:
                obj["visible"] = True
            if tic - integer(obj.get("seen_tic"), tic) > 420:
                del self.landmarks[ident]

        while self.damage_window and tic - self.damage_window[0][0] > 105:
            self.damage_window.popleft()
        self.previous = obs

    # ----- route and interaction memory --------------------------------

    def current_damage(self):
        return sum(finite(row[1]) for row in self.damage_window)

    def same_level(self, obj, obs):
        return abs(finite(obj.get("dz"))) <= 52.0

    def landmark_value(self, obj, obs):
        """Rank only observations, with exit/key capability above optional loot."""
        if not obj.get("visible", True):
            return -10000.0
        if not self.same_level(obj, obs):
            return -10000.0
        d = finite(obj.get("distance"), 99999.0)
        if d > 950.0:
            return -10000.0
        cat = category_lower(obj.get("category"))
        name = str(obj.get("name", ""))
        value = 0.0
        if cat == "exit" or "exit" in name.lower():
            value = 1200.0
        elif cat == "switch" or "switch" in name.lower() or "teleport" in name.lower():
            value = 430.0
        elif cat == "key":
            value = 600.0
        elif cat == "weapon":
            slot = WEAPON_SLOT.get(name)
            value = 430.0 if slot in (3, 4, 6) and not self.weapon_owned(obs, slot) else 75.0
        elif cat == "health":
            value = 460.0 if obs["health"] < 35 else 260.0 if obs["health"] < 70 else 15.0
        elif cat == "armor":
            value = 230.0 if obs["armor"] < 40 else 40.0
        elif cat == "ammo":
            value = 100.0 if obs["weapon_ammo"] < 12 else 15.0
        elif cat in ("door", "teleporter", "teleport"):
            value = 330.0
        else:
            value = 0.0
        if self.plan.get("priority") == "key" and cat == "key":
            value += 120.0
        if self.plan.get("priority") == "weapon" and cat in ("weapon", "ammo"):
            value += 90.0
        if self.plan.get("priority") == "recover" and cat in ("health", "armor"):
            value += 100.0
        if self.plan.get("focus_id") == obj.get("id"):
            value += 80.0
        value -= d * 0.32
        value -= max(0.0, abs(finite(obj.get("dz"))) - 20.0) * 2.0
        return value

    def useful_landmarks(self, obs):
        result = []
        for obj in self.visible_objects:
            if is_live_monster(obj):
                continue
            value = self.landmark_value(obj, obs)
            if value > 0:
                result.append((value, obj))
        result.sort(key=lambda pair: pair[0], reverse=True)
        return [obj for _, obj in result[:8]]

    def direction_point(self, p, angle, length=192.0):
        radians = math.radians(angle % 360.0)
        return [p[0] + length * math.cos(radians),
                p[1] + length * math.sin(radians), p[2]]

    def explore_point(self, obs):
        """Choose a finite radial probe, or walk the measured breadcrumb graph."""
        p = obs["p"]
        current = point_cell(p)
        current_angle = obs["angle"]
        candidates = []
        for index in range(8):
            angle = index * 45.0
            face = (current, index)
            if self.blocked.get(face, -1) > obs["tic"]:
                continue
            destination = self.direction_point(p, angle)
            dest_cell = point_cell(destination)
            tries = self.attempts.get(face, 0)
            novelty = 5.0 if dest_cell not in self.cells else 0.0
            turn_cost = abs(wrap(angle - current_angle)) / 90.0
            # The Astra side is a tie-breaker, not a route claim.
            side_bias = 0.3 if (self.plan.get("side") == "left" and angle > current_angle) else 0.0
            if self.plan.get("side") == "right" and angle < current_angle:
                side_bias += 0.3
            score = novelty + side_bias - tries * 2.8 - turn_cost
            candidates.append((score, index, destination))
        if candidates:
            _, index, destination = max(candidates, key=lambda row: row[0])
            self.attempts[(current, index)] += 1
            return {
                "kind": "explore",
                "id": None,
                "p": destination,
                "direction": index,
                "started": obs["tic"],
                "best": distance(p, destination),
                "last_improve": obs["tic"],
                "serial": self.goal_serial,
            }

        # All local headings were tested.  Return to the least visited measured
        # neighbor, never to an invented coordinate or hidden sector.
        neighbors = list(self.edges.get(current, ()))
        if neighbors:
            neighbor = min(neighbors, key=lambda cell: self.cells.get(cell, {}).get("visits", 0))
            q = self.cells.get(neighbor, {}).get("p", p)
            return {
                "kind": "backtrack", "id": None, "p": list(q),
                "direction": angle_octant(bearing(p, q)),
                "started": obs["tic"], "best": distance(p, q),
                "last_improve": obs["tic"], "serial": self.goal_serial,
            }

        # A fresh deterministic sweep is safer than an absorbing forward loop.
        angle = (self.sweep_index % 8) * 45.0
        self.sweep_index += 1
        q = self.direction_point(p, angle, 160.0)
        return {
            "kind": "sweep", "id": None, "p": q,
            "direction": angle_octant(angle), "started": obs["tic"],
            "best": distance(p, q), "last_improve": obs["tic"],
            "serial": self.goal_serial,
        }

    def choose_goal(self, obs, force=False):
        if self.goal is not None and not force:
            goal = self.goal
            if goal.get("kind") == "landmark":
                current = next((x for x in self.visible_objects if x.get("id") == goal.get("id")), None)
                if current is not None and self.landmark_value(current, obs) > 0:
                    goal["p"] = list(current["p"])
                    return goal
                # Do not chase a label that has disappeared for more than a
                # short approach window; it is not a map oracle.
                if obs["tic"] - integer(goal.get("last_seen"), obs["tic"]) < 20:
                    return goal
                self.goal = None
            else:
                d = distance(obs["p"], goal.get("p", obs["p"]))
                if d < 34:
                    self.goal = None
                elif obs["tic"] - integer(goal.get("last_improve"), obs["tic"]) > 26:
                    self.blocked[(point_cell(obs["p"]), integer(goal.get("direction"), 0) % 8)] = obs["tic"] + 175
                    self.goal = None
                elif obs["tic"] - integer(goal.get("started"), obs["tic"]) < 145:
                    return goal
                else:
                    self.goal = None

        if self.goal is not None and not force:
            return self.goal
        self.goal_serial += 1
        candidates = self.useful_landmarks(obs)
        selected = None
        if self.intent_target:
            selected = next((x for x in candidates if x.get("id") == self.intent_target), None)
        if selected is None and self.plan.get("focus_id"):
            selected = next((x for x in candidates if x.get("id") == self.plan.get("focus_id")), None)
        if selected is None and candidates:
            selected = candidates[0]
        if selected is not None:
            self.goal = {
                "kind": "landmark", "id": selected["id"], "p": list(selected["p"]),
                "started": obs["tic"], "last_improve": obs["tic"],
                "best": distance(obs["p"], selected["p"]),
                "last_seen": obs["tic"], "serial": self.goal_serial,
            }
        else:
            self.goal = self.explore_point(obs)
            self.goal["serial"] = self.goal_serial
        return self.goal

    def weapon_owned(self, obs, slot):
        weapons = obs.get("weapons", {})
        def value(key):
            return bool(weapons.get(str(key), weapons.get(key, False)))
        if slot == 1:
            return value(0) or value(7) or value(1)
        if slot == 3:
            return value(2) or value(8) or value(9) or value(3)
        return value(slot - 1) or value(slot)

    def weapon_ammo(self, obs, slot):
        if integer(obs.get("weapon"), -1) == slot:
            return max(0.0, finite(obs.get("weapon_ammo")))
        ammo = obs.get("ammo", {})
        values = [finite(ammo.get(str(slot))), finite(ammo.get(str(slot - 1)))]
        if slot == 3:
            values.extend([finite(ammo.get("8")), finite(ammo.get("9"))])
        return max(values or [0.0])

    # ----- deterministic motor -----------------------------------------

    def legal_action(self, *names):
        for name in names:
            if name in self.legal:
                return name
        return self.noop

    def choose_turn(self, error):
        # right_angle_sign is the observed sign of TURN_RIGHT.  Move in the
        # direction of the signed shortest error, with no repeated chattering.
        if wrap(error) * self.right_angle_sign > 0:
            return self.legal_action("turn_right", "turn_left")
        return self.legal_action("turn_left", "turn_right")

    def steer(self, obs, destination, retreat=False, fire=False):
        error = wrap(bearing(obs["p"], destination) - obs["angle"])
        abs_error = abs(error)
        if abs_error > 22.0 and self.turn_streak < 2:
            self.turn_streak += 1
            ticks = min(3, max(1, int(abs_error / max(3.0, self.turn_rate))))
            self.action_until = obs["tic"] + ticks
            return self.choose_turn(error)
        self.turn_streak = 0
        if abs_error > 9.0:
            # Translation while the view settles prevents the 0004 failure
            # mode: a target cannot freeze the player into a turn-in-place loop.
            if error > 0:
                return self.legal_action("forward_left", "strafe_left", "forward")
            return self.legal_action("forward_right", "strafe_right", "forward")
        if retreat:
            return self.legal_action("backward_left", "backward_right", "backward")
        if fire:
            side = "left" if self.dodge_side < 0 else "right"
            return self.legal_action(
                "fire_strafe_" + side,
                "fire_forward", "fire", "strafe_" + side, "forward",
            )
        return self.legal_action("forward", "forward_left", "strafe_left")

    def evade(self, obs):
        tic = obs["tic"]
        if tic - self.last_side_change > 18:
            self.dodge_side *= -1
            self.last_side_change = tic
        side = "left" if self.dodge_side < 0 else "right"
        if tic - self.last_damage_tic < 20 or self.current_damage() > 18:
            return self.legal_action("backward_" + side, "strafe_" + side, "backward")
        return self.legal_action("strafe_" + side, "forward_" + side, "forward")

    def retreat_target(self, obs):
        p = obs["p"]
        choices = []
        for tic, q, old_cell, new_cell in self.trail:
            d = distance(p, q)
            if d < 80:
                continue
            age = obs["tic"] - tic
            if 0 <= age < 420:
                choices.append((age + d * 0.01, q))
        if choices:
            # A recent breadcrumb is a measured route, not a guessed map exit.
            return min(choices, key=lambda pair: pair[0])[1]
        return None

    def switch_action(self, obs, threat):
        if obs["tic"] - self.last_weapon_switch < 30:
            return None
        current = integer(obs.get("weapon"), 2)
        name = str(threat.get("name", "")) if threat else ""
        if threat and (threat.get("major") or len(self.visible_threats) >= 3):
            order = [6, 4, 3, 2]
        elif threat and (threat.get("hitscan") or finite(threat.get("distance"), 9999) > 300):
            order = [4, 3, 6, 2]
        else:
            order = [3, 4, 2, 6, 1]
        # Do not choose splash weapons in an unobserved close geometry.
        if threat and finite(threat.get("distance"), 9999) < 190:
            order = [3, 4, 6, 2, 1]
        for slot in order:
            if not self.weapon_owned(obs, slot) or self.weapon_ammo(obs, slot) <= 0:
                continue
            if current == slot:
                return None
            action = "select_weapon_" + str(slot)
            if action in self.legal:
                self.last_weapon_switch = obs["tic"]
                return action
        return None

    def combat_action(self, obs, threat):
        if threat is None:
            return None
        d = finite(threat.get("distance"), 9999.0)
        error = wrap(threat.get("error"))
        current = integer(obs.get("weapon"), 2)
        if current in (5, 7) and d < 190:
            return self.evade(obs)
        if obs["weapon_ammo"] <= 0 and current not in (1, 8):
            return self.evade(obs)
        if abs(error) > 13.0:
            # Permit at most two short view corrections.  The next decision
            # must translate/fire-strafe even if the target is still off-axis;
            # this prevents a visible actor from recreating 0004's full-turn
            # oscillation.
            if self.turn_streak < 2:
                self.turn_streak += 1
                self.action_until = obs["tic"] + 2
                return self.choose_turn(error)
            self.turn_streak = 0
            side = "left" if error > 0 else "right"
            return self.legal_action(
                "fire_strafe_" + side, "fire_forward", "strafe_" + side, "forward"
            )
        self.turn_streak = 0
        if d < 90 or obs["health"] < 28 or obs["tic"] - self.last_damage_tic < 18:
            fire_side = "left" if self.dodge_side < 0 else "right"
            return self.legal_action(
                "fire_strafe_" + fire_side,
                "fire_forward", "fire", "strafe_" + fire_side,
            )
        return self.legal_action(
            "fire_strafe_left" if self.dodge_side < 0 else "fire_strafe_right",
            "fire_forward", "fire", "strafe_left", "strafe_right",
        )

    def interact_action(self, obs, obj):
        if obj is None:
            return None
        tic = obs["tic"]
        d = finite(obj.get("distance"), 9999.0)
        error = wrap(obj.get("error"))
        if d > 82.0:
            return None
        if abs(error) > 28.0:
            return self.choose_turn(error)
        key = (str(obj.get("id")), point_cell(obs["p"]))
        if self.use_faces.get(key, -1) > tic or tic - self.last_use_tic < 12:
            return self.legal_action("forward", "forward_left", "strafe_left")
        self.use_faces[key] = tic + 70
        self.last_use_tic = tic
        self.after_use = {"tic": tic, "p": list(obs["p"]), "id": str(obj.get("id"))}
        self.goal = None
        self.ctx.log("use_contact", tic=tic, object_id=str(obj.get("id")),
                     category=str(obj.get("category", "")))
        return self.legal_action("use", "forward")

    def blocked_action(self, obs):
        event = self.blocked_event
        if event is None:
            return None
        tic = obs["tic"]
        face = event.get("face")
        # A single pulse is a cheap door/switch test.  Cooldown is per measured
        # face, so an unopened wall cannot consume the entire episode.
        if (face is not None and self.use_faces.get(("face", str(face)), -1) <= tic
                and tic - self.last_use_tic >= 18
                and tic - self.last_damage_tic > 10):
            self.use_faces[("face", str(face))] = tic + 105
            self.last_use_tic = tic
            self.after_use = {"tic": tic, "p": list(obs["p"]), "id": None}
            self.goal = None
            self.ctx.log("use_blocked_face", tic=tic, face=list(face) if isinstance(face, tuple) else str(face))
            return self.legal_action("use", "backward", "turn_left")
        # The failed face is already recorded; a short moving turn and a new
        # radial probe are preferable to another straight collision.
        self.goal = None
        self.turn_streak = 0
        self.sweep_index += 1
        return self.legal_action("backward_left", "backward_right", "strafe_left", "turn_left")

    def should_fight(self, obs, threat):
        if threat is None:
            return False
        d = finite(threat.get("distance"), 9999.0)
        immediate = (
            d < 145.0 or threat.get("hitscan") and d < 650.0
            or threat.get("major") and d < 800.0
            or threat.get("body_blocker") and d < 230.0
            or obs["tic"] - self.last_damage_tic < 20
        )
        if immediate:
            return True
        if self.intent == "fight" and self.intent_expires >= obs["now"]:
            return True
        if self.plan.get("combat") == "clear_all" and d < 450.0:
            return True
        return False

    def local_action(self, obs):
        tic = obs["tic"]
        if obs["terminal"] or obs["health"] <= 0:
            return self.noop

        # A turn is a bounded view correction, not a fresh permission to
        # reconsider the same correction every engine tic.  Emergency damage
        # and projectiles can interrupt it; otherwise let the already-posted
        # correction finish before selecting a translating action.
        if (self.current_action in TURN_ACTIONS and obs["tic"] < self.action_until
                and obs["health"] >= 20
                and obs["tic"] - self.last_damage_tic >= 8
                and not (obs["projectiles"] and obs["projectiles"][0].get("distance", 9999) < 230)):
            return self.current_action
        if (self.current_action in MOVE_ACTIONS and obs["tic"] < self.translate_until
                and self.blocked_event is None
                and obs["health"] >= 20
                and obs["tic"] - self.last_damage_tic >= 8
                and not (obs["projectiles"] and obs["projectiles"][0].get("distance", 9999) < 230)):
            return self.current_action

        threats = obs["threats"]
        threat = threats[0] if threats else None
        nearby_interact = [x for x in obs["objects"] if is_interactable(x)]
        nearby_interact.sort(key=lambda x: finite(x.get("distance"), 9999.0))
        interact = nearby_interact[0] if nearby_interact else None

        # Emergency evasion is code-owned and cannot be vetoed by a stale
        # semantic answer.
        if obs["projectiles"] and obs["projectiles"][0].get("distance", 9999) < 230:
            return self.evade(obs)
        # A distant visible actor must not suppress a measured collision
        # recovery.  Close/body-blocking danger still wins and is handled by
        # combat immediately below.
        if self.blocked_event is not None and threat is not None:
            threat_distance = finite(threat.get("distance"), 9999.0)
            if (threat_distance > 170.0 and not threat.get("major")
                    and tic - self.last_damage_tic >= 20):
                alternate = self.blocked_action(obs)
                if alternate:
                    return alternate
        if threat is not None and self.should_fight(obs, threat):
            if obs["health"] < 20 and finite(threat.get("distance"), 9999) > 95:
                return self.evade(obs)
            switch = self.switch_action(obs, threat)
            if switch:
                return switch
            self.last_target_id = threat.get("id")
            self.last_shot_tic = tic
            return self.combat_action(obs, threat)

        # A visible exit/switch is more valuable than optional combat.  The
        # pulse is restricted to measured contact and facing, then confirmed by
        # the terminal observation or movement change.
        if interact is not None:
            value = self.landmark_value(interact, obs)
            if value > 150.0 or self.intent == "interact":
                action = self.interact_action(obs, interact)
                if action:
                    return action

        if self.intent == "retreat" and self.intent_expires >= obs["now"]:
            target = self.retreat_target(obs)
            if target is not None and obs["health"] < 55:
                return self.steer(obs, target, retreat=True)
            return self.evade(obs)
        if obs["health"] < 24 and self.current_damage() > 10:
            target = self.retreat_target(obs)
            if target is not None:
                return self.steer(obs, target, retreat=True)
            return self.evade(obs)
        if tic - self.last_damage_tic < 22:
            return self.evade(obs)

        blocked = self.blocked_action(obs)
        if blocked:
            return blocked

        goal = self.choose_goal(obs)
        if goal is None:
            return self.legal_action("forward_right", "forward", "strafe_right")
        if goal.get("kind") == "landmark":
            obj = next((x for x in obs["objects"] if x.get("id") == goal.get("id")), None)
            if obj is not None:
                goal["p"] = list(obj["p"])
                goal["last_seen"] = tic
                d = distance(obs["p"], obj["p"])
                if d < 86 and is_interactable(obj):
                    action = self.interact_action(obs, obj)
                    if action:
                        return action
                if d < 32:
                    self.goal = None
                    return self.legal_action("forward", "strafe_left", "forward_left")
                if d < goal.get("best", d) - 5:
                    goal["best"] = d
                    goal["last_improve"] = tic
                return self.steer(obs, obj["p"])
            self.goal = None
        else:
            d = distance(obs["p"], goal.get("p", obs["p"]))
            if d < goal.get("best", d) - 5:
                goal["best"] = d
                goal["last_improve"] = tic
            return self.steer(obs, goal.get("p", obs["p"]))

        return self.legal_action("forward_right", "forward", "strafe_right")

    # ----- typed Jev interface -----------------------------------------

    def jev_state(self, obs):
        def describe(obj):
            return {
                "id": str(obj.get("id", "")),
                "name": str(obj.get("name", "")),
                "category": str(obj.get("category", "")),
                "range": str(obj.get("range", range_bucket(obj.get("distance")))),
                "side": str(obj.get("side", side_bucket(obj.get("error")))),
                "level": "same" if abs(finite(obj.get("dz"))) <= 52 else "different",
                "active": bool(obj.get("active")),
                "hitscan": bool(obj.get("hitscan")),
                "body_blocker": bool(obj.get("body_blocker")),
            }
        goal = self.goal or {}
        target_options = [describe(x) for x in obs["threats"][:4]]
        landmark_options = [describe(x) for x in self.useful_landmarks(obs)[:5]]
        return {
            "task": "Reach and activate the visible map exit alive; optional kills are secondary.",
            "rules": "Only currently rendered labels are supplied. Code computes distances, aiming, legality, collisions and route memory. A label does not prove a straight walkable path.",
            "plan": dict(self.plan),
            "health": "critical" if obs["health"] < 25 else "injured" if obs["health"] < 60 else "healthy",
            "armor": "protected" if obs["armor"] >= 20 else "unprotected",
            "ammo": "empty" if obs["weapon_ammo"] <= 0 else "low" if obs["weapon_ammo"] < 10 else "usable",
            "weapon_slot": integer(obs.get("weapon"), 2),
            "recent_damage": self.current_damage() > 0 or obs["tic"] - self.last_damage_tic < 35,
            "visible_threats": target_options,
            "visible_landmarks": landmark_options,
            "motion": {
                "blocked": self.blocked_event is not None,
                "known_cells": len(self.cells),
                "known_edges": sum(len(v) for v in self.edges.values()),
                "stalled": obs["tic"] - self.last_progress > 50,
                "goal": str(goal.get("kind", "none")),
            },
            "current_intent": self.intent if self.intent_expires >= obs["now"] else "none",
            "candidate_rule": "Pick a semantic intent only; the code will keep moving, pulse USE once, aim, and override danger.",
        }

    def make_jev_questions(self, state, obs):
        intents = {
            "fight": "Engage the supplied live threat now, using moving fire; choose only when it is an immediate blocker or dangerous active threat.",
            "retreat": "Create distance along measured travel or by a moving dodge because health/damage/close pressure makes commitment unsafe.",
            "interact": "Approach and pulse USE for a supplied nearby exit, switch or door; do not infer an unseen door.",
            "advance": "Continue the current exit/key/weapon objective or a measured exploration macro.",
            "scan": "No supplied threat or landmark is urgent; change the view with a short moving sweep.",
            "hold": "Evidence is insufficient for a new commitment; let the deterministic short macro continue briefly.",
        }
        # Ask the same high-consequence semantic question twice with different
        # order.  This neutralizes first-option attraction without adding a
        # second network round trip.
        ordered = list(intents)
        if self.jev_request % 2:
            ordered.reverse()
        first = {key: intents[key] for key in ordered}
        second_order = list(reversed(ordered))
        second = {key: intents[key] for key in second_order}
        target_criteria = {"none": "No listed visible monster is the best immediate target."}
        target_keys = {}
        for index, obj in enumerate(obs["threats"][:4]):
            key = "t" + str(index)
            target_keys[key] = str(obj.get("id"))
            target_criteria[key] = {
                "name": str(obj.get("name", "")),
                "range": str(obj.get("range", "")),
                "side": str(obj.get("side", "")),
                "active": bool(obj.get("active")),
                "hitscan": bool(obj.get("hitscan")),
            }
        side_criteria = {
            "left": "Use a left-side moving dodge or sweep when a side must be chosen.",
            "right": "Use a right-side moving dodge or sweep when a side must be chosen.",
            "forward": "Keep the current forward route when neither side is clearly safer.",
        }
        questions = {
            "intent_a": Choice(
                "Using the named fields in state, choose the local semantic intent. Do not calculate numbers, invent geometry, or output a button.",
                first,
            ),
            "intent_b": Choice(
                "Independently choose the same local intent from the same state. An immediate active threat or critical damage outranks an optional pickup; uncertainty is hold.",
                second,
            ),
            "target": Choice(
                "Which supplied visible threat, if any, deserves current aim attention? Use active evidence, hitscan and body-blocking pressure; do not choose a hidden or remembered actor.",
                target_criteria,
            ),
            "side": Choice(
                "Which movement side is safer or more useful for the current local intent? Choose forward when side evidence is absent.",
                side_criteria,
            ),
            "use_now": Noul(
                "Is a USE pulse warranted at the current measured contact or blocked face? Yes only for the supplied nearby interactable or a first bounded obstruction test; no for an unseen door or a distant label.",
                {"true": "The current state supplies a nearby interactable or a newly blocked face where one pulse is justified.",
                 "false": "There is no measured contact or a USE pulse was recently attempted."},
            ),
        }
        return questions, target_keys

    def dispatch_jev(self, obs):
        if self.jev_future is not None:
            return
        if obs["now"] < self.next_jev:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if finite(remaining.get("system1_calls"), 0) <= 4 or finite(remaining.get("wall_s"), 0) <= 2:
            return
        self.jev_request += 1
        state = self.jev_state(obs)
        questions, target_keys = self.make_jev_questions(state, obs)
        meta = {
            "request": self.jev_request,
            "tic": obs["tic"],
            "now": obs["now"],
            "plan_version": self.plan_version,
            "target_keys": target_keys,
        }
        self.jev_meta = meta
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=2.5)
            )
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
            return
        self.next_jev = obs["now"] + (0.48 if obs["threats"] or obs["projectiles"] else 1.05)

    def poll_jev(self, obs):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        meta = self.jev_meta or {}
        try:
            answers = future.result()
        except (BudgetExceeded, EpisodeOver):
            return
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError, CancelledError) as exc:
            self.last_jev_failure = obs["now"]
            self.ctx.log("jev_failure", error=type(exc).__name__)
            return
        except Exception as exc:
            # A stand-in or provider may return an unexpected exception; a
            # failed semantic update is never allowed to break the game loop.
            self.last_jev_failure = obs["now"]
            self.ctx.log("jev_failure", error=type(exc).__name__)
            return

        age = obs["now"] - finite(meta.get("now"), obs["now"])
        if age > 2.2 or integer(meta.get("plan_version"), -1) != self.plan_version:
            self.ctx.log("jev_discard", request=meta.get("request"), age_s=round(age, 3))
            return
        legal_intents = {"fight", "retreat", "interact", "advance", "scan", "hold"}
        first = choice_answer(answers, "intent_a", legal_intents, 0.52)
        second = choice_answer(answers, "intent_b", legal_intents, 0.52)
        if first is not None and second == first:
            self.intent = first
            self.intent_expires = obs["now"] + 1.65
        elif first is not None and second is None and first in {"advance", "scan", "hold"}:
            self.intent = first
            self.intent_expires = obs["now"] + 1.0
        target_key = choice_answer(answers, "target", set(["none"] + list(meta.get("target_keys", {}).keys())), 0.52)
        if target_key == "none":
            self.intent_target = None
        elif target_key is not None:
            ident = meta.get("target_keys", {}).get(target_key)
            if ident and any(obj.get("id") == ident for obj in obs["threats"]):
                self.intent_target = ident
        side = choice_answer(answers, "side", {"left", "right", "forward"}, 0.52)
        if side in ("left", "right"):
            self.intent_side = side
            self.dodge_side = -1 if side == "left" else 1
        if noul_yes(answers, "use_now") and self.blocked_event is not None:
            # This is advisory only; blocked_action still enforces the pulse
            # cooldown and legal face evidence.
            self.last_use_tic = min(self.last_use_tic, obs["tic"] - 18)
        self.ctx.log("jev_intent", request=meta.get("request"), intent=self.intent,
                     target=self.intent_target, side=self.intent_side, age_s=round(age, 3))

    # ----- bounded Astra policy ----------------------------------------

    def astra_payload(self, obs):
        visible = []
        for obj in (obs["threats"][:6] + self.useful_landmarks(obs)[:8]):
            visible.append({
                "id": str(obj.get("id", "")),
                "name": str(obj.get("name", "")),
                "category": str(obj.get("category", "")),
                "range": str(obj.get("range", "")),
                "side": str(obj.get("side", "")),
                "level": "same" if abs(finite(obj.get("dz"))) <= 52 else "different",
                "active": bool(obj.get("active")),
            })
        return {
            "episode": {
                "map": str(obs["raw"].get("map", "")),
                "iwad": str(obs["raw"].get("iwad", "")),
                "skill": integer(obs["raw"].get("skill")),
                "tic": obs["tic"],
            },
            "survival": {"health": round(obs["health"], 1), "armor": round(obs["armor"], 1),
                         "ammo": "empty" if obs["weapon_ammo"] <= 0 else "low" if obs["weapon_ammo"] < 10 else "usable",
                         "damage_recent": self.current_damage() > 0},
            "visible": visible,
            "measured_route": {
                "cells": len(self.cells),
                "edges": sum(len(v) for v in self.edges.values()),
                "failed_faces": sum(1 for v in self.attempts.values() if v > 0),
                "stalled": obs["tic"] - self.last_progress > 70,
                "goal": self.goal.get("kind") if self.goal else "none",
            },
            "recent": [
                {"tic": int(t), "position": [round(finite(q[0])), round(finite(q[1])), round(finite(q[2]))]}
                for t, q, _, _ in list(self.trail)[-8:]
            ],
            "current_plan": dict(self.plan),
        }

    def dispatch_astra(self, obs, reason):
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        if obs["now"] < self.next_astra:
            return
        if not hasattr(self.ctx, "think"):
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if finite(remaining.get("system2_calls"), 0) <= 0 or finite(remaining.get("wall_s"), 0) <= 120:
            return
        payload = self.astra_payload(obs)
        prompt = (
            "You are the slow strategic layer for a real-time classic Doom controller. "
            "The controller continues moving while you answer. Do not output buttons, coordinates, "
            "a memorized seed route, code, or an invented wall/key/exit. Use only the supplied public "
            "observations and measured traversed cells. Exit success dominates optional kills. "
            "Revise a bounded durable phase for a fast code executor. Return ONLY one JSON object: "
            "phase=sweep|approach|clear|interact|retreat, priority=exit|key|weapon|recover|explore, "
            "combat=clear_near|bypass|clear_all, side=left|right, interaction=use_on_contact|use_on_block, "
            "focus_id=null or exactly one supplied visible object id, reason=short evidence. "
            "The code itself owns distance arithmetic, weapon legality, aiming, collision recovery, "
            "USE cooldowns and emergency evasion. Choose interact only when a supplied interactable is "
            "near or repeated measured progress suggests it; choose retreat only from observed damage or "
            "critical health. No prose outside JSON.\nPUBLIC_STATE="
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        )
        self.astra_calls += 1
        self.last_astra = obs["now"]
        self.next_astra = obs["now"] + 150.0
        try:
            session = self.astra_session
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, session_id=session, timeout_s=240.0)
            )
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None
            return
        self.ctx.log("astra_requested", reason=reason, call=self.astra_calls,
                     cells=len(self.cells), goal=self.goal.get("kind") if self.goal else "none")

    def poll_astra(self, obs):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
        except (BudgetExceeded, EpisodeOver, CancelledError):
            return
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
            self.ctx.log("astra_failure", error=type(exc).__name__)
            return
        except Exception as exc:
            self.ctx.log("astra_failure", error=type(exc).__name__)
            return
        proposed = parse_json_object(getattr(result, "text", ""))
        updated = dict(self.plan)
        valid = False
        for key, options in {
            "phase": {"sweep", "approach", "clear", "interact", "retreat"},
            "priority": {"exit", "key", "weapon", "recover", "explore"},
            "combat": {"clear_near", "bypass", "clear_all"},
            "side": {"left", "right"},
            "interaction": {"use_on_contact", "use_on_block"},
        }.items():
            value = proposed.get(key)
            if isinstance(value, str) and value in options:
                updated[key] = value
                valid = True
        focus = proposed.get("focus_id")
        if focus is None:
            updated["focus_id"] = None
        elif isinstance(focus, (str, int)):
            focus = str(focus)
            if any(obj.get("id") == focus for obj in self.visible_objects):
                updated["focus_id"] = focus
                valid = True
        if valid:
            self.plan = updated
            self.plan_version += 1
            self.intent = "advance"
            self.intent_expires = -1
            self.intent_target = None
            self.goal = None
            self.astra_session = getattr(result, "session_id", self.astra_session)
            self.ctx.log("astra_plan", version=self.plan_version, plan=dict(self.plan),
                         reason=str(proposed.get("reason", ""))[:240])
        else:
            self.ctx.log("astra_invalid", fallback="keep_plan")

    # ----- episode loop -------------------------------------------------

    def run_episode(self, game):
        ctx = self.ctx
        spec = getattr(game, "action_spec", None)
        names = getattr(spec, "names", ()) if spec is not None else ()
        self.legal = set(names or ())
        if not self.legal:
            return
        candidate_noop = getattr(spec, "noop", None)
        self.noop = candidate_noop if candidate_noop in self.legal else next(iter(self.legal))
        self.started = finite(ctx.clock.now())
        self.current_action = self.noop
        self.previous_action = self.noop
        try:
            ctx.act(self.noop)
        except Exception:
            # The action spec was checked; if a test double rejects the first
            # pulse, let the next safe legal action be attempted normally.
            pass

        while not ctx.done and not bool(getattr(game, "done", False)):
            try:
                raw = ctx.observe()
            except (EpisodeOver, BudgetExceeded):
                return
            now = max(0.0, finite(ctx.clock.now()) - self.started)
            obs = self.normalize(raw, now)
            if obs["terminal"] or bool(getattr(game, "done", False)):
                break
            if obs["tic"] <= self.last_tic:
                try:
                    ctx.wait_for_tick()
                except (EpisodeOver, BudgetExceeded):
                    return
                continue
            self.last_tic = obs["tic"]
            self.update(obs)
            self.poll_jev(obs)
            self.poll_astra(obs)
            if ctx.done or bool(getattr(game, "done", False)):
                break

            # Astra is deliberately event-gated: one initial durable review,
            # then one only after a long measured stall or damage crisis.
            stalled = obs["tic"] - self.last_progress > 175
            crisis = obs["health"] < 28 and self.current_damage() > 18
            if self.astra_calls == 0:
                self.dispatch_astra(obs, "initial_phase")
            elif stalled or crisis:
                self.dispatch_astra(obs, "stall_or_crisis")

            self.dispatch_jev(obs)
            try:
                action = self.local_action(obs)
            except Exception as exc:
                # Deterministic code is defensive too: a malformed observation
                # must degrade to a legal moving/noop action, never an error.
                self.ctx.log("controller_fallback", error=type(exc).__name__)
                action = self.legal_action("forward_right", "forward", "strafe_right", self.noop)
            if action not in self.legal:
                action = self.legal_action("forward_right", "forward", "strafe_right", self.noop)
            same_action = action == self.current_action
            if action in TURN_ACTIONS:
                # Do not extend a turn merely because the bounded hold was
                # returned by the next observation.  Only a new turn request,
                # or an expired one, receives a fresh two-tic horizon.
                if not same_action or self.action_until <= obs["tic"]:
                    self.action_until = max(self.action_until, obs["tic"] + 2)
                self.translate_until = obs["tic"]
            elif action in MOVE_ACTIONS:
                self.action_until = obs["tic"]
                # Once a turn correction has yielded translation, preserve the
                # moving macro for several tics.  A blocked event, damage or a
                # projectile can interrupt it at the top of local_action.
                if not same_action or self.translate_until <= obs["tic"]:
                    self.translate_until = obs["tic"] + 7
            else:
                self.action_until = obs["tic"]
                self.translate_until = obs["tic"]
            self.current_action = action
            self.previous_action = action
            self.last_action_tic = obs["tic"]
            try:
                ctx.act(action)
            except (EpisodeOver, BudgetExceeded):
                return
            except Exception as exc:
                # A rejected action is recovered with the advertised noop; the
                # public contract treats game-side action errors as terminal,
                # but a test double may throw for an optional action.
                self.ctx.log("action_failure", action=action, error=type(exc).__name__)
                try:
                    ctx.act(self.noop)
                except Exception:
                    return
            try:
                ctx.wait_for_tick()
            except (EpisodeOver, BudgetExceeded):
                return

        try:
            ctx.memory["moving_macro_summary"] = {
                "cells": len(self.cells),
                "edges": sum(len(v) for v in self.edges.values()),
                "goal": self.goal.get("kind") if self.goal else "none",
                "astra_calls": self.astra_calls,
                "jev_requests": self.jev_request,
                "plan_version": self.plan_version,
            }
        except Exception:
            pass


def build(ctx):
    return MovingMacroController(ctx)
