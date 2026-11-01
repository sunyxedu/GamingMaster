"""Landmark-survey controller for asynchronous classic Doom.

The controller treats the rendered labels as intermittent landmarks rather than
as a complete map.  It periodically takes a bounded panorama, remembers the
world positions and height layers of useful landmarks, and then commits to a
measured approach or to an information-seeking departure.  A small portal
experiment records the result of USE + same-heading crossing instead of
repeating the same wall probe.  Models choose among these code-generated
objectives; they never hold the real-time motor clock.
"""
from __future__ import annotations

import collections
import json
import math
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


MOVE = {
    "forward", "backward", "strafe_left", "strafe_right",
    "forward_left", "forward_right", "backward_left", "backward_right",
    "fire_forward", "fire_strafe_left", "fire_strafe_right",
}
TURN = {"turn_left", "turn_right"}
HITSCAN = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS",
           "SpiderMastermind"}
MAJOR = {"Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind",
         "Revenant"}
MELEE = {"Demon", "Spectre", "LostSoul", "Pinkie", "Pinky"}
MISSILES = {
    "DoomImpBall", "CacodemonBall", "BaronBall", "RevenantTracer",
    "Rocket", "ArachnotronPlasma", "PlasmaBall", "FatShot", "BFGShot",
    "Tracer", "PlasmaFire",
}
WEAPON_SLOTS = {
    "shotgun": 3, "supershotgun": 3, "chaingun": 4,
    "rocketlauncher": 5, "plasmarifle": 6, "bfg9000": 7,
    "chainsaw": 1,
}
VALID_PROGRAM = {"seek_keys", "seek_exit", "survey", "probe", "recover"}


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


def obj(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def wrap(angle):
    return (number(angle) + 180.0) % 360.0 - 180.0


def distance(a, b):
    return math.hypot(number(a[0]) - number(b[0]),
                     number(a[1]) - number(b[1]))


def bearing(a, b):
    return math.degrees(math.atan2(number(b[1]) - number(a[1]),
                                   number(b[0]) - number(a[0]))) % 360.0


def cell(p, size=64.0):
    return (math.floor(number(p[0]) / size),
            math.floor(number(p[1]) / size),
            math.floor(number(p[2]) / 48.0))


def heading_bin(angle):
    return int((number(angle) % 360.0 + 22.5) // 45.0) % 8


def point(p, angle, length):
    r = math.radians(angle)
    return (number(p[0]) + math.cos(r) * length,
            number(p[1]) + math.sin(r) * length,
            number(p[2]))


def direction(error):
    return "ahead" if abs(error) < 12 else "left" if error > 0 else "right"


def parse_json_object(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for index, char in enumerate(text[:16000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            continue
    return {}


def parse_observation(raw, fallback_tic):
    """Normalize the public projection and tolerate arbitrary pre-flight data."""
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
    p = [number(player.get(k)) for k in ("x", "y", "z")]
    velocity = raw.get("velocity")
    if not isinstance(velocity, dict):
        velocity = {}
    v = [number(velocity.get(k)) for k in ("x", "y", "z")]
    tic_value = raw.get("tic")
    tic = integer(tic_value, fallback_tic) if tic_value is not None else fallback_tic
    if tic < fallback_tic - 100000:
        tic = fallback_tic
    angle = number(player.get("angle")) % 360.0
    health = number(raw.get("health"), 100.0)
    armor = number(raw.get("armor"), 0.0)
    labels = raw.get("visible")
    if not isinstance(labels, list):
        labels = []
    objects = []
    for index, label in enumerate(labels[:300]):
        if not isinstance(label, dict):
            continue
        world = label.get("world")
        if not isinstance(world, (list, tuple)) or len(world) < 3:
            world = label.get("world_position")
        if not isinstance(world, (list, tuple)) or len(world) < 3:
            continue
        q = [number(x) for x in world[:3]]
        name = str(label.get("name", label.get("object_name", "unknown")))[:80]
        category = str(label.get("category", label.get("object_category", "unknown")))[:48]
        low_name = name.lower()
        low_cat = category.lower()
        if low_cat == "self" or name == "DoomPlayer":
            continue
        vv = label.get("velocity")
        if not isinstance(vv, (list, tuple)) or len(vv) < 3:
            vv = [0.0, 0.0, 0.0]
        box = label.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            box = [160.0, 120.0, 0.0, 0.0]
        ident = str(label.get("id", label.get("object_id", index)))[:80]
        d = distance(p, q)
        err = wrap(bearing(p, q) - angle)
        monster = low_cat == "monster" and not low_name.startswith("dead")
        projectile = (low_cat in {"projectile", "missile"} or
                      name in MISSILES or
                      (low_cat == "explosive" and
                       any(x in low_name for x in ("ball", "rocket", "tracer", "plasma", "fatshot"))))
        # BulletPuff is an impact marker, not an incoming missile.
        if name == "BulletPuff":
            projectile = False
        key = (low_cat == "key" or "key" in low_name or
               low_name.endswith("card"))
        interact = (low_cat in {"door", "exit", "switch", "teleporter", "teleport"} or
                    any(x in low_name for x in ("door", "exit", "switch", "teleport")))
        pickup = (low_cat in {"weapon", "health", "armor", "ammo", "key",
                              "powerup", "power-up"} or key or interact)
        objects.append({
            "id": ident,
            "name": name,
            "category": category,
            "cat": low_cat,
            "p": q,
            "v": [number(x) for x in vv[:3]],
            "box": [number(x) for x in box[:4]],
            "d": d,
            "dz": q[2] - p[2],
            "err": err,
            "monster": monster,
            "projectile": projectile,
            "key": key,
            "interact": interact,
            "pickup": pickup,
            "hitscan": monster and name in HITSCAN,
            "major": monster and name in MAJOR,
            "melee": monster and name in MELEE,
        })
    threats = [x for x in objects if x["monster"]]
    for x in threats:
        score = (245 if x["major"] else 0) + (210 if x["hitscan"] else 0)
        score += 130 if x["melee"] else 0
        score += 260 if x["d"] < 110 else 100 if x["d"] < 230 else 0
        score -= .30 * x["d"] + .08 * abs(x["err"])
        x["threat_score"] = score
    threats.sort(key=lambda x: x["threat_score"], reverse=True)
    weapons = raw.get("weapons")
    ammo = raw.get("ammo")
    return {
        "raw": raw,
        "tic": tic,
        "p": p,
        "v": v,
        "angle": angle,
        "health": health,
        "armor": armor,
        "weapon": integer(raw.get("weapon"), 2),
        "weapon_ammo": number(raw.get("weapon_ammo"), 0),
        "weapons": weapons if isinstance(weapons, dict) else {},
        "ammo": ammo if isinstance(ammo, dict) else {},
        "kills": integer(raw.get("kills")),
        "items": integer(raw.get("items")),
        "secrets": integer(raw.get("secrets")),
        "damage": number(raw.get("damage_taken")),
        "dead": bool(raw.get("dead")),
        "terminal": bool(raw.get("terminal")),
        "objects": objects,
        "threats": threats,
    }


def accepted_choice(answers, key, allowed, threshold):
    if not isinstance(answers, dict):
        return None
    answer = answers.get(key)
    value = getattr(answer, "value", None)
    if not isinstance(value, str) or value not in allowed:
        return None
    probabilities = getattr(answer, "probabilities", None)
    if isinstance(probabilities, dict) and number(probabilities.get(value)) < threshold:
        return None
    confidence = getattr(answer, "confidence", None)
    if confidence is not None and number(confidence) < 0.10:
        return None
    return value


def accepted_yes(answers, key, threshold):
    if not isinstance(answers, dict):
        return False
    answer = answers.get(key)
    return number(getattr(answer, "value", 0.0)) >= threshold


class LandmarkSurvey:
    """A code-owned motor with a persistent, height-aware landmark memory."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.noop = "noop"
        self.last_tic = -1
        self.previous = None
        self.previous_action = "noop"
        self.previous_position = None
        self.turn_sign = 1.0  # positive angle change caused by turn_left
        self.turn_rate = 5.5
        self.turn_lease = None
        self.stationary = 0
        self.blocked = 0
        self.last_damage_tic = -10000
        self.last_health = 100.0

        self.cells = collections.Counter()
        self.heading_visits = collections.Counter()
        self.view_visits = collections.Counter()
        self.view_evidence = collections.defaultdict(float)
        self.last_new_cell = -1
        self.last_transition = -10000
        self.recent_events = collections.deque(maxlen=32)
        self.landmarks = {}
        self.keys_seen = set()
        self.keys_collected = set()
        self.last_counts = (0, 0, 0)

        self.mode = "travel"
        self.travel_heading = None
        self.travel_until = -1
        self.travel_started = -1
        self.travel_origin = None
        self.goal = None
        self.goal_best = 1e30
        self.goal_last_progress = -1
        self.goal_failures = collections.Counter()
        self.goal_cooldown = {}

        self.scan = None
        self.last_scan = -10000
        self.scan_count = 0
        self.scan_side = 1
        self.scan_bearings = []
        self.force_scan = False

        self.portal = None
        self.portal_faces = {}
        self.last_use = -10000
        self.force_use = False
        self.escape = collections.deque()
        self.last_escape = -10000
        self.dodge_side = 1

        self.combat_id = None
        self.combat_until = -1
        self.last_weapon = -10000
        self.last_fire = -10000

        self.jev_future = None
        self.jev_request = None
        self.next_jev = 0
        self.jev_calls = 0
        self.jev_hint = None
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.next_astra = 80
        self.program = collections.deque()
        self.program_until = -1
        self.policy = {"mission": "key_first", "scan": "normal",
                       "combat": "blockers"}

    # ---------- defensive and geometric helpers ----------
    def legal(self, *actions):
        for action in actions:
            if action in self.allowed:
                return action
        return self.noop

    def turn_action(self, desired_sign):
        positive_by_left = self.turn_sign > 0
        return self.legal("turn_left" if (desired_sign > 0) == positive_by_left
                          else "turn_right", "forward")

    def turn_to(self, error, tic, tolerance=8.0, lease=6):
        if abs(error) <= tolerance:
            self.turn_lease = None
            return None
        wanted = 1 if error > 0 else -1
        if self.turn_lease and tic < self.turn_lease[1]:
            return self.legal(self.turn_lease[0], "forward")
        action = self.turn_action(wanted)
        if action in TURN:
            self.turn_lease = (action, tic + lease)
        return action

    def object_descriptions(self, values, limit=5):
        result = []
        for x in values[:limit]:
            result.append({
                "id": x["id"], "name": x["name"], "category": x["category"],
                "range": "contact" if x["d"] < 75 else "near" if x["d"] < 220 else
                "medium" if x["d"] < 550 else "far",
                "direction": direction(x["err"]),
                "height": "same_floor" if abs(x["dz"]) <= 40 else
                "above" if x["dz"] > 0 else "below",
                "hitscan": bool(x["hitscan"]), "major": bool(x["major"]),
                "body_blocker": bool(x["melee"] and x["d"] < 145),
            })
        return result

    def update_kinematics(self, obs):
        t = obs["tic"]
        if self.previous is None or self.previous_position is None:
            self.stationary = 0
            self.blocked = 0
            self.last_health = obs["health"]
            return
        dt = max(1, t - self.previous["tic"])
        moved_xy = distance(obs["p"], self.previous_position)
        moved = math.sqrt(moved_xy * moved_xy +
                          (obs["p"][2] - self.previous_position[2]) ** 2)
        delta_angle = wrap(obs["angle"] - self.previous["angle"])
        if self.previous_action in TURN and dt <= 10 and abs(delta_angle) > .15:
            rate = abs(delta_angle) / dt
            if .2 < rate < 25:
                self.turn_rate = .82 * self.turn_rate + .18 * rate
                if self.previous_action == "turn_left":
                    self.turn_sign = 1.0 if delta_angle > 0 else -1.0
                else:
                    self.turn_sign = -1.0 if delta_angle > 0 else 1.0
        if self.previous_action in MOVE and dt <= 16:
            if moved < max(1.4, .48 * dt):
                self.blocked += dt
            else:
                self.blocked = 0
        else:
            self.blocked = 0
        if moved < .75:
            self.stationary += dt
        else:
            self.stationary = 0
        if obs["health"] < self.last_health - .1 or obs["damage"] > self.previous["damage"] + .1:
            self.last_damage_tic = t
            self.recent_events.append({"tic": t, "kind": "damage",
                                       "amount": round(max(0, self.last_health - obs["health"]))})
        self.last_health = obs["health"]
        # A large z change or an unexpectedly large displacement is evidence of
        # a lift/teleporter/sector transition, not merely another new cell.
        if abs(obs["p"][2] - self.previous_position[2]) > 20 or moved_xy > 240:
            self.last_transition = t
            self.recent_events.append({"tic": t, "kind": "layer_transition",
                                       "from_z": round(self.previous_position[2]),
                                       "to_z": round(obs["p"][2])})
            self.mode = "travel"
            self.scan = None

    def landmark_value(self, record, obs, mission=None):
        cat = record["category"].lower()
        name = record["name"].lower()
        if record.get("acquired"):
            return -1
        if "exit" in name or cat in {"exit", "switch", "door", "teleport", "teleporter"}:
            return 12000
        if record.get("key"):
            return 9000 if mission != "seek_exit" else 4500
        if cat == "weapon":
            slot = WEAPON_SLOTS.get(name.replace(" ", ""))
            owned = bool(obs["weapons"].get(str(slot), False)) if slot else False
            return 3800 if not owned else 300
        if cat == "health":
            return 3600 if obs["health"] < 48 else 700 if obs["health"] < 78 else 0
        if cat == "armor":
            return 3000 if obs["armor"] < 30 else 450 if obs["armor"] < 75 else 0
        if cat == "ammo":
            return 1200 if obs["weapon_ammo"] < 8 else 200
        return 0

    def remember_landmarks(self, obs):
        t = obs["tic"]
        now_ids = set()
        for x in obs["objects"]:
            if not x["pickup"] or x["monster"] or x["projectile"]:
                continue
            ident = x["id"]
            now_ids.add(ident)
            old = self.landmarks.get(ident)
            if old is None or distance(old["p"], x["p"]) > 220:
                old = {"id": ident, "name": x["name"], "category": x["category"],
                       "p": list(x["p"]), "key": x["key"], "interact": x["interact"],
                       "seen": t, "last_seen": t, "seen_count": 0,
                       "last_d": x["d"], "best_d": x["d"], "acquired": False,
                       "attempts": 0}
                self.landmarks[ident] = old
            old["name"] = x["name"]
            old["category"] = x["category"]
            old["p"] = list(x["p"])
            old["key"] = x["key"]
            old["interact"] = x["interact"]
            old["last_seen"] = t
            old["seen_count"] += 1
            old["last_d"] = x["d"]
            old["best_d"] = min(old.get("best_d", x["d"]), x["d"])
            if x["key"]:
                self.keys_seen.add(x["name"])
        # Disappearance immediately after a close overlap is a useful local
        # pickup/key event.  It is not treated as an inventory oracle otherwise.
        for ident, old in list(self.landmarks.items()):
            if ident in now_ids or old.get("acquired"):
                continue
            if (t - old.get("last_seen", t) <= 12 and
                    old.get("last_d", 9999) < 48 and
                    (old.get("key") or old.get("category", "").lower() in
                     {"weapon", "health", "armor", "ammo"})):
                old["acquired"] = True
                if old.get("key"):
                    self.keys_collected.add(old["name"])
                self.recent_events.append({"tic": t, "kind": "landmark_contact",
                                           "name": old["name"], "id": ident})
                self.ctx.log("landmark_contact", name=old["name"], id=ident,
                             key=bool(old.get("key")))

    def update(self, obs):
        t = obs["tic"]
        if t <= self.last_tic:
            return False
        self.update_kinematics(obs)
        c = cell(obs["p"])
        was_new = c not in self.cells
        self.cells[c] += 1
        self.heading_visits[(c, heading_bin(obs["angle"]))] += 1
        if was_new:
            self.last_new_cell = t
        self.remember_landmarks(obs)
        counts = (obs["kills"], obs["items"], obs["secrets"])
        if counts != self.last_counts:
            self.last_counts = counts
            self.recent_events.append({"tic": t, "kind": "counter_change",
                                       "counts": list(counts)})
        # A compact view signature gives the survey a memory that is different
        # from merely counting the number of visited coordinate cells.
        names = sorted({x["name"] for x in obs["objects"]
                        if not x["monster"] and not x["projectile"]})[:8]
        signature = (tuple(names), round(obs["p"][2] / 32.0),
                     "threat" if obs["threats"] else "quiet")
        self.view_visits[(c, heading_bin(obs["angle"]), signature)] += 1
        self.previous = obs
        self.previous_position = list(obs["p"])
        self.last_tic = t
        return True

    # ---------- landmark and survey planning ----------
    def candidate_landmarks(self, obs, mission=None):
        t = obs["tic"]
        rows = []
        for ident, record in self.landmarks.items():
            if record.get("acquired"):
                continue
            age = t - record.get("last_seen", t)
            if age > 130:
                continue
            value = self.landmark_value(record, obs, mission)
            if value <= 0:
                continue
            d = distance(obs["p"], record["p"])
            if d > 1500:
                continue
            q = dict(record)
            q["id"] = ident
            q["d"] = d
            q["dz"] = record["p"][2] - obs["p"][2]
            q["err"] = wrap(bearing(obs["p"], record["p"]) - obs["angle"])
            q["age"] = age
            q["value"] = value - .22 * d - .8 * age
            rows.append(q)
        rows.sort(key=lambda x: x["value"], reverse=True)
        return rows

    def mission(self):
        if self.program:
            current = self.program[0]
            if current in VALID_PROGRAM:
                return current
        return self.policy.get("mission", "key_first")

    def choose_goal(self, obs):
        mission = self.mission()
        candidates = self.candidate_landmarks(obs, mission)
        if mission == "seek_exit":
            exit_rows = [x for x in candidates if x["interact"] or
                         "exit" in x["name"].lower()]
            chosen = exit_rows[0] if exit_rows else (candidates[0] if candidates else None)
        elif mission == "recover":
            safe = [x for x in candidates if x["category"].lower() in {"health", "armor"}]
            chosen = safe[0] if safe else (candidates[0] if candidates else None)
        else:
            chosen = candidates[0] if candidates else None
        if chosen is None:
            self.goal = None
            return None
        if self.goal and self.goal.get("id") == chosen["id"]:
            if chosen["d"] < self.goal_best - 8:
                self.goal_best = chosen["d"]
                self.goal_last_progress = obs["tic"]
            self.goal["record"] = chosen
            if (obs["tic"] - self.goal_last_progress > 72 or
                    obs["tic"] - self.goal.get("started", obs["tic"]) > 190):
                self.goal_failures[chosen["id"]] += 1
                self.goal["record"]["attempts"] = self.goal_failures[chosen["id"]]
                self.goal_cooldown[chosen["id"]] = obs["tic"] + 240
                self.recent_events.append({"tic": obs["tic"], "kind": "landmark_stalled",
                                           "id": chosen["id"]})
                self.goal = None
                return None
            return self.goal
        if self.goal_cooldown.get(chosen["id"], -1) > obs["tic"]:
            return None
        self.goal = {"id": chosen["id"], "record": chosen,
                     "started": obs["tic"], "best": chosen["d"]}
        self.goal_best = chosen["d"]
        self.goal_last_progress = obs["tic"]
        self.ctx.log("landmark_goal", id=chosen["id"], name=chosen["name"],
                     category=chosen["category"], height=round(chosen["dz"]),
                     distance=round(chosen["d"]), mission=mission)
        return self.goal

    def choose_heading(self, obs, preferred=None):
        """Choose a departure from surveyed evidence, not only heading counts."""
        here = cell(obs["p"])
        base = obs["angle"]
        preferred = preferred if preferred in (-1, 1) else self.scan_side
        options = []
        for h in range(8):
            angle = h * 45.0
            projected = cell(point(obs["p"], angle, 150))
            visits = self.cells.get(projected, 0)
            local = self.heading_visits.get((here, h), 0)
            failed = self.portal_faces.get((here, h), -10000)
            penalty = 100 if failed > obs["tic"] else 0
            evidence = self.view_evidence.get((here, h), 0.0)
            turn_cost = abs(wrap(angle - base)) / 180.0
            side_bias = 0 if ((1 if wrap(angle - base) >= 0 else -1) == preferred) else .3
            # An observed useful landmark gets a large bonus; a route through a
            # repeatedly seen cell gets a cost.  This is view-conditioned search,
            # not a blind square sweep.
            score = penalty + 2.2 * local + .55 * visits - 1.7 * evidence
            score += turn_cost + side_bias
            options.append((score, h, angle))
        return min(options)[-1]

    def start_travel(self, obs, heading=None, duration=65, reason="survey_departure"):
        if heading is None:
            heading = self.choose_heading(obs)
        self.mode = "travel"
        self.travel_heading = number(heading) % 360.0
        self.travel_started = obs["tic"]
        self.travel_until = obs["tic"] + max(20, int(duration))
        self.travel_origin = list(obs["p"])
        self.ctx.log("travel_commit", tic=obs["tic"], heading=round(self.travel_heading, 1),
                     duration=int(duration), reason=reason)

    def start_scan(self, obs, reason="scheduled_survey"):
        if self.scan is not None:
            return
        if obs["tic"] - self.last_scan < (180 if self.policy.get("scan") == "frequent" else 300):
            return
        if obs["health"] < 32 or obs["tic"] - self.last_damage_tic < 25:
            return
        self.scan = {"remaining": 8, "segment": 0, "samples": [],
                     "anchor": obs["angle"], "cell": cell(obs["p"]),
                     "start": obs["tic"], "reason": reason}
        self.last_scan = obs["tic"]
        self.scan_count += 1
        self.goal = None
        self.ctx.log("panorama_start", tic=obs["tic"], reason=reason,
                     cell=list(self.scan["cell"]), count=self.scan_count)

    def sample_scan(self, obs):
        if self.scan is None:
            return
        visible = []
        for x in obs["objects"]:
            if x["monster"] or x["projectile"]:
                continue
            if x["pickup"] or x["d"] > 120:
                visible.append({"name": x["name"], "id": x["id"],
                                "category": x["category"], "p": list(x["p"]),
                                "d": round(x["d"], 1), "key": bool(x["key"]),
                                "interact": bool(x["interact"])})
        self.scan["samples"].append({"angle": obs["angle"], "objects": visible})

    def finish_scan(self, obs):
        scan = self.scan
        if scan is None:
            return
        best = None
        for sample in scan["samples"]:
            h = heading_bin(sample["angle"])
            score = 0.0
            for row in sample["objects"]:
                if row["interact"]:
                    score += 12
                elif row["key"]:
                    score += 10
                elif row["category"].lower() in {"weapon", "health", "armor"}:
                    score += 5
                else:
                    score += .4
                self.view_evidence[(scan["cell"], h)] += score
                if best is None or score > best[0]:
                    best = (score, row, sample["angle"])
        self.scan_bearings = []
        # Prefer the absolute world position of an observed capability/exit;
        # the next approach may be vertical and therefore cannot use a simple
        # same-floor distance filter.
        for ident, record in self.landmarks.items():
            if record.get("last_seen", -1) >= scan["start"] and not record.get("acquired"):
                if record.get("key") or record.get("interact"):
                    self.scan_bearings.append(bearing(obs["p"], record["p"]))
        if self.scan_bearings:
            desired = self.scan_bearings[0]
            reason = "new_landmark_bearing"
        elif best is not None and best[0] > 1.0:
            desired = best[2]
            reason = "useful_view"
        else:
            desired = self.choose_heading(obs, self.scan_side)
            reason = "least_repeated_view"
        self.scan = None
        self.scan_side *= -1
        self.start_travel(obs, desired, duration=82, reason=reason)
        self.ctx.log("panorama_finish", tic=obs["tic"], samples=8,
                     selected_heading=round(desired, 1), evidence=reason)

    def scan_action(self, obs):
        if self.scan is None:
            return None
        if self.scan["segment"] <= 0:
            self.sample_scan(obs)
            self.scan["remaining"] -= 1
            if self.scan["remaining"] <= 0:
                self.finish_scan(obs)
                return self.legal("forward")
            self.scan["segment"] = 9
        self.scan["segment"] -= 1
        return self.legal("turn_left" if self.scan_side > 0 else "turn_right", "forward")

    # ---------- interaction and movement ----------
    def start_portal_test(self, obs, reason="blocked_face"):
        if self.portal is not None:
            return
        face = (cell(obs["p"]), heading_bin(obs["angle"]))
        if self.portal_faces.get(face, -10000) > obs["tic"]:
            self.start_travel(obs, self.choose_heading(obs, -self.scan_side), 55,
                               "tested_face_alternate")
            return
        self.portal = {"phase": "settle", "start": obs["tic"],
                       "origin": list(obs["p"]), "face": face,
                       "heading": obs["angle"], "before_z": obs["p"][2]}
        self.ctx.log("portal_test_start", tic=obs["tic"], face=str(face), reason=reason)

    def portal_action(self, obs):
        portal = self.portal
        if portal is None:
            return None
        t = obs["tic"]
        if t - self.last_damage_tic < 12:
            self.portal = None
            self.portal_faces[portal["face"]] = t + 420
            self.start_escape(obs, "damage_during_portal")
            return self.escape.popleft() if self.escape else self.legal("backward")
        if portal["phase"] == "settle":
            if t - portal["start"] < 6:
                return self.legal("forward", "strafe_left", "strafe_right")
            if t - self.last_use < 12:
                return self.legal("forward")
            self.last_use = t
            portal["phase"] = "wait"
            portal["pulse"] = t
            self.ctx.log("portal_use", tic=t, face=str(portal["face"]))
            return self.legal("use", "forward")
        if portal["phase"] == "wait":
            moved = distance(obs["p"], portal["origin"])
            if moved > 24 or abs(obs["p"][2] - portal["before_z"]) > 18:
                portal["phase"] = "cross"
                portal["origin"] = list(obs["p"])
                return self.legal("forward")
            if t - portal["pulse"] < 10:
                return self.legal("forward", "strafe_left", "strafe_right")
            portal["phase"] = "cross"
            portal["origin"] = list(obs["p"])
            portal["cross_start"] = t
            return self.legal("forward")
        moved = distance(obs["p"], portal["origin"])
        if moved > 42 or abs(obs["p"][2] - portal["before_z"]) > 18:
            self.recent_events.append({"tic": t, "kind": "portal_pass", "face": str(portal["face"]),
                                       "moved": round(moved)})
            self.ctx.log("portal_pass", tic=t, face=str(portal["face"]), moved=round(moved))
            self.portal = None
            self.last_transition = t
            self.start_travel(obs, portal["heading"], 100, "portal_passage")
            return self.legal("forward")
        if t - portal.get("cross_start", t) > 22:
            self.portal_faces[portal["face"]] = t + 600
            self.ctx.log("portal_blocked", tic=t, face=str(portal["face"]))
            self.recent_events.append({"tic": t, "kind": "portal_blocked",
                                       "face": str(portal["face"])})
            self.portal = None
            self.force_scan = True
            self.start_travel(obs, self.choose_heading(obs, -self.scan_side), 45,
                               "portal_alternate")
            return self.legal("backward", "strafe_left", "strafe_right")
        return self.legal("forward")

    def target_action(self, obs, goal):
        record = goal.get("record") or {}
        q = record.get("p", obs["p"])
        err = wrap(bearing(obs["p"], q) - obs["angle"])
        t = obs["tic"]
        # A vertical landmark is a relay objective: approach its horizontal
        # projection, then resurvey rather than declaring the target unreachable.
        tolerance = max(7.0, self.turn_rate * 1.25)
        turned = self.turn_to(err, t, tolerance, lease=7)
        if turned is not None:
            return turned
        if record.get("interact") and record.get("d", 9999) < 58:
            if t - self.last_use >= 12:
                self.last_use = t
                self.ctx.log("landmark_use", tic=t, name=record.get("name"), id=record.get("id"))
                return self.legal("use", "forward")
            return self.legal("forward")
        if record.get("d", 9999) < 34:
            self.goal = None
            return self.legal("forward")
        return self.legal("forward")

    def movement_action(self, obs):
        t = obs["tic"]
        if self.travel_heading is None or t >= self.travel_until:
            self.start_travel(obs, self.choose_heading(obs), 70, "travel_expired")
        err = wrap(self.travel_heading - obs["angle"])
        turned = self.turn_to(err, t, max(8.0, self.turn_rate * 1.35), lease=8)
        if turned is not None:
            return turned
        if self.blocked >= 5:
            self.start_portal_test(obs, "measured_no_displacement")
            if self.portal is not None:
                return self.portal_action(obs)
        if t - self.travel_started > 90 and distance(obs["p"], self.travel_origin or obs["p"]) < 18:
            self.start_escape(obs, "travel_no_displacement")
            return self.escape.popleft() if self.escape else self.legal("backward")
        return self.legal("forward", "forward_left", "forward_right")

    def start_escape(self, obs, reason):
        if self.escape or obs["tic"] - self.last_escape < 75:
            return
        side = self.scan_side
        lateral = "left" if side > 0 else "right"
        retreat = self.legal("backward_" + lateral, "backward", "strafe_" + lateral)
        depart = self.legal("forward_" + lateral, "forward", "strafe_" + lateral)
        desired = self.choose_heading(obs, side)
        turn = self.turn_to(wrap(desired - obs["angle"]), obs["tic"], 0, lease=1)
        if turn not in TURN:
            turn = self.legal("turn_left" if side > 0 else "turn_right", "forward")
        self.escape.extend([retreat] * 6 + [turn] * max(8, int(35 / max(2, self.turn_rate))) +
                           [depart] * 18)
        self.last_escape = obs["tic"]
        self.stationary = 0
        self.blocked = 0
        self.portal = None
        self.goal = None
        self.ctx.log("escape_commit", tic=obs["tic"], reason=reason,
                     heading=round(desired, 1), side="left" if side > 0 else "right")

    # ---------- combat ----------
    def incoming(self, obs):
        for x in obs["objects"]:
            if not x["projectile"] or x["d"] > 300:
                continue
            rx = x["p"][0] - obs["p"][0]
            ry = x["p"][1] - obs["p"][1]
            vx = x["v"][0] - obs["v"][0]
            vy = x["v"][1] - obs["v"][1]
            vv = vx * vx + vy * vy
            tau = max(0, min(16, -(rx * vx + ry * vy) / vv)) if vv > 1 else 0
            close = math.hypot(rx + vx * tau, ry + vy * tau)
            if ((0 < tau < 14 and close < 62 and -15 < x["dz"] < 72) or
                    (x["d"] < 72 and -15 < x["dz"] < 72)):
                return x
        return None

    def weapon_action(self, obs, t):
        if t - self.last_weapon < 38:
            return None
        current = obs["weapon"]
        current_ammo = obs["weapon_ammo"]
        desired = None
        # Do not spend a pulse if the current workhorse still has ammunition.
        for slot in (6, 4, 3, 2):
            if not bool(obs["weapons"].get(str(slot), False)):
                continue
            available = number(obs["ammo"].get(str(slot)), 0)
            if slot == current:
                available = max(available, current_ammo)
            if available > 0:
                desired = slot
                break
        if desired is None or desired == current:
            return None
        if current_ammo > 0 and current in {3, 4, 6}:
            return None
        action = "select_weapon_" + str(desired)
        if action in self.allowed:
            self.last_weapon = t
            return action
        return None

    def combat_action(self, obs, threat):
        t = obs["tic"]
        switched = self.weapon_action(obs, t)
        if switched:
            return switched
        self.combat_id = threat["id"]
        self.combat_until = t + 100
        error = threat["err"]
        turned = self.turn_to(error, t, max(8, self.turn_rate * 1.35), lease=5)
        if turned is not None and self.stationary < 24:
            return turned
        if obs["health"] < 25 and threat["d"] > 120 and not threat["major"]:
            self.start_escape(obs, "critical_health_combat")
            return self.escape.popleft() if self.escape else self.legal("backward")
        if obs["weapon_ammo"] <= 0:
            side = "left" if self.dodge_side > 0 else "right"
            return self.legal("backward_" + side, "backward", "strafe_" + side)
        self.last_fire = t
        self.dodge_side *= -1 if threat["melee"] or threat["d"] < 150 else 1
        fire = "fire_strafe_left" if self.dodge_side > 0 else "fire_strafe_right"
        if threat["d"] < 100 or threat["melee"]:
            return self.legal(fire, "backward_left" if self.dodge_side > 0 else "backward_right",
                              "fire_forward", "fire")
        return self.legal(fire, "fire_forward", "fire")

    def immediate_threat(self, obs):
        for x in obs["threats"]:
            if x["d"] < 135 or x["major"] or (x["hitscan"] and x["d"] < 560):
                return x
            if obs["tic"] - self.last_damage_tic < 22 and x["d"] < 520:
                return x
        return None

    # ---------- typed model bridge ----------
    def compact_state(self, obs):
        goal = self.goal.get("record") if self.goal else None
        candidates = self.candidate_landmarks(obs, self.mission())
        options = []
        for row in candidates[:6]:
            options.append({"option": "lm:" + row["id"],
                            "id": row["id"], "name": row["name"],
                            "category": row["category"],
                            "range": "contact" if row["d"] < 75 else "near" if row["d"] < 220 else "far",
                            "height": "same_floor" if abs(row["dz"]) <= 40 else
                            "above" if row["dz"] > 0 else "below",
                            "kind": "exit_or_switch" if row["interact"] else
                            "key" if row["key"] else row["category"].lower()})
        p = self.policy
        return {
            "task": "Reach and activate the map exit alive; keys/capability are prerequisites and optional loot is secondary.",
            "rules": "Labels are currently rendered only. Code computes bearings, height, distance, ammo, collision, and legality. A vertical landmark is a route clue, not proof of a reachable straight line. Choose only supplied objectives.",
            "mission": self.mission(),
            "policy": dict(p),
            "player": {"health": "critical" if obs["health"] < 25 else "wounded" if obs["health"] < 60 else "healthy",
                       "armor": "none" if obs["armor"] < 10 else "present",
                       "layer": round(obs["p"][2] / 32.0),
                       "damage_recent": obs["tic"] - self.last_damage_tic < 30},
            "motion": {"mode": "scan" if self.scan else "portal_test" if self.portal else self.mode,
                       "blocked": self.blocked >= 4, "stationary_tics": min(60, self.stationary),
                       "new_cell_recent": obs["tic"] - self.last_new_cell < 35,
                       "visited_cells": min(500, len(self.cells)),
                       "view_count": min(500, len(self.view_visits)),
                       "layer_transition_recent": obs["tic"] - self.last_transition < 80},
            "threats": self.object_descriptions(obs["threats"], 4),
            "landmark_options": options,
            "current_goal": {"id": goal.get("id"), "name": goal.get("name"),
                             "height": "same_floor" if abs(goal.get("dz", 0)) <= 40 else "different_floor"}
            if goal else None,
            "portal": {"phase": self.portal.get("phase"), "known_faces": len(self.portal_faces)}
            if self.portal else {"phase": "none", "known_faces": len(self.portal_faces)},
            "recent_evidence": list(self.recent_events)[-5:],
        }

    def questions(self, state, reverse=False):
        route_options = {
            "keep": "Keep the code-owned current movement or verified combat skill; no listed objective justifies a change.",
            "scan": "Start one bounded panorama when safe to identify a new landmark or route bearing.",
            "probe": "A measured blocked face or supplied interaction merits one controlled USE experiment; do not invent a door.",
        }
        for row in state.get("landmark_options", []):
            route_options[row["option"]] = (
                "Approach this currently or recently rendered landmark: " +
                str(row["name"]) + "; it is an explicit code-supplied candidate, not hidden map data.")
        if reverse:
            route_options = dict(reversed(list(route_options.items())))
        return {
            "objective": Choice(
                "Choose one supplied local objective using `mission`, `motion`, `threats`, and `landmark_options`. "
                "Never invent a map route, target, key, or exit. Keep is safest when evidence is weak.",
                route_options),
            "stance": Choice(
                "Choose the useful combat/traversal side only from the named facts. This changes a future strafe bias, not the target or legality.",
                {"left": "Use left-side circle movement while preserving a retreat lane.",
                 "right": "Use right-side circle movement while preserving a retreat lane.",
                 "back": "Create separation from a close or damaging threat.",
                 "none": "No special side is justified."}),
            "use": Noul(
                "Do the supplied facts justify one USE pulse now or at the next controlled contact?",
                {"true": "A rendered interaction or a measured blocked face supports one bounded USE test.",
                 "false": "No supplied interaction evidence supports USE yet."}),
        }

    def launch_jev(self, obs):
        if self.ctx.system1 is None or self.jev_future is not None or obs["tic"] < self.next_jev:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if number(remaining.get("system1_calls")) <= 8 or number(remaining.get("wall_s")) < 3:
            return
        state = self.compact_state(obs)
        questions = self.questions(state, bool(self.jev_calls % 2))
        self.jev_request = {"tic": obs["tic"], "state": state,
                            "options": {x["option"] for x in state["landmark_options"]}}
        self.jev_calls += 1
        self.next_jev = obs["tic"] + (22 if obs["threats"] else 52)
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=1.35))
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
        except Exception as exc:
            self.jev_future = None
            self.ctx.log("jev_launch_failure", error=type(exc).__name__)

    def poll_jev(self, obs):
        if self.jev_future is None or not self.jev_future.done():
            return
        future = self.jev_future
        request = self.jev_request or {}
        self.jev_future = None
        self.jev_request = None
        try:
            answers = future.result()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, EpisodeOver, CancelledError):
            return
        except Exception as exc:
            self.ctx.log("jev_failure", error=type(exc).__name__)
            return
        age = obs["tic"] - integer(request.get("tic"), obs["tic"])
        if age > 52:
            self.ctx.log("jev_discard", age_tics=age, reason="stale")
            return
        allowed = {"keep", "scan", "probe"} | set(request.get("options", set()))
        # The model cannot be allowed to return a newly invented lm:id.
        objective = accepted_choice(answers, "objective", allowed, .61)
        stance = accepted_choice(answers, {"left": "stance"}.get("left", "stance"),
                                 {"left", "right", "back", "none"}, .56)
        if objective and objective.startswith("lm:"):
            ident = objective[3:]
            if any(x["id"] == ident for x in self.candidate_landmarks(obs, self.mission())):
                self.jev_hint = {"id": ident, "expires": obs["tic"] + 55}
                self.ctx.log("jev_landmark_hint", id=ident, age_tics=age)
        elif objective == "scan":
            self.force_scan = True
        elif objective == "probe":
            self.force_use = True
        if stance == "left":
            self.dodge_side = 1
        elif stance == "right":
            self.dodge_side = -1
        elif stance == "back":
            self.start_escape(obs, "jev_separation")
        if accepted_yes(answers, "use", .74):
            self.force_use = True
        self.ctx.log("jev_survey_bridge", objective=objective, stance=stance,
                     use=self.force_use, age_tics=age)

    def launch_astra(self, obs):
        if self.astra_future is not None or self.astra_calls >= 2 or obs["tic"] < self.next_astra:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if number(remaining.get("system2_calls")) <= 0 or number(remaining.get("wall_s")) < 180:
            return
        records = []
        for row in sorted(self.landmarks.values(), key=lambda x: x.get("last_seen", 0), reverse=True)[:12]:
            records.append({"id": row.get("id"), "name": row.get("name"),
                            "category": row.get("category"), "p": [round(number(v), 1) for v in row.get("p", [0, 0, 0])],
                            "last_seen_age": obs["tic"] - integer(row.get("last_seen"), obs["tic"]),
                            "key": bool(row.get("key")), "interact": bool(row.get("interact")),
                            "acquired": bool(row.get("acquired"))})
        payload = {"map": str(obs["raw"].get("map", "unknown")),
                   "iwad": str(obs["raw"].get("iwad", "unknown")),
                   "skill": integer(obs["raw"].get("skill")),
                   "tic": obs["tic"], "health": round(obs["health"]),
                   "armor": round(obs["armor"]), "weapon": obs["weapon"],
                   "ammo": round(obs["weapon_ammo"]), "counts": [obs["kills"], obs["items"], obs["secrets"]],
                   "position": [round(number(v), 1) for v in obs["p"]],
                   "cells": len(self.cells), "views": len(self.view_visits),
                   "portal_faces": len(self.portal_faces), "events": list(self.recent_events)[-12:],
                   "landmarks": records, "policy": dict(self.policy)}
        prompt = (
            "You are Astra's strategic layer for a real-time Classic Doom controller. "
            "The motor below already owns aiming, ammo legality, collision, bounded panorama scans, "
            "and verified USE experiments. Exit completion dominates optional loot. Based only on the "
            "public evidence, return ONLY compact JSON with fields: "
            "program (a list of at most 4 tokens from seek_keys, seek_exit, survey, probe, recover), "
            "mission (key_first|seek_exit|explore|recover), scan (rare|normal|frequent), "
            "combat (blockers|avoid_optional), reason (under 120 characters). "
            "A program token is an abstract future objective, never a button or hidden coordinate. "
            "Use survey when landmarks are absent or stale; use seek_keys only when key evidence exists; "
            "never claim that a visible object proves a walkable route. Garbage or missing fields will be ignored.\n" +
            json.dumps(payload, separators=(",", ":"), allow_nan=False))
        session = self.astra_session
        self.astra_calls += 1
        self.next_astra = obs["tic"] + 4200
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, session_id=session, timeout_s=190.0))
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("astra_launch_failure", error=type(exc).__name__)
        self.ctx.log("astra_survey_review_requested", call=self.astra_calls,
                     landmarks=len(records), cells=len(self.cells))

    def poll_astra(self, obs):
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
        except Exception as exc:
            self.ctx.log("astra_failure", error=type(exc).__name__)
            return
        proposal = parse_json_object(getattr(result, "text", ""))
        program = proposal.get("program")
        valid_program = []
        if isinstance(program, list):
            for token in program[:4]:
                if isinstance(token, str) and token in VALID_PROGRAM:
                    valid_program.append(token)
        changed = bool(valid_program)
        if valid_program:
            self.program.clear()
            self.program.extend(valid_program)
            self.program_until = obs["tic"] + 650
        for key, values in {
            "mission": {"key_first", "seek_exit", "explore", "recover"},
            "scan": {"rare", "normal", "frequent"},
            "combat": {"blockers", "avoid_optional"},
        }.items():
            value = proposal.get(key)
            if isinstance(value, str) and value in values:
                if key == "mission":
                    self.policy["mission"] = value
                else:
                    self.policy[key] = value
                changed = True
        self.astra_session = getattr(result, "session_id", self.astra_session)
        if changed:
            self.ctx.log("astra_survey_program", program=list(self.program),
                         policy=dict(self.policy), reason=str(proposal.get("reason", ""))[:120])

    # ---------- main priority bridge ----------
    def action(self, obs):
        t = obs["tic"]
        missile = self.incoming(obs)
        if missile is not None:
            self.dodge_side *= -1
            self.ctx.log("missile_dodge", tic=t, name=missile["name"])
            return self.legal("strafe_left" if self.dodge_side > 0 else "strafe_right",
                              "backward_left", "backward_right")
        if self.escape:
            return self.escape.popleft()
        if self.scan is not None:
            # Abort the panorama rather than spend its remaining turn lease in a
            # newly dangerous room.
            if self.immediate_threat(obs) is not None or obs["health"] < 32:
                self.scan = None
            else:
                return self.scan_action(obs)
        threat = self.immediate_threat(obs)
        if threat is not None:
            self.goal = None
            return self.combat_action(obs, threat)
        if self.stationary >= 30 and t - self.last_escape > 75:
            close_interaction = any(x["interact"] and x["d"] < 125 for x in obs["objects"])
            if not close_interaction:
                self.start_escape(obs, "stationary_watchdog")
                if self.escape:
                    return self.escape.popleft()
        if self.portal is not None:
            return self.portal_action(obs)

        goal = self.choose_goal(obs)
        if self.jev_hint and self.jev_hint.get("expires", -1) >= t:
            hinted = next((x for x in self.candidate_landmarks(obs, self.mission())
                           if x["id"] == self.jev_hint.get("id")), None)
            if hinted:
                if not goal or hinted["value"] >= goal.get("record", {}).get("value", -1):
                    self.goal = {"id": hinted["id"], "record": hinted,
                                 "started": t, "best": hinted["d"]}
                    goal = self.goal
            else:
                self.jev_hint = None
        if goal is not None:
            record = goal.get("record") or {}
            if self.force_use and record.get("interact") and record.get("d", 9999) < 90:
                self.force_use = False
                if t - self.last_use >= 12:
                    self.last_use = t
                    return self.legal("use", "forward")
            return self.target_action(obs, goal)

        if self.force_use and (self.blocked >= 4 or any(x["interact"] for x in obs["objects"])):
            self.force_use = False
            if t - self.last_use >= 12:
                self.last_use = t
                return self.legal("use", "forward")
        if self.blocked >= 5:
            self.start_portal_test(obs, "collision_before_goal")
            if self.portal is not None:
                return self.portal_action(obs)
        if self.force_scan or (t - self.last_scan > 300 and not obs["threats"]):
            self.force_scan = False
            self.start_scan(obs, "forced" if self.force_scan else "scheduled")
            if self.scan is not None:
                return self.scan_action(obs)
        if t - self.last_use >= 105 and not obs["threats"] and self.blocked < 3:
            self.last_use = t
            return self.legal("use", "forward")
        return self.movement_action(obs)

    def run_episode(self, game):
        ctx = self.ctx
        try:
            spec = getattr(game, "action_spec", None)
            self.allowed = set(getattr(spec, "names", ()) or ())
            if not self.allowed:
                return
            self.noop = getattr(spec, "noop", "noop")
            if self.noop not in self.allowed:
                self.noop = next(iter(self.allowed))
            while not ctx.done and not bool(getattr(game, "done", False)):
                raw = ctx.observe()
                fallback = self.last_tic + 1 if self.last_tic >= 0 else 0
                obs = parse_observation(raw, fallback)
                if ctx.done or bool(getattr(game, "done", False)) or obs["terminal"] or obs["dead"]:
                    break
                if not self.update(obs):
                    ctx.wait_for_tick()
                    continue
                self.poll_jev(obs)
                self.poll_astra(obs)
                self.launch_jev(obs)
                self.launch_astra(obs)
                action = self.legal(self.action(obs))
                ctx.act(action)
                self.previous_action = action
                ctx.wait_for_tick()
        except (EpisodeOver, BudgetExceeded):
            return
        finally:
            for future in (self.jev_future, self.astra_future):
                if future is not None and not future.done():
                    try:
                        future.cancel()
                    except Exception:
                        pass
            try:
                ctx.memory["landmark_survey_summary"] = {
                    "cells": len(self.cells), "views": len(self.view_visits),
                    "landmarks": len(self.landmarks), "scans": self.scan_count,
                    "portal_faces": len(self.portal_faces),
                    "portal_passes": sum(1 for x in self.recent_events if x.get("kind") == "portal_pass"),
                    "jev_calls": self.jev_calls, "astra_calls": self.astra_calls,
                }
            except Exception:
                pass


def build(ctx):
    return LandmarkSurvey(ctx)
