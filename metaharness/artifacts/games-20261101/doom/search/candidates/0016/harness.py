"""0016: effect-aligned skills for asynchronous Doom.

Only public observations and advertised actions are used. Code identifies the
input/effect delay and executes skills with postconditions. Jev selects among
concrete, observed affordances and diagnoses failed skills. Astra revises the
recognition objective from a ledger of those outcomes, not from raw buttons.
"""
from __future__ import annotations

import bisect
import collections
import json
import math
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


MOVE = {
    "forward": 0, "backward": 180, "strafe_left": 90, "strafe_right": -90,
    "forward_left": 45, "forward_right": -45,
    "backward_left": 135, "backward_right": -135,
    "fire_forward": 0, "fire_strafe_left": 90, "fire_strafe_right": -90,
}
HITSCAN = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS", "SpiderMastermind"}
MAJOR = {"Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind", "Revenant"}
MELEE = {"Demon", "Spectre", "LostSoul", "Pinkie", "Pinky"}
MISSILES = {
    "DoomImpBall", "CacodemonBall", "BaronBall", "Rocket", "RevenantTracer",
    "ArachnotronPlasma", "PlasmaBall", "FatShot", "BFGShot", "Tracer",
}
WEAPON_PICKUPS = {"Shotgun": 3, "SuperShotgun": 3, "Chaingun": 4,
                  "RocketLauncher": 5, "PlasmaRifle": 6, "BFG9000": 7,
                  "Chainsaw": 1}


def num(x, default=0.0):
    try:
        x = float(x)
        return x if math.isfinite(x) else default
    except (ValueError, TypeError, OverflowError):
        return default


def integer(x, default=0):
    return int(num(x, default))


def objdict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            return y if isinstance(y, dict) else {}
        except (ValueError, TypeError):
            pass
    return {}


def parse_model_json(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for i, char in enumerate(text[:10000]):
        if char == "{":
            try:
                result, _ = decoder.raw_decode(text[i:])
                if isinstance(result, dict):
                    return result
            except (ValueError, TypeError):
                pass
    return {}


def wrap(a):
    return (num(a) + 180) % 360 - 180


def distance(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def bearing(p, q):
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 360


def point(p, angle, length):
    r = math.radians(angle)
    return [p[0] + math.cos(r) * length, p[1] + math.sin(r) * length, p[2]]


def cell(p, size=96):
    return (math.floor(p[0] / size), math.floor(p[1] / size), math.floor((p[2] + 8) / 48))


def octant(a):
    return int((a % 360 + 22.5) // 45) % 8


def direction(err):
    return "ahead" if abs(err) < 12 else "left" if err > 0 else "right"


def probability_choice(answers, key, allowed, threshold=0.64):
    if not isinstance(answers, dict):
        return None
    ans = answers.get(key)
    value = getattr(ans, "value", None)
    probs = getattr(ans, "probabilities", None)
    if not isinstance(value, str) or value not in allowed or not isinstance(probs, dict):
        return None
    if num(probs.get(value)) < threshold:
        return None
    confidence = getattr(ans, "confidence", None)
    if confidence is not None and num(confidence) < 0.15:
        return None
    return value


def normalize(raw, fallback_tic):
    raw = objdict(raw)
    pl = objdict(raw.get("player"))
    vel = objdict(raw.get("velocity"))
    p = [num(pl.get(k)) for k in ("x", "y", "z")]
    angle = num(pl.get("angle")) % 360
    obs = {
        "raw": raw, "t": integer(raw.get("tic"), fallback_tic), "p": p, "a": angle,
        "v": [num(vel.get(k)) for k in ("x", "y", "z")],
        "hp": num(raw.get("health"), 100), "armor": num(raw.get("armor")),
        "weapon": integer(raw.get("weapon"), 2), "wa": num(raw.get("weapon_ammo")),
        "ammo": objdict(raw.get("ammo")), "weapons": objdict(raw.get("weapons")),
        "kills": integer(raw.get("kills")), "items": integer(raw.get("items")),
        "secrets": integer(raw.get("secrets")), "damage": num(raw.get("damage_taken")),
        "ground": bool(raw.get("on_ground", True)),
        "done": bool(raw.get("terminal") or raw.get("dead")), "objects": [],
    }
    labels = raw.get("visible")
    if not isinstance(labels, list):
        labels = []
    for index, label in enumerate(labels[:256]):
        if not isinstance(label, dict):
            continue
        w = label.get("world")
        if not isinstance(w, (list, tuple)) or len(w) < 3:
            continue
        name = str(label.get("name", "unknown"))[:64]
        cat = str(label.get("category", "unknown")).lower()[:32]
        if cat == "self" or name == "DoomPlayer":
            continue
        q = [num(x) for x in w[:3]]
        vv = label.get("velocity")
        vv = [num(x) for x in vv[:3]] if isinstance(vv, (list, tuple)) and len(vv) >= 3 else [0, 0, 0]
        box = label.get("bbox")
        screen = [num(x) for x in box[:4]] if isinstance(box, (list, tuple)) and len(box) >= 4 else [160, 120, 0, 0]
        ident = str(label.get("id", index))[:80]
        low = name.lower()
        monster = cat == "monster" and not low.startswith("dead")
        # In this build BulletPuff and actual missiles both have category
        # Explosive. A hitscan impact puff is not an incoming projectile.
        missile = (name in MISSILES or cat in {"projectile", "missile"} or
                   (cat == "explosive" and any(s in low for s in ("ball", "rocket", "tracer", "plasma", "fatshot"))))
        interaction = (cat in {"door", "exit", "switch", "teleporter", "teleport"} or
                       any(s in low for s in ("exit", "switch", "teleport", "door")))
        ob = {
            "id": ident, "name": name, "cat": cat, "p": q, "v": vv,
            "box": screen, "d": distance(p, q), "dz": q[2] - p[2],
            "err": wrap(bearing(p, q) - angle), "monster": monster,
            "missile": missile, "interact": interaction,
            "key": cat == "key", "hitscan": monster and name in HITSCAN,
            "major": monster and name in MAJOR, "melee": monster and name in MELEE,
        }
        # A clipped box at the screen edge is weaker evidence of a direct
        # line-of-fire than a box crossing the reticle.  This is only a local
        # urgency feature; it is not a visibility/geometry oracle.
        left, width = screen[0], screen[2]
        ob["reticle_overlap"] = left <= 160 <= left + max(0.0, width)
        obs["objects"].append(ob)
    obs["by_id"] = {x["id"]: x for x in obs["objects"]}
    obs["threats"] = sorted([x for x in obs["objects"] if x["monster"]],
                            key=lambda x: (220 if x["hitscan"] else 0) +
                            (240 if x["major"] else 0) + (260 if x["d"] < 110 else 0) - x["d"] * .35,
                            reverse=True)
    return obs


class EffectModel:
    """Fit input delay from observed yaw; predict already queued yaw commands."""
    def __init__(self):
        self.times = []
        self.commands = []
        self.lag = 3
        self.rate = 3.515625
        self.left_sign = 1
        self.errors = [20.0] * 8
        self.samples = 0
        self.last = None
        self.calibration_logged = False

    def yaw(self, action):
        return self.left_sign if action == "turn_left" else -self.left_sign if action == "turn_right" else 0

    def at(self, tic):
        i = bisect.bisect_right(self.times, tic) - 1
        return self.commands[i] if i >= 0 else "noop"

    def post(self, tic, action):
        if self.times and self.times[-1] == tic:
            self.commands[-1] = action
        else:
            self.times.append(tic)
            self.commands.append(action)
        if len(self.times) > 300:
            self.times = self.times[-180:]
            self.commands = self.commands[-180:]

    def update(self, obs):
        old = self.last
        self.last = obs
        if old is None:
            return
        dt = obs["t"] - old["t"]
        if not 0 < dt <= 4:
            return
        delta = wrap(obs["a"] - old["a"])
        signs = [self.yaw(self.at(s)) for s in range(obs["t"] - 12, obs["t"])]
        if signs and all(x == signs[0] and x != 0 for x in signs) and .3 < abs(delta / dt) < 12:
            if delta * signs[0] < 0:
                self.left_sign *= -1
            self.rate = .94 * self.rate + .06 * abs(delta / dt)
        # Only transitions identify latency; steady yaw alone does not.
        if len(set(signs)) > 1 and any(signs):
            for lag in range(8):
                predicted = sum(self.yaw(self.at(s - lag)) * self.rate
                                for s in range(old["t"], obs["t"]))
                self.errors[lag] = .97 * self.errors[lag] + .03 * min(200, (delta - predicted) ** 2)
            self.samples += 1
            if self.samples > 20:
                self.lag = min(range(8), key=lambda l: self.errors[l] + .03 * abs(l - 3))

    def predicted_angle(self, obs):
        future_yaw = sum(self.yaw(self.at(s)) * self.rate
                         for s in range(obs["t"] - self.lag, obs["t"]))
        return (obs["a"] + future_yaw) % 360

    def predicted_position(self, obs):
        return [obs["p"][i] + obs["v"][i] * min(4, self.lag) for i in range(3)]

    def error_to(self, obs, destination, target_velocity=None):
        q = list(destination)
        if target_velocity:
            q = [q[i] + target_velocity[i] * self.lag for i in range(3)]
        return wrap(bearing(self.predicted_position(obs), q) - self.predicted_angle(obs))

    def turn(self, error):
        return "turn_left" if error * self.left_sign > 0 else "turn_right"


class SkillController:
    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.noop = "noop"
        self.effect = EffectModel()
        self.last_tic = -1
        self.last = None
        self.last_action = "noop"
        self.started = 0.0
        self.skill = None
        self.serial = 0
        self.outcomes = collections.deque(maxlen=36)
        self.outcome_serial = 0
        self.visits = collections.Counter()
        self.departures = collections.Counter()
        self.contacts = {}
        self.failed_headings = {}
        self.cooldowns = {}
        self.trail = collections.deque(maxlen=220)
        self.last_novel = 0
        self.last_damage = -1000
        self.damage_recent = collections.deque(maxlen=32)
        self.action_window = collections.deque(maxlen=100)
        self.key_near = {}
        self.inferred_keys = set()
        self.key_epoch = 0
        self.next_use = -1
        self.weapon_pending = None
        self.weapon_last = -1000
        self.dodge_side = 1
        self.dodge_until = -1
        self.dodge_action = None
        self.last_side_change = -1000
        self.fire_hold = -1
        self.fire_action = None
        self.contact_passes = 0
        self.contact_tests = 0
        self.jev_future = None
        self.jev_meta = None
        self.next_jev = 10
        self.jev_calls = 0
        self.jev_hint = None
        self.diagnosis = "unknown"
        self.astra_future = None
        self.astra_calls = 0
        self.astra_session = None
        self.next_astra = 30
        self.interface_version = 0
        self.policy = {"focus": "capability", "encounter": "blockers",
                       "travel": "persistent", "interaction": "patient"}
        self.next_summary = 175

    def legal(self, *actions):
        for action in actions:
            if action in self.allowed:
                return action
        return self.noop

    def update(self, obs):
        self.effect.update(obs)
        t, p = obs["t"], obs["p"]
        old = self.last
        if old:
            loss = max(0, old["hp"] - obs["hp"], obs["damage"] - old["damage"])
            if loss > 0:
                self.last_damage = t
                self.damage_recent.append((t, loss))
            for ident, key in list(self.key_near.items()):
                if ident not in obs["by_id"] and t - key["t"] < 20:
                    self.inferred_keys.add(key["name"])
                    self.key_epoch += 1
                    self.ctx.log("key_contact_inference", name=key["name"], epoch=self.key_epoch,
                                 evidence="rendered key disappeared after same-level overlap; not inventory confirmation")
                    self.key_near.pop(ident, None)
                elif t - key["t"] > 30:
                    self.key_near.pop(ident, None)
        for x in obs["objects"]:
            if x["key"] and x["d"] < 34 and abs(x["dz"]) < 32 and x["name"] not in self.inferred_keys:
                self.key_near[x["id"]] = {"t": t, "name": x["name"]}
        c = cell(p)
        if c not in self.visits:
            self.last_novel = t
        self.visits[c] += 1
        if not self.trail or distance(self.trail[-1][1], p) > 30 or abs(self.trail[-1][1][2] - p[2]) > 20:
            self.trail.append((t, list(p), obs["hp"]))
        self.action_window.append((t, list(p), self.last_action, obs["a"]))
        while self.damage_recent and t - self.damage_recent[0][0] > 105:
            self.damage_recent.popleft()
        self.last = obs

    def blocked(self, obs):
        """Account for queued input, acceleration, and pulse interruptions."""
        t = obs["t"]
        window = [r for r in self.action_window if t - 12 <= r[0] < t]
        if len(window) < 8 or not obs["ground"]:
            return False
        translations = [r for r in window if r[2] in MOVE]
        if len(translations) < len(window) - 2:
            return False
        if distance(window[0][1], obs["p"]) > 9:
            return False
        return True

    def failed_direction(self, obs, angle):
        return self.failed_headings.get((cell(obs["p"]), octant(angle)), -1) > obs["t"]

    def safe_side(self, obs, target=None):
        candidates = []
        for side in (self.dodge_side, -self.dodge_side):
            a = obs["a"] + side * 90
            dest = point(obs["p"], a, 90)
            penalty = 100 if self.failed_direction(obs, a) else 0
            # Prefer measured same-level ground to a completely unknown edge
            # while firing. This is a preference, not an invented walkable ray.
            known = cell(dest) in self.visits
            candidates.append((penalty + (0 if known else 2), 0 if side == self.dodge_side else 1, side))
        return min(candidates)[2]

    def missile_dodge(self, obs):
        t = obs["t"]
        imminent = None
        for x in obs["objects"]:
            if not x["missile"] or x["d"] > 270:
                continue
            rx, ry = x["p"][0] - obs["p"][0], x["p"][1] - obs["p"][1]
            vx, vy = x["v"][0] - obs["v"][0], x["v"][1] - obs["v"][1]
            vv = vx * vx + vy * vy
            tau = max(0, min(14, -(rx * vx + ry * vy) / vv)) if vv > 1 else 0
            closest = math.hypot(rx + vx * tau, ry + vy * tau)
            height = x["dz"] + (x["v"][2] - obs["v"][2]) * tau
            if (closest < 55 and 0 < tau < 13 and -12 < height < 70) or (x["d"] < 70 and -12 < height < 70):
                imminent = x
                break
        if imminent is None:
            return None
        if t < self.dodge_until and self.dodge_action:
            return self.dodge_action
        self.dodge_side = self.safe_side(obs, imminent)
        self.dodge_action = self.legal("strafe_left" if self.dodge_side > 0 else "strafe_right", "backward")
        self.dodge_until = t + 8
        self.ctx.log("incoming_missile", id=imminent["id"], action=self.dodge_action)
        return self.dodge_action

    def finish(self, obs, result, evidence=None):
        if not self.skill:
            return
        s = self.skill
        row = {"serial": s["serial"], "kind": s["kind"], "target_name": s.get("name"),
               "result": result, "tics": obs["t"] - s["start"],
               "distance_moved": round(distance(s["origin"], obs["p"])),
               "health_change": round(obs["hp"] - s["hp"]),
               "kill_change": obs["kills"] - s["kills"],
               "ammo_spent": max(0, s["ammo"] - obs["wa"]) if s["weapon"] == obs["weapon"] else None,
               "evidence": str(evidence or "")[:120]}
        self.outcome_serial += 1
        row["outcome_serial"] = self.outcome_serial
        self.outcomes.append(row)
        self.ctx.log("skill_outcome", **row)
        self.skill = None

    def start(self, obs, kind, obj=None, heading=None, destination=None, reason="code"):
        if self.skill:
            self.finish(obs, "preempted", reason)
        self.serial += 1
        s = {"serial": self.serial, "kind": kind, "start": obs["t"], "phase_start": obs["t"],
             "origin": list(obs["p"]), "hp": obs["hp"], "kills": obs["kills"],
             "ammo": obs["wa"], "weapon": obs["weapon"], "phase": "align",
             "id": obj["id"] if obj else None, "name": obj["name"] if obj else None,
             "q": list(obj["p"]) if obj else list(destination) if destination else None,
             "heading": heading, "best": obj["d"] if obj else 99999,
             "progress": obs["t"], "seen": obs["t"], "aim_ticks": 0,
             "shot_tic": obs["t"], "fire_ammo": obs["wa"], "reason": reason}
        if kind == "transit":
            s["phase"] = "travel"
            s["heading"] = heading if heading is not None else obs["a"]
            self.departures[(cell(obs["p"]), octant(s["heading"]))] += 1
        if kind == "contact":
            s["heading"] = heading if heading is not None else obs["a"]
            s["face"] = (cell(obs["p"], 48), octant(s["heading"]), self.key_epoch)
            s["phase"] = "settle"
            self.contact_tests += 1
        if kind == "recover":
            s["phase"] = "backstep"
        self.skill = s
        self.fire_hold = -1
        self.ctx.log("skill_start", serial=s["serial"], kind=kind, target=s["name"],
                     reason=reason, heading=round(s["heading"], 1) if s["heading"] is not None else None)

    def departure(self, obs, exclude=None):
        c, a = cell(obs["p"]), obs["a"]
        options = []
        # Motion attempts, not merely looking in a direction, consume a branch.
        for h in range(8):
            angle = h * 45.0
            q = point(obs["p"], angle, 180)
            attempts = self.departures[(c, h)]
            blocked = self.failed_direction(obs, angle)
            score = 5 * attempts + (30 if blocked else 0)
            score += min(6, self.visits.get(cell(q), 0) / 25)
            score += abs(wrap(angle - a)) / 180
            if exclude is not None and abs(wrap(angle - exclude)) < 50:
                score += 20
            # A long run is preserved until an actual contact or postcondition.
            if self.policy["travel"] == "persistent" and abs(wrap(angle - a)) < 25:
                score -= 1
            options.append((score, abs(wrap(angle - a)), h, angle))
        return min(options)[-1]

    def useful(self, x, obs):
        if self.cooldowns.get("pickup:" + x["id"], -1) > obs["t"]:
            return -1
        if x["monster"] or x["missile"] or x["d"] > 1000:
            return -1
        # A rendered object above/below is not a traversable staircase. Keys
        # remain in the semantic report, but are not direct dive targets.
        if x["dz"] < -30 or x["dz"] > 44:
            return -1
        if x["interact"]:
            value = 1800 if "exit" in x["name"].lower() else 900
        elif x["key"]:
            value = 1100
        elif x["cat"] == "weapon":
            slot = WEAPON_PICKUPS.get(x["name"])
            owned = bool(obs["weapons"].get(str(slot), False)) if slot else False
            value = 1050 if slot in (3, 4, 6) and not owned else 170 if obs["wa"] < 8 else 0
        elif x["cat"] == "health":
            bonus = "bonus" in x["name"].lower()
            # A medikit/stimpack is a survival capability, not optional loot;
            # below 38 health it outranks a newly rendered weapon.
            value = (180 if bonus else 1450) if obs["hp"] < 38 else (80 if bonus else 500) if obs["hp"] < 75 else 0
        elif x["cat"] == "armor":
            bonus = "bonus" in x["name"].lower()
            value = (120 if bonus else 800) if obs["armor"] < 25 else (20 if bonus else 150) if obs["armor"] < 80 else 0
        elif x["cat"] == "ammo":
            shell = "shell" in x["name"].lower()
            value = 550 if obs["wa"] < 5 else 140 if shell and obs["weapons"].get("3") else 35
        else:
            value = 0
        if self.policy["focus"] == "keys" and x["key"]:
            value += 200
        if self.policy["focus"] == "recover" and x["cat"] in {"health", "armor"}:
            value += 120
        return value - .45 * x["d"] if value > 0 else -1

    def urgent(self, obs):
        # Do not let every rendered routine monster seize the route.  A
        # close/body-blocking threat, a major threat, a reticle-overlapping
        # hitscanner, or recent damage is actionable; distant optional actors
        # can be bypassed until their observed attacks make them urgent.
        for x in obs["threats"]:
            if self.cooldowns.get("enemy:" + x["id"], -1) > obs["t"] and x["d"] > 140:
                continue
            close = x["d"] < 125
            near_blocker = x["d"] < 205 and (x["melee"] or x["hitscan"] or x["major"])
            line_hitscan = x["hitscan"] and x["d"] < 430 and x["reticle_overlap"]
            major = x["major"] and x["d"] < 560
            damaged = obs["t"] - self.last_damage < 24 and x["d"] < 560
            if close or near_blocker or line_hitscan or major or damaged:
                return x
        return None

    def affordances(self, obs):
        choices = []
        for x in obs["threats"][:3]:
            if x["d"] < 800:
                choices.append(("enemy:" + x["id"], "fight", x))
        pickups = sorted([(self.useful(x, obs), x) for x in obs["objects"]],
                         key=lambda pair: pair[0], reverse=True)
        for value, x in pickups[:4]:
            if value > 0:
                choices.append(("pickup:" + x["id"], "pickup", x))
        return choices[:6]

    def choose_skill(self, obs):
        hint = self.jev_hint
        if hint and hint["expires"] < obs["t"]:
            self.jev_hint = None
            hint = None
        if hint:
            for key, kind, x in self.affordances(obs):
                if key != hint["key"]:
                    continue
                if kind == "fight" and (x["d"] < 400 or x["hitscan"] or x["major"]):
                    self.jev_hint = None
                    self.start(obs, kind, obj=x, reason="jev_affordance")
                    return
                if kind == "pickup" and self.useful(x, obs) > 0:
                    self.jev_hint = None
                    self.start(obs, kind, obj=x, reason="jev_affordance")
                    return
        urgent = self.urgent(obs)
        if urgent and obs["wa"] > 0:
            self.start(obs, "fight", obj=urgent, reason="immediate_threat")
            return
        pickups = sorted([(self.useful(x, obs), x) for x in obs["objects"]],
                         key=lambda pair: pair[0], reverse=True)
        if pickups and pickups[0][0] > 0:
            self.start(obs, "pickup", obj=pickups[0][1], reason="observed_capability")
            return
        self.start(obs, "transit", heading=self.departure(obs), reason="measured_departure")

    def maybe_switch(self, obs):
        t = obs["t"]
        if self.weapon_pending:
            slot, started = self.weapon_pending
            if obs["weapon"] == slot or t - started > 22:
                self.weapon_pending = None
            else:
                # Selection is a one-tic pulse; keep a translating dodge while
                # the engine acknowledges it rather than parking in noop.
                return self.legal(
                    "strafe_left" if self.dodge_side > 0 else "strafe_right",
                    "backward_left" if self.dodge_side > 0 else "backward_right",
                    self.noop)
        if t - self.weapon_last < 40:
            return None
        for slot in (6, 4, 3, 2):
            if not obs["weapons"].get(str(slot), False):
                continue
            available = num(obs["ammo"].get(str(slot)))
            if slot == obs["weapon"]:
                available = max(available, obs["wa"])
            if available <= 0:
                continue
            if slot == obs["weapon"] or (obs["weapon"] in (3, 4, 6) and obs["wa"] > 0):
                return None
            action = "select_weapon_" + str(slot)
            if action in self.allowed:
                self.weapon_pending = (slot, t)
                self.weapon_last = t
                return action
        return None

    def fight(self, obs, s):
        t = obs["t"]
        x = obs["by_id"].get(s["id"])
        if not x or not x["monster"]:
            if t - s["seen"] > 10:
                self.finish(obs, "target_no_longer_rendered", "kill confirmed" if obs["kills"] > s["kills"] else "visibility loss only")
                return None
            return self.legal("fire" if obs["wa"] > 0 else "strafe_left")
        s["seen"] = t
        s["q"] = list(x["p"])
        switched = self.maybe_switch(obs)
        if switched is not None:
            return switched
        if obs["wa"] <= 0 and obs["weapon"] != 1:
            self.cooldowns["enemy:" + x["id"]] = t + 70
            self.finish(obs, "no_usable_ammo")
            return None
        limit = 75 if x["melee"] else 105 if obs["hp"] < 45 else 150
        if (t - s["start"] > limit and obs["kills"] == s["kills"]
                and (x["d"] > 110 or x["melee"])):
            self.cooldowns["enemy:" + x["id"]] = t + 100
            self.finish(obs, "combat_no_kill", "bounded engagement expired")
            self.start(obs, "recover", heading=self.departure(obs), reason="combat_cost")
            return self.recover(obs, self.skill)
        error = self.effect.error_to(obs, x["p"], x["v"])
        blocked = self.blocked(obs)
        if blocked and self.last_action in MOVE:
            a = obs["a"] + MOVE[self.last_action]
            self.failed_headings[(cell(obs["p"]), octant(a))] = t + 140
            if t - self.last_side_change > 12:
                self.dodge_side *= -1
                self.last_side_change = t
                self.fire_hold = -1
        if obs["wa"] < s["fire_ammo"]:
            s["shot_tic"] = t
            s["fire_ammo"] = obs["wa"]
        # The target ID stays locked. Queued yaw is subtracted before issuing
        # another correction, rather than reacting to a three-tic-old view.
        if self.fire_hold > t and self.fire_action and abs(error) < 24:
            return self.fire_action
        if obs["hp"] < 24 and x["d"] > 125 and not x["major"]:
            # At critical health, first restore separation; a later skill can
            # reacquire the same visible ID instead of spending the last margin
            # on an in-place correction.
            return self.legal(
                "backward_left" if self.dodge_side > 0 else "backward_right",
                "strafe_left" if self.dodge_side > 0 else "strafe_right")
        if abs(error) > 9:
            s["aim_ticks"] += 1
            if s["aim_ticks"] % 28 < 22:
                return self.legal(self.effect.turn(error))
            # Six-tic translating/neutral interval lets input effects drain.
            return self.legal("fire" if abs(x["err"]) < 22 else "strafe_left" if self.dodge_side > 0 else "strafe_right")
        s["aim_ticks"] = 0
        self.dodge_side = self.safe_side(obs, x)
        # Close melee actors are exactly where a stationary fire action is
        # lethal.  Keep ATTACK held together with a translating strafe (or a
        # backward diagonal fallback); this preserves the retreat lane while
        # still applying damage.  The same moving fire policy is useful for a
        # hitscanner that has already reached the near band.
        moving_fire = ("fire_strafe_left" if self.dodge_side > 0
                       else "fire_strafe_right")
        if x["melee"] or x["d"] < 125 or x["major"] or blocked:
            action = self.legal(
                moving_fire,
                "backward_left" if self.dodge_side > 0 else "backward_right",
                "fire")
        else:
            action = self.legal(moving_fire, "fire")
        self.fire_action = action
        self.fire_hold = t + 8
        return action

    def contact(self, obs, s):
        t = obs["t"]
        err = wrap(s["heading"] - self.effect.predicted_angle(obs))
        if s["phase"] == "settle":
            if abs(err) > 6:
                s["phase_start"] = t
                return self.legal(self.effect.turn(err))
            if t - s["phase_start"] < self.effect.lag + 5:
                return self.noop
            s["phase"] = "pulse"
        if s["phase"] == "pulse":
            if t < self.next_use:
                return self.noop
            self.next_use = t + 12
            s["phase"] = "wait"
            s["phase_start"] = t
            s["wait_origin"] = list(obs["p"])
            self.ctx.log("interaction_pulse", serial=s["serial"], face=str(s["face"]),
                         settled_heading=round(obs["a"], 1), lag=self.effect.lag)
            return self.legal("use")
        if s["phase"] == "wait":
            # Keep a forward probe held while the one-tic USE pulse takes
            # effect.  A stationary noop was safe from collisions but allowed
            # monsters to own the whole 18--28 tic opening interval.
            if distance(s.get("wait_origin", obs["p"]), obs["p"]) > 70:
                heading = s["heading"]
                self.contacts[s["face"]] = (t, "passage_after_use")
                self.contact_passes += 1
                self.finish(obs, "passage_after_use", "displacement during opening wait")
                self.start(obs, "transit", heading=heading, reason="interaction_crossing")
                return self.legal("forward")
            wait = 8 if self.policy["interaction"] == "patient" else 5
            if t - s["phase_start"] < wait:
                return self.legal("forward", "strafe_left", "strafe_right", self.noop)
            s["phase"] = "cross"
            s["phase_start"] = t
            s["cross_origin"] = list(obs["p"])
        if s["phase"] == "cross":
            moved = distance(s["cross_origin"], obs["p"])
            if moved > 70:
                heading = s["heading"]
                self.contacts[s["face"]] = (t, "passage_after_use")
                self.contact_passes += 1
                self.finish(obs, "passage_after_use", "displacement, not proof of a door or exit")
                self.start(obs, "transit", heading=heading, reason="interaction_crossing")
                return self.legal("forward")
            if t - s["phase_start"] > 18:
                heading = s["heading"]
                self.contacts[s["face"]] = (t, "blocked_after_settled_use")
                self.failed_headings[(cell(obs["p"]), octant(heading))] = t + 350
                self.finish(obs, "blocked_after_settled_use")
                self.start(obs, "recover", heading=self.departure(obs, exclude=heading), reason="tested_barrier")
                return self.recover(obs, self.skill)
            if abs(err) > 9:
                return self.legal(self.effect.turn(err))
            return self.legal("forward")
        return self.noop

    def recover(self, obs, s):
        t = obs["t"]
        if s["phase"] == "backstep":
            if t - s["phase_start"] < 8 + self.effect.lag:
                return self.legal("backward")
            s["phase"] = "align"
            s["phase_start"] = t
        err = wrap(s["heading"] - self.effect.predicted_angle(obs))
        if s["phase"] == "align":
            if abs(err) > 7 and t - s["phase_start"] < 65:
                return self.legal(self.effect.turn(err))
            s["phase"] = "translate"
            s["phase_start"] = t
        if t - s["phase_start"] < 18 + self.effect.lag:
            return self.legal("forward")
        self.finish(obs, "recovery_displaced" if distance(s["origin"], obs["p"]) > 40 else "recovery_no_displacement")
        return None

    def travel(self, obs, s):
        t = obs["t"]
        if s["kind"] == "pickup":
            x = obs["by_id"].get(s["id"])
            if x:
                s["q"] = list(x["p"])
                s["seen"] = t
                if x["dz"] < -34 or x["dz"] > 50:
                    self.cooldowns["pickup:" + s["id"]] = t + 220
                    self.finish(obs, "wrong_height", "visible is not traversable")
                    return None
                if x["d"] < s["best"] - 8:
                    s["best"] = x["d"]
                    s["progress"] = t
                if x["d"] < 43 and x["interact"]:
                    heading = bearing(obs["p"], x["p"])
                    self.finish(obs, "interaction_contact")
                    self.start(obs, "contact", heading=heading, reason="rendered_interaction")
                    return self.contact(obs, self.skill)
            elif t - s["seen"] > 14:
                self.finish(obs, "pickup_no_longer_rendered", "disappearance is not inventory confirmation")
                return None
            if t - s["start"] > 135 or t - s["progress"] > 58:
                self.cooldowns["pickup:" + s["id"]] = t + 210
                self.finish(obs, "approach_no_progress")
                return None
            if distance(obs["p"], s["q"]) < 27:
                # Walk through the pickup instead of steering back and forth
                # over its centre before the pending input reaches the engine.
                if s["phase"] != "overlap":
                    s["phase"] = "overlap"
                    s["phase_start"] = t
                if t - s["phase_start"] < 7:
                    return self.legal("forward")
                self.cooldowns["pickup:" + s["id"]] = t + 70
                self.finish(obs, "pickup_overlap")
                return None
            error = self.effect.error_to(obs, s["q"])
        else:
            error = wrap(s["heading"] - self.effect.predicted_angle(obs))
            if distance(s["origin"], obs["p"]) > 500 or t - s["start"] > 170:
                self.finish(obs, "traverse_completed")
                return None
        if abs(error) > 8:
            if t - s["start"] > 95 and distance(s["origin"], obs["p"]) < 12:
                self.finish(obs, "alignment_timeout")
                return None
            return self.legal(self.effect.turn(error))
        if self.blocked(obs) and t - s["start"] > 14:
            heading = s["heading"] if s["heading"] is not None else bearing(obs["p"], s["q"])
            face = (cell(obs["p"], 48), octant(heading), self.key_epoch)
            prior = self.contacts.get(face)
            self.finish(obs, "measured_contact")
            # A fresh blocked face is worth one USE test only while the player
            # has a usable safety margin.  At low health or immediately after
            # damage, retreat first; an explicitly rendered interaction still
            # reaches contact through the pickup skill above.
            unsafe_probe = obs["hp"] < 45 or t - self.last_damage < 35
            if (prior is None or t - prior[0] > 700) and not unsafe_probe:
                self.start(obs, "contact", heading=heading, reason="new_blocked_face")
                return self.contact(obs, self.skill)
            self.failed_headings[(cell(obs["p"]), octant(heading))] = t + (210 if unsafe_probe else 350)
            self.start(obs, "recover", heading=self.departure(obs, exclude=heading),
                       reason="unsafe_contact" if unsafe_probe else "known_tested_face")
            return self.recover(obs, self.skill)
        return self.legal("forward")

    def action(self, obs):
        emergency = self.missile_dodge(obs)
        if emergency:
            return emergency
        urgent = self.urgent(obs)
        s = self.skill
        # A contact probe is a deliberate commitment. Damage during its settle,
        # wait, or crossing phase is evidence that the current interaction is
        # unsafe; abandon it immediately instead of spending the remaining wait
        # interval in the line of fire.
        if s and s["kind"] == "contact" and obs["t"] - self.last_damage < 20:
            if urgent and obs["wa"] > 0:
                self.start(obs, "fight", obj=urgent, reason="damage_interrupts_contact")
            else:
                self.finish(obs, "damage_interrupts_contact")
                self.start(obs, "recover", heading=self.departure(obs), reason="damage_interrupts_contact")
        s = self.skill
        if (urgent and obs["hp"] < 18 and urgent["d"] > 140
                and not urgent["major"]):
            # A low-health line-of-fire threat is not a reason to stand and
            # trade pistol shots. Preserve the life margin and re-enter only
            # after displacement or a newly close body blocker.
            if not s or s["kind"] not in {"recover", "contact"}:
                if s:
                    self.finish(obs, "critical_disengage")
                self.start(obs, "recover", heading=self.departure(obs),
                           reason="critical_health")
            s = self.skill
        if urgent and obs["wa"] > 0:
            # Keep the current rendered target unless a new contact threat is
            # substantially closer. This removes target-selection oscillation.
            old_target = obs["by_id"].get(s.get("id")) if s else None
            if not s or (s["kind"] != "fight" and (s["kind"] not in {"contact", "recover"} or urgent["d"] < 145 or urgent["hitscan"])):
                self.start(obs, "fight", obj=urgent, reason="threat_precondition")
            elif s["kind"] == "fight" and old_target and urgent["id"] != old_target["id"] and urgent["d"] < 80 and old_target["d"] > 180:
                self.start(obs, "fight", obj=urgent, reason="contact_blocker_preempts")
        for _ in range(3):
            if not self.skill:
                self.choose_skill(obs)
            s = self.skill
            if not s:
                return self.noop
            if s["kind"] == "fight":
                result = self.fight(obs, s)
            elif s["kind"] == "contact":
                result = self.contact(obs, s)
            elif s["kind"] == "recover":
                result = self.recover(obs, s)
            else:
                result = self.travel(obs, s)
            if result is not None:
                return result
        return self.legal("forward")

    def describe_object(self, x):
        return {"id": x["id"], "name": x["name"], "category": x["cat"],
                "range": "contact" if x["d"] < 90 else "near" if x["d"] < 240 else "medium" if x["d"] < 550 else "far",
                "direction": direction(x["err"]),
                "height": "same_floor" if abs(x["dz"]) < 30 else "above" if x["dz"] > 0 else "below",
                "hitscan": x["hitscan"], "body_blocker": x["melee"] and x["d"] < 140,
                "major": x["major"]}

    def model_state(self, obs, choices):
        s = self.skill
        facts = {"blocked_after_sustained_motion": self.blocked(obs),
                 "damaged_recently": obs["t"] - self.last_damage < 45,
                 "new_ground_recently": obs["t"] - self.last_novel < 70,
                 "usable_ammo": obs["wa"] > 0,
                 "better_gun_owned": any(obs["weapons"].get(str(i), False) for i in (3, 4, 6)),
                 "critical_health": obs["hp"] < 28}
        candidates = []
        for key, kind, x in choices:
            row = self.describe_object(x)
            row.update({"option": key, "skill": kind,
                        "direct_approach_permitted": -30 <= x["dz"] <= 44,
                        "needed_capability": kind == "pickup" and x["cat"] in {"key", "weapon"}})
            candidates.append(row)
        return {"task": "Finish the Doom map by reaching and activating its exit alive.",
                "rules": "Only rendered objects are listed. Visibility is NOT evidence of a walkable route. Static exits/doors may have no actor label. Code owns exact geometry arithmetic, input lag, ammo, aiming, contact tests, and emergency action.",
                "recognizer_focus": dict(self.policy), "facts": facts,
                "skill": {"kind": s["kind"], "phase": s["phase"], "target": s.get("name"),
                          "target_currently_rendered": s.get("id") in obs["by_id"]} if s else None,
                "candidates": candidates,
                "last_outcome": self.outcomes[-1] if self.outcomes else None,
                "obstruction_evidence": {"close_monster": any(x["d"] < 105 for x in obs["threats"]),
                                          "height_mismatch_target": bool(s and s.get("id") in obs["by_id"] and abs(obs["by_id"][s["id"]]["dz"]) > 44),
                                          "settled_USE_failed": bool(self.outcomes and self.outcomes[-1]["result"] == "blocked_after_settled_use")}}

    def launch_jev(self, obs):
        if self.jev_future is not None or obs["t"] < self.next_jev or getattr(self.ctx, "system1", True) is None:
            return
        remaining = self.ctx.budget.remaining()
        if num(remaining.get("system1_calls")) < 4 or num(remaining.get("wall_s")) < 4:
            return
        choices = self.affordances(obs)
        # Empty space needs odometry, not a model asked to guess a door.
        if not choices and not self.outcomes:
            self.next_jev = obs["t"] + 35
            return
        state = self.model_state(obs, choices)
        criteria = {key: {"skill": kind, "object": self.describe_object(x)} for key, kind, x in choices}
        criteria["keep_skill"] = "Keep the current measured skill; no supplied object justifies changing its objective."
        order = list(criteria.items())
        if self.jev_calls % 2:
            order.reverse()
        focus = {
            "capability": "Prefer a missing workhorse gun or required key over optional bonuses; immediate threats still matter.",
            "keys": "Prioritize a supplied reachable key or exit interaction; do not dive toward a key on another height.",
            "exit": "Preserve traversal/contact-testing unless a rendered object changes the route or survival.",
            "recover": "Prefer substantial nearby health/armor or a defensible engagement; not distant one-point bonuses.",
        }[self.policy["focus"]]
        questions = {
            "objective": Choice("Choose the best supplied skill objective using candidates, facts, and recognizer_focus. " + focus + " Do not invent layout or compare raw numbers. keep_skill means no change.", dict(order)),
            "encounter": Choice("Classify the local encounter from facts and candidates, independently of objective. What must be dealt with before traversal?", {
                "clear_blocker": "An exposed hitscanner, close body blocker, or immediately dangerous monster needs aimed fire.",
                "preserve_travel": "No immediate monster justifies abandoning the current movement/contact skill.",
                "recover_capability": "Empty ammo or critical health makes an observed substantial supply more useful than optional combat.",
                "unknown": "Evidence does not establish a change."}),
            "obstruction": Choice("Diagnose ONLY the measured obstruction_evidence and last_outcome. A bare collision does not prove a door. This classifies an outcome, not hidden architecture.", {
                "enemy": "A close currently rendered monster plausibly blocks movement.",
                "height": "The supplied target has a directly reported height mismatch.",
                "tested_barrier": "Settled USE and a same-heading crossing attempt explicitly failed.",
                "unresolved_contact": "Sustained movement is blocked but a settled interaction test has not established the cause.",
                "unknown": "There is no current obstruction or its cause is not established."}),
        }
        meta = {"tic": obs["t"], "version": self.interface_version,
                "keys": set(criteria), "serial": self.skill["serial"] if self.skill else None,
                "outcome_serial": self.outcome_serial}
        self.jev_meta = meta
        self.jev_calls += 1
        self.next_jev = obs["t"] + (22 if obs["threats"] else 46)
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=3.0)
            )
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
        except Exception as exc:
            self.jev_future = None
            self.ctx.log("jev_launch_failure", error=type(exc).__name__)

    def poll_jev(self, obs):
        if self.jev_future is None or not self.jev_future.done():
            return
        f, meta = self.jev_future, self.jev_meta or {}
        self.jev_future = None
        try:
            answers = f.result()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, EpisodeOver, CancelledError):
            return
        except Exception as exc:
            self.ctx.log("jev_failure", error=type(exc).__name__)
            return
        age = obs["t"] - integer(meta.get("tic"), obs["t"])
        if age > 48 or meta.get("version") != self.interface_version:
            self.ctx.log("jev_discard", age_tics=age, reason="age_or_interface")
            return
        objective = probability_choice(answers, "objective", meta.get("keys", set()), .64)
        encounter = probability_choice(answers, "encounter", {"clear_blocker", "preserve_travel", "recover_capability", "unknown"}, .68)
        diagnosis = probability_choice(answers, "obstruction", {"enemy", "height", "tested_barrier", "unresolved_contact", "unknown"}, .68)
        if objective and objective != "keep_skill":
            still_valid = next(((kind, x) for key, kind, x in self.affordances(obs) if key == objective), None)
            if still_valid:
                kind, x = still_valid
                self.jev_hint = {"key": objective, "expires": obs["t"] + 45}
                if self.skill and self.skill["kind"] == "transit" and (kind == "pickup" or encounter == "clear_blocker"):
                    self.start(obs, kind, obj=x, reason="jev_current_affordance")
                    self.jev_hint = None
        if diagnosis:
            self.diagnosis = diagnosis
            # A diagnosis may invalidate a pickup, but never creates a door or
            # preempts an emergency with a speculative long recovery.
            s = self.skill
            if (diagnosis == "height" and s and s["kind"] == "pickup" and s["serial"] == meta.get("serial")):
                x = obs["by_id"].get(s["id"])
                if x and (x["dz"] < -30 or x["dz"] > 44):
                    self.cooldowns["pickup:" + x["id"]] = obs["t"] + 210
                    self.finish(obs, "jev_height_invalidated")
        self.ctx.log("jev_skill_bridge", objective=objective, encounter=encounter,
                     diagnosis=diagnosis, age_tics=age, changed_goal=bool(self.jev_hint))

    def launch_astra(self, obs):
        if self.astra_future is not None or self.astra_calls >= 3 or obs["t"] < self.next_astra:
            return
        remaining = self.ctx.budget.remaining()
        if num(remaining.get("system2_calls")) <= 0 or num(remaining.get("wall_s")) < 180:
            return
        # The first review authors Jev's recognition emphasis. Later reviews
        # use actual skill outcomes instead of repeating an old scene snapshot.
        payload = {"public_scene": self.model_state(obs, self.affordances(obs)),
                   "map_family": str(obs["raw"].get("iwad", "")),
                   "measured_effects": {"input_delay_tics": self.effect.lag,
                                        "yaw_degrees_per_tic": round(self.effect.rate, 3),
                                        "delay_samples": self.effect.samples},
                   "skill_ledger": list(self.outcomes)[-18:],
                   "coverage": {"visited_cells": len(self.visits),
                                "new_ground_recent": obs["t"] - self.last_novel < 175,
                                "settled_contact_tests": self.contact_tests,
                                "crossings_after_USE": self.contact_passes},
                   "current_interface": self.policy}
        guide = (
            "Local Jev contract: ctx.decide(state, questions, timeout_s=3) returns dict[id,answer]. "
            "Choice(instructions, criteria={key:description}) returns .value string and .probabilities; "
            "Noul returns yes probability .value. No prose, numeric reasoning or hidden context. "
            "Each request carries its own rules, derived semantic facts and explicit candidate objects. "
            "Code, not Jev, does arithmetic, input delay prediction, validation and execution. "
            "Atomic independent questions can share one request. Unknown, low probability, age over "
            "48 tics or changed interface preserve the code skill. Jev chooses an observed objective, "
            "classifies encounter pressure and diagnoses already measured obstruction evidence. "
            "Astra can revise the following enumerated recognition focus, not invent questions/buttons/geometry."
        )
        prompt = (
            "You are the strategic interface author for a real-time Doom agent. The world advances "
            "while you think, so revise DURABLE recognition priorities from measured skill outcomes, "
            "not a per-frame action. Exit success dominates optional kills. The code now predicts queued "
            "yaw, locks a target ID, and tests obstructions with settle -> USE -> wait -> same-heading "
            "crossing. Static doors and exit switches have not been supplied as labels, so never depend "
            "on hallucinating one. Additional frontier bookkeeping alone and blind wall following have "
            "not established completion. Choose which existing affordances Jev should emphasize. "
            "Return ONLY a small JSON object with fields: focus=capability|keys|exit|recover, "
            "encounter=blockers|secure, travel=persistent|radial, interaction=patient|quick, "
            "reason=one short explanation citing the ledger. Do not output code or a memorized map route. "
            "Choose secure only if optional close threats repeatedly prevent skill completion; patient "
            "keeps a longer settling/opening interval. Never claim that a crossing proves a door or exit.\n" +
            guide + "\nOBSERVED_EVIDENCE=" + json.dumps(payload, separators=(",", ":"), allow_nan=False)
        )
        session = self.astra_session
        self.astra_calls += 1
        self.next_astra = obs["t"] + 4200
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, session_id=session, timeout_s=240.0)
            )
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("astra_launch_failure", error=type(exc).__name__)
        self.ctx.log("astra_review_requested", call=self.astra_calls, outcome_serial=self.outcome_serial)

    def poll_astra(self, obs):
        if self.astra_future is None or not self.astra_future.done():
            return
        f = self.astra_future
        self.astra_future = None
        try:
            result = f.result()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, EpisodeOver, CancelledError):
            return
        except Exception as exc:
            self.ctx.log("astra_failure", error=type(exc).__name__)
            return
        proposal = parse_model_json(getattr(result, "text", ""))
        valid = False
        for key, allowed in {"focus": {"capability", "keys", "exit", "recover"},
                             "encounter": {"blockers", "secure"},
                             "travel": {"persistent", "radial"},
                             "interaction": {"patient", "quick"}}.items():
            value = proposal.get(key)
            if isinstance(value, str) and value in allowed:
                self.policy[key] = value
                valid = True
        if valid:
            self.interface_version += 1
            self.jev_hint = None
            self.astra_session = getattr(result, "session_id", self.astra_session)
            self.ctx.log("astra_recognizer_revision", version=self.interface_version,
                         policy=dict(self.policy), reason=str(proposal.get("reason", ""))[:240])
        else:
            self.ctx.log("astra_invalid", fallback="retain_skill_interface")

    def run_episode(self, game):
        ctx = self.ctx
        spec = getattr(game, "action_spec", None)
        self.allowed = set(getattr(spec, "names", ()) or ())
        if not self.allowed:
            return
        self.noop = getattr(spec, "noop", "noop")
        if self.noop not in self.allowed:
            self.noop = sorted(self.allowed)[0]
        self.started = num(ctx.clock.now())
        try:
            while not ctx.done and not bool(getattr(game, "done", False)):
                now = max(0, num(ctx.clock.now()) - self.started)
                obs = normalize(ctx.observe(), integer(now * 35))
                if obs["done"] or ctx.done or bool(getattr(game, "done", False)):
                    break
                if obs["t"] <= self.last_tic:
                    ctx.wait_for_tick()
                    continue
                self.last_tic = obs["t"]
                self.update(obs)
                self.poll_jev(obs)
                self.poll_astra(obs)
                action = self.legal(self.action(obs))
                ctx.act(action)
                self.effect.post(obs["t"], action)
                self.last_action = action
                # Post first, then launch background judgments. Neither model
                # is the actuator clock and a slow answer does not hold a turn.
                self.launch_jev(obs)
                self.launch_astra(obs)
                if obs["t"] >= self.next_summary:
                    self.next_summary = obs["t"] + 175
                    self.ctx.log("effect_skill_status", tic=obs["t"], hp=obs["hp"],
                                 skill=self.skill["kind"] if self.skill else None,
                                 phase=self.skill["phase"] if self.skill else None,
                                 lag=self.effect.lag, yaw_rate=round(self.effect.rate, 3),
                                 cells=len(self.visits), contact_tests=self.contact_tests,
                                 crossings=self.contact_passes, outcomes=self.outcome_serial,
                                 diagnosis=self.diagnosis)
                ctx.wait_for_tick()
        except (EpisodeOver, BudgetExceeded):
            return
        finally:
            for future in (self.jev_future, self.astra_future):
                if future is not None and not future.done():
                    future.cancel()
            ctx.memory["effect_skill_summary"] = {
                "cells": len(self.visits), "outcomes": list(self.outcomes)[-12:],
                "lag": self.effect.lag, "yaw_rate": self.effect.rate,
                "contact_tests": self.contact_tests, "crossings": self.contact_passes,
                "jev_calls": self.jev_calls, "astra_calls": self.astra_calls,
            }


def build(ctx):
    return SkillController(ctx)
