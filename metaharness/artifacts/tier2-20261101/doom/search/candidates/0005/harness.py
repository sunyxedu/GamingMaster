"""Candidate 0005: guarded semantic options with an odometric executor.

Only public observations are used. No IWAD, map geometry, seed routes, native
engine, filesystem, or direct model clients are accessed by the controller.
"""
from __future__ import annotations

import collections
import json
import math

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def decode(value):
    if isinstance(value, dict):
        return value
    try:
        value = json.loads(value)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bearing(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def point(obs):
    p = obs.get("player") or {}
    return (number(p.get("x")), number(p.get("y")), number(p.get("z")))


def cell(p):
    return (math.floor(p[0] / 80), math.floor(p[1] / 80), math.floor(p[2] / 64))


def octant(angle):
    return int(math.floor((angle % 360 + 22.5) / 45.0)) % 8


def choice(answers, key, threshold=0.56):
    a = answers.get(key) if isinstance(answers, dict) else None
    v = getattr(a, "value", None)
    p = getattr(a, "probabilities", None)
    if not isinstance(v, str) or not isinstance(p, dict):
        return None
    if number(p.get(v)) < threshold:
        return None
    confidence = getattr(a, "confidence", None)
    if confidence is not None and number(confidence) < 0.15:
        return None
    return v


def yes(answers, key, threshold=0.76):
    a = answers.get(key) if isinstance(answers, dict) else None
    return number(getattr(a, "value", 0)) >= threshold


MOVEMENT = {
    "forward": 0, "fire_forward": 0, "backward": 180,
    "strafe_left": 90, "fire_strafe_left": 90,
    "strafe_right": -90, "fire_strafe_right": -90,
    "forward_left": 45, "forward_right": -45,
    "backward_left": 135, "backward_right": -135,
}
HITSCANNERS = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS", "SpiderMastermind"}
URGENT = {"Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind"}
WEAPON_SLOT = {"Pistol": 2, "Shotgun": 3, "SuperShotgun": 3,
               "Chaingun": 4, "RocketLauncher": 5, "PlasmaRifle": 6,
               "BFG9000": 7, "Chainsaw": 1}


def weapon_owned(weapons, action_slot):
    """Map classic key slots to the public zero-based WEAPON variables."""
    if action_slot == 1:
        return bool(weapons.get("0")) or bool(weapons.get("7"))
    return bool(weapons.get(str(action_slot - 1)))


def weapon_ammo(obs, action_slot):
    selected = int(number(obs.get("weapon"), -1))
    if selected + 1 == action_slot:
        return number(obs.get("weapon_ammo"))
    ammo = obs.get("ammo") or {}
    # The first spelling is the usual internal weapon-aligned projection;
    # the fallback keeps this conservative when a scenario exposes aliases.
    return max(number(ammo.get(str(action_slot - 1))), number(ammo.get(str(action_slot))))


class ObservedWorld:
    """Sparse collision/traversal memory, not an inferred omniscient map."""
    def __init__(self):
        self.nodes = {}
        self.edges = collections.defaultdict(set)
        self.tried = collections.Counter()
        self.blocked = {}
        self.landmarks = {}
        self.visits = collections.Counter()
        self.previous = None
        self.previous_action = "noop"
        self.previous_tic = 0
        self.last_cell = None
        self.motion = None
        self.turn_rate = 5.625
        self.turn_left_sign = 1
        self.damage = collections.deque()
        self.last_damage = -1000.0
        self.last_progress = 0.0
        self.collision_count = 0
        self.trail = collections.deque(maxlen=400)
        self.history = collections.deque(maxlen=30)
        self.last_history = -10.0
        self.last_counts = (0, 0, 0)
        self.last_monsters = {}
        self.visible = []
        self.threats = []
        self.projectiles = []
        self.collision = None

    def update(self, obs, now):
        p = point(obs)
        angle = number((obs.get("player") or {}).get("angle"))
        tic = int(number(obs.get("tic"), now * 35))
        c = cell(p)
        self.collision = None
        if c != self.last_cell:
            self.visits[c] += 1
            if c not in self.nodes:
                self.last_progress = now
            self.nodes[c] = p
            if self.last_cell is not None and self.previous is not None:
                old = point(self.previous)
                if distance(old, p) < 100 and abs(old[2] - p[2]) < 32:
                    self.edges[c].add(self.last_cell)
                    self.edges[self.last_cell].add(c)
            self.last_cell = c
        if not self.trail or distance(self.trail[-1][1], p) > 48:
            self.trail.append((now, p, number(obs.get("health"), 100)))
        if self.previous is not None:
            oldp = point(self.previous)
            oldangle = number((self.previous.get("player") or {}).get("angle"))
            dt = tic - self.previous_tic
            loss = max(0, number(self.previous.get("health")) - number(obs.get("health")))
            if loss:
                self.damage.append((now, loss))
                self.last_damage = now
            if self.previous_action in ("turn_left", "turn_right") and 0 < dt <= 8:
                delta = wrap(angle - oldangle)
                if 0.2 < abs(delta) / dt < 25:
                    self.turn_rate = 0.7 * self.turn_rate + 0.3 * abs(delta) / dt
                    self.turn_left_sign = (1 if delta > 0 else -1) * (1 if self.previous_action == "turn_left" else -1)
            if self.previous_action in MOVEMENT:
                heading = (oldangle + MOVEMENT[self.previous_action]) % 360
                if self.motion is None or abs(wrap(heading - self.motion[2])) > 40:
                    self.motion = (oldp, self.previous_tic, heading)
                origin, start, direction = self.motion
                if distance(origin, p) >= 10:
                    self.motion = (p, tic, heading)
                elif tic - start >= 10:
                    k = (c, octant(direction))
                    self.blocked[k] = now + 35
                    self.tried[k] += 2
                    self.collision_count += 1
                    self.collision = {"cell": c, "heading": direction, "action": self.previous_action}
                    self.motion = (p, tic, heading)
            else:
                self.motion = None
        while self.damage and now - self.damage[0][0] > 6:
            self.damage.popleft()
        counts = tuple(int(number(obs.get(k))) for k in ("kills", "items", "secrets"))
        if counts != self.last_counts:
            self.last_progress = now
        self.last_counts = counts
        self.visible, self.threats, self.projectiles = [], [], []
        raw = obs.get("visible") or []
        for v in raw if isinstance(raw, list) else []:
            if not isinstance(v, dict):
                continue
            w = v.get("world")
            if not isinstance(w, (list, tuple)) or len(w) < 3:
                continue
            q = tuple(number(x) for x in w[:3])
            name, category = str(v.get("name", "")), str(v.get("category", ""))
            ident = str(v.get("id", name))
            d = distance(p, q)
            rel = wrap(bearing(p, q) - angle)
            e = {"id": ident, "name": name, "category": category, "pos": q,
                 "distance": d, "relative": rel, "height": q[2] - p[2], "seen": now}
            self.visible.append(e)
            if category == "Monster" and not name.startswith("Dead"):
                old = self.last_monsters.get(ident)
                moving = old is not None and distance(old["pos"], q) > 1
                e["active"] = moving or d < 180 or now - self.last_damage < 3
                e["hitscan"] = name in HITSCANNERS
                e["priority"] = (260 if name in URGENT else 0) + (170 if e["hitscan"] else 0) + (200 if d < 130 else 0) - d * 0.2
                self.threats.append(e)
                self.last_monsters[ident] = e
            elif category == "Projectile":
                self.projectiles.append(e)
            elif (category in ("Weapon", "Health", "Armor", "Ammo", "Key", "Powerup", "Exit", "Switch", "Door")
                  or any(token in name.lower() for token in ("exit", "switch", "door"))):
                old = self.landmarks.get(ident, {})
                e.update({"defer": old.get("defer", 0), "failures": old.get("failures", 0), "gone": False})
                self.landmarks[ident] = e
        visible_ids = {e["id"] for e in self.visible}
        for e in self.landmarks.values():
            if e["id"] not in visible_ids and distance(p, e["pos"]) < 30 and abs(e["pos"][2] - p[2]) < 28:
                e["gone"] = True  # inferred pickup, not a claim about hidden key inventory
        self.threats.sort(key=lambda e: e["priority"], reverse=True)
        if now - self.last_history >= 2:
            self.history.append({"t": round(now, 1), "health": obs.get("health"),
                                 "position": [round(x) for x in p], "kills": counts[0],
                                 "cells": len(self.nodes), "collisions": self.collision_count,
                                 "action": self.previous_action})
            self.last_history = now
        self.previous, self.previous_tic = obs, tic

    def resource_candidates(self, obs, now, policy):
        p = point(obs)
        health, armor = number(obs.get("health"), 100), number(obs.get("armor"))
        weapons, ammo = obs.get("weapons") or {}, obs.get("ammo") or {}
        result = []
        for e in self.landmarks.values():
            if e.get("gone") or e.get("defer", 0) > now or now - e["seen"] > 150:
                continue
            cat, name = e["category"], e["name"]
            d, z = distance(p, e["pos"]), e["pos"][2] - p[2]
            utility, why = 0, ""
            if cat == "Key":
                utility, why = 110, "required-route key candidate"
            elif cat == "Weapon":
                slot = WEAPON_SLOT.get(name)
                owned = weapon_owned(weapons, slot) if slot else False
                if name == "SuperShotgun":
                    owned = bool(weapons.get("8")) or bool(weapons.get("9"))
                if not owned:
                    utility, why = (160 if slot in (3, 4, 6) else 70), "new combat capability"
                elif slot and weapon_ammo(obs, slot) < 12:
                    utility, why = 50, "weapon also supplies scarce ammunition"
            elif cat in ("Exit", "Switch", "Door") or any(token in name.lower() for token in ("exit", "switch", "door")):
                utility, why = (1000 if cat == "Exit" or "exit" in name.lower() else 210), "visible exit or interaction landmark"
            elif cat == "Health" and health < 95:
                utility, why = (220 if health < 40 else 125), "needed healing"
            elif cat == "Armor" and armor < (180 if name == "BlueArmor" else 90):
                utility, why = (105 if "Bonus" not in name else 20), "protection"
            elif cat == "Ammo":
                slot = 3 if "Shell" in name else 6 if "Cell" in name else 5 if "Rocket" in name else 2
                if weapon_owned(weapons, slot) and weapon_ammo(obs, slot) < (20 if slot == 3 else 70):
                    utility, why = 55, "usable ammunition"
            elif cat == "Powerup":
                utility, why = 90, "powerup"
            if not utility or d > 1600 or z > 48:
                continue
            if policy["priority"] == "exit_first" and cat == "Key":
                utility += 70
            if policy["priority"] == "recover" and cat in ("Health", "Armor"):
                utility += 60
            score = utility - d * 0.14 - max(0, abs(z) - 20) * 2 - e.get("failures", 0) * 35
            candidate = dict(e, distance=d, height=z, score=score, benefit=why)
            result.append(candidate)
        return sorted(result, key=lambda e: e["score"], reverse=True)[:5]

    def frontier(self, obs, now, handedness):
        p = point(obs)
        c = cell(p)
        angle = number((obs.get("player") or {}).get("angle"))
        options = []
        for i in range(8):
            theta = i * 45
            dest = (p[0] + 100 * math.cos(math.radians(theta)), p[1] + 100 * math.sin(math.radians(theta)), p[2])
            if self.blocked.get((c, i), 0) > now:
                continue
            dc = cell(dest)
            novelty = 7 * self.visits.get(dc, 0) + 5 * self.tried[(c, i)]
            turncost = abs(wrap(theta - angle)) / 90
            bias = 0.1 if wrap(theta - angle) * handedness < 0 else 0
            options.append((novelty + turncost + bias, i, dest))
        if options:
            _, i, dest = min(options)
            self.tried[(c, i)] += 1
            return dest
        # A predecessor is a measured location, unlike an invented passage.
        neighbors = list(self.edges.get(c, ()))
        if neighbors:
            n = min(neighbors, key=lambda n: self.visits[n])
            return self.nodes[n]
        return (p[0] - 72 * math.cos(math.radians(angle)), p[1] - 72 * math.sin(math.radians(angle)), p[2])


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.world = ObservedWorld()
        self.policy = {"priority": "resource_first", "profile": "ordinary", "handedness": "left"}
        self.version = 0
        self.intent = None
        self.intent_until = -1
        self.votes = {}
        self.goal = None
        self.recovery = None
        self.interaction = None
        self.use_cooldown = {}
        self.last_weapon = -100.0
        self.last_shot = -100.0
        self.last_target = None
        self.combat_started = -100.0
        self.last_dodge_change = -100.0
        self.dodge = "strafe_left"
        self.future1 = None
        self.future2 = None
        self.request1 = None
        self.next1 = 0.0
        self.next2 = 20.0
        self.calls2 = 0
        self.session = None
        self.started = None
        self.last_tic = -1
        self.log_until = 0
        self.allowed = set()

    def resources(self, obs, now):
        return self.world.resource_candidates(obs, now, self.policy)

    def summary(self, obs, now):
        def describe(e):
            d, a, z = e["distance"], e.get("relative", 0), e.get("height", 0)
            return {"id": e["id"], "name": e["name"], "category": e["category"],
                    "range": "touching" if d < 48 else "near" if d < 180 else "medium" if d < 450 else "far",
                    "side": "ahead" if abs(a) < 12 else "left" if a > 0 else "right",
                    "height": "above_reach" if z > 28 else "below" if z < -28 else "same_level",
                    "active_evidence": e.get("active", False), "hitscanner": e.get("hitscan", False),
                    "benefit": e.get("benefit", "")}
        w = self.world
        return {"rules": "Reach and activate the exit. Only rendered actors are visible; doors, walls, and walkable floor are NOT in this observation. A visible pickup does not prove a straight path. No jump. Code aims and detects collisions.",
                "policy": dict(self.policy), "health": "critical" if number(obs.get("health")) < 30 else "wounded" if number(obs.get("health")) < 65 else "healthy",
                "damage_recent": now - w.last_damage < 2.5,
                "ammo": "empty" if number(obs.get("weapon_ammo")) <= 0 else "low" if number(obs.get("weapon_ammo")) < 8 else "available",
                "threats": [describe(e) for e in w.threats[:3]],
                "resources": [describe(e) for e in self.resources(obs, now)[:4]],
                "motion": {"blocked": bool(w.collision) or self.interaction is not None,
                           "goal_kind": self.goal.get("kind") if self.goal else "none",
                           "progress_stalled": now - w.last_progress > 10,
                           "known_retreat": len(w.trail) > 2},
                "proposed_fallback": "shoot/evade immediate active threats; otherwise approach useful same-floor resources, then short unexplored-heading probes; brake on collision"}

    def questions(self, state):
        criteria = {
            "collect": "A listed useful resource can improve health, equipment, or key progress; approach it carefully, never assume an unobstructed path.",
            "engage": "A visible active monster or hitscanner must be dealt with before advancing; aim from standoff rather than rush it.",
            "explore": "No more urgent listed threat or useful resource; cautiously probe an untried direction toward the exit.",
            "recover": "Recent damage, critical health, or no usable ammo warrants retreat along observed travel before another commitment.",
            "wait": "The evidence is genuinely insufficient; a brief quiet brake is safer than committing. This is not a permanent strategy.",
        }
        if self.version % 2:
            criteria = dict(reversed(list(criteria.items())))
        focus = {"ordinary": "Prefer useful resources, route progress, and clearing blockers.",
                 "collision": "Repeated forward motion did not work. Do not recommend that failed path; prefer a different heading or observed backtrack.",
                 "survival": "Avoid optional exposure. Preserve a known escape lane; urgently prioritize health and active hitscanners."}[self.policy["profile"]]
        q = {"mode": Choice("Choose the next local intention using threats, resources, motion and health. " + focus, criteria),
             "attack": Noul("Does the visible threat evidence justify initiating or continuing combat NOW rather than quietly acquiring resources or probing? Say yes for an active dangerous blocker; no for a harmless distant inactive monster or no visible threat."),
             "continue": Noul("Is continuing the CURRENT motion goal supported? Say no if motion.blocked is true, health is critical under recent damage, or geometry is unknown and the goal is stalled. Visible objects alone do not establish a traversable path.")}
        if state["threats"]:
            c = {e["id"]: {"what": e["name"], "facts": e} for e in state["threats"]}
            c["none"] = "No listed monster warrants immediate targeting."
            q["target"] = Choice("Which listed visible monster is the most urgent immediate threat? Use active evidence, hitscan, proximity, and blocking pressure, not hypothetical enemies.", c)
        if state["resources"]:
            c = {e["id"]: {"what": e["name"], "facts": e} for e in state["resources"]}
            c["none"] = "None of these resources is worth approaching now."
            q["pickup"] = Choice("Which listed resource best supports the current policy and survival? Favor new useful weapons and needed healing, then keys; above_reach is not directly collectible. Ignore optional ammo at full supply.", c)
        if state["motion"]["blocked"]:
            q["obstruction"] = Choice("What should follow a collision? There is no door label or guaranteed geometry. A use pulse may test a door but cannot confirm one. Repeated failed probes require a different route.", {
                "use_once": "A first collision might be a door or lift: try one USE pulse, wait briefly, then test motion once.",
                "change_heading": "The direction has repeatedly failed: turn to a different untried heading.",
                "backtrack": "Damage or resource shortage makes a known return path safer.",
                "unknown": "Insufficient evidence; let the bounded collision executor test it.",
            })
        return q

    def poll(self, obs, now):
        if self.future1 is not None and self.future1.done():
            f, meta = self.future1, self.request1
            self.future1 = None
            try:
                answers = f.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                self.next1 = now + 0.5
                self.ctx.log("jev_transient_failure", fallback="odometric_executor")
            else:
                age = now - meta["time"]
                if age <= 1.5 and meta["version"] == self.version:
                    mode = choice(answers, "mode")
                    old = self.intent
                    self.intent = mode
                    self.intent_until = now + 1.25
                    self.votes = {"target": choice(answers, "target"), "pickup": choice(answers, "pickup"),
                                  "attack": yes(answers, "attack"), "continue": yes(answers, "continue", 0.65),
                                  "obstruction": choice(answers, "obstruction")}
                    if mode != old:
                        self.ctx.log("jev_intent", before=old, after=mode, age_s=round(age, 3),
                                     target=self.votes["target"], pickup=self.votes["pickup"])
                    if mode == "collect" and self.goal and self.goal["kind"] == "probe":
                        if self.votes["pickup"] in self.world.landmarks:
                            self.goal = None
                else:
                    self.ctx.log("jev_stale_discard", age_s=round(age, 3))
        if self.future2 is not None and self.future2.done():
            f = self.future2
            self.future2 = None
            try:
                result = f.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                self.ctx.log("astra_transient_failure", fallback="existing_policy")
            else:
                text = str(getattr(result, "text", ""))
                start, end = text.find("{"), text.rfind("}")
                policy = decode(text[start:end + 1]) if start >= 0 and end >= start else {}
                accepted = {}
                for key, legal in (("priority", ("resource_first", "exit_first", "recover")),
                                   ("profile", ("ordinary", "collision", "survival")),
                                   ("handedness", ("left", "right"))):
                    if policy.get(key) in legal:
                        accepted[key] = policy[key]
                if accepted:
                    self.policy.update(accepted)
                    self.version += 1
                    self.intent_until = -1
                    self.goal = None
                    self.ctx.log("astra_interface_revision", version=self.version, policy=dict(self.policy))
                else:
                    self.ctx.log("astra_invalid_text", fallback="existing_policy")
                self.session = getattr(result, "session_id", self.session)

    def dispatch(self, obs, now):
        remaining = self.ctx.budget.remaining()
        if self.future1 is None and now >= self.next1 and remaining.get("system1_calls", 0) > 8 and remaining.get("wall_s", 0) > 2:
            state = self.summary(obs, now)
            questions = self.questions(state)
            self.request1 = {"time": now, "version": self.version}
            self.future1 = self.ctx.spawn(lambda: self.ctx.decide(state, questions, timeout_s=3.0))
            pacing = number(remaining.get("wall_s")) / max(1, number(remaining.get("system1_calls")) - 16)
            self.next1 = now + max(0.32 if self.world.threats else 0.75, pacing)
        stalled = now - self.world.last_progress > 14
        endangered = sum(x[1] for x in self.world.damage) > 25
        if self.future2 is None and now >= self.next2 and self.calls2 < 3 and (stalled or endangered or self.calls2 == 0) and remaining.get("system2_calls", 0) > 0 and remaining.get("wall_s", 0) > 100:
            state = self.summary(obs, now)
            evidence = list(self.world.history)[-12:]
            prompt = (
                "Revise a Doom agent's bounded policy and fast recognition interface. No map geometry, screenshots, or map-specific route is available. "
                "Infer only from these public observations; do not invent walls, keys, switches or an exit location. "
                "The controller already computes distances/bearings, aims, detects blocked movement, probes USE once per obstruction, and remembers traversed positions. "
                "It keeps acting during this slow request. Return ONLY a small JSON object with priority: resource_first|exit_first|recover, "
                "profile: ordinary|collision|survival, handedness: left|right. These select the next Jev question emphasis; do not output raw actions or coordinates. "
                "ordinary emphasizes useful resources and blockers; collision explicitly rejects failed straight paths; survival emphasizes active threats and a safe retreat. "
                "Choose collision when cells stop increasing despite movement; survival for falling health; exit_first once adequately equipped. "
                "Local Jev contract: ctx.decide(state, questions, timeout_s=3.0) returns a dict of typed answers. Choice(instructions, criteria={key:description}) "
                "returns .value as an option key with .probabilities/.confidence; Noul returns probability yes as .value. Independent questions share state but not answers. "
                "Jev sees only the compact state, cannot do reliable arithmetic, and does not inherit this conversation. Our code sends semantic distance/height buckets and legal candidates. "
                "Mode/target/pickup need top probability >=.56; optional combat also needs Noul >=.76; emergency safety overrides both. "
                "Your policy is long-lived but situational decisions expire in 1.25 seconds, and current legality is checked every tic. "
                "Current state=" + json.dumps(state, separators=(",", ":")) + "\nRecent measured history=" + json.dumps(evidence, separators=(",", ":"))
            )
            session = self.session
            self.future2 = self.ctx.spawn(lambda: self.ctx.think(prompt, session_id=session, timeout_s=180.0))
            self.calls2 += 1
            self.next2 = now + 100
            self.ctx.log("astra_requested", reason="stalled" if stalled else "damage" if endangered else "first_policy_review")

    def turn(self, difference):
        left = difference * self.world.turn_left_sign > 0
        return "turn_left" if left else "turn_right"

    def steer(self, obs, dest, precise=False):
        p = point(obs)
        angle = number((obs.get("player") or {}).get("angle"))
        diff = wrap(bearing(p, dest) - angle)
        deadzone = max(3.0, self.world.turn_rate * 0.65) if precise else 12
        if abs(diff) > deadzone:
            return self.turn(diff)
        return "forward"

    def pick_weapon(self, obs, now, threat):
        if now - self.last_weapon < 1.5:
            return None
        weapons, ammo = obs.get("weapons") or {}, obs.get("ammo") or {}
        current = int(number(obs.get("weapon"), 2))
        current_slot = current + 1
        health = number(obs.get("health"), 100)
        order = [6, 4, 3, 2] if health < 35 or (threat and threat["name"] in URGENT) else [4, 3, 6, 2]
        for slot in order:
            owned = weapon_owned(weapons, slot)
            count = weapon_ammo(obs, slot)
            if slot == 3:
                count = max(count, number(ammo.get("8")), number(ammo.get("9")))
            if owned and count > 0:
                if current_slot == slot:
                    return None
                a = "select_weapon_" + str(slot)
                if a in self.allowed:
                    self.last_weapon = now
                    return a
        return None

    def fight(self, obs, now, threat):
        w = self.world
        if self.last_target != threat["id"]:
            self.last_target = threat["id"]
            self.combat_started = now
        selected = self.pick_weapon(obs, now, threat)
        if selected:
            return selected
        if now - self.last_dodge_change > 1.2 or (w.collision and "strafe" in w.collision["action"]):
            self.dodge = "strafe_right" if self.dodge == "strafe_left" else "strafe_left"
            self.last_dodge_change = now
        diff = threat["relative"]
        close = threat["distance"] < 100
        if close and now - w.last_damage < 0.5:
            return "backward_left" if self.dodge == "strafe_left" else "backward_right"
        if abs(diff) > max(3.5, w.turn_rate * 0.7):
            return self.turn(diff)
        if number(obs.get("weapon_ammo")) <= 0:
            return "backward_left" if self.dodge == "strafe_left" else "backward_right"
        if int(number(obs.get("weapon"))) in (4, 6):
            # Rocket/BFG splash clearance cannot be certified without geometry.
            return self.dodge
        aim_window = max(1.4, math.degrees(math.atan2(13, max(20, threat["distance"]))))
        if abs(diff) > aim_window and threat["distance"] > 200:
            return "strafe_left" if diff > 0 else "strafe_right"
        self.last_shot = now
        if close or threat["active"] or now - w.last_damage < 2 or w.projectiles:
            return "fire_" + self.dodge
        return "fire"

    def fail_goal(self, now):
        if self.goal and self.goal.get("id") in self.world.landmarks:
            e = self.world.landmarks[self.goal["id"]]
            e["failures"] += 1
            e["defer"] = now + min(90, 20 * e["failures"])
        self.goal = None

    def collision_action(self, obs, now):
        w = self.world
        tic = int(number(obs.get("tic"), now * 35))
        if self.interaction:
            q = self.interaction
            if distance(point(obs), q["pos"]) > 36 and q["phase"] == "retry":
                w.blocked.pop(q["key"], None)
                self.interaction = None
                self.goal = None
                self.ctx.log("door_probe_progress", confirmed="movement_only")
                return None
            if tic < q["until"]:
                return "noop" if q["phase"] == "wait" else "forward" if q["phase"] == "retry" else "backward"
            if q["phase"] == "wait":
                q.update(phase="retry", until=tic + 12)
                return "forward"
            if q["phase"] == "retry":
                q.update(phase="backoff", until=tic + 7)
                self.fail_goal(now)
                return "backward"
            self.interaction = None
            self.goal = None
            w.motion = None
        if w.collision and w.collision["action"] in ("forward", "fire_forward"):
            key = (cell(point(obs)), octant(number((obs.get("player") or {}).get("angle"))))
            if self.use_cooldown.get(key, 0) < now and now - w.last_damage > 1:
                self.use_cooldown[key] = now + 25
                self.interaction = {"phase": "wait", "until": tic + 18, "key": key, "pos": point(obs)}
                self.ctx.log("bounded_use_probe", position=[round(x) for x in point(obs)])
                return "use"
            self.fail_goal(now)
        return None

    def navigation(self, obs, now):
        w, p = self.world, point(obs)
        if self.goal:
            g = self.goal
            d = distance(p, g["pos"])
            if d < 24 or now - g["start"] > 6:
                if d >= 24:
                    self.fail_goal(now)
                else:
                    self.goal = None
            elif d < g["best"] - 12:
                g["best"], g["progress"] = d, now
            elif now - g["progress"] > 2.5:
                self.fail_goal(now)
        if self.goal is None:
            candidates = self.resources(obs, now)
            best = candidates[0] if candidates else None
            desired = self.votes.get("pickup") if now <= self.intent_until else None
            if desired:
                best = next((e for e in candidates if e["id"] == desired), best)
            if best and best["score"] > 15:
                self.goal = {"kind": "pickup", "id": best["id"], "pos": best["pos"]}
            else:
                hand = 1 if self.policy["handedness"] == "left" else -1
                self.goal = {"kind": "probe", "pos": w.frontier(obs, now, hand)}
            self.goal.update(start=now, progress=now, best=distance(p, self.goal["pos"]))
        return self.steer(obs, self.goal["pos"])

    def retreat(self, obs, now):
        p = point(obs)
        if self.recovery is None:
            choices = [(t, q) for t, q, hp in self.world.trail if distance(p, q) > 90 and now - t < 30]
            if choices:
                _, q = choices[-1]
                self.recovery = {"pos": q, "until": now + 3}
        if self.recovery and now < self.recovery["until"]:
            if distance(p, self.recovery["pos"]) > 32:
                return self.steer(obs, self.recovery["pos"])
            self.recovery = None
            return "noop"
        self.recovery = None
        return "backward_left" if self.dodge == "strafe_left" else "backward_right"

    def visible_interaction(self, obs, now):
        p = point(obs)
        for e in self.world.landmarks.values():
            name = e.get("name", "").lower()
            category = e.get("category", "")
            if now - e.get("seen", -1000) > 0.35 or e.get("gone"):
                continue
            if category not in ("Exit", "Switch", "Door") and not any(token in name for token in ("exit", "switch", "door")):
                continue
            if distance(p, e["pos"]) > 72 or abs(e["pos"][2] - p[2]) > 48:
                continue
            key = ("visible", e["id"])
            if self.use_cooldown.get(key, 0) <= now:
                self.use_cooldown[key] = now + 2.0
                self.goal = None
                self.ctx.log("visible_interaction", name=e["name"], category=category)
                return "use"
        return None

    def action(self, obs, now):
        w = self.world
        health = number(obs.get("health"), 100)
        live = now <= self.intent_until
        mode = self.intent if live else None
        threats = w.threats
        target = threats[0] if threats else None
        if live and self.votes.get("target"):
            target = next((e for e in threats if e["id"] == self.votes["target"]), target)
        # An expiring semantic answer never owns aiming or motor duration.
        urgent = target is not None and (target["active"] or target["distance"] < 230 or now - w.last_damage < 2)
        authorize = target is not None and live and mode == "engage" and self.votes.get("attack")
        recently_fighting = target is not None and now - self.last_shot < 1.5
        if target and (urgent or authorize or recently_fighting):
            self.interaction = None
            if health < 22 and number(obs.get("weapon_ammo")) <= 0:
                return self.retreat(obs, now)
            return self.fight(obs, now, target)
        if now - w.last_damage < 1.0 and not target:
            self.goal = None
            # Blind damage is not evidence that forward is safe. Reacquire by a
            # brief turn; use a measured return waypoint if available.
            if len(w.trail) > 3:
                return self.retreat(obs, now)
            return "turn_left"
        if w.projectiles and min(e["distance"] for e in w.projectiles) < 180:
            return self.dodge
        interaction = self.visible_interaction(obs, now)
        if interaction:
            return interaction
        collision = self.collision_action(obs, now)
        if collision:
            return collision
        if mode == "recover" and now - w.last_damage < 3 and len(w.trail) > 2:
            return self.retreat(obs, now)
        if mode == "wait" and not target and self.future1 is not None and self.goal is None:
            return "noop"
        # The initial brake is safe and brief, not an entire Astra latency.
        if now < 0.6 and self.intent is None:
            return "noop"
        return self.navigation(obs, now)

    def run_episode(self, game):
        ctx = self.ctx
        self.allowed = set(game.action_spec.names)
        self.started = ctx.clock.now()
        ctx.act(game.action_spec.noop)
        while not ctx.done and not getattr(game, "done", False):
            obs = decode(ctx.observe())
            if ctx.done or getattr(game, "done", False):
                return
            if obs.get("terminal") or obs.get("dead"):
                return
            if not obs or not isinstance(obs.get("player"), dict):
                ctx.act(game.action_spec.noop)
                ctx.wait_for_tick()
                continue
            now = max(0, ctx.clock.now() - self.started)
            tic = int(number(obs.get("tic"), now * 35))
            if tic == self.last_tic:
                ctx.wait_for_tick()
                continue
            self.last_tic = tic
            self.world.update(obs, now)
            self.poll(obs, now)
            self.dispatch(obs, now)
            action = self.action(obs, now)
            if action not in self.allowed:
                action = game.action_spec.noop
            self.world.previous_action = action
            ctx.act(action)
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
