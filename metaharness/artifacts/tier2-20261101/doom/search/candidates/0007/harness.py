"""Candidate 0007: a checkpointed Doom explorer with semantic arbitration.

The controller deliberately separates three jobs.  A small deterministic motor
uses observed odometry to traverse bounded macro-actions and to recover from a
collision.  Jev classifies the current local situation (it never emits a raw
button sequence).  Astra is an infrequent policy reviewer whose answer only
changes a few search/combat preferences.  Thus a slow or malformed answer
cannot turn an otherwise moving player into a permanent turn-in-place loop.
"""
from __future__ import annotations

import collections
import json
import math

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


# These are action names, not native button indexes.  The adaptor validates
# every name against action_spec before it reaches the engine.
TRANSLATION = {
    "forward", "backward", "strafe_left", "strafe_right",
    "forward_left", "forward_right", "backward_left", "backward_right",
    "fire_forward", "fire_strafe_left", "fire_strafe_right",
}
TURNING = {"turn_left", "turn_right"}

# This taxonomy is only applied after the public label category says Monster.
# It is a threat prior, not a claim that an arbitrary name is hostile.
HITSCAN = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS",
           "SpiderMastermind"}
URGENT = {"Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind"}


def number(value, default=0.0):
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def integer(value, default=0):
    try:
        return int(number(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def wrap(angle):
    return (number(angle) + 180.0) % 360.0 - 180.0


def position(observation):
    player = observation.get("player") if isinstance(observation, dict) else None
    if not isinstance(player, dict):
        return None
    values = tuple(number(player.get(k), 0.0) for k in ("x", "y", "z"))
    return values if all(math.isfinite(x) for x in values) else None


def distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.sqrt(sum((number(x) - number(y)) ** 2 for x, y in zip(a, b)))


def xy_distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.hypot(number(a[0]) - number(b[0]), number(a[1]) - number(b[1]))


def cell_of(p):
    if p is None:
        return (0, 0, 0)
    return (math.floor(p[0] / 64.0), math.floor(p[1] / 64.0),
            math.floor(p[2] / 64.0))


def heading_bucket(angle):
    return int(math.floor((number(angle) % 360.0 + 22.5) / 45.0)) % 8


def bearing(a, b):
    if a is None or b is None:
        return 0.0
    return math.degrees(math.atan2(number(b[1]) - number(a[1]),
                                   number(b[0]) - number(a[0]))) % 360.0


def finite_list(values, length=3):
    if not isinstance(values, (list, tuple)) or len(values) < length:
        return None
    return [number(x) for x in values[:length]]


def extract_json(text):
    """Extract one object from an Astra answer without trusting its prose."""
    if not isinstance(text, str):
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        value = json.loads(text[start:end + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def choice_answer(answers, key, allowed, threshold=0.56):
    """Read a Jev Choice defensively, including arbitrary preflight answers."""
    if not isinstance(answers, dict):
        return None
    answer = answers.get(key)
    value = getattr(answer, "value", None)
    if not isinstance(value, str) or value not in allowed:
        return None
    probabilities = getattr(answer, "probabilities", None)
    if isinstance(probabilities, dict) and value in probabilities:
        if number(probabilities.get(value), 0.0) < threshold:
            return None
    confidence = getattr(answer, "confidence", None)
    if confidence is not None and number(confidence, 0.0) < 0.08:
        return None
    return value


def noul_answer(answers, key, threshold=0.72):
    if not isinstance(answers, dict):
        return False
    answer = answers.get(key)
    value = getattr(answer, "value", None)
    if isinstance(value, bool):
        return value
    return number(value, 0.0) >= threshold


class Harness:
    """One episode controller; all episode state is intentionally instance-local."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.noop = "noop"
        self.descriptions = {}
        self.started = None

        # Observation/odometry state.
        self.last_tic = None
        self.last_position = None
        self.last_angle = None
        self.last_health = 100.0
        self.last_damage_counter = 0.0
        self.last_counts = (0, 0, 0)
        self.last_observation = {}
        self.last_progress = 0.0
        self.still_ticks = 0
        self.total_displacement = 0.0
        self.visited = collections.Counter()
        self.edges = collections.defaultdict(set)
        self.trail = collections.deque(maxlen=80)
        self.history = collections.deque(maxlen=20)
        self.visible = []
        self.threats = []
        self.projectiles = []
        self.landmarks = {}
        self.collision_count = 0
        self.recovery_count = 0
        self.heading_attempts = collections.Counter()
        self.damage_until = -1.0
        self.damage_spike = 0.0
        self.last_counts_change = 0.0

        # Motor macro state.  A macro is finite; it is never an unbounded
        # commitment to the result of a model call.
        self.current_action = self.noop
        self.remaining = 0
        self.last_post_tic = -1
        self.goal = None
        self.recovery_queue = []
        self.recovery_base = None
        self.recovery_side = 1
        self.recovery_attempt = 0
        self.use_cooldown = {}
        self.interaction_attempts = collections.Counter()
        self.retreat_until = -1.0
        self.retreat_target = None
        self.combat_target = None
        self.combat_started = -1.0
        self.combat_kills = 0

        # Native angle conventions are measured, rather than assumed.
        self.turn_rate = 5.625
        self.left_angle_sign = 1.0

        # Astra's bounded persistent policy.
        self.plan = {
            "wall_side": "right",
            "combat": "cautious",
            "resource_radius": "near",
            "resource_radius_m": 260.0,
            "probe_doors": True,
        }
        self.plan_version = 0
        self.astra_future = None
        self.astra_meta = None
        self.astra_session = None
        self.astra_calls = 0
        self.astra_next = 0.25
        self.last_astra_accept = -1000.0
        self.last_astra_block_count = 0

        # Jev supplies only a short-lived semantic vote.
        self.jev_future = None
        self.jev_meta = None
        self.jev_next = 0.15
        self.jev_mode = None
        self.jev_mode_until = -1.0
        self.jev_target = None
        self.jev_use = False
        self.jev_retreat = False
        self.jev_request_id = 0
        self.system1_enabled = True

    # ---------- public-state derivation ----------

    def _set_action_name(self, action):
        if action in self.allowed:
            return action
        if self.noop in self.allowed:
            return self.noop
        return next(iter(self.allowed), self.noop)

    def _parse_visible(self, observation, now):
        p = position(observation)
        visible = observation.get("visible") if isinstance(observation, dict) else []
        if not isinstance(visible, list):
            visible = []
        self.visible = []
        self.threats = []
        self.projectiles = []
        seen_ids = set()

        for raw in visible:
            if not isinstance(raw, dict):
                continue
            ident = str(raw.get("id", raw.get("object_id", raw.get("name", "?"))))
            name = str(raw.get("name", raw.get("object_name", "")))
            category = str(raw.get("category", raw.get("object_category", "")))
            q = finite_list(raw.get("world"), 3)
            if q is None:
                q = finite_list(raw.get("position"), 3)
            if q is None:
                continue
            d = distance(p, q)
            angle = number((observation.get("player") or {}).get("angle"))
            rel = wrap(bearing(p, q) - angle)
            velocity = finite_list(raw.get("velocity"), 3) or [0.0, 0.0, 0.0]
            speed = math.sqrt(sum(x * x for x in velocity))
            bbox = raw.get("bbox")
            area = 0.0
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                area = max(0.0, number(bbox[2])) * max(0.0, number(bbox[3]))
            entry = {
                "id": ident, "name": name, "category": category,
                "pos": tuple(q), "distance": d, "relative": rel,
                "height": q[2] - (p[2] if p is not None else 0.0),
                "speed": speed, "area": area, "seen": now,
            }
            self.visible.append(entry)
            seen_ids.add(ident)

            if category == "Monster" and not name.startswith("Dead"):
                active = speed > 0.15 or d < 280.0 or now <= self.damage_until
                entry["active"] = active
                entry["hitscan"] = name in HITSCAN
                entry["urgent_name"] = name in URGENT
                entry["threat_score"] = (
                    (180.0 if name in URGENT else 0.0)
                    + (105.0 if name in HITSCAN else 0.0)
                    + (140.0 if d < 120.0 else 70.0 if d < 240.0 else 0.0)
                    + (65.0 if active else 0.0)
                    + min(55.0, area / 18.0)
                    - min(180.0, d * 0.12)
                )
                self.threats.append(entry)
            elif category == "Projectile":
                self.projectiles.append(entry)
            else:
                token = name.lower()
                is_landmark = category in {
                    "Exit", "Switch", "Door", "Key", "Weapon", "Health",
                    "Armor", "Ammo", "Powerup",
                } or any(word in token for word in ("exit", "switch", "door"))
                if is_landmark:
                    old = self.landmarks.get(ident, {})
                    entry.update({
                        "failures": integer(old.get("failures"), 0),
                        "defer": number(old.get("defer"), -1.0),
                        "last_seen": now,
                        "gone": False,
                    })
                    self.landmarks[ident] = entry

        # A label disappearing close to the player is useful pickup evidence,
        # but it is not treated as omniscient knowledge of the map.
        for ident, old in list(self.landmarks.items()):
            if ident not in seen_ids and old.get("last_seen", -1000.0) < now - 0.05:
                if distance(p, old.get("pos")) < 34.0 and abs(number(old.get("height"))) < 36:
                    old["gone"] = True

        self.threats.sort(key=lambda e: e.get("threat_score", 0.0), reverse=True)
        self.projectiles.sort(key=lambda e: e.get("distance", 1e9))

    def update(self, observation, now):
        observation = as_dict(observation)
        if not observation:
            return False
        tic = integer(observation.get("tic"), integer(observation.get("elapsed_tics"), 0))
        p = position(observation)
        angle = number((observation.get("player") or {}).get("angle"))
        dt = 1 if self.last_tic is None else max(1, tic - self.last_tic)

        # The action that produced the last interval is already known locally.
        if self.last_angle is not None and self.current_action in TURNING:
            delta = wrap(angle - self.last_angle)
            if abs(delta) > 0.25:
                measured = abs(delta) / float(dt)
                if 0.2 < measured < 20.0:
                    self.turn_rate = 0.75 * self.turn_rate + 0.25 * measured
                    if self.current_action == "turn_left":
                        self.left_angle_sign = 1.0 if delta > 0 else -1.0
                    else:
                        # Right is the opposite direction in this interface.
                        self.left_angle_sign = -1.0 if delta > 0 else 1.0

        if self.last_position is not None and p is not None:
            moved = distance(self.last_position, p)
            self.total_displacement += min(moved, 128.0)
            if moved >= 2.5:
                self.last_progress = now
                self.still_ticks = 0
            elif self.current_action in TRANSLATION:
                self.still_ticks += dt
            else:
                self.still_ticks = 0
            old_cell = cell_of(self.last_position)
            new_cell = cell_of(p)
            if old_cell != new_cell:
                self.edges[old_cell].add(new_cell)
                self.edges[new_cell].add(old_cell)
                self.last_progress = now
        else:
            moved = 0.0
            self.still_ticks = 0

        current_cell = cell_of(p)
        self.visited[current_cell] += 1
        if not self.trail or distance(self.trail[-1][1], p) > 28.0:
            self.trail.append((now, p, number(observation.get("health"), 100.0)))

        health = number(observation.get("health"), self.last_health)
        damage_counter = number(observation.get("damage_taken"), self.last_damage_counter)
        health_loss = max(0.0, self.last_health - health)
        counter_loss = max(0.0, damage_counter - self.last_damage_counter)
        self.damage_spike = max(health_loss, counter_loss)
        if self.damage_spike >= 1.0:
            self.damage_until = max(self.damage_until, now + 2.8)
            if self.damage_spike >= 8.0:
                self.retreat_until = max(self.retreat_until, now + 2.5)
                self.retreat_target = None

        counts = tuple(integer(observation.get(k), 0) for k in ("kills", "items", "secrets"))
        if counts != self.last_counts:
            self.last_counts_change = now
            self.last_progress = now

        self._parse_visible(observation, now)
        if now - self.last_counts_change < 0.05 or now - (self.history[-1]["t"] if self.history else -1000) >= 1.0:
            self.history.append({
                "t": round(now, 2), "cell": list(current_cell),
                "health": round(health, 1), "armor": round(number(observation.get("armor")), 1),
                "kills": counts[0], "items": counts[1], "secrets": counts[2],
                "action": self.current_action, "still": self.still_ticks,
            })

        self.last_tic = tic
        self.last_position = p
        self.last_angle = angle
        self.last_health = health
        self.last_damage_counter = damage_counter
        self.last_counts = counts
        self.last_observation = observation
        return True

    # ---------- compact model interface ----------

    @staticmethod
    def _range_bucket(d):
        return "touching" if d < 48 else "near" if d < 160 else "medium" if d < 360 else "far"

    @staticmethod
    def _side_bucket(relative):
        return "ahead" if abs(relative) < 15 else "left" if relative > 0 else "right"

    def _label_for_model(self, entry, include_benefit=False):
        result = {
            "id": entry.get("id", ""), "name": entry.get("name", ""),
            "category": entry.get("category", ""),
            "range": self._range_bucket(number(entry.get("distance"), 9999.0)),
            "side": self._side_bucket(number(entry.get("relative"))),
            "height": "above" if number(entry.get("height")) > 32 else "below" if number(entry.get("height")) < -32 else "same_level",
        }
        if entry.get("category") == "Monster":
            result.update({
                "active_evidence": bool(entry.get("active")),
                "hitscanner": bool(entry.get("hitscan")),
                "urgent_name": bool(entry.get("urgent_name")),
            })
        if include_benefit:
            result["approachable_now"] = (
                number(entry.get("distance"), 9999.0) <= number(self.plan.get("resource_radius_m"), 260.0)
                and abs(number(entry.get("height"))) <= 52.0
            )
        return result

    def model_state(self, observation, now):
        p = position(observation)
        health = number(observation.get("health"), 100.0)
        armor = number(observation.get("armor"), 0.0)
        ammo_now = number(observation.get("weapon_ammo"), 0.0)
        resources = []
        for entry in self._resource_candidates(observation, now):
            resources.append(self._label_for_model(entry, include_benefit=True))
            if len(resources) >= 4:
                break
        threats = [self._label_for_model(x) for x in self.threats[:4]]
        projectiles = [self._label_for_model(x) for x in self.projectiles[:2]]
        return {
            "rules": "Reach and activate the map exit. Labels are only currently rendered objects; no wall, sector, key-inventory, or exit map is supplied.",
            "policy": dict(self.plan),
            "health": "critical" if health < 25 else "wounded" if health < 60 else "healthy",
            "armor": "none" if armor < 1 else "light" if armor < 50 else "substantial",
            "ammo": "empty" if ammo_now <= 0 else "low" if ammo_now < 8 else "available",
            "damage_recent": now <= self.damage_until,
            "player": {
                "cell": list(cell_of(p)),
                "facing_sector": heading_bucket((observation.get("player") or {}).get("angle")),
                "moving": self.current_action in TRANSLATION and self.still_ticks < 5,
            },
            "threats": threats, "projectiles": projectiles, "resources": resources,
            "motion": {
                "last_action": self.current_action,
                "blocked": self._blocked(),
                "stalled_seconds": round(max(0.0, now - self.last_progress), 1),
                "visited_cells": min(999, len(self.visited)),
                "known_retreat": len(self.trail) >= 3,
                "recovery_count": self.recovery_count,
            },
            "progress": {
                "kills": self.last_counts[0], "items": self.last_counts[1],
                "secrets": self.last_counts[2],
                "last_count_change_seconds": round(max(0.0, now - self.last_counts_change), 1),
            },
            "semantic_vote": self.jev_mode if now <= self.jev_mode_until else "expired",
        }

    def _question_set(self, state):
        modes = {
            "advance": "Move through the current passage or approach a small, currently rendered useful landmark. Do not infer a straight path from visibility alone.",
            "fight": "Deal with a currently rendered active or dangerous monster before committing to exploration. Code will aim and verify ammunition.",
            "evade": "Preserve health and a retreat lane because recent damage, a close projectile, or a dangerous close threat makes exposure unsafe.",
            "interact": "Test a nearby rendered door, switch, or exit with a single USE pulse, or support the bounded collision probe. Do not claim an unseen exit.",
            "search": "Continue the finite wall-following/search pattern because no listed threat or interaction should dominate now.",
            "hold": "The local evidence is genuinely ambiguous; briefly retain the safe finite motor action while the code re-observes. This is not a permanent idle policy.",
        }
        # The duplicate, reversed Choice is a low-cost guard against the
        # provider's documented first-option bias.  Code acts on agreement.
        ordered = list(modes.items())
        if self.jev_request_id % 2:
            ordered.reverse()
        reverse = list(reversed(ordered))
        q = {
            "mode_a": Choice(
                "Choose the best short-lived local intention from the exact facts in state. "
                "Do not invent geometry, count hidden actors, or output an action. "
                "Prefer survival and exit progress over optional completion.",
                dict(ordered),
            ),
            "mode_b": Choice(
                "Independently choose the same local intention. Use only state.threats, "
                "state.projectiles, state.resources, state.motion, health and policy. "
                "The code will reject disagreement or low-probability answers.",
                dict(reverse),
            ),
            "side": Choice(
                "If a short dodge or search turn is useful, which side is preferable? "
                "Choose straight when neither side is supported by the visible evidence.",
                {
                    "left": "Strafe or turn left; use when the visible danger is on the right or the left side is the safer lane.",
                    "right": "Strafe or turn right; use when the visible danger is on the left or the right side is the safer lane.",
                    "straight": "Do not bias a side; retain the measured finite route action.",
                },
            ),
            "retreat": Noul(
                "Does the exact current state justify retreating along observed travel rather than advancing? "
                "Yes only for recent damage, critical health, a close projectile, or an active close threat; "
                "a distant harmless label is not enough."
            ),
            "use": Noul(
                "Is a single USE pulse warranted now? Say yes only when motion.blocked is true or a "
                "currently rendered door, switch, or exit is nearby; never infer an unseen interaction."
            ),
        }
        if state.get("threats"):
            criteria = {str(x.get("id")): {
                "what": x.get("name", "monster"),
                "facts": "Use the supplied range, side, active_evidence, hitscanner and urgent_name fields."
            } for x in state["threats"]}
            criteria["none"] = "No listed monster should be targeted during this short interval."
            q["target"] = Choice(
                "Which currently rendered monster is the most urgent immediate target? "
                "Choose none for inactive/distant monsters when exploration is safer.", criteria
            )
        return q

    # ---------- local map/search bookkeeping ----------

    def _blocked(self):
        return self.current_action in TRANSLATION and self.still_ticks >= 8

    def _resource_candidates(self, observation, now):
        health = number(observation.get("health"), 100.0)
        armor = number(observation.get("armor"), 0.0)
        weapons = observation.get("weapons") if isinstance(observation.get("weapons"), dict) else {}
        radius = number(self.plan.get("resource_radius_m"), 260.0)
        result = []
        for ident, entry in self.landmarks.items():
            if entry.get("gone") or number(entry.get("defer"), -1.0) > now:
                continue
            if now - number(entry.get("last_seen"), -1000.0) > 0.8:
                continue
            category = entry.get("category", "")
            name = str(entry.get("name", ""))
            if category not in {"Key", "Weapon", "Health", "Armor", "Ammo", "Powerup"}:
                continue
            d = number(entry.get("distance"), 9999.0)
            z = abs(number(entry.get("height"), 9999.0))
            if d > radius or z > 52.0:
                continue
            utility = 0.0
            reason = ""
            if category == "Key":
                utility, reason = 125.0, "key"
            elif category == "Weapon":
                # Match rendered classic weapon names to the public WEAPON
                # variables.  A shotgun/chain weapon already owned should not
                # become a stale steering target merely because it is visible.
                weapon_slots = {
                    "Pistol": 2, "Shotgun": 3, "SuperShotgun": 3,
                    "Chaingun": 4, "RocketLauncher": 5, "PlasmaRifle": 6,
                    "BFG9000": 7, "Chainsaw": 1,
                }
                slot = weapon_slots.get(name)
                owned = ((bool(weapons.get(str(slot - 1))) or bool(weapons.get(str(slot))))
                         if slot is not None else False)
                if owned:
                    continue
                utility, reason = 95.0, "new weapon"
            elif category == "Health":
                if health >= 92.0:
                    continue
                utility, reason = (100.0 if health < 35 else 52.0), "healing"
            elif category == "Armor":
                if armor >= 80.0:
                    continue
                utility, reason = (72.0 if armor < 30 else 38.0), "armor"
            elif category == "Ammo":
                if number(observation.get("weapon_ammo"), 0.0) >= 12.0:
                    continue
                utility, reason = 32.0, "usable ammo"
            elif category == "Powerup":
                utility, reason = 45.0, "powerup"
            score = utility - d * 0.16 - z * 0.8 - integer(entry.get("failures"), 0) * 42.0
            result.append(dict(entry, score=score, benefit=reason))
        result.sort(key=lambda x: number(x.get("score"), -1e9), reverse=True)
        return result[:5]

    def _near_interaction(self, now):
        candidates = []
        for entry in self.visible:
            category = entry.get("category", "")
            name = str(entry.get("name", ""))
            token = name.lower()
            if category not in {"Exit", "Switch", "Door"} and not any(
                    x in token for x in ("exit", "switch", "door")):
                continue
            if number(entry.get("distance"), 9999.0) > 110.0:
                continue
            if abs(number(entry.get("height"), 9999.0)) > 55.0:
                continue
            candidates.append(entry)
        if not candidates:
            return None
        candidates.sort(key=lambda e: (0 if e.get("category") == "Exit" or "exit" in str(e.get("name", "")).lower() else 1,
                                       number(e.get("distance"), 9999.0)))
        return candidates[0]

    def _turn_for(self, difference):
        if difference * self.left_angle_sign > 0:
            return self._set_action_name("turn_left")
        return self._set_action_name("turn_right")

    def _turn_ticks(self, degrees, minimum=2, maximum=14):
        return max(minimum, min(maximum, int(math.ceil(abs(number(degrees)) / max(1.0, self.turn_rate))) + 1))

    def _steer_to(self, destination):
        p = self.last_position
        if p is None or destination is None:
            return self._set_action_name("forward"), 8, "no_destination"
        diff = wrap(bearing(p, destination) - self.last_angle)
        if abs(diff) > 12.0:
            return self._turn_for(diff), self._turn_ticks(diff), "align"
        return self._set_action_name("forward"), 10, "approach"

    def _preferred_side(self):
        preferred = 1 if self.plan.get("wall_side") == "right" else -1
        c = cell_of(self.last_position)
        h = heading_bucket(self.last_angle)
        # Prefer a less-tested adjacent heading.  The policy preference breaks
        # ties; every third failed attempt deliberately changes handedness.
        left_h = (h + 2) % 8
        right_h = (h - 2) % 8
        left_count = self.heading_attempts[(c, left_h)]
        right_count = self.heading_attempts[(c, right_h)]
        if left_count < right_count:
            preferred = -1
        elif right_count < left_count:
            preferred = 1
        if self.recovery_count and self.recovery_count % 3 == 0:
            preferred *= -1
        return preferred

    def _start_recovery(self, now):
        if self.recovery_queue:
            return
        p = self.last_position
        c = cell_of(p)
        h = heading_bucket(self.last_angle)
        self.heading_attempts[(c, h)] += 1
        self.collision_count += 1
        self.recovery_count += 1
        self.recovery_attempt = 1
        self.recovery_side = self._preferred_side()
        self.recovery_base = p
        self.goal = None
        self.jev_mode_until = min(self.jev_mode_until, now)

        queue = []
        use_key = (c, h)
        if self.plan.get("probe_doors", True) and self.use_cooldown.get(use_key, -1.0) <= now:
            if "use" in self.allowed:
                self.use_cooldown[use_key] = now + 4.0
                queue.append(("use", 1))
        # After USE, give an opening a short straight test before rotating
        # away.  A door/exit can become usable without changing labels.
        if queue:
            queue.append((self._set_action_name("forward"), 10))
        turn = "turn_right" if self.recovery_side > 0 else "turn_left"
        queue.extend([(self._set_action_name(turn), self._turn_ticks(82.0, 10, 17)),
                      (self._set_action_name("forward"), 26)])
        self.recovery_queue = queue
        self.ctx.log("checkpoint_recovery", cell=list(c), side="right" if self.recovery_side > 0 else "left",
                     attempt=self.recovery_count, queue=[x[0] for x in queue])

    def _recovery_decision(self, now):
        # A successful post-USE opening should release the remaining branch
        # immediately instead of turning away after the test segment.
        if (self.recovery_queue and self.recovery_base is not None
                and self.current_action in TRANSLATION
                and distance(self.recovery_base, self.last_position) >= 18.0):
            self.recovery_queue = []
            self.still_ticks = 0
            self.recovery_base = None
            return None
        if self.recovery_queue:
            action, duration = self.recovery_queue[0]
            if self.current_action == action and self.remaining > 0:
                return action, self.remaining, "recovery_hold"
            if self.current_action == action and self.remaining <= 0:
                self.recovery_queue.pop(0)
            if self.recovery_queue:
                action, duration = self.recovery_queue[0]
                return action, duration, "recovery_step"

        # The first turn/advance pair is over.  Leaving a cell is enough to
        # release the normal policy; a repeated collision gets one opposite
        # branch, then a short backtrack rather than an infinite spin.
        moved = distance(self.recovery_base, self.last_position) >= 18.0
        if moved:
            self.recovery_queue = []
            self.still_ticks = 0
            self.recovery_base = None
            return None
        if self.recovery_attempt == 1:
            self.recovery_attempt = 2
            side = -self.recovery_side
            turn = "turn_right" if side > 0 else "turn_left"
            self.recovery_queue = [(self._set_action_name(turn), self._turn_ticks(92.0, 11, 18)),
                                   (self._set_action_name("forward"), 30)]
            return self.recovery_queue[0][0], self.recovery_queue[0][1], "opposite_recovery"
        if self.recovery_attempt == 2:
            self.recovery_attempt = 3
            self.recovery_queue = [(self._set_action_name("backward"), 15),
                                   (self._set_action_name("turn_left" if self.recovery_side > 0 else "turn_right"), 16),
                                   (self._set_action_name("forward"), 30)]
            return self.recovery_queue[0][0], self.recovery_queue[0][1], "backtrack_recovery"
        self.recovery_queue = []
        self.recovery_base = None
        self.still_ticks = 0
        return self._set_action_name("forward"), 14, "new_search_segment"

    # ---------- combat and survival ----------

    def _target(self, now):
        visible_ids = {x.get("id") for x in self.threats}
        if self.jev_target in visible_ids:
            chosen = next((x for x in self.threats if x.get("id") == self.jev_target), None)
            if chosen is not None:
                return chosen
        for entry in self.threats:
            if entry.get("active") or number(entry.get("distance"), 9999.0) < 190.0:
                return entry
        if self.jev_mode == "fight" and now <= self.jev_mode_until:
            limit = 260.0 if self.plan.get("combat") == "cautious" else 430.0
            return self.threats[0] if self.threats and number(self.threats[0].get("distance"), 9999.0) < limit else None
        return None

    def _weapon_pulse(self, observation, now):
        selected_ammo = number(observation.get("weapon_ammo"), 0.0)
        if selected_ammo > 0 or now - number(getattr(self, "last_weapon_change", -1000.0)) < 1.5:
            return None
        weapons = observation.get("weapons") if isinstance(observation.get("weapons"), dict) else {}
        ammo = observation.get("ammo") if isinstance(observation.get("ammo"), dict) else {}
        current = integer(observation.get("weapon"), -1)
        # Conventional low slots are preferred.  A wrong/unowned slot is
        # harmless because the adaptor ignores unavailable selection pulses.
        for slot in (3, 4, 6, 2, 1):
            action = "select_weapon_" + str(slot)
            if action not in self.allowed or current + 1 == slot:
                continue
            owned = bool(weapons.get(str(slot - 1))) or bool(weapons.get(str(slot)))
            available_ammo = max(number(ammo.get(str(slot - 1))), number(ammo.get(str(slot))))
            if owned and available_ammo > 0:
                self.last_weapon_change = now
                return action
        return None

    def _combat_decision(self, observation, now, target):
        if target is None:
            return None
        health = number(observation.get("health"), 100.0)
        d = number(target.get("distance"), 9999.0)
        diff = number(target.get("relative"), 0.0)

        if health < 22.0 and (now <= self.damage_until or d < 150.0):
            return self._retreat_decision(now)
        pulse = self._weapon_pulse(observation, now)
        if pulse:
            return pulse, 1, "weapon_switch"
        if abs(diff) > 10.0:
            return self._turn_for(diff), self._turn_ticks(diff, 2, 9), "aim"

        # A nearby projectile or body blocker gets a lateral escape.  At
        # standoff, fire while strafing rather than holding a stationary shot.
        dodge = "right" if self.recovery_side > 0 else "left"
        if self.projectiles and number(self.projectiles[0].get("distance"), 9999.0) < 190.0:
            action = "strafe_right" if dodge == "right" else "strafe_left"
            return self._set_action_name(action), 6, "projectile_dodge"
        if d < 78.0 and now <= self.damage_until:
            action = "backward_right" if dodge == "right" else "backward_left"
            return self._set_action_name(action), 6, "close_retreat"

        # Refuse to keep a splash weapon pressed against an unverified nearby
        # target.  The exact selected slot is native data, so only the obvious
        # high slots are treated as splash-risk here.
        selected = integer(observation.get("weapon"), -1)
        if selected >= 4 and d < 180.0:
            fallback = self._set_action_name("select_weapon_3")
            if fallback in self.allowed and fallback != self.noop:
                self.last_weapon_change = now
                return fallback, 1, "splash_safe_switch"
        if "fire_strafe_left" in self.allowed or "fire_strafe_right" in self.allowed:
            action = "fire_strafe_right" if dodge == "right" else "fire_strafe_left"
        else:
            action = "fire" if "fire" in self.allowed else "forward"
        return self._set_action_name(action), 5, "fire_and_dodge"

    def _retreat_decision(self, now):
        p = self.last_position
        if self.retreat_target is None:
            choices = [(t, q) for t, q, hp in self.trail
                       if q is not None and now - t > 0.8 and distance(p, q) > 90.0]
            if choices:
                self.retreat_target = choices[-1][1]
        if self.retreat_target is not None and now <= self.retreat_until + 1.0:
            action, duration, reason = self._steer_to(self.retreat_target)
            if distance(p, self.retreat_target) < 42.0:
                self.retreat_target = None
                self.retreat_until = min(self.retreat_until, now + 0.4)
            return action, duration, "retreat_" + reason
        self.retreat_target = None
        action = "backward_left" if self.recovery_side < 0 else "backward_right"
        return self._set_action_name(action), 8, "blind_retreat"

    # ---------- action selection ----------

    def _goal_decision(self, observation, now):
        if self.goal:
            ident = self.goal.get("id")
            live = next((x for x in self.visible if x.get("id") == ident), None)
            if live is None or now - number(self.goal.get("last_seen"), -1000.0) > 0.8:
                old = self.landmarks.get(ident)
                if old is not None:
                    old["failures"] = integer(old.get("failures"), 0) + 1
                    old["defer"] = now + min(45.0, 8.0 * old["failures"])
                self.goal = None
            else:
                self.goal["last_seen"] = now
                d = number(live.get("distance"), 9999.0)
                if d < 34.0:
                    self.goal = None
                elif d < number(self.goal.get("best"), 1e9) - 8.0:
                    self.goal["best"] = d
                    self.goal["last_progress"] = now
                elif now - number(self.goal.get("last_progress"), now) > 1.8:
                    old = self.landmarks.get(ident)
                    if old is not None:
                        old["failures"] = integer(old.get("failures"), 0) + 1
                        old["defer"] = now + min(45.0, 8.0 * old["failures"])
                    self.goal = None
                else:
                    return self._steer_to(live.get("pos"))

        # A rendered exit/switch/door is a route objective, not merely a
        # one-tic interaction.  Approach only recent, same-level labels and
        # let the normal bounded failure rule abandon an obstructed one.
        interactions = []
        for entry in self.visible:
            category = entry.get("category", "")
            name = str(entry.get("name", ""))
            token = name.lower()
            if category not in {"Exit", "Switch", "Door"} and not any(
                    word in token for word in ("exit", "switch", "door")):
                continue
            if number(entry.get("distance"), 9999.0) > 340.0 or abs(number(entry.get("height"), 9999.0)) > 58.0:
                continue
            is_exit = category == "Exit" or "exit" in token
            score = (1200.0 if is_exit else 260.0) - number(entry.get("distance"), 9999.0) * 0.25
            interactions.append((score, entry))
        if interactions and self.jev_mode not in {"evade", "hold"}:
            _, best = max(interactions, key=lambda x: x[0])
            self.goal = {"id": best.get("id"), "pos": best.get("pos"),
                         "best": number(best.get("distance"), 9999.0),
                         "last_progress": now, "last_seen": now}
            return self._steer_to(best.get("pos"))

        candidates = self._resource_candidates(observation, now)
        if candidates and (self.jev_mode != "search" or candidates[0].get("category") == "Key"):
            best = candidates[0]
            self.goal = {"id": best.get("id"), "pos": best.get("pos"),
                         "best": number(best.get("distance"), 9999.0),
                         "last_progress": now, "last_seen": now}
            return self._steer_to(best.get("pos"))
        return None

    def decide_action(self, observation, now):
        health = number(observation.get("health"), 100.0)

        # A recovery queue has precedence over model semantics.  It is the
        # position-based anti-stall mechanism carried over from the strong
        # parent, but upgraded from a three-action nudge to a finite search
        # branch with USE and opposite-side checkpoints.
        if self.recovery_queue or self.recovery_base is not None:
            choice = self._recovery_decision(now)
            if choice is not None:
                return choice

        immediate_projectile = bool(self.projectiles and number(self.projectiles[0].get("distance"), 9999.0) < 150.0)
        active_target = self._target(now)
        urgent_target = active_target is not None and (
            bool(active_target.get("active"))
            or number(active_target.get("distance"), 9999.0) < 150.0
            or now <= self.damage_until
        )

        # A collision is measured from the player's position, never inferred
        # from a visible pickup.  Use once, turn, and test a new corridor.
        if self._blocked():
            self._start_recovery(now)
            return self._recovery_decision(now) or (self._set_action_name("turn_right"), 12, "blocked_turn")

        if health < 24.0 and (now <= self.damage_until or self.damage_spike >= 5.0):
            self.retreat_until = max(self.retreat_until, now + 2.0)
            return self._retreat_decision(now)
        if immediate_projectile:
            self.retreat_until = max(self.retreat_until, now + 1.0)
            return self._retreat_decision(now)
        if now <= self.retreat_until:
            return self._retreat_decision(now)

        # Fight only a currently rendered threat.  The semantic target can
        # refine the code ranking, but cannot authorize an invented monster.
        if active_target is not None and (urgent_target or self.jev_mode == "fight"):
            if self.combat_target != active_target.get("id"):
                self.combat_target = active_target.get("id")
                self.combat_started = now
                self.combat_kills = self.last_counts[0]
            if now - self.combat_started < 13.0 or urgent_target:
                combat = self._combat_decision(observation, now, active_target)
                if combat is not None:
                    return combat
        else:
            self.combat_target = None

        # Interactions are attempted only when rendered and close.  Alignment
        # is a separate finite turn macro, so the one-tic USE pulse is not lost.
        interaction = self._near_interaction(now)
        if interaction is not None and not urgent_target:
            ident = interaction.get("id")
            key = ("landmark", ident)
            if now >= number(self.use_cooldown.get(key), -1.0):
                diff = number(interaction.get("relative"), 0.0)
                if abs(diff) > 13.0:
                    return self._turn_for(diff), self._turn_ticks(diff, 2, 8), "align_interaction"
                self.use_cooldown[key] = now + 2.2
                self.interaction_attempts[ident] += 1
                self.ctx.log("visible_use", id=ident, name=interaction.get("name", ""),
                             attempt=self.interaction_attempts[ident])
                if "use" in self.allowed:
                    return "use", 1, "visible_interaction"

        goal = self._goal_decision(observation, now)
        if goal is not None and self.jev_mode not in {"evade", "hold"}:
            return goal

        if self.jev_mode == "evade" or self.jev_retreat:
            self.retreat_until = max(self.retreat_until, now + 1.2)
            return self._retreat_decision(now)

        # The normal search is deliberately a long-enough straight segment to
        # discover a new room, but every segment is bounded and checkpointed.
        return self._set_action_name("forward"), 18, "search_forward"

    # ---------- model scheduling ----------

    def _poll_jev(self, now):
        if self.jev_future is None or not self.jev_future.done():
            return
        future, meta = self.jev_future, self.jev_meta
        self.jev_future = None
        try:
            answers = future.result()
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError):
            self.ctx.log("jev_failure", fallback="finite_motor")
            self.jev_next = now + 0.8
            return
        age = now - number(meta.get("time"), now)
        if age > 2.4 or integer(meta.get("tic"), -1) != integer(self.last_tic, -2) and age > 1.2:
            self.ctx.log("jev_discard", age_s=round(age, 2), reason="late")
            return
        if integer(meta.get("plan_version"), -1) != self.plan_version:
            self.ctx.log("jev_discard", age_s=round(age, 2), reason="policy_changed")
            return
        allowed_modes = {"advance", "fight", "evade", "interact", "search", "hold"}
        first = choice_answer(answers, "mode_a", allowed_modes)
        second = choice_answer(answers, "mode_b", allowed_modes)
        mode = first if first is not None and first == second else None
        if mode is not None:
            self.jev_mode = mode
            self.jev_mode_until = now + 1.6
        side = choice_answer(answers, "side", {"left", "right", "straight"})
        if side in {"left", "right"} and self._blocked():
            self.recovery_side = -1 if side == "left" else 1
        self.jev_retreat = noul_answer(answers, "retreat")
        self.jev_use = noul_answer(answers, "use")
        target = choice_answer(answers, "target", {x.get("id") for x in self.threats} | {"none"})
        self.jev_target = None if target in (None, "none") else target
        self.ctx.log("jev_vote", mode=self.jev_mode, target=self.jev_target,
                     retreat=self.jev_retreat, use=self.jev_use, age_s=round(age, 2))

    def _poll_astra(self, now):
        if self.astra_future is None or not self.astra_future.done():
            return
        future = self.astra_future
        self.astra_future = None
        try:
            result = future.result()
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError):
            self.ctx.log("astra_failure", fallback="current_policy")
            self.astra_next = now + 60.0
            return
        data = extract_json(getattr(result, "text", ""))
        accepted = {}
        if data.get("wall_side") in {"left", "right"}:
            accepted["wall_side"] = data["wall_side"]
        if data.get("combat") in {"cautious", "normal", "aggressive"}:
            accepted["combat"] = data["combat"]
        if data.get("resource_radius") in {"near", "moderate"}:
            accepted["resource_radius"] = data["resource_radius"]
            accepted["resource_radius_m"] = 260.0 if data["resource_radius"] == "near" else 390.0
        if isinstance(data.get("probe_doors"), bool):
            accepted["probe_doors"] = data["probe_doors"]
        if accepted:
            self.plan.update(accepted)
            self.plan_version += 1
            self.jev_mode_until = -1.0
            self.ctx.log("astra_policy", version=self.plan_version, policy=dict(self.plan))
        else:
            self.ctx.log("astra_invalid", fallback="current_policy")
        self.astra_session = getattr(result, "session_id", self.astra_session)
        self.last_astra_accept = now
        self.astra_next = now + 75.0

    def _dispatch_jev(self, observation, now):
        if not self.system1_enabled or self.jev_future is not None or now < self.jev_next:
            return
        remaining = self.ctx.budget.remaining()
        if number(remaining.get("system1_calls"), 0.0) <= 12 or number(remaining.get("wall_s"), 0.0) <= 3.0:
            return
        state = self.model_state(observation, now)
        questions = self._question_set(state)
        self.jev_request_id += 1
        self.jev_meta = {"time": now, "tic": self.last_tic,
                         "plan_version": self.plan_version}
        try:
            self.jev_future = self.ctx.spawn(
                lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=2.6)
            )
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError):
            self.jev_future = None
            self.jev_next = now + 0.8
            return
        combat = bool(self.threats) or now <= self.damage_until
        self.jev_next = now + (0.42 if combat or self._blocked() else 0.85)

    def _astra_prompt(self, observation, now):
        state = self.model_state(observation, now)
        evidence = list(self.history)[-10:]
        return (
            "You are the slow policy reviewer for a real-time classic Doom controller. "
            "The player must survive and activate the map exit; labels are only rendered "
            "objects, and no map geometry or hidden inventory is available. The fast code "
            "already computes bearings, odometry, finite collision recovery, weapon legality, "
            "and emergency retreat. Do not output raw actions, coordinates, map-specific routes, "
            "code, or claims about unseen doors. Return ONLY one short JSON object with exactly "
            "optional fields: wall_side (left|right), combat (cautious|normal|aggressive), "
            "resource_radius (near|moderate), probe_doors (true|false). Choose near when a "
            "far visible pickup is likely to induce a blocked steering loop. Choose cautious "
            "when health is falling; choose aggressive only for an active blocker. This changes "
            "preferences, not the safety layer. Current state="
            + json.dumps(state, separators=(",", ":"))
            + " Recent measured evidence=" + json.dumps(evidence, separators=(",", ":"))
        )

    def _dispatch_astra(self, observation, now):
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        remaining = self.ctx.budget.remaining()
        if number(remaining.get("system2_calls"), 0.0) <= 0 or number(remaining.get("wall_s"), 0.0) <= 100.0:
            return
        blocked_events = self.recovery_count
        initial = self.astra_calls == 0 and now >= self.astra_next
        learned_failure = blocked_events >= self.last_astra_block_count + 2 and now >= self.astra_next
        if not (initial or learned_failure):
            return
        prompt = self._astra_prompt(observation, now)
        session = self.astra_session
        try:
            self.astra_future = self.ctx.spawn(
                lambda p=prompt, s=session: self.ctx.think(p, session_id=s, timeout_s=180.0)
            )
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError):
            self.astra_future = None
            self.astra_next = now + 60.0
            return
        self.astra_meta = {"time": now, "blocks": blocked_events}
        self.astra_calls += 1
        self.last_astra_block_count = blocked_events
        self.astra_next = now + 75.0
        self.ctx.log("astra_request", reason="initial" if initial else "repeated_collision")

    # ---------- episode loop ----------

    def _post_if_needed(self, action, duration, tic, reason):
        action = self._set_action_name(action)
        duration = max(1, integer(duration, 1))
        # If a macro has not expired, keeping it is intentional.  A changed
        # emergency decision or a recovery step always has a different action
        # (or arrives after remaining reaches zero).
        if action != self.current_action or self.remaining <= 0:
            self.ctx.act(action)
            self.current_action = action
            self.remaining = duration
            self.last_post_tic = tic
            if reason not in {"search_forward", "recovery_hold", "approach", "hold"}:
                self.ctx.log("motor", action=action, duration=duration, reason=reason)

    def run_episode(self, game):
        self.allowed = set(getattr(game.action_spec, "names", ()) or ())
        if not self.allowed:
            return
        self.descriptions = getattr(game.action_spec, "descriptions", {}) or {}
        if not isinstance(self.descriptions, dict):
            self.descriptions = {}
        candidate_noop = getattr(game.action_spec, "noop", "noop")
        self.noop = candidate_noop if candidate_noop in self.allowed else next(iter(self.allowed))
        self.current_action = self.noop
        self.system1_enabled = getattr(self.ctx, "system1", True) is not None
        self.started = self.ctx.clock.now()

        # Safe initial action is posted before the first slow request.
        ctx = self.ctx
        ctx.act(self.noop)
        self.remaining = 1

        while not ctx.done and not bool(getattr(game, "done", False)):
            observation = as_dict(ctx.observe())
            if not observation:
                ctx.wait_for_tick()
                continue
            if bool(observation.get("terminal")) or bool(observation.get("dead")):
                return
            tic = integer(observation.get("tic"), integer(observation.get("elapsed_tics"), 0))
            if self.last_tic is not None and tic == self.last_tic:
                ctx.wait_for_tick()
                continue
            now = max(0.0, number(ctx.clock.now()) - number(self.started))
            dt = 1 if self.last_tic is None else max(1, tic - self.last_tic)
            self.remaining -= dt
            if not self.update(observation, now):
                ctx.wait_for_tick()
                continue
            self._poll_jev(now)
            self._poll_astra(now)
            self._dispatch_jev(observation, now)
            self._dispatch_astra(observation, now)

            # Expired semantic votes never become raw actions; they simply
            # return control to the deterministic search/combat policy.
            if now > self.jev_mode_until:
                self.jev_mode = None
                self.jev_target = None
                self.jev_retreat = False
                self.jev_use = False

            # Do not recalculate a safe macro on every real-time tick.  This is
            # what prevents a far visible pickup from generating left/right
            # oscillation, while the checks below still interrupt danger.
            urgent = bool(self.projectiles and number(self.projectiles[0].get("distance"), 9999.0) < 150.0)
            danger = self.damage_spike >= 8.0 or (number(observation.get("health"), 100.0) < 24.0 and now <= self.damage_until)
            must_reconsider = self.remaining <= 0 or self._blocked() or urgent or danger
            if not must_reconsider:
                self._post_if_needed(self.current_action, self.remaining, tic, "hold")
            else:
                action, duration, reason = self.decide_action(observation, now)
                self._post_if_needed(action, duration, tic, reason)
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
