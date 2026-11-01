"""Candidate 0018: a landmark-ledger expedition controller for async Doom.

The controller deliberately separates three jobs.  Code owns the motor, combat
safety, and a persistent ledger of public landmarks.  Jev classifies the local
situation and chooses a high-level lease.  Astra is an infrequent mission-card
editor.  No inference result is allowed to hold the real-time loop hostage.
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
MELEE = {"Demon", "Spectre", "Pinkie", "Pinky", "LostSoul"}
MAJOR = {"Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind",
         "Revenant"}
STATIC_CATEGORIES = {
    "exit", "switch", "door", "teleporter", "teleport", "key", "weapon",
    "health", "armor", "ammo", "powerup", "power-up",
}


def num(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    try:
        return int(num(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_obs(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    result = dict(raw)
    player = raw.get("player")
    result["player"] = dict(player) if isinstance(player, dict) else {}
    for key in ("x", "y", "z", "angle"):
        result["player"][key] = num(result["player"].get(key))
    visible = raw.get("visible")
    result["visible"] = visible if isinstance(visible, list) else []
    return result


def json_from_text(text):
    """Find the first valid object in arbitrary Astra output."""
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


def position(obs):
    p = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    return num(p.get("x")), num(p.get("y")), num(p.get("z"))


def coarse_cell(point):
    # This is an explored-location memory, not an asserted map grid.
    return (math.floor(point[0] / 96.0), math.floor(point[1] / 96.0),
            math.floor(point[2] / 96.0))


def angle_bin(angle):
    return int(math.floor((angle % 360.0 + 22.5) / 45.0)) % 8


def label_facts(obs):
    here = position(obs)
    facing = num((obs.get("player") or {}).get("angle")) % 360.0
    facts = []
    for raw in obs.get("visible", []):
        if not isinstance(raw, dict):
            continue
        world = raw.get("world")
        if not isinstance(world, (list, tuple)) or len(world) < 3:
            continue
        point = (num(world[0]), num(world[1]), num(world[2]))
        name = str(raw.get("name", raw.get("object_name", "unknown")))[:80]
        category = str(raw.get("category", raw.get("object_category", "Unknown")))[:40]
        lc = category.lower()
        ln = name.lower()
        if lc == "self" or name == "DoomPlayer":
            continue
        d = distance(here, point)
        bearing = math.degrees(math.atan2(point[1] - here[1], point[0] - here[0]))
        error = wrap(bearing - facing)
        monster = lc == "monster" and not ln.startswith("dead")
        projectile = ("projectile" in lc or "missile" in lc or
                      lc == "explosive" or ln in {"doomimpball", "rocket", "plasmafire", "fireball"})
        interact = (lc in {"exit", "switch", "door", "teleporter", "teleport"} or
                    any(word in ln for word in ("exit", "switch", "door", "teleport")))
        key = lc == "key" or any(word in ln for word in ("redcard", "bluecard", "yellowcard", "key"))
        static = interact or key or lc in STATIC_CATEGORIES
        fact = {
            "id": str(raw.get("id", raw.get("object_id", name))),
            "name": name,
            "category": category,
            "point": point,
            "distance": d,
            "height": point[2] - here[2],
            "error": error,
            "monster": monster,
            "projectile": projectile,
            "interact": interact,
            "key": key,
            "static": static,
            "hitscan": monster and name in HITSCAN,
            "melee": monster and name in MELEE,
            "major": monster and name in MAJOR,
        }
        facts.append(fact)
    return facts


def answer_choice(answers, key, options, threshold):
    if not isinstance(answers, dict):
        return None
    answer = answers.get(key)
    value = getattr(answer, "value", None)
    if value not in options:
        return None
    probabilities = getattr(answer, "probabilities", None)
    if isinstance(probabilities, dict):
        if num(probabilities.get(value), 0.0) < threshold:
            return None
    confidence = getattr(answer, "confidence", None)
    if confidence is not None and num(confidence, 0.0) < 0.08:
        return None
    return value


def answer_yes(answers, key, threshold=0.70):
    if not isinstance(answers, dict):
        return False
    value = num(getattr(answers.get(key), "value", 0.0), 0.0)
    return value >= threshold


class Expedition:
    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.noop = "noop"
        self.prev_obs = None
        self.prev_pos = None
        self.prev_tic = -1
        self.prev_action = "noop"
        self.last_health = 100.0
        self.damage_until = -100
        self.blocked = 0
        self.still = 0
        self.turn_rate = 5.5
        self.turn_sign = 1.0

        self.cells = collections.Counter()
        self.rays = collections.Counter()
        self.failed_rays = collections.Counter()
        self.last_new_tic = 0
        self.last_cell = None
        self.last_counts = (0, 0, 0)

        # id -> a public, previously rendered landmark.  It is intentionally
        # not a map oracle: entries exist only after a label was visible.
        self.landmarks = {}
        self.failed_targets = {}
        self.target_id = None
        self.target_started = 0
        self.target_best = float("inf")
        self.target_last_seen = -1
        self.target_close_since = None

        self.escape = collections.deque()
        self.nav_lease = collections.deque()
        self.scan_lease = collections.deque()
        self.explore_side = 1
        self.use_requested = False
        self.last_use = -10000
        self.face_use = {}
        self.aim_streak = 0
        self.orbit = collections.deque()
        self.last_weapon = -10000
        self.weapon_hint = None

        self.mode_hint = "survey"
        self.side_hint = None
        self.next_jev = 0
        self.jev_future = None
        self.jev_request = None
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.next_astra = 1
        self.policy_version = 0
        self.policy = {
            "priority": "exit",
            "mode": "landmark_first",
            "side": "left",
            "caution": "normal",
            "use": "barrier",
        }
        self.events = collections.deque(maxlen=18)

    # ---------- legality and exact local facts ----------
    def legal(self, action):
        if action in self.allowed:
            return action
        alternatives = {
            "forward_left": ("forward", "strafe_left", "turn_left"),
            "forward_right": ("forward", "strafe_right", "turn_right"),
            "backward_left": ("backward", "strafe_left", "turn_left"),
            "backward_right": ("backward", "strafe_right", "turn_right"),
            "fire_strafe_left": ("fire_forward", "fire", "strafe_left"),
            "fire_strafe_right": ("fire_forward", "fire", "strafe_right"),
            "fire_forward": ("fire", "forward"),
        }
        for candidate in alternatives.get(action, ()):
            if candidate in self.allowed:
                return candidate
        if self.noop in self.allowed:
            return self.noop
        for candidate in ("forward", "backward", "strafe_left", "turn_left"):
            if candidate in self.allowed:
                return candidate
        return next(iter(self.allowed), self.noop)

    def turn(self, sign):
        # sign is desired positive/negative angular correction.  The mapping is
        # calibrated from observed angle deltas, so it does not assume a button
        # index or a particular renderer convention.
        return self.legal("turn_left" if (sign > 0) == (self.turn_sign > 0) else "turn_right")

    def side_move(self, side, firing=False, backward=False):
        if backward:
            name = "backward_left" if side > 0 else "backward_right"
        elif firing:
            name = "fire_strafe_left" if side > 0 else "fire_strafe_right"
        else:
            name = "forward_left" if side > 0 else "forward_right"
        return self.legal(name)

    def record(self, event, **fields):
        self.events.append(event)
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    # ---------- public observation -> ledger ----------
    def update(self, obs):
        tic = integer(obs.get("tic"), self.prev_tic + 1)
        if tic <= self.prev_tic:
            return tic, False
        point = position(obs)
        facing = num((obs.get("player") or {}).get("angle"))
        if self.prev_obs is not None and self.prev_pos is not None:
            dt = max(1, tic - self.prev_tic)
            moved = distance(point, self.prev_pos)
            old_angle = num((self.prev_obs.get("player") or {}).get("angle"))
            delta = wrap(facing - old_angle)
            if self.prev_action in TURN and 0 < dt <= 12 and abs(delta) > 0.1:
                rate = abs(delta) / dt
                if 0.1 < rate < 30:
                    self.turn_rate = 0.8 * self.turn_rate + 0.2 * rate
                    self.turn_sign = 1.0 if (
                        (self.prev_action == "turn_left" and delta > 0) or
                        (self.prev_action == "turn_right" and delta < 0)
                    ) else -1.0
            if self.prev_action in MOVE and dt <= 16:
                if moved < max(2.0, dt * 0.55):
                    self.blocked += dt
                    self.still += dt
                else:
                    self.blocked = 0
                    self.still = 0
            else:
                self.blocked = 0
                self.still = 0
            old_health = num(self.prev_obs.get("health"), self.last_health)
            health = num(obs.get("health"), old_health)
            if health < old_health - 0.1:
                self.damage_until = tic + 38
            self.last_health = health
        cell = coarse_cell(point)
        fresh_cell = cell not in self.cells
        self.cells[cell] += 1
        self.rays[(cell, angle_bin(facing))] += 1
        if fresh_cell:
            self.last_new_tic = tic
            self.last_cell = cell
        counts = tuple(integer(obs.get(name), 0) for name in ("kills", "items", "secrets"))
        if counts != self.last_counts:
            self.last_counts = counts
            self.last_new_tic = tic
        facts = label_facts(obs)
        seen_ids = set()
        for item in facts:
            if not item["static"]:
                continue
            ident = item["id"]
            seen_ids.add(ident)
            old = self.landmarks.get(ident)
            self.landmarks[ident] = {
                "id": ident,
                "name": item["name"],
                "category": item["category"],
                "point": list(item["point"]),
                "first": old.get("first", tic) if old else tic,
                "last": tic,
                "seen": True,
                "claimed": old.get("claimed", False) if old else False,
            }
            if self.target_id == ident:
                self.target_last_seen = tic
        # A static pickup that disappeared immediately after contact was likely
        # collected.  Interaction labels are retained, because doors can reopen.
        if self.target_id is not None:
            target = self.landmarks.get(self.target_id)
            if target and target.get("category", "").lower() not in {"exit", "switch", "door", "teleporter", "teleport"}:
                if tic - self.target_last_seen > 8 and self.target_close_since is not None:
                    target["claimed"] = True
                    self.failed_targets[self.target_id] = tic + 500
        self.prev_obs = obs
        self.prev_pos = point
        self.prev_tic = tic
        return tic, True

    def update_target_close(self, target, tic):
        if target is None:
            self.target_close_since = None
            return
        if target["distance"] < 58:
            if self.target_close_since is None:
                self.target_close_since = tic
        else:
            self.target_close_since = None

    # ---------- derived model interface ----------
    def threats(self, facts):
        result = []
        for item in facts:
            if not item["monster"]:
                continue
            score = 0.0
            if item["hitscan"]:
                score += 170
            if item["major"]:
                score += 150
            if item["melee"]:
                score += 90
            if item["distance"] < 120:
                score += 180
            elif item["distance"] < 260:
                score += 80
            score -= item["distance"] * 0.16
            score -= abs(item["error"]) * 0.08
            item["threat_score"] = score
            result.append(item)
        return sorted(result, key=lambda x: x["threat_score"], reverse=True)

    def target_from_landmark(self, obs, facts, tic):
        here = position(obs)
        facing = num((obs.get("player") or {}).get("angle"))
        visible_static = {x["id"]: x for x in facts if x["static"]}
        candidates = []
        health = num(obs.get("health"), 100)
        armor = num(obs.get("armor"), 0)
        ammo = num(obs.get("weapon_ammo"), 0)
        for ident, landmark in list(self.landmarks.items()):
            if landmark.get("claimed") or self.failed_targets.get(ident, -1) > tic:
                continue
            age = tic - integer(landmark.get("last"), tic)
            if age > 4500:
                continue
            point = tuple(num(x) for x in landmark.get("point", (0, 0, 0)))
            d = distance(here, point)
            if d > 2400:
                continue
            bearing = math.degrees(math.atan2(point[1] - here[1], point[0] - here[0]))
            error = wrap(bearing - facing)
            cat = str(landmark.get("category", "")).lower()
            name = str(landmark.get("name", "")).lower()
            if cat == "exit" or "exit" in name:
                base = 260.0
            elif cat in {"switch", "door", "teleporter", "teleport"} or any(x in name for x in ("switch", "door", "teleport")):
                base = 190.0
            elif landmark.get("category", "").lower() == "key" or "card" in name or "key" in name:
                base = 170.0
            elif cat == "weapon":
                base = 88.0 if ammo < 12 else 24.0
            elif cat == "health":
                base = 110.0 if health < 72 else 18.0
            elif cat == "armor":
                base = 90.0 if armor < 60 else 14.0
            elif cat == "ammo":
                base = 52.0 if ammo < 10 else 8.0
            else:
                continue
            if self.policy["priority"] == "survival" and cat in {"health", "armor"}:
                base += 28
            if self.policy["priority"] == "capability" and (cat == "weapon" or "card" in name or "key" in name):
                base += 25
            if self.policy["mode"] == "survey_first" and cat not in {"exit", "switch", "door", "teleporter", "teleport", "key"}:
                base -= 20
            base -= d * 0.035
            base -= min(30.0, age * 0.003)
            if ident == self.target_id:
                base += 16
            if ident in visible_static:
                base += 12
            candidates.append((base, ident, point, d, error, landmark))
        if not candidates:
            self.target_id = None
            self.target_close_since = None
            return None
        best = max(candidates, key=lambda x: x[0])
        if self.target_id != best[1]:
            self.target_id = best[1]
            self.target_started = tic
            self.target_best = best[3]
            self.target_close_since = None
            self.record("landmark_target", name=best[5].get("name"), category=best[5].get("category"), distance=round(best[3]))
        elif best[3] < self.target_best - 7:
            self.target_best = best[3]
            self.target_started = tic
        elif tic - self.target_started > 180 or tic - self.target_started > 0 and best[3] > self.target_best + 130:
            self.failed_targets[self.target_id] = tic + 700
            self.record("landmark_abandon", name=best[5].get("name"), distance=round(best[3]))
            self.target_id = None
            self.target_close_since = None
            return None
        item = dict(best[5])
        item.update({"point": list(best[2]), "distance": best[3], "error": best[4]})
        self.update_target_close(item, tic)
        return item

    def compact_fact(self, item):
        d = item["distance"]
        return {
            "id": item["id"],
            "name": item["name"],
            "category": item["category"],
            "range": "contact" if d < 70 else "near" if d < 220 else "medium" if d < 600 else "far",
            "direction": "center" if abs(item["error"]) < 14 else "left" if item["error"] > 0 else "right",
            "hitscan": bool(item.get("hitscan")),
            "major": bool(item.get("major")),
            "blocker": bool(item.get("melee")) and d < 150,
        }

    def decision_state(self, obs, facts, threats, target, tic):
        health = num(obs.get("health"), 100)
        armor = num(obs.get("armor"), 0)
        current_ammo = num(obs.get("weapon_ammo"), 0)
        return {
            "task": "Reach and activate the map exit alive; optional completion is secondary.",
            "rules": "Only rendered labels are observations. Code computes bearings, ranges, legality, collision and resources. Pick a lease, not a raw button or invented route.",
            "vital": {
                "health": "critical" if health < 25 else "wounded" if health < 60 else "healthy",
                "armor": "none" if armor < 10 else "present",
                "damage_recent": tic <= self.damage_until,
                "ammo": "empty" if current_ammo <= 0 else "limited" if current_ammo < 8 else "available",
            },
            "motion": {
                "phase": "escape" if self.escape else "navigate" if self.nav_lease else "scan" if self.scan_lease else "survey",
                "blocked": self.blocked >= 3,
                "stuck": self.still >= 24,
                "novel_cell": tic - self.last_new_tic < 28,
                "visited_cells": min(255, len(self.cells)),
                "local_rays_seen": min(12, self.rays.get((coarse_cell(position(obs)), angle_bin(num((obs.get("player") or {}).get("angle")))), 0)),
            },
            "threats": [self.compact_fact(x) for x in threats[:4]],
            "landmarks_visible": [self.compact_fact(x) for x in facts if x["static"]][:5],
            "target": (self.compact_fact(target) if target and "category" in target else None),
            "policy": dict(self.policy),
            "last_action": self.prev_action,
        }

    def questions(self, reverse=False):
        leases = {
            "scan": "No reliable target is established; use a short view scan and then commit to new translation.",
            "push": "Advance through unknown space; do not stop for optional distant pickups.",
            "landmark": "Approach the named visible or remembered key, weapon, health item, or interaction landmark.",
            "fight": "A visible threat is the immediate blocker; circle while firing only when code says the weapon is legal.",
            "disengage": "Recent damage, critical health, or a close threat makes a finite retreat safer than pursuit.",
            "interact": "A named door, switch, teleporter or exit is near enough that a bounded USE attempt is worthwhile.",
        }
        if reverse:
            leases = dict(reversed(list(leases.items())))
        return {
            "lease": Choice(
                "Choose the next high-level lease from the current `vital`, `motion`, `threats`, `landmarks_visible`, and `target`. Never invent a map route or a label that is absent.",
                leases,
            ),
            "side": Choice(
                "Choose the safer lateral side for a moving orbit or finite escape. If the facts do not distinguish sides, choose neutral.",
                {
                    "right": "Use the player's right side for a strafe/orbit.",
                    "neutral": "Do not change the existing side bias.",
                    "left": "Use the player's left side for a strafe/orbit.",
                },
            ),
            "use": Noul(
                "Do the named visible facts justify one USE pulse at the next safe boundary?",
                {"true": "A nearby rendered interaction or blocked-face evidence supports USE.",
                 "false": "No nearby interaction or blocked-face evidence supports USE."},
            ),
        }

    # ---------- asynchronous Jev ----------
    def dispatch_jev(self, obs, facts, threats, target, tic):
        if getattr(self.ctx, "system1", None) is None:
            return
        if self.jev_future is not None or tic < self.next_jev:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if remaining.get("system1_calls", 0) <= 6 or remaining.get("wall_s", 0) < 2:
            return
        state = self.decision_state(obs, facts, threats, target, tic)
        qs = self.questions(bool((tic // 20) % 2))
        self.jev_request = {"tic": tic, "state": state}
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, qs, timeout_s=0.85)
            )
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
            return
        self.next_jev = tic + (20 if threats else 48)

    def poll_jev(self, obs, tic):
        if self.jev_future is None or not self.jev_future.done():
            return
        future = self.jev_future
        request = self.jev_request
        self.jev_future = None
        self.jev_request = None
        try:
            answers = future.result()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, EpisodeOver, CancelledError):
            return
        except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
            self.record("jev_failed", error=type(exc).__name__)
            return
        if not request:
            return
        age = tic - integer(request.get("tic"), tic)
        if age > 70:
            self.record("jev_discarded", age=age)
            return
        lease = answer_choice(answers, "lease", {"scan", "push", "landmark", "fight", "disengage", "interact"}, 0.54)
        side = answer_choice(answers, "side", {"left", "right", "neutral"}, 0.50)
        if lease:
            self.mode_hint = lease
        if side in {"left", "right"}:
            self.side_hint = 1 if side == "left" else -1
        if answer_yes(answers, "use"):
            self.use_requested = True
        self.record("jev_lease", lease=lease, side=side, use=self.use_requested, age=age)

    # ---------- slow mission-card editor ----------
    def launch_astra(self, obs, facts, threats, target, tic):
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if remaining.get("system2_calls", 0) <= 0 or remaining.get("wall_s", 0) < 110:
            return
        state = self.decision_state(obs, facts, threats, target, tic)
        payload = {
            "current": dict(self.policy),
            "state": state,
            "ledger": {
                "visited_cells": len(self.cells),
                "landmarks": [{"name": x.get("name"), "category": x.get("category"), "age": tic - integer(x.get("last"), tic)} for x in list(self.landmarks.values())[-12:] if not x.get("claimed")],
                "failed_targets": len(self.failed_targets),
            },
            "recent_events": list(self.events),
        }
        prompt = (
            "You are the slow mission layer of a real-time Classic Doom controller. "
            "The code below owns exact movement, target bearings, combat safety, resources, "
            "and a persistent ledger containing only labels that were actually rendered. "
            "Do not propose buttons, coordinates, hidden geometry, or a seed-specific route. "
            "Write a compact mission card that changes priorities, not the motor. Exit and "
            "survival outrank optional kills/items. Return ONLY JSON with exactly these bounded "
            "fields: priority=exit|capability|survival|explore, "
            "mode=landmark_first|survey_first|survival, side=left|right, "
            "caution=normal|high, use=barrier|visible. Include a short reason. "
            "If evidence is insufficient, repeat current.\n" +
            json.dumps(payload, separators=(",", ":"), allow_nan=False)
        )
        self.astra_calls += 1
        session = self.astra_session
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, session_id=session, timeout_s=165.0)
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
            self.record("astra_failed", error=type(exc).__name__)
            return
        self.astra_session = getattr(result, "session_id", self.astra_session)
        proposal = json_from_text(getattr(result, "text", ""))
        changed = False
        for key, values in {
            "priority": {"exit", "capability", "survival", "explore"},
            "mode": {"landmark_first", "survey_first", "survival"},
            "side": {"left", "right"},
            "caution": {"normal", "high"},
            "use": {"barrier", "visible"},
        }.items():
            value = proposal.get(key)
            if value in values and value != self.policy[key]:
                self.policy[key] = value
                changed = True
        if changed:
            self.policy_version += 1
            self.explore_side = 1 if self.policy["side"] == "left" else -1
            self.record("astra_card", policy=dict(self.policy), reason=str(proposal.get("reason", ""))[:180])

    # ---------- exact action programs ----------
    def remember_failed_ray(self, obs):
        cell = coarse_cell(position(obs))
        facing = angle_bin(num((obs.get("player") or {}).get("angle")))
        self.failed_rays[(cell, facing)] += 1

    def start_escape(self, tic, obs, reason):
        if self.escape:
            return
        here = coarse_cell(position(obs))
        angle = num((obs.get("player") or {}).get("angle"))
        preferred = self.side_hint if self.side_hint in (-1, 1) else self.explore_side
        candidates = []
        for side in (preferred, -preferred):
            for magnitude in (2, 3, 1, 4):
                target_heading = angle_bin(angle + side * 45.0 * magnitude)
                key = (here, target_heading)
                score = self.rays.get(key, 0) + self.failed_rays.get(key, 0) * 4
                candidates.append((score, 0 if side == preferred else 1, magnitude, side, target_heading))
        _, _, magnitude, side, target_heading = min(candidates)
        self.explore_side = side
        turn_steps = max(8, min(28, int(round(45.0 * magnitude / max(2.0, self.turn_rate)))))
        turn_action = self.turn(side)
        escape = [self.legal("backward")] * 6
        escape += [turn_action] * turn_steps
        escape += [self.side_move(side)] * 20
        escape += [self.legal("forward")] * 12
        self.escape.extend(escape)
        self.blocked = 0
        self.still = 0
        self.failed_targets.pop(self.target_id, None) if self.target_id in self.failed_targets else None
        self.target_id = None
        self.target_close_since = None
        self.remember_failed_ray(obs)
        self.record("escape_program", tic=tic, reason=reason, side=side, heading=target_heading, length=len(escape))

    def weapon_action(self, obs, tic):
        if tic - self.last_weapon < 35:
            return None
        weapons = obs.get("weapons") if isinstance(obs.get("weapons"), dict) else {}
        ammo = obs.get("ammo") if isinstance(obs.get("ammo"), dict) else {}
        current = integer(obs.get("weapon"), 2)
        current_ammo = num(obs.get("weapon_ammo"), 0)
        desired = None
        hint = self.weapon_hint if isinstance(self.weapon_hint, int) else None
        order = [hint] if hint is not None else []
        order += [3, 4, 6, 2, 1]
        for slot in order:
            if slot is None or not bool(weapons.get(str(slot), False)):
                continue
            available = num(ammo.get(str(slot)), 0)
            if slot == current:
                available = max(available, current_ammo)
            # Deliberately do not select rocket/BFG merely from an unseen target.
            if available > 0 and slot not in {5, 7}:
                desired = slot
                break
        if desired is None or desired == current or current_ammo > 0:
            return None
        action = "select_weapon_" + str(desired)
        if action in self.allowed:
            self.last_weapon = tic
            return action
        return None

    def combat(self, obs, threat, tic):
        switch = self.weapon_action(obs, tic)
        if switch:
            return switch
        error = threat["error"]
        side = self.side_hint if self.side_hint in (-1, 1) else self.explore_side
        if self.orbit:
            return self.orbit.popleft()
        if abs(error) > max(9.0, self.turn_rate * 2.0):
            self.aim_streak += 1
            # A turn-only lease is bounded.  After three correction tics the
            # player translates, so aim error cannot create an absorbing trap.
            if self.aim_streak <= 3:
                return self.turn(1 if error > 0 else -1)
            self.aim_streak = 0
            self.orbit.extend([self.side_move(side)] * 4)
            return self.orbit.popleft()
        self.aim_streak = 0
        health = num(obs.get("health"), 100)
        ammo = num(obs.get("weapon_ammo"), 0)
        if threat["distance"] < 125 and (threat["melee"] or health < 25 or tic <= self.damage_until):
            return self.side_move(side, backward=True)
        if ammo <= 0:
            return self.side_move(side, backward=True)
        return self.side_move(side, firing=True)

    def steer_landmark(self, target, tic):
        error = num(target.get("error"), 0)
        d = num(target.get("distance"), 9999)
        if self.nav_lease:
            return self.nav_lease.popleft()
        if d < 58:
            cat = str(target.get("category", "")).lower()
            name = str(target.get("name", "")).lower()
            interact = cat in {"exit", "switch", "door", "teleporter", "teleport"} or any(x in name for x in ("exit", "switch", "door", "teleport"))
            if interact:
                if tic - self.last_use >= (8 if self.policy["caution"] == "high" else 12):
                    self.last_use = tic
                    self.use_requested = False
                    return self.legal("use")
                return self.legal("forward")
            # A pickup should be crossed rather than circled indefinitely.
            if self.target_close_since is not None and tic - self.target_close_since > 42:
                self.landmarks.get(self.target_id, {}).update({"claimed": True})
                self.failed_targets[self.target_id] = tic + 450
                self.target_id = None
            return self.legal("forward")
        if abs(error) > 32:
            steps = min(5, max(2, int(abs(error) / max(3.0, self.turn_rate))))
            self.nav_lease.extend([self.turn(1 if error > 0 else -1)] * steps)
            return self.nav_lease.popleft()
        # Translation leases are the key difference from a per-observation
        # target controller: the bearing is corrected only at lease boundaries.
        side = 1 if error > 5 else -1 if error < -5 else self.explore_side
        length = 10 if d < 180 else 18 if d < 600 else 28
        self.nav_lease.extend([self.side_move(side)] * length)
        return self.nav_lease.popleft()

    def scan_or_survey(self, obs, tic):
        if self.scan_lease:
            return self.scan_lease.popleft()
        if self.nav_lease:
            return self.nav_lease.popleft()
        if tic - self.last_new_tic > 92 or self.blocked >= 4 or self.still >= 28:
            self.start_escape(tic, obs, "no_novel_space" if tic - self.last_new_tic > 92 else "blocked")
            return self.escape.popleft() if self.escape else self.legal("backward")
        if self.use_requested and tic - self.last_use > 12 and not self.policy["caution"] == "high":
            self.use_requested = False
            self.last_use = tic
            return self.legal("use")
        if self.scan_lease:
            return self.scan_lease.popleft()
        # Survey is a bounded forward run followed by a short view rotation.
        if not hasattr(self, "survey_left"):
            self.survey_left = 34
        if self.survey_left > 0:
            self.survey_left -= 1
            return self.legal("forward")
        self.survey_left = 34 if self.policy["mode"] != "survey_first" else 24
        side = self.side_hint if self.side_hint in (-1, 1) else self.explore_side
        self.explore_side = side
        self.scan_lease.extend([self.turn(side)] * 11)
        return self.scan_lease.popleft()

    def choose_action(self, obs, facts, threats, tic):
        close_projectile = next((x for x in facts if x["projectile"] and x["distance"] < 220), None)
        if close_projectile is not None:
            side = -1 if close_projectile["error"] > 0 else 1
            self.explore_side = side
            return self.side_move(side)
        threat = threats[0] if threats else None
        immediate = threat is not None and (
            threat["major"] or threat["distance"] < 285 or
            (threat["hitscan"] and threat["distance"] < 700) or tic <= self.damage_until
        )
        if immediate:
            self.target_id = None
            return self.combat(obs, threat, tic)
        if self.escape:
            return self.escape.popleft()
        target = self.target_from_landmark(obs, facts, tic)
        if target is not None and self.mode_hint not in {"scan", "push"}:
            return self.steer_landmark(target, tic)
        if self.use_requested and tic - self.last_use > 12:
            self.use_requested = False
            self.last_use = tic
            return self.legal("use")
        if self.mode_hint == "disengage":
            self.start_escape(tic, obs, "jev_disengage")
            return self.escape.popleft() if self.escape else self.legal("backward")
        if self.blocked >= 4:
            face = (coarse_cell(position(obs)), angle_bin(num((obs.get("player") or {}).get("angle"))))
            if self.face_use.get(face, -1) < tic and tic - self.last_use > 15:
                self.face_use[face] = tic + 180
                self.last_use = tic
                return self.legal("use")
            self.start_escape(tic, obs, "collision")
            return self.escape.popleft() if self.escape else self.legal("backward")
        if tic - self.last_use > 93 and not threats and self.policy["use"] == "barrier" and self.blocked >= 2:
            self.last_use = tic
            return self.legal("use")
        return self.scan_or_survey(obs, tic)

    # ---------- lifecycle ----------
    def run_episode(self, game):
        spec = getattr(game, "action_spec", None)
        names = getattr(spec, "names", ()) or ()
        self.allowed = set(names)
        if not self.allowed:
            return
        self.noop = getattr(spec, "noop", "noop")
        if self.noop not in self.allowed:
            self.noop = next(iter(self.allowed))
        try:
            while not self.ctx.done and not getattr(game, "done", False):
                obs = parse_obs(self.ctx.observe())
                if self.ctx.done or getattr(game, "done", False):
                    break
                if obs.get("terminal") or obs.get("dead"):
                    break
                tic, fresh = self.update(obs)
                if not fresh:
                    self.ctx.wait_for_tick()
                    continue
                facts = label_facts(obs)
                threats = self.threats(facts)
                target = self.target_from_landmark(obs, facts, tic)
                self.poll_jev(obs, tic)
                self.poll_astra(tic)
                self.dispatch_jev(obs, facts, threats, target, tic)
                if self.astra_future is None and tic >= self.next_astra:
                    self.launch_astra(obs, facts, threats, target, tic)
                    self.next_astra = tic + 6200
                action = self.legal(self.choose_action(obs, facts, threats, tic))
                self.ctx.act(action)
                self.prev_action = action
                self.ctx.wait_for_tick()
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
                self.ctx.memory["expedition_summary"] = {
                    "visited_cells": len(self.cells),
                    "landmarks": len(self.landmarks),
                    "failed_targets": len(self.failed_targets),
                    "policy": dict(self.policy),
                    "events": list(self.events),
                }
            except Exception:
                pass


def build(ctx):
    return Expedition(ctx)
