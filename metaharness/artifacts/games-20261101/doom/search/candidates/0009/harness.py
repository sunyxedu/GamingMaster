"""Candidate 0009: frontier graph search with a continuously refreshed motor.

This controller is deliberately different from a periodic sweep.  It turns
measured collisions into directed, local heading edges, remembers which edges
have been tried at each coarse position, and walks back along observed
breadcrumbs when a local node is exhausted.  Jev and Astra may choose semantic
preferences, but neither is on the real-time motor path.
"""
from __future__ import annotations

import collections
import json
import math
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


TRANSLATION = {
    "forward", "backward", "strafe_left", "strafe_right",
    "forward_left", "forward_right", "backward_left", "backward_right",
    "fire_forward", "fire_strafe_left", "fire_strafe_right",
}
TURNING = {"turn_left", "turn_right"}

# Names are used only after the public label category identifies a Monster.
HITSCAN = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS",
           "SpiderMastermind"}
MAJOR = {"Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind",
         "Revenant"}
MELEE = {"Demon", "Spectre", "LostSoul", "Pinkie", "Pinky"}


def num(value, default=0.0):
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def integer(value, default=0):
    try:
        return int(num(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def finite(value, default=0.0):
    value = num(value, default)
    return value if math.isfinite(value) else default


def wrap(angle):
    return (finite(angle) + 180.0) % 360.0 - 180.0


def point_distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.sqrt(sum((finite(x) - finite(y)) ** 2 for x, y in zip(a, b)))


def xy_distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.hypot(finite(a[0]) - finite(b[0]),
                     finite(a[1]) - finite(b[1]))


def position(obs):
    player = obs.get("player") if isinstance(obs, dict) else None
    if not isinstance(player, dict):
        return None
    values = tuple(finite(player.get(k), 0.0) for k in ("x", "y", "z"))
    return values if all(math.isfinite(x) for x in values) else None


def heading_bin(angle):
    return int(math.floor((finite(angle) % 360.0 + 22.5) / 45.0)) % 8


def node_of(p):
    """A deliberately coarse memory of observed space, not a map oracle."""
    if p is None:
        return (0, 0, 0)
    return (math.floor(finite(p[0]) / 96.0),
            math.floor(finite(p[1]) / 96.0),
            math.floor(finite(p[2]) / 64.0))


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


def extract_object(text):
    """Extract the first valid JSON object from an unconstrained Astra reply."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for index, char in enumerate(text[:16000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        return value if isinstance(value, dict) else {}
    return {}


def choice_value(answers, key, allowed, threshold=0.56):
    if not isinstance(answers, dict):
        return None
    answer = answers.get(key)
    value = getattr(answer, "value", None)
    if not isinstance(value, str) or value not in allowed:
        return None
    probabilities = getattr(answer, "probabilities", None)
    if isinstance(probabilities, dict) and value in probabilities:
        if num(probabilities.get(value), 0.0) < threshold:
            return None
    confidence = getattr(answer, "confidence", None)
    if confidence is not None and num(confidence, 0.0) < 0.06:
        return None
    return value


def yes_value(answers, key, threshold=0.72):
    if not isinstance(answers, dict):
        return False
    answer = answers.get(key)
    value = getattr(answer, "value", None)
    if isinstance(value, bool):
        return value
    return num(value, 0.0) >= threshold


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.noop = "noop"

        # Observation and motion bookkeeping.
        self.last_tic = None
        self.last_position = None
        self.last_angle = None
        self.last_action = "noop"
        self.last_health = 100.0
        self.last_damage_counter = 0.0
        self.damage_until = -1
        self.damage_amount = 0.0
        self.still_ticks = 0
        self.last_progress_tic = 0
        self.last_counts = (0, 0, 0)
        self.current_node = None
        self.last_node = None
        self.current_obs = {}
        self.objects = []
        self.threats = []
        self.projectiles = []

        # Frontier graph.  A node is a position bin actually visited by the
        # player.  An attempted heading is never inferred from unseen space.
        self.node_positions = {}
        self.node_heading = {}
        self.attempted = collections.defaultdict(set)
        self.node_visits = collections.Counter()
        self.node_path = []
        self.trail = collections.deque(maxlen=320)
        self.frontier_events = 0
        self.collision_count = 0
        self.heading_sign = 1.0
        self.turn_rate = 5.625
        self.visited_cells = collections.Counter()

        # Finite route motor.
        self.phase = "travel"       # travel | turn | probe | backtrack
        self.macro_left = 0
        self.probe_left = 0
        self.probe_node = None
        self.probe_heading = None
        self.probe_origin = None
        self.turn_target = None
        self.backtrack_target = None
        self.backtrack_node = None
        self.backtrack_deadline = -1
        self.branch_hint = None
        self.fire_side = 1
        self.retreat_until = -1
        self.retreat_target = None

        # Interaction and capability goals are bounded and label-backed.
        self.goal_id = None
        self.goal_best = float("inf")
        self.goal_started = -1
        self.goal_progress = -1
        self.goal_defer = {}
        self.last_use_tic = -10000
        self.probed_faces = {}
        self.force_use = False
        self.last_weapon_tic = -10000

        # Astra policy is advisory and finite.
        self.policy = {
            "hand": "right",
            "combat": "cautious",
            "probe": "normal",
            "resources": "capability",
        }
        self.policy_version = 0
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.astra_due = 180

        # Jev is a short-lived classifier, never the action clock.
        self.jev_future = None
        self.jev_meta = None
        self.jev_next = 0
        self.jev_mode = None
        self.jev_mode_until = -1
        self.jev_retreat = False
        self.jev_target = None
        self.jev_request = 0
        self.history = collections.deque(maxlen=24)

    # ---------- safe public-state conversion ----------
    def legal(self, action):
        if action in self.allowed:
            return action
        if self.noop in self.allowed:
            return self.noop
        return next(iter(self.allowed), self.noop)

    def _parse_objects(self, obs, tic):
        p = position(obs)
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
            q = tuple(finite(x) for x in world[:3])
            name = str(raw.get("name", raw.get("object_name", "unknown")))[:80]
            category = str(raw.get("category", raw.get("object_category", "Unknown")))[:40]
            lower_name = name.lower()
            lower_category = category.lower()
            if lower_category == "self" or name == "DoomPlayer":
                continue
            d = point_distance(p, q)
            bearing = math.degrees(math.atan2(q[1] - (p[1] if p else 0.0),
                                              q[0] - (p[0] if p else 0.0))) % 360.0
            velocity = raw.get("velocity")
            if not isinstance(velocity, (list, tuple)) or len(velocity) < 3:
                velocity = (0.0, 0.0, 0.0)
            speed = math.sqrt(sum(finite(x) ** 2 for x in velocity[:3]))
            bbox = raw.get("bbox")
            area = 0.0
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                area = max(0.0, finite(bbox[2])) * max(0.0, finite(bbox[3]))
            monster = lower_category == "monster" and not lower_name.startswith("dead")
            projectile = (
                "projectile" in lower_category or "missile" in lower_category
                or lower_name in {"doomimpball", "cacodemonball", "plasmaball",
                                  "rocket", "fireball"}
            )
            interact = (
                lower_category in {"exit", "switch", "door", "teleporter", "teleport"}
                or any(token in lower_name for token in ("exit", "switch", "door", "teleport"))
            )
            key = lower_category == "key" or "key" in lower_name
            pickup = lower_category in {
                "key", "weapon", "health", "armor", "ammo", "powerup", "power-up"
            }
            entry = {
                "id": str(raw.get("id", raw.get("object_id", name))),
                "name": name,
                "category": category,
                "pos": q,
                "distance": d,
                "relative": wrap(bearing - angle),
                "height": q[2] - (p[2] if p else 0.0),
                "speed": speed,
                "area": area,
                "monster": monster,
                "projectile": projectile,
                "interact": interact,
                "key": key,
                "pickup": pickup,
            }
            if monster:
                entry["hitscan"] = name in HITSCAN
                entry["major"] = name in MAJOR
                entry["melee"] = name in MELEE
                active = speed > 0.15 or d < 270.0 or tic <= self.damage_until
                entry["active"] = active
                entry["threat_score"] = (
                    (190.0 if entry["major"] else 0.0)
                    + (115.0 if entry["hitscan"] else 0.0)
                    + (105.0 if entry["melee"] and d < 180.0 else 0.0)
                    + (150.0 if d < 105.0 else 65.0 if d < 240.0 else 0.0)
                    + (60.0 if active else 0.0)
                    + min(50.0, area / 20.0)
                    - min(170.0, d * 0.12)
                )
            objects.append(entry)
        self.objects = objects
        self.threats = sorted(
            [x for x in objects if x.get("monster")],
            key=lambda x: num(x.get("threat_score"), -1e9), reverse=True,
        )
        self.projectiles = sorted(
            [x for x in objects if x.get("projectile")],
            key=lambda x: num(x.get("distance"), 1e9),
        )

    def update(self, raw):
        obs = as_dict(raw)
        if not obs:
            return False
        tic = integer(obs.get("tic"), integer(obs.get("elapsed_tics"), 0))
        if self.last_tic is not None and tic <= self.last_tic:
            return False
        p = position(obs)
        angle = finite((obs.get("player") or {}).get("angle"), 0.0)
        dt = 1 if self.last_tic is None else max(1, tic - self.last_tic)

        if self.last_angle is not None and self.last_action in TURNING:
            delta = wrap(angle - self.last_angle)
            if abs(delta) > 0.2 and dt <= 14:
                rate = abs(delta) / float(dt)
                if 0.2 < rate < 25.0:
                    self.turn_rate = 0.78 * self.turn_rate + 0.22 * rate
                    if self.last_action == "turn_left":
                        self.heading_sign = 1.0 if delta > 0 else -1.0
                    else:
                        self.heading_sign = -1.0 if delta > 0 else 1.0

        moved = (0.0 if self.last_position is None or p is None
                 else point_distance(self.last_position, p))
        if self.last_position is not None and self.last_action in TRANSLATION:
            if moved < max(1.5, 0.35 * dt):
                self.still_ticks += dt
            else:
                self.still_ticks = 0
        else:
            self.still_ticks = 0

        health = num(obs.get("health"), self.last_health)
        damage_counter = num(obs.get("damage_taken"), self.last_damage_counter)
        loss = max(0.0, self.last_health - health)
        counter_loss = max(0.0, damage_counter - self.last_damage_counter)
        self.damage_amount = max(loss, counter_loss)
        if self.damage_amount >= 1.0:
            self.damage_until = max(self.damage_until, tic + 38)
            self.fire_side *= -1
            if self.damage_amount >= 8.0:
                self.retreat_until = max(self.retreat_until, tic + 70)
                self.retreat_target = None
        if self.phase == "travel":
            self.macro_left -= dt
        elif self.phase == "probe":
            self.probe_left -= dt
            # A probe has found usable space even when it has not yet crossed
            # the deliberately coarse 96-unit node boundary.
            if self.probe_origin is not None and moved >= 20.0:
                self._attempts(self.probe_node or self.current_node).add(
                    integer(self.probe_heading, 0) % 8
                )
                self.phase = "travel"
                self.macro_left = self.segment_length()
                self.probe_left = 0
                self.probe_node = None
                self.probe_origin = None

        node = node_of(p)
        was_backtrack = self.phase == "backtrack"
        if self.current_node is None:
            self.current_node = node
            self.node_path = [node]
            self.node_positions[node] = p
            self.node_heading[node] = heading_bin(angle)
        elif node != self.current_node:
            old_node = self.current_node
            old_heading = heading_bin(self.last_angle if self.last_angle is not None else angle)
            # The traversed heading is a discovered edge.  Do not immediately
            # choose the reverse corridor as a fresh branch at the new node.
            self.node_heading[old_node] = old_heading
            self.node_positions[node] = p
            self.node_heading[node] = heading_bin(angle)
            self._attempts(old_node).add(old_heading)
            self._attempts(node).add((old_heading + 4) % 8)
            self.current_node = node
            self.node_path.append(node)
            self.last_progress_tic = tic
            self.still_ticks = 0
            self.macro_left = max(self.macro_left, self.segment_length())
            reached_backtrack = was_backtrack and node == self.backtrack_node
            if not was_backtrack or reached_backtrack:
                self.phase = "travel"
                self.backtrack_target = None
                self.backtrack_node = None
            else:
                # Passing through an intermediate breadcrumb node should not
                # cancel the return to the selected frontier.
                self.phase = "backtrack"
            self.probe_left = 0
            self.probe_node = None
            self.probe_origin = None
            self.history.append({"t": tic, "event": "new_node", "node": list(node)})

        if moved >= 2.5:
            self.last_progress_tic = tic
        if self.last_position is None or point_distance(self.last_position, p) >= 24.0:
            self.trail.append((tic, p, node, health))
        self.visited_cells[node] += 1
        self.node_visits[node] += 1
        counts = tuple(integer(obs.get(k), 0) for k in ("kills", "items", "secrets"))
        if counts != self.last_counts:
            self.last_progress_tic = tic

        self.current_obs = obs
        self._parse_objects(obs, tic)
        self.last_tic = tic
        self.last_position = p
        self.last_angle = angle
        self.last_health = health
        self.last_damage_counter = damage_counter
        self.last_counts = counts
        return True

    def node_attempts(self, node):
        return self.attempted[node]

    def attempts_for(self, node):
        return self.node_attempts(node)

    def segment_length(self):
        if self.policy.get("probe") == "rare":
            return 62
        return 42 if self.policy.get("resources") == "keys_only" else 34

    def _attempts(self, node):
        return self.attempted[node]

    # ---------- route frontier ----------
    def candidate_heading(self, node, base=None):
        if base is None:
            base = self.node_heading.get(node, heading_bin(self.last_angle))
        base = integer(base) % 8
        hand = 1 if self.policy.get("hand") == "right" else -1
        hint = self.branch_hint
        if hint == "left":
            offsets = [2 * -hand, 1 * -hand, 3 * -hand, 2 * hand,
                       1 * hand, 4, 3 * hand, -3 * hand]
        elif hint == "right":
            offsets = [2 * hand, 1 * hand, 3 * hand, 2 * -hand,
                       1 * -hand, 4, -3 * hand, 3 * hand]
        elif hint == "reverse":
            offsets = [4, 3 * hand, 3 * -hand, 2 * hand,
                       2 * -hand, 1 * hand, 1 * -hand, 0]
        else:
            offsets = [2 * hand, 2 * -hand, 4, 1 * hand,
                       1 * -hand, 3 * hand, 3 * -hand, 0]
        tried = self._attempts(node)
        # Start each newly discovered node by continuing through the current
        # opening.  Once that edge has been sampled, the side/reverse order
        # supplies the frontier-search behavior.
        if self.node_visits.get(node, 0) <= 1 and 0 not in tried:
            offsets = [0] + offsets
        for offset in offsets:
            h = (base + offset) % 8
            if h not in tried:
                return h
        return None

    def start_probe(self, node, h, tic, reason):
        if node is None:
            node = self.current_node
        if node is None:
            return
        h = integer(h) % 8
        self._attempts(node).add(h)
        self.probe_node = node
        self.probe_heading = h
        self.probe_origin = self.last_position
        self.probe_left = 30 if self.policy.get("probe") != "rare" else 22
        self.turn_target = h * 45.0
        self.phase = "turn"
        self.macro_left = 0
        self.branch_hint = None
        self.frontier_events += 1
        self.ctx.log("frontier_probe", node=list(node), heading=h, reason=reason,
                     event_number=self.frontier_events)

    def _choose_backtrack(self, tic):
        if self.current_node is None:
            return False
        # Search backward through the observed route, not through an imagined
        # map.  Prefer the newest earlier node with an untried direction.
        seen = set()
        for index in range(len(self.node_path) - 2, -1, -1):
            node = self.node_path[index]
            if node in seen or node == self.current_node:
                continue
            seen.add(node)
            base = self.node_heading.get(node, heading_bin(self.last_angle))
            if self.candidate_heading(node, base) is None:
                continue
            target = self.node_positions.get(node)
            if target is None:
                continue
            self.phase = "backtrack"
            self.backtrack_target = target
            self.backtrack_node = node
            self.backtrack_deadline = tic + 125
            self.ctx.log("frontier_backtrack", node=list(node), depth=index,
                         frontier=len(self.frontier_nodes()))
            return True
        return False

    def frontier_nodes(self):
        result = []
        for node in self.node_positions:
            if self.candidate_heading(node, self.node_heading.get(node)) is not None:
                result.append(node)
        return result

    def handle_block(self, tic):
        node = self.current_node or node_of(self.last_position)
        h = heading_bin(self.last_angle)
        self._attempts(node).add(h)
        self.collision_count += 1
        self.history.append({"t": tic, "event": "collision", "node": list(node), "heading": h})
        # Give a newly blocked face one USE pulse.  The cooldown is keyed by
        # observed node+heading, so this cannot turn into the old USE flood.
        face = (node, h)
        if self.probed_faces.get(face, -1) <= tic and tic - self.last_use_tic >= 15:
            self.probed_faces[face] = tic + 180
            self.last_use_tic = tic
            if self.phase != "backtrack":
                self.phase = "travel"
            self.still_ticks = 0
            self.ctx.log("frontier_use_probe", node=list(node), heading=h)
            return self.legal("use")
        next_h = self.candidate_heading(node, h)
        if next_h is not None:
            self.start_probe(node, next_h, tic, "collision")
            self.still_ticks = 0
            return self.route_action(tic)
        if self._choose_backtrack(tic):
            self.still_ticks = 0
            return self.route_action(tic)
        # A fully sampled local node is not a reason to spin forever.  Change
        # heading once and keep a moving fallback even if all known frontier
        # nodes have been exhausted.
        self.phase = "travel"
        self.macro_left = 55
        self.still_ticks = 0
        self.turn_target = (h * 45.0 + 180.0) % 360.0
        self.phase = "turn"
        self.probe_left = 20
        self.probe_node = node
        self.probe_heading = (h + 4) % 8
        self.probe_origin = self.last_position
        self._attempts(node).add((h + 4) % 8)
        self.ctx.log("frontier_exhausted", node=list(node), heading=(h + 4) % 8)
        return self.route_action(tic)

    # ---------- labels, goals and combat ----------
    def is_interaction(self, item):
        return bool(item.get("interact"))

    def _goal_item(self, obs, tic):
        visible = [x for x in self.objects
                   if abs(num(x.get("height"), 9999.0)) <= 60.0]
        interactions = [x for x in visible if self.is_interaction(x)
                        and num(x.get("distance"), 9999.0) <= 420.0]
        if interactions:
            interactions.sort(key=lambda x: (
                0 if x.get("category", "").lower() == "exit"
                or "exit" in x.get("name", "").lower() else 1,
                num(x.get("distance"), 9999.0)))
            return interactions[0]

        health = num(obs.get("health"), 100.0)
        armor = num(obs.get("armor"), 0.0)
        weapons = obs.get("weapons") if isinstance(obs.get("weapons"), dict) else {}
        if self.policy.get("resources") == "keys_only":
            permitted = {"key"}
        else:
            permitted = {"key", "weapon", "health", "armor"}
        candidates = []
        for item in visible:
            category = item.get("category", "").lower()
            if category not in permitted and not item.get("key"):
                continue
            d = num(item.get("distance"), 9999.0)
            if d > 280.0:
                continue
            name = item.get("name", "")
            useful = False
            priority = 0.0
            if item.get("key") or category == "key":
                useful, priority = True, 900.0
            elif category == "weapon":
                # A visible weapon is a capability target only if its common
                # slot is not already owned; unknown names are not chased.
                slots = {"shotgun": 3, "supershotgun": 3, "chaingun": 4,
                         "rocketlauncher": 5, "plasmarifle": 6,
                         "bfg9000": 7, "chainsaw": 1}
                slot = slots.get(name.lower())
                owned = bool(weapons.get(str(slot))) if slot is not None else True
                useful, priority = slot is not None and not owned, 520.0
            elif category == "health" and health < 55.0:
                useful, priority = True, 500.0 if health < 30 else 270.0
            elif category == "armor" and armor < 30.0:
                useful, priority = True, 430.0
            if useful:
                candidates.append((priority - d * 0.55, item))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates[0][1]

    def current_goal(self, obs, tic):
        item = None
        if self.goal_id is not None:
            item = next((x for x in self.objects if x.get("id") == self.goal_id), None)
            if item is None:
                if tic - self.goal_started > 35:
                    self.goal_id = None
                else:
                    return None
            else:
                d = num(item.get("distance"), 9999.0)
                if d < self.goal_best - 8.0:
                    self.goal_best = d
                    self.goal_progress = tic
                elif tic - self.goal_progress > 72:
                    self.goal_defer[self.goal_id] = tic + 360
                    self.goal_id = None
                    item = None
                if item is not None:
                    return item
        candidate = self._goal_item(obs, tic)
        if candidate is None:
            return None
        ident = candidate.get("id")
        if self.goal_defer.get(ident, -1) > tic:
            return None
        self.goal_id = ident
        self.goal_best = num(candidate.get("distance"), 9999.0)
        self.goal_started = tic
        self.goal_progress = tic
        self.ctx.log("frontier_goal", name=candidate.get("name", ""),
                     category=candidate.get("category", ""),
                     distance=round(self.goal_best))
        return candidate

    def _turn_for(self, difference):
        # heading_sign is learned from observed angle changes.
        if difference * self.heading_sign > 0:
            return self.legal("turn_left")
        return self.legal("turn_right")

    def _steer_to(self, target):
        if target is None or self.last_position is None:
            return self.legal("forward")
        angle = finite(self.last_angle)
        desired = math.degrees(math.atan2(target[1] - self.last_position[1],
                                          target[0] - self.last_position[0])) % 360.0
        diff = wrap(desired - angle)
        if abs(diff) > 12.0:
            return self._turn_for(diff)
        return self.legal("forward")

    def _safe_point(self):
        p = self.last_position
        if p is None:
            return None
        for tic, q, node, health in reversed(self.trail):
            if self.last_tic - tic >= 18 and xy_distance(p, q) >= 110.0:
                return q
        return None

    def _weapon_action(self, obs, tic):
        if tic - self.last_weapon_tic < 30:
            return None
        if num(obs.get("weapon_ammo"), 0.0) > 0:
            return None
        weapons = obs.get("weapons") if isinstance(obs.get("weapons"), dict) else {}
        ammo = obs.get("ammo") if isinstance(obs.get("ammo"), dict) else {}
        current = integer(obs.get("weapon"), -1)
        for slot in (3, 4, 6, 2, 1):
            action = "select_weapon_" + str(slot)
            if action not in self.allowed:
                continue
            owned = bool(weapons.get(str(slot))) or bool(weapons.get(str(slot - 1)))
            available = max(num(ammo.get(str(slot))), num(ammo.get(str(slot - 1))))
            if owned and available > 0 and slot != current:
                self.last_weapon_tic = tic
                return action
        return None

    def _retreat_action(self, tic):
        if self.retreat_target is None:
            self.retreat_target = self._safe_point()
        if self.retreat_target is not None:
            if xy_distance(self.last_position, self.retreat_target) < 48.0:
                self.retreat_target = None
            else:
                return self._steer_to(self.retreat_target)
        return self.legal("backward_left" if self.fire_side > 0 else "backward_right")

    def combat_action(self, obs, threat, tic):
        health = num(obs.get("health"), 100.0)
        distance = num(threat.get("distance"), 9999.0)
        relative = num(threat.get("relative"), 0.0)
        if health < 28.0 or (distance < 85.0 and threat.get("melee")):
            self.retreat_until = max(self.retreat_until, tic + 45)
            return self._retreat_action(tic)
        weapon = self._weapon_action(obs, tic)
        if weapon:
            return weapon
        if abs(relative) > max(9.0, self.turn_rate * 1.2):
            return self._turn_for(relative)
        if num(obs.get("weapon_ammo"), 0.0) <= 0:
            return self.legal("backward_left" if self.fire_side > 0 else "backward_right")
        if distance < 145.0 and self.damage_until >= tic:
            return self.legal("backward_left" if self.fire_side > 0 else "backward_right")
        action = "fire_strafe_left" if self.fire_side > 0 else "fire_strafe_right"
        if action not in self.allowed:
            action = "fire_forward" if "fire_forward" in self.allowed else "fire"
        return self.legal(action)

    def urgent_threat(self, tic):
        if self.projectiles and num(self.projectiles[0].get("distance"), 9999.0) < 185.0:
            return self.threats[0] if self.threats else None
        if not self.threats:
            return None
        target = self.threats[0]
        d = num(target.get("distance"), 9999.0)
        if target.get("major") or target.get("hitscan") and d < 560.0:
            return target
        if target.get("active") and d < 270.0:
            return target
        if d < 125.0:
            return target
        if self.jev_mode == "fight" and d < 330.0:
            return target
        if tic <= self.damage_until and d < 380.0:
            return target
        return None

    def goal_action(self, item, tic):
        d = num(item.get("distance"), 9999.0)
        if d < 58.0 and self.is_interaction(item) and tic - self.last_use_tic >= 14:
            self.last_use_tic = tic
            self.force_use = False
            return self.legal("use")
        if d < 42.0 and not self.is_interaction(item):
            return self.legal("forward")
        if abs(num(item.get("relative"), 0.0)) > 13.0:
            return self._turn_for(num(item.get("relative"), 0.0))
        return self.legal("forward")

    # ---------- route action ----------
    def route_action(self, tic):
        if self.phase == "turn":
            diff = wrap(num(self.turn_target) - finite(self.last_angle))
            if abs(diff) <= 9.0:
                self.phase = "probe"
                self.probe_left = max(1, self.probe_left)
                return self.legal("forward")
            return self._turn_for(diff)
        if self.phase == "probe":
            if self.probe_left <= 0:
                node = self.probe_node or self.current_node
                next_h = self.candidate_heading(node, self.probe_heading)
                self.phase = "travel"
                self.still_ticks = 0
                if next_h is not None:
                    self.start_probe(node, next_h, tic, "failed_probe")
                    return self.route_action(tic)
                if self._choose_backtrack(tic):
                    return self.route_action(tic)
                self.macro_left = 25
                return self.legal("backward")
            return self.legal("forward")
        if self.phase == "backtrack":
            if self.backtrack_target is None or tic > self.backtrack_deadline:
                self.phase = "travel"
                self.macro_left = 0
                self.backtrack_target = None
                return self.legal("backward")
            if xy_distance(self.last_position, self.backtrack_target) < 46.0:
                target_node = self.backtrack_node
                self.phase = "travel"
                self.macro_left = 0
                self.backtrack_target = None
                self.backtrack_node = None
                if target_node in self.node_path:
                    self.node_path = self.node_path[:self.node_path.index(target_node) + 1]
                return self.legal("forward")
            return self._steer_to(self.backtrack_target)

        if self.macro_left <= 0:
            node = self.current_node or node_of(self.last_position)
            h = self.candidate_heading(node, heading_bin(self.last_angle))
            if h is not None:
                self.start_probe(node, h, tic, "segment_boundary")
                return self.route_action(tic)
            if self._choose_backtrack(tic):
                return self.route_action(tic)
            self.macro_left = 40
        self.macro_left -= 1
        return self.legal("forward")

    def choose_action(self, obs, tic):
        # Emergency layers are deterministic and outrank Jev/Astra.
        projectile = self.projectiles[0] if self.projectiles else None
        if projectile is not None and num(projectile.get("distance"), 9999.0) < 170.0:
            self.fire_side *= -1
            action = "strafe_left" if self.fire_side > 0 else "strafe_right"
            return self.legal(action)

        urgent = self.urgent_threat(tic)
        health = num(obs.get("health"), 100.0)
        if (health < 35.0 and (tic <= self.damage_until or self.damage_amount >= 5.0)):
            self.retreat_until = max(self.retreat_until, tic + 50)
        if self.retreat_until >= tic:
            return self._retreat_action(tic)
        if urgent is not None:
            return self.combat_action(obs, urgent, tic)

        # A rendered interaction or a bounded capability goal outranks search.
        # A Jev USE vote is deliberately gated by the same visible/blocked
        # evidence; it cannot pulse USE merely because a model said "yes".
        goal = self.current_goal(obs, tic)
        use_supported = (
            goal is not None and self.is_interaction(goal)
            and num(goal.get("distance"), 9999.0) < 115.0
        ) or self.still_ticks >= 5
        if self.force_use and use_supported and tic - self.last_use_tic >= 14:
            self.force_use = False
            self.last_use_tic = tic
            return self.legal("use")
        if goal is not None and self.jev_mode not in {"evade", "hold"}:
            return self.goal_action(goal, tic)

        if self.jev_retreat and (tic <= self.damage_until or health < 55.0):
            self.retreat_until = max(self.retreat_until, tic + 30)
            self.jev_retreat = False
            return self._retreat_action(tic)

        if self.still_ticks >= 5 and self.last_action in TRANSLATION:
            return self.handle_block(tic)

        # Periodic USE is sparse; it is a door/switch safety net, not a main
        # search action.  Collision faces get their own stricter probe above.
        if (tic - self.last_use_tic >= 175 and not self.threats
                and self.phase == "travel"):
            self.last_use_tic = tic
            return self.legal("use")

        return self.route_action(tic)

    # ---------- typed Jev interface ----------
    def compact(self, item):
        d = num(item.get("distance"), 9999.0)
        rel = num(item.get("relative"), 0.0)
        return {
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "category": item.get("category", ""),
            "range": "touching" if d < 48 else "near" if d < 170 else "medium" if d < 380 else "far",
            "side": "ahead" if abs(rel) < 15 else "left" if rel > 0 else "right",
            "level": "same" if abs(num(item.get("height"))) <= 42 else "different",
            "active": bool(item.get("active")),
            "hitscan": bool(item.get("hitscan")),
            "major": bool(item.get("major")),
        }

    def model_state(self, obs, tic):
        health = num(obs.get("health"), 100.0)
        armor = num(obs.get("armor"), 0.0)
        ammo = num(obs.get("weapon_ammo"), 0.0)
        goal_items = [x for x in self.objects if not x.get("monster") and not x.get("projectile")]
        goal_items.sort(key=lambda x: num(x.get("distance"), 9999.0))
        return {
            "task": "Survive and reach/activate the map exit; optional completion is secondary.",
            "visibility": "Only currently rendered labels are present. Do not infer hidden map geometry or inventory.",
            "health": "critical" if health < 25 else "wounded" if health < 60 else "healthy",
            "armor": "none" if armor < 1 else "light" if armor < 50 else "present",
            "ammo": "empty" if ammo <= 0 else "low" if ammo < 8 else "available",
            "damage_recent": tic <= self.damage_until,
            "player": {
                "node": list(self.current_node or (0, 0, 0)),
                "heading_sector": heading_bin(self.last_angle),
            },
            "motor": {
                "phase": self.phase,
                "action": self.last_action,
                "blocked": self.still_ticks >= 5,
                "stalled_tics": min(255, max(0, tic - self.last_progress_tic)),
                "node_visits": min(20, self.node_visits.get(self.current_node, 0)),
                "route_depth": min(255, len(self.node_path)),
                "frontier_nodes": min(255, len(self.frontier_nodes())),
                "collisions": min(255, self.collision_count),
            },
            "threats": [self.compact(x) for x in self.threats[:4]],
            "visible_goals": [self.compact(x) for x in goal_items[:4]],
            "policy": dict(self.policy),
            "vote": self.jev_mode if tic <= self.jev_mode_until else "expired",
        }

    def question_set(self, state):
        modes = {
            "explore": "Continue the measured frontier search and use a finite branch; no immediate listed danger dominates.",
            "fight": "Deal with a currently rendered active or dangerous monster before continuing the route.",
            "evade": "Keep moving away from a close projectile, recent damage, or dangerous threat to preserve health.",
            "interact": "Approach a currently rendered door, switch, teleporter, key, or exit and use it when close.",
            "retreat": "Return toward an observed breadcrumb because the current route is unsafe or blocked.",
            "hold": "The evidence is ambiguous; retain the current finite motor only briefly while re-observing.",
        }
        first = list(modes.items())
        if self.jev_request % 2:
            first.reverse()
        second = list(reversed(first))
        target_criteria = {"none": "No listed monster merits a short-lived target."}
        for item in self.threats[:4]:
            target_criteria[str(item.get("id"))] = {
                "what": item.get("name", "monster"),
                "facts": "Use only its supplied range, side, active, hitscan and major fields.",
            }
        return {
            "mode_a": Choice(
                "Choose one short-lived semantic mode from the exact JSON state. Do not output a button, calculate a distance, or invent an unseen route.",
                dict(first),
            ),
            "mode_b": Choice(
                "Independently choose the safest useful local mode from state.motor, state.threats, state.visible_goals and state.health. Code owns bearings and action legality.",
                dict(second),
            ),
            "branch": Choice(
                "At the next measured frontier boundary, choose which side to try first. This only orders code-owned finite probes; choose straight when no side is supported.",
                {
                    "left": "Try the left-side branch first.",
                    "right": "Try the right-side branch first.",
                    "reverse": "Prefer a reverse/backtrack branch after a blocked or repeated node.",
                    "straight": "Do not bias the code's measured least-tried branch.",
                },
            ),
            "target": Choice(
                "Which currently rendered monster is the most urgent, or none? Do not choose an absent actor.",
                target_criteria,
            ),
            "use": Noul(
                "Does the current state justify one USE pulse soon? Yes only for a listed interaction/exit or a measured blocked frontier face; never for an imagined door."
            ),
            "retreat": Noul(
                "Does current evidence justify retreating toward an observed breadcrumb? Yes only for recent damage, critical health, a close projectile, or a dangerous close threat."
            ),
        }

    def dispatch_jev(self, obs, tic):
        if getattr(self.ctx, "system1", True) is None:
            return
        if self.jev_future is not None or tic < self.jev_next:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if num(remaining.get("system1_calls"), 0.0) <= 10 or num(remaining.get("wall_s"), 0.0) < 2.0:
            return
        state = self.model_state(obs, tic)
        self.jev_request += 1
        questions = self.question_set(state)
        self.jev_meta = {"tic": tic, "policy": self.policy_version}
        danger = bool(self.threats or self.projectiles or tic <= self.damage_until)
        self.jev_next = tic + (21 if danger else 52)
        try:
            self.jev_future = self.ctx.spawn(
                lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=0.95)
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
        if age > 58 or integer(meta.get("policy"), -1) != self.policy_version:
            self.ctx.log("jev_discard", age=age, reason="stale_or_replanned")
            return
        modes = {"explore", "fight", "evade", "interact", "retreat", "hold"}
        a = choice_value(answers, "mode_a", modes)
        b = choice_value(answers, "mode_b", modes)
        if a is not None and a == b:
            self.jev_mode = a
            self.jev_mode_until = tic + 54
        branch = choice_value(answers, "branch", {"left", "right", "reverse", "straight"}, 0.54)
        if branch in {"left", "right", "reverse"}:
            self.branch_hint = branch
        target = choice_value(answers, "target", {str(x.get("id")) for x in self.threats} | {"none"}, 0.54)
        self.jev_target = None if target in (None, "none") else target
        if yes_value(answers, "use"):
            self.force_use = True
        self.jev_retreat = yes_value(answers, "retreat")
        self.ctx.log("jev_frontier_vote", mode=self.jev_mode, branch=self.branch_hint,
                     target=self.jev_target, use=self.force_use,
                     retreat=self.jev_retreat, age=age)

    # ---------- bounded Astra policy ----------
    def astra_prompt(self, obs, tic):
        state = self.model_state(obs, tic)
        evidence = list(self.history)[-10:]
        return (
            "You are the slow policy reviewer for a real-time Classic Doom controller. "
            "The code already owns aiming, collision detection, legal actions, USE cooldowns, "
            "and an observed-position frontier search with breadcrumb backtracking. Exit success "
            "and survival dominate optional kills. Labels are view-visible only. Do not output "
            "buttons, coordinates, hidden geometry, a route, code, or prose. Return ONLY a short "
            "JSON object with any of these bounded fields: hand (left|right), combat "
            "(cautious|normal), probe (normal|rare), resources (keys_only|capability). "
            "Change policy only when the measured evidence supports it; otherwise repeat the "
            "current policy.\nSTATE=" + json.dumps(state, separators=(",", ":"), allow_nan=False)
            + "\nEVIDENCE=" + json.dumps(evidence, separators=(",", ":"), allow_nan=False)
            + "\nCURRENT=" + json.dumps(self.policy, separators=(",", ":"), allow_nan=False)
        )

    def dispatch_astra(self, obs, tic):
        if self.astra_future is not None or self.astra_calls >= 2 or tic < self.astra_due:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if num(remaining.get("system2_calls"), 0.0) <= 0 or num(remaining.get("wall_s"), 0.0) < 180.0:
            return
        prompt = self.astra_prompt(obs, tic)
        session = self.astra_session
        self.astra_calls += 1
        self.astra_due = tic + 4200
        try:
            self.astra_future = self.ctx.spawn(
                lambda p=prompt, s=session: self.ctx.think(
                    p, session_id=s, timeout_s=170.0
                )
            )
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError):
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
        except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
            self.ctx.log("astra_failure", error=type(exc).__name__)
            return
        data = extract_object(getattr(result, "text", ""))
        accepted = {}
        for key, values in {
            "hand": {"left", "right"},
            "combat": {"cautious", "normal"},
            "probe": {"normal", "rare"},
            "resources": {"keys_only", "capability"},
        }.items():
            if data.get(key) in values:
                accepted[key] = data[key]
        self.astra_session = getattr(result, "session_id", self.astra_session)
        if accepted:
            self.policy.update(accepted)
            self.policy_version += 1
            self.jev_mode = None
            self.jev_mode_until = -1
            self.ctx.log("astra_frontier_policy", policy=dict(self.policy),
                         version=self.policy_version)
        else:
            self.ctx.log("astra_invalid", fallback="frontier_policy")

    # ---------- lifecycle ----------
    def run_episode(self, game):
        self.allowed = set(getattr(game.action_spec, "names", ()) or ())
        if not self.allowed:
            return
        candidate_noop = getattr(game.action_spec, "noop", "noop")
        self.noop = candidate_noop if candidate_noop in self.allowed else next(iter(self.allowed))
        ctx = self.ctx
        try:
            # Establish a legal held action before any model request.
            ctx.act(self.noop)
            while not ctx.done and not bool(getattr(game, "done", False)):
                raw = ctx.observe()
                obs = as_dict(raw)
                if not obs:
                    ctx.wait_for_tick()
                    continue
                if obs.get("terminal") or obs.get("dead"):
                    break
                tic = integer(obs.get("tic"), integer(obs.get("elapsed_tics"), 0))
                if self.last_tic is not None and tic <= self.last_tic:
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
                self.dispatch_astra(obs, tic)
                if tic > self.jev_mode_until:
                    self.jev_mode = None
                    self.jev_target = None
                    self.jev_retreat = False
                action = self.legal(self.choose_action(obs, tic))
                # Refresh on every processed state.  This keeps stale ticks low
                # while all pending inference remains off the engine thread.
                ctx.act(action)
                self.last_action = action
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
                ctx.memory["frontier_controller_summary"] = {
                    "nodes": len(self.node_positions),
                    "frontier_nodes": len(self.frontier_nodes()),
                    "collisions": self.collision_count,
                    "probes": self.frontier_events,
                    "policy": dict(self.policy),
                }
            except Exception:
                pass


def build(ctx):
    return Harness(ctx)
