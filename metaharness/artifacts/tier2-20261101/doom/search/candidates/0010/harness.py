"""Candidate 0010: keyed frontier search for label-poor Doom exits.

The native label projection contains keys and actors, but archived runs never
rendered Exit/Switch/Door labels.  This controller therefore treats movement
as an online graph-discovery problem.  Every discovered local node has finite,
remembered direction trials; a blocked face gets a one-tic USE attempt and a
retry after the door has had time to react.  Models can choose among measured
frontier branches, but never own the motor or invent a route.
"""
from __future__ import annotations

import collections
import json
import math
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


# These are names in the adaptor's public action vocabulary.  The legal set is
# still checked at runtime because pre-flight may expose a reduced action set.
TRANSLATION = {
    "forward", "backward", "strafe_left", "strafe_right",
    "forward_left", "forward_right", "backward_left", "backward_right",
    "fire_forward", "fire_strafe_left", "fire_strafe_right",
}
TURNING = {"turn_left", "turn_right"}
FIRE = {"fire", "fire_forward", "fire_strafe_left", "fire_strafe_right"}

HITSCAN = {
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS",
    "SpiderMastermind",
}
MAJOR = {"Archvile", "PainElemental", "Cyberdemon", "SpiderMastermind", "Revenant"}
MELEE = {"Demon", "Spectre", "LostSoul", "Pinky", "Pinkie"}
SPLASH_WEAPONS = {5, 7}

INTERACT_CATEGORIES = {"exit", "switch", "door", "teleporter", "teleport", "exitdoor"}
INTERACT_WORDS = ("exit", "switch", "door", "teleport")
PROJECTILE_CATEGORIES = {"projectile", "missile"}
PROJECTILE_WORDS = ("ball", "rocket", "plasma", "fireball", "revenant")

DIRECTION_NAMES = (
    "east", "northeast", "north", "northwest",
    "west", "southwest", "south", "southeast",
)


def finite(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    try:
        return int(finite(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def wrapped(angle):
    return (finite(angle) + 180.0) % 360.0 - 180.0


def distance(a, b):
    try:
        return math.sqrt(sum((finite(a[i]) - finite(b[i])) ** 2 for i in range(3)))
    except (TypeError, IndexError):
        return 99999.0


def xy_distance(a, b):
    try:
        return math.hypot(finite(a[0]) - finite(b[0]), finite(a[1]) - finite(b[1]))
    except (TypeError, IndexError):
        return 99999.0


def bearing(a, b):
    try:
        return math.degrees(math.atan2(finite(b[1]) - finite(a[1]),
                                       finite(b[0]) - finite(a[0]))) % 360.0
    except (TypeError, IndexError):
        return 0.0


def cell_of(point, size=64.0):
    try:
        return (int(math.floor(finite(point[0]) / size)),
                int(math.floor(finite(point[1]) / size)),
                int(math.floor((finite(point[2]) + 24.0) / 48.0)))
    except (TypeError, IndexError):
        return (0, 0, 0)


def heading_of(angle):
    return int(math.floor((finite(angle) % 360.0 + 22.5) / 45.0)) % 8


def heading_angle(index):
    return (int(index) % 8) * 45.0


def direction_point(point, heading, length=160.0):
    r = math.radians(heading_angle(heading))
    return (finite(point[0]) + length * math.cos(r),
            finite(point[1]) + length * math.sin(r), finite(point[2]))


def as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def extract_json(text):
    """Find the first bounded JSON object in an arbitrary Astra response."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for i, char in enumerate(text[:18000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            if isinstance(value, dict):
                return value
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return {}


def choice_value(answers, key, allowed, threshold=0.56):
    """Read a typed Choice without trusting stand-in or malformed answers."""
    if not isinstance(answers, dict):
        return None
    try:
        answer = answers.get(key)
        value = getattr(answer, "value", None)
        if not isinstance(value, str) or value not in allowed:
            return None
        probabilities = getattr(answer, "probabilities", None)
        if isinstance(probabilities, dict):
            if finite(probabilities.get(value), 0.0) < threshold:
                return None
        confidence = getattr(answer, "confidence", None)
        if confidence is not None and finite(confidence, 0.0) < 0.08:
            return None
        return value
    except Exception:
        return None


def noul_yes(answers, key, threshold=0.72):
    if not isinstance(answers, dict):
        return False
    try:
        value = getattr(answers.get(key), "value", 0.0)
        return bool(value) if isinstance(value, bool) else finite(value, 0.0) >= threshold
    except Exception:
        return False


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.legal = set()
        self.noop = "noop"

        # Current observation and proprioception.
        self.previous = None
        self.previous_action = "noop"
        self.last_tic = -1
        self.last_position = (0.0, 0.0, 0.0)
        self.last_angle = 0.0
        self.last_health = 100.0
        self.last_damage_counter = 0.0
        self.last_counts = (0, 0, 0)
        self.last_progress_tic = 0
        self.last_new_cell_tic = 0
        self.damage_until = -1
        self.damage_spike = 0.0
        self.current = None
        self.visible = []
        self.threats = []
        self.projectiles = []
        self.keys = []
        self.interactions = []
        self.last_use_tic = -10000
        self.last_weapon_tic = -10000
        self.fire_side = 1
        self.combat_turn_streak = 0
        self.turn_streak = 0
        self.turn_rate = 5.4
        self.left_angle_sign = 1.0

        # The observed graph.  A node records only places actually visited;
        # dirs are world-heading tests, not asserted map geometry.
        self.nodes = {}
        self.edges = collections.defaultdict(set)
        self.trail = collections.deque(maxlen=180)
        self.visited = collections.Counter()
        self.total_displacement = 0.0
        self.current_cell = (0, 0, 0)
        self.blocked_faces = collections.Counter()
        self.open_faces = collections.Counter()
        self.failed_routes = collections.Counter()
        self.key_epoch = 0
        self.key_names = []
        self.key_memory = {}
        self.last_key_tic = -10000

        # One finite local branch program or a route to an already observed
        # frontier.  No model response can create a program with raw buttons.
        self.probe = None
        self.route = None
        self.goal = None
        self.goal_fail_until = {}
        self.preferred_heading = None
        self.preferred_node = None
        self.jev_target = None
        self.jev_use = False

        # Durable policy is intentionally small and map-agnostic.
        self.policy = {
            "order": "least_tried",
            "probe": "barrier_and_junction",
            "key_radius": "near",
            "combat": "blockers",
            "side": "right",
        }
        self.policy_version = 0

        # At most one outstanding request of each speed.  Futures are bounded
        # inference work; there is no custom background loop to leak.
        self.jev_future = None
        self.jev_meta = None
        self.jev_request_id = 0
        self.next_jev_tic = 0
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.next_astra_tic = 700
        self.last_astra_tic = -10000

        self.history = collections.deque(maxlen=18)
        self.event_counts = collections.Counter()

    # ---------- safe action and observation projection ----------
    def action(self, *names):
        for name in names:
            if name in self.legal:
                return name
        if self.noop in self.legal:
            return self.noop
        return next(iter(self.legal), self.noop)

    def normalize(self, raw):
        raw = as_dict(raw)
        player = raw.get("player") if isinstance(raw.get("player"), dict) else {}
        p = (finite(player.get("x")), finite(player.get("y")), finite(player.get("z")))
        tic = integer(raw.get("tic"), self.last_tic + 1)
        angle = finite(player.get("angle")) % 360.0
        result = {
            "raw": raw, "tic": tic, "p": p, "angle": angle,
            "health": finite(raw.get("health"), 100.0),
            "armor": finite(raw.get("armor")),
            "damage": finite(raw.get("damage_taken")),
            "weapon": integer(raw.get("weapon"), 2),
            "weapon_ammo": finite(raw.get("weapon_ammo")),
            "ammo": raw.get("ammo") if isinstance(raw.get("ammo"), dict) else {},
            "weapons": raw.get("weapons") if isinstance(raw.get("weapons"), dict) else {},
            "kills": integer(raw.get("kills")),
            "items": integer(raw.get("items")),
            "secrets": integer(raw.get("secrets")),
            "terminal": bool(raw.get("terminal") or raw.get("dead")),
            "dead": bool(raw.get("dead")),
            "labels": [], "threats": [], "projectiles": [],
            "keys": [], "interactions": [],
        }
        labels = raw.get("visible")
        if not isinstance(labels, list):
            labels = []
        for index, label in enumerate(labels[:300]):
            if not isinstance(label, dict):
                continue
            world = label.get("world", label.get("position"))
            if not isinstance(world, (list, tuple)) or len(world) < 3:
                continue
            q = (finite(world[0]), finite(world[1]), finite(world[2]))
            name = str(label.get("name", label.get("object_name", "unknown")))[:80]
            category = str(label.get("category", label.get("object_category", "Unknown")))[:36]
            ident = str(label.get("id", label.get("object_id", "%s:%d" % (name, index))))[:120]
            bbox = label.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                box = [finite(x) for x in bbox[:4]]
            else:
                box = [0.0, 0.0, 0.0, 0.0]
            cat = category.lower()
            low = name.lower()
            if cat == "self" or name == "DoomPlayer":
                continue
            d = distance(p, q)
            err = wrapped(bearing(p, q) - angle)
            dead = low.startswith("dead") or "corpse" in low
            monster = cat == "monster" and not dead
            projectile = ((cat in PROJECTILE_CATEGORIES or
                           (cat == "explosive" and "barrel" not in low and not dead)) and
                          not ("puff" in low and d > 1000))
            if any(word in low for word in PROJECTILE_WORDS) and "barrel" not in low:
                projectile = True
            key = cat == "key" or "card" in low or low.endswith("key")
            interact = cat in INTERACT_CATEGORIES or any(word in low for word in INTERACT_WORDS)
            obj = {
                "id": ident, "name": name, "category": category, "p": q,
                "distance": d, "height": q[2] - p[2], "error": err,
                "bbox": box, "monster": monster, "projectile": projectile,
                "key": key, "interact": interact, "hitscan": monster and name in HITSCAN,
                "major": monster and name in MAJOR, "melee": monster and name in MELEE,
                "seen_tic": tic,
            }
            result["labels"].append(obj)
            if monster:
                result["threats"].append(obj)
            if projectile:
                result["projectiles"].append(obj)
            if key:
                result["keys"].append(obj)
            if interact:
                result["interactions"].append(obj)
        result["threats"].sort(key=lambda x: self.threat_score(x), reverse=True)
        result["projectiles"].sort(key=lambda x: x["distance"])
        result["keys"].sort(key=lambda x: x["distance"])
        result["interactions"].sort(key=lambda x: (0 if "exit" in x["name"].lower() else 1,
                                                    x["distance"]))
        return result

    def threat_score(self, obj):
        d = finite(obj.get("distance"), 99999.0)
        value = 0.0
        value += 260.0 if obj.get("major") else 0.0
        value += 165.0 if obj.get("hitscan") else 0.0
        value += 115.0 if obj.get("melee") else 0.0
        value += 190.0 if d < 90 else 100.0 if d < 190 else 35.0 if d < 550 else 0.0
        value -= d * 0.15 + abs(finite(obj.get("error"))) * 0.06
        return value

    def update(self, o):
        tic, p, angle = o["tic"], o["p"], o["angle"]
        old = self.current
        if old is not None:
            dt = max(1, tic - old["tic"])
            delta = wrapped(angle - old["angle"])
            if self.previous_action in TURNING and 0 < dt <= 12 and abs(delta) > 0.2:
                rate = abs(delta) / float(dt)
                if 0.2 < rate < 25.0:
                    self.turn_rate = 0.78 * self.turn_rate + 0.22 * rate
                    if self.previous_action == "turn_left":
                        self.left_angle_sign = 1.0 if delta > 0 else -1.0
                    else:
                        self.left_angle_sign = -1.0 if delta > 0 else 1.0

            moved = distance(old["p"], p)
            self.total_displacement += min(128.0, moved)
            if moved > 2.0:
                self.last_progress_tic = tic
            old_cell = cell_of(old["p"])
            new_cell = cell_of(p)
            if old_cell != new_cell:
                self.edges[old_cell].add(new_cell)
                if abs(p[2] - old["p"][2]) <= 60.0:
                    self.edges[new_cell].add(old_cell)
                self.last_new_cell_tic = tic
                self.last_progress_tic = tic
            health_loss = max(0.0, old["health"] - o["health"])
            counter_loss = max(0.0, o["damage"] - old["damage"])
            self.damage_spike = max(health_loss, counter_loss)
            if self.damage_spike > 0.0:
                self.damage_until = tic + 38
            if o["kills"] != old["kills"] or o["items"] != old["items"] or o["secrets"] != old["secrets"]:
                self.last_progress_tic = tic

            # A branch which left its origin has proved that this world heading
            # is open.  It is released before a new direction can be selected.
            if self.probe and self.probe.get("origin") == old_cell:
                if moved > 2.0:
                    self.probe["moved"] = self.probe.get("moved", 0.0) + moved
                    self.probe["last_move_tic"] = tic
                else:
                    self.probe["no_move"] = self.probe.get("no_move", 0) + dt

        self.current = o
        self.current_cell = cell_of(p)
        node = self.nodes.setdefault(self.current_cell, self.new_node(p, tic))
        node["p"] = p
        node["visits"] += 1
        self.visited[self.current_cell] += 1
        if old is None or cell_of(old["p"]) != self.current_cell:
            self.last_new_cell_tic = tic
        if not self.trail or xy_distance(self.trail[-1][1], p) > 20.0:
            self.trail.append((tic, p, self.current_cell, o["health"]))

        # Keep key evidence, but do not treat a key merely seen in another view
        # as inventory.  A near label disappearing after a key goal is evidence.
        current_key_ids = {x["id"] for x in o["keys"]}
        for item in o["keys"]:
            self.key_memory[item["id"]] = {
                "name": item["name"], "p": item["p"], "last": tic,
                "near": item["distance"] < 52.0,
            }
        if self.goal and self.goal.get("kind") == "key":
            gid = self.goal.get("id")
            if gid not in current_key_ids and self.goal.get("best", 9999.0) < 58.0:
                self.mark_key(self.goal.get("name", "key"), tic, "near_label_disappeared")
                self.goal = None
        if old is not None and o["items"] > old["items"] and self.goal is not None:
            if self.goal.get("kind") == "key" and self.goal.get("best", 9999.0) < 75.0:
                self.mark_key(self.goal.get("name", "key"), tic, "item_count_after_key_goal")
                self.goal = None

        # A stale direct target must not survive a long loss of rendering.
        if self.goal and tic - self.goal.get("last_seen", tic) > 18:
            if self.goal.get("kind") == "key":
                self.goal_fail_until[self.goal.get("id")] = tic + 420
            self.goal = None
        self.last_tic, self.last_position, self.last_angle = tic, p, angle
        self.last_health, self.last_damage_counter = o["health"], o["damage"]
        self.last_counts = (o["kills"], o["items"], o["secrets"])
        self.history.append({
            "tic": tic, "cell": list(self.current_cell), "health": round(o["health"], 1),
            "armor": round(o["armor"], 1), "kills": o["kills"], "items": o["items"],
            "keys_visible": len(o["keys"]), "action": self.previous_action,
        })

    @staticmethod
    def new_node(p, tic):
        return {"p": p, "visits": 0, "first": tic,
                "dirs": {str(i): {"status": "unknown", "tries": 0,
                                   "retry": 0, "last": -1} for i in range(8)}}

    def mark_key(self, name, tic, reason):
        label = str(name or "key")[:80]
        if label not in self.key_names:
            self.key_names.append(label)
            self.key_names = self.key_names[-8:]
            self.key_epoch += 1
            self.last_key_tic = tic
            # A locked door that was correctly classified as a wall is worth
            # retrying after a capability milestone; preserve attempt counts.
            for node in self.nodes.values():
                for info in node["dirs"].values():
                    if info["status"] == "blocked":
                        info["status"] = "unknown"
                        info["retry"] = tic
            self.event_counts["key_milestone"] += 1
            try:
                self.ctx.log("key_milestone", name=label, epoch=self.key_epoch, reason=reason)
            except Exception:
                pass

    # ---------- graph frontier and finite movement programs ----------
    def node_frontier(self, node, tic):
        if not isinstance(node, dict) or not isinstance(node.get("dirs"), dict):
            return []
        result = []
        for h in range(8):
            info = node["dirs"].get(str(h))
            if not isinstance(info, dict):
                continue
            status = info["status"]
            if status == "unknown":
                result.append(h)
            elif status == "blocked" and tic >= integer(info.get("retry"), 0):
                result.append(h)
        return result

    def branch_options(self, tic):
        node = self.nodes.get(self.current_cell)
        if node is None:
            return []
        choices = self.node_frontier(node, tic)
        if not choices:
            return []
        # Sort least-tested headings first.  Astra may change only this tie
        # policy; a branch is still code-generated and locally rechecked.
        order = self.policy.get("order", "least_tried")
        if order == "clockwise":
            base = heading_of(self.last_angle)
            choices.sort(key=lambda h: ((h - base) % 8, node["dirs"][str(h)]["tries"]))
        elif order == "counterclockwise":
            base = heading_of(self.last_angle)
            choices.sort(key=lambda h: ((base - h) % 8, node["dirs"][str(h)]["tries"]))
        else:
            choices.sort(key=lambda h: (node["dirs"][str(h)]["tries"],
                                        abs(((h - heading_of(self.last_angle) + 4) % 8) - 4)))
        return choices

    def start_probe(self, heading, tic):
        heading = int(heading) % 8
        node = self.nodes.setdefault(self.current_cell, self.new_node(self.last_position, tic))
        info = node["dirs"][str(heading)]
        info["tries"] += 1
        info["last"] = tic
        # Mark a pending test so a response cannot repeatedly choose it.
        info["status"] = "testing"
        self.probe = {
            "origin": self.current_cell, "heading": heading, "phase": "align",
            "started": tic, "moved": 0.0, "no_move": 0, "last_move_tic": tic,
            "use_done": False, "retry_started": -1,
            "use_at_start": self.policy.get("probe") == "every_branch",
            "after_use": "advance",
        }
        self.preferred_heading = None
        self.preferred_node = None
        self.event_counts["probe_started"] += 1

    def finish_probe(self, status, tic):
        if not self.probe:
            return
        origin = self.probe.get("origin")
        heading = int(self.probe.get("heading", 0)) % 8
        node = self.nodes.get(origin)
        if node is not None:
            info = node["dirs"][str(heading)]
            info["status"] = status
            info["last"] = tic
            if status == "blocked":
                info["retry"] = tic + (260 if self.key_epoch == 0 else 150)
                self.blocked_faces[(origin, heading)] += 1
            else:
                self.open_faces[(origin, heading)] += 1
        self.event_counts["probe_" + status] += 1
        self.probe = None

    def angle_action(self, error):
        return self.action("turn_left") if wrapped(error) * self.left_angle_sign > 0 else self.action("turn_right", "turn_left")

    def move_toward(self, point, allow_turn=True):
        error = wrapped(bearing(self.last_position, point) - self.last_angle)
        if allow_turn and abs(error) > 10.0 and self.turn_streak < 5:
            self.turn_streak += 1
            return self.angle_action(error)
        # Deliberate translation after a few corrections prevents a target
        # label from recreating the old infinite turn-in-place loop.
        self.turn_streak = 0
        if error > 9.0:
            return self.action("forward_left", "strafe_left", "forward")
        if error < -9.0:
            return self.action("forward_right", "strafe_right", "forward")
        return self.action("forward", "forward_left", "forward_right")

    def probe_action(self, o):
        p = self.probe
        if p is None:
            return None
        tic = o["tic"]
        heading = int(p["heading"]) % 8
        desired = heading_angle(heading)
        error = wrapped(desired - o["angle"])
        if p["phase"] == "align":
            if abs(error) > 7.5 and tic - p["started"] < 34:
                return self.angle_action(error)
            p["phase"] = "use" if p.get("use_at_start") else "advance"
        if p["phase"] == "use":
            after_use = p.get("after_use", "advance")
            p["phase"] = after_use
            p["use_done"] = True
            p["retry_started"] = tic if after_use == "retry" else -1
            if tic - self.last_use_tic >= 10:
                self.last_use_tic = tic
                self.event_counts["use_probe"] += 1
                return self.action("use", "forward")
        if p["phase"] in ("advance", "retry"):
            # A substantial displacement or a new cell is enough to certify
            # that the face is open; stop before a successful corridor becomes
            # another long straight-line sweep.
            # Cross at least most of a 64-unit cell before declaring an
            # open face complete.  The old sweep controllers often sampled
            # only a short nudge and therefore made no measured edge.
            if p.get("moved", 0.0) >= 56.0 or self.current_cell != p["origin"]:
                self.finish_probe("open", tic)
                return None
            if p["phase"] == "advance" and p.get("no_move", 0) >= 8:
                # Only a measured barrier earns a USE pulse in the default
                # protocol; an Astra/explicit every_branch policy may pulse at
                # the junction before this test.
                p["phase"] = "use"
                p["after_use"] = "retry"
                p["no_move"] = 0
            elif p["phase"] == "retry" and tic - p.get("retry_started", tic) >= 14:
                self.finish_probe("blocked", tic)
                return None
            if tic - p["started"] > 60:
                self.finish_probe("open" if p.get("moved", 0.0) > 4 else "blocked", tic)
                return None
            return self.action("forward", "forward_left")
        return self.action("forward")

    def nearest_frontier_route(self, start, tic):
        """BFS only on measured edges, to a node with an untried face."""
        if start not in self.nodes:
            return None
        parents = {start: None}
        queue = collections.deque([start])
        found = None
        while queue and len(parents) < 700:
            here = queue.popleft()
            if here != start and self.node_frontier(self.nodes.get(here, {}), tic):
                found = here
                break
            for nxt in self.edges.get(here, ()):
                if nxt not in parents:
                    parents[nxt] = here
                    queue.append(nxt)
        if found is None:
            return None
        path = []
        cur = found
        while cur is not None and cur != start:
            path.append(cur)
            cur = parents.get(cur)
        path.reverse()
        return path

    def route_action(self, o):
        if not self.route:
            return None
        tic = o["tic"]
        target_cell = self.route[0]
        target = self.nodes.get(target_cell, {}).get("p")
        if target is None:
            self.route = None
            return None
        if xy_distance(o["p"], target) < 30.0:
            self.route.pop(0)
            if not self.route:
                return None
            target = self.nodes.get(self.route[0], {}).get("p", o["p"])
        # Route entries are measured positions, and the failure clock is reset
        # whenever the current cell changes (update() does that globally).
        if tic - self.last_progress_tic > 55:
            self.failed_routes[target_cell] += 1
            self.route = None
            return None
        return self.move_toward(target)

    def select_exploration(self, o):
        tic = o["tic"]
        if self.probe:
            return self.probe_action(o)
        if self.route:
            result = self.route_action(o)
            if result is not None:
                return result
        choices = self.branch_options(tic)
        if choices:
            heading = None
            if self.preferred_node == self.current_cell and self.preferred_heading in choices:
                heading = self.preferred_heading
            if heading is None:
                heading = choices[0]
            self.start_probe(heading, tic)
            return self.probe_action(o)
        path = self.nearest_frontier_route(self.current_cell, tic)
        if path:
            self.route = path
            return self.route_action(o)
        # The measured component has exhausted its known faces.  Re-test the
        # least-tried open face after a quiet interval: an open branch that did
        # not cross a coarse cell may still lead farther once the player starts
        # from a slightly different point.  This remains finite and ordered,
        # unlike a blind square sweep.
        if tic - self.last_new_cell_tic > 70:
            node = self.nodes.get(self.current_cell)
            if node:
                h = min(range(8), key=lambda x: node["dirs"][str(x)]["tries"])
            else:
                h = heading_of(o["angle"])
            self.start_probe(h, tic)
            return self.probe_action(o)
        return self.action("forward_left", "forward", "strafe_left")

    # ---------- visible milestones and combat ----------
    def visible_interaction(self, o):
        if not o["interactions"]:
            return None
        for item in o["interactions"]:
            if item["distance"] <= 700 and abs(item["height"]) <= 75:
                return item
        return None

    def interaction_action(self, o, item):
        if item is None:
            return None
        tic = o["tic"]
        if item["distance"] > 58:
            return self.move_toward(item["p"])
        if abs(item["error"]) > 15:
            return self.angle_action(item["error"])
        if tic - self.last_use_tic >= 12:
            self.last_use_tic = tic
            self.event_counts["use_visible"] += 1
            return self.action("use", "forward")
        return self.action("forward", "strafe_left")

    def key_goal(self, o):
        tic = o["tic"]
        # Retain only a short, measurable approach to one currently rendered
        # key.  This is intentionally narrower than the parents' all-pickup
        # attraction, which produced many broad coverage loops.
        if self.goal is not None:
            if self.goal.get("kind") == "key":
                item = next((x for x in o["keys"] if x["id"] == self.goal.get("id")), None)
                if item is None:
                    self.goal = None
                    return None
                self.goal["last_seen"] = tic
                d = item["distance"]
                if d < self.goal.get("best", 99999.0) - 6:
                    self.goal["best"] = d
                    self.goal["last_progress"] = tic
                elif tic - self.goal.get("last_progress", tic) > 60:
                    self.goal_fail_until[item["id"]] = tic + 420
                    self.goal = None
                    return None
                self.goal["p"] = item["p"]
                return self.move_toward(item["p"])
            self.goal = None

        radius = 780.0 if self.policy.get("key_radius") == "wide" else 430.0
        candidates = []
        for item in o["keys"]:
            if item["distance"] > radius or abs(item["height"]) > 70:
                continue
            if self.goal_fail_until.get(item["id"], -1) > tic:
                continue
            value = 1100.0 - item["distance"] * 0.9
            if item["distance"] < 160:
                value += 180.0
            candidates.append((value, item))
        if not candidates:
            return None
        _, item = max(candidates, key=lambda x: x[0])
        self.goal = {"kind": "key", "id": item["id"], "name": item["name"],
                     "p": item["p"], "best": item["distance"],
                     "last_progress": tic, "last_seen": tic}
        self.event_counts["key_goal"] += 1
        return self.move_toward(item["p"])

    def weapon_owned(self, o, slot):
        weapons = o["weapons"]
        return bool(weapons.get(str(slot), weapons.get(slot, False))) or bool(
            weapons.get(str(slot - 1), weapons.get(slot - 1, False)))

    def weapon_ammo(self, o, slot):
        if o["weapon"] == slot:
            return max(0.0, o["weapon_ammo"])
        ammo = o["ammo"]
        values = [finite(ammo.get(str(slot))), finite(ammo.get(slot)),
                  finite(ammo.get(str(slot - 1))), finite(ammo.get(slot - 1))]
        if slot == 3:
            values += [finite(ammo.get("8")), finite(ammo.get("9"))]
        return max(values or [0.0])

    def weapon_switch(self, o, threat):
        tic = o["tic"]
        if tic - self.last_weapon_tic < 30:
            return None
        current = o["weapon"]
        order = [6, 4, 3, 2, 1]
        if threat and (threat.get("major") or len(o["threats"]) >= 3):
            order = [6, 4, 3, 2, 1]
        if threat and threat["distance"] < 150:
            order = [3, 4, 6, 2, 1]
        for slot in order:
            if slot == current or not self.weapon_owned(o, slot):
                continue
            if self.weapon_ammo(o, slot) <= 0:
                continue
            action = "select_weapon_" + str(slot)
            if action in self.legal:
                self.last_weapon_tic = tic
                return action
        return None

    def choose_threat(self, o):
        if not o["threats"]:
            return None
        visible_ids = {x["id"] for x in o["threats"]}
        if self.jev_target in visible_ids:
            return next(x for x in o["threats"] if x["id"] == self.jev_target)
        return max(o["threats"], key=self.threat_score)

    def combat_action(self, o, threat):
        if threat is None:
            self.combat_turn_streak = 0
            return None
        tic = o["tic"]
        d = threat["distance"]
        immediate = (d < 150 or (threat["hitscan"] and d < 650) or
                     threat["major"] or (threat["melee"] and d < 250) or
                     tic <= self.damage_until)
        # Distant routine actors are not allowed to derail frontier search;
        # Astra's avoid policy can tighten that boundary without touching
        # emergency combat.
        if not immediate and (d > 360 or self.policy.get("combat") == "avoid"):
            return None
        if o["health"] < 25 and (d < 250 or tic <= self.damage_until):
            self.fire_side *= -1
            return self.action("backward_left" if self.fire_side < 0 else "backward_right",
                               "backward", "strafe_left")
        switch = self.weapon_switch(o, threat)
        if switch:
            return switch
        if o["weapon"] in SPLASH_WEAPONS and d < 190:
            fallback = self.action("select_weapon_3", "select_weapon_2")
            if fallback != self.noop:
                return fallback
        if abs(threat["error"]) > 13.0 and self.combat_turn_streak < 3:
            self.combat_turn_streak += 1
            return self.angle_action(threat["error"])
        self.combat_turn_streak = 0
        if o["projectiles"] and o["projectiles"][0]["distance"] < 210:
            self.fire_side *= -1
            return self.action("strafe_left" if self.fire_side < 0 else "strafe_right",
                               "forward_left", "forward_right")
        if d < 85 and (threat["melee"] or tic <= self.damage_until):
            self.fire_side *= -1
            return self.action("backward_left" if self.fire_side < 0 else "backward_right",
                               "backward", "strafe_left")
        if o["weapon_ammo"] <= 0 and o["weapon"] not in (1, 8):
            return self.action("backward_left", "backward_right", "backward")
        self.fire_side *= -1 if tic % 21 == 0 else 1
        side = "left" if self.fire_side < 0 else "right"
        return self.action("fire_strafe_" + side, "fire_forward", "fire", "forward_" + side)

    # ---------- Jev typed bridge ----------
    def compact_obj(self, item):
        d = item["distance"]
        return {
            "id": item["id"], "name": item["name"], "category": item["category"],
            "range": "contact" if d < 75 else "near" if d < 200 else "medium" if d < 500 else "far",
            "side": "center" if abs(item["error"]) < 12 else "left" if item["error"] > 0 else "right",
            "level": "same" if abs(item["height"]) <= 55 else "different",
            "hitscan": bool(item.get("hitscan")), "major": bool(item.get("major")),
            "body_blocker": bool(item.get("melee") and d < 170),
        }

    def branch_state(self, tic):
        node = self.nodes.get(self.current_cell)
        rows = []
        if node:
            for h in range(8):
                info = node["dirs"][str(h)]
                rows.append({"id": "d" + str(h), "heading": DIRECTION_NAMES[h],
                             "status": info["status"], "tries": min(4, info["tries"])})
        return rows

    def model_state(self, o):
        node = self.nodes.get(self.current_cell, self.new_node(o["p"], o["tic"]))
        blocked = bool(self.probe and self.probe.get("phase") in ("retry",))
        return {
            "task": "Reach and activate the unseen map exit alive; keys are capability milestones, kills are secondary.",
            "rules": "Only currently rendered labels are supplied. Code owns distances, graph search, aiming, collision tests, USE timing and legal actions. Choose among code-provided branches; never invent a route or button.",
            "policy": dict(self.policy), "policy_version": self.policy_version,
            "health": "critical" if o["health"] < 25 else "wounded" if o["health"] < 60 else "healthy",
            "armor": "none" if o["armor"] < 10 else "present",
            "ammo": "empty" if o["weapon_ammo"] <= 0 else "low" if o["weapon_ammo"] < 10 else "available",
            "recent_damage": o["tic"] <= self.damage_until,
            "milestone": {"key_epoch": self.key_epoch, "keys_observed": self.key_names[-5:],
                          "key_visible": bool(o["keys"]), "interaction_label_visible": bool(o["interactions"])},
            "player": {"cell": list(self.current_cell), "facing": DIRECTION_NAMES[heading_of(o["angle"])]},
            "motion": {
                "phase": "probe" if self.probe else "route" if self.route else "key_goal" if self.goal else "frontier",
                "blocked_face": blocked, "new_cell_recent": o["tic"] - self.last_new_cell_tic < 35,
                "stagnant": o["tic"] - self.last_progress_tic > 90,
                "visited_cells": min(500, len(self.nodes)),
                "known_edges": min(800, sum(len(x) for x in self.edges.values()) // 2),
            },
            "branches": self.branch_state(o["tic"]),
            "threats": [self.compact_obj(x) for x in o["threats"][:5]],
            "projectiles": [self.compact_obj(x) for x in o["projectiles"][:3]],
            "keys": [self.compact_obj(x) for x in o["keys"][:4]],
            "interactions": [self.compact_obj(x) for x in o["interactions"][:3]],
            "last_action": self.previous_action,
        }

    def questions(self, state, options, reverse=False):
        criteria = {}
        for h in options:
            criteria["d" + str(h)] = {
                "heading": DIRECTION_NAMES[h],
                "meaning": "Try this code-generated local face; prefer unknown over already tested faces.",
            }
        criteria["none"] = "Do not bias the frontier choice; let the deterministic least-tried rule select it."
        ordered = list(criteria.items())
        if reverse:
            ordered.reverse()
        branch = dict(ordered)
        target_criteria = {x["id"]: {
            "what": x["name"], "facts": "Use only the supplied range, side, level and threat flags."
        } for x in state.get("threats", [])}
        target_criteria["none"] = "No listed threat should own the controller; continue the safe graph search."
        return {
            "branch_a": Choice(
                "Choose a local search branch from state.branches. Select only a supplied dN option. "
                "Prefer an untested face, but do not infer a hidden door or exit.", branch),
            "branch_b": Choice(
                "Independently choose the same local branch using only the named branch status and motion facts. "
                "The code rejects disagreement and stale answers.", dict(reversed(ordered))),
            "threat": Choice(
                "Which currently rendered monster, if any, is the most urgent immediate threat? "
                "Do not choose a distant routine actor when code can safely explore.", target_criteria),
            "use_barrier": Noul(
                "Do current facts justify one USE pulse at a measured blocked face? "
                "Yes only for a current retry/barrier or a currently rendered interaction; never invent one.",
                {"true": "A measured barrier or rendered interaction makes a pulse worthwhile.",
                 "false": "No measured or rendered interaction supports a pulse."}),
        }

    def dispatch_jev(self, o):
        if getattr(self.ctx, "system1", object()) is None:
            return
        if self.jev_future is not None or o["tic"] < self.next_jev_tic:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if remaining.get("system1_calls", 0) <= 5 or remaining.get("wall_s", 0) < 3:
            return
        options = self.branch_options(o["tic"])
        state = self.model_state(o)
        self.jev_request_id += 1
        request_id = self.jev_request_id
        questions = self.questions(state, options, reverse=(request_id % 2 == 0))
        self.jev_meta = {"id": request_id, "tic": o["tic"], "node": self.current_cell,
                         "version": self.policy_version, "epoch": self.key_epoch,
                         "options": options, "state": state}
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=1.45)
            )
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
            return
        self.next_jev_tic = o["tic"] + (18 if o["threats"] or o["tic"] <= self.damage_until else 34)

    def poll_jev(self, o):
        if self.jev_future is None or not self.jev_future.done():
            return
        future, meta = self.jev_future, self.jev_meta
        self.jev_future = None
        self.jev_meta = None
        try:
            answers = future.result()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, EpisodeOver, CancelledError):
            return
        except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
            self.event_counts["jev_failure"] += 1
            try:
                self.ctx.log("keyed_search_jev_failure", error=type(exc).__name__)
            except Exception:
                pass
            return
        if not meta:
            return
        age = o["tic"] - integer(meta.get("tic"), o["tic"])
        if (age > 52 or meta.get("version") != self.policy_version
                or meta.get("epoch") != self.key_epoch):
            self.event_counts["jev_stale"] += 1
            return
        # The node and candidate face must still be current.  This prevents a
        # fast answer from selecting a branch after a key/teleport transition.
        options = list(meta.get("options") or [])
        allowed = {"d" + str(x) for x in options} | {"none"}
        a = choice_value(answers, "branch_a", allowed, 0.56)
        b = choice_value(answers, "branch_b", allowed, 0.56)
        adopted = {}
        if a is not None and a == b and a.startswith("d"):
            try:
                h = int(a[1:]) % 8
            except (TypeError, ValueError):
                h = -1
            if h in self.branch_options(o["tic"]) and meta.get("node") == self.current_cell:
                self.preferred_heading = h
                self.preferred_node = self.current_cell
                adopted["branch"] = DIRECTION_NAMES[h]
        target_allowed = {x["id"] for x in o["threats"]} | {"none"}
        target = choice_value(answers, "threat", target_allowed, 0.58)
        if target and target != "none":
            self.jev_target = target
            adopted["threat"] = target
        elif target == "none":
            self.jev_target = None
        if noul_yes(answers, "use_barrier"):
            self.jev_use = True
            adopted["use_barrier"] = True
        self.event_counts["jev_adopted"] += bool(adopted)
        try:
            self.ctx.log("keyed_search_jev", age_tics=age, adopted=adopted,
                         agreement=(a is not None and a == b))
        except Exception:
            pass

    # ---------- Astra policy revision ----------
    def astra_payload(self, o):
        return {
            "observation": {
                "map": o["raw"].get("map"), "iwad": o["raw"].get("iwad"),
                "skill": o["raw"].get("skill"), "tic": o["tic"],
                "health": o["health"], "armor": o["armor"], "weapon": o["weapon"],
                "weapon_ammo": o["weapon_ammo"], "kills": o["kills"], "items": o["items"],
                "visible_keys": [self.compact_obj(x) for x in o["keys"][:8]],
                "visible_interactions": [self.compact_obj(x) for x in o["interactions"][:5]],
                "visible_threats": [self.compact_obj(x) for x in o["threats"][:5]],
            },
            "current_policy": self.policy, "key_epoch": self.key_epoch,
            "key_names": self.key_names[-8:], "visited_cells": len(self.nodes),
            "known_edges": sum(len(x) for x in self.edges.values()) // 2,
            "blocked_faces": sum(self.blocked_faces.values()),
            "open_faces": sum(self.open_faces.values()),
            "recent": list(self.history),
        }

    def launch_astra(self, o):
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if remaining.get("system2_calls", 0) <= 0 or remaining.get("wall_s", 0) < 100:
            return
        payload = self.astra_payload(o)
        prompt = (
            "You are the slow strategic reviewer for a real-time Classic Doom controller. "
            "The primary objective is to activate an exit that may not be rendered as a label. "
            "The code has an online graph of only traversed cells. At each node it tests finite "
            "world-heading branches; a blocked face receives USE and a delayed retry. Keys are "
            "observed capability milestones, not a memorized route. Do not output buttons, raw "
            "coordinates, a map-specific path, hidden geometry, or code. Choose only a durable "
            "search protocol. Return ONLY JSON with these fields: "
            "order=least_tried|clockwise|counterclockwise, "
            "probe=barrier_and_junction|every_branch, "
            "key_radius=near|wide, combat=blockers|avoid, side=left|right, reason=short. "
            "Use the measured counts and labels below. A late answer is applied only as policy; "
            "emergency movement, combat, legality, terminal checks and graph evidence remain code-owned.\n" +
            json.dumps(payload, separators=(",", ":"), allow_nan=False)
        )
        self.astra_calls += 1
        self.last_astra_tic = o["tic"]
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, session_id=self.astra_session, timeout_s=220.0)
            )
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None

    def poll_astra(self, o):
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
            try:
                self.ctx.log("keyed_search_astra_failure", error=type(exc).__name__)
            except Exception:
                pass
            return
        self.astra_session = getattr(result, "session_id", self.astra_session)
        proposal = extract_json(getattr(result, "text", ""))
        updated = dict(self.policy)
        changed = False
        for key, allowed in {
            "order": {"least_tried", "clockwise", "counterclockwise"},
            "probe": {"barrier_and_junction", "every_branch"},
            "key_radius": {"near", "wide"},
            "combat": {"blockers", "avoid"},
            "side": {"left", "right"},
        }.items():
            value = proposal.get(key)
            if isinstance(value, str) and value in allowed:
                updated[key] = value
                changed = True
        if changed:
            self.policy = updated
            self.policy_version += 1
            self.preferred_heading = None
            self.preferred_node = None
            self.jev_target = None
            self.jev_use = False
            self.event_counts["astra_applied"] += 1
            try:
                self.ctx.log("keyed_search_astra", policy=dict(self.policy),
                             reason=str(proposal.get("reason", ""))[:300])
            except Exception:
                pass

    # ---------- top-level control ----------
    def action_for(self, o):
        if o["terminal"] or o["health"] <= 0:
            return self.action("noop", "backward")

        # Emergency layer outranks any key goal or graph decision.
        projectile = next((x for x in o["projectiles"] if x["distance"] < 230), None)
        if projectile is not None:
            self.fire_side *= -1
            return self.action("strafe_left" if self.fire_side < 0 else "strafe_right",
                               "forward_left", "forward_right", "backward")
        threat = self.choose_threat(o)
        if threat is not None:
            combat = self.combat_action(o, threat)
            if combat is not None:
                return combat

        interaction = self.visible_interaction(o)
        if interaction is not None:
            self.goal = None
            result = self.interaction_action(o, interaction)
            if result is not None:
                return result

        # Keys are a capability subgoal, but only for a bounded, currently
        # rendered approach; absence immediately returns authority to search.
        key_action = self.key_goal(o)
        if key_action is not None:
            return key_action

        # Model Noul can request an extra pulse, but never bypasses cooldown or
        # the requirement that a measured branch or visible object exists.
        if self.jev_use and self.probe and self.probe.get("phase") in ("advance", "retry"):
            self.jev_use = False
            if o["tic"] - self.last_use_tic >= 12:
                self.last_use_tic = o["tic"]
                return self.action("use", "forward")

        explored = self.select_exploration(o)
        if explored is not None:
            return explored
        return self.action("forward_left", "forward", "strafe_left", "noop")

    def run_episode(self, game):
        ctx = self.ctx
        names = getattr(getattr(game, "action_spec", None), "names", ()) or ()
        self.legal = set(names)
        self.noop = getattr(getattr(game, "action_spec", None), "noop", "noop")
        if self.noop not in self.legal:
            self.noop = "noop" if "noop" in self.legal else next(iter(self.legal), "noop")
        if not self.legal:
            return
        try:
            while not ctx.done and not getattr(game, "done", False):
                raw = ctx.observe()
                o = self.normalize(raw)
                if o["terminal"] or ctx.done or getattr(game, "done", False):
                    break
                if o["tic"] <= self.last_tic:
                    ctx.wait_for_tick()
                    continue
                self.visible, self.threats = o["labels"], o["threats"]
                self.projectiles, self.keys = o["projectiles"], o["keys"]
                self.interactions = o["interactions"]
                self.update(o)
                self.poll_jev(o)
                self.poll_astra(o)

                # Post a deterministic action before launching any new model
                # request.  This is the late-decision fallback on every tic.
                action = self.action_for(o)
                ctx.act(action)
                self.previous_action = action

                self.dispatch_jev(o)
                if (self.astra_future is None and o["tic"] >= self.next_astra_tic and
                        self.astra_calls < 2):
                    self.launch_astra(o)
                    self.next_astra_tic = o["tic"] + (5600 if self.event_counts["probe_blocked"] < 8 else 3500)

                if o["tic"] % 175 == 0:
                    try:
                        ctx.log("keyed_search_summary", tic=o["tic"], cells=len(self.nodes),
                                 edges=sum(len(x) for x in self.edges.values()) // 2,
                                 key_epoch=self.key_epoch, blocked=sum(self.blocked_faces.values()),
                                 open=sum(self.open_faces.values()), health=o["health"],
                                 kills=o["kills"], items=o["items"], action=action)
                    except Exception:
                        pass
                ctx.wait_for_tick()
        except (EpisodeOver, BudgetExceeded):
            return
        finally:
            for future_name in ("jev_future", "astra_future"):
                future = getattr(self, future_name, None)
                if future is not None and not future.done():
                    try:
                        future.cancel()
                    except Exception:
                        pass
            try:
                ctx.memory["keyed_frontier_summary"] = {
                    "cells": len(self.nodes),
                    "edges": sum(len(x) for x in self.edges.values()) // 2,
                    "key_epoch": self.key_epoch,
                    "blocked_faces": sum(self.blocked_faces.values()),
                    "open_faces": sum(self.open_faces.values()),
                    "policy": dict(self.policy),
                }
            except Exception:
                pass


def build(ctx):
    return Harness(ctx)
