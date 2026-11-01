"""Crossover controller: transactional goals with coordinate-staged interactions.

The controller keeps the useful compiled-action idea from the transactional parent,
but treats a block interaction as a geometry transaction rather than as one string
that can be retried.  Resource targets are approached with observed safe feet
coordinates (``goto``), while block references are recompiled from the newest
snapshot.  Jev chooses among these already legal commands; Astra only changes a
slow macro bias after a real stall.
"""
from __future__ import annotations

import json
import math
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


WOOD_SUFFIXES = ("_log", "_stem")
FOODS = {
    "apple", "bread", "baked_potato", "carrot", "cooked_beef", "cooked_chicken",
    "cooked_mutton", "cooked_porkchop", "cooked_rabbit", "cooked_cod",
    "cooked_salmon", "sweet_berries", "melon_slice", "beef", "chicken",
    "mutton", "porkchop", "rabbit", "cod", "salmon", "potato", "beetroot",
    "wheat", "dried_kelp",
}
STONE_NAMES = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON_NAMES = {"iron_ore", "deepslate_iron_ore"}
COAL_NAMES = {"coal_ore", "deepslate_coal_ore"}
PASSIVE = {"cow", "pig", "sheep", "chicken", "rabbit"}
WEAPONS = ("stone_sword", "iron_sword", "diamond_sword", "stone_axe", "iron_axe")
PICKS = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")
DANGERS = {"lava", "fire", "soul_fire", "magma_block", "cactus", "powder_snow"}
SOLID_SUPPORTS = {
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "stone", "cobblestone", "cobbled_deepslate", "deepslate", "sand", "red_sand",
    "gravel", "terracotta", "netherrack", "end_stone", "andesite", "diorite",
    "granite", "tuff", "calcite", "blackstone", "basalt", "oak_planks",
}
REPLACEABLE = {
    "air", "cave_air", "void_air", "short_grass", "tall_grass", "fern",
    "large_fern", "vine", "snow", "torch", "flower", "dandelion", "poppy",
    "blue_orchid", "allium", "azure_bluet", "oxeye_daisy", "cornflower",
    "lily_of_the_valley", "wheat", "sweet_berry_bush", "dead_bush",
}
IDENT = re.compile(r"^[a-z0-9_]+$")


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def item_name(value):
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("block", ""))))
    value = str(value or "").removeprefix("minecraft:")
    return value if IDENT.fullmatch(value) else ""


def rows(value):
    return [dict(x) for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def position(value):
    if isinstance(value, dict):
        value = value.get("position", value)
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result = [number(value[i], math.inf) for i in range(3)]
    return result if all(math.isfinite(x) for x in result) else None


def distance(a, b):
    if a is None or b is None:
        return 999.0
    try:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
    except (TypeError, IndexError):
        return 999.0


def cell(point, size=4):
    if point is None:
        return None
    return (math.floor(point[0] / size), math.floor(point[1] / size), math.floor(point[2] / size))


def coord_text(point):
    return "%.2f %.2f %.2f" % (number(point[0]), number(point[1]), number(point[2]))


def counts_from(source):
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source["items"]
    result = {}
    if isinstance(source, dict):
        iterator = source.items()
        for key, value in iterator:
            name = item_name(key)
            count = value.get("count", 0) if isinstance(value, dict) else value
            count = max(0, int(number(count)))
            if name and count:
                result[name] = result.get(name, 0) + count
    elif isinstance(source, list):
        for row in source:
            if not isinstance(row, dict):
                continue
            name = item_name(row)
            count = max(0, int(number(row.get("count", 0))))
            if name and count:
                result[name] = result.get(name, 0) + count
    return result


def inventory(obs):
    """Prefer the audit, but use labelled personal views while an audit catches up."""
    for key in ("inventory", "inventory_authoritative"):
        value = counts_from(obs.get(key))
        if value:
            return value
    value = counts_from(obs.get("inventory_slots"))
    if value:
        return value
    value = counts_from(obs.get("inventory_client"))
    if value:
        return value
    return {}


def feedback_text(obs):
    values = []
    for key in ("action_feedback", "action_result", "last_result", "feedback"):
        value = obs.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (dict, list)):
            try:
                value = json.dumps(value, sort_keys=True, default=str)
            except Exception:
                value = str(value)
        values.append(str(value))
    return " ".join(values).lower()


def failure_feedback(text):
    return any(word in text for word in (
        "failed", "failure", "error", "unreachable", "timeout", "cannot",
        "rejected", "illegal", "not enough", "not in reach", "no path",
        "busy", "invalid", "out_of_reach", "unknown_or_stale",
    ))


def success_feedback(text):
    return any(word in text for word in (
        "success", "started", "completed", "crafted", "placed", "collected",
        "equipped", "opened", "slept", "accepted",
    )) and not failure_feedback(text)


def exact_coords(identifier):
    """Decode only coordinate-bearing public candidate IDs, never invent terrain."""
    text = str(identifier or "")
    if ":" not in text:
        return None
    tail = text.split(":", 1)[1].split(",")
    if len(tail) != 3:
        return None
    point = [number(x, math.inf) for x in tail]
    return point if all(math.isfinite(x) for x in point) else None


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.last_posted = "noop"
        self.active = None
        self.cancel_pending = False
        self.feedback_marker = None
        self.next_work = 0.0

        # A failure is attached to a spatial goal, not to a count-bearing string.
        self.bad_until = {}
        self.cluster_bad_until = {}
        self.failure_count = {}
        self.approaches = {}
        self.recent_gotos = {}
        self.visited_cells = set()
        self.frontier_failures = {}
        self.last_frontier = None

        self.furnace_started = {}
        self.table_ever = False
        self.bed_ever = False
        self.last_inventory = {}
        self.last_world_mark = None
        self.last_progress = 0.0
        self.stall_count = 0
        self.phase = "opening"

        self.macro = "progress"
        self.macro_version = 0
        self.astra_future = None
        self.astra_session = None
        self.astra_next = 28.0

        self.jev_future = None
        self.jev_started = 0.0
        self.jev_seq = None
        self.jev_episode = None
        self.jev_version = 0
        self.jev_options = ()
        self.jev_committed = False
        self.jev_cooldown_until = 0.0

    def now(self):
        return self.ctx.clock.now()

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def legal(self, action):
        return str(action).split(" ", 1)[0] in self.allowed

    def facts(self, obs):
        me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
        origin = position(me) or [0.0, 64.0, 0.0]
        blocks = []
        stands = []
        for raw in rows(obs.get("candidates")):
            if not raw.get("id"):
                continue
            row = dict(raw)
            row["_name"] = item_name(raw.get("name", raw.get("block", "")))
            row["_pos"] = position(raw)
            row["_distance"] = number(raw.get("distance"), distance(origin, row["_pos"]))
            if raw.get("kind") == "stand" or str(raw.get("id")).startswith("s:"):
                stands.append(row)
            elif raw.get("kind") == "block" or str(raw.get("id")).startswith("b:"):
                blocks.append(row)
        entities = []
        for raw in rows(obs.get("entities", obs.get("nearby_entities", []))):
            row = dict(raw)
            row["_name"] = item_name(raw.get("name", raw.get("type", "")))
            row["_pos"] = position(raw)
            row["_distance"] = number(raw.get("distance"), distance(origin, row["_pos"]))
            entities.append(row)
        return me, origin, inventory(obs), blocks, stands, entities

    def world_mark(self, obs, me, inv):
        progress = obs.get("progress")
        score = number(obs.get("score"), number(progress.get("score", 0) if isinstance(progress, dict) else 0))
        adv = tuple(sorted(str(x) for x in (obs.get("advancements") or [])))
        bed = str(me.get("bed_position"))
        return (tuple(sorted(inv.items())), round(score, 2), adv, bed,
                str(me.get("dimension", "")), bool(me.get("dead")))

    def observe_progress(self, obs, me, inv):
        mark = self.world_mark(obs, me, inv)
        now = self.now()
        if self.last_world_mark is None:
            self.last_world_mark = mark
            self.last_progress = now
        elif mark != self.last_world_mark:
            self.last_world_mark = mark
            self.last_progress = now
            self.stall_count = 0
        if inv != self.last_inventory:
            self.last_inventory = dict(inv)
            self.last_progress = now
            self.stall_count = 0
        elif now - self.last_progress > 45.0:
            self.stall_count += 1
            self.last_progress = now
        self.table_ever = self.table_ever or bool(inv.get("crafting_table"))
        self.bed_ever = self.bed_ever or any(k.endswith("_bed") and v for k, v in inv.items())
        self.bed_ever = self.bed_ever or bool(me.get("bed_position"))
        self.bed_ever = self.bed_ever or any("sleep_in_bed" in str(x) or "set_spawn" in str(x)
                                             for x in (obs.get("advancements") or []))

    def goal_key(self, action):
        parts = str(action or "").split()
        if not parts:
            return ""
        verb = parts[0]
        if verb == "gather" and len(parts) >= 2:
            point = exact_coords(parts[1])
            return "gather:" + ("%.0f,%.0f,%.0f" % tuple(point) if point else parts[1])
        if verb == "place" and len(parts) >= 3:
            point = exact_coords(parts[2])
            return "place:" + parts[1] + ":" + ("%.0f,%.0f,%.0f" % tuple(point) if point else parts[2]) + ":" + (parts[3] if len(parts) > 3 else "up")
        if verb == "goto" and len(parts) >= 4:
            return "goto:" + ",".join("%.0f" % number(x) for x in parts[1:4])
        if verb == "navigate" and len(parts) >= 2:
            return "navigate:" + parts[1]
        if verb in {"open_container", "sleep", "activate_block"} and len(parts) >= 2:
            return verb + ":" + parts[1]
        return str(action)

    def resource_cluster(self, row):
        p = row.get("_pos")
        return "resource:%s:%s" % (row.get("_name", ""), cell(p, 5))

    def blocked(self, action, now=None):
        now = self.now() if now is None else now
        key = self.goal_key(action)
        if self.bad_until.get(key, -1.0) > now:
            return True
        if str(action).startswith("gather "):
            parts = str(action).split()
            if len(parts) >= 2:
                p = exact_coords(parts[1])
                # Block the cluster after repeated failures; one bad log should
                # not blacklist a whole tree immediately.
                cluster = "gather-cell:" + str(cell(p, 5))
                if self.cluster_bad_until.get(cluster, -1.0) > now:
                    return True
        return False

    def invalidate(self, action, now, reason, obs=None):
        if not action or action in {"noop", "cancel"}:
            return
        key = self.goal_key(action)
        verb = str(action).split()[0]
        n = self.failure_count.get(key, 0) + 1
        self.failure_count[key] = n
        if verb == "gather":
            duration = min(360.0, 100.0 + 80.0 * min(n, 3))
            parts = str(action).split()
            p = exact_coords(parts[1]) if len(parts) > 1 else None
            if n >= 2:
                self.cluster_bad_until["gather-cell:" + str(cell(p, 5))] = now + 420.0
        elif verb == "place":
            # A placement has a two-cell geometric goal.  Do not try this same
            # support/target again merely because a cooldown expired.
            duration = min(700.0, 260.0 + 120.0 * min(n, 3))
        elif verb in {"goto", "navigate"}:
            duration = min(150.0, 35.0 + 35.0 * min(n, 3))
            self.frontier_failures[key] = now + duration
        else:
            duration = min(90.0, 20.0 + 15.0 * min(n, 3))
        self.bad_until[key] = max(self.bad_until.get(key, 0.0), now + duration)
        if self.active and self.active.get("action") == action:
            self.active = None
        self.next_work = now + 0.25
        self.log("goal_invalidated", action=action, goal=key, reason=str(reason)[:160],
                 blocked_s=round(duration, 1), failures=n)

    def feedback_changed(self, obs):
        marker = feedback_text(obs)
        changed = marker != self.feedback_marker
        if changed:
            self.feedback_marker = marker
        return marker, changed

    def action_timeout(self, action):
        verb = str(action).split()[0] if action else ""
        if verb == "gather":
            return 30.0
        if verb in {"goto", "navigate"}:
            return 19.0
        if verb == "place":
            return 11.0
        if verb in {"craft", "equip", "hotbar", "furnace_start", "furnace_take", "eat", "sleep"}:
            return 9.0
        if verb in {"attack", "activate_entity", "activate_block", "use_item"}:
            return 4.0
        return 7.0

    def block_change_mark(self, obs):
        changes = rows(obs.get("block_changes"))
        if not changes:
            return ()
        return tuple((str(x.get("position")), str(x.get("name")), str(x.get("at"))) for x in changes[-8:])

    def active_progress(self, obs, me, inv):
        active = self.active
        if active is None:
            return False
        action = active["action"]
        verb = action.split()[0]
        if verb == "respawn" and not me.get("dead"):
            return True
        if inv != active["inventory"]:
            return True
        if str(me.get("dimension", "")) != str(active.get("dimension", "")):
            return True
        if item_name(me.get("held_item")) != active.get("held"):
            return True
        if str(me.get("bed_position")) != str(active.get("bed")):
            return True
        if number(me.get("health"), 20) != number(active.get("health"), 20):
            return True
        if number(me.get("food"), 20) != number(active.get("food"), 20):
            return True
        if verb in {"goto", "navigate"} and distance(position(me), active.get("position")) >= 0.65:
            return True
        if verb == "place":
            parts = action.split()
            support = exact_coords(parts[2]) if len(parts) > 2 else None
            face = parts[3] if len(parts) > 3 else "up"
            target = list(support) if support else None
            if target is not None:
                if face == "up":
                    target[1] += 1
                elif face == "down":
                    target[1] -= 1
                elif face == "east":
                    target[0] += 1
                elif face == "west":
                    target[0] -= 1
                elif face == "south":
                    target[2] += 1
                elif face == "north":
                    target[2] -= 1
                wanted = item_name(parts[1]) if len(parts) > 1 else ""
                for row in rows(obs.get("candidates")):
                    if item_name(row.get("name")) == wanted and distance(position(row), target) < 0.2:
                        return True
            if active.get("item_before", 0) > inv.get(item_name(parts[1]) if len(parts) > 1 else "", 0):
                return True
        if self.block_change_mark(obs) != active.get("block_changes") and verb in {"gather", "place", "dig"}:
            return True
        if verb == "attack" and not obs.get("task") and self.now() - active["started"] >= 0.75:
            return True
        if verb in {"noop", "controls", "shield"}:
            return True
        return success_feedback(feedback_text(obs)) and feedback_text(obs) != active.get("feedback", "")

    def update_active(self, obs, me, inv, feedback, feedback_is_new):
        if self.active is None:
            return
        action = self.active["action"]
        now = self.now()
        if feedback_is_new and failure_feedback(feedback):
            self.invalidate(action, now, "adapter_feedback", obs)
            return
        if self.active_progress(obs, me, inv) and not obs.get("task"):
            self.log("transaction_complete", action=action, elapsed=round(now - self.active["started"], 2))
            self.active = None
            self.next_work = now + 0.15
            return
        elapsed = now - self.active["started"]
        limit = self.action_timeout(action)
        if elapsed >= limit + (5.0 if obs.get("task") else 0.0):
            self.invalidate(action, now, "deadline_without_postcondition", obs)

    def post(self, action, obs, purpose=""):
        if self.ctx.done:
            return "noop"
        action = str(action or "noop").strip() or "noop"
        if not self.legal(action):
            action = "noop"
        verb = action.split()[0]
        repeatable = verb in {"noop", "controls", "attack", "shield", "cancel"}
        # Identical one-shots are latched by the adaptor.  A noop is an explicit
        # arm between retries and also makes every deadline tick fresh.
        if action == self.last_posted and not repeatable:
            action = "noop"
            verb = "noop"
        self.ctx.act(action)
        self.last_posted = action
        now = self.now()
        if action == "noop":
            return action
        if verb == "cancel":
            self.active = None
            self.cancel_pending = True
            return action
        if verb in {"controls", "shield"}:
            return action
        me, p, inv, _, _, _ = self.facts(obs if isinstance(obs, dict) else {})
        self.active = {
            "action": action,
            "started": now,
            "position": list(p),
            "dimension": me.get("dimension", ""),
            "held": item_name(me.get("held_item")),
            "bed": str(me.get("bed_position")),
            "health": number(me.get("health"), 20),
            "food": number(me.get("food"), 20),
            "inventory": dict(inv),
            "item_before": inv.get(item_name(action.split()[1]), 0) if verb == "place" and len(action.split()) > 1 else 0,
            "block_changes": self.block_change_mark(obs if isinstance(obs, dict) else {}),
            "feedback": feedback_text(obs if isinstance(obs, dict) else {}),
        }
        if verb == "goto" and len(action.split()) >= 4:
            self.last_frontier = [number(x) for x in action.split()[1:4]]
            target_cell = cell(self.last_frontier)
            if target_cell is not None:
                self.visited_cells.add(target_cell)
            self.recent_gotos[self.goal_key(action)] = now
        if verb == "furnace_start" and len(action.split()) >= 2:
            self.furnace_started[action.split()[1]] = now
        self.log("posted", action=action, purpose=purpose, phase=self.phase)
        return action

    def emergency(self, obs, me, p, inv, blocks, stands, entities):
        now = self.now()
        if me.get("dead"):
            if self.active and str(self.active.get("action", "")).split()[0] == "respawn":
                return None, "finish respawn"
            return "respawn", "ordinary respawn after death"
        if number(obs.get("observation_age_s"), 0) > 3.0:
            return ("cancel" if obs.get("task") or self.active else "noop"), "stale observation"
        if me.get("is_in_lava") or me.get("is_in_fire"):
            if obs.get("task") or self.active:
                return "cancel", "interrupt continuous fire/lava hazard"
            return "controls forward,jump", "escape fire/lava"
        if me.get("is_in_water") and number(me.get("oxygen"), 300) < 80:
            if obs.get("task") or self.active:
                return "cancel", "interrupt drowning risk"
            return "controls jump", "surface for oxygen"
        food = next((x for x in FOODS if inv.get(x)), None)
        active_verb = str(self.active.get("action", "")).split()[0] if self.active else ""
        held = item_name(me.get("held_item"))
        if food and (number(me.get("health"), 20) <= 7 or number(me.get("food"), 20) <= 8):
            if active_verb in {"eat", "equip"}:
                return None, "finish food handling"
            if held != food:
                if obs.get("task") or self.active:
                    return "cancel", "interrupt work before equipping food"
                return "equip " + food, "equip carried food before eating"
            if obs.get("task") or self.active:
                return "cancel", "interrupt work before eating"
            return "eat " + food, "restore health or hunger"
        origin = p
        threats = []
        for entity in entities:
            if not entity.get("hostile") or not entity.get("visible"):
                continue
            if entity.get("_pos") is not None and distance(entity["_pos"], origin) <= 6.5:
                threats.append(entity)
        if not threats:
            return None, ""
        if active_verb in {"equip", "attack"}:
            return None, "finish the defensive action"
        if obs.get("task") or self.active:
            return "cancel", "interrupt work for close hostile"
        close = [x for x in threats if distance(x.get("_pos"), origin) <= 3.2 and x.get("id") is not None]
        weapon = next((x for x in WEAPONS if inv.get(x)), None)
        held = item_name(me.get("held_item"))
        if close and weapon:
            if held != weapon:
                return "equip " + weapon, "equip weapon against close hostile"
            return "attack " + str(close[0]["id"]), "attack a visible in-range hostile"
        if inv.get("shield"):
            return "shield on", "block a visible hostile while retreating"
        safe = [x for x in stands if x.get("_pos") and 5.0 < x["_distance"] < 24.0]
        if safe:
            safe.sort(key=lambda x: distance(x["_pos"], threats[0].get("_pos") or origin), reverse=True)
            q = safe[0]["_pos"]
            return "goto " + coord_text(q), "retreat to an observed safe stand"
        return "controls back,jump", "back away from a hostile"

    def stand_for(self, target, stands, origin, now, goal_bias=True):
        tp = target.get("_pos") if isinstance(target, dict) else target
        choices = []
        for stand in stands:
            sp = stand.get("_pos")
            if sp is None or distance(sp, origin) > 48.0:
                continue
            if distance(sp, origin) < 0.75:
                continue
            action = "goto " + coord_text(sp)
            if self.blocked(action, now):
                continue
            if self.recent_gotos.get(self.goal_key(action), -1000.0) + 8.0 > now:
                continue
            toward = distance(sp, tp) if tp is not None else 0.0
            # Safe stands are supplied by the adaptor.  Prefer a stand next to
            # the resource, not a farthest point that merely grows a frontier.
            score = toward * (1.5 if goal_bias and tp is not None else 0.05)
            score += distance(sp, origin) * 0.10
            score += 7.0 if cell(sp) in self.visited_cells else 0.0
            choices.append((score, stand))
        if not choices:
            return None
        choices.sort(key=lambda pair: pair[0])
        return choices[0][1]

    def target_block(self, blocks, names, now, reachable_only=False):
        names = set(names)
        candidates = []
        for row in blocks:
            name = row.get("_name", "")
            matched = name in names or any(str(x).startswith("_") and name.endswith(str(x)) for x in names)
            if not matched or row.get("harvestable") is False or not row.get("id"):
                continue
            if self.blocked("gather %s 1" % row["id"], now):
                continue
            if reachable_only and not row.get("in_reach"):
                continue
            # A raycast-hidden log can still be an exposed, loaded target.  It
            # is legal as an approach goal but never as a direct gather target.
            # For non-wood resources, retain the conservative visible/in-reach
            # filter because hidden solid interiors are not a map oracle.
            if not reachable_only and not row.get("visible") and not row.get("in_reach"):
                if not any(str(x).startswith("_") and name.endswith(str(x)) for x in names):
                    continue
            if not reachable_only and self.approaches.get(self.goal_key("gather " + str(row["id"]) + " 1"), 0) >= 3:
                continue
            candidates.append(row)
        candidates.sort(key=lambda x: (not bool(x.get("in_reach")), not bool(x.get("visible")), x.get("_distance", 999)))
        return candidates[0] if candidates else None

    def add_resource(self, options, blocks, stands, origin, names, purpose, priority, now, count=1):
        target = self.target_block(blocks, names, now, reachable_only=True)
        if target is not None:
            self.add_option(options, "gather %s %d" % (target["id"], count), purpose, priority, now)
            return True
        # A visible but distant target is a real goal; do not call gather on it.
        distant = self.target_block(blocks, names, now, reachable_only=False)
        if distant is not None:
            stand = self.stand_for(distant, stands, origin, now, goal_bias=True)
            if stand is not None:
                self.approaches[self.goal_key("gather " + str(distant["id"]) + " 1")] = self.approaches.get(self.goal_key("gather " + str(distant["id"]) + " 1"), 0) + 1
                self.add_option(options, "goto " + coord_text(stand["_pos"]),
                                purpose + " by an observed safe stand", priority - 3, now)
                return True
        return False

    def add_option(self, options, action, purpose, priority, now):
        action = str(action or "").strip()
        if not action or not self.legal(action) or self.blocked(action, now):
            return
        if any(row["action"] == action for row in options):
            return
        options.append({"action": action, "purpose": str(purpose), "priority": float(priority)})

    def placement_plan(self, obs, item, blocks, stands, origin, now):
        """Return a fresh support/stand/place transaction from the current view."""
        if not item:
            return None
        occupied = {}
        for row in blocks:
            q = row.get("_pos")
            if q is not None:
                occupied[tuple(round(x) for x in q)] = row.get("_name", "")
        options = []
        for support in blocks:
            sp = support.get("_pos")
            if sp is None or support.get("visible") is False or not support.get("in_reach"):
                continue
            if support.get("_name") not in SOLID_SUPPORTS:
                continue
            target = [round(sp[0]), round(sp[1] + 1), round(sp[2])]
            target_name = occupied.get(tuple(target))
            if target_name and target_name not in REPLACEABLE:
                continue
            place = "place %s %s up" % (item, support["id"])
            if self.blocked(place, now):
                continue
            # A local collision label can reject a geometrically occupied target
            # even when it was not emitted as a candidate.  Only reject known solid.
            collision = {}
            for raw in ((obs.get("world") or {}).get("local_collision") or []):
                if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                    collision[(int(number(raw[0])), int(number(raw[1])), int(number(raw[2])))] = str(raw[3])
            rel = tuple(int(target[i] - math.floor(origin[i])) for i in range(3))
            if collision.get(rel) == "solid":
                continue
            stand_choices = []
            for stand in stands:
                q = stand.get("_pos")
                if q is None or distance(q, origin) > 48.0:
                    continue
                # Keep the player out of the block being placed, but remain close
                # enough for an ordinary right-click reach.
                dtarget = distance(q, target)
                if dtarget < 0.9 or dtarget > 4.6:
                    continue
                if abs(q[1] - target[1]) > 2.0:
                    continue
                goto = "goto " + coord_text(q)
                if self.blocked(goto, now):
                    continue
                stand_choices.append((dtarget + 0.08 * distance(q, origin) +
                                      (5.0 if cell(q) in self.visited_cells else 0.0), stand))
            stand_choices.sort(key=lambda x: x[0])
            chosen_stand = stand_choices[0][1] if stand_choices else None
            current_safe = 0.9 <= distance(origin, target) <= 4.6
            if not current_safe and chosen_stand is None:
                continue
            score = distance(sp, origin) + (0.4 if current_safe else 0.0)
            options.append((score, support, chosen_stand, place, current_safe))
        if not options:
            return None
        options.sort(key=lambda x: x[0])
        _, support, stand, place, current_safe = options[0]
        return {"support": support, "stand": stand, "place": place, "current_safe": current_safe}

    def near_table(self, blocks):
        candidates = [x for x in blocks if x.get("_name") == "crafting_table" and x.get("_pos") is not None]
        candidates.sort(key=lambda x: x.get("_distance", 999))
        return candidates[0] if candidates and candidates[0].get("_distance", 999) <= 4.6 else None

    def table_any(self, blocks):
        candidates = [x for x in blocks if x.get("_name") == "crafting_table" and x.get("_pos") is not None]
        candidates.sort(key=lambda x: x.get("_distance", 999))
        return candidates[0] if candidates else None

    def table_waypoint(self, table, stands, origin, now):
        if table is None:
            return None
        stand = self.stand_for(table, stands, origin, now, goal_bias=True)
        return "goto " + coord_text(stand["_pos"]) if stand is not None else None

    def frontier(self, options, blocks, stands, origin, now, goal=None, purpose="Scout a new safe frontier", priority=20):
        # If a known resource exists, finish its local approach before arbitrary
        # scouting.  This is the key difference from a farthest-stand orbit.
        if goal is not None:
            stand = self.stand_for({"_pos": goal}, stands, origin, now, goal_bias=True)
            if stand is not None:
                self.add_option(options, "goto " + coord_text(stand["_pos"]), purpose, priority + 12, now)
                return True
        choices = []
        for stand in stands:
            q = stand.get("_pos")
            if q is None or distance(q, origin) < 1.0 or distance(q, origin) > 42.0:
                continue
            action = "goto " + coord_text(q)
            if self.blocked(action, now) or self.recent_gotos.get(self.goal_key(action), -1000.0) + 12.0 > now:
                continue
            novelty = 100.0 if cell(q) not in self.visited_cells else 0.0
            direction_change = 0.0 if self.last_frontier is None else distance(q, self.last_frontier) * 0.05
            score = novelty + min(distance(q, origin), 30.0) + direction_change
            choices.append((score, stand))
        if not choices:
            return False
        choices.sort(key=lambda pair: pair[0], reverse=True)
        stand = choices[0][1]
        self.add_option(options, "goto " + coord_text(stand["_pos"]), purpose, priority, now)
        return True

    def visible_animal(self, entities, origin):
        animals = [x for x in entities if x.get("_name") in PASSIVE and x.get("visible") and x.get("_pos") is not None]
        animals.sort(key=lambda x: x.get("_distance", distance(x["_pos"], origin)))
        return animals[0] if animals else None

    def options_for(self, obs, me, origin, inv, blocks, stands, entities):
        now = self.now()
        options = []
        planks = sum(v for k, v in inv.items() if k.endswith("_planks"))
        logs = sum(v for k, v in inv.items() if k.endswith(WOOD_SUFFIXES))
        sticks = inv.get("stick", 0)
        food_count = sum(inv.get(x, 0) for x in FOODS)
        cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
        raw_iron = inv.get("raw_iron", 0) + inv.get("iron_ore", 0)
        iron = inv.get("iron_ingot", 0)
        wood_pick = next((x for x in PICKS if inv.get(x)), None)
        stone_pick = any(inv.get(x) for x in ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"))
        table = self.near_table(blocks)
        any_table = self.table_any(blocks)
        self.table_ever = self.table_ever or bool(inv.get("crafting_table")) or any_table is not None

        def craft_for_log(count):
            log = next((k for k, v in inv.items() if v and k.endswith(WOOD_SUFFIXES)), None)
            if not log:
                return None
            return log.replace("_log", "_planks").replace("_stem", "_planks")

        # A carried table must first become a real, observed nearby workstation.
        if not logs and not planks and not inv.get("crafting_table") and any_table is None:
            if not self.add_resource(options, blocks, stands, origin, set(WOOD_SUFFIXES),
                                     "Collect an exposed log for the opening", 120, now, 1):
                self.frontier(options, blocks, stands, origin, now,
                              purpose="Approach the nearest observed wood goal", priority=100)
            self.phase = "find_wood"
            return self.rank(options), self.phase

        if not inv.get("crafting_table") and any_table is None:
            plank_item = craft_for_log(4)
            if planks < 4 and plank_item:
                before = len(options)
                self.add_option(options, "craft %s 4" % plank_item,
                                "Convert a carried log into workstation planks", 120, now)
                if len(options) == before and not self.add_resource(
                        options, blocks, stands, origin, set(WOOD_SUFFIXES),
                        "Collect another exposed log after a failed plank transaction", 116, now, 1):
                    self.frontier(options, blocks, stands, origin, now,
                                  purpose="Search for wood after a failed target", priority=98)
            elif planks >= 4:
                before = len(options)
                self.add_option(options, "craft crafting_table 1",
                                "Craft the opening workstation", 120, now)
                if len(options) == before and not self.add_resource(
                        options, blocks, stands, origin, set(WOOD_SUFFIXES),
                        "Collect a spare log after a failed table transaction", 116, now, 1):
                    self.frontier(options, blocks, stands, origin, now,
                                  purpose="Search for wood after a failed table transaction", priority=98)
            elif not self.add_resource(options, blocks, stands, origin, set(WOOD_SUFFIXES),
                                       "Collect another exposed log for the workstation", 116, now, 1):
                self.frontier(options, blocks, stands, origin, now,
                              purpose="Search for wood after a failed target", priority=98)
            self.phase = "make_table"
            return self.rank(options), self.phase

        if inv.get("crafting_table") and table is None:
            plan = self.placement_plan(obs, "crafting_table", blocks, stands, origin, now)
            if plan is not None:
                if plan["current_safe"]:
                    self.add_option(options, plan["place"],
                                    "Place the carried table on a verified free top face", 116, now)
                elif plan["stand"] is not None:
                    self.add_option(options, "goto " + coord_text(plan["stand"]["_pos"]),
                                    "Stage at a safe stand before placing the table", 118, now)
            if not options:
                self.frontier(options, blocks, stands, origin, now,
                              purpose="Find a different safe support for the workstation", priority=92)
            self.phase = "stage_table"
            return self.rank(options), self.phase

        if table is None and any_table is not None:
            waypoint = self.table_waypoint(any_table, stands, origin, now)
            if waypoint:
                self.add_option(options, waypoint, "Approach the observed workstation by a safe stand", 113, now)
            self.phase = "approach_table"
            return self.rank(options), self.phase

        # Food is treated as a branch, but only an observed source can enter it.
        if food_count < 3 or number(me.get("food"), 20) <= 12:
            if self.add_resource(options, blocks, stands, origin,
                                 {"hay_block", "sweet_berry_bush", "melon"},
                                 "Collect an exposed food source before a long trip",
                                 111 if self.macro == "food" else 82, now, 1):
                pass

        if wood_pick is None:
            plank_item = craft_for_log(8)
            # A table consumes the first log.  Do not assume that one log also
            # supplies the pick and sticks; ask for another observed log when
            # the exact craft is blocked or the carried stock is insufficient.
            needed_planks = 5 if sticks < 2 else 3
            if planks < needed_planks:
                added = False
                if plank_item:
                    log_item = next((k for k, v in inv.items()
                                     if v and k.endswith(WOOD_SUFFIXES) and
                                     k.replace("_log", "_planks").replace("_stem", "_planks") == plank_item), "")
                    available_logs = inv.get(log_item, 0)
                    craft_count = min(8, max(4, available_logs * 4))
                    before = len(options)
                    self.add_option(options, "craft %s %d" % (plank_item, craft_count),
                                    "Make enough planks for the first mining tool", 108, now)
                    added = len(options) > before
                if not added:
                    self.add_resource(options, blocks, stands, origin, set(WOOD_SUFFIXES),
                                      "Collect more exposed wood for tool materials", 106, now, 1)
            elif sticks < 2:
                before = len(options)
                self.add_option(options, "craft stick 4", "Craft tool handles", 107, now)
                if len(options) == before:
                    self.add_resource(options, blocks, stands, origin, set(WOOD_SUFFIXES),
                                      "Collect wood after a failed handle transaction", 104, now, 1)
            elif table is not None and table.get("id"):
                self.add_option(options, "craft wooden_pickaxe 1 %s" % table["id"],
                                "Craft a wooden pickaxe at the observed table", 109, now)
            elif table is not None:
                pass
            else:
                waypoint = self.table_waypoint(any_table, stands, origin, now)
                if waypoint:
                    self.add_option(options, waypoint, "Return to the workstation for tool crafting", 106, now)
            if not options:
                self.frontier(options, blocks, stands, origin, now,
                              purpose="Find the next observed wood or workstation", priority=80)
            self.phase = "wood_pick"
            return self.rank(options), self.phase

        if not stone_pick:
            if cobble < 3:
                self.add_resource(options, blocks, stands, origin, STONE_NAMES,
                                  "Mine exposed stone for the stone pickaxe", 103, now, 3)
            elif sticks < 2:
                self.add_option(options, "craft stick 4", "Make handles for the stone pickaxe", 102, now)
            elif table is not None:
                self.add_option(options, "craft stone_pickaxe 1 %s" % table["id"],
                                "Craft the stone pickaxe", 104, now)
            if not options:
                self.frontier(options, blocks, stands, origin, now,
                              purpose="Explore a new safe stand for exposed stone", priority=74)
            self.phase = "stone_pick"
            return self.rank(options), self.phase

        # A stone sword is a cheap, observable safety upgrade before mining farther.
        if not any(inv.get(x) for x in ("stone_sword", "iron_sword", "diamond_sword")) and cobble >= 2 and sticks >= 1 and table is not None:
            self.add_option(options, "craft stone_sword 1 %s" % table["id"],
                            "Craft a basic defensive weapon", 96, now)

        if not inv.get("coal") and not inv.get("charcoal"):
            self.add_resource(options, blocks, stands, origin, COAL_NAMES,
                              "Collect exposed furnace fuel", 78 if self.macro != "iron" else 88, now, 2)
        if raw_iron < 3:
            self.add_resource(options, blocks, stands, origin, IRON_NAMES,
                              "Mine exposed iron with the stone pickaxe", 99 if self.macro == "iron" else 91, now, 3)

        furnace_rows = [x for x in blocks if x.get("_name") == "furnace"]
        furnace_rows.sort(key=lambda x: x.get("_distance", 999))
        furnace = furnace_rows[0] if furnace_rows else None
        if raw_iron and furnace is None and not inv.get("furnace") and cobble >= 8 and table is not None:
            self.add_option(options, "craft furnace 1 %s" % table["id"],
                            "Craft a real furnace for smelting", 94, now)
        elif raw_iron and furnace is None and inv.get("furnace"):
            plan = self.placement_plan(obs, "furnace", blocks, stands, origin, now)
            if plan:
                self.add_option(options, plan["place"] if plan["current_safe"] else "goto " + coord_text(plan["stand"]["_pos"]),
                                "Place the carried furnace on a verified support", 93, now)
        elif raw_iron and furnace is not None and furnace.get("_distance", 999) <= 4.6:
            fuel = next((x for x in ("coal", "charcoal") if inv.get(x)), None)
            if fuel and str(furnace["id"]) not in self.furnace_started:
                self.add_option(options, "furnace_start %s raw_iron %d %s %d" % (
                    furnace["id"], min(raw_iron, 3), fuel, min(inv.get(fuel, 1), 3)),
                    "Load real iron and fuel into the furnace", 92, now)
            elif str(furnace["id"]) in self.furnace_started:
                started = self.furnace_started[str(furnace["id"])]
                if now - started >= 12.0:
                    self.add_option(options, "furnace_take " + str(furnace["id"]),
                                    "Retrieve finished iron after furnace time", 93, now)
                else:
                    self.next_work = max(self.next_work, started + 12.0)
        elif raw_iron and furnace is not None:
            waypoint = self.table_waypoint(furnace, stands, origin, now)
            if waypoint:
                self.add_option(options, waypoint, "Approach the observed furnace", 89, now)

        if iron >= 1 and not inv.get("shield") and planks >= 6 and table is not None:
            self.add_option(options, "craft shield 1 %s" % table["id"],
                            "Craft the shield milestone", 101, now)
        if iron >= 3 and not inv.get("bucket") and table is not None:
            self.add_option(options, "craft bucket 1 %s" % table["id"],
                            "Craft a bucket for later portal safety", 67, now)

        # Bed support is handled with the same geometry transaction as the table.
        if not self.bed_ever:
            beds = [x for x in blocks if x.get("_name", "").endswith("_bed")]
            night = number((obs.get("world") or {}).get("time_of_day"), 0) % 24000 >= 12500
            for bed in beds:
                if night and bed.get("in_reach"):
                    self.add_option(options, "sleep " + str(bed["id"]),
                                    "Set the ordinary respawn point in the observed bed", 86, now)
                elif not bed.get("in_reach"):
                    stand = self.stand_for(bed, stands, origin, now, goal_bias=True)
                    if stand:
                        self.add_option(options, "goto " + coord_text(stand["_pos"]),
                                        "Approach an observed bed safely", 76, now)
                break
            wool = next((x for x, v in inv.items() if x.endswith("_wool") and v >= 3), None)
            if wool and planks >= 3 and table is not None:
                bed_item = wool[:-5] + "_bed"
                self.add_option(options, "craft %s 1 %s" % (bed_item, table["id"]),
                                "Craft a bed from carried wool and planks", 80, now)
            bed_item = next((x for x in inv if x.endswith("_bed") and inv.get(x)), None)
            if bed_item and not beds:
                plan = self.placement_plan(obs, bed_item, blocks, stands, origin, now)
                if plan:
                    self.add_option(options, plan["place"] if plan["current_safe"] else "goto " + coord_text(plan["stand"]["_pos"]),
                                    "Place the carried bed on a free observed support", 83, now)

        # A visible passive mob is an optional food/wool branch, never a blind ID.
        animal = self.visible_animal(entities, origin)
        if animal is not None and (food_count < 5 or not self.bed_ever):
            weapon = next((x for x in WEAPONS if inv.get(x)), None)
            if weapon and animal.get("_distance", 999) <= 3.2 and animal.get("id") is not None:
                if item_name(me.get("held_item")) != weapon:
                    self.add_option(options, "equip " + weapon, "Equip the carried weapon for food or wool", 73, now)
                else:
                    self.add_option(options, "attack " + str(animal["id"]), "Harvest a visible passive animal", 72, now)
            elif animal.get("_pos") is not None and animal.get("_distance", 999) <= 40:
                self.add_option(options, "goto " + coord_text(animal["_pos"]),
                                "Approach a visible passive animal", 62, now)

        # If no milestone action is available, scout; if a resource target is
        # still listed, scout toward that target rather than away from it.
        if not options:
            target = self.target_block(blocks, set(WOOD_SUFFIXES) | STONE_NAMES | IRON_NAMES | COAL_NAMES,
                                       now, reachable_only=False)
            self.frontier(options, blocks, stands, origin, now,
                          goal=target.get("_pos") if target else None,
                          purpose="Scout a new loaded safe frontier", priority=45)
        if not options and self.stall_count >= 1:
            self.frontier(options, blocks, stands, origin, now,
                          purpose="Break a local stall with a different observed stand", priority=35)
        self.phase = "resource_progress"
        return self.rank(options), self.phase

    def rank(self, options):
        options.sort(key=lambda x: (-x.get("priority", 0), x.get("action", "")))
        return options[:7]

    def maybe_finish_astra(self):
        if self.astra_future is None or not self.astra_future.done():
            return
        future = self.astra_future
        self.astra_future = None
        try:
            result = future.result()
            text = str(getattr(result, "text", ""))
            start, end = text.find("{"), text.rfind("}")
            parsed = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
            mode = str(parsed.get("mode", parsed.get("priority", ""))).lower()
            aliases = {"balanced": "progress", "iron": "progress", "explore": "scout", "food": "food", "bed": "food"}
            mode = aliases.get(mode, mode)
            if mode in {"progress", "food", "scout", "recover"} and mode != self.macro:
                self.macro = mode
                self.macro_version += 1
                self.log("astra_macro", mode=mode, version=self.macro_version)
            self.astra_session = getattr(result, "session_id", self.astra_session)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception:
            self.log("astra_unusable")

    def maybe_start_astra(self, obs, me, inv, options):
        self.maybe_finish_astra()
        now = self.now()
        remaining = self.ctx.budget.remaining()
        if self.astra_future is not None or now < self.astra_next:
            return
        if remaining.get("system2_calls", 0) <= 1 or remaining.get("wall_s", 0) < 180:
            return
        if now - self.last_progress < 35.0 and self.stall_count < 1:
            return
        snapshot = {
            "objective": "Free the End in ordinary Minecraft survival",
            "phase": self.phase,
            "macro": self.macro,
            "dimension": str(me.get("dimension", "overworld")),
            "health_band": "critical" if number(me.get("health"), 20) < 7 else "low" if number(me.get("health"), 20) < 12 else "stable",
            "food": int(number(me.get("food"), 20)),
            "inventory": {k: int(v) for k, v in sorted(inv.items()) if v and (k in FOODS or k in {"crafting_table", "stick", "cobblestone", "raw_iron", "iron_ingot", "shield", "furnace", "bucket"} or k.endswith(("_log", "_planks", "_wool", "_bed")))},
            "available": [{"action": x["action"], "purpose": x["purpose"]} for x in options[:7]],
            "stall_count": self.stall_count,
            "remaining_wall_s": int(number(remaining.get("wall_s"), 0)),
        }
        prompt = (
            "You are a slow macro adviser for a vanilla Minecraft survival controller. "
            "Code owns recipes, coordinates, legality, inventory counts, hazards and all exact commands. "
            "Choose one bias only: progress (finish the next verified prerequisite), food (secure food/bed), "
            "scout (seek a new loaded safe area), or recover (conservative safety). Do not emit commands, "
            "coordinates, or prose. Reply exactly JSON {\"mode\":\"progress|food|scout|recover\"}. "
            "This is an adaptation after a local stall, not a license to skip prerequisites. State: "
            + json.dumps(snapshot, sort_keys=True, default=str)
        )
        session = self.astra_session
        self.astra_future = self.ctx.spawn(lambda: self.ctx.think(prompt, session_id=session, timeout_s=120.0))
        self.astra_next = now + 185.0
        self.log("astra_started", phase=self.phase, stall_count=self.stall_count)

    def jev_state(self, obs, me, inv, entities, options):
        threats = []
        origin = position(me)
        for entity in entities:
            if not entity.get("hostile") or not entity.get("visible"):
                continue
            d = entity.get("_distance", distance(entity.get("_pos"), origin))
            threats.append({"kind": entity.get("_name", "hostile"),
                            "range": "melee" if d <= 3.2 else "near" if d <= 8 else "far",
                            "id": str(entity.get("id", ""))})
        return {
            "objective": "safe verified progress toward Free the End",
            "phase": self.phase,
            "macro_bias": self.macro,
            "dimension": str(me.get("dimension", "overworld")),
            "health_band": "critical" if number(me.get("health"), 20) < 7 else "low" if number(me.get("health"), 20) < 12 else "stable",
            "hunger_band": "hungry" if number(me.get("food"), 20) < 12 else "fed",
            "inventory": {k: int(v) for k, v in sorted(inv.items()) if v and (k in FOODS or k in {"crafting_table", "stick", "cobblestone", "raw_iron", "iron_ingot", "shield", "furnace", "bucket"} or k.endswith(("_log", "_planks", "_wool", "_bed")))},
            "visible_threats": threats[:6],
            "options": [{"command": x["action"], "purpose": x["purpose"]} for x in options],
            "rules": "Choose only an exact command in options. Coordinates, IDs and counts must never be invented. Code overrides lethal choices.",
        }

    def start_jev(self, obs, me, inv, entities, options):
        if self.jev_future is not None or not options or self.now() < self.jev_cooldown_until:
            return
        remaining = self.ctx.budget.remaining()
        if remaining.get("system1_calls", 0) <= 2:
            return
        state = self.jev_state(obs, me, inv, entities, options)
        criteria = {str(x["action"]): str(x["purpose"]) for x in options[:7]}
        questions = {
            "action": Choice(
                "Select the safest useful complete command from the supplied options for this exact phase. Prefer a verified prerequisite over scouting, but do not select work when visible threats make it unsafe.",
                criteria,
            ),
            "safe": Noul(
                "Does the supplied state lack a visible close hostile that should interrupt ordinary work?",
                {"true": "No visible close hostile is listed and the health band is not critical.",
                 "false": "A visible close hostile or critical health is explicitly listed."},
            ),
        }
        self.jev_future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=1.8))
        self.jev_started = self.now()
        self.jev_seq = obs.get("seq")
        self.jev_episode = obs.get("episode")
        self.jev_version = self.macro_version
        self.jev_options = tuple(x["action"] for x in options[:7])
        self.jev_committed = False
        self.log("jev_started", phase=self.phase, options=len(self.jev_options))

    def consume_jev(self, obs, options):
        if self.jev_future is None:
            return None
        future = self.jev_future
        now = self.now()
        if not future.done():
            if not self.jev_committed and now - self.jev_started >= 0.70:
                self.jev_committed = True
                fallback = options[0]["action"] if options else "noop"
                self.log("jev_deadline_fallback", action=fallback, age=round(now - self.jev_started, 2))
                return fallback
            return "__WAIT__"
        self.jev_future = None
        try:
            answers = future.result()
            if self.jev_committed:
                self.jev_cooldown_until = now + 0.20
                return None
            answer = answers.get("action") if isinstance(answers, dict) else None
            safe = answers.get("safe") if isinstance(answers, dict) else None
            selected = getattr(answer, "value", None)
            probabilities = getattr(answer, "probabilities", {}) or {}
            probability = number(probabilities.get(selected), 0.0) if isinstance(probabilities, dict) else 0.0
            current = {x["action"] for x in options}
            seq_now = number(obs.get("seq"), 0)
            seq_then = number(self.jev_seq, seq_now)
            fresh = (selected in current and obs.get("episode") == self.jev_episode and
                     seq_now - seq_then <= 10 and self.macro_version == self.jev_version)
            safe_probability = number(getattr(safe, "value", 1.0), 1.0)
            # Noul is advisory only: the code emergency gate is authoritative,
            # so a low safety answer cannot deadlock the whole episode.
            if fresh and probability >= 0.43:
                self.log("jev_selected", action=selected, probability=round(probability, 3),
                         safe_probability=round(safe_probability, 3), phase=self.phase)
                self.jev_cooldown_until = now + 0.05
                return selected
            self.log("jev_rejected", selected=str(selected), probability=round(probability, 3),
                     safe_probability=round(safe_probability, 3), fresh=fresh)
            self.jev_cooldown_until = now + 1.0
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception:
            self.jev_cooldown_until = now + 1.0
            self.log("jev_unusable")
        return None

    def run_episode(self, game):
        self.allowed = set(getattr(getattr(game, "action_spec", None), "names", ()))
        while not self.ctx.done:
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                self.post("noop", {}, "missing observation")
                self.ctx.wait_for_tick()
                continue
            now = self.now()
            me, origin, inv, blocks, stands, entities = self.facts(obs)
            self.observe_progress(obs, me, inv)
            feedback, feedback_is_new = self.feedback_changed(obs)
            self.update_active(obs, me, inv, feedback, feedback_is_new)

            urgent, reason = self.emergency(obs, me, origin, inv, blocks, stands, entities)
            if urgent:
                # A pending Jev answer is tied to the old menu.  It may finish in
                # the background, but it can no longer replace this safety action.
                if self.jev_future is not None:
                    self.jev_committed = True
                if urgent == "cancel" and self.last_posted == "cancel":
                    self.post("noop", obs, reason)
                else:
                    self.post(urgent, obs, reason)
                self.ctx.wait_for_tick()
                continue

            if self.cancel_pending:
                if obs.get("task"):
                    self.post("cancel", obs, "finish cancellation")
                    self.ctx.wait_for_tick()
                    continue
                self.cancel_pending = False

            if obs.get("task") or self.active is not None:
                # A bounded task owns the hands.  Reposting its command would
                # restart/latch it; noop is the lease renewal.
                self.post("noop", obs, "continue active transaction")
                self.ctx.wait_for_tick()
                continue

            if now < self.next_work:
                self.post("noop", obs, "brief postcondition observation lease")
                self.ctx.wait_for_tick()
                continue

            options, phase = self.options_for(obs, me, origin, inv, blocks, stands, entities)
            self.phase = phase
            self.maybe_start_astra(obs, me, inv, options)
            if not options:
                self.post("noop", obs, "no currently legal observed goal")
                self.ctx.wait_for_tick()
                continue

            chosen = self.consume_jev(obs, options)
            if chosen == "__WAIT__":
                self.post("noop", obs, "Jev pending; preserve world position")
                self.ctx.wait_for_tick()
                continue
            if chosen is None:
                self.start_jev(obs, me, inv, entities, options)
                if self.jev_future is not None and not self.jev_committed:
                    self.post("noop", obs, "Jev pending; preserve current safe state")
                    self.ctx.wait_for_tick()
                    continue
                chosen = options[0]["action"]
                self.log("compiled_fallback", action=chosen, phase=self.phase)
            current = {x["action"] for x in options}
            if chosen not in current:
                chosen = options[0]["action"]
            purpose = next((x["purpose"] for x in options if x["action"] == chosen), "compiled goal")
            self.post(chosen, obs, purpose)
            self.ctx.wait_for_tick()


def build(ctx):
    return Controller(ctx)
