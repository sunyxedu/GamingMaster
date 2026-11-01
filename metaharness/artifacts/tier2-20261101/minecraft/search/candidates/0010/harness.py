"""Candidate 0010: hand-staged interaction transactions for Minecraft survival.

The controller treats an item interaction as a small transaction rather than a
single command.  In particular, a table/furnace placement first equips the
item, then places it against a freshly observed support.  A failed support is
retired for the episode and the controller searches or moves on instead of
repeating the same placement forever.
"""
from __future__ import annotations

import json
import math
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


_ID = re.compile(r"^[bs]:-?\d+,-?\d+,-?\d+$")
_ITEM = re.compile(r"^[a-z0-9_]+$")

WOOD_SUFFIXES = ("_log", "_stem")
STONE = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON = {"iron_ore", "deepslate_iron_ore"}
COAL = {"coal_ore", "deepslate_coal_ore"}
FOOD_BLOCKS = {
    "hay_block", "sweet_berry_bush", "melon", "wheat", "carrots", "carrot",
    "potatoes", "potato", "beetroots", "beetroot", "cocoa",
}
FOODS = {
    "apple", "bread", "baked_potato", "carrot", "golden_carrot",
    "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
    "cooked_rabbit", "cooked_cod", "cooked_salmon", "sweet_berries",
    "melon_slice", "beef", "porkchop", "mutton", "chicken", "rabbit",
    "cod", "salmon", "potato", "beetroot", "wheat", "mushroom_stew",
}
ANIMALS = {"cow", "pig", "sheep", "chicken", "rabbit"}
HOSTILES = {
    "zombie", "skeleton", "creeper", "spider", "drowned", "husk", "stray",
    "witch", "phantom", "enderman", "slime", "magma_cube", "blaze", "ghast",
    "piglin", "piglin_brute", "zombified_piglin", "hoglin", "wither_skeleton",
    "ender_dragon", "shulker", "silverfish",
}
HAZARDS = {
    "lava", "fire", "soul_fire", "magma_block", "cactus", "powder_snow",
    "campfire", "soul_campfire", "pointed_dripstone",
}
SUPPORTS = {
    "stone", "cobblestone", "cobbled_deepslate", "deepslate", "dirt",
    "grass_block", "coarse_dirt", "sand", "gravel", "terracotta", "netherrack",
    "blackstone", "oak_log", "birch_log", "spruce_log", "jungle_log",
    "acacia_log", "dark_oak_log", "mangrove_log", "podzol", "mycelium",
    "dirt_path", "sandstone", "red_sand", "snow_block", "snow", "clay",
    "moss_block", "basalt", "end_stone", "crafting_table", "furnace",
}
WEAPONS = ("iron_sword", "stone_sword", "diamond_sword", "stone_axe", "iron_axe", "wooden_axe")
PICKS = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def rows(value):
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def clean_item(value):
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", "")))
    if not isinstance(value, str):
        return ""
    value = value.removeprefix("minecraft:")
    return value if _ITEM.fullmatch(value) else ""


def point(value):
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z")):
            value = [value.get("x"), value.get("y"), value.get("z")]
        else:
            value = value.get("position")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    answer = tuple(number(value[i], math.inf) for i in range(3))
    return answer if all(math.isfinite(x) for x in answer) else None


def distance(a, b):
    pa, pb = point(a), point(b)
    if pa is None or pb is None:
        return math.inf
    return math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))


def candidate_id(value, prefix=None):
    value = value if isinstance(value, str) else ""
    if not _ID.fullmatch(value):
        return ""
    if prefix and not value.startswith(prefix + ":"):
        return ""
    return value


def id_point(value):
    value = candidate_id(value)
    if not value:
        return None
    try:
        return tuple(float(x) for x in value[2:].split(","))
    except (TypeError, ValueError):
        return None


def parse_observation(value):
    if isinstance(value, dict):
        # A few test doubles wrap the JSON observation one level deep.
        if "self" not in value and isinstance(value.get("obs"), (dict, str)):
            value = value.get("obs")
        else:
            return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def self_row(obs):
    value = obs.get("self")
    return value if isinstance(value, dict) else {}


def inventory(obs):
    """Return only personal stacks; never count a cursor or open container."""
    sources = []
    for key in ("inventory", "inventory_authoritative"):
        value = obs.get(key)
        if value not in (None, {}, []):
            sources.append(value)
    # inventory_slots is the labelled personal slot view in this adaptor.
    if not sources and isinstance(obs.get("inventory_slots"), list):
        sources.append(obs.get("inventory_slots"))
    if not sources and isinstance(obs.get("inventory_client"), dict):
        sources.append(obs.get("inventory_client"))

    answer = {}
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
            source = source.get("items")
        if isinstance(source, dict):
            iterable = []
            for key, raw in source.items():
                iterable.append({"name": key, "count": raw.get("count", 0) if isinstance(raw, dict) else raw})
        else:
            iterable = source if isinstance(source, list) else []
        for raw in iterable:
            if not isinstance(raw, dict):
                continue
            item = clean_item(raw)
            count = max(0, int(number(raw.get("count", 0))))
            if item and count:
                answer[item] = answer.get(item, 0) + count
        if answer:
            # The authoritative view is preferred, but a partially populated
            # client view is useful for feasibility between audit snapshots.
            break
    if not answer and isinstance(obs.get("inventory_client"), dict):
        for key, raw in obs["inventory_client"].items():
            item = clean_item(key)
            count = max(0, int(number(raw)))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    return answer


def inventory_signature(inv):
    return tuple(sorted((str(k), int(v)) for k, v in inv.items() if v))


def feedback_value(obs):
    for key in ("action_feedback", "action_result", "last_result", "feedback"):
        value = obs.get(key)
        if value in (None, "", {}):
            continue
        if isinstance(value, dict):
            value = " ".join(str(value.get(k, "")) for k in ("status", "result", "message", "error", "reason"))
        return str(value).lower()
    return ""


def is_failure(text):
    text = str(text or "").lower()
    return any(word in text for word in (
        "rejected", "reject", "failed", "failure", "error", "unreachable",
        "timed out", "timeout", "cannot", "can't", "invalid", "not in reach",
        "no path", "not enough", "out of range",
    ))


def is_task(obs):
    task = obs.get("task")
    return task not in (None, "", {}, [])


def visible_name(row):
    return clean_item(row.get("name", row.get("block", row.get("type", "")))) if isinstance(row, dict) else ""


def food_name(inv):
    order = (
        "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
        "bread", "baked_potato", "carrot", "apple", "sweet_berries", "beef",
        "porkchop", "mutton", "chicken", "rabbit", "cod", "salmon", "potato",
        "wheat",
    )
    return next((x for x in order if inv.get(x, 0)), "")


def food_total(inv):
    return sum(int(v) for k, v in inv.items() if k in FOODS)


def wood_total(inv):
    return sum(int(v) for k, v in inv.items() if k.endswith(WOOD_SUFFIXES))


def plank_total(inv):
    return sum(int(v) for k, v in inv.items() if k.endswith("_planks"))


class HandTransaction:
    """A placement or use operation with an explicit equip stage."""

    def __init__(self):
        self.intent = None

    def begin(self, kind, item, reference, face="up"):
        if not item or not reference:
            return False
        self.intent = {"kind": kind, "item": item, "reference": reference, "face": face}
        return True

    def clear(self):
        self.intent = None


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.names = set()
        self.last_posted = "noop"
        self.active = None
        self.cancel_pending = False
        self.hand = HandTransaction()
        self.failed = {}
        self.failed_blocks = set()
        self.failed_supports = set()
        self.nav_bad = {}
        self.nav_visits = {}
        self.frontier_cells = set()
        self.explore_attempts = 0
        self.last_feedback = None
        self.last_inventory = None
        self.last_score = None
        self.last_progress = 0.0
        self.table_placed = False
        self.table_position = None
        self.table_expected_until = -1.0
        self.furnace_position = None
        self.furnace_expected_until = -1.0
        self.smelting = {}
        self.manual_until = -1.0
        self.manual_pattern = 0
        self.manual_segments = 0
        self.replan_until = -1.0
        self.dead_seen = False
        self.phase = "opening"
        self.tactical_future = None
        self.tactical_submitted = -100.0
        self.tactical_signature = None
        self.tactical_options = []
        self.tactical_cached = None
        self.tactical_cached_at = -100.0

    def now(self):
        try:
            return float(self.ctx.clock.now())
        except Exception:
            return 0.0

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def permitted(self, action):
        verb = str(action or "").split(" ", 1)[0]
        if not self.names:
            return verb in {
                "noop", "cancel", "navigate", "goto", "gather", "equip", "craft", "place",
                "furnace_start", "furnace_take", "eat", "attack", "shield", "wait", "controls",
                "respawn", "sleep", "wake",
            }
        return verb in self.names

    def position(self, obs):
        return point(self_row(obs).get("position")) or (0.0, 64.0, 0.0)

    def blocks(self, obs):
        answer = []
        here = self.position(obs)
        for row in rows(obs.get("candidates")):
            ident = candidate_id(row.get("id"), "b")
            if not ident:
                continue
            item = dict(row)
            item["_id"] = ident
            item["_name"] = visible_name(row)
            item["_pos"] = point(row.get("position")) or id_point(ident)
            item["_distance"] = number(row.get("distance"), distance(item["_pos"], here))
            answer.append(item)
        answer.sort(key=lambda x: x["_distance"])
        return answer

    def stands(self, obs):
        answer = []
        here = self.position(obs)
        for row in rows(obs.get("candidates")):
            ident = candidate_id(row.get("id"), "s")
            if not ident:
                continue
            item = dict(row)
            item["_id"] = ident
            item["_pos"] = point(row.get("position")) or id_point(ident)
            item["_distance"] = number(row.get("distance"), distance(item["_pos"], here))
            answer.append(item)
        answer.sort(key=lambda x: x["_distance"])
        return answer

    def entities(self, obs):
        answer = []
        here = self.position(obs)
        for row in rows(obs.get("entities", obs.get("nearby_entities"))):
            item = dict(row)
            item["_name"] = clean_item(row.get("name", row.get("type", "")))
            item["_pos"] = point(row.get("position"))
            item["_distance"] = number(row.get("distance"), distance(item["_pos"], here))
            answer.append(item)
        return answer

    def collision(self, obs):
        result = {}
        world = obs.get("world") if isinstance(obs.get("world"), dict) else {}
        raw = world.get("local_collision")
        for row in raw if isinstance(raw, list) else []:
            if isinstance(row, (list, tuple)) and len(row) >= 4:
                try:
                    result[tuple(int(number(row[i])) for i in range(3))] = str(row[3])
                except (TypeError, ValueError):
                    pass
        return result

    def record_progress(self, obs, now):
        inv = inventory(obs)
        score = number(obs.get("score"), 0.0)
        if isinstance(obs.get("progress"), dict):
            score = number(obs["progress"].get("score"), score)
        advancements = tuple(sorted(str(x) for x in (obs.get("advancements") or [])))
        marker = (inventory_signature(inv), round(score, 3), advancements)
        old = (self.last_inventory, self.last_score, getattr(self, "last_advancements", ()))
        if self.last_inventory is None or marker != old:
            self.last_progress = now
        self.last_inventory = marker[0]
        self.last_score = marker[1]
        self.last_advancements = marker[2]
        # An observed table is an objective fact, not merely an inventory guess.
        for block in self.blocks(obs):
            if block["_name"] == "crafting_table":
                self.table_position = block["_pos"] or self.table_position
                self.table_placed = True
            if block["_name"] == "furnace":
                self.furnace_position = block["_pos"] or self.furnace_position

    def feedback_marker(self, obs):
        value = feedback_value(obs)
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)

    def operation_key(self, action):
        fields = str(action or "").split()
        if len(fields) >= 2 and fields[0] in {"gather", "navigate", "goto", "place", "sleep", "open_container"}:
            if fields[0] == "place" and len(fields) >= 3:
                return "place " + fields[1] + " " + fields[2]
            return fields[0] + " " + fields[1]
        return str(action or "")

    def action_limit(self, action):
        verb = str(action or "").split(" ", 1)[0]
        return {
            "gather": 25.0, "navigate": 15.0, "goto": 15.0,
            "craft": 8.0, "place": 8.0, "equip": 5.0, "furnace_start": 8.0,
            "furnace_take": 8.0, "eat": 6.0, "sleep": 8.0, "activate_block": 8.0,
        }.get(verb, 6.0)

    def current_action_target(self, action):
        fields = str(action or "").split()
        if not fields:
            return None
        # place ITEM REFERENCE FACE names the reference in field 2; the other
        # coordinate-bearing commands name it in field 1.
        index = 2 if fields[0] == "place" and len(fields) >= 3 else 1
        if len(fields) > index:
            return id_point(fields[index])
        return None

    def mark_failed(self, action, now, reason):
        action = str(action or "")
        if not action or action in {"noop", "cancel"}:
            return
        fields = action.split()
        verb = fields[0] if fields else ""
        key = self.operation_key(action)
        cooldown = {"gather": 180.0, "navigate": 90.0, "goto": 45.0, "place": 999999.0}.get(verb, 12.0)
        self.failed[key] = max(self.failed.get(key, -1.0), now + cooldown)
        if verb == "gather" and len(fields) > 1:
            self.failed_blocks.add(fields[1])
        if verb == "navigate" and len(fields) > 1:
            self.nav_bad[fields[1]] = now + 120.0
        if verb == "place" and len(fields) >= 3:
            self.failed_supports.add((fields[1], fields[2]))
            self.hand.clear()
        if self.active and self.active.get("action") == action:
            self.active = None
        self.log("operation_failed", action=action, reason=str(reason)[:160], key=key)

    def blocked(self, action, now):
        key = self.operation_key(action)
        return now < self.failed.get(action, -1.0) or now < self.failed.get(key, -1.0)

    def block_change_for(self, obs, target):
        if target is None:
            return False
        # block_changes are recent, bounded events.  Coordinate equality is
        # enough; the event name is intentionally not trusted as a success flag.
        for change in rows(obs.get("block_changes")):
            if point(change.get("position")) == tuple(target):
                return True
        return False

    def post(self, action, obs, reason=""):
        if self.ctx.done:
            return "noop"
        action = str(action or "noop").strip() or "noop"
        if not self.permitted(action):
            action = "noop"
        now = self.now()
        if action not in {"noop", "cancel"} and self.blocked(action, now):
            self.log("blocked_action", action=action)
            if action.startswith("place "):
                self.hand.clear()
            action = "noop"
        verb = action.split(" ", 1)[0]
        repeatable = verb in {"noop", "controls", "attack", "shield"}
        # One-shot commands are latched by the adaptor.  A noop is an explicit
        # re-arm; it does not replace a bounded job because noop continues it.
        if action == self.last_posted and not repeatable:
            action = "noop"
            verb = "noop"
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("post_error", action=action, error=type(exc).__name__)
            action = "noop"
            verb = "noop"
        self.last_posted = action
        self.last_feedback = self.feedback_marker(obs)
        if action == "cancel":
            self.active = None
            self.cancel_pending = False
            return action
        if action == "noop":
            return action
        if verb == "controls":
            return action
        if verb == "shield":
            return action
        self.active = {
            "action": action,
            "at": now,
            "inventory": inventory_signature(inventory(obs)),
            "inventory_dict": dict(inventory(obs)),
            "position": self.position(obs),
            "held": clean_item(self_row(obs).get("held_item")),
            "dimension": self_row(obs).get("dimension"),
            "target": self.current_action_target(action),
            "food": number(self_row(obs).get("food"), 20),
            "advancements": tuple(str(x) for x in (obs.get("advancements") or [])),
            "limit": self.action_limit(action),
        }
        if verb == "navigate" and len(action.split()) >= 2:
            ident = action.split()[1]
            self.nav_visits[ident] = self.nav_visits.get(ident, 0) + 1
            self.explore_attempts += 1
            p = id_point(ident)
            if p:
                self.frontier_cells.add((int(math.floor(p[0] / 6.0)), int(math.floor(p[2] / 6.0))))
        self.log("posted", action=action, reason=reason, phase=self.phase)
        return action

    def complete_place(self, action, obs, now):
        fields = action.split()
        if len(fields) < 3:
            return
        item = fields[1]
        ref = fields[2]
        ref_pos = id_point(ref)
        placed_pos = (ref_pos[0], ref_pos[1] + 1.0, ref_pos[2]) if ref_pos else None
        if item == "crafting_table":
            self.table_placed = True
            self.table_position = placed_pos or self.table_position
            self.table_expected_until = now + 8.0
        elif item == "furnace":
            self.furnace_position = placed_pos or self.furnace_position
            self.furnace_expected_until = now + 8.0
        self.hand.clear()
        self.log("placement_confirmed", item=item, reference=ref, position=placed_pos)

    def active_changed(self, active, obs):
        action = active["action"]
        verb = action.split(" ", 1)[0]
        now_inv = inventory(obs)
        old_inv = active.get("inventory_dict", {})
        inv_changed = inventory_signature(now_inv) != active.get("inventory", ())
        me = self_row(obs)
        held = clean_item(me.get("held_item"))
        old_pos = active.get("position")
        pos = self.position(obs)
        moved = distance(old_pos, pos) >= 0.65
        target = active.get("target")
        closer = target is not None and distance(pos, target) + 0.35 < distance(old_pos, target)
        if verb in {"navigate", "goto"}:
            return moved or closer
        if verb == "equip":
            desired = clean_item(action.split()[1] if len(action.split()) > 1 else "")
            return held == desired
        if verb == "eat":
            return inv_changed or number(me.get("food"), 20) > number(active.get("food", me.get("food", 20)), 20) + 0.2
        if verb == "place":
            if inv_changed or self.block_change_for(obs, target):
                return True
            fields = action.split()
            item = fields[1] if len(fields) > 1 else ""
            if target is not None:
                placed = (target[0], target[1] + 1.0, target[2])
                return any(x["_name"] == item and x["_pos"] == placed for x in self.blocks(obs))
            return False
        if verb == "sleep":
            return bool(me.get("bed_position")) or inv_changed or tuple(str(x) for x in (obs.get("advancements") or [])) != active.get("advancements", ())
        if verb in {"gather", "dig"}:
            return inv_changed or self.block_change_for(obs, target)
        if verb == "attack":
            return False
        return inv_changed or self.block_change_for(obs, target)

    def update_active(self, obs, now):
        active = self.active
        if active is None:
            return False
        text = feedback_value(obs)
        marker = self.feedback_marker(obs)
        if marker != self.last_feedback:
            self.last_feedback = marker
            if is_failure(text):
                self.mark_failed(active["action"], now, "adapter_feedback")
                self.cancel_pending = bool(is_task(obs))
                return False
        if self.active is None:
            return False
        active = self.active
        if self.active_changed(active, obs):
            action = active["action"]
            if action.startswith("place "):
                self.complete_place(action, obs, now)
            if action.startswith("furnace_start "):
                fields = action.split()
                if len(fields) > 1:
                    self.smelting[fields[1]] = now
            self.log("operation_confirmed", action=action)
            self.active = None
            return False
        elapsed = now - number(active.get("at"), now)
        task = is_task(obs)
        if elapsed < 0.8:
            return True
        if elapsed >= number(active.get("limit"), 8.0) and not task:
            self.mark_failed(active["action"], now, "no_postcondition")
            return False
        if elapsed >= number(active.get("limit"), 8.0) + 3.0:
            self.mark_failed(active["action"], now, "job_deadline")
            self.cancel_pending = task
            return False
        return task or elapsed < number(active.get("limit"), 8.0)

    def support_candidates(self, obs, item):
        here = self.position(obs)
        floor = (math.floor(here[0]), math.floor(here[1]), math.floor(here[2]))
        collision = self.collision(obs)
        answer = []
        for block in self.blocks(obs):
            ref = block["_id"]
            pos = block["_pos"]
            if not pos or block["_name"] not in SUPPORTS:
                continue
            if not block.get("visible") and not block.get("in_reach"):
                continue
            if not block.get("in_reach"):
                continue
            if int(math.floor(pos[1])) != floor[1] - 1:
                continue
            if int(math.floor(pos[0])) == floor[0] and int(math.floor(pos[2])) == floor[2]:
                continue
            if (item, ref) in self.failed_supports:
                continue
            rel = (int(math.floor(pos[0])) - floor[0], int(math.floor(pos[1])) + 1 - floor[1], int(math.floor(pos[2])) - floor[2])
            # If the adaptor supplied a collision label for the placement cell,
            # reject an occupied cell.  Unknown cells remain admissible because
            # the bounded candidate stream may omit a vertical neighbour.
            if collision.get(rel) == "solid":
                continue
            answer.append(block)
        answer.sort(key=lambda x: (number(x.get("distance"), 99), not bool(x.get("visible"))))
        return answer

    def start_placement(self, obs, item, kind):
        inv = inventory(obs)
        if inv.get(item, 0) <= 0 or self.hand.intent is not None:
            return None
        supports = self.support_candidates(obs, item)
        if not supports:
            return None
        support = supports[0]
        self.hand.begin(kind, item, support["_id"], "up")
        return self.hand_action(obs)

    def hand_action(self, obs):
        intent = self.hand.intent
        if not intent:
            return None
        item = intent["item"]
        held = clean_item(self_row(obs).get("held_item"))
        if held != item:
            action = "equip %s hand" % item
            return None if self.blocked(action, self.now()) else action
        return "place %s %s %s" % (item, intent["reference"], intent["face"])

    def current_table(self, obs):
        choices = [x for x in self.blocks(obs) if x["_name"] == "crafting_table"]
        if not choices:
            return None
        choices.sort(key=lambda x: x["_distance"])
        self.table_placed = True
        self.table_position = choices[0]["_pos"] or self.table_position
        return choices[0]

    def usable_table(self, obs):
        table = self.current_table(obs)
        if table and table.get("in_reach") and table["_distance"] <= 4.8:
            return table
        return None

    def current_furnace(self, obs):
        choices = [x for x in self.blocks(obs) if x["_name"] == "furnace"]
        if not choices:
            return None
        choices.sort(key=lambda x: x["_distance"])
        self.furnace_position = choices[0]["_pos"] or self.furnace_position
        return choices[0]

    def select_resource(self, obs, names, tool=None, count=1):
        now = self.now()
        blocks = [x for x in self.blocks(obs)
                  if x["_name"] in names or any(x["_name"].endswith(s) for s in names if s.startswith("_"))]
        blocks = [x for x in blocks if x.get("harvestable") is not False and x["_id"] not in self.failed_blocks]
        blocks.sort(key=lambda x: (not bool(x.get("visible")), not bool(x.get("in_reach")), x["_distance"]))
        if not blocks:
            return None
        target = blocks[0]
        if tool and clean_item(self_row(obs).get("held_item")) != tool:
            if inventory(obs).get(tool, 0):
                action = "equip %s hand" % tool
                if not self.blocked(action, now):
                    return action
        if target.get("in_reach") or target.get("visible"):
            return "gather %s %d" % (target["_id"], max(1, min(4, int(count))))
        # A block candidate is itself a legal navigation goal.  This is useful
        # for a visible tree trunk hidden behind foliage, where a stand candidate
        # may leave the bot one interaction cell short.
        nav = "navigate " + target["_id"]
        if (target["_distance"] <= 40.0 and self.nav_visits.get(target["_id"], 0) == 0
                and not self.blocked(nav, now) and now >= self.nav_bad.get(target["_id"], -1.0)):
            return nav
        # One direct approach is enough for this bounded target.  If the
        # candidate remains occluded, let the general frontier choose a new
        # interaction position rather than retrying the same block forever.
        return self.choose_navigation(obs, "approach resource")

    def animal_action(self, obs, inv):
        entities = [x for x in self.entities(obs) if x["_name"] in ANIMALS and x.get("visible") and not x.get("dead")]
        if not entities:
            return None
        entities.sort(key=lambda x: (x["_name"] != "sheep", x["_distance"]))
        animal = entities[0]
        weapon = next((x for x in WEAPONS if inv.get(x, 0)), "")
        if not weapon:
            return None
        held = clean_item(self_row(obs).get("held_item"))
        if held != weapon:
            action = "equip %s hand" % weapon
            return None if self.blocked(action, self.now()) else action
        if animal["_distance"] <= 3.2 and animal.get("id") is not None:
            return "attack " + str(animal["id"])
        if animal.get("_pos") and animal["_distance"] <= 40:
            p = animal["_pos"]
            return "goto %.2f %.2f %.2f" % p
        return None

    def drop_action(self, obs):
        drops = [x for x in self.entities(obs) if x["_name"] in {"item", "item_stack"} and x.get("_pos") and x["_distance"] < 8]
        if not drops:
            return None
        drops.sort(key=lambda x: x["_distance"])
        p = drops[0]["_pos"]
        return "goto %.2f %.2f %.2f" % p

    def choose_navigation(self, obs, purpose="search"):
        now = self.now()
        here = self.position(obs)
        candidates = []
        for stand in self.stands(obs):
            ident = stand["_id"]
            d = stand["_distance"]
            if d < 2.0 or d > 48.0 or now < self.nav_bad.get(ident, -1.0):
                continue
            visits = self.nav_visits.get(ident, 0)
            pos = stand.get("_pos")
            cell = (int(math.floor(pos[0] / 6.0)), int(math.floor(pos[2] / 6.0))) if pos else None
            fresh = cell not in self.frontier_cells
            # Novel cells dominate, then low visit count, then a moderate
            # distance.  The list is a frontier, not a nearest-stand orbit.
            rank = (0 if fresh else 1, visits, -min(d, 36.0), distance(here, pos))
            if visits < 2:
                candidates.append((rank, ident))
        if not candidates:
            for stand in self.stands(obs):
                ident = stand["_id"]
                if stand["_distance"] >= 2.0 and stand["_distance"] <= 48.0 and now >= self.nav_bad.get(ident, -1.0):
                    candidates.append(((2, self.nav_visits.get(ident, 0), -stand["_distance"], 0), ident))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            self.log("frontier_choice", candidate=candidates[0][1], purpose=purpose)
            return "navigate " + candidates[0][1]
        return None

    def maybe_manual_escape(self, obs, now):
        if not self.permitted("controls"):
            return None
        if self.manual_until > now:
            patterns = ("controls forward,jump", "controls left,forward,jump", "controls right,forward,jump")
            return patterns[self.manual_pattern % len(patterns)]
        # Manual motion is only a recovery from a local navigation basin.  It
        # is not the ordinary policy and is disabled once the workstation is
        # usable.
        if self.table_placed or self.explore_attempts < 7 or now - self.last_progress < 28.0:
            return None
        self.manual_until = now + 5.0
        self.manual_pattern += 1
        self.manual_segments += 1
        self.log("manual_frontier_escape", segment=self.manual_segments)
        return self.maybe_manual_escape(obs, now)

    def normal_action(self, obs):
        now = self.now()
        me = self_row(obs)
        inv = inventory(obs)
        blocks = self.blocks(obs)
        table = self.usable_table(obs)
        planks = plank_total(inv)
        logs = [k for k, v in inv.items() if v and k.endswith(WOOD_SUFFIXES)]
        sticks = inv.get("stick", 0)
        cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
        has_wood_pick = any(inv.get(x, 0) for x in PICKS)
        has_stone_pick = any(inv.get(x, 0) for x in ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"))
        food = food_total(inv)
        raw_iron = inv.get("raw_iron", 0)
        iron_ingot = inv.get("iron_ingot", 0)
        self.phase = "opening"

        if self.replan_until > now and self.hand.intent is None:
            recovery = self.choose_navigation(obs, "recover from failed operation")
            if recovery:
                self.phase = "replan_frontier"
                return recovery
            manual = self.maybe_manual_escape(obs, now)
            if manual:
                self.phase = "replan_manual"
                return manual

        # Drops are real entities and must be collected before wandering away.
        drop = self.drop_action(obs)
        if drop and not table:
            self.phase = "collect_drops"
            return drop

        # A low hunger band always wins over optional gathering.  Food use is
        # itself hand-staged because eat requires the selected carried item.
        if number(me.get("food"), 20) <= 10:
            edible = food_name(inv)
            if edible:
                self.phase = "eat"
                held = clean_item(me.get("held_item"))
                return ("equip %s hand" % edible) if held != edible else "eat " + edible

        # Opening workstation transaction.  The crucial difference from the
        # previous candidates is the equip stage before place.
        if table is None:
            nearby_table = self.current_table(obs)
            if nearby_table is not None:
                if nearby_table.get("in_reach") and nearby_table["_distance"] <= 4.8:
                    table = nearby_table
                else:
                    self.phase = "approach_table"
                    action = "navigate " + nearby_table["_id"]
                    return action if not self.blocked(action, now) else self.choose_navigation(obs, "approach another workstation") or "noop"
            if table is None and inv.get("crafting_table", 0):
                action = self.hand_action(obs)
                if self.hand.intent is None:
                    action = self.start_placement(obs, "crafting_table", "table")
                if action:
                    self.phase = "place_table"
                    return action
                if self.table_placed and self.table_position:
                    p = self.table_position
                    self.phase = "find_placed_table"
                    return "goto %.2f %.2f %.2f" % (p[0] + 1.0, p[1], p[2] + 1.0)
            if table is None and self.table_placed and not inv.get("crafting_table"):
                # The item was consumed, so do not craft another table merely
                # because the candidate snapshot has not caught up yet.
                if now < self.table_expected_until:
                    self.phase = "await_table_update"
                    return "noop"
                nav = self.choose_navigation(obs, "locate placed workstation")
                if nav:
                    self.phase = "locate_table"
                    return nav
            if table is None:
                if planks < 4 and logs:
                    self.phase = "make_table_planks"
                    output = min(4, inv.get(logs[0], 0) * 4)
                    return "craft %s %d" % (logs[0].replace("_log", "_planks").replace("_stem", "_planks"), max(4, output))
                if planks >= 4 and not inv.get("crafting_table"):
                    self.phase = "craft_table"
                    return "craft crafting_table 1"
                if logs:
                    self.phase = "make_more_planks"
                    output = min(8, inv.get(logs[0], 0) * 4)
                    return "craft %s %d" % (logs[0].replace("_log", "_planks").replace("_stem", "_planks"), max(4, output))
                resource = self.select_resource(obs, set(WOOD_SUFFIXES), count=1)
                if resource:
                    self.phase = "find_wood"
                    return resource
                self.phase = "frontier_wood"
                return self.choose_navigation(obs, "find exposed wood") or self.maybe_manual_escape(obs, now) or "noop"

        if table is None:
            self.phase = "table_missing"
            return self.choose_navigation(obs, "return to workstation") or "noop"

        # Wooden and stone tools are a strict dependency chain.  Tool gathers
        # also stage the held tool rather than relying on an implicit auto-equip.
        if not has_wood_pick:
            if planks < 3:
                if logs:
                    self.phase = "wooden_pick_planks"
                    output = min(8, inv.get(logs[0], 0) * 4)
                    return "craft %s %d" % (logs[0].replace("_log", "_planks").replace("_stem", "_planks"), max(4, output))
                action = self.select_resource(obs, set(WOOD_SUFFIXES), count=1)
                self.phase = "wooden_pick_wood"
                return action or self.choose_navigation(obs, "find wood for pick") or self.maybe_manual_escape(obs, now) or "noop"
            if sticks < 2:
                self.phase = "wooden_pick_sticks"
                return "craft stick 4"
            self.phase = "wooden_pick"
            return "craft wooden_pickaxe 1 %s" % table["_id"]

        if not has_stone_pick:
            if cobble < 3:
                action = self.select_resource(obs, STONE, tool="wooden_pickaxe", count=3)
                self.phase = "stone_gather"
                return action or self.choose_navigation(obs, "find exposed stone") or self.maybe_manual_escape(obs, now) or "noop"
            if sticks < 2:
                self.phase = "stone_sticks"
                return "craft stick 4"
            self.phase = "stone_pick"
            return "craft stone_pickaxe 1 %s" % table["_id"]

        # A weapon is cheap insurance and enables the animal path for food/wool.
        weapon = next((x for x in WEAPONS if inv.get(x, 0)), "")
        if not weapon and cobble >= 2 and sticks >= 1:
            self.phase = "weapon"
            return "craft stone_sword 1 %s" % table["_id"]

        # Try visible/exposed food and sheep before a long mining expedition.
        if food < 3:
            action = self.select_resource(obs, FOOD_BLOCKS, count=1)
            if action:
                self.phase = "food_block"
                return action
            action = self.animal_action(obs, inv)
            if action:
                self.phase = "food_animal"
                return action

        # Acquire wool/bed when the world offers a sheep.  Possessing a bed is
        # sufficient for the opening flag; sleeping is attempted later only
        # when a real bed block is observed and vanilla permits it.
        has_bed = any(k.endswith("_bed") and v for k, v in inv.items())
        wool = next((k for k, v in inv.items() if k.endswith("_wool") and v >= 3), "")
        if not has_bed and wool and planks >= 3:
            self.phase = "bed_craft"
            return "craft %s_bed 1 %s" % (wool[:-5], table["_id"])
        if not has_bed:
            sheep = [x for x in self.entities(obs) if x["_name"] == "sheep" and x.get("visible")]
            if sheep:
                action = self.animal_action(obs, inv)
                if action:
                    self.phase = "bed_sheep"
                    return action

        # Iron first uses only exposed candidates.  If there is no local ore,
        # the frontier branch below reveals new loaded terrain rather than
        # inventing underground coordinates.
        if iron_ingot <= 0 and raw_iron < 3:
            action = self.select_resource(obs, IRON, tool="stone_pickaxe", count=3)
            if action:
                self.phase = "iron_ore"
                return action
        if not inv.get("coal") and not inv.get("charcoal"):
            action = self.select_resource(obs, COAL, tool="stone_pickaxe", count=2)
            if action:
                self.phase = "coal"
                return action

        furnace = self.current_furnace(obs)
        if raw_iron and furnace is None and not inv.get("furnace") and cobble >= 8:
            self.phase = "furnace_craft"
            return "craft furnace 1 %s" % table["_id"]
        if raw_iron and furnace is None and inv.get("furnace"):
            action = self.hand_action(obs)
            if self.hand.intent is None:
                action = self.start_placement(obs, "furnace", "furnace")
            if action:
                self.phase = "furnace_place"
                return action
            if now < self.furnace_expected_until:
                self.phase = "furnace_wait_place"
                return "noop"
        if raw_iron and furnace is not None:
            if furnace["_distance"] > 4.8:
                self.phase = "approach_furnace"
                action = "navigate " + furnace["_id"]
                return action if not self.blocked(action, now) else self.choose_navigation(obs, "approach another furnace") or "noop"
            fid = furnace["_id"]
            started = self.smelting.get(fid)
            fuel = next((x for x in ("coal", "charcoal") if inv.get(x)), "")
            if not fuel:
                fuel = next((x for x in inv if x.endswith("_planks") and inv.get(x)), "")
            if started is None and fuel:
                self.phase = "smelt_start"
                return "furnace_start %s raw_iron %d %s %d" % (fid, min(3, raw_iron), fuel, min(3, inv.get(fuel, 1)))
            if started is not None and now - started >= 31.0:
                self.phase = "smelt_take"
                return "furnace_take " + fid
            self.phase = "smelt_wait"
            return "noop"

        if iron_ingot >= 1 and not inv.get("shield"):
            if planks < 6 and logs:
                self.phase = "shield_planks"
                output = min(8, inv.get(logs[0], 0) * 4)
                return "craft %s %d" % (logs[0].replace("_log", "_planks").replace("_stem", "_planks"), max(4, output))
            self.phase = "shield"
            return "craft shield 1 %s" % table["_id"] if planks >= 6 else "noop"

        # A nearby bed block can establish the respawn point once the world is
        # night.  Do not force the run to wait for night or a village.
        beds = [x for x in blocks if x["_name"].endswith("_bed") and (x.get("in_reach") or x.get("visible"))]
        if not has_bed and beds:
            bed = beds[0]
            if bed["_distance"] <= 4.5:
                self.phase = "sleep"
                return "sleep " + bed["_id"]
            self.phase = "approach_bed"
            action = "navigate " + bed["_id"]
            return action if not self.blocked(action, now) else self.choose_navigation(obs, "approach another bed") or "noop"

        drop = self.drop_action(obs)
        if drop:
            self.phase = "collect_drops"
            return drop

        # No exposed immediate milestone: reveal another local slice.  After a
        # repeated navigation basin, briefly use renewed manual movement to
        # escape terrain where all listed stands are poor interaction positions.
        self.phase = "frontier"
        nav = self.choose_navigation(obs, "reveal resources")
        if nav:
            return nav
        manual = self.maybe_manual_escape(obs, now)
        if manual:
            return manual
        return "noop"

    def hazard_rows(self, obs):
        here = self.position(obs)
        found = []
        me = self_row(obs)
        if me.get("is_in_lava") or me.get("is_in_fire") or number(me.get("fall_distance_estimate"), 0) > 2:
            found.append({"kind": "self_hazard", "id": "self", "distance": 0.0})
        if me.get("is_in_water") and number(me.get("oxygen"), 300) < 80:
            found.append({"kind": "drowning", "id": "self", "distance": 0.0})
        for block in self.blocks(obs):
            if block["_name"] in HAZARDS and block["_distance"] <= 5.0:
                found.append({"kind": block["_name"], "id": block["_id"], "distance": block["_distance"]})
        for ent in self.entities(obs):
            if (ent.get("hostile") or ent.get("conditional_threat") or ent["_name"] in HOSTILES) and ent.get("visible") and ent["_distance"] <= 8.5:
                found.append({"kind": ent["_name"], "id": str(ent.get("id", "")), "distance": ent["_distance"], "entity": ent})
        return found

    def retreat_action(self, obs, threats, inv):
        close = [x for x in threats if x.get("entity") and number(x.get("distance"), 99) <= 3.2]
        if inv.get("shield"):
            return "shield on"
        weapon = next((x for x in WEAPONS if inv.get(x, 0)), "")
        if close and weapon:
            held = clean_item(self_row(obs).get("held_item"))
            if held != weapon:
                return "equip %s hand" % weapon
            entity = close[0].get("entity") or {}
            if entity.get("id") is not None:
                return "attack " + str(entity["id"])
        stands = []
        here = self.position(obs)
        for stand in self.stands(obs):
            if 4.0 <= stand["_distance"] <= 22.0 and stand.get("_pos"):
                nearest = min((distance(stand["_pos"], x.get("entity", {}).get("position")) for x in threats if x.get("entity")), default=0)
                stands.append((nearest, stand))
        if stands:
            stands.sort(key=lambda x: x[0], reverse=True)
            return "navigate " + stands[0][1]["_id"]
        return "controls back,jump" if self.permitted("controls") else "noop"

    def tactical_menu(self, obs, threats, inv):
        options = []
        me = self_row(obs)
        if inv.get("shield"):
            options.append("shield on")
        weapon = next((x for x in WEAPONS if inv.get(x, 0)), "")
        for threat in threats:
            ent = threat.get("entity") or {}
            if ent.get("id") is not None and number(threat.get("distance"), 99) <= 3.2 and weapon:
                if clean_item(me.get("held_item")) != weapon:
                    options.append("equip %s hand" % weapon)
                options.append("attack " + str(ent["id"]))
        retreat = self.retreat_action(obs, threats, inv)
        if retreat:
            options.append(retreat)
        for stand in self.stands(obs):
            if 4.0 < stand["_distance"] < 20.0:
                options.append("navigate " + stand["_id"])
                break
        options.append("noop")
        answer = []
        for action in options:
            if action not in answer and self.permitted(action):
                answer.append(action)
        return answer[:7]

    def poll_tactical(self, obs, options, signature, now):
        if self.tactical_future is not None and self.tactical_future.done():
            future = self.tactical_future
            old_sig = self.tactical_signature
            self.tactical_future = None
            try:
                result = future.result()
                answer = result.get("response") if isinstance(result, dict) else None
                value = getattr(answer, "value", None)
                probs = getattr(answer, "probabilities", {}) or {}
                probability = number(probs.get(value), 0.0) if isinstance(probs, dict) else 0.0
                if old_sig == signature and value in options and probability >= 0.55:
                    self.tactical_cached = value
                    self.tactical_cached_at = now
                    self.log("jev_tactical", action=value, probability=probability)
                else:
                    self.tactical_cached = None
            except (BudgetExceeded, ModelUnavailable):
                # Budget closure is a normal episode end; an unavailable model
                # must retain the contract's stop semantics.
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:
                self.log("jev_tactical_invalid", error=type(exc).__name__)
                self.tactical_cached = None
        if self.tactical_cached in options and now - self.tactical_cached_at <= 2.5:
            return self.tactical_cached
        if self.tactical_future is None and self.ctx.budget.remaining().get("system1_calls", 0) > 2:
            state = {
                "purpose": "Immediate survival response; code owns legality and movement.",
                "self": {
                    "health_band": "critical" if number(self_row(obs).get("health"), 20) < 7 else "low" if number(self_row(obs).get("health"), 20) < 12 else "stable",
                    "food_band": "low" if number(self_row(obs).get("food"), 20) < 10 else "fed",
                    "dimension": str(self_row(obs).get("dimension", "")),
                    "lava_or_fire": bool(self_row(obs).get("is_in_lava") or self_row(obs).get("is_in_fire")),
                },
                "threats": [{"kind": str(x.get("kind", "")), "range": "touching" if number(x.get("distance"), 99) <= 3.2 else "near"} for x in threats[:6]],
                "legal_actions": list(options),
                "rule": "Choose only one supplied complete action. Survival takes priority. Never invent an id or attack a hazard.",
            }
            questions = {"response": Choice(
                "Select the safest supplied immediate response. Use noop only when none of the supplied defensive actions is supported.",
                {x: "Legal response: " + x for x in options},
            )}
            try:
                self.tactical_future = self.ctx.spawn(
                    lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=2.0)
                )
                self.tactical_submitted = now
                self.tactical_signature = signature
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self.log("jev_tactical_start_failed", error=type(exc).__name__)
                self.tactical_future = None
        # Conservative code fallback while Jev is in flight.
        safe = self.retreat_action(obs, threats, inventory(obs))
        return safe if safe in options else (options[0] if options else "noop")

    def emergency(self, obs):
        now = self.now()
        me = self_row(obs)
        inv = inventory(obs)
        if me.get("dead"):
            self.manual_until = -1.0
            return "respawn"
        if number(obs.get("observation_age_s"), 0) > 3.0:
            if is_task(obs):
                return "cancel"
            if self.manual_until > now:
                return "controls none"
            return "noop"
        if me.get("is_in_lava") or me.get("is_in_fire"):
            self.manual_until = -1.0
            return "cancel" if is_task(obs) else ("controls forward,jump" if self.permitted("controls") else "noop")
        if me.get("is_in_water") and number(me.get("oxygen"), 300) < 80:
            self.manual_until = -1.0
            return "cancel" if is_task(obs) else ("controls jump" if self.permitted("controls") else "noop")
        if number(me.get("health"), 20) <= 6 and food_name(inv):
            if is_task(obs):
                return "cancel"
            edible = food_name(inv)
            held = clean_item(me.get("held_item"))
            return "equip %s hand" % edible if held != edible else "eat " + edible
        threats = self.hazard_rows(obs)
        if not threats:
            return None
        if is_task(obs):
            return "cancel"
        options = self.tactical_menu(obs, threats, inv)
        sig = tuple(sorted((str(x.get("kind")), str(x.get("id")), "touching" if number(x.get("distance"), 99) <= 3.2 else "near") for x in threats))
        return self.poll_tactical(obs, options, sig, now)

    def run_episode(self, game):
        self.names = set(getattr(getattr(game, "action_spec", None), "names", ()) or ())
        while not self.ctx.done:
            now = self.now()
            obs = parse_observation(self.ctx.observe())
            if not obs:
                self.post("noop", {}, "missing_observation")
                self.ctx.wait_for_tick()
                continue
            self.record_progress(obs, now)
            self.process_feedback_for_active(obs, now)
            me = self_row(obs)
            if me.get("dead"):
                if not self.dead_seen:
                    self.dead_seen = True
                    self.post("respawn", obs, "ordinary_respawn")
                else:
                    self.post("noop", obs, "await_respawn")
                self.ctx.wait_for_tick()
                continue
            self.dead_seen = False

            urgent = self.emergency(obs)
            if urgent:
                if urgent.startswith("cancel"):
                    self.hand.clear() if self.active is None else None
                self.post(urgent, obs, "safety_override")
                self.ctx.wait_for_tick()
                continue

            if self.cancel_pending:
                if is_task(obs):
                    self.post("cancel", obs, "finish_failed_job_cancel")
                    self.ctx.wait_for_tick()
                    continue
                self.cancel_pending = False

            if self.active is not None:
                still = self.update_active(obs, now)
                if self.active is not None or still or is_task(obs):
                    self.post("noop", obs, "continue_transaction")
                    self.ctx.wait_for_tick()
                    continue

            # A failed job can clear the local transaction before the adaptor
            # clears its task field.  Never replace that still-busy job.
            if is_task(obs):
                self.post("noop", obs, "wait_for_task_clear")
                self.ctx.wait_for_tick()
                continue

            if self.manual_until > now:
                self.post(self.maybe_manual_escape(obs, now) or "controls none", obs, "manual_frontier")
                self.ctx.wait_for_tick()
                continue
            if self.manual_until > 0 and self.manual_until <= now:
                self.manual_until = -1.0
                self.post("controls none", obs, "end_manual_frontier")
                self.ctx.wait_for_tick()
                continue

            if self.hand.intent is not None:
                action = self.hand_action(obs)
                if action:
                    self.post(action, obs, "hand_staged_interaction")
                    self.ctx.wait_for_tick()
                    continue
                self.hand.clear()

            action = self.normal_action(obs)
            if not action:
                action = "noop"
            self.post(action, obs, "compiled_progression")
            self.ctx.wait_for_tick()

    def process_feedback_for_active(self, obs, now):
        # Keep this separate from update_active so a failure can cancel a task
        # before the normal compiler observes its stale task field.
        if self.active is None:
            self.last_feedback = self.feedback_marker(obs)
            return
        marker = self.feedback_marker(obs)
        if marker == self.last_feedback:
            return
        self.last_feedback = marker
        text = feedback_value(obs)
        if is_failure(text):
            action = self.active.get("action")
            self.mark_failed(action, now, "adapter_feedback")
            self.cancel_pending = is_task(obs)


def build(ctx):
    return Controller(ctx)
