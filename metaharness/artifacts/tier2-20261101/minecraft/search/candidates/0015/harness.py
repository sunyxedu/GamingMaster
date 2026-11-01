"""A closed-loop Minecraft opener with a precondition-complete action compiler.

The parent controller already had a useful code-owned progression loop and a
small Jev tactical menu.  This version keeps that division of labour, but
compiles one-shot work from the real inventory and hand state: recipe counts
are never larger than the available inputs, and placing an item is preceded by
an explicit equip when necessary.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


FOODS = (
    "bread", "cooked_beef", "cooked_porkchop", "cooked_mutton",
    "cooked_chicken", "baked_potato", "apple", "carrot", "sweet_berries",
    "beef", "porkchop", "mutton", "chicken", "cod", "salmon",
)
WOOD_SUFFIXES = ("_log", "_stem")
STONE_BLOCKS = ("stone", "cobblestone", "deepslate", "cobbled_deepslate")
SUPPORT_BLOCKS = (
    "grass_block", "dirt", "coarse_dirt", "podzol", "mycelium", "stone",
    "sand", "gravel", "cobblestone", "deepslate", "cobbled_deepslate",
)
PICKAXES = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")
WEAPONS = ("wooden_sword", "stone_sword", "iron_sword", "diamond_sword", "netherite_sword", "wooden_axe", "stone_axe", "iron_axe")
FAIL_WORDS = (
    "failed", "error", "unreachable", "timeout", "cannot", "rejected",
    "illegal", "not enough", "not in reach", "no path", "busy", "invalid",
)


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def item_name(value):
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("id", ""))))
    return str(value or "").removeprefix("minecraft:")


def rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def point(value):
    if isinstance(value, dict):
        if value.get("position") is not None:
            value = value.get("position")
        elif all(k in value for k in ("x", "y", "z")):
            value = [value.get("x"), value.get("y"), value.get("z")]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [number(value[0]), number(value[1]), number(value[2])]
    return None


def distance(a, b):
    try:
        return math.dist(a, b)
    except (TypeError, ValueError):
        return 999.0


def personal_inventory(observation):
    """Read only personal inventory; do not count a cursor or an open window."""
    source = observation.get("inventory")
    if source is None:
        source = observation.get("inventory_authoritative", {})
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source["items"]
    answer = {}
    if isinstance(source, dict):
        for key, value in source.items():
            count = value.get("count", 0) if isinstance(value, dict) else value
            count = int(max(0, number(count)))
            name = item_name(key)
            if name and count:
                answer[name] = answer.get(name, 0) + count
    elif isinstance(source, list):
        for row in source:
            if not isinstance(row, dict):
                continue
            name = item_name(row)
            count = int(max(0, number(row.get("count", 0))))
            if name and count:
                answer[name] = answer.get(name, 0) + count
    # The authoritative audit can be empty for one snapshot while the labelled
    # client inventory has caught up.  It is still not a container/cursor view.
    if not answer and isinstance(observation.get("inventory_client"), dict):
        for key, value in observation["inventory_client"].items():
            count = int(max(0, number(value)))
            if count:
                answer[item_name(key)] = count
    return answer


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last_posted = "noop"
        self.active = None
        self.pending = None
        self.feedback_signature = None
        self.bad_until = {}
        self.recent_stands = {}
        self.seen_cells = set()
        self.last_phase = "opening"
        self.table_mark = None
        self.table_wait_until = -1000.0
        self.furnace_started = {}
        self.last_attack = -1000.0

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def facts(self, observation):
        me = observation.get("self") if isinstance(observation.get("self"), dict) else {}
        here = point(me) or [0.0, 64.0, 0.0]
        inv = personal_inventory(observation)
        blocks = []
        for raw in rows(observation.get("candidates")):
            if not raw.get("id"):
                continue
            row = dict(raw)
            row["_name"] = item_name(raw.get("name", raw.get("block", "")))
            row["_point"] = point(raw)
            row["_distance"] = number(raw.get("distance"), distance(here, row["_point"]) if row["_point"] else 999.0)
            blocks.append(row)
        entities = []
        for raw in rows(observation.get("entities", observation.get("nearby_entities", []))):
            row = dict(raw)
            row["_name"] = item_name(raw.get("name", raw.get("type", "")))
            row["_point"] = point(raw)
            row["_distance"] = number(raw.get("distance"), distance(here, row["_point"]) if row["_point"] else 999.0)
            entities.append(row)
        return me, here, inv, blocks, entities

    def action_key(self, action):
        parts = str(action).split()
        if len(parts) >= 2 and parts[0] in ("gather", "navigate", "goto", "open_container", "sleep"):
            return parts[0] + " " + parts[1]
        return str(action)

    def is_blocked(self, action, now):
        key = self.action_key(action)
        return now < max(self.bad_until.get(key, -1.0), self.bad_until.get(str(action), -1.0))

    def invalidate(self, action, now, reason):
        if not action or action in ("noop", "cancel"):
            return
        verb = str(action).split()[0]
        duration = 150.0 if verb == "gather" else 65.0 if verb in ("navigate", "goto") else 25.0
        key = self.action_key(action)
        self.bad_until[key] = max(self.bad_until.get(key, -1.0), now + duration)
        if verb in ("navigate", "goto"):
            parts = str(action).split()
            if len(parts) > 1:
                self.recent_stands[parts[1]] = now + 120.0
        if self.active and self.active.get("action") == action:
            self.active = None
        self.log("action_invalidated", action=action, reason=str(reason)[:120], blocked_s=duration)

    def read_feedback(self, observation, now):
        values = [observation.get(name) for name in ("action_feedback", "action_result", "last_result", "feedback")]
        try:
            signature = json.dumps(values, sort_keys=True, default=str)
        except Exception:
            signature = repr(values)
        if signature == self.feedback_signature:
            return
        self.feedback_signature = signature
        if self.active and any(word in signature.lower() for word in FAIL_WORDS):
            self.invalidate(self.active.get("action"), now, "adapter_feedback")

    def timeout_for(self, action):
        verb = str(action).split()[0] if action else ""
        if verb == "gather":
            return 28.0
        if verb in ("navigate", "goto"):
            return 18.0
        if verb in ("craft", "equip", "place", "hotbar", "furnace_start", "furnace_take", "activate_block"):
            return 8.0
        if verb in ("attack", "use_item", "activate_entity"):
            return 4.0
        return 7.0

    def update_active(self, observation, now, me, here, inv):
        if not self.active:
            return
        action = self.active.get("action", "")
        before_inv = self.active.get("inventory", {})
        before_here = self.active.get("position") or here
        changed_inventory = inv != before_inv
        moved = action.split()[0] in ("navigate", "goto") and distance(here, before_here) >= 1.0
        dimension_changed = me.get("dimension") != self.active.get("dimension")
        hand_changed = item_name(me.get("held_item")) != self.active.get("held")
        if changed_inventory or moved or dimension_changed or hand_changed:
            if changed_inventory and action.startswith("place crafting_table "):
                parts = action.split()
                if len(parts) >= 3 and parts[2].startswith("b:"):
                    try:
                        xyz = [int(v) for v in parts[2][2:].split(",")]
                        if len(xyz) == 3:
                            self.table_mark = [xyz[0], xyz[1] + 1, xyz[2]]
                            self.table_wait_until = now + 7.0
                            self.log("table_place_observed", position=self.table_mark, wait_s=7.0)
                    except (TypeError, ValueError):
                        pass
            self.log("action_progress", action=action, inventory_changed=changed_inventory, moved=moved)
            self.active = None
            return
        elapsed = now - self.active.get("at", now)
        if elapsed >= self.timeout_for(action) and not observation.get("task"):
            self.invalidate(action, now, "no_observed_progress")
        elif elapsed >= self.timeout_for(action) + 8.0:
            self.invalidate(action, now, "job_deadline_without_progress")

    def post(self, action, observation, now, reason=""):
        if self.ctx.done:
            return
        action = str(action or "noop").strip() or "noop"
        verb = action.split()[0]
        # One-shot actions are latched.  A noop is the explicit re-arm barrier.
        if action == self.last_posted and verb not in ("noop", "controls", "attack", "shield"):
            action = "noop"
            verb = "noop"
        self.ctx.act(action)
        self.last_posted = action
        if action == "noop" or verb in ("controls", "shield", "wait"):
            return
        if verb == "cancel":
            self.active = None
            return
        me, here, inv, _, _ = self.facts(observation)
        self.active = {
            "action": action,
            "at": now,
            "inventory": dict(inv),
            "position": list(here),
            "dimension": me.get("dimension"),
            "held": item_name(me.get("held_item")),
        }
        if verb in ("navigate", "goto") and len(action.split()) > 1:
            self.recent_stands[action.split()[1]] = now
            candidate = action.split()[1]
            if candidate.startswith("s:"):
                try:
                    xyz = [int(v) for v in candidate[2:].split(",")]
                    if len(xyz) == 3:
                        self.seen_cells.add((xyz[0] // 5, xyz[2] // 5))
                except (TypeError, ValueError):
                    pass
        if verb == "furnace_start" and len(action.split()) > 1:
            self.furnace_started[action.split()[1]] = now
        self.log("posted", action=action, reason=reason, phase=self.last_phase)

    def support(self, here, blocks):
        floor_y = math.floor(here[1]) - 1
        floor_x, floor_z = math.floor(here[0]), math.floor(here[2])
        possible = []
        for row in blocks:
            if row["_name"] not in SUPPORT_BLOCKS:
                continue
            if not row.get("in_reach") or row.get("visible") is False:
                continue
            q = row.get("_point")
            if not q or abs(q[1] - floor_y) > 1:
                continue
            # An up-face placement directly below the player targets occupied
            # space; an adjacent support is the safe placement transaction.
            if math.floor(q[0]) == floor_x and math.floor(q[2]) == floor_z:
                continue
            possible.append(row)
        possible.sort(key=lambda row: row["_distance"])
        return possible[0] if possible else None

    def nearby_table(self, blocks):
        choices = [row for row in blocks if row["_name"] == "crafting_table" and row["_distance"] <= 4.5]
        choices.sort(key=lambda row: row["_distance"])
        return choices[0] if choices else None

    def usable(self, row, names, now):
        name = row.get("_name", "")
        if not (name in names or any(str(suffix).startswith("_") and name.endswith(str(suffix)) for suffix in names)):
            return False
        if row.get("harvestable") is False or not (row.get("visible") or row.get("in_reach")):
            return False
        return not self.is_blocked("gather %s 1" % row.get("id"), now)

    def add(self, options, action, purpose, priority, now):
        if not action or self.is_blocked(action, now):
            return False
        if any(entry["action"] == action for entry in options):
            return False
        options.append({"action": str(action), "purpose": purpose, "priority": priority})
        return True

    def add_gather(self, options, blocks, names, purpose, priority, now, count=1):
        choices = [row for row in blocks if row.get("id") and self.usable(row, names, now)]
        choices.sort(key=lambda row: (not bool(row.get("visible")), not bool(row.get("in_reach")), row["_distance"]))
        if not choices:
            return False
        return self.add(options, "gather %s %d" % (choices[0]["id"], count), purpose, priority, now)

    def scout(self, options, here, blocks, purpose, priority, now):
        stands = []
        for row in blocks:
            ident = str(row.get("id", ""))
            if not ident.startswith("s:") or not (3.0 <= row["_distance"] <= 36.0):
                continue
            if self.recent_stands.get(ident, -1000.0) + 20.0 > now:
                continue
            try:
                xyz = [int(v) for v in ident[2:].split(",")]
                if len(xyz) != 3:
                    continue
            except (TypeError, ValueError):
                continue
            cell = (xyz[0] // 5, xyz[2] // 5)
            novelty = 1 if cell not in self.seen_cells else 0
            stands.append((novelty, min(row["_distance"], 30.0), row))
        if not stands:
            return False
        # Novel cells dominate, then a farther safe stand opens the next local
        # observation window without replaying a failed exact candidate.
        stands.sort(key=lambda value: (value[0], value[1]), reverse=True)
        row = stands[0][2]
        return self.add(options, "navigate " + str(row["id"]), purpose, priority, now)

    @staticmethod
    def total(inv, suffix):
        return sum(int(value) for key, value in inv.items() if key.endswith(suffix) and value)

    @staticmethod
    def wood_plank_item(log_name):
        if log_name.endswith("_log"):
            return log_name[:-4] + "_planks"
        if log_name.endswith("_stem"):
            return log_name[:-5] + "_planks"
        return None

    def feasible_outputs(self, item, inv):
        """Return the largest legal NEW_COUNT for the known recipe inputs."""
        if item.endswith("_planks"):
            base = item[:-7]
            source = inv.get(base + "_log", 0) + inv.get(base + "_stem", 0)
            return source * 4
        planks = self.total(inv, "_planks")
        sticks = inv.get("stick", 0)
        cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
        if item == "stick":
            return (planks // 2) * 4
        if item == "crafting_table":
            return planks // 4
        if item == "wooden_pickaxe":
            return min(planks // 3, sticks // 2)
        if item == "stone_pickaxe":
            return min(cobble // 3, sticks // 2)
        if item == "stone_sword":
            return min(cobble // 2, sticks)
        if item == "furnace":
            return cobble // 8
        if item == "shield":
            # The shield recipe consumes five planks and one iron ingot.
            return min(inv.get("iron_ingot", 0), planks // 5)
        if item == "bucket":
            return inv.get("iron_ingot", 0) // 3
        return 0

    def add_craft(self, options, item, desired, inv, table, purpose, priority, now):
        available = int(max(0, self.feasible_outputs(item, inv)))
        count = min(int(max(1, desired)), available)
        if count < 1:
            return False
        action = "craft %s %d" % (item, count)
        if table is not None:
            action += " " + str(table["id"])
        return self.add(options, action, purpose, priority, now)

    def place_transaction(self, options, item, support, me, inv, purpose, priority, now):
        if not inv.get(item) or support is None:
            return False
        held = item_name(me.get("held_item"))
        if held != item:
            return self.add(options, "equip %s hand" % item, "Hold the carried %s before placing it" % item, priority + 1, now)
        return self.add(options, "place %s %s up" % (item, support["id"]), purpose, priority, now)

    def table_action(self, options, me, inv, table, support, purpose, priority, now):
        if table is not None:
            if table["_distance"] > 4.0:
                return self.add(options, "navigate " + str(table["id"]), purpose, priority, now)
            return False
        if inv.get("crafting_table"):
            return self.place_transaction(options, "crafting_table", support, me, inv, "Place the carried crafting table on an adjacent support", priority, now)
        return False

    def emergency(self, observation, me, here, inv, blocks, entities, now):
        if me.get("dead"):
            return "respawn"
        if number(observation.get("observation_age_s"), 0) > 3.0:
            return "cancel" if observation.get("task") else "noop"
        if me.get("is_in_lava"):
            return "controls forward,jump"
        if me.get("is_in_water") and number(me.get("oxygen"), 20) < 6:
            return "controls jump"
        if number(me.get("health"), 20) <= 7:
            food = next((food for food in FOODS if inv.get(food)), None)
            if food:
                return "eat " + food
        threats = [row for row in entities if row.get("hostile") and row.get("visible") and row["_distance"] < 8.5]
        if not threats:
            return None
        if observation.get("task"):
            return "cancel"
        weapon = next((weapon for weapon in WEAPONS if inv.get(weapon)), None)
        close = [row for row in threats if row["_distance"] <= 3.2 and row.get("id") is not None]
        held = item_name(me.get("held_item"))
        if close and weapon and held != weapon:
            return "equip " + weapon
        if close and weapon and now - self.last_attack >= 0.9:
            self.last_attack = now
            return "attack " + str(close[0]["id"])
        if inv.get("shield"):
            return "shield on"
        safe = [row for row in blocks if str(row.get("id", "")).startswith("s:") and 4.0 < row["_distance"] < 20.0]
        if safe:
            threat_point = threats[0].get("_point") or here
            safe.sort(key=lambda row: distance(row.get("_point") or here, threat_point), reverse=True)
            return "navigate " + str(safe[0]["id"])
        return "controls back,jump"

    def compile_options(self, observation, now):
        me, here, inv, blocks, entities = self.facts(observation)
        options = []
        table = self.nearby_table(blocks)
        support = self.support(here, blocks)
        logs = [key for key, value in inv.items() if value and key.endswith(WOOD_SUFFIXES)]
        planks = self.total(inv, "_planks")
        sticks = inv.get("stick", 0)
        cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
        pick = next((key for key in PICKAXES if inv.get(key)), None)
        stone_pick = any(inv.get(key) for key in ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"))
        food_count = sum(inv.get(key, 0) for key in FOODS)

        # A successful place consumes the carried table before the block view
        # catches up.  The inventory delta arms this short, non-duplicating wait.
        if table is None and self.table_mark and not inv.get("crafting_table"):
            if now < self.table_wait_until:
                self.add(options, "noop", "Wait for the placed table to enter the candidate view", 150, now)
                return options, "await_table"
            q = self.table_mark
            self.add(options, "goto %.2f %.2f %.2f" % (q[0] + 1.0, q[1], q[2] + 1.0), "Return to the observed workstation", 145, now)
            return options, "locate_table"

        # Hunger is a high-priority legal action, but the progression branch is
        # also compiled so a fed player never becomes an idle-only controller.
        if number(me.get("food"), 20) <= 14:
            food = next((key for key in FOODS if inv.get(key)), None)
            if food:
                self.add(options, "eat " + food, "Restore hunger before the next bounded job", 140, now)

        # Exclusive opening: do not leave the spawn basin until the first log
        # and the workstation transaction have a verified postcondition.
        if not logs and planks == 0 and not inv.get("crafting_table") and table is None:
            if not self.add_gather(options, blocks, set(WOOD_SUFFIXES), "Collect an exposed first log", 125, now):
                self.scout(options, here, blocks, "Search a new safe stand for exposed wood", 110, now)
            return self.rank(options), "find_wood"

        if not inv.get("crafting_table") and table is None:
            if logs and planks < 4:
                self.add_craft(options, self.wood_plank_item(logs[0]) or "oak_planks", 4, inv, None, "Convert one carried log into enough planks for the table", 128, now)
            elif planks >= 4:
                self.add_craft(options, "crafting_table", 1, inv, None, "Craft the opening workstation", 128, now)
            elif logs:
                self.add_craft(options, self.wood_plank_item(logs[0]) or "oak_planks", 4, inv, None, "Make planks for the workstation", 128, now)
            if not options:
                self.add_gather(options, blocks, set(WOOD_SUFFIXES), "Collect another exposed log for the workstation", 120, now)
                if not options:
                    self.scout(options, here, blocks, "Scout beyond a failed wood target", 108, now)
            return self.rank(options), "make_table"

        if inv.get("crafting_table") and table is None:
            self.place_transaction(options, "crafting_table", support, me, inv, "Place the carried workstation", 126, now)
            if not options:
                self.scout(options, here, blocks, "Move to a safe support for the workstation", 105, now)
            return self.rank(options), "place_table"

        # Wooden pick transaction.  Every craft request below is bounded by its
        # actual inputs, rather than asking for a fixed output count.
        if not pick:
            if planks < 3:
                if logs:
                    self.add_craft(options, self.wood_plank_item(logs[0]) or "oak_planks", 4, inv, None, "Make feasible planks for the wooden pick", 120, now)
                else:
                    self.add_gather(options, blocks, set(WOOD_SUFFIXES), "Collect wood for the first pickaxe", 115, now)
            elif sticks < 2:
                if planks >= 2:
                    self.add_craft(options, "stick", 4, inv, None, "Craft feasible pickaxe handles", 119, now)
            elif table is not None and table["_distance"] <= 4.0:
                self.add_craft(options, "wooden_pickaxe", 1, inv, table, "Craft the wooden pickaxe at the observed table", 118, now)
            else:
                self.table_action(options, me, inv, table, support, "Approach the workstation for the wooden pick", 118, now)
            if not options:
                self.scout(options, here, blocks, "Find wood or the observed workstation", 103, now)
            return self.rank(options), "wood_pick"

        # A stone weapon is a cheap defensive transaction before the first
        # longer stone search.
        if not any(inv.get(key) for key in ("stone_sword", "iron_sword", "diamond_sword")) and cobble >= 2 and sticks >= 1:
            if table is not None and table["_distance"] <= 4.0:
                self.add_craft(options, "stone_sword", 1, inv, table, "Craft a basic defensive weapon", 114, now)
            else:
                self.table_action(options, me, inv, table, support, "Approach the table for a defensive weapon", 114, now)
            return self.rank(options), "weapon"

        if not stone_pick:
            if cobble < 3:
                if not self.add_gather(options, blocks, set(STONE_BLOCKS), "Mine exposed stone with the wooden pick", 112, now, 1):
                    self.scout(options, here, blocks, "Scout for an exposed stone seam", 96, now)
            elif sticks < 2:
                self.add_craft(options, "stick", 4, inv, None, "Craft handles for the stone pick", 111, now)
            elif table is not None and table["_distance"] <= 4.0:
                self.add_craft(options, "stone_pickaxe", 1, inv, table, "Craft the stone pickaxe", 110, now)
            else:
                self.table_action(options, me, inv, table, support, "Return to the table for the stone pick", 110, now)
            if not options:
                self.scout(options, here, blocks, "Search for stone instead of replaying old stands", 95, now)
            return self.rank(options), "stone_pick"

        raw_iron = inv.get("raw_iron", 0) + inv.get("iron_ore", 0)
        if raw_iron < 3 and not inv.get("iron_ingot"):
            if not self.add_gather(options, blocks, {"iron_ore", "deepslate_iron_ore"}, "Collect exposed iron", 108, now):
                self.add_gather(options, blocks, {"coal_ore", "deepslate_coal_ore"}, "Collect exposed furnace fuel", 78, now)
        if not inv.get("coal") and not inv.get("charcoal"):
            self.add_gather(options, blocks, {"coal_ore", "deepslate_coal_ore"}, "Collect coal for smelting", 76, now)

        furnace_rows = [row for row in blocks if row["_name"] == "furnace"]
        furnace_rows.sort(key=lambda row: row["_distance"])
        furnace = furnace_rows[0] if furnace_rows else None
        if raw_iron and furnace is None and not inv.get("furnace") and cobble >= 8:
            if table is not None and table["_distance"] <= 4.0:
                self.add_craft(options, "furnace", 1, inv, table, "Craft a furnace for the iron", 104, now)
            else:
                self.table_action(options, me, inv, table, support, "Approach the table for a furnace", 104, now)
        elif raw_iron and furnace is None and inv.get("furnace"):
            self.place_transaction(options, "furnace", support, me, inv, "Place the carried furnace", 103, now)
        elif raw_iron and furnace is not None and furnace["_distance"] > 4.0:
            self.add(options, "navigate " + str(furnace["id"]), "Approach the observed furnace", 102, now)
        elif raw_iron and furnace is not None and furnace["_distance"] <= 4.0:
            fuel = next((key for key in ("coal", "charcoal") if inv.get(key)), None)
            if fuel is None:
                fuel = next((key for key, value in inv.items() if key.endswith("_planks") and value), None)
            started = self.furnace_started.get(str(furnace["id"]), -1000.0)
            if fuel and now - started > 35.0:
                self.add(options, "furnace_start %s raw_iron %d %s %d" % (furnace["id"], min(3, raw_iron), fuel, min(3, inv[fuel])), "Load real iron and fuel", 101, now)
            elif now - started >= 30.0:
                self.add(options, "furnace_take " + str(furnace["id"]), "Retrieve finished iron", 100, now)

        if inv.get("iron_ingot") and not inv.get("shield"):
            if planks < 6 and logs:
                self.add_craft(options, self.wood_plank_item(logs[0]) or "oak_planks", 8, inv, None, "Make feasible shield planks", 107, now)
            elif table is not None and table["_distance"] <= 4.0:
                self.add_craft(options, "shield", 1, inv, table, "Craft the shield milestone", 106, now)
            else:
                self.table_action(options, me, inv, table, support, "Return to the table for the shield", 106, now)

        # Opportunistic bed/food work is compiled only from visible entities or
        # real blocks, so it cannot invent a village or a drop.
        if not any(key.endswith("_bed") and value for key, value in inv.items()):
            bed_rows = [row for row in blocks if row["_name"].endswith("_bed") and self.usable(row, {row["_name"]}, now)]
            if bed_rows:
                bed_rows.sort(key=lambda row: row["_distance"])
                self.add(options, "gather %s 1" % bed_rows[0]["id"], "Collect an observed bed", 90, now)
            wool = next((key for key, value in inv.items() if key.endswith("_wool") and value >= 3), None)
            if wool and planks >= 3 and table is not None and table["_distance"] <= 4.0:
                self.add_craft(options, wool.replace("_wool", "_bed"), 1, inv, table, "Craft a bed from observed wool", 89, now)

        animals = [row for row in entities if row["_name"] in ("sheep", "cow", "pig", "chicken") and row.get("visible")]
        animals.sort(key=lambda row: (row["_name"] != "sheep", row["_distance"]))
        if food_count < 5 or not any(key.endswith("_bed") and value for key, value in inv.items()):
            if not self.add_gather(options, blocks, {"hay_block", "sweet_berry_bush", "melon"}, "Collect a visible food block", 88, now) and animals:
                animal = animals[0]
                weapon = next((key for key in WEAPONS if inv.get(key)), None)
                if weapon and animal["_distance"] <= 3.2:
                    if item_name(me.get("held_item")) != weapon:
                        self.add(options, "equip " + weapon, "Equip a weapon before harvesting", 87, now)
                    else:
                        self.add(options, "attack " + str(animal["id"]), "Harvest a visible animal", 86, now)
                elif animal.get("_point") and animal["_distance"] <= 40:
                    q = animal["_point"]
                    self.add(options, "goto %.2f %.2f %.2f" % tuple(q), "Approach a visible animal", 75, now)

        drops = [row for row in entities if row["_name"] in ("item", "item_stack") and row.get("_point") and row["_distance"] < 8]
        if drops:
            q = drops[0]["_point"]
            self.add(options, "goto %.2f %.2f %.2f" % tuple(q), "Collect a nearby real item drop", 80, now)
        if not options:
            self.scout(options, here, blocks, "Scout a new safe resource frontier", 55, now)
        if not options:
            self.add_gather(options, blocks, set(WOOD_SUFFIXES), "Collect a cooled fallback log", 40, now)
        return self.rank(options), "resource_frontier"

    @staticmethod
    def rank(options):
        options.sort(key=lambda row: row.get("priority", 0), reverse=True)
        return options[:7]

    def start_jev(self, observation, options, phase, now):
        if self.pending is not None or not options:
            return
        remaining = self.ctx.budget.remaining()
        if remaining.get("system1_calls", 0) <= 2:
            return
        me, here, inv, _, entities = self.facts(observation)
        threats = []
        for row in entities:
            if row.get("hostile") and row.get("visible"):
                threats.append({
                    "kind": row["_name"],
                    "range": "melee" if row["_distance"] <= 3.2 else "near",
                    "id": str(row.get("id", "")),
                })
        state = {
            "objective": "Free the End in ordinary survival; maximize safe verified milestones",
            "phase": phase,
            "dimension": str(me.get("dimension", "overworld")),
            "health": "critical" if number(me.get("health"), 20) < 7 else "low" if number(me.get("health"), 20) < 12 else "stable",
            "hunger": "hungry" if number(me.get("food"), 20) < 14 else "fed",
            "inventory": {key: int(value) for key, value in sorted(inv.items()) if value and (key in FOODS or key in PICKAXES or key in WEAPONS or key in {"crafting_table", "stick", "cobblestone", "raw_iron", "iron_ore", "iron_ingot", "shield", "furnace"} or key.endswith(("_log", "_planks", "_wool", "_bed")))},
            "threats": threats[:5],
            "options": [{"command": row["action"], "purpose": row["purpose"]} for row in options],
            "recent_failures": list(sorted(self.bad_until))[-5:],
            "rule": "Return exactly one command string from options. Never invent an ID, count, coordinate, or verb; Python owns legality and arithmetic.",
        }
        criteria = {row["action"]: row["purpose"] for row in options}
        questions = {
            "action": Choice(
                "Choose the safest useful exact command from the current compiled options. Prefer verified progression, but avoid work when a visible close hostile makes it unsafe.",
                criteria,
            ),
            "safe": Noul(
                "Is there no visible close hostile threat that makes the compiled work unsafe?",
                {
                    "true": "No visible hostile is close enough to make the listed work unsafe.",
                    "false": "A visible close hostile or explicit immediate hazard makes work unsafe.",
                },
            ),
        }
        seq = observation.get("seq")
        episode = observation.get("episode")
        try:
            future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=2.0))
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception:
            self.log("jev_spawn_failed", phase=phase)
            return
        self.pending = {
            "future": future,
            "started": now,
            "seq": seq,
            "episode": episode,
            "commands": [row["action"] for row in options],
            "committed": False,
        }

    def finish_jev(self, observation, options, now):
        if self.pending is None:
            return None
        pending = self.pending
        future = pending["future"]
        if not future.done():
            if not pending["committed"] and now - pending["started"] >= 0.8:
                pending["committed"] = True
                return options[0]["action"] if options else "noop"
            return "noop"
        self.pending = None
        if pending["committed"]:
            try:
                future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                pass
            return None
        try:
            result = future.result()
            answer = result.get("action") if isinstance(result, dict) else None
            safe = result.get("safe") if isinstance(result, dict) else None
            value = getattr(answer, "value", None)
            probabilities = getattr(answer, "probabilities", {}) or {}
            current = {row["action"] for row in options}
            seq_now = observation.get("seq")
            fresh = (
                pending.get("episode") == observation.get("episode")
                and (pending.get("seq") is None or seq_now is None or number(seq_now) - number(pending.get("seq")) <= 10)
                and value in current
            )
            probability = number(probabilities.get(value), 0.0) if isinstance(probabilities, dict) else 0.0
            safe_probability = number(getattr(safe, "value", 1.0), 1.0)
            if fresh and probability >= 0.55 and safe_probability >= 0.35:
                self.log("jev_command", action=value, probability=probability, phase=self.last_phase)
                return value
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError, KeyError):
            self.log("jev_fallback", phase=self.last_phase)
        return None

    def run_episode(self, game):
        ctx = self.ctx
        while not ctx.done:
            now = ctx.clock.now()
            observation = ctx.observe()
            if not isinstance(observation, dict):
                self.post("noop", {}, now, "missing_observation")
                ctx.wait_for_tick()
                continue
            me, here, inv, blocks, entities = self.facts(observation)
            self.read_feedback(observation, now)
            self.update_active(observation, now, me, here, inv)
            urgent = self.emergency(observation, me, here, inv, blocks, entities, now)
            if urgent:
                self.post(urgent, observation, now, "safety_override")
                ctx.wait_for_tick()
                continue
            if observation.get("task"):
                self.post("noop", observation, now, "continue_bounded_job")
                ctx.wait_for_tick()
                continue
            if self.active and now - self.active.get("at", now) < 1.0:
                self.post("noop", observation, now, "one_shot_observation_lease")
                ctx.wait_for_tick()
                continue
            if number(observation.get("observation_age_s"), 0) > 3.0:
                self.post("cancel" if observation.get("task") else "noop", observation, now, "stale_snapshot")
                ctx.wait_for_tick()
                continue
            options, phase = self.compile_options(observation, now)
            self.last_phase = phase
            if self.pending is None:
                self.start_jev(observation, options, phase, now)
            chosen = self.finish_jev(observation, options, now)
            if chosen is None and self.pending is not None:
                self.post("noop", observation, now, "jev_pending")
                ctx.wait_for_tick()
                continue
            if chosen is None:
                chosen = options[0]["action"] if options else "noop"
            allowed = {row["action"] for row in options}
            if chosen not in allowed:
                chosen = options[0]["action"] if options else "noop"
            self.post(chosen, observation, now, "compiled_or_jev")
            ctx.wait_for_tick()


def build(ctx):
    return Controller(ctx)
