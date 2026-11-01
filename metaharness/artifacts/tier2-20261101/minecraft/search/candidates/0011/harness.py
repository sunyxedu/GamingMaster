"""Closed-loop Minecraft opener: invalidate failed targets and keep a frontier.

The controller deliberately keeps exact action construction and local safety in
Python.  Jev only chooses between the currently executable commands.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


FAIL_WORDS = (
    "failed", "error", "unreachable", "timeout", "cannot", "rejected",
    "illegal", "not enough", "not in reach", "no path", "busy", "invalid",
)
FOODS = (
    "bread", "cooked_beef", "cooked_porkchop", "cooked_mutton",
    "cooked_chicken", "baked_potato", "apple", "carrot", "sweet_berries",
    "beef", "porkchop", "mutton", "chicken", "cod", "salmon",
)
WOOD_SUFFIXES = ("_log", "_stem")
STONE = ("stone", "cobblestone", "deepslate", "cobbled_deepslate")
SOLID_SUPPORT = ("grass_block", "dirt", "coarse_dirt", "stone", "sand", "gravel", "cobblestone")
WEAPONS = ("iron_sword", "stone_sword", "diamond_sword", "stone_axe", "iron_axe", "wooden_axe")
PICKS = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")


def num(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def text_name(value):
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("id", ""))))
    return str(value or "").removeprefix("minecraft:")


def dict_rows(value):
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def position(value):
    if isinstance(value, dict):
        value = value.get("position")
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [num(value[0]), num(value[1]), num(value[2])]
    return None


def distance(a, b):
    try:
        return math.dist(a, b)
    except (TypeError, ValueError):
        return 999.0


def inventory(obs):
    """Use the audited personal inventory, never a container or cursor."""
    source = obs.get("inventory")
    if source is None:
        source = obs.get("inventory_authoritative", {})
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source["items"]
    result = {}
    if isinstance(source, dict):
        for key, value in source.items():
            count = value.get("count", 0) if isinstance(value, dict) else value
            count = int(max(0, num(count)))
            if count:
                result[text_name(key)] = result.get(text_name(key), 0) + count
    elif isinstance(source, list):
        for row in source:
            if not isinstance(row, dict):
                continue
            key = text_name(row)
            count = int(max(0, num(row.get("count", 0))))
            if key and count:
                result[key] = result.get(key, 0) + count
    # Some adaptor revisions expose an empty authoritative object briefly while
    # the labelled client view is already useful for feasibility.  This fallback
    # never treats a cursor/window as personal inventory.
    if not result and isinstance(obs.get("inventory_client"), dict):
        for key, value in obs["inventory_client"].items():
            count = int(max(0, num(value)))
            if count:
                result[text_name(key)] = count
    return result


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last_posted = "noop"
        self.last_issue = -1000.0
        self.active = None
        self.pending = None
        self.feedback_marker = None
        self.failures = {}
        # A failed block interaction is a spatial fact for this episode, not
        # merely a command cooldown.  This is the only structural change from
        # 0008: do not return to the same gather node or placement support.
        self.failed_blocks = set()
        self.frontier_cells = set()
        self.frontier_failures = {}
        self.recent_targets = {}
        self.last_inventory = {}
        self.last_position = None
        self.last_dimension = None
        self.last_phase = "opening"
        self.last_seq = None
        self.furnace_started = {}
        self.tables_seen = set()
        self.beds_seen = set()
        self.deaths = 0

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def obs_facts(self, obs):
        me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
        p = position(me) or [0.0, 64.0, 0.0]
        inv = inventory(obs)
        candidates = []
        for row in dict_rows(obs.get("candidates")):
            if not row.get("id"):
                continue
            item = dict(row)
            item["_name"] = text_name(row.get("name", row.get("block", "")))
            item["_pos"] = position(row)
            item["_distance"] = num(row.get("distance"), distance(p, item["_pos"]) if item["_pos"] else 999.0)
            candidates.append(item)
            if item["_name"] == "crafting_table":
                self.tables_seen.add(str(item["id"]))
            if item["_name"].endswith("_bed"):
                self.beds_seen.add(str(item["id"]))
        entities = []
        for row in dict_rows(obs.get("entities", obs.get("nearby_entities", []))):
            item = dict(row)
            item["_name"] = text_name(row.get("name", row.get("type", "")))
            item["_pos"] = position(row)
            item["_distance"] = num(row.get("distance"), distance(p, item["_pos"]) if item["_pos"] else 999.0)
            entities.append(item)
        return me, p, inv, candidates, entities

    def blocked(self, action, now):
        # Gather retries may use a different count, but the observed block
        # target is the same failed resource.  Normalize before checking.
        block = self.block_target(action)
        if block is not None and block in self.failed_blocks:
            return True
        key = self.target_key(action)
        return now < max(self.failures.get(action, -1.0), self.failures.get(key, -1.0))

    def block_target(self, action):
        """Return the spatial block ID for a gather/place interaction."""
        parts = str(action or "").split()
        if len(parts) >= 2 and parts[0] == "gather":
            return parts[1] or None
        if len(parts) >= 3 and parts[0] == "place":
            # place ITEM REFERENCE_BLOCK_ID FACE: the reference is the block
            # whose face is being clicked, not the item being placed.
            return parts[2] or None
        return None

    def target_key(self, action):
        parts = str(action).split()
        if len(parts) >= 2 and parts[0] in ("gather", "navigate", "open_container", "sleep"):
            return parts[0] + " " + parts[1]
        return str(action)

    def mark_failure(self, action, now, reason="unknown"):
        if not action or action in ("noop", "cancel"):
            return
        key = self.target_key(action)
        verb = str(action).split()[0]
        duration = 150.0 if verb == "gather" else 60.0 if verb == "navigate" else 25.0
        self.failures[key] = max(self.failures.get(key, 0.0), now + duration)
        block = self.block_target(action)
        if block is not None:
            # A failed gather or failed placement is tied to an observed
            # coordinate.  A short timeout caused 0008 to replay the same
            # coordinate every 25/150 seconds for the rest of the episode.
            self.failed_blocks.add(block)
            self.log("target_quarantined", target=block, verb=verb,
                     reason=str(reason)[:120])
        if verb == "navigate":
            parts = str(action).split()
            if len(parts) > 1:
                self.frontier_failures[parts[1]] = now + 120.0
        if self.active and self.active.get("action") == action:
            self.active = None
        self.log("action_invalidated", action=action, reason=str(reason)[:120], blocked_s=duration)

    def process_feedback(self, obs, now):
        values = [obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")]
        marker = json.dumps(values, sort_keys=True, default=str)
        if marker == self.feedback_marker:
            return
        self.feedback_marker = marker
        if self.active and any(word in marker.lower() for word in FAIL_WORDS):
            self.mark_failure(self.active.get("action"), now, "adapter_feedback")

    def active_timeout(self, action):
        verb = str(action).split()[0] if action else ""
        if verb == "gather":
            return 28.0
        if verb in ("navigate", "goto"):
            return 18.0
        if verb in ("craft", "place", "equip", "hotbar", "furnace_start", "furnace_take", "eat", "activate_block"):
            return 7.0
        if verb in ("attack", "use_item", "activate_entity"):
            return 4.0
        return 6.0

    def update_active(self, obs, now, me, p, inv):
        if not self.active:
            return
        action = self.active.get("action", "")
        before_inv = self.active.get("inventory", {})
        before_pos = self.active.get("position") or p
        verb = action.split()[0] if action else ""
        changed_inv = inv != before_inv
        moved = verb in ("navigate", "goto") and distance(p, before_pos) >= 1.0
        dimension_changed = me.get("dimension") != self.active.get("dimension")
        held_changed = text_name(me.get("held_item")) != self.active.get("held")
        if changed_inv or moved or dimension_changed or held_changed:
            self.log("action_progress", action=action, inventory_changed=changed_inv, moved=moved)
            self.active = None
            return
        elapsed = now - self.active.get("at", now)
        # A job that is still reported busy is allowed its documented deadline;
        # an accepted one-shot with no state change is invalidated sooner.
        if elapsed >= self.active_timeout(action) and not obs.get("task"):
            self.mark_failure(action, now, "no_observed_progress")
        elif elapsed >= self.active_timeout(action) + 8.0:
            self.mark_failure(action, now, "job_deadline_without_progress")

    def post(self, action, obs, now, reason=""):
        if self.ctx.done:
            return
        action = str(action or "noop").strip() or "noop"
        verb = action.split()[0]
        # One-shot actions are latched by the adaptor.  A fresh noop is the
        # explicit re-arm barrier before any deliberate identical retry.
        if action == self.last_posted and verb not in ("noop", "controls", "attack", "shield"):
            action = "noop"
            verb = "noop"
        self.ctx.act(action)
        self.last_posted = action
        if action == "noop":
            return
        self.last_issue = now
        if verb == "cancel":
            self.active = None
            return
        if verb in ("controls", "shield"):
            return
        me, p, inv, _, _ = self.obs_facts(obs)
        self.active = {
            "action": action,
            "at": now,
            "inventory": dict(inv),
            "position": list(p),
            "dimension": me.get("dimension"),
            "held": text_name(me.get("held_item")),
        }
        if verb == "navigate":
            parts = action.split()
            if len(parts) > 1:
                self.recent_targets[parts[1]] = now
                row = parts[1]
                if row.startswith("s:"):
                    coords = row[2:].split(",")
                    if len(coords) == 3:
                        cell = (int(num(coords[0]) // 4), int(num(coords[2]) // 4))
                        self.frontier_cells.add(cell)
        if verb == "furnace_start":
            parts = action.split()
            if len(parts) > 1:
                self.furnace_started[parts[1]] = now
        self.log("posted", action=action, reason=reason, phase=self.last_phase)

    def support_block(self, p, candidates):
        floor_y = math.floor(p[1]) - 1
        good = []
        for row in candidates:
            if str(row.get("id", "")) in self.failed_blocks:
                continue
            if row["_name"] not in SOLID_SUPPORT or not row.get("in_reach") or row.get("visible") is False:
                continue
            q = row.get("_pos")
            if not q:
                continue
            if abs(q[1] - floor_y) <= 1 and distance(p, q) > 1.0:
                good.append(row)
        good.sort(key=lambda x: x["_distance"])
        return good[0] if good else None

    def near_table(self, candidates):
        rows = [x for x in candidates if x["_name"] == "crafting_table" and x["_distance"] <= 4.5]
        rows.sort(key=lambda x: x["_distance"])
        return rows[0] if rows else None

    def usable_block(self, row, names):
        exact = row["_name"] in names
        suffix = any(str(part).startswith("_") and row["_name"].endswith(str(part)) for part in names)
        if not (exact or suffix) or row.get("harvestable") is False:
            return False
        if self.blocked("gather %s 1" % row["id"], self.ctx.clock.now()):
            return False
        # A hidden solid interior is not an exposed target.  In-reach is an
        # explicit exception because it is a direct interaction opportunity.
        return bool(row.get("visible") or row.get("in_reach"))

    def add(self, options, action, purpose, priority=0, now=None):
        if not action:
            return
        now = self.ctx.clock.now() if now is None else now
        if self.blocked(action, now):
            return
        if any(x["action"] == action for x in options):
            return
        options.append({"action": action, "purpose": purpose, "priority": priority})

    def add_gather(self, options, candidates, names, purpose, priority, count=1, now=None):
        rows = [x for x in candidates if self.usable_block(x, names)]
        rows.sort(key=lambda x: (not bool(x.get("visible")), not bool(x.get("in_reach")), x["_distance"]))
        if rows:
            self.add(options, "gather %s %d" % (rows[0]["id"], count), purpose, priority, now)
            return True
        return False

    def explore(self, options, p, candidates, purpose="Scout a new safe frontier", priority=15, now=None):
        now = self.ctx.clock.now() if now is None else now
        stands = []
        for row in candidates:
            ident = str(row.get("id", ""))
            if not ident.startswith("s:") or row["_distance"] < 4.0 or row["_distance"] > 32.0:
                continue
            if self.frontier_failures.get(ident, -1000.0) > now:
                continue
            if self.recent_targets.get(ident, -1000.0) + 18.0 > now:
                continue
            coords = ident[2:].split(",")
            if len(coords) != 3:
                continue
            cell = (int(num(coords[0]) // 4), int(num(coords[2]) // 4))
            novelty = 0 if cell in self.frontier_cells else 1000
            # Prefer far, novel stands.  This turns the bounded stand list into
            # a frontier rather than the parent’s repeated local orbit.
            score = novelty + min(row["_distance"], 28.0)
            stands.append((score, row))
        if not stands:
            # If every local stand has been seen, still take the farthest safe
            # candidate after a cooldown; otherwise the bot can freeze in a
            # small basin forever.
            for row in candidates:
                ident = str(row.get("id", ""))
                if ident.startswith("s:") and 4.0 <= row["_distance"] <= 32.0:
                    if (self.frontier_failures.get(ident, -1000.0) <= now
                            and self.recent_targets.get(ident, -1000.0) + 18.0 <= now):
                        stands.append((row["_distance"], row))
        if stands:
            stands.sort(key=lambda pair: pair[0], reverse=True)
            self.add(options, "navigate " + str(stands[0][1]["id"]), purpose, priority, now)
            return True
        return False

    def emergency(self, obs, me, p, inv, candidates, entities, now):
        if me.get("dead"):
            self.deaths += 1
            return "respawn"
        if num(obs.get("observation_age_s"), 0) > 3.0:
            return "cancel" if obs.get("task") else "noop"
        if me.get("is_in_lava"):
            return "controls forward,jump"
        if me.get("is_in_water") and num(me.get("oxygen"), 20) < 6:
            return "controls jump"
        if num(me.get("health"), 20) <= 7 and any(inv.get(x) for x in FOODS):
            food = next(x for x in FOODS if inv.get(x))
            return "eat " + food
        threats = [e for e in entities if e.get("hostile") and e.get("visible") and e["_distance"] < 8.5]
        if not threats:
            return None
        if obs.get("task"):
            return "cancel"
        # Close melee gets one attack when a real weapon is carried; ranged and
        # uncertain threats are handled by a shield or by leaving the basin.
        weapon = next((x for x in WEAPONS if inv.get(x)), None)
        held = text_name(me.get("held_item"))
        close = [e for e in threats if e["_distance"] <= 3.2 and e.get("id") is not None]
        if close and weapon and held != weapon:
            return "equip " + weapon
        if close and weapon and num(now - self.recent_targets.get("attack", -1000.0)) >= 0.9:
            self.recent_targets["attack"] = now
            return "attack " + str(close[0]["id"])
        if inv.get("shield"):
            return "shield on"
        # Without a shield, select a safe stand farthest from the nearest threat.
        safe = [x for x in candidates if str(x.get("id", "")).startswith("s:") and 4 < x["_distance"] < 20]
        if safe:
            threat_p = threats[0].get("_pos") or p
            safe.sort(key=lambda x: distance(x.get("_pos") or p, threat_p), reverse=True)
            return "navigate " + str(safe[0]["id"])
        return "controls back,jump"

    def options(self, obs, now):
        me, p, inv, candidates, entities = self.obs_facts(obs)
        options = []

        def ranked():
            options.sort(key=lambda row: row.get("priority", 0), reverse=True)
            return options[:7]

        table = self.near_table(candidates)
        support = self.support_block(p, candidates)
        logs = [k for k, v in inv.items() if v and k.endswith(WOOD_SUFFIXES)]
        planks = sum(v for k, v in inv.items() if k.endswith("_planks"))
        sticks = inv.get("stick", 0)
        cobble = sum(inv.get(k, 0) for k in ("cobblestone", "cobbled_deepslate"))
        has_pick = next((k for k in PICKS if inv.get(k)), None)
        stone_pick = any(inv.get(k) for k in ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"))
        food_count = sum(inv.get(k, 0) for k in FOODS)
        now = float(now)

        def craft(item, count=1, table_row=None, why="Craft a required item", priority=70):
            action = "craft %s %d" % (item, count)
            if table_row:
                action += " " + str(table_row["id"])
            self.add(options, action, why, priority, now)

        def need_table(why="Approach a nearby workstation", priority=75):
            if table:
                if table["_distance"] > 4.0:
                    self.add(options, "navigate " + str(table["id"]), why, priority, now)
                return True
            if inv.get("crafting_table") and support:
                self.add(options, "place crafting_table %s up" % support["id"], "Place the carried table on observed solid ground", priority, now)
                return True
            return False

        # Hunger is an immediate resource constraint, not a late combat action.
        if num(me.get("food"), 20) <= 14:
            food = next((x for x in FOODS if inv.get(x)), None)
            if food:
                self.add(options, "eat " + food, "Restore hunger before movement or mining", 120, now)

        # Opening dependency chain.  It is deliberately exclusive until each
        # prerequisite is observed, preventing exploration from starving craft.
        if not logs and not planks and not inv.get("crafting_table") and not table:
            if not self.add_gather(options, candidates, set(FOODS) | set(WOOD_SUFFIXES), "Collect an exposed food or tree block", 105, 1, now):
                self.add_gather(options, candidates, set(WOOD_SUFFIXES), "Collect the first exposed log", 110, 1, now)
            if not options:
                self.explore(options, p, candidates, "Search for exposed wood instead of retrying hidden blocks", 95, now)
            return ranked(), "find_wood"

        if not inv.get("crafting_table") and not table:
            if logs and planks < 4:
                log = logs[0]
                craft(log.replace("_log", "_planks").replace("_stem", "_planks"), 4, None, "Turn a carried log into table planks", 115)
            elif planks >= 4:
                craft("crafting_table", 1, None, "Craft the opening workstation", 115)
            elif logs:
                log = logs[0]
                craft(log.replace("_log", "_planks").replace("_stem", "_planks"), 4, None, "Make planks for the workstation", 115)
            else:
                self.add_gather(options, candidates, set(WOOD_SUFFIXES), "Collect an exposed log for the workstation", 110, 1, now)
            if not options:
                self.explore(options, p, candidates, "Scout for wood after a failed local target", 95, now)
            return ranked(), "make_table"

        if inv.get("crafting_table") and not table:
            if support:
                self.add(options, "place crafting_table %s up" % support["id"], "Place the carried workstation beside the player", 112, now)
            else:
                self.explore(options, p, candidates, "Move to a safe support position for the workstation", 88, now)
            return ranked(), "place_table"

        if not has_pick:
            if planks < 3:
                if logs:
                    log = logs[0]
                    craft(log.replace("_log", "_planks").replace("_stem", "_planks"), 8, None, "Make planks for tools", 105)
                else:
                    self.add_gather(options, candidates, set(WOOD_SUFFIXES), "Collect wood for the first pickaxe", 100, 1, now)
            elif sticks < 2:
                craft("stick", 4, None, "Craft pickaxe handles", 104)
            elif need_table("Approach the table before crafting the wooden pickaxe", 103):
                if table and table["_distance"] <= 4.0:
                    craft("wooden_pickaxe", 1, table, "Craft the first pickaxe", 103)
            if not options:
                self.explore(options, p, candidates, "Find a table or wood without retrying a failed target", 85, now)
            return ranked(), "wood_pick"

        # A cheap weapon makes night exploration and sheep/animal harvesting
        # survivable.  It is scheduled before a blind long-range scout.
        if not any(inv.get(k) for k in ("stone_sword", "iron_sword", "diamond_sword")) and cobble >= 2 and sticks >= 1:
            if need_table("Approach the table to craft a defensive weapon", 98) and table and table["_distance"] <= 4.0:
                craft("stone_sword", 1, table, "Craft a basic defensive weapon", 98)
                return ranked(), "weapon"

        if not stone_pick:
            if cobble < 3:
                if not self.add_gather(options, candidates, set(STONE), "Mine exposed stone for the stone pickaxe", 96, 2, now):
                    self.explore(options, p, candidates, "Scout for exposed stone rather than orbiting old stands", 78, now)
            elif sticks < 2:
                craft("stick", 4, None, "Make handles for the stone pickaxe", 96)
            elif need_table("Approach the table before crafting the stone pickaxe", 95) and table and table["_distance"] <= 4.0:
                craft("stone_pickaxe", 1, table, "Craft the stone pickaxe", 95)
            if not options:
                self.explore(options, p, candidates, "Search for the next exposed stone seam", 78, now)
            return ranked(), "stone_pick"

        # Food and bed resources are gathered opportunistically, but exposed
        # iron/fuel wins when it is actually in the current observation.
        raw_iron = inv.get("raw_iron", 0) + inv.get("iron_ore", 0)
        if not inv.get("iron_ingot") and raw_iron < 3:
            if not self.add_gather(options, candidates, {"iron_ore", "deepslate_iron_ore"}, "Mine an exposed iron ore block", 94, 1, now):
                self.add_gather(options, candidates, {"coal_ore", "deepslate_coal_ore"}, "Collect exposed fuel for smelting", 60, 1, now)
        if not inv.get("coal") and not inv.get("charcoal"):
            self.add_gather(options, candidates, {"coal_ore", "deepslate_coal_ore"}, "Mine exposed coal for the furnace", 62, 1, now)

        furnace_rows = [x for x in candidates if x["_name"] == "furnace"]
        furnace_rows.sort(key=lambda x: x["_distance"])
        furnace = furnace_rows[0] if furnace_rows else None
        if raw_iron and not inv.get("furnace") and not furnace and cobble >= 8:
            if need_table("Approach the table to craft a furnace", 91) and table and table["_distance"] <= 4.0:
                craft("furnace", 1, table, "Craft a real furnace for iron", 91)
        elif raw_iron and not furnace and inv.get("furnace"):
            if support:
                self.add(options, "place furnace %s up" % support["id"], "Place the carried furnace on solid ground", 90, now)
        elif raw_iron and furnace and furnace["_distance"] > 4.0:
            self.add(options, "navigate " + str(furnace["id"]), "Approach the observed furnace", 89, now)
        elif raw_iron and furnace and furnace["_distance"] <= 4.0:
            fuel = next((x for x in ("coal", "charcoal") if inv.get(x)), None)
            fuel = fuel or next((x for x in inv if x.endswith("_planks") and inv.get(x)), None)
            started = self.furnace_started.get(str(furnace["id"]), -1000.0)
            if fuel and now - started > 35.0:
                self.add(options, "furnace_start %s raw_iron %d %s %d" % (
                    furnace["id"], min(3, raw_iron), fuel, min(3, inv[fuel])),
                    "Load actual iron and fuel into the furnace", 88, now)
            elif now - started >= 30.0:
                self.add(options, "furnace_take " + str(furnace["id"]), "Retrieve smelted iron after real furnace time", 87, now)

        if inv.get("iron_ingot") and not inv.get("shield"):
            if planks < 6:
                if logs:
                    log = logs[0]
                    craft(log.replace("_log", "_planks").replace("_stem", "_planks"), 8, None, "Make planks for the shield", 100)
            elif need_table("Approach the table to craft the shield", 100) and table and table["_distance"] <= 4.0:
                craft("shield", 1, table, "Craft the shield milestone", 100)

        # Bed blocks and wool are both valid paths to the opening bed flag.
        if not any(k.endswith("_bed") and inv.get(k) for k in inv):
            bed_blocks = [x for x in candidates if x["_name"].endswith("_bed") and self.usable_block(x, {x["_name"]})]
            if bed_blocks:
                bed_blocks.sort(key=lambda x: x["_distance"])
                self.add(options, "gather %s 1" % bed_blocks[0]["id"], "Collect an observed bed for respawn", 84, now)
            wool = next((k for k in inv if k.endswith("_wool") and inv[k] >= 3), None)
            if wool and planks >= 3 and table and table["_distance"] <= 4.0:
                craft(wool.replace("_wool", "_bed"), 1, table, "Craft a bed for a respawn point", 83)

        # Harvest easy food and wool from visible entities only.  The command
        # is compiled from the current entity ID and is invalidated on failure.
        animals = [x for x in entities if x["_name"] in ("sheep", "cow", "pig", "chicken") and x.get("visible")]
        animals.sort(key=lambda x: (x["_name"] != "sheep", x["_distance"]))
        if food_count < 5 or not any(k.endswith("_bed") and inv.get(k) for k in inv):
            if not self.add_gather(options, candidates, {"hay_block", "sweet_berry_bush", "melon"}, "Gather an exposed food source", 82, 1, now):
                if animals:
                    animal = animals[0]
                    weapon = next((x for x in WEAPONS if inv.get(x)), None)
                    held = text_name(me.get("held_item"))
                    if weapon and animal["_distance"] <= 3.2 and animal.get("id") is not None:
                        if held != weapon:
                            self.add(options, "equip " + weapon, "Equip the carried weapon before harvesting", 81, now)
                        else:
                            self.add(options, "attack " + str(animal["id"]), "Harvest a visible animal for food or wool", 80, now)
                    elif animal.get("_pos") and animal["_distance"] <= 40:
                        q = animal["_pos"]
                        self.add(options, "goto %.2f %.2f %.2f" % tuple(q), "Approach a visible animal by bounded pathfinding", 72, now)

        # Pick up a nearby real drop before leaving it behind.
        drops = [x for x in entities if x["_name"] in ("item", "item_stack") and x.get("_pos") and x["_distance"] < 8]
        if drops:
            q = drops[0]["_pos"]
            self.add(options, "goto %.2f %.2f %.2f" % tuple(q), "Walk to an observed dropped item", 76, now)

        if not options:
            self.explore(options, p, candidates, "Scout a novel safe frontier for resources", 45, now)
        if not options:
            self.add_gather(options, candidates, set(WOOD_SUFFIXES), "Collect an exposed fallback log", 35, 1, now)
        return ranked(), "resource_frontier"

    def start_decision(self, obs, options, phase, now):
        if self.pending is not None:
            return
        budget = self.ctx.budget.remaining()
        if budget.get("system1_calls", 0) <= 2 or not options:
            return
        me, p, inv, _, entities = self.obs_facts(obs)
        threats = []
        for row in entities:
            if row.get("hostile") and row.get("visible"):
                threats.append({"kind": row["_name"], "range": "melee" if row["_distance"] <= 3.2 else "near", "id": str(row.get("id", ""))})
        threats = threats[:5]
        state = {
            "objective": "Free the End in vanilla survival; currently maximize safe verified milestones",
            "phase": phase,
            "dimension": str(me.get("dimension", "overworld")),
            "health_band": "critical" if num(me.get("health"), 20) < 7 else "low" if num(me.get("health"), 20) < 12 else "stable",
            "hunger_band": "hungry" if num(me.get("food"), 20) < 14 else "fed",
            "inventory": {k: int(v) for k, v in sorted(inv.items()) if k in set(FOODS) | set(PICKS) | {"crafting_table", "stick", "cobblestone", "raw_iron", "iron_ingot", "shield", "furnace"} or k.endswith(("_log", "_planks", "_wool", "_bed"))},
            "threats": threats,
            "options": [{"command": x["action"], "purpose": x["purpose"]} for x in options],
            "recent_failure_memory": list(sorted(self.failures))[-5:],
            "quarantined_block_targets": sorted(self.failed_blocks)[-8:],
            "rule": "Select only one exact command from options; do not invent coordinates or verbs. Code handles arithmetic and lethal overrides.",
        }
        criteria = {x["action"]: x["purpose"] for x in options}
        questions = {
            "action": Choice(
                "Choose the safest useful exact command for the supplied phase. Prefer a verified progress action over exploration, but do not choose resource work when threats explicitly make it unsafe.",
                criteria,
            ),
            "safe": Noul(
                "Does the supplied state show no visible close hostile threat that makes the listed work unsafe?",
                {"true": "No visible hostile is close; ordinary uncertainty is not a threat.", "false": "A visible hostile is close or the state explicitly shows immediate danger."},
            ),
        }
        seq = obs.get("seq")
        episode = obs.get("episode")
        future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=2.0))
        self.pending = {"future": future, "at": now, "seq": seq, "episode": episode, "options": [x["action"] for x in options], "committed": False}

    def decision(self, obs, options, now):
        if self.pending is None:
            return None
        pending = self.pending
        future = pending["future"]
        if not future.done():
            # Do not let a slow classifier stall the real-time world.  The
            # deterministic first option is used after a short lease; the same
            # future is still harvested when it finishes, so no worker loop is
            # leaked and no second request is launched over it.
            if not pending["committed"] and now - pending["at"] >= 0.8:
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
            answers = future.result()
            answer = answers.get("action") if isinstance(answers, dict) else None
            safe = answers.get("safe") if isinstance(answers, dict) else None
            value = getattr(answer, "value", None)
            probs = getattr(answer, "probabilities", {}) or {}
            current = {x["action"] for x in options}
            fresh = (
                pending.get("episode") == obs.get("episode")
                and (pending.get("seq") is None or obs.get("seq") is None or num(obs.get("seq")) - num(pending.get("seq")) <= 10)
                and value in current
            )
            probability = num(probs.get(value), 0.0) if isinstance(probs, dict) else 0.0
            safe_probability = num(getattr(safe, "value", 1.0), 1.0)
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
            obs = ctx.observe()
            if not isinstance(obs, dict):
                self.post("noop", {}, now, "missing_observation")
                ctx.wait_for_tick()
                continue
            me, p, inv, candidates, entities = self.obs_facts(obs)
            self.process_feedback(obs, now)
            self.update_active(obs, now, me, p, inv)
            if me.get("dead"):
                self.post("respawn", obs, now, "normal_respawn")
                ctx.wait_for_tick()
                continue
            urgent = self.emergency(obs, me, p, inv, candidates, entities, now)
            if urgent:
                self.post(urgent, obs, now, "safety_override")
                ctx.wait_for_tick()
                continue
            if num(obs.get("observation_age_s"), 0) > 3.0:
                self.post("cancel" if obs.get("task") else "noop", obs, now, "stale_snapshot")
                ctx.wait_for_tick()
                continue
            if obs.get("task"):
                self.post("noop", obs, now, "continue_bounded_job")
                ctx.wait_for_tick()
                continue
            # A recently issued one-shot gets a small observation lease so a
            # lagging task field cannot be overwritten by a second job.
            if self.active and now - self.active.get("at", now) < 1.0:
                self.post("noop", obs, now, "action_lease")
                ctx.wait_for_tick()
                continue
            options, phase = self.options(obs, now)
            self.last_phase = phase
            if self.pending is None:
                self.start_decision(obs, options, phase, now)
            chosen = self.decision(obs, options, now)
            if chosen is None and self.pending is not None:
                self.post("noop", obs, now, "jev_pending")
                ctx.wait_for_tick()
                continue
            if chosen is None:
                chosen = options[0]["action"] if options else "noop"
            # Revalidate a model answer against this exact fresh menu.
            if chosen not in {x["action"] for x in options}:
                chosen = options[0]["action"] if options else "noop"
            self.post(chosen, obs, now, "compiled_or_jev")
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
