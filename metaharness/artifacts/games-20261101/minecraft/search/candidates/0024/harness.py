"""Milestone scavenger for the Minecraft real-time adaptor.

This controller deliberately prioritizes the cheap, permanent opening flags
(food and a bed) before starting a risky stone expedition.  It also treats a
placed crafting table as a local tool, never as a distant coordinate to which
it must repeatedly return.  All decisions are compiled from the current public
observation; Jev may select among those exact transactions but cannot invent an
action or coordinate.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


FOODS = {
    "apple", "baked_potato", "beetroot", "beetroot_soup", "bread",
    "carrot", "chorus_fruit", "cooked_beef", "cooked_chicken",
    "cooked_cod", "cooked_mutton", "cooked_porkchop", "cooked_rabbit",
    "cooked_salmon", "dried_kelp", "glow_berries", "golden_apple",
    "melon_slice", "mushroom_stew", "potato", "pumpkin_pie", "rabbit_stew",
    "beef", "chicken", "cod", "mutton", "porkchop", "rabbit", "salmon",
    "raw_beef", "raw_chicken", "raw_cod", "raw_mutton", "raw_porkchop",
    "raw_rabbit", "raw_salmon", "rotten_flesh", "sweet_berries",
}
WOOD_ENDINGS = ("_log", "_stem")
STONE_NAMES = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
SUPPORT_NAMES = {
    "grass_block", "dirt", "coarse_dirt", "podzol", "mycelium", "stone",
    "cobblestone", "deepslate", "cobbled_deepslate", "sand", "gravel",
}
PICKS = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")
WEAPONS = ("iron_sword", "stone_sword", "diamond_sword", "netherite_sword", "iron_axe", "stone_axe", "wooden_sword", "wooden_axe")
FAIL_WORDS = ("failed", "error", "unreachable", "timeout", "rejected", "not enough", "not in reach", "no path", "busy", "invalid")
USEFUL_LOOT = FOODS | {
    "crafting_table", "iron_ingot", "raw_iron", "iron_nugget", "iron_pickaxe",
    "iron_sword", "iron_axe", "iron_helmet", "iron_chestplate", "iron_leggings",
    "iron_boots", "shield", "obsidian", "flint_and_steel", "oak_log", "spruce_log",
    "birch_log", "jungle_log", "acacia_log", "dark_oak_log", "mangrove_log",
    "cherry_log", "crimson_stem", "warped_stem", "oak_planks", "spruce_planks",
    "birch_planks", "jungle_planks", "acacia_planks", "dark_oak_planks",
    "mangrove_planks", "cherry_planks", "stick", "string", "wheat", "wool",
}


def num(value, default=0.0):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def name(value):
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("id", ""))))
    return str(value or "").removeprefix("minecraft:")


def rows(value):
    return [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []


def pos(value):
    if not isinstance(value, dict):
        return None
    value = value.get("position", value)
    if isinstance(value, dict) and all(k in value for k in ("x", "y", "z")):
        value = [value["x"], value["y"], value["z"]]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [num(value[0]), num(value[1]), num(value[2])]
    return None


def dist(a, b):
    try:
        return math.dist(a, b)
    except (TypeError, ValueError):
        return 999.0


def inventory(obs):
    """Return audited personal items; a labelled client view is a lag fallback."""
    src = obs.get("inventory")
    if src is None:
        src = obs.get("inventory_authoritative", {})
    if isinstance(src, dict) and isinstance(src.get("items"), (dict, list)):
        src = src["items"]
    out = {}
    if isinstance(src, dict):
        for k, v in src.items():
            n = name(k)
            c = int(max(0, num(v.get("count", 0) if isinstance(v, dict) else v)))
            if n and c:
                out[n] = out.get(n, 0) + c
    elif isinstance(src, list):
        for row in src:
            if isinstance(row, dict):
                n = name(row)
                c = int(max(0, num(row.get("count", 0))))
                if n and c:
                    out[n] = out.get(n, 0) + c
    if not out and isinstance(obs.get("inventory_client"), dict):
        for k, v in obs["inventory_client"].items():
            n, c = name(k), int(max(0, num(v)))
            if n and c:
                out[n] = c
    return out


class Program:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last_posted = "noop"
        self.active = None
        self.pending = None
        self.feedback_seen = None
        self.cooldown = {}
        self.last_attack = {}
        self.last_scout = {}
        self.smelting_since = {}
        self.last_phase = "opening"

    def log(self, event, **data):
        self.ctx.log(event, **data)

    def facts(self, obs):
        me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
        here = pos(me) or [0.0, 64.0, 0.0]
        inv = inventory(obs)
        blocks = []
        for raw in rows(obs.get("candidates")):
            if not raw.get("id"):
                continue
            b = dict(raw)
            b["_name"] = name(raw.get("name", raw.get("block", "")))
            b["_pos"] = pos(raw)
            b["_dist"] = num(raw.get("distance"), dist(here, b["_pos"]) if b["_pos"] else 999.0)
            blocks.append(b)
        entities = []
        for raw in rows(obs.get("entities", obs.get("nearby_entities", []))):
            e = dict(raw)
            e["_name"] = name(raw.get("name", raw.get("type", "")))
            e["_pos"] = pos(raw)
            e["_dist"] = num(raw.get("distance"), dist(here, e["_pos"]) if e["_pos"] else 999.0)
            entities.append(e)
        return me, here, inv, blocks, entities

    @staticmethod
    def key(action):
        p = str(action).split()
        if len(p) > 1 and p[0] in ("gather", "navigate", "open_container", "sleep", "goto"):
            return p[0] + " " + p[1]
        return str(action)

    def is_blocked(self, action, now):
        return now < self.cooldown.get(self.key(action), -1.0)

    def mark_bad(self, action, now, reason):
        if not action or action in ("noop", "cancel"):
            return
        verb = action.split()[0]
        duration = 140.0 if verb == "gather" else 75.0 if verb in ("navigate", "goto") else 35.0
        k = self.key(action)
        self.cooldown[k] = max(self.cooldown.get(k, -1.0), now + duration)
        self.log("transaction_failed", action=action, reason=str(reason)[:100], cooldown_s=duration)
        if self.active and self.active.get("action") == action:
            self.active = None

    def feedback(self, obs, now):
        # Read only the most recent adaptor fields; do not reinterpret the whole
        # event history, which can contain old failures after a new success.
        vals = [obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")]
        try:
            marker = json.dumps(vals, sort_keys=True, default=str)
        except Exception:
            marker = repr(vals)
        if marker == self.feedback_seen:
            return
        self.feedback_seen = marker
        if self.active and any(word in marker.lower() for word in FAIL_WORDS):
            self.mark_bad(self.active["action"], now, "latest_feedback")

    def action_timeout(self, action):
        verb = action.split()[0]
        if verb == "gather":
            return 28.0
        if verb in ("navigate", "goto"):
            return 17.0
        if verb in ("craft", "equip", "place", "open_container", "transfer", "furnace_start", "furnace_take", "activate_block", "sleep"):
            return 8.0
        return 6.0

    def update_active(self, obs, now, me, here, inv):
        if not self.active:
            return
        a = self.active["action"]
        before = self.active
        moved = a.startswith(("navigate ", "goto ")) and dist(here, before["position"]) >= 1.0
        changed_inv = inv != before["inventory"]
        changed_hand = name(me.get("held_item")) != before["held"]
        dim_change = me.get("dimension") != before["dimension"]
        if changed_inv or moved or changed_hand or dim_change:
            self.log("transaction_progress", action=a, inventory_changed=changed_inv, moved=moved)
            self.active = None
            return
        elapsed = now - before["at"]
        if elapsed >= self.action_timeout(a) and not obs.get("task"):
            self.mark_bad(a, now, "no_observed_postcondition")
        elif elapsed > self.action_timeout(a) + 8 and obs.get("task"):
            # Do not interrupt jobs before their documented bounded deadline.
            self.log("job_still_running", action=a, elapsed_s=round(elapsed, 1))

    def post(self, action, obs, now, reason):
        if self.ctx.done:
            return
        action = str(action or "noop").strip() or "noop"
        verb = action.split()[0]
        if action == self.last_posted and verb not in ("noop", "controls", "attack", "shield"):
            action = "noop"
            verb = "noop"
        self.ctx.act(action)
        self.last_posted = action
        if action in ("noop", "controls none"):
            return
        if verb == "cancel":
            self.active = None
            return
        if verb == "attack":
            parts = action.split()
            if len(parts) > 1:
                self.last_attack[parts[1]] = now
            return
        if verb in ("controls", "shield", "wait", "respawn", "wake", "close_window"):
            return
        me, here, inv, _, _ = self.facts(obs)
        self.active = {
            "action": action, "at": now, "inventory": dict(inv), "position": list(here),
            "held": name(me.get("held_item")), "dimension": me.get("dimension"),
        }
        if verb == "furnace_start":
            parts = action.split()
            if len(parts) > 1:
                self.smelting_since[parts[1]] = now
        if verb == "navigate" and len(action.split()) > 1:
            self.last_scout[action.split()[1]] = now
        self.log("action_posted", action=action, phase=self.last_phase, reason=reason)

    def add(self, out, action, purpose, priority, now):
        if not action or self.is_blocked(action, now):
            return
        if any(row["action"] == action for row in out):
            return
        out.append({"action": str(action), "purpose": str(purpose), "priority": int(priority)})

    def choose_candidate(self, blocks, accepts, here, now):
        found = [b for b in blocks if accepts(b) and (b.get("visible") or b.get("in_reach")) and b.get("harvestable") is not False and not self.is_blocked("gather %s 1" % b["id"], now)]
        found.sort(key=lambda b: (not bool(b.get("in_reach")), not bool(b.get("visible")), b["_dist"]))
        return found[0] if found else None

    def add_gather(self, out, blocks, accepts, why, priority, now, count=1):
        row = self.choose_candidate(blocks, accepts, None, now)
        if row:
            self.add(out, "gather %s %d" % (row["id"], max(1, min(4, count))), why, priority, now)
            return True
        return False

    def stands(self, out, blocks, here, why, priority, now, near_point=None):
        choices = []
        for b in blocks:
            ident = str(b.get("id", ""))
            if not ident.startswith("s:") or not b.get("_pos") or not (4 <= b["_dist"] <= 38):
                continue
            if now - self.last_scout.get(ident, -1000.0) < 28:
                continue
            if self.is_blocked("navigate " + ident, now):
                continue
            # Favor an unvisited outward stand, or one that is close to the
            # visible resource actually being sought. No seed/map inference.
            novelty = min(b["_dist"], 32.0)
            target = dist(b["_pos"], near_point) if near_point else 0.0
            rank = novelty - target * 0.5 + min(b["_dist"], 20.0) * 0.1
            choices.append((rank, b))
        if not choices:
            return False
        choices.sort(key=lambda pair: pair[0], reverse=True)
        row = choices[0][1]
        self.add(out, "navigate " + str(row["id"]), why, priority, now)
        return True

    def support(self, here, blocks):
        fx, fy, fz = math.floor(here[0]), math.floor(here[1]), math.floor(here[2])
        ok = []
        for b in blocks:
            q = b.get("_pos")
            if b["_name"] not in SUPPORT_NAMES or not q or not b.get("in_reach") or b.get("visible") is False:
                continue
            if abs(q[1] - (fy - 1)) > 1 or (math.floor(q[0]) == fx and math.floor(q[2]) == fz):
                continue
            ok.append(b)
        ok.sort(key=lambda x: x["_dist"])
        return ok[0] if ok else None

    def nearby(self, blocks, item, maximum=4.0):
        options = [b for b in blocks if b["_name"] == item and b["_dist"] <= maximum]
        options.sort(key=lambda b: b["_dist"])
        return options[0] if options else None

    def emergency(self, obs, me, inv, blocks, entities, now):
        if me.get("dead"):
            return "respawn"
        if num(obs.get("observation_age_s"), 0.0) > 3.0:
            return "cancel" if obs.get("task") else "noop"
        if me.get("is_in_lava"):
            return "controls forward,jump"
        if me.get("is_in_water") and num(me.get("oxygen"), 20) < 6:
            return "controls jump"
        health = num(me.get("health"), 20)
        food = num(me.get("food"), 20)
        eat = next((x for x in FOODS if inv.get(x)), None)
        if eat and (health <= 8 or (health <= 13 and food < 18)):
            return "eat " + eat
        threats = [e for e in entities if e.get("hostile") and e.get("visible") and e["_dist"] < 8]
        if not threats:
            return None
        if obs.get("task"):
            return "cancel"
        if inv.get("shield"):
            return "shield on"
        close = [e for e in threats if e["_dist"] <= 3.2 and e.get("id") is not None]
        weapon = next((w for w in WEAPONS if inv.get(w)), None)
        if close and weapon:
            if name(me.get("held_item")) != weapon:
                return "equip " + weapon
            target = close[0]
            tid = str(target["id"])
            if now - self.last_attack.get(tid, -1000.0) >= 1.0:
                return "attack " + tid
        # Default to leaving the encounter rather than trading unarmored hits.
        safe = [b for b in blocks if str(b.get("id", "")).startswith("s:") and 4 < b["_dist"] < 24]
        if safe:
            danger = threats[0].get("_pos") or [0, 0, 0]
            safe.sort(key=lambda b: dist(b.get("_pos") or [0, 0, 0], danger), reverse=True)
            return "navigate " + str(safe[0]["id"])
        return "controls back,jump"

    def loot_options(self, obs, out, inv, blocks, me, here, entities, now):
        window = obs.get("window")
        if isinstance(window, dict):
            contents = window.get("items", window.get("slots", []))
            counts = {}
            if isinstance(contents, dict):
                for k, v in contents.items():
                    n = name(k)
                    c = v.get("count", 0) if isinstance(v, dict) else v
                    if n:
                        counts[n] = counts.get(n, 0) + int(max(0, num(c)))
            elif isinstance(contents, list):
                for row in contents:
                    if isinstance(row, dict):
                        n = name(row)
                        c = int(max(0, num(row.get("count", 0))))
                        if n and c:
                            counts[n] = counts.get(n, 0) + c
            transfer = next(((n, c) for n, c in counts.items() if c and n in USEFUL_LOOT and inv.get(n, 0) < (3 if n in FOODS else 1)), None)
            if transfer:
                n, c = transfer
                self.add(out, "transfer take %s %d" % (n, min(64, c)), "Take useful real loot from the open container", 135, now)
            else:
                self.add(out, "close_window", "Close a container after useful contents are exhausted", 134, now)
            return True
        # Chest access is a seed-agnostic option, never a command to open an
        # unobserved or distant container. Skip it when a visible threat exists.
        if any(e.get("hostile") and e.get("visible") and e["_dist"] < 8 for e in entities):
            return False
        dim = str(me.get("dimension", "overworld"))
        if "nether" in dim or "end" in dim:
            return False
        if sum(inv.get(f, 0) for f in FOODS) >= 3 and any(k.endswith("_bed") and v for k, v in inv.items()):
            return False
        containers = [b for b in blocks if b["_name"] in ("chest", "barrel") and b.get("in_reach") and b["_dist"] <= 4.2]
        containers.sort(key=lambda b: b["_dist"])
        for b in containers:
            self.add(out, "open_container " + str(b["id"]), "Inspect an observed nearby chest/barrel for food or bed resources", 92, now)
            return bool(out)
        return False

    def compile(self, obs, now):
        me, here, inv, blocks, entities = self.facts(obs)
        out = []
        self.last_phase = "opening"
        table = self.nearby(blocks, "crafting_table", 4.0)
        any_table = self.nearby(blocks, "crafting_table", 30.0)
        support = self.support(here, blocks)
        logs = {k: v for k, v in inv.items() if v and k.endswith(WOOD_ENDINGS)}
        planks = sum(v for k, v in inv.items() if k.endswith("_planks"))
        sticks = inv.get("stick", 0)
        food_total = sum(inv.get(f, 0) for f in FOODS)
        has_bed = any(k.endswith("_bed") and v for k, v in inv.items())
        wool = next((k for k, v in inv.items() if k.endswith("_wool") and v >= 3), None)
        held = name(me.get("held_item"))

        # 1. Keep environmental food/bed opportunities from being postponed
        # behind the entire mining/Nether chain.
        self.loot_options(obs, out, inv, blocks, me, here, entities, now)
        if not has_bed:
            beds = [b for b in blocks if b["_name"].endswith("_bed") and b.get("in_reach") and b.get("harvestable") is not False]
            beds.sort(key=lambda b: b["_dist"])
            if beds:
                day = num((obs.get("world") or {}).get("time_of_day"), 1000) % 24000
                if 12500 <= day <= 23000:
                    self.add(out, "sleep " + str(beds[0]["id"]), "Use an observed bed at night to secure the bed/sleep milestone", 125, now)
                else:
                    self.add(out, "gather %s 1" % beds[0]["id"], "Collect a genuinely observed bed block", 122, now)
            elif wool and table:
                self.add(out, "craft %s_bed 1 %s" % (wool[:-5], table["id"]), "Craft a bed from three carried same-color wool", 119, now)
            elif wool and any_table:
                self.add(out, "navigate " + str(any_table["id"]), "Approach the currently observed table for the real bed recipe", 117, now)
            elif wool and not table and inv.get("crafting_table") and support:
                if held != "crafting_table":
                    self.add(out, "equip crafting_table hand", "Hold the table before placing a local workstation", 111, now)
                else:
                    self.add(out, "place crafting_table %s up" % support["id"], "Place a local workstation for the bed recipe", 110, now)
            elif not wool or sum(v for k, v in inv.items() if k.endswith("_wool")) < 3:
                sheep = [e for e in entities if e["_name"] == "sheep" and e.get("visible") and e.get("id") is not None]
                sheep.sort(key=lambda e: e["_dist"])
                if sheep:
                    animal = sheep[0]
                    if animal["_dist"] <= 3.2:
                        tid = str(animal["id"])
                        if now - self.last_attack.get(tid, -1000.0) >= 1.0:
                            self.add(out, "attack " + tid, "Harvest a visible sheep for real wool toward a bed", 121, now)
                    elif animal.get("_pos") and animal["_dist"] <= 32:
                        self.add(out, "goto %.2f %.2f %.2f" % tuple(animal["_pos"]), "Approach a visible sheep to seek bed wool", 96, now)

        # 2. Score a real food possession before spending the remaining opening
        # on an uncertain stone search. This is still bounded by visible targets.
        if food_total == 0:
            edible = self.choose_candidate(blocks, lambda b: b["_name"] in {"sweet_berry_bush", "melon"}, here, now)
            if edible:
                self.add(out, "gather %s 1" % edible["id"], "Gather an observed food source for the opening milestone", 127, now)
            else:
                animals = [e for e in entities if e["_name"] in {"cow", "pig", "chicken", "rabbit", "sheep"} and e.get("visible") and e.get("id") is not None]
                animals.sort(key=lambda e: e["_dist"])
                if animals:
                    animal = animals[0]
                    if animal["_dist"] <= 3.2:
                        tid = str(animal["id"])
                        if now - self.last_attack.get(tid, -1000.0) >= 1.0:
                            self.add(out, "attack " + tid, "Harvest a visible animal for real food drops", 126, now)
                    elif animal.get("_pos") and animal["_dist"] <= 32:
                        self.add(out, "goto %.2f %.2f %.2f" % tuple(animal["_pos"]), "Approach a visible food animal", 103, now)

        # 3. The opening dependency chain is deterministic, with feasible output
        # counts derived only from personal inventory.
        if not logs and not planks and not inv.get("crafting_table") and not any_table:
            wood = self.choose_candidate(blocks, lambda b: b["_name"].endswith(WOOD_ENDINGS), here, now)
            if wood:
                self.add(out, "gather %s 1" % wood["id"], "Collect one exposed real log", 145, now)
            else:
                self.stands(out, blocks, here, "Scout a different loaded stand for exposed trees", 120, now)
            self.last_phase = "find_wood"
            return self.rank(out), self.last_phase

        if not inv.get("crafting_table") and not any_table:
            if logs:
                log = next(iter(logs))
                self.add(out, "craft %s_planks %d" % (log[:-4] if log.endswith("_log") else log[:-5], min(4, logs[log] * 4)), "Make only the planks supported by carried logs", 140, now)
            elif planks >= 4:
                self.add(out, "craft crafting_table 1", "Craft the opening workstation", 138, now)
            else:
                wood = self.choose_candidate(blocks, lambda b: b["_name"].endswith(WOOD_ENDINGS), here, now)
                if wood:
                    self.add(out, "gather %s 1" % wood["id"], "Gather another exposed log for the table", 130, now)
                else:
                    self.stands(out, blocks, here, "Search outward for exposed wood rather than repeating a failed target", 115, now)
            self.last_phase = "make_table"
            return self.rank(out), self.last_phase

        if inv.get("crafting_table") and not any_table:
            if held != "crafting_table" and support:
                self.add(out, "equip crafting_table hand", "Stage the table in hand before placement", 139, now)
            elif held == "crafting_table" and support:
                self.add(out, "place crafting_table %s up" % support["id"], "Place the held table on an adjacent non-occupied support", 138, now)
            elif not support:
                self.stands(out, blocks, here, "Find an adjacent safe support for the carried crafting table", 110, now)
            self.last_phase = "place_table"
            return self.rank(out), self.last_phase

        # If a table was left behind, do not navigate indefinitely to its cached
        # coordinate. Keep moving with the current task and create a local table
        # only when a recipe actually needs one and materials permit it.
        has_pick = next((p for p in PICKS if inv.get(p)), None)
        if not has_pick:
            if planks < 3 and logs:
                log = next(iter(logs))
                plank_item = (log[:-4] if log.endswith("_log") else log[:-5]) + "_planks"
                self.add(out, "craft %s %d" % (plank_item, min(4, logs[log] * 4)), "Convert a carried log into feasible tool planks", 120, now)
            elif planks >= 2 and sticks < 2:
                self.add(out, "craft stick 4", "Craft sticks only when two planks are present", 119, now)
            elif planks >= 3 and sticks >= 2 and table:
                self.add(out, "craft wooden_pickaxe 1 %s" % table["id"], "Craft a wooden pick at the currently observed workstation", 118, now)
            elif planks >= 3 and sticks >= 2 and any_table:
                self.add(out, "navigate " + str(any_table["id"]), "Approach the currently observed workstation by its fresh candidate ID", 117, now)
            elif planks >= 3 and sticks >= 2 and not table and inv.get("crafting_table") and support:
                if held != "crafting_table":
                    self.add(out, "equip crafting_table hand", "Stage a carried workbench for the wooden pick", 116, now)
                else:
                    self.add(out, "place crafting_table %s up" % support["id"], "Place a local workbench for the wooden pick", 115, now)
            elif planks >= 3 and sticks >= 2 and not any_table and not inv.get("crafting_table") and planks >= 7:
                self.add(out, "craft crafting_table 1", "Craft a replacement workstation from carried planks", 114, now)
            self.last_phase = "wood_pick"
            return self.rank(out), self.last_phase

        # 3. Food is a scored milestone, not just a late combat consumable.
        # Gather berries/melon and make use of real animal drops where possible.
        if food_total == 0:
            berry = self.choose_candidate(blocks, lambda b: b["_name"] in {"sweet_berry_bush", "melon"}, here, now)
            if berry:
                self.add(out, "gather %s 1" % berry["id"], "Gather an observed renewable/edible surface resource", 126, now)
            else:
                animals = [e for e in entities if e["_name"] in {"sheep", "cow", "pig", "chicken", "rabbit"} and e.get("visible") and e.get("id") is not None]
                animals.sort(key=lambda e: e["_dist"])
                if animals:
                    animal = animals[0]
                    if animal["_dist"] <= 3.2:
                        tid = str(animal["id"])
                        if now - self.last_attack.get(tid, -1000.0) >= 1.0:
                            self.add(out, "attack " + tid, "Harvest a visible animal for actual food drops", 124, now)
                    elif animal.get("_pos") and animal["_dist"] <= 36:
                        self.add(out, "goto %.2f %.2f %.2f" % tuple(animal["_pos"]), "Approach an observed animal through bounded navigation", 104, now)
                drops = [e for e in entities if e["_name"] in {"item", "item_stack"} and e["_pos"] and e["_dist"] < 10]
                if drops:
                    q = drops[0]["_pos"]
                    self.add(out, "goto %.2f %.2f %.2f" % tuple(q), "Collect a nearby real dropped food item", 128, now)

        # 4. A bed block can be found later in the same loaded observation;
        # when absent, never invent wool or claim a respawn point.
        if not has_bed and wool is None:
            nearby_bed = [b for b in blocks if b["_name"].endswith("_bed") and b.get("in_reach") and b.get("harvestable") is not False]
            if nearby_bed:
                self.add(out, "gather %s 1" % nearby_bed[0]["id"], "Acquire a visible bed item for the opening milestone", 123, now)

        # 5. Stone/iron are opportunistic only from actual exposed candidates.
        cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
        stone_pick = any(inv.get(p) for p in PICKS[1:])
        if not stone_pick:
            if cobble < 3:
                target = self.choose_candidate(blocks, lambda b: b["_name"] in STONE_NAMES, here, now)
                if target:
                    if held not in PICKS and inv.get("wooden_pickaxe"):
                        self.add(out, "equip wooden_pickaxe hand", "Equip the real wooden pick before mining exposed stone", 104, now)
                    else:
                        self.add(out, "gather %s 1" % target["id"], "Mine one exposed stone candidate with the carried pick", 102, now)
                else:
                    self.stands(out, blocks, here, "Search for a real exposed stone seam", 84, now)
            elif sticks >= 2 and table:
                self.add(out, "craft stone_pickaxe 1 %s" % table["id"], "Craft a stone pick only with three cobble and two sticks", 112, now)
            elif cobble >= 3 and sticks >= 2 and not table and not inv.get("crafting_table") and planks >= 4:
                self.add(out, "craft crafting_table 1", "Create a local workstation rather than returning to a remote one", 111, now)
            elif cobble >= 3 and sticks >= 2 and inv.get("crafting_table") and not table and support:
                if held != "crafting_table":
                    self.add(out, "equip crafting_table hand", "Carry the workstation to the current stone handoff", 110, now)
                else:
                    self.add(out, "place crafting_table %s up" % support["id"], "Place a local workbench at the current stone handoff", 109, now)
            elif cobble >= 3 and sticks < 2 and planks >= 2:
                self.add(out, "craft stick 4", "Make the missing stone-pick handles", 108, now)
            self.last_phase = "stone_pick"
            return self.rank(out), self.last_phase

        raw_iron = inv.get("raw_iron", 0)
        if not raw_iron and not inv.get("iron_ingot"):
            ore = self.choose_candidate(blocks, lambda b: b["_name"] in {"iron_ore", "deepslate_iron_ore"}, here, now)
            if ore:
                self.add(out, "gather %s 1" % ore["id"], "Mine only an observed exposed iron-ore block", 105, now)
            else:
                self.stands(out, blocks, here, "Look for exposed ore candidates", 78, now)
        furnace = self.nearby(blocks, "furnace", 4.0)
        if raw_iron and not furnace and not inv.get("furnace") and cobble >= 8 and table:
            self.add(out, "craft furnace 1 %s" % table["id"], "Craft a furnace from eight real cobblestone", 101, now)
        elif raw_iron and not furnace and inv.get("furnace") and support:
            if held != "furnace":
                self.add(out, "equip furnace hand", "Hold a real furnace before local placement", 100, now)
            else:
                self.add(out, "place furnace %s up" % support["id"], "Place the carried furnace beside the current position", 99, now)
        elif raw_iron and furnace:
            fuel = next((x for x in ("coal", "charcoal") if inv.get(x)), None)
            if not fuel:
                fuel = next((x for x in inv if x.endswith("_planks") and inv[x]), None)
            started = self.smelting_since.get(str(furnace["id"]), -1000.0)
            if fuel and now - started > 35:
                self.add(out, "furnace_start %s raw_iron %d %s %d" % (furnace["id"], min(3, raw_iron), fuel, min(3, inv[fuel])), "Smelt actual iron with carried fuel", 98, now)
            elif now - started >= 30:
                self.add(out, "furnace_take " + str(furnace["id"]), "Retrieve completed real furnace output", 97, now)
            elif not fuel:
                coal = self.choose_candidate(blocks, lambda b: b["_name"] in {"coal_ore", "deepslate_coal_ore"}, here, now)
                if coal:
                    self.add(out, "gather %s 1" % coal["id"], "Gather actual exposed fuel for smelting", 94, now)
        if inv.get("iron_ingot") and not inv.get("shield"):
            if planks >= 5 and table:
                self.add(out, "craft shield 1 %s" % table["id"], "Craft a shield from one ingot and five planks", 110, now)
            elif not table and inv.get("crafting_table") and support:
                if held != "crafting_table":
                    self.add(out, "equip crafting_table hand", "Carry a workbench to the current shield handoff", 109, now)
                else:
                    self.add(out, "place crafting_table %s up" % support["id"], "Place a local workbench for shield crafting", 108, now)
            elif planks < 5 and logs:
                log = next(iter(logs))
                plank_item = (log[:-4] if log.endswith("_log") else log[:-5]) + "_planks"
                self.add(out, "craft %s %d" % (plank_item, min(8, logs[log] * 4)), "Make actual planks for a shield", 106, now)

        # If no score-bearing work is immediately visible, move to a new loaded
        # stand; do not chase a stale cached workstation coordinate.
        if not out:
            self.stands(out, blocks, here, "Explore a safe loaded frontier for the next visible resource", 40, now)
        return self.rank(out), self.last_phase

    @staticmethod
    def rank(options):
        options.sort(key=lambda x: x["priority"], reverse=True)
        return options[:7]

    def launch(self, obs, options, phase, now):
        if self.pending is not None or not options:
            return
        if self.ctx.budget.remaining().get("system1_calls", 0) <= 1:
            return
        me, here, inv, _, entities = self.facts(obs)
        threats = [{"kind": e["_name"], "range": "melee" if e["_dist"] <= 3.2 else "near"}
                   for e in entities if e.get("hostile") and e.get("visible") and e["_dist"] < 8][:4]
        state = {
            "objective": "Earn permanent Free the End preparation milestones without inventing facts.",
            "phase": phase,
            "dimension": str(me.get("dimension", "overworld")),
            "health": "critical" if num(me.get("health"), 20) < 7 else "low" if num(me.get("health"), 20) < 12 else "stable",
            "hunger": "hungry" if num(me.get("food"), 20) < 14 else "fed",
            "position_context": "in_water" if me.get("is_in_water") else "in_lava" if me.get("is_in_lava") else "on_ground" if me.get("on_ground") else "airborne",
            "inventory": {k: int(v) for k, v in inv.items() if k in FOODS or k in {"crafting_table", "stick", "cobblestone", "raw_iron", "iron_ingot", "shield"} or k.endswith(("_log", "_planks", "_wool", "_bed", "_pickaxe"))},
            "visible_threats": threats,
            "options": [{"action": x["action"], "purpose": x["purpose"]} for x in options],
            "rule": "Choose only an exact action listed in options. Python owns counts, coordinates, legality, safety overrides, and postconditions.",
        }
        criteria = {x["action"]: {"purpose": x["purpose"]} for x in options}
        questions = {
            "action": Choice("Which exact compiled action best advances the supplied phase safely? Prefer a real score milestone, but do not choose a close hostile interaction.", criteria),
            "clear": Noul("Do visible_threats show no nearby hostile that makes the chosen ordinary resource action unsafe?", {"true": "No nearby visible hostile is a practical immediate danger.", "false": "A visible close hostile makes ordinary gathering, opening, or crafting unsafe."}),
        }
        try:
            fut = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=2.0))
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("jev_launch_fallback", error=type(exc).__name__)
            return
        self.pending = {"future": fut, "at": now, "seq": obs.get("seq"), "episode": obs.get("episode"), "commands": [x["action"] for x in options], "committed": False}

    def decision(self, obs, options, now):
        if self.pending is None:
            return None
        p = self.pending
        fut = p["future"]
        if not fut.done():
            if not p["committed"] and now - p["at"] >= 0.7:
                p["committed"] = True
                return options[0]["action"] if options else "noop"
            return "noop"
        self.pending = None
        try:
            answers = fut.result()
            if p["committed"]:
                return None
            ans = answers.get("action") if isinstance(answers, dict) else None
            safe = answers.get("clear") if isinstance(answers, dict) else None
            value = getattr(ans, "value", None)
            probs = getattr(ans, "probabilities", {}) or {}
            current = {x["action"] for x in options}
            same_episode = p.get("episode") == obs.get("episode")
            age = num(obs.get("seq"), 0) - num(p.get("seq"), 0)
            probability = num(probs.get(value), 0) if isinstance(probs, dict) else 0
            clear_p = num(getattr(safe, "value", 1.0), 1.0)
            if same_episode and 0 <= age <= 10 and value in current and probability >= 0.55 and clear_p >= 0.35:
                self.log("jev_selected", action=value, probability=probability, phase=self.last_phase)
                return value
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError, KeyError, Exception) as exc:
            # Garbage stand-in answers and provider failures are nonfatal here.
            self.log("jev_answer_fallback", error=type(exc).__name__)
        return None

    def run_episode(self, game):
        try:
            while not self.ctx.done:
                now = self.ctx.clock.now()
                obs = self.ctx.observe()
                if not isinstance(obs, dict):
                    self.post("noop", {}, now, "no_observation")
                    self.ctx.wait_for_tick()
                    continue
                me, here, inv, blocks, entities = self.facts(obs)
                self.feedback(obs, now)
                self.update_active(obs, now, me, here, inv)
                urgent = self.emergency(obs, me, inv, blocks, entities, now)
                if urgent:
                    self.post(urgent, obs, now, "safety_override")
                    self.ctx.wait_for_tick()
                    continue
                if obs.get("task"):
                    self.post("noop", obs, now, "continue_bounded_job")
                    self.ctx.wait_for_tick()
                    continue
                if self.active and now - self.active["at"] < 0.9:
                    self.post("noop", obs, now, "transaction_observation_lease")
                    self.ctx.wait_for_tick()
                    continue
                if num(obs.get("observation_age_s"), 0.0) > 3:
                    self.post("cancel" if obs.get("task") else "noop", obs, now, "stale_snapshot")
                    self.ctx.wait_for_tick()
                    continue
                options, phase = self.compile(obs, now)
                self.last_phase = phase
                if self.pending is None:
                    self.launch(obs, options, phase, now)
                chosen = self.decision(obs, options, now)
                if chosen is None and self.pending is not None:
                    self.post("noop", obs, now, "jev_pending")
                    self.ctx.wait_for_tick()
                    continue
                if chosen not in {x["action"] for x in options}:
                    chosen = options[0]["action"] if options else "noop"
                self.post(chosen, obs, now, "compiled_or_typed_choice")
                self.ctx.wait_for_tick()
        finally:
            # A bounded Jev request is the only background work. Do not return
            # while leaving a worker spinning or waiting unboundedly.
            if self.pending is not None:
                try:
                    self.pending["future"].result(timeout=2.5)
                except Exception:
                    pass
                self.pending = None


def build(ctx):
    return Program(ctx)
