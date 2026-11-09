"""Lease-aware Minecraft progression controller.

The controller keeps the useful part of 0004 (a code-owned dependency chain and
compiled commands) and the useful part of 0008 (failure memory and tactical
interrupts), but replaces stand cycling with an active, renewed movement survey.
Every resource transaction is closed by an inventory/block postcondition.
"""
from __future__ import annotations

import json
import math
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


FAIL_WORDS = (
    "failed", "failure", "error", "unreachable", "timeout", "cannot",
    "rejected", "illegal", "not enough", "not in reach", "no path",
    "busy", "invalid", "unknown", "out of range",
)
ID_RE = re.compile(r"^[bs]:-?\d+,-?\d+,-?\d+$")
ITEM_RE = re.compile(r"^[a-z0-9_]+$")

WOOD_SUFFIXES = ("_log", "_stem")
PICKAXES = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")
STONE_PICKAXES = ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")
WEAPONS = ("iron_sword", "diamond_sword", "stone_sword", "iron_axe", "stone_axe", "wooden_sword", "wooden_axe")
FOODS = (
    "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
    "cooked_rabbit", "bread", "baked_potato", "carrot", "apple",
    "sweet_berries", "melon_slice", "beef", "porkchop", "mutton",
    "chicken", "rabbit", "cod", "salmon", "potato", "beetroot",
    "wheat", "mushroom_stew",
)
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
    "stone", "cobblestone", "deepslate", "cobbled_deepslate", "dirt",
    "grass_block", "coarse_dirt", "sand", "gravel", "terracotta",
    "netherrack", "blackstone", "oak_log", "birch_log", "spruce_log",
    "jungle_log", "acacia_log", "dark_oak_log", "mangrove_log", "chest",
    "crafting_table", "furnace", "podzol", "mycelium", "sandstone",
    "red_sand", "snow_block", "clay", "moss_block", "basalt", "end_stone",
}


def num(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def item_name(value):
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("id", ""))))
    if not isinstance(value, str):
        return ""
    value = value.removeprefix("minecraft:")
    return value if ITEM_RE.fullmatch(value) else ""


def rows(value):
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def point(value):
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result = tuple(num(x, math.inf) for x in value[:3])
    return result if all(math.isfinite(x) for x in result) else None


def dist(a, b):
    pa, pb = point(a), point(b)
    if pa is None or pb is None:
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def valid_id(value, prefix=None):
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        return ""
    if prefix and not value.startswith(prefix + ":"):
        return ""
    return value


def inv_of(obs):
    """Return only personal inventory; never count a window or cursor."""
    source = obs.get("inventory")
    if source is None or source == {} or source == []:
        source = obs.get("inventory_authoritative", source)
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source["items"]
    answer = {}
    if isinstance(source, dict):
        for key, raw in source.items():
            item = item_name(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(num(raw_count)))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    elif isinstance(source, list):
        for raw in rows(source):
            item = item_name(raw)
            count = max(0, int(num(raw.get("count", 0))))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    # The client view is explicitly labelled personal inventory.  It is useful
    # during the short audit lag at join, but containers are never a fallback.
    if not answer and isinstance(obs.get("inventory_client"), dict):
        for key, raw in obs["inventory_client"].items():
            item = item_name(key)
            count = max(0, int(num(raw.get("count", 0) if isinstance(raw, dict) else raw)))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    return answer


def inv_sig(inv):
    return tuple(sorted((str(k), int(v)) for k, v in inv.items() if v))


def block_name(row):
    return item_name(row.get("name", row.get("block", "")))


def entity_name(row):
    return item_name(row.get("name", row.get("type", "")))


def self_row(obs):
    return obs.get("self") if isinstance(obs.get("self"), dict) else {}


def self_pos(obs):
    return point(self_row(obs).get("position"))


def has_task(obs):
    task = obs.get("task")
    return task not in (None, "", {})


def is_hazard(name):
    return name in HAZARDS or name in {"lava", "water", "bubble_column"}


def food_total(inv):
    return sum(inv.get(x, 0) for x in FOODS)


def wood_total(inv):
    return sum(v for k, v in inv.items() if k.endswith(WOOD_SUFFIXES))


def plank_total(inv):
    return sum(v for k, v in inv.items() if k.endswith("_planks"))


def first_item(inv, choices):
    return next((x for x in choices if inv.get(x, 0) > 0), "")


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.names = set()
        self.last_posted = "noop"
        self.one = None
        self.work = None
        self.portal = None
        self.manual = None
        self.manual_index = 0
        self.manual_last_end = -100.0
        self.failed = {}
        self.completed = set()
        self.bad_support = set()
        self.visits = {}
        self.last_feedback = None
        self.last_inv = ()
        self.last_adv = ()
        self.last_progress = 0.0
        self.last_position = None
        self.furnace_started = {}
        self.furnace_ready = {}
        self.table_expected = None
        self.furnace_expected = None
        self.escape_ticks = 0
        self.last_attack = -100.0
        self.threat_future = None
        self.threat_sig = ()
        self.threat_at = -100.0
        self.threat_cached = None
        self.astra_future = None
        self.astra_session = None
        self.astra_last = -1000.0
        self.astra_version = 0
        self.focus = "balanced"
        self.focus_deadline = 0.0
        self.recent = []

    def now(self):
        return self.ctx.clock.now()

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            # Logging must not turn a normal malformed observation into a game
            # error.  Inference and game calls are never hidden here.
            pass

    def allowed(self, action):
        verb = str(action or "").split(" ", 1)[0]
        return not self.names or verb in self.names

    def target_key(self, action):
        parts = str(action or "").split()
        if not parts:
            return ""
        if parts[0] in {"gather", "dig", "navigate", "goto", "open_container", "sleep"} and len(parts) > 1:
            return parts[0] + " " + parts[1]
        if parts[0] == "place" and len(parts) > 2:
            return "place " + parts[1] + " " + parts[2]
        return str(action)

    def is_blocked(self, key, now=None):
        now = self.now() if now is None else now
        return now < self.failed.get(str(key), -1.0)

    def mark_failed(self, action, now, reason="", duration=None):
        action = str(action or "")
        if not action or action in {"noop", "cancel"}:
            return
        verb = action.split()[0]
        if duration is None:
            duration = 420.0 if verb in {"dig", "gather"} else 180.0 if verb in {"navigate", "goto", "place"} else 45.0
        key = self.target_key(action)
        self.failed[key] = max(self.failed.get(key, -1.0), now + duration)
        parts = action.split()
        if len(parts) > 1 and (parts[1].startswith("b:") or parts[1].startswith("s:")):
            self.failed[parts[1]] = max(self.failed.get(parts[1], -1.0), now + duration)
        if verb == "place" and len(parts) > 2:
            # The reference block, not the carried item, is the failed target.
            self.bad_support.add(parts[2])
            self.failed[parts[2]] = max(self.failed.get(parts[2], -1.0), now + duration)
        self.log("target_quarantined", action=action, key=key, reason=str(reason)[:100], blocked_s=duration)
        if self.one and self.one.get("action") == action:
            self.one = None
        if self.work and self.work.get("target") in {key, parts[1] if len(parts) > 1 else ""}:
            self.log("resource_job_failed", target=self.work.get("target"), reason=str(reason)[:100])
            self.work = None

    def feedback_marker(self, obs):
        return json.dumps([obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")], sort_keys=True, default=str)

    def process_feedback(self, obs, now):
        marker = self.feedback_marker(obs)
        if marker == self.last_feedback:
            return
        self.last_feedback = marker
        if not self.one:
            return
        lower = marker.lower()
        if any(word in lower for word in FAIL_WORDS):
            action = self.one.get("action", "")
            self.mark_failed(action, now, "adapter_feedback")
            if self.work and action.startswith(("navigate ", "goto ", "dig ", "gather ")):
                resource_target = self.work.get("target")
                if (self.work.get("kind") == "mine" and
                        self.work.get("stage") in {"pickup", "collect", "pickup_look", "pickup_move"} and
                        action.startswith(("navigate ", "goto "))):
                    # A failed pickup route says nothing about whether the
                    # block was mined.  Keep the transaction and switch to a
                    # short, renewed manual pickup attempt.
                    self.work["stage"] = "pickup_look"
                    self.work["pickup_attempted"] = "manual"
                    self.work["pickup_started"] = now
                    self.log("pickup_route_failed", action=action, target=resource_target)
                elif resource_target and self.work.get("kind") == "mine":
                    self.mark_failed("dig " + str(resource_target), now, "resource_action_feedback", duration=420.0)
                    self.work = None
                else:
                    self.work = None

    def observe_rows(self, obs, prefix=None):
        here = self_pos(obs)
        result = []
        for raw in rows(obs.get("candidates")):
            ident = valid_id(raw.get("id"), prefix)
            if not ident:
                continue
            row = dict(raw)
            row["_id"] = ident
            row["_name"] = block_name(raw)
            row["_pos"] = point(raw.get("position"))
            row["_distance"] = num(raw.get("distance"), dist(here, row["_pos"]))
            result.append(row)
        result.sort(key=lambda x: x["_distance"])
        return result

    def entities(self, obs):
        here = self_pos(obs)
        result = []
        for raw in rows(obs.get("entities", obs.get("nearby_entities", []))):
            row = dict(raw)
            row["_name"] = entity_name(raw)
            row["_pos"] = point(raw.get("position"))
            row["_distance"] = num(raw.get("distance"), dist(here, row["_pos"]))
            result.append(row)
        return result

    def post(self, action, obs=None, reason=""):
        if self.ctx.done:
            return "noop"
        action = str(action or "noop").strip() or "noop"
        if not self.allowed(action):
            self.log("unsupported_compiled_action", action=action)
            action = "noop"
        verb = action.split()[0]
        # One-shot commands are latched.  A noop barrier is required before a
        # deliberate retry, while controls/attack/shield are intentionally
        # repeatable in the real-time adaptor.
        if action == self.last_posted and verb not in {"noop", "controls", "attack", "shield"}:
            action = "noop"
            verb = "noop"
        if verb not in {"noop", "controls", "attack", "shield", "cancel", "respawn"} and self.is_blocked(self.target_key(action), self.now()):
            self.log("blocked_retry_suppressed", action=action)
            action = "noop"
            verb = "noop"
        self.ctx.act(action)
        self.last_posted = action
        now = self.now()
        if verb == "cancel":
            self.one = None
            self.recent.append(action)
            self.recent = self.recent[-12:]
            self.log("posted", action=action, reason=reason, focus=self.focus)
            return action
        if verb == "respawn":
            self.one = None
            self.recent.append(action)
            self.recent = self.recent[-12:]
            self.log("posted", action=action, reason=reason, focus=self.focus)
            return action
        if verb not in {"noop", "controls", "attack", "shield"}:
            before = inv_sig(inv_of(obs or {}))
            self.one = {"action": action, "started": now, "before_inv": before,
                        "before_pos": self_pos(obs or {}), "dimension": self_row(obs or {}).get("dimension")}
            parts = action.split()
            if verb == "place" and len(parts) >= 3:
                self.one["item"] = item_name(parts[1])
                self.one["support"] = parts[2]
                self.one["face"] = parts[3] if len(parts) > 3 else "up"
            elif verb == "equip" and len(parts) >= 2:
                self.one["item"] = item_name(parts[1])
            elif verb == "craft" and len(parts) >= 2:
                self.one["item"] = item_name(parts[1])
            elif verb in {"dig", "gather"} and len(parts) >= 2:
                self.one["target"] = parts[1]
            elif verb in {"navigate", "goto"} and len(parts) >= 2:
                self.one["target"] = parts[1:]
            if verb == "furnace_start" and len(parts) >= 2:
                self.furnace_started[parts[1]] = now
                self.furnace_ready[parts[1]] = now + 38.0
            if verb == "place" and len(parts) >= 3:
                if item_name(parts[1]) == "crafting_table":
                    self.table_expected = now + 10.0
                elif item_name(parts[1]) == "furnace":
                    self.furnace_expected = now + 10.0
            self.recent.append(action)
            self.recent = self.recent[-12:]
            self.log("posted", action=action, reason=reason, focus=self.focus)
        elif verb == "attack":
            self.last_attack = now
        return action

    def row_by_id(self, obs, ident):
        return next((x for x in self.observe_rows(obs, "b") if x["_id"] == ident), None)

    def table_rows(self, obs):
        return [x for x in self.observe_rows(obs, "b") if x["_name"] == "crafting_table"]

    def nearby_table(self, obs):
        tables = [x for x in self.table_rows(obs) if x["_distance"] <= 4.8 and x.get("in_reach")]
        return tables[0] if tables else None

    def furnace_rows(self, obs):
        return [x for x in self.observe_rows(obs, "b") if x["_name"] == "furnace"]

    def nearby_support(self, obs):
        here = self_pos(obs)
        if here is None:
            return None
        floor_y = math.floor(here[1]) - 1
        choices = []
        for row in self.observe_rows(obs, "b"):
            if row["_name"] not in SUPPORTS or not row.get("in_reach") or row.get("visible") is False:
                continue
            q = row.get("_pos")
            if q is None or math.floor(q[1]) != floor_y:
                continue
            if math.floor(q[0]) == math.floor(here[0]) and math.floor(q[2]) == math.floor(here[2]):
                continue
            if row["_id"] in self.bad_support or self.is_blocked(row["_id"]):
                continue
            choices.append(row)
        choices.sort(key=lambda x: x["_distance"])
        return choices[0] if choices else None

    def valid_resource(self, row, names=None, suffixes=(), now=None):
        now = self.now() if now is None else now
        name = row.get("_name", "")
        if names is not None and name not in names and not any(name.endswith(s) for s in suffixes):
            return False
        if names is None and not any(name.endswith(s) for s in suffixes):
            return False
        if row.get("harvestable") is False or row.get("visible") is not True:
            return False
        if is_hazard(name):
            return False
        ident = row.get("_id", "")
        if ident in self.completed or self.is_blocked(ident, now) or self.is_blocked("dig " + ident, now):
            return False
        here = self_pos_from_row(row)
        me = self.last_position
        # Do not mine the support block directly below the player's feet.
        if me and here and tuple(math.floor(x) for x in here) == (math.floor(me[0]), math.floor(me[1]) - 1, math.floor(me[2])):
            return False
        return True

    def find_resource(self, obs, names=None, suffixes=(), now=None):
        now = self.now() if now is None else now
        choices = [x for x in self.observe_rows(obs, "b") if self.valid_resource(x, names, suffixes, now)]
        choices.sort(key=lambda x: (0 if x.get("in_reach") else 1, x["_distance"], x["_id"]))
        return choices[0] if choices else None

    def begin_mine(self, row, inv, tool, purpose):
        self.work = {"kind": "mine", "stage": "approach", "target": row["_id"],
                     "name": row["_name"], "pos": row.get("_pos"), "tool": tool,
                     "before_inv": inv_sig(inv), "before_count": inv.get(row["_name"], 0),
                     "started": self.now(), "purpose": purpose, "approach_count": 0,
                     "pickup_started": None, "pickup_attempted": False}
        self.log("mine_begin", target=row["_id"], block=row["_name"], purpose=purpose)

    def expected_drop(self, name):
        if name in {"stone", "deepslate", "cobblestone", "cobbled_deepslate"}:
            return {"cobblestone", "cobbled_deepslate", "stone", "deepslate"}
        if name in {"iron_ore", "deepslate_iron_ore"}:
            return {"raw_iron", "iron_ore", "deepslate_iron_ore"}
        if name in {"coal_ore", "deepslate_coal_ore"}:
            return {"coal", "coal_ore", "deepslate_coal_ore"}
        if name == "hay_block":
            return {"wheat", "hay_block"}
        if name == "sweet_berry_bush":
            return {"sweet_berries", "sweet_berry_bush"}
        return {name}

    def work_action(self, obs, now):
        job = self.work
        if not job:
            return None
        inv = inv_of(obs)
        stage = job.get("stage")
        target = job.get("target", "")
        row = self.row_by_id(obs, target)
        if stage == "approach":
            if row is None:
                self.work = None
                self.mark_failed("dig " + target, now, "target_disappeared")
                return None
            if row.get("visible") and row.get("in_reach"):
                job["stage"] = "equip"
                return None
            if job.get("approach_count", 0) >= 8 or now - job.get("started", now) > 22.0:
                self.work = None
                self.mark_failed("navigate " + target, now, "resource_approach_timeout")
                self.mark_failed("dig " + target, now, "resource_approach_timeout", duration=420.0)
                return None
            job["approach_count"] += 1
            return "navigate " + target
        if stage == "equip":
            tool = item_name(job.get("tool"))
            held = item_name(self_row(obs).get("held_item"))
            if tool:
                if not inv.get(tool):
                    self.work = None
                    self.mark_failed("dig " + target, now, "required_tool_missing")
                    return None
                if held != tool:
                    job["stage"] = "equip_wait"
                    return "equip " + tool + " hand"
            job["stage"] = "look"
            return None
        if stage == "equip_wait":
            tool = item_name(job.get("tool"))
            if item_name(self_row(obs).get("held_item")) == tool:
                job["stage"] = "look"
            return None
        if stage == "look":
            q = row.get("_pos") if row else job.get("pos")
            if not q:
                self.work = None
                return None
            job["stage"] = "dig"
            return "look_at %.2f %.2f %.2f" % q
        if stage == "dig":
            row = self.row_by_id(obs, target)
            if row is None:
                job["stage"] = "pickup"
                job["pickup_started"] = now
                return None
            if not (row.get("visible") and row.get("in_reach")):
                self.work = None
                self.mark_failed("dig " + target, now, "target_not_visible_and_in_reach")
                return None
            q = row.get("_pos") or job.get("pos")
            if not q:
                self.work = None
                return None
            job["stage"] = "pickup"
            job["pickup_started"] = now
            return "dig %d %d %d" % tuple(int(round(x)) for x in q)
        if stage == "pickup_look":
            q = job.get("pos")
            if q:
                job["stage"] = "pickup_move"
                return "look_at %.2f %.2f %.2f" % q
            job["stage"] = "pickup_move"
            return None
        if stage == "pickup_move":
            before = dict(job.get("before_inv", ()))
            if any(inv.get(x, 0) > before.get(x, 0) for x in self.expected_drop(job.get("name", ""))):
                self.completed.add(target)
                self.log("mine_confirmed", target=target, block=job.get("name"), method="manual_pickup")
                self.work = None
                return None
            if now - job.get("pickup_started", now) < 6.0:
                return "controls forward"
            self.work = None
            self.mark_failed("dig " + target, now, "drop_not_collected")
            return None
        if stage in {"pickup", "collect"}:
            before = dict(job.get("before_inv", ()))
            if any(inv.get(x, 0) > before.get(x, 0) for x in self.expected_drop(job.get("name", ""))):
                self.completed.add(target)
                self.log("mine_confirmed", target=target, block=job.get("name"), method="inventory")
                self.work = None
                return None
            if job.get("pickup_started") is None:
                job["pickup_started"] = now
            if now - job["pickup_started"] < 2.0:
                return None
            drops = []
            for ent in self.entities(obs):
                if ent["_name"] in {"item", "item_stack", "item_entity"} and ent.get("_pos") and ent["_distance"] <= 10:
                    if job.get("pos") is None or dist(ent["_pos"], job.get("pos")) <= 9:
                        drops.append(ent)
            drops.sort(key=lambda x: x["_distance"])
            q = drops[0].get("_pos") if drops else job.get("pos")
            if q and not job.get("pickup_attempted") and now - job.get("pickup_started", now) < 14:
                # Prefer a listed safe stand around the drop.  Arbitrary goto
                # feet coordinates often land inside a block or water; a stand
                # candidate is the adaptor's intended pickup primitive.
                stands = [s for s in self.observe_rows(obs, "s")
                          if s.get("_pos") and dist(s.get("_pos"), q) <= 4.8
                          and not self.is_blocked("navigate " + s["_id"], now)]
                stands.sort(key=lambda s: dist(s.get("_pos"), q))
                job["pickup_attempted"] = "stand" if stands else "goto"
                job["stage"] = "collect"
                if stands:
                    return "navigate " + stands[0]["_id"]
                return "goto %.2f %.2f %.2f" % (q[0], q[1] - 1.0, q[2])
            if q and job.get("pickup_attempted") == "stand" and now - job.get("pickup_started", now) > 8.0:
                job["pickup_attempted"] = "goto"
                return "goto %.2f %.2f %.2f" % (q[0], q[1] - 1.0, q[2])
            if now - job.get("pickup_started", now) > 18:
                job["stage"] = "pickup_look"
                job["pickup_started"] = now
            return None
        self.work = None
        return None

    def begin_hunt(self, ent, inv):
        self.work = {"kind": "hunt", "entity": str(ent.get("id", "")), "name": ent.get("_name", ""),
                     "pos": ent.get("_pos"), "before_inv": inv_sig(inv), "started": self.now(),
                     "approach_attempted": False, "last_seen": self.now()}
        self.log("hunt_begin", entity=str(ent.get("id", "")), kind=ent.get("_name", ""))

    def hunt_action(self, obs, now):
        job = self.work
        inv = inv_of(obs)
        eid = str(job.get("entity", ""))
        found = next((e for e in self.entities(obs) if str(e.get("id", "")) == eid), None)
        if found is None or found.get("dead"):
            if inv_sig(inv) != tuple(sorted(job.get("before_inv", ()))) or now - job.get("started", now) > 4:
                self.log("hunt_finished", entity=eid)
                self.work = None
                return None
            if now - job.get("started", now) > 15:
                self.work = None
            return None
        job["pos"] = found.get("_pos") or job.get("pos")
        job["last_seen"] = now
        weapon = first_item(inv, WEAPONS)
        if not weapon:
            self.work = None
            return None
        held = item_name(self_row(obs).get("held_item"))
        if held != weapon:
            return "equip " + weapon + " hand"
        if num(found.get("distance"), 99) <= 3.3:
            if now - self.last_attack >= 0.75:
                self.last_attack = now
                return "attack " + eid
            return None
        if found.get("_pos") and not job.get("approach_attempted"):
            job["approach_attempted"] = True
            return "goto %.2f %.2f %.2f" % tuple(found["_pos"])
        if now - job.get("started", now) > 24:
            self.work = None
        return None

    def update_one(self, obs, now):
        if not self.one:
            return
        action = str(self.one.get("action", ""))
        verb = action.split()[0] if action else ""
        inv = inv_of(obs)
        before = dict(self.one.get("before_inv", ()))
        changed = inv_sig(inv) != tuple(sorted(before.items()))
        me = self_row(obs)
        if verb == "equip":
            if item_name(me.get("held_item")) == item_name(self.one.get("item")):
                self.one = None
                return
        elif verb == "craft":
            item = item_name(self.one.get("item"))
            if changed or inv.get(item, 0) > before.get(item, 0):
                self.one = None
                return
        elif verb in {"eat", "furnace_take", "trade", "enchant"}:
            if changed or verb == "eat" and num(me.get("food"), 20) > 18:
                if verb == "furnace_take":
                    parts = action.split()
                    if len(parts) > 1:
                        self.furnace_ready.pop(parts[1], None)
                        self.furnace_started.pop(parts[1], None)
                self.one = None
                return
        elif verb == "place":
            item = item_name(self.one.get("item"))
            support = self.one.get("support")
            support_row = self.row_by_id(obs, support) if support else None
            support_pos = support_row.get("_pos") if support_row else None
            expected_pos = None
            if support_pos:
                expected_pos = (support_pos[0], support_pos[1] + 1, support_pos[2])
            for row in self.observe_rows(obs, "b"):
                if row["_name"] == item and (expected_pos is None or dist(row.get("_pos"), expected_pos) <= 1.1):
                    self.one = None
                    if item == "crafting_table":
                        self.table_expected = None
                    if item == "furnace":
                        self.furnace_expected = None
                    return
            if changed and before.get(item, 0) > inv.get(item, 0):
                self.one = None
                return
        elif verb in {"navigate", "goto"}:
            old = self.one.get("before_pos")
            here = self_pos(obs)
            if old and here and dist(old, here) >= 0.8:
                self.one = None
                return
            target = self.one.get("target", [])
            if verb == "navigate" and target:
                row = self.row_by_id(obs, target[0])
                if row and (row.get("in_reach") or row.get("_distance", 99) <= 2.2):
                    self.one = None
                    return
        elif verb in {"dig", "gather"}:
            target = self.one.get("target")
            if changed:
                self.one = None
                return
            if target and self.row_by_id(obs, target) is None:
                self.one = None
                return
        elif verb == "look_at":
            if now - self.one.get("started", now) >= 0.7:
                self.one = None
                return
        elif verb == "furnace_start":
            if now - self.one.get("started", now) >= 1.0 and not has_task(obs):
                self.one = None
                return
        elif verb == "sleep":
            if me.get("bed_position") or now - self.one.get("started", now) > 2:
                self.one = None
                return
        age = now - self.one.get("started", now)
        limits = {"gather": 31, "dig": 18, "navigate": 18, "goto": 18,
                  "place": 9, "craft": 8, "equip": 5, "look_at": 3,
                  "furnace_start": 7, "furnace_take": 8, "sleep": 8,
                  "activate_block": 8, "use_item": 8}.get(verb, 8)
        if age > limits and not has_task(obs):
            self.mark_failed(action, now, "postcondition_timeout")

    def update_progress(self, obs, now):
        inv = inv_of(obs)
        adv = tuple(sorted(str(x) for x in obs.get("advancements", []) if isinstance(x, str)))
        pos = self_pos(obs)
        if inv_sig(inv) != self.last_inv or adv != self.last_adv:
            self.last_progress = now
            self.last_inv = inv_sig(inv)
            self.last_adv = adv
        self.last_position = pos

    def threats(self, obs):
        me = self_row(obs)
        result = []
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            result.append({"kind": "lava_or_fire", "id": "self", "range": "touching", "distance": 0.0})
        if me.get("is_in_water") and num(me.get("oxygen"), 20) < 6:
            result.append({"kind": "drowning", "id": "self", "range": "touching", "distance": 0.0})
        if num(me.get("fall_distance_estimate"), 0) > 3:
            result.append({"kind": "fall", "id": "self", "range": "touching", "distance": 0.0})
        for row in self.observe_rows(obs, "b"):
            if row["_name"] in HAZARDS and row["_distance"] <= 4.5:
                result.append({"kind": row["_name"], "id": row["_id"], "range": "near", "distance": row["_distance"]})
        for ent in self.entities(obs):
            kind = ent["_name"]
            if (ent.get("hostile") or ent.get("conditional_threat") or kind in HOSTILES) and ent.get("visible") and ent["_distance"] <= 8:
                result.append({"kind": kind, "id": str(ent.get("id", "")),
                               "range": "touching" if ent["_distance"] <= 3.3 else "near",
                               "distance": ent["_distance"]})
        return result

    def threat_fallback(self, obs, threats):
        me = self_row(obs)
        inv = inv_of(obs)
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            return "cancel" if (self.one or has_task(obs)) else "controls back,jump"
        if me.get("is_in_water") and num(me.get("oxygen"), 20) < 6:
            return "cancel" if (self.one or has_task(obs)) else "controls jump"
        # The adaptor has one hand/navigation owner.  Cancel before replacing
        # a mining, navigation, or use job with combat equipment or movement.
        if self.one or has_task(obs):
            return "cancel"
        close = [x for x in threats if x.get("id") != "self" and x.get("range") == "touching"]
        if close:
            kind = close[0].get("kind", "")
            weapon = first_item(inv, WEAPONS)
            if weapon and kind not in {"creeper", "skeleton", "stray", "ghast", "blaze"}:
                if item_name(me.get("held_item")) != weapon:
                    return "equip " + weapon + " hand"
                if self.now() - self.last_attack >= 0.75:
                    self.last_attack = self.now()
                    return "attack " + str(close[0].get("id", ""))
            if inv.get("shield"):
                return "shield on"
            return "controls back,jump"
        if inv.get("shield"):
            return "shield on"
        return "controls back,jump"

    def threat_action(self, obs, now):
        threats = self.threats(obs)
        if not threats:
            self.threat_cached = None
            self.threat_sig = ()
            return None
        fallback = self.threat_fallback(obs, threats)
        options = []
        if fallback:
            options.append(fallback)
        inv = inv_of(obs)
        weapon = first_item(inv, WEAPONS)
        for t in threats:
            if t.get("id") == "self":
                continue
            if t.get("range") == "touching" and weapon:
                options.append("equip " + weapon + " hand")
                options.append("attack " + str(t.get("id", "")))
        if inv.get("shield"):
            options.append("shield on")
        if self.one or has_task(obs):
            options.append("cancel")
        options.append("noop")
        legal = []
        for action in options:
            if action and action not in legal and self.allowed(action):
                legal.append(action)
        if not legal:
            return fallback
        sig = tuple(sorted((str(x.get("kind")), str(x.get("id")), str(x.get("range"))) for x in threats))
        if self.threat_cached and self.threat_cached[1] == sig and now - self.threat_cached[2] < 2.0:
            return self.threat_cached[0] if self.threat_cached[0] in legal else fallback
        if self.threat_future is None and self.ctx.budget.remaining().get("system1_calls", 0) > 2:
            state = {
                "purpose": "Immediate Minecraft survival interrupt; code owns legality and coordinates.",
                "self": {"health_band": "critical" if num(self_row(obs).get("health"), 20) < 7 else "stable",
                         "dimension": str(self_row(obs).get("dimension", "")),
                         "lava_fire": bool(self_row(obs).get("is_in_lava") or self_row(obs).get("is_in_fire"))},
                "threats": [{"kind": x["kind"], "range": x["range"]} for x in threats[:6]],
                "legal_actions": legal,
                "rule": "Choose only one exact legal_actions value. Survival and escape beat work; never invent an id.",
            }
            criteria = {x: "Exact permitted response: " + x for x in legal}
            questions = {
                "response": Choice("Choose the safest supplied response to the immediate threats.", criteria),
                "continue_safe": Noul("Is it safe to continue the current work without first responding to a threat?",
                                      {"true": "No immediate touching or environmental hazard is present.",
                                       "false": "A visible or self hazard requires defense or escape."}),
            }
            try:
                self.threat_future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=1.5))
                self.threat_sig = sig
                self.threat_at = now
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                self.threat_future = None
        if self.threat_future is not None and self.threat_future.done():
            future = self.threat_future
            self.threat_future = None
            try:
                answers = future.result()
                answer = answers.get("response") if isinstance(answers, dict) else None
                value = getattr(answer, "value", None)
                probs = getattr(answer, "probabilities", {}) or {}
                probability = num(probs.get(value), 0.0) if isinstance(probs, dict) else 0.0
                if value in legal and self.threat_sig == sig and probability >= 0.55:
                    self.threat_cached = (value, sig, now)
                    self.log("jev_threat_choice", action=value, probability=round(probability, 3))
                    return value
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError, KeyError):
                self.log("jev_threat_fallback")
        return fallback

    def parse_focus(self, text):
        text = str(text or "").lower()
        values = ("food", "bed", "stone", "iron", "explore", "balanced")
        try:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                value = json.loads(text[start:end + 1]).get("focus")
                if value in values:
                    return value
        except (TypeError, ValueError, json.JSONDecodeError, AttributeError):
            pass
        for value in values:
            if re.search(r"\b" + value + r"\b", text):
                return value
        return None

    def astra(self, obs, phase, options, now):
        if self.astra_future is not None and self.astra_future.done():
            future = self.astra_future
            self.astra_future = None
            try:
                result = future.result()
                value = self.parse_focus(getattr(result, "text", ""))
                if value:
                    self.focus = value
                    self.focus_deadline = self.now() + 60.0
                    self.astra_version += 1
                    self.astra_session = getattr(result, "session_id", self.astra_session)
                    self.log("astra_focus", focus=value, version=self.astra_version)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError):
                self.log("astra_invalid")
        remaining = self.ctx.budget.remaining()
        stalled = now - self.last_progress > 35.0
        if (self.astra_future is None and remaining.get("system2_calls", 0) > 1 and
                remaining.get("wall_s", 0) > 180 and now - self.astra_last > 180 and stalled):
            inv = inv_of(obs)
            visible = [(x["_name"], round(x["_distance"], 1), bool(x.get("in_reach")))
                       for x in self.observe_rows(obs, "b") if x.get("visible")][:16]
            prompt = (
                "You are a Minecraft survival macro adviser. Code owns all recipes, coordinates, "
                "legal commands and safety. Choose one focus from food, bed, stone, iron, explore, "
                "balanced; do not emit commands or coordinates. Prefer a verified next milestone "
                "from the supplied inventory and visible opportunities. Reply only JSON "
                "{\"focus\":\"...\"}. " + json.dumps({"phase": phase, "focus": self.focus,
                "inventory": inv, "visible": visible, "recent_failures": list(self.failed)[-8:],
                "recent_actions": self.recent[-8:], "remaining_s": remaining.get("wall_s")}, default=str))
            try:
                session = self.astra_session
                self.astra_future = self.ctx.spawn(lambda p=prompt, s=session: self.ctx.think(
                    p, session_id=s, timeout_s=120.0))
                self.astra_last = now
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                self.astra_future = None

    def start_manual(self, now, reason=""):
        if self.manual is None:
            self.manual = {"dir": self.manual_index % 8, "stage": "look", "started": now,
                           "move_started": now, "reason": reason}
            self.manual_index += 1
            self.log("survey_begin", direction=self.manual["dir"], reason=reason)

    def manual_action(self, obs, now):
        if self.manual is None:
            return None
        # A survey ends only for resources relevant to the current dependency
        # phase.  An already-known tree is not a reason to stop a stone/iron
        # survey; this prevents a benign wood candidate from producing a
        # start/stop loop after the opening.
        inv = inv_of(obs)
        wanted = None
        if wood_total(inv) == 0 and not inv.get("crafting_table"):
            wanted = self.find_resource(obs, suffixes=WOOD_SUFFIXES, now=now)
        elif not inv.get("wooden_pickaxe"):
            wanted = self.find_resource(obs, suffixes=WOOD_SUFFIXES, now=now)
        elif not first_item(inv, STONE_PICKAXES):
            wanted = self.find_resource(obs, names={"stone", "cobblestone", "deepslate", "cobbled_deepslate"}, now=now)
        elif not inv.get("iron_ingot") and inv.get("raw_iron", 0) < 3:
            wanted = self.find_resource(obs, names={"iron_ore", "deepslate_iron_ore", "coal_ore", "deepslate_coal_ore", "hay_block", "sweet_berry_bush", "melon", "wheat"}, now=now)
        elif food_total(inv) < 3:
            wanted = self.find_resource(obs, names={"hay_block", "sweet_berry_bush", "melon", "wheat", "carrots", "potatoes", "beetroots"}, now=now)
        if wanted:
            self.manual = None
            self.manual_last_end = now
            return "controls none"
        if food_total(inv) < 3:
            weapon = first_item(inv, WEAPONS)
            if weapon and any(e["_name"] in ANIMALS and e.get("visible") for e in self.entities(obs)):
                self.manual = None
                self.manual_last_end = now
                return "controls none"
        stage = self.manual.get("stage")
        if stage == "look":
            p = self_pos(obs) or (0.0, 64.0, 0.0)
            dirs = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
            dx, dz = dirs[self.manual.get("dir", 0) % len(dirs)]
            self.manual["stage"] = "move"
            self.manual["move_started"] = now
            return "look_at %.2f %.2f %.2f" % (p[0] + dx * 2.5, p[1] + 0.5, p[2] + dz * 2.5)
        if stage == "move":
            if now - self.manual.get("move_started", now) >= 10.0:
                self.manual = None
                self.manual_last_end = now
                return "controls none"
            return "controls forward,jump"
        self.manual = None
        return "controls none"

    def portal_action(self, obs, now):
        if self.portal is None:
            return None
        row = next((x for x in self.observe_rows(obs, "b") if x["_id"] == self.portal.get("id")), None)
        if row is None:
            self.portal = None
            return None
        stage = self.portal.get("stage")
        if stage == "approach":
            if row.get("in_reach") or row["_distance"] <= 3:
                self.portal["stage"] = "look"
                return None
            self.portal["tries"] = self.portal.get("tries", 0) + 1
            if self.portal["tries"] > 2:
                self.portal = None
                return None
            return "navigate " + row["_id"]
        if stage == "look":
            self.portal["stage"] = "move"
            self.portal["move_started"] = now
            return "look_at %.2f %.2f %.2f" % row["_pos"] if row.get("_pos") else None
        if stage == "move":
            if self_row(obs).get("dimension") != self.portal.get("dimension"):
                self.log("portal_entered")
                self.portal = None
                return "controls none"
            if now - self.portal.get("move_started", now) > 7:
                self.portal = None
                return "controls none"
            return "controls forward"
        self.portal = None
        return None

    def choose_resource_tool(self, block):
        name = block.get("_name", "") if block else ""
        if name in {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}:
            return first_item(inv_of(self._obs_for_tool), STONE_PICKAXES) if hasattr(self, "_obs_for_tool") else "wooden_pickaxe"
        if name in {"iron_ore", "deepslate_iron_ore", "coal_ore", "deepslate_coal_ore", "gold_ore", "deepslate_gold_ore"}:
            return "stone_pickaxe"
        return ""

    def craft_action(self, obs, inv, table, now):
        """Return a prerequisite-first, real recipe command or None."""
        planks = plank_total(inv)
        logs = next((k for k, v in inv.items() if v and k.endswith(WOOD_SUFFIXES)), "")
        sticks = inv.get("stick", 0)
        if planks < 4 and logs:
            pn = logs.replace("_log", "_planks").replace("_stem", "_planks")
            return "craft %s %d" % (pn, 4 if inv.get(logs, 0) < 2 else 8)
        if not inv.get("crafting_table") and not self.table_rows(obs) and planks >= 4:
            return "craft crafting_table 1"
        # A wooden pick is the first hard gate.  It is worth replenishing wood
        # before consuming the only log in the table recipe.
        if not inv.get("wooden_pickaxe"):
            if planks < 5 and logs:
                pn = logs.replace("_log", "_planks").replace("_stem", "_planks")
                return "craft %s 8" % pn
            if sticks < 2 and planks >= 2:
                return "craft stick 4"
            if table and table["_distance"] <= 4.8:
                return "craft wooden_pickaxe 1 " + table["_id"]
        # Make a cheap weapon for food/wool as soon as two cobblestone are
        # available; otherwise a wooden sword is still a legitimate route.
        if table and table["_distance"] <= 4.8 and sticks >= 1:
            cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
            if cobble >= 2 and not first_item(inv, ("stone_sword", "iron_sword", "diamond_sword")):
                return "craft stone_sword 1 " + table["_id"]
            if planks >= 2 and not first_item(inv, ("wooden_sword", "wooden_axe", "stone_sword", "stone_axe", "iron_sword")):
                return "craft wooden_sword 1 " + table["_id"]
        if inv.get("wooden_pickaxe") and not inv.get("stone_pickaxe") and not inv.get("iron_pickaxe"):
            cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
            if cobble >= 3 and sticks >= 2 and table and table["_distance"] <= 4.8:
                return "craft stone_pickaxe 1 " + table["_id"]
        if inv.get("iron_ingot") and not inv.get("shield") and planks >= 6 and table and table["_distance"] <= 4.8:
            return "craft shield 1 " + table["_id"]
        return None

    def strategic_action(self, obs, now):
        inv = inv_of(obs)
        me = self_row(obs)
        dim = str(me.get("dimension", "overworld"))
        table = self.nearby_table(obs)
        tables = self.table_rows(obs)
        # Hunger is checked before every expedition and not delegated to Jev.
        if num(me.get("food"), 20) <= 14:
            food = first_item(inv, FOODS)
            if food:
                if item_name(me.get("held_item")) != food:
                    return "equip " + food + " hand", "eat"
                return "eat " + food, "eat"
        # Direct jobs are transactional.  No new strategic command can replace
        # one until its postcondition or failure has been recorded.
        if self.work:
            action = self.hunt_action(obs, now) if self.work.get("kind") == "hunt" else self.work_action(obs, now)
            return action, "transaction" if action else "transaction_wait"
        # A natural active portal is a safe optional shortcut only after a
        # minimal return kit.  It is never invented from coordinates.
        if dim == "overworld" and inv.get("shield") and food_total(inv) >= 3 and self.portal is None:
            portals = [x for x in self.observe_rows(obs, "b") if x["_name"] == "nether_portal" and x.get("visible")]
            if portals:
                portals.sort(key=lambda x: x["_distance"])
                self.portal = {"id": portals[0]["_id"], "dimension": dim, "stage": "approach", "tries": 0}
        if self.portal:
            action = self.portal_action(obs, now)
            return action, "portal" if action else "portal_wait"
        # Opening: acquire an exposed log, then actually craft the table.
        if wood_total(inv) == 0 and not inv.get("crafting_table") and not tables:
            row = self.find_resource(obs, suffixes=WOOD_SUFFIXES, now=now)
            if row:
                self.begin_mine(row, inv, "", "opening wood")
                return self.work_action(obs, now), "opening_wood"
            food_row = self.find_resource(obs, names={"hay_block", "sweet_berry_bush", "melon", "wheat"}, now=now)
            if food_row:
                self.begin_mine(food_row, inv, "", "opening food")
                return self.work_action(obs, now), "opening_food"
            return self.manual_action(obs, now) if self.manual else (self.start_manual(now, "no exposed opening resource") or None), "survey_opening"
        # Crafting and table placement are staged and have an explicit support
        # failure ledger; this prevents 0008's 34-repeat placement loop.
        action = self.craft_action(obs, inv, table, now)
        if action:
            return action, "opening_recipes"
        if not table:
            if tables:
                tables.sort(key=lambda x: x["_distance"])
                return "navigate " + tables[0]["_id"], "approach_table"
            if inv.get("crafting_table"):
                support = self.nearby_support(obs)
                if support:
                    held = item_name(me.get("held_item"))
                    if held != "crafting_table":
                        return "equip crafting_table hand", "equip_table"
                    return "place crafting_table %s up" % support["_id"], "place_table"
            # A table may be expected just outside the current candidate window.
            if self.table_expected and now < self.table_expected:
                return None, "table_wait"
        # Obtain a bed/food from present observations, but do not stall the
        # stone dependency forever when a biome has neither local opportunity.
        if food_total(inv) < 3:
            row = self.find_resource(obs, names={"hay_block", "sweet_berry_bush", "melon", "wheat", "carrots", "potatoes", "beetroots"}, now=now)
            if row:
                self.begin_mine(row, inv, "", "visible food")
                return self.work_action(obs, now), "food"
        # Hunt only visible passive entities and only with a real carried weapon.
        animal = [e for e in self.entities(obs) if e["_name"] in ANIMALS and e.get("visible") and e.get("id") is not None and not e.get("dead")]
        animal.sort(key=lambda x: x["_distance"])
        weapon = first_item(inv, WEAPONS)
        if animal and weapon and (food_total(inv) < 3 or not any(k.endswith("_wool") for k in inv)):
            self.begin_hunt(animal[0], inv)
            return self.hunt_action(obs, now), "hunt"
        # Astra may request one bounded detour when the local menu has no
        # immediately visible food/bed opportunity.  It never overrides the
        # prerequisite chain and expires automatically, so a bad macro answer
        # cannot recreate a 900-second exploration stall.
        if self.focus in {"food", "bed", "explore"} and now < self.focus_deadline:
            if self.manual is None:
                self.start_manual(now, "bounded_astra_focus_" + self.focus)
            return self.manual_action(obs, now), "astra_focus_survey"
        # Visible stone is always more valuable than another blind stand.  A
        # wooden pick is held for stone and a stone pick for ore.
        stone = self.find_resource(obs, names={"stone", "cobblestone", "deepslate", "cobbled_deepslate"}, now=now)
        have_stone_pick = bool(first_item(inv, STONE_PICKAXES))
        if inv.get("wooden_pickaxe") and not have_stone_pick and stone:
            self.begin_mine(stone, inv, "wooden_pickaxe", "stone for first stone pick")
            return self.work_action(obs, now), "stone"
        # Build the first stone pick as soon as its verified material exists.
        action = self.craft_action(obs, inv, table, now)
        if action:
            return action, "tools"
        if inv.get("wooden_pickaxe") and not have_stone_pick:
            if stone:
                self.begin_mine(stone, inv, "wooden_pickaxe", "stone for first stone pick")
                return self.work_action(obs, now), "stone"
        # Food/wool can be gathered after tools even if no animal happened to be
        # in the first view.  A visible sheep is handled above.
        wool_count = sum(v for k, v in inv.items() if k.endswith("_wool"))
        if wool_count >= 3 and not any(k.endswith("_bed") and v for k, v in inv.items()):
            if table:
                color = next(k[:-5] for k, v in inv.items() if k.endswith("_wool") and v >= 3)
                return "craft %s_bed 1 %s" % (color, table["_id"]), "bed_craft"
        # Gather exposed iron/coal only with the proper observed pickaxe.
        if have_stone_pick:
            raw = inv.get("raw_iron", 0) + inv.get("iron_ore", 0)
            if raw < 3 and not inv.get("iron_ingot"):
                iron = self.find_resource(obs, names={"iron_ore", "deepslate_iron_ore"}, now=now)
                if iron:
                    self.begin_mine(iron, inv, first_item(inv, STONE_PICKAXES), "exposed iron")
                    return self.work_action(obs, now), "iron"
            if not inv.get("coal") and not inv.get("charcoal"):
                coal = self.find_resource(obs, names={"coal_ore", "deepslate_coal_ore"}, now=now)
                if coal:
                    self.begin_mine(coal, inv, first_item(inv, STONE_PICKAXES), "furnace fuel")
                    return self.work_action(obs, now), "coal"
        # Furnace construction and real-time smelting.
        furnaces = self.furnace_rows(obs)
        if inv.get("raw_iron") and not furnaces:
            if inv.get("furnace") and self.furnace_expected is None:
                support = self.nearby_support(obs)
                if support:
                    if item_name(me.get("held_item")) != "furnace":
                        return "equip furnace hand", "equip_furnace"
                    return "place furnace %s up" % support["_id"], "place_furnace"
            if inv.get("furnace") and self.furnace_expected and now < self.furnace_expected:
                return None, "furnace_wait"
            if not inv.get("furnace") and table and table["_distance"] <= 4.8 and inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0) >= 8:
                return "craft furnace 1 " + table["_id"], "furnace_craft"
        if furnaces:
            furnace = furnaces[0]
            fid = furnace["_id"]
            if fid in self.furnace_ready and now >= self.furnace_ready[fid]:
                return "furnace_take " + fid, "furnace_take"
            if inv.get("raw_iron") and fid not in self.furnace_started:
                fuel = first_item(inv, ("coal", "charcoal")) or next((k for k in inv if k.endswith("_planks") and inv[k]), "")
                if fuel:
                    return "furnace_start %s raw_iron %d %s %d" % (fid, min(3, inv["raw_iron"]), fuel, min(inv[fuel], 8)), "furnace_start"
        # Craft a bed once wool and a workstation exist, and use a known night
        # bed when available.  The score flag is permanent, so this is optional
        # after the recipe has succeeded.
        beds = [x for x in self.observe_rows(obs, "b") if x["_name"].endswith("_bed") and x.get("visible")]
        if beds and num(obs.get("world", {}).get("time_of_day"), 0) >= 12500:
            beds.sort(key=lambda x: x["_distance"])
            if beds[0]["_distance"] > 4.5:
                return "navigate " + beds[0]["_id"], "approach_bed"
            return "sleep " + beds[0]["_id"], "sleep"
        # If no immediate resource is exposed, use the active survey.  Unlike a
        # stand-only frontier this renews controls every 0.5 s, so the 0.8 s
        # lease cannot silently leave the bot at the same tree.
        if self.manual is None:
            self.start_manual(now, "no current milestone resource")
        return self.manual_action(obs, now), "survey"

    def run_episode(self, game):
        self.names = set(getattr(getattr(game, "action_spec", None), "names", ()) or ())
        while not self.ctx.done:
            now = self.now()
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                self.post("noop", {}, "missing_observation")
                self.ctx.wait_for_tick()
                continue
            self.update_progress(obs, now)
            self.process_feedback(obs, now)
            self.update_one(obs, now)
            me = self_row(obs)
            if me.get("dead"):
                self.work = None
                self.manual = None
                self.portal = None
                self.one = None
                self.post("respawn", obs, "normal_respawn")
                self.ctx.wait_for_tick()
                continue
            urgent = self.threat_action(obs, now)
            if urgent:
                # A safety command owns this tick.  It is allowed to cancel an
                # active bounded job; the job's observed target remains in the
                # failure/progress ledger and can be resumed or replaced.
                self.post(urgent, obs, "safety_override")
                self.ctx.wait_for_tick()
                continue
            if num(obs.get("observation_age_s"), 0) > 3.0:
                self.post("cancel" if self.one or has_task(obs) else "noop", obs, "stale_observation")
                self.ctx.wait_for_tick()
                continue
            # Renew a movement survey even while no inference is available.
            if self.manual is not None:
                if self.one:
                    self.post("noop", obs, "finish_look")
                else:
                    action = self.manual_action(obs, now) or "noop"
                    self.post(action, obs, "active_survey")
                self.ctx.wait_for_tick()
                continue
            if self.portal is not None:
                if self.one:
                    self.post("noop", obs, "finish_portal_step")
                else:
                    action = self.portal_action(obs, now) or "noop"
                    self.post(action, obs, "portal_transaction")
                self.ctx.wait_for_tick()
                continue
            if self.one:
                self.post("noop", obs, "bounded_job")
                self.ctx.wait_for_tick()
                continue
            # Ask Astra only when a verified inventory/advancement change has
            # not occurred for a while.  It can reorder safe branches but cannot
            # author an action or interrupt a survival veto.
            inv = inv_of(obs)
            phase = "progression"
            if not wood_total(inv):
                phase = "wood"
            elif not inv.get("wooden_pickaxe"):
                phase = "table_and_tools"
            elif not inv.get("stone_pickaxe"):
                phase = "stone"
            elif not inv.get("iron_ingot"):
                phase = "iron"
            else:
                phase = "consolidate"
            self.astra(obs, phase, [], now)
            action, purpose = self.strategic_action(obs, now)
            # Focus is a macro hint: only alter the order where two viable
            # local choices exist; prerequisites above always win.  Keeping
            # this as an advisory event avoids trusting arbitrary Astra prose.
            if action is None:
                if self.manual is None and not self.work:
                    self.start_manual(now, "strategic_wait")
                action = self.manual_action(obs, now) if self.manual else "noop"
            self.post(action or "noop", obs, purpose)
            self.ctx.wait_for_tick()


def self_pos_from_row(row):
    return row.get("_pos") if isinstance(row, dict) else None


def build(ctx):
    return Harness(ctx)
