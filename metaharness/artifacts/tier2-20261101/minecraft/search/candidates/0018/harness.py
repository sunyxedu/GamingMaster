"""Candidate 0018: a two-level expedition controller for Minecraft.

This controller deliberately does not treat a block-candidate list as a route.
It has a small, transactional milestone compiler for work that is already
observable, and a separate directional expedition state machine for changing
the observation window.  Jev chooses between the currently legal transactions;
Astra chooses only a coarse expedition mode and bearing.  Neither model is
allowed to manufacture coordinates, recipes, or action arguments.
"""
from __future__ import annotations

import json
import math
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


ID_RE = re.compile(r"^[bs]:-?\d+,-?\d+,-?\d+$")
ITEM_RE = re.compile(r"^[a-z0-9_]+$")
FAIL_WORDS = (
    "failed", "failure", "error", "unreachable", "timeout", "cannot",
    "rejected", "illegal", "not enough", "not in reach", "no path",
    "busy", "invalid", "unknown", "out of range",
)

WOOD_SUFFIXES = ("_log", "_stem")
PICKAXES = ("netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
STONE_PICKAXES = ("netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe")
WEAPONS = (
    "netherite_sword", "diamond_sword", "iron_sword", "stone_sword",
    "wooden_sword", "iron_axe", "stone_axe", "wooden_axe",
)
FOODS = (
    "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
    "cooked_rabbit", "bread", "baked_potato", "carrot", "apple",
    "sweet_berries", "melon_slice", "mushroom_stew", "beetroot",
    "potato", "beef", "porkchop", "mutton", "chicken", "rabbit",
    "cod", "salmon", "wheat",
)
FOOD_BLOCKS = {"hay_block", "melon", "sweet_berry_bush", "wheat", "carrots", "potatoes", "beetroots"}
STONE_BLOCKS = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON_BLOCKS = {"iron_ore", "deepslate_iron_ore"}
FUEL_BLOCKS = {"coal_ore", "deepslate_coal_ore"}
EXCAVATABLE = {
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "sand", "red_sand", "gravel", "clay", "snow_block", "snow", "mud",
}
SUPPORTS = EXCAVATABLE | STONE_BLOCKS | {
    "terracotta", "sandstone", "red_sandstone", "netherrack", "blackstone",
    "basalt", "oak_log", "birch_log", "spruce_log", "jungle_log",
    "acacia_log", "dark_oak_log", "mangrove_log", "crafting_table", "furnace",
    "chest", "end_stone",
}
HAZARDS = {
    "lava", "fire", "soul_fire", "magma_block", "cactus", "powder_snow",
    "campfire", "soul_campfire", "pointed_dripstone", "bubble_column",
}
HOSTILES = {
    "zombie", "skeleton", "creeper", "spider", "drowned", "husk", "stray",
    "witch", "phantom", "enderman", "slime", "magma_cube", "blaze", "ghast",
    "piglin", "piglin_brute", "hoglin", "wither_skeleton", "zombified_piglin",
    "ender_dragon", "shulker", "silverfish",
}


# The observation stream is deliberately treated as untrusted.  In particular,
# these helpers make model stand-ins and a lagging adaptor harmless.
def number(value, default=0.0):
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


def dict_rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def point(value):
    if isinstance(value, dict):
        if isinstance(value.get("position"), (list, tuple, dict)):
            value = value.get("position")
        elif all(key in value for key in ("x", "y", "z")):
            value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result = tuple(number(value[i], math.inf) for i in range(3))
    return result if all(math.isfinite(x) for x in result) else None


def id_point(ident):
    if not isinstance(ident, str) or not ID_RE.fullmatch(ident):
        return None
    try:
        result = tuple(float(x) for x in ident[2:].split(","))
        return result if len(result) == 3 and all(math.isfinite(x) for x in result) else None
    except (TypeError, ValueError):
        return None


def distance(a, b):
    pa, pb = point(a), point(b)
    if pa is None or pb is None:
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def self_row(obs):
    return obs.get("self") if isinstance(obs.get("self"), dict) else {}


def self_position(obs):
    return point(self_row(obs).get("position"))


def has_task(obs):
    value = obs.get("task")
    return value not in (None, "", {})


def inventory(obs):
    """Return only the player's inventory, never a window or cursor preview."""
    source = obs.get("inventory")
    if source in (None, {}, []):
        source = obs.get("inventory_authoritative", source)
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source.get("items")
    result = {}
    if isinstance(source, dict):
        for key, raw in source.items():
            name = item_name(key)
            count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(number(count)))
            if name and count:
                result[name] = result.get(name, 0) + count
    elif isinstance(source, list):
        for raw in dict_rows(source):
            name = item_name(raw)
            count = max(0, int(number(raw.get("count", 0))))
            if name and count:
                result[name] = result.get(name, 0) + count
    # The adaptor labels this as personal inventory.  It is safe as a temporary
    # fallback while the atomic audit catches up, unlike window/cursor fields.
    if not result and isinstance(obs.get("inventory_client"), dict):
        for key, raw in obs["inventory_client"].items():
            name = item_name(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(number(raw_count)))
            if name and count:
                result[name] = result.get(name, 0) + count
    return result


def inv_sig(inv):
    return tuple(sorted((str(key), int(value)) for key, value in inv.items() if int(value) > 0))


def block_rows(obs, prefix=None):
    here = self_position(obs)
    result = []
    for raw in dict_rows(obs.get("candidates")):
        ident = raw.get("id")
        if not isinstance(ident, str) or not ID_RE.fullmatch(ident):
            continue
        if prefix and not ident.startswith(prefix + ":"):
            continue
        row = dict(raw)
        row["_id"] = ident
        row["_name"] = item_name(raw.get("name", raw.get("block", "")))
        row["_pos"] = point(raw.get("position")) or id_point(ident)
        row["_distance"] = number(raw.get("distance"), distance(here, row["_pos"]))
        result.append(row)
    result.sort(key=lambda row: row.get("_distance", math.inf))
    return result


def entity_rows(obs):
    here = self_position(obs)
    source = obs.get("entities", obs.get("nearby_entities", []))
    result = []
    for raw in dict_rows(source):
        row = dict(raw)
        row["_name"] = item_name(raw.get("name", raw.get("type", "")))
        row["_pos"] = point(raw.get("position"))
        row["_distance"] = number(raw.get("distance"), distance(here, row["_pos"]))
        result.append(row)
    return result


def action_verb(action):
    return str(action or "").split(" ", 1)[0]


def food_total(inv):
    return sum(int(inv.get(name, 0)) for name in FOODS)


def wood_total(inv):
    return sum(int(value) for key, value in inv.items() if key.endswith(WOOD_SUFFIXES))


def plank_total(inv):
    return sum(int(value) for key, value in inv.items() if key.endswith("_planks"))


def first_item(inv, choices):
    return next((name for name in choices if inv.get(name, 0) > 0), "")


def progress_sig(obs):
    progress = obs.get("progress") if isinstance(obs.get("progress"), dict) else {}
    milestones = progress.get("milestones", {})
    if isinstance(milestones, dict):
        milestones = sorted((str(key), bool(value)) for key, value in milestones.items())
    else:
        milestones = str(milestones)
    advances = obs.get("advancements", [])
    if isinstance(advances, dict):
        advances = sorted(str(key) for key, value in advances.items() if value)
    elif isinstance(advances, list):
        advances = sorted(str(value) for value in advances)
    else:
        advances = str(advances)
    return json.dumps([milestones, advances], sort_keys=True, default=str)


def failed_feedback(values):
    def one(value):
        if isinstance(value, dict):
            if value.get("ok") is False or value.get("success") is False:
                return True
            status = str(value.get("status", value.get("result", ""))).lower()
            if any(word in status for word in FAIL_WORDS):
                return True
            for key in ("message", "feedback", "reason", "error"):
                text = value.get(key)
                if isinstance(text, str) and any(word in text.lower() for word in FAIL_WORDS):
                    return True
            return False
        if isinstance(value, str):
            text = value.lower()
            return any(word in text for word in FAIL_WORDS) and "success" not in text
        return False
    return any(one(value) for value in values)


class Controller:
    """Code-owned transactions plus model-authored macro/exploration choices."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.names = set()
        self.last_posted = "noop"
        self.active = None
        self.bad_until = {}
        self.completed_blocks = set()
        self.excavated = set()
        self.aimed = None
        self.exposure_count = 0
        self.table_hint = None
        self.table_expected_until = -1.0
        self.furnace_expected_until = -1.0
        self.smelt_started = {}
        self.feedback_marker = None
        self.last_inventory = ()
        self.last_progress = ""
        self.last_progress_at = 0.0
        self.phase = "opening"
        self.phase_changed_at = 0.0
        self.last_attack = -100.0
        self.dead_seen = False

        # A survey is an actual movement expedition, not a list of repeatedly
        # retried stands.  Astra may supply its next bearing, but the fallback is
        # a deterministic rotation independent of the seed.
        self.survey = None
        self.heading_index = 0
        self.astra_future = None
        self.astra_session = None
        self.astra_last_started = -1000.0
        self.astra_mode = "surface"
        self.astra_bearing = None
        self.astra_until = -1.0
        self.astra_phase = ""

        # Keep emergency and ordinary-menu inference independent.  A threat
        # response may finish after the hazard vanishes; it must not occupy the
        # transaction menu's mailbox forever.
        self.threat_future = None
        self.threat_snapshot = None
        self.threat_options = []
        self.threat_started = -1000.0
        self.menu_future = None
        self.menu_snapshot = None
        self.menu_options = []
        self.menu_started = -1000.0
        self.menu_committed = False

    def now(self):
        return number(self.ctx.clock.now())

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def allowed(self, action):
        verb = action_verb(action)
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

    def blocked(self, action_or_key, now=None):
        now = self.now() if now is None else now
        key = self.target_key(action_or_key)
        return now < max(self.bad_until.get(key, -1.0), self.bad_until.get(str(action_or_key), -1.0))

    def suppress(self, action, now, reason="", duration=None):
        action = str(action or "")
        if not action or action in {"noop", "cancel"}:
            return
        verb = action_verb(action)
        if duration is None:
            duration = 95.0 if verb in {"gather", "dig"} else 42.0 if verb in {"navigate", "goto"} else 28.0
        key = self.target_key(action)
        self.bad_until[key] = max(self.bad_until.get(key, -1.0), now + duration)
        parts = action.split()
        if len(parts) > 1 and parts[1].startswith(("b:", "s:")):
            self.bad_until[parts[1]] = max(self.bad_until.get(parts[1], -1.0), now + duration)
        if self.active and self.active.get("action") == action:
            self.active = None
        self.log("transaction_suppressed", action=action, reason=str(reason)[:120], blocked_s=duration)

    def facts(self, obs):
        me = self_row(obs)
        inv = inventory(obs)
        return me, self_position(obs) or (0.0, 64.0, 0.0), inv, block_rows(obs), entity_rows(obs)

    def process_feedback(self, obs, now):
        values = [obs.get(key) for key in ("action_feedback", "action_result", "last_result", "feedback")]
        try:
            marker = json.dumps(values, sort_keys=True, default=str)
        except Exception:
            marker = repr(values)
        if marker == self.feedback_marker:
            return
        self.feedback_marker = marker
        if self.active and failed_feedback(values):
            action = str(self.active.get("action", ""))
            self.suppress(action, now, "adapter_feedback")
            self.log("operation_rejected", action=action)

    def update_progress(self, obs, now):
        current_inv = inv_sig(inventory(obs))
        current_progress = progress_sig(obs)
        changed_inventory = current_inv != self.last_inventory
        if changed_inventory or current_progress != self.last_progress:
            self.last_inventory = current_inv
            self.last_progress = current_progress
            self.last_progress_at = now
            self.log("verified_change", inventory=changed_inventory)

    def row(self, obs, ident):
        return next((row for row in block_rows(obs, "b") if row["_id"] == ident), None)

    def near_table(self, obs):
        rows = [row for row in block_rows(obs, "b")
                if row["_name"] == "crafting_table" and row["_distance"] <= 4.8]
        rows.sort(key=lambda row: row["_distance"])
        return rows[0] if rows else None

    def table_rows(self, obs):
        return [row for row in block_rows(obs, "b") if row["_name"] == "crafting_table"]

    def furnace_rows(self, obs):
        return [row for row in block_rows(obs, "b") if row["_name"] == "furnace"]

    def support_block(self, obs, now):
        here = self_position(obs)
        if here is None:
            return None
        floor_y = math.floor(here[1]) - 1
        choices = []
        for row in block_rows(obs, "b"):
            q = row.get("_pos")
            if (row["_name"] not in SUPPORTS or row["_name"] in HAZARDS or
                    not row.get("in_reach") or row.get("visible") is False or q is None):
                continue
            if math.floor(q[1]) != floor_y:
                continue
            if math.floor(q[0]) == math.floor(here[0]) and math.floor(q[2]) == math.floor(here[2]):
                continue
            if self.blocked("place crafting_table " + row["_id"] + " up", now):
                continue
            choices.append(row)
        choices.sort(key=lambda row: row["_distance"])
        return choices[0] if choices else None

    def post(self, action, obs, why=""):
        """Post one validated action and install a postcondition latch."""
        if self.ctx.done:
            return "noop"
        action = str(action or "noop").strip() or "noop"
        if action != "noop" and not self.allowed(action):
            self.log("unsupported_action", action=action)
            action = "noop"
        verb = action_verb(action)
        if action == self.last_posted and verb not in {"noop", "controls", "attack", "shield"}:
            # The adaptor latches identical one-shot strings.  A noop is an
            # explicit re-arm barrier, so the next compiler pass can retry only
            # after it has re-read the world.
            action = "noop"
            verb = "noop"
        now = self.now()
        if verb not in {"noop", "controls", "attack", "shield", "cancel", "respawn"} and self.blocked(action, now):
            self.log("suppressed_retry", action=action)
            action = "noop"
            verb = "noop"
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            # Malformed stand-in answers or a disappearing candidate must not
            # turn a normal adaptor rejection into a harness error.
            self.log("post_exception", action=action, error=type(exc).__name__)
            try:
                self.ctx.act("noop")
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                pass
            action, verb = "noop", "noop"
        self.last_posted = action
        if action != "noop":
            self.log("posted", action=action, reason=why, phase=self.phase)
        if verb in {"cancel", "respawn"}:
            self.active = None
            return action
        if verb not in {"noop", "controls", "attack", "shield"}:
            parts = action.split()
            self.active = {
                "action": action,
                "started": now,
                "before_inv": inv_sig(inventory(obs or {})),
                "before_pos": self_position(obs or {}),
                "kind": verb,
            }
            if verb == "gather" and len(parts) > 1:
                self.active["target"] = parts[1]
            if verb == "dig" and len(parts) >= 4:
                self.active["target"] = "b:" + ",".join(parts[1:4])
                self.active["exposure"] = True
                self.exposure_count += 1
            if verb in {"navigate", "goto"} and len(parts) > 1:
                self.active["target"] = parts[1]
            if verb == "place" and len(parts) > 2:
                self.active["item"] = item_name(parts[1])
                self.active["support"] = parts[2]
                if self.active["item"] == "crafting_table":
                    support = self.row(obs or {}, parts[2])
                    if support and support.get("_pos"):
                        q = support["_pos"]
                        self.table_hint = (q[0], q[1] + 1.0, q[2])
                    self.table_expected_until = now + 10.0
                elif self.active["item"] == "furnace":
                    self.furnace_expected_until = now + 10.0
            if verb == "furnace_start" and len(parts) > 1:
                self.smelt_started[parts[1]] = now
        return action

    def update_active(self, obs, now):
        active = self.active
        if not active:
            return
        action = str(active.get("action", ""))
        verb = action_verb(action)
        inv = inventory(obs)
        before = dict(active.get("before_inv", ()))
        changed = inv_sig(inv) != tuple(sorted(before.items()))
        age = now - number(active.get("started"), now)
        me = self_row(obs)
        if verb == "equip":
            wanted = item_name(action.split()[1] if len(action.split()) > 1 else "")
            if item_name(me.get("held_item")) == wanted:
                self.active = None
                return
        elif verb == "craft":
            item = item_name(action.split()[1] if len(action.split()) > 1 else "")
            if changed or inv.get(item, 0) > before.get(item, 0):
                self.active = None
                return
        elif verb == "place":
            item = item_name(active.get("item"))
            support = self.row(obs, active.get("support", ""))
            expected = None
            if support and support.get("_pos"):
                q = support["_pos"]
                expected = (q[0], q[1] + 1.0, q[2])
            found = any(row["_name"] == item and (expected is None or distance(row.get("_pos"), expected) <= 1.2)
                        for row in block_rows(obs, "b"))
            if found or (item and before.get(item, 0) > inv.get(item, 0)):
                if item == "crafting_table":
                    self.table_expected_until = now + 7.0
                    self.log("table_placement_seen", position=self.table_hint)
                self.active = None
                return
        elif verb in {"navigate", "goto"}:
            if active.get("pickup_wait"):
                if changed:
                    self.completed_blocks.add(str(active.get("target", "")))
                    self.active = None
                    self.log("resource_confirmed", target=active.get("target", ""), method="drop_route")
                    return
                if age < 10.0:
                    # Keep the pickup latch alive after the short route moves;
                    # movement alone is not a resource postcondition.
                    return
                self.suppress(action, now, "drop_not_audited")
                self.active = None
                return
            moved = distance(active.get("before_pos"), self_position(obs)) >= 0.7
            target = active.get("target")
            reached = False
            if verb == "navigate" and target:
                row = self.row(obs, target)
                reached = bool(row and (row.get("in_reach") or row.get("_distance", 99) <= 2.5))
            if moved or reached or age >= (16.0 if verb == "navigate" else 14.0):
                if not moved and not reached and age >= (16.0 if verb == "navigate" else 14.0):
                    self.suppress(action, now, "navigation_no_progress")
                else:
                    self.active = None
                return
        elif verb == "look_at":
            if age >= 0.7:
                self.active = None
                return
        elif verb == "dig":
            target = active.get("target", "")
            row = self.row(obs, target)
            if row is None or row.get("_name") in {"air", "water", "lava"}:
                self.excavated.add(str(target))
                self.aimed = None
                self.active = None
                self.log("excavation_confirmed", target=target)
                return
            if changed and active.get("resource"):
                self.completed_blocks.add(str(target))
                self.active = None
                return
        elif verb == "gather":
            target = active.get("target", "")
            if changed:
                self.completed_blocks.add(str(target))
                self.active = None
                if self.phase == "stone":
                    self.exposure_count = 0
                self.log("resource_confirmed", target=target, method="inventory")
                return
            # A gather job can remove a block before the atomic inventory audit
            # catches the item.  Give its drop a short window; do not reissue
            # the same latched gather string.
            if self.row(obs, target) is None and age > 4.0:
                active["pickup_wait"] = True
                if age > 12.0:
                    self.suppress(action, now, "drop_not_audited")
                    self.active = None
                    return
        elif verb in {"furnace_start", "furnace_take", "eat", "sleep"}:
            if changed or age >= 7.0:
                self.active = None
                return
        # A task owns the hands, but it cannot be allowed to consume the whole
        # episode without a postcondition.
        limits = {
            "gather": 30.0, "dig": 11.0, "navigate": 18.0, "goto": 16.0,
            "place": 10.0, "craft": 8.0, "equip": 5.0, "look_at": 3.0,
            "furnace_start": 8.0, "furnace_take": 8.0, "eat": 7.0,
            "sleep": 8.0,
        }
        if age >= limits.get(verb, 8.0) and not has_task(obs):
            self.suppress(action, now, "postcondition_timeout")
            self.active = None

    def active_followup(self, obs, now):
        """Return a movement needed to collect a vanished direct-gather drop."""
        active = self.active
        if not active or not active.get("pickup_wait"):
            return None
        target = active.get("target", "")
        q = id_point(target)
        drops = []
        for ent in entity_rows(obs):
            if ent["_name"] not in {"item", "item_stack", "item_entity"} or not ent.get("_pos"):
                continue
            if ent["_distance"] <= 10.0 and (q is None or distance(ent["_pos"], q) <= 8.0):
                drops.append(ent)
        drops.sort(key=lambda row: row["_distance"])
        if drops and now - number(active.get("started"), now) < 10.0:
            p = drops[0]["_pos"]
            action = "goto %.2f %.2f %.2f" % (p[0], p[1] - 1.0, p[2])
            # The goto becomes the owned latch, but retains the expected
            # inventory signature and a pickup marker.
            before = active.get("before_inv", ())
            self.post(action, obs, "collect_real_drop")
            if self.active:
                self.active["pickup_wait"] = True
                self.active["before_inv"] = before
            return action
        return None

    def valid_target(self, row, names=None, suffixes=(), now=None):
        now = self.now() if now is None else now
        name = row.get("_name", "")
        if names is not None and name not in names:
            return False
        if suffixes and not any(name.endswith(suffix) for suffix in suffixes):
            return False
        if not (row.get("visible") or row.get("in_reach")):
            return False
        if row.get("harvestable") is False or name in HAZARDS or name in {"air", "water", "lava"}:
            return False
        ident = row.get("_id", "")
        return ident not in self.completed_blocks and not self.blocked(ident, now) and not self.blocked("gather " + ident, now)

    def choose_target(self, obs, names=None, suffixes=(), now=None):
        now = self.now() if now is None else now
        rows = [row for row in block_rows(obs, "b") if self.valid_target(row, names, suffixes, now)]
        rows.sort(key=lambda row: (
            0 if row.get("in_reach") else 1,
            0 if row.get("visible") else 1,
            row.get("_distance", math.inf),
            row.get("_id", ""),
        ))
        return rows[0] if rows else None

    def safe_excavation_target(self, obs, now):
        here = self_position(obs)
        if here is None:
            return None
        choices = []
        for row in block_rows(obs, "b"):
            if row["_name"] not in EXCAVATABLE or row.get("_id") in self.excavated:
                continue
            if not row.get("visible") or not row.get("in_reach") or row.get("harvestable") is False:
                continue
            if row["_name"] in HAZARDS or self.blocked(row["_id"], now) or self.blocked("dig " + row["_id"], now):
                continue
            q = row.get("_pos")
            if q is None:
                continue
            if abs(q[1] - here[1]) > 1.5 or abs(q[0] - here[0]) + abs(q[2] - here[2]) > 5.0:
                continue
            # or a lower adjacent block is a safe exposure transaction.
            if math.floor(q[0]) == math.floor(here[0]) and math.floor(q[2]) == math.floor(here[2]) and q[1] <= here[1] - 0.5:
                continue
            side = abs(q[0] - here[0]) + abs(q[2] - here[2]) >= 1.0
            below = q[1] < here[1]
            rank = (
                0 if row["_name"] in {"dirt", "coarse_dirt", "rooted_dirt"} else 1,
                0 if below else 1,
                0 if side else 1,
                row.get("_distance", math.inf),
            )
            choices.append((rank, row))
        choices.sort(key=lambda pair: pair[0])
        return choices[0][1] if choices else None

    def table_action(self, obs, inv, now, options):
        table = self.near_table(obs)
        if table:
            return table
        tables = self.table_rows(obs)
        if tables:
            tables.sort(key=lambda row: row.get("_distance", math.inf))
            row = tables[0]
            if not self.blocked("navigate " + row["_id"], now):
                options.append({"action": "navigate " + row["_id"], "purpose": "Approach the observed crafting table", "priority": 100})
            return None
        if self.table_hint and now < self.table_expected_until:
            # Wait for the block stream rather than crafting another table.
            options.append({"action": "noop", "purpose": "Wait for the placed table to enter the observation window", "priority": 98})
        elif self.table_hint and not self.blocked("goto %.2f %.2f %.2f" % (self.table_hint[0] + 1.0, self.table_hint[1], self.table_hint[2] + 1.0), now):
            q = self.table_hint
            options.append({"action": "goto %.2f %.2f %.2f" % (q[0] + 1.0, q[1], q[2] + 1.0), "purpose": "Recover the observed workstation after the block stream lag", "priority": 101})
        elif inv.get("crafting_table"):
            support = self.support_block(obs, now)
            if support:
                held = item_name(self_row(obs).get("held_item"))
                action = ("equip crafting_table hand" if held != "crafting_table"
                          else "place crafting_table %s up" % support["_id"])
                options.append({"action": action, "purpose": "Complete the staged workstation placement", "priority": 120})
        return None

    @staticmethod
    def wood_planks(log_name):
        if log_name.endswith("_log"):
            return log_name[:-4] + "_planks"
        if log_name.endswith("_stem"):
            return log_name[:-5] + "_planks"
        return "oak_planks"

    def feasible(self, item, inv):
        """Return a legal output count from actual personal inventory."""
        if item.endswith("_planks"):
            base = item[:-7]
            return (inv.get(base + "_log", 0) + inv.get(base + "_stem", 0)) * 4
        planks = plank_total(inv)
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
            return min(inv.get("iron_ingot", 0), planks // 5)
        if item == "bucket":
            return inv.get("iron_ingot", 0) // 3
        return 0

    def craft_option(self, options, item, desired, inv, table, purpose, priority):
        count = min(max(1, int(desired)), max(0, int(self.feasible(item, inv))))
        if count < 1:
            return False
        action = "craft %s %d" % (item, count)
        if table:
            action += " " + table["_id"]
        if not self.blocked(action):
            options.append({"action": action, "purpose": purpose, "priority": priority})
            return True
        return False

    def phase_for(self, obs, inv):
        tables = self.table_rows(obs)
        if wood_total(inv) == 0 and plank_total(inv) == 0 and not inv.get("crafting_table") and not tables and not self.table_hint:
            return "wood"
        # A carried table is a pending placement; no table at all means the
        # recipe gate must run before tools.  This is the opening dependency
        # that the previous draft accidentally skipped after the first log.
        if not tables and not self.table_hint and not inv.get("crafting_table"):
            return "table"
        if not tables and not self.table_hint and inv.get("crafting_table"):
            return "table"
        if not inv.get("wooden_pickaxe") and not any(inv.get(name) for name in STONE_PICKAXES):
            return "wood_tools"
        if not any(inv.get(name) for name in STONE_PICKAXES):
            return "stone"
        raw = inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0)
        if raw and not inv.get("iron_ingot"):
            return "smelt"
        if not inv.get("iron_ingot") and not inv.get("shield"):
            return "iron"
        if inv.get("iron_ingot") and not inv.get("shield"):
            return "shield"
        return "expedition"

    def food_options(self, obs, inv, options, now):
        if food_total(inv) >= 5:
            return
        target = self.choose_target(obs, names=FOOD_BLOCKS, now=now)
        if target:
            options.append({
                "action": "gather %s 1" % target["_id"],
                "purpose": "Collect an observed food source before a longer expedition",
                "priority": 64,
            })

    def compile_options(self, obs, now):
        me, here, inv, blocks, entities = self.facts(obs)
        phase = self.phase_for(obs, inv)
        if phase != self.phase:
            self.phase = phase
            self.phase_changed_at = now
            self.aimed = None
            self.exposure_count = 0
            self.log("phase_change", phase=phase)
        options = []
        table = self.near_table(obs)

        # Hunger is an exact safety precondition, not a Jev judgment.
        if number(me.get("food"), 20) <= 12:
            food = first_item(inv, FOODS)
            if food:
                if item_name(me.get("held_item")) != food:
                    options.append({"action": "equip %s hand" % food, "purpose": "Hold food before eating", "priority": 150})
                else:
                    options.append({"action": "eat " + food, "purpose": "Restore hunger before movement", "priority": 151})

        if phase == "wood":
            target = self.choose_target(obs, suffixes=WOOD_SUFFIXES, now=now)
            if target:
                options.append({"action": "gather %s 1" % target["_id"], "purpose": "Collect the first exposed log", "priority": 130})
            self.food_options(obs, inv, options, now)
            return options, phase

        if phase == "table":
            logs = next((key for key, value in inv.items() if value and key.endswith(WOOD_SUFFIXES)), "")
            planks = plank_total(inv)
            if planks < 4 and logs:
                self.craft_option(options, self.wood_planks(logs), 4, inv, None, "Make the planks needed for the workstation", 125)
            elif planks >= 4 and not inv.get("crafting_table"):
                self.craft_option(options, "crafting_table", 1, inv, None, "Craft the opening workstation", 126)
            self.table_action(obs, inv, now, options)
            return options, phase

        if phase == "wood_tools":
            logs = next((key for key, value in inv.items() if value and key.endswith(WOOD_SUFFIXES)), "")
            planks = plank_total(inv)
            sticks = inv.get("stick", 0)
            if planks < 3 and logs:
                self.craft_option(options, self.wood_planks(logs), 4, inv, None, "Make feasible planks for the wooden pick", 122)
            elif planks < 3:
                target = self.choose_target(obs, suffixes=WOOD_SUFFIXES, now=now)
                if target:
                    options.append({"action": "gather %s 1" % target["_id"], "purpose": "Collect more exposed wood for the wooden pick", "priority": 121})
            elif sticks < 2 and planks >= 2:
                self.craft_option(options, "stick", 4, inv, None, "Make pickaxe handles from carried planks", 121)
            elif table:
                self.craft_option(options, "wooden_pickaxe", 1, inv, table, "Craft the first pickaxe at the observed table", 120)
            else:
                self.table_action(obs, inv, now, options)
            self.food_options(obs, inv, options, now)
            return options, phase

        if phase == "stone":
            cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
            if cobble < 3:
                target = self.choose_target(obs, names=STONE_BLOCKS, now=now)
                if target:
                    options.append({"action": "gather %s 1" % target["_id"], "purpose": "Mine one exposed stone block with the wooden pick", "priority": 116})
                else:
                    # A short, safe face transaction is preferable to another
                    # navigation retry when terrain is merely covered by dirt.
                    ground = self.safe_excavation_target(obs, now) if self.exposure_count < 6 else None
                    if ground:
                        q = ground["_pos"]
                        held = item_name(me.get("held_item"))
                        if held != "wooden_pickaxe" and inv.get("wooden_pickaxe"):
                            options.append({"action": "equip wooden_pickaxe hand", "purpose": "Equip the tool for the exposure cut", "priority": 114})
                        elif self.aimed != ground["_id"]:
                            options.append({"action": "look_at %.2f %.2f %.2f" % q, "purpose": "Aim at a safe side block before exposing the next layer", "priority": 113})
                        else:
                            options.append({"action": "dig %d %d %d" % tuple(int(round(x)) for x in q), "purpose": "Remove one safe surface block to expose stone", "priority": 112})
                    else:
                        self.food_options(obs, inv, options, now)
            elif inv.get("stick", 0) < 2 and plank_total(inv) >= 2:
                self.craft_option(options, "stick", 4, inv, None, "Restore the handles needed for the stone pick", 115)
            elif table:
                self.craft_option(options, "stone_pickaxe", 1, inv, table, "Craft the stone pick from verified cobblestone", 114)
            else:
                self.table_action(obs, inv, now, options)
            self.food_options(obs, inv, options, now)
            return options, phase

        if phase == "iron":
            target = self.choose_target(obs, names=IRON_BLOCKS, now=now)
            if target:
                options.append({"action": "gather %s 1" % target["_id"], "purpose": "Mine observed iron with the stone pick", "priority": 112})
            else:
                target = self.choose_target(obs, names=FUEL_BLOCKS, now=now)
                if target and not inv.get("coal") and not inv.get("charcoal"):
                    options.append({"action": "gather %s 1" % target["_id"], "purpose": "Collect observed fuel for the furnace", "priority": 95})
            self.food_options(obs, inv, options, now)
            return options, phase

        if phase == "smelt":
            furnaces = self.furnace_rows(obs)
            raw = inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0)
            if not furnaces and not inv.get("furnace"):
                cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
                if table and cobble >= 8:
                    self.craft_option(options, "furnace", 1, inv, table, "Craft a real furnace for the iron", 109)
                elif cobble < 8:
                    target = self.choose_target(obs, names=STONE_BLOCKS, now=now)
                    if target:
                        options.append({"action": "gather %s 1" % target["_id"], "purpose": "Gather stone for the furnace", "priority": 105})
            elif not furnaces and inv.get("furnace"):
                support = self.support_block(obs, now)
                if support:
                    held = item_name(me.get("held_item"))
                    options.append({"action": ("equip furnace hand" if held != "furnace" else "place furnace %s up" % support["_id"]), "purpose": "Place the furnace before smelting", "priority": 108})
            elif furnaces:
                furnace = min(furnaces, key=lambda row: row.get("_distance", math.inf))
                fid = furnace["_id"]
                if furnace["_distance"] > 4.8:
                    options.append({"action": "navigate " + fid, "purpose": "Approach the observed furnace", "priority": 104})
                else:
                    fuel = first_item(inv, ("coal", "charcoal"))
                    if not fuel:
                        fuel = next((key for key, value in inv.items() if key.endswith("_planks") and value), "")
                    if raw and fuel and fid not in self.smelt_started:
                        options.append({"action": "furnace_start %s raw_iron %d %s %d" % (fid, min(3, raw), fuel, min(3, inv.get(fuel, 0))), "purpose": "Load real iron and fuel into the furnace", "priority": 103})
                    elif fid in self.smelt_started and now - self.smelt_started[fid] >= 32.0:
                        options.append({"action": "furnace_take " + fid, "purpose": "Retrieve the finished iron", "priority": 102})
            return options, phase

        if phase == "shield":
            logs = next((key for key, value in inv.items() if value and key.endswith(WOOD_SUFFIXES)), "")
            if plank_total(inv) < 6 and logs:
                self.craft_option(options, self.wood_planks(logs), 8, inv, None, "Make shield planks without over-requesting a recipe", 110)
            elif table:
                self.craft_option(options, "shield", 1, inv, table, "Craft the iron shield milestone", 109)
            else:
                self.table_action(obs, inv, now, options)
            return options, phase

        # After the first shield, prefer any naturally observed portal and then
        # continue finding food/resources.  This branch is intentionally
        # evidence-gated; it never assumes a portal or a structure.
        portals = [row for row in blocks if row["_name"] == "nether_portal" and row.get("visible")]
        if portals and inv.get("flint_and_steel"):
            portals.sort(key=lambda row: row.get("_distance", math.inf))
            portal = portals[0]
            if portal["_distance"] > 4.5:
                options.append({"action": "navigate " + portal["_id"], "purpose": "Approach the observed natural portal", "priority": 100})
            else:
                held = item_name(me.get("held_item"))
                options.append({"action": ("equip flint_and_steel hand" if held != "flint_and_steel" else "activate_block %s up" % portal["_id"]), "purpose": "Activate the observed portal with ordinary items", "priority": 100})
        self.food_options(obs, inv, options, now)
        return options, phase

    def begin_survey(self, now, reason):
        bearing = self.astra_bearing if now < self.astra_until else None
        if bearing not in {"north", "south", "east", "west"}:
            bearing = ("east", "south", "west", "north")[self.heading_index % 4]
            self.heading_index += 1
        self.survey = {"bearing": bearing, "side": "left" if self.heading_index % 2 else "right", "stage": "aim", "started": now, "aimed_at": None, "last_pos": None, "reason": reason}
        self.log("expedition_begin", bearing=bearing, reason=reason)

    def survey_vector(self, bearing):
        return {"east": (1.0, 0.0), "west": (-1.0, 0.0), "south": (0.0, 1.0), "north": (0.0, -1.0)}.get(bearing, (1.0, 0.0))

    def survey_action(self, obs, now):
        if not self.survey:
            return None
        job = self.survey
        here = self_position(obs) or (0.0, 64.0, 0.0)
        dx, dz = self.survey_vector(job.get("bearing"))
        if job.get("stage") == "aim":
            job["stage"] = "move"
            job["move_started"] = now
            job["last_pos"] = here
            return "look_at %.2f %.2f %.2f" % (here[0] + dx * 12.0, here[1] + 0.4, here[2] + dz * 12.0)
        if job.get("stage") == "move":
            elapsed = now - number(job.get("move_started"), now)
            moved = distance(job.get("last_pos"), here)
            if elapsed >= 13.0 or (elapsed >= 4.0 and moved < 0.6):
                self.survey = None
                self.exposure_count = 0
                self.log("expedition_end", bearing=job.get("bearing"), moved=round(moved, 2), elapsed=round(elapsed, 1))
                return "controls none"
            keys = "forward,jump," + str(job.get("side", "right"))
            if number(self_row(obs).get("food"), 20) >= 14 and number(self_row(obs).get("health"), 20) >= 12:
                keys += ",sprint"
            return "controls " + keys
        self.survey = None
        return "controls none"

    def relevant_target_exists(self, obs, phase, now):
        if phase == "stone":
            if any(self.valid_target(row, STONE_BLOCKS, (), now) for row in block_rows(obs, "b")):
                return True
            return False
        names = None
        suffixes = ()
        if phase in {"wood", "wood_tools", "shield"}:
            suffixes = WOOD_SUFFIXES
        elif phase == "stone":
            names = STONE_BLOCKS | EXCAVATABLE
        elif phase == "iron":
            names = IRON_BLOCKS | FUEL_BLOCKS
        elif phase == "smelt":
            names = STONE_BLOCKS
        elif phase == "expedition":
            names = FOOD_BLOCKS | {"nether_portal", "nether_bricks"}
        if names is None and not suffixes:
            return False
        return any(self.valid_target(row, names, suffixes, now) for row in block_rows(obs, "b"))

    def threats(self, obs):
        me = self_row(obs)
        result = []
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            result.append({"kind": "lava_or_fire", "id": "self", "range": "touching", "distance": 0.0})
        if me.get("is_in_water") and number(me.get("oxygen"), 20) < 6:
            result.append({"kind": "drowning", "id": "self", "range": "touching", "distance": 0.0})
        for row in block_rows(obs, "b"):
            if row["_name"] in HAZARDS and row.get("_distance", 99) <= 4.5:
                result.append({"kind": row["_name"], "id": row["_id"], "range": "near", "distance": row["_distance"]})
        for row in entity_rows(obs):
            hostile = bool(row.get("hostile") or row.get("conditional_threat") or row["_name"] in HOSTILES)
            if hostile and row.get("visible") and row.get("_distance", 99) <= 8.0:
                result.append({
                    "kind": row["_name"], "id": str(row.get("id", "")),
                    "range": "touching" if row["_distance"] <= 3.3 else "near",
                    "distance": row["_distance"],
                })
        return result

    def safety_fallback(self, obs, hazards):
        me = self_row(obs)
        inv = inventory(obs)
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            return "controls forward,jump" if self.allowed("controls") else "cancel"
        if me.get("is_in_water") and number(me.get("oxygen"), 20) < 6:
            return "controls jump" if self.allowed("controls") else "cancel"
        if number(me.get("health"), 20) <= 7 or number(me.get("food"), 20) <= 5:
            food = first_item(inv, FOODS)
            if food:
                if item_name(me.get("held_item")) != food and self.allowed("equip"):
                    return "equip %s hand" % food
                if self.allowed("eat"):
                    return "eat " + food
        close = [row for row in hazards if row.get("id") != "self" and row.get("range") == "touching"]
        weapon = first_item(inv, WEAPONS)
        if close and weapon:
            if item_name(me.get("held_item")) != weapon and self.allowed("equip"):
                return "equip " + weapon + " hand"
            if self.allowed("attack") and close[0].get("id") and self.now() - self.last_attack >= 0.8:
                self.last_attack = self.now()
                return "attack " + str(close[0]["id"])
        if inv.get("shield") and self.allowed("shield"):
            return "shield on"
        return "controls back,jump" if self.allowed("controls") else "noop"

    def tactical_action(self, obs, now):
        hazards = self.threats(obs)
        if not hazards:
            return None
        fallback = self.safety_fallback(obs, hazards)
        if has_task(obs) or self.active:
            if fallback not in {"shield on", "noop"} and self.allowed("cancel"):
                return "cancel"
            if fallback == "shield on" and self.allowed("cancel"):
                return "cancel"
        options = []
        for action in (fallback, "shield on", "controls back,jump", "noop"):
            if action and self.allowed(action) and action not in options:
                options.append(action)
        weapon = first_item(inventory(obs), WEAPONS)
        for threat in hazards:
            if threat.get("id") != "self" and threat.get("range") == "touching" and weapon:
                for action in ("equip %s hand" % weapon, "attack " + str(threat.get("id", ""))):
                    if self.allowed(action) and action not in options:
                        options.append(action)
        options = options[:7]
        if not options:
            return fallback

        signature = tuple(sorted((x.get("kind"), x.get("id"), x.get("range")) for x in hazards))
        # Tactical calls are independent of the longer transaction menu.  The
        # fallback is immediate; Jev can replace it only on the same hazard set.
        if self.threat_future is None and self.ctx.budget.remaining().get("system1_calls", 0) > 2:
            state = {
                "purpose": "Immediate survival interrupt. Code has already removed illegal actions.",
                "self": {
                    "health_band": "critical" if number(self_row(obs).get("health"), 20) < 7 else "stable",
                    "dimension": str(self_row(obs).get("dimension", "")),
                    "environment": "lava_or_fire" if self_row(obs).get("is_in_lava") or self_row(obs).get("is_in_fire") else "ordinary",
                },
                "hazards": [{"kind": x.get("kind", ""), "range": x.get("range", "near")} for x in hazards[:6]],
                "legal_actions": options,
                "rule": "Choose only a legal_actions string. Escape and shield beat work; never attack environmental hazards.",
            }
            criteria = {action: "Exact supplied response: " + action for action in options}
            questions = {
                "response": Choice("Choose the safest supplied immediate response.", criteria),
                "same_response": Choice("Recheck the same emergency with the options in reverse order.", dict(reversed(list(criteria.items())))),
                "continue": Noul("Is it safe to continue the current work without a defensive action?", {"true": "No immediate hazard requires interruption.", "false": "A hazard requires escape, defense, or interruption."}),
            }
            try:
                self.threat_future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=1.5))
                self.threat_snapshot = (signature, now)
                self.threat_options = list(options)
                self.threat_started = now
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                self.threat_future = None
        if self.threat_future is not None and self.threat_snapshot:
            future = self.threat_future
            if not future.done():
                return fallback
            self.threat_future = None
            try:
                answers = future.result()
                first = answers.get("response") if isinstance(answers, dict) else None
                second = answers.get("same_response") if isinstance(answers, dict) else None
                value = getattr(first, "value", None)
                value2 = getattr(second, "value", None)
                probs = getattr(first, "probabilities", {}) or {}
                p = number(probs.get(value), 0.0) if isinstance(probs, dict) else 0.0
                if value in options and value == value2 and p >= 0.45 and self.threat_snapshot[0] == signature:
                    self.log("jev_threat", action=value, probability=round(p, 3))
                    return value
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                self.log("jev_threat_fallback")
        return fallback

    def compile_jev(self, obs, options, now):
        if not options:
            return None
        # Keep the menu small and deterministic.  Priority is code-owned; the
        # model sees descriptions and exact commands, not raw candidate clutter.
        options = sorted(options, key=lambda row: (-number(row.get("priority"), 0), str(row.get("action", ""))))[:7]
        actions = [str(row.get("action", "noop")) for row in options]
        if self.menu_future is None and self.ctx.budget.remaining().get("system1_calls", 0) > 2:
            me, here, inv, _, entities = self.facts(obs)
            state = {
                "objective": "Free the End through the next verified survival milestone",
                "phase": self.phase,
                "dimension": str(me.get("dimension", "overworld")),
                "health_band": "critical" if number(me.get("health"), 20) < 7 else "low" if number(me.get("health"), 20) < 12 else "stable",
                "food_band": "low" if number(me.get("food"), 20) < 10 else "fed",
                "inventory": {key: int(value) for key, value in sorted(inv.items())
                              if key in FOODS or key in {"crafting_table", "wooden_pickaxe", "stone_pickaxe", "cobblestone", "raw_iron", "iron_ingot", "shield", "furnace", "coal", "charcoal", "stick"}
                              or key.endswith(("_log", "_planks", "_wool", "_bed"))},
                "visible_threats": [{"kind": row.get("_name", ""), "range": "touching" if row.get("_distance", 99) <= 3.3 else "near"}
                                    for row in entities if row.get("visible") and (row.get("hostile") or row.get("_name") in HOSTILES)][:5],
                "options": [{"command": action, "purpose": str(row.get("purpose", ""))} for action, row in zip(actions, options)],
                "rule": "Return only an exact command from options. Never invent an ID, count, recipe, coordinate, or action. Prefer a verified next milestone and avoid unnecessary danger.",
            }
            criteria = {action: str(row.get("purpose", "Exact supplied command")) for action, row in zip(actions, options)}
            questions = {
                "action": Choice("Choose the most useful safe exact command from the supplied transaction menu.", criteria),
                "action_check": Choice("Check that choice with the options in reverse order; choose the same supplied command if safe.", dict(reversed(list(criteria.items())))),
                "continue": Noul("Is the highest-priority supplied transaction safe to attempt now?", {"true": "The compiled preconditions and visible surroundings support it.", "false": "A visible hazard or missing precondition makes it unsafe."}),
            }
            try:
                self.menu_future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=1.5))
                self.menu_snapshot = (obs.get("episode"), obs.get("seq"), now)
                self.menu_options = actions
                self.menu_started = now
                self.menu_committed = False
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                self.menu_future = None
        fallback = actions[0]
        if self.menu_future is None or not self.menu_snapshot:
            return fallback
        future = self.menu_future
        if not future.done():
            if now - self.menu_started >= 0.45:
                self.menu_committed = True
            return fallback
        self.menu_future = None
        if self.menu_committed:
            try:
                future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                pass
            return fallback
        try:
            answers = future.result()
            first = answers.get("action") if isinstance(answers, dict) else None
            second = answers.get("action_check") if isinstance(answers, dict) else None
            value = getattr(first, "value", None)
            value2 = getattr(second, "value", None)
            probs = getattr(first, "probabilities", {}) or {}
            p = number(probs.get(value), 0.0) if isinstance(probs, dict) else 0.0
            old_seq = self.menu_snapshot[1]
            current_seq = obs.get("seq")
            fresh = (self.menu_snapshot[0] == obs.get("episode") and
                     (old_seq is None or current_seq is None or number(current_seq) - number(old_seq) <= 8))
            if fresh and value == value2 and value in actions and p >= 0.45:
                self.log("jev_transaction", action=value, probability=round(p, 3), phase=self.phase)
                return value
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception:
            self.log("jev_transaction_fallback")
        return fallback

    def parse_astra(self, text):
        text = str(text or "").lower()
        modes = {"cave", "surface", "food", "shelter", "portal", "progress"}
        bearings = {"north", "south", "east", "west"}
        mode = None
        bearing = None
        try:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(text[start:end + 1])
                if isinstance(data, dict):
                    if data.get("mode") in modes:
                        mode = data.get("mode")
                    if data.get("bearing") in bearings:
                        bearing = data.get("bearing")
        except Exception:
            pass
        if mode is None:
            for value in modes:
                if re.search(r"\b" + re.escape(value) + r"\b", text):
                    mode = value
                    break
        if bearing is None:
            for value in bearings:
                if re.search(r"\b" + value + r"\b", text):
                    bearing = value
                    break
        return mode, bearing

    def poll_astra(self, now):
        if self.astra_future is None or not self.astra_future.done():
            return
        future = self.astra_future
        self.astra_future = None
        try:
            result = future.result()
            mode, bearing = self.parse_astra(getattr(result, "text", ""))
            if mode:
                self.astra_mode = mode
            if bearing:
                self.astra_bearing = bearing
            self.astra_until = now + 100.0
            session = getattr(result, "session_id", None)
            if session:
                self.astra_session = session
            self.log("astra_policy", mode=self.astra_mode, bearing=self.astra_bearing)
        except (BudgetExceeded, ModelUnavailable):
            # If the episode is closing, the background call is normally
            # cancelled.  Other backend failures are not useful route advice.
            if not self.ctx.done:
                self.log("astra_closed")
        except Exception:
            self.log("astra_unusable")

    def maybe_astra(self, obs, now):
        self.poll_astra(now)
        if self.astra_future is not None:
            return
        remaining = self.ctx.budget.remaining()
        if remaining.get("system2_calls", 0) <= 0 or remaining.get("wall_s", 0) < 220:
            return
        stalled = now - self.last_progress_at >= 45.0
        phase_due = self.phase != self.astra_phase and now - self.astra_last_started > 35.0
        if not (stalled or phase_due) or now - self.astra_last_started < 120.0:
            return
        inv = inventory(obs)
        visible = []
        for row in block_rows(obs, "b"):
            if row["_name"] in (set(STONE_BLOCKS) | IRON_BLOCKS | FUEL_BLOCKS | EXCAVATABLE | FOOD_BLOCKS | {"nether_portal", "nether_bricks"}):
                visible.append({
                    "name": row["_name"],
                    "visibility": "visible" if row.get("visible") else "occluded",
                    "range": "reach" if row.get("in_reach") else "near" if row.get("_distance", 99) < 10 else "far",
                })
        state = {
            "phase": self.phase,
            "dimension": str(self_row(obs).get("dimension", "overworld")),
            "inventory_bands": {
                "wood": wood_total(inv), "planks": plank_total(inv), "food": food_total(inv),
                "stone": inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0),
                "raw_iron": inv.get("raw_iron", 0), "iron": inv.get("iron_ingot", 0),
                "shield": bool(inv.get("shield")),
            },
            "nearby_exposed": visible[:24],
            "expeditions_completed": self.heading_index,
            "recent_suppressed": list(self.bad_until.keys())[-8:],
            "rule": "Reply with very short JSON {\"mode\":\"cave|surface|food|shelter|portal|progress\",\"bearing\":\"north|south|east|west\"}. Choose only from evidence. Never emit commands, coordinates, seed facts, or hidden ores.",
        }
        prompt = (
            "You are the slow macro planner for an ordinary Minecraft survival client. "
            "Code owns recipes, exact actions, safety vetoes, and all coordinates. "
            "Choose a coarse search mode and a compass bearing for the next bounded expedition. "
            "A cave mode means seek exposed terrain/cave entrances; surface means change the loaded "
            "window; food means prioritize a visible food opportunity; shelter means look for a bed or "
            "safe night stop. Do not assume anything absent from the supplied state. Return only the "
            "requested compact JSON.\n" + json.dumps(state, separators=(",", ":"), default=str)
        )
        self.astra_last_started = now
        self.astra_phase = self.phase
        try:
            session = self.astra_session
            self.astra_future = self.ctx.spawn(lambda p=prompt, s=session: self.ctx.think(p, session_id=s, timeout_s=105.0))
            self.log("astra_started", phase=self.phase)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception:
            self.astra_future = None

    def emergency_or_food(self, obs, now):
        me = self_row(obs)
        inv = inventory(obs)
        if me.get("dead"):
            return "respawn"
        if number(obs.get("observation_age_s"), 0) > 3.0:
            return "cancel" if has_task(obs) else "noop"
        tactical = self.tactical_action(obs, now)
        if tactical:
            return tactical
        return None

    def start_action(self, obs, now):
        """Compile the next transaction or start a directional expedition."""
        options, phase = self.compile_options(obs, now)
        # A newly visible resource always interrupts a survey before it can
        # consume another movement leg.
        if not options:
            if not self.survey:
                self.begin_survey(now, "no observable transaction for " + phase)
            return self.survey_action(obs, now)
        return self.compile_jev(obs, options, now)

    def run_episode(self, game):
        self.names = set(getattr(getattr(game, "action_spec", None), "names", ()) or ())
        self.last_progress_at = self.now()
        while not self.ctx.done:
            now = self.now()
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                self.post("noop", {}, "missing_observation")
                self.ctx.wait_for_tick()
                continue
            self.process_feedback(obs, now)
            self.update_progress(obs, now)
            self.update_active(obs, now)
            me = self_row(obs)
            if me.get("dead"):
                self.active = None
                self.survey = None
                self.aimed = None
                if not self.dead_seen:
                    self.dead_seen = True
                    self.post("respawn", obs, "ordinary_respawn")
                else:
                    self.post("noop", obs, "await_respawn")
                self.ctx.wait_for_tick()
                continue
            self.dead_seen = False
            urgent = self.emergency_or_food(obs, now)
            if urgent:
                if has_task(obs) and urgent not in {"cancel", "noop"}:
                    self.post("cancel", obs, "hazard_interrupt")
                else:
                    self.post(urgent, obs, "safety_policy")
                self.ctx.wait_for_tick()
                continue
            if has_task(obs):
                self.post("noop", obs, "continue_bounded_job")
                self.ctx.wait_for_tick()
                continue
            follow = self.active_followup(obs, now)
            if follow:
                self.ctx.wait_for_tick()
                continue
            if self.active:
                self.post("noop", obs, "await_transaction_postcondition")
                self.ctx.wait_for_tick()
                continue

            # A current observed target outranks continuing a blind expedition.
            current_phase = self.phase_for(obs, inventory(obs))
            if self.survey and self.relevant_target_exists(obs, current_phase, now):
                self.survey = None
                self.post("controls none", obs, "new_observed_opportunity")
                self.ctx.wait_for_tick()
                continue
            if self.survey:
                action = self.survey_action(obs, now) or "noop"
                self.post(action, obs, "directional_expedition")
                self.ctx.wait_for_tick()
                continue

            self.maybe_astra(obs, now)
            action = self.start_action(obs, now)
            self.post(action or "noop", obs, "compiled_transaction")
            # A direct dig target becomes the aimed target only after the
            # look_at action; this keeps the one-hand job ordering explicit.
            if action and action.startswith("look_at "):
                # Identify the currently selected excavation target from the
                # fresh menu without asking a model to parse coordinates.
                target = self.safe_excavation_target(obs, now)
                if target:
                    self.aimed = target["_id"]
            self.ctx.wait_for_tick()


def build(ctx):
    return Controller(ctx)
