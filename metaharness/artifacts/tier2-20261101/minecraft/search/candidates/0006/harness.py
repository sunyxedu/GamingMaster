"""Candidate 0006: a deterministic milestone executor with modelled threat interrupts.

The controller does not ask Jev to invent a Minecraft plan.  Code maintains a
small, restartable build order and compiles only observed, legal commands.
Jev is reserved for the short-lived judgement that code cannot safely encode:
which response is best when a visible threat competes with the current job.
Astra is a slow macro adviser used only to choose among already implemented
resource phases when observation-driven progress stalls.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


_ID = re.compile(r"^[bs]:-?\d+,-?\d+,-?\d+$")
_ITEM = re.compile(r"^[a-z0-9_]+$")

FOODS = {
    "apple", "bread", "baked_potato", "carrot", "golden_carrot",
    "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
    "cooked_rabbit", "cooked_cod", "cooked_salmon", "sweet_berries",
    "melon_slice", "beef", "porkchop", "mutton", "chicken", "rabbit",
    "cod", "salmon", "potato", "beetroot", "mushroom_stew",
}
_ANIMALS = {"cow", "pig", "sheep", "chicken", "rabbit"}
_HOSTILE_NAMES = {
    "zombie", "skeleton", "creeper", "spider", "drowned", "husk", "stray",
    "witch", "phantom", "enderman", "slime", "magma_cube", "blaze",
    "ghast", "piglin", "piglin_brute", "zombified_piglin", "hoglin",
    "wither_skeleton", "ender_dragon", "shulker", "silverfish",
}
_DANGEROUS_BLOCKS = {
    "lava", "fire", "soul_fire", "magma_block", "cactus", "powder_snow",
    "campfire", "soul_campfire", "pointed_dripstone",
}
_SUPPORTS = {
    "stone", "cobblestone", "cobbled_deepslate", "deepslate", "dirt",
    "grass_block", "sand", "gravel", "terracotta", "netherrack",
    "blackstone", "oak_log", "birch_log", "spruce_log", "jungle_log",
    "acacia_log", "dark_oak_log", "mangrove_log", "chest", "crafting_table",
    "furnace", "podzol", "mycelium", "dirt_path", "sandstone", "red_sand",
    "snow_block", "snow", "clay", "moss_block", "basalt", "end_stone",
}


def num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def rows(value: Any) -> list[dict]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def parse_obs(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def clean_item(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", value.get("item", ""))
    if not isinstance(value, str):
        return ""
    value = value.removeprefix("minecraft:")
    return value if _ITEM.fullmatch(value) else ""


def point(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        value = [value.get(k) for k in ("x", "y", "z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    p = tuple(num(v, math.inf) for v in value[:3])
    return p if all(math.isfinite(v) for v in p) else None


def dist(a: Any, b: Any) -> float:
    pa, pb = point(a), point(b)
    if pa is None or pb is None:
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def inv_of(observation: dict) -> dict[str, int]:
    """Read only the adaptor's audited personal inventory."""
    source = observation.get("inventory")
    if source is None or source == {} or source == []:
        source = observation.get("inventory_authoritative", source)
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source["items"]
    answer: dict[str, int] = {}
    if isinstance(source, dict):
        for key, raw in source.items():
            item = clean_item(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(num(raw_count)))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    elif isinstance(source, list):
        for raw in rows(source):
            item = clean_item(raw)
            count = max(0, int(num(raw.get("count", 0))))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    return answer


def block_name(record: dict) -> str:
    return clean_item(record.get("name", ""))


def candidate_id(record: dict, prefix: str | None = None) -> str:
    value = record.get("id")
    if not isinstance(value, str) or not _ID.fullmatch(value):
        return ""
    if prefix and not value.startswith(prefix + ":"):
        return ""
    return value


def position_of(observation: dict) -> tuple[float, float, float] | None:
    return point(obj(observation.get("self")).get("position"))


def count_wood(inventory: dict[str, int]) -> int:
    return sum(v for k, v in inventory.items() if k.endswith("_log") or k.endswith("_stem"))


def count_planks(inventory: dict[str, int]) -> int:
    return sum(v for k, v in inventory.items() if k.endswith("_planks"))


def has_pick(inventory: dict[str, int], stone_or_better: bool = False) -> bool:
    names = ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")
    if stone_or_better:
        return any(inventory.get(k, 0) for k in names)
    return any(inventory.get(k, 0) for k in ("wooden_pickaxe",) + names)


def best_food(inventory: dict[str, int]) -> str:
    order = (
        "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
        "bread", "baked_potato", "carrot", "apple", "sweet_berries", "beef",
        "porkchop", "mutton", "chicken", "rabbit", "cod", "salmon", "potato",
    )
    return next((x for x in order if inventory.get(x, 0)), "")


def food_count(inventory: dict[str, int]) -> int:
    return sum(v for k, v in inventory.items() if k in FOODS)


def signature(inventory: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(inventory.items()))


def progress_signature(observation: dict) -> str:
    p = obj(observation.get("progress"))
    milestones = obj(p.get("milestones"))
    advancements = observation.get("advancements")
    return json.dumps([sorted((str(k), bool(v)) for k, v in milestones.items()), advancements], sort_keys=True, default=str)


def current_task(observation: dict) -> bool:
    task = observation.get("task")
    return task is not None and task != "" and task != {}


def is_solid_support(record: dict) -> bool:
    name = block_name(record)
    if name in _DANGEROUS_BLOCKS or name in {"air", "water", "bubble_column", "lava"}:
        return False
    return name in _SUPPORTS


class MilestoneExecutor:
    """Owns exact recipes, target binding, retries, and a small exploration ledger."""

    def __init__(self, ctx: Any, names: set[str]):
        self.ctx = ctx
        self.names = names
        self.active: dict | None = None
        self.failed_until: dict[str, float] = {}
        self.visits: dict[str, int] = {}
        self.seen_targets: set[str] = set()
        self.last_inventory: tuple[tuple[str, int], ...] = ()
        self.last_progress = ""
        self.last_change_time = 0.0
        self.last_post_time = -100.0
        self.last_posted = "noop"
        self.dead_sent = False
        # Placement is asynchronous: the inventory can update before the new
        # block appears in the bounded candidate list.  These expectations
        # prevent an immediate duplicate workstation craft.
        self.table_expected_until = -1.0
        self.furnace_expected_until = -1.0
        self.table_placed = False
        self.furnace_placed = False
        self.furnace_started: dict[str, float] = {}

    def log(self, event: str, **fields: Any) -> None:
        self.ctx.log(event, **fields)

    def permitted(self, action: str) -> bool:
        verb = action.split(" ", 1)[0] if action else ""
        return verb in self.names or (verb == "noop" and not self.names)

    def blocked(self, action: str, now: float) -> bool:
        return now < self.failed_until.get(action, -1.0)

    def record_observation(self, observation: dict, now: float) -> None:
        # Block discovery is also progress for placement operations.
        if self._tables(observation):
            self.table_expected_until = -1.0
        if any(block_name(x) == "furnace" for x in self._records(observation, "b")):
            self.furnace_expected_until = -1.0
        inventory = inv_of(observation)
        progress = progress_signature(observation)
        changed = signature(inventory) != self.last_inventory or progress != self.last_progress
        if changed:
            self.last_inventory = signature(inventory)
            self.last_progress = progress
            self.last_change_time = now
            if self.active:
                active_action = str(self.active.get("action", ""))
                if active_action.startswith("furnace_take "):
                    self.furnace_started.pop(active_action.split()[1], None)
                self.log("operation_progress", action=active_action)
                self.active = None
        if self.active is None:
            return
        action = str(self.active.get("action", ""))
        age = now - num(self.active.get("started"), now)
        before_pos = self.active.get("position")
        moved = dist(before_pos, position_of(observation)) > 1.5
        if action.startswith("navigate ") or action.startswith("goto "):
            if moved:
                self.active = None
                return
        # A task is allowed to run to the adaptor's own deadline.  If it has
        # ended without evidence of progress, quarantine this exact command.
        limits = 31.0 if action.startswith("gather ") else 18.0 if action.startswith(("navigate ", "goto ")) else 7.0
        if age > limits:
            self.failed_until[action] = now + 45.0
            self.log("operation_quarantined", action=action, age_s=round(age, 2))
            self.active = None

    def post(self, action: str, observation: dict, now: float) -> bool:
        if not action or not self.permitted(action):
            action = "noop"
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("action_post_error", action=action, error=type(exc).__name__)
            action = "noop"
        self.last_posted = action
        self.last_post_time = now
        if action == "cancel":
            self.active = None
        elif action != "noop":
            if action.startswith("place crafting_table "):
                self.table_expected_until = now + 12.0
                self.table_placed = True
            if action.startswith("place furnace "):
                self.furnace_expected_until = now + 12.0
                self.furnace_placed = True
            if action.startswith("furnace_start "):
                fields = action.split()
                if len(fields) > 1:
                    self.furnace_started[fields[1]] = now
            # Attacks are explicitly repeatable and shield-on is a held
            # defensive control.  Neither should be followed by a noop lease:
            # the realtime clock repeats the last posted action and another
            # noop would release the shield.
            if not action.startswith("attack ") and action != "shield on":
                self.active = {
                    "action": action,
                    "started": now,
                    "inventory": signature(inv_of(observation)),
                    "progress": progress_signature(observation),
                    "position": position_of(observation),
                }
            self.log("compiled_action", action=action)
        return action == "noop"

    def _records(self, observation: dict, prefix: str | None = None) -> list[dict]:
        records = []
        for item in rows(observation.get("candidates")):
            if candidate_id(item, prefix):
                records.append(item)
        records.sort(key=lambda x: num(x.get("distance"), math.inf))
        return records

    def _tables(self, observation: dict) -> list[dict]:
        return [x for x in self._records(observation, "b") if block_name(x) == "crafting_table"]

    def _near_table(self, observation: dict) -> dict | None:
        tables = [x for x in self._tables(observation) if x.get("in_reach") and num(x.get("distance"), 99) <= 5]
        return tables[0] if tables else None

    def _stand_for(self, observation: dict, target: dict | None, now: float) -> str:
        stands = self._records(observation, "s")
        target_position = target.get("position") if target else None
        here = position_of(observation)
        fresh_exists = bool(target is None and any(self.visits.get(candidate_id(s, "s"), 0) == 0 for s in stands))
        ranked = []
        for stand in stands:
            sid = candidate_id(stand, "s")
            if not sid or self.blocked("navigate " + sid, now):
                continue
            # One failed approach per listed stand is enough before the
            # resource search changes context.  Without this bound an exposed
            # but water-trapped log can monopolize every subsequent decision.
            if target is not None and self.visits.get(sid, 0) >= 1:
                continue
            if fresh_exists and target is None and self.visits.get(sid, 0) >= 1:
                continue
            if num(stand.get("distance"), 999) > 48:
                continue
            if target_position is not None and num(stand.get("distance"), 99) < 1.25:
                # This is usually the current feet candidate.  Reusing it
                # cannot make an out-of-reach target reachable.
                continue
            if target_position is not None:
                closeness = dist(stand.get("position"), target_position)
                # A stand well away from a resource is exploration, not a
                # useful approach; keep it as a fallback only.
                rank = (0 if closeness <= 7 else 1, closeness, self.visits.get(sid, 0), num(stand.get("distance"), 999))
            else:
                # Prefer a new, reasonably distant safe stand to force the
                # adaptor to reveal another local slice of terrain.
                rank = (self.visits.get(sid, 0), -min(num(stand.get("distance"), 0), 42), dist(stand.get("position"), here))
            ranked.append((rank, sid))
        if target is None and stands and not fresh_exists:
            return ""
        if not ranked:
            return ""
        ranked.sort(key=lambda pair: pair[0])
        sid = ranked[0][1]
        self.visits[sid] = self.visits.get(sid, 0) + 1
        return "navigate " + sid

    def _resource(self, observation: dict, now: float, predicate, count: int = 1) -> tuple[str, dict] | None:
        here = position_of(observation)
        choices = []
        for block in self._records(observation, "b"):
            name = block_name(block)
            bid = candidate_id(block, "b")
            if not bid or not predicate(name) or block.get("harvestable") is False:
                continue
            if name in _DANGEROUS_BLOCKS or dist(block.get("position"), here) > 48:
                continue
            if here is not None and point(block.get("position")) == (math.floor(here[0]), math.floor(here[1]) - 1, math.floor(here[2])):
                continue
            command = "gather %s %d" % (bid, max(1, min(4, int(count))))
            if self.blocked(command, now):
                continue
            # Visible and nearer candidates are preferable, but a resource
            # that was not previously attempted gets a deterministic bonus.
            rank = (0 if block.get("visible") else 1, 0 if block.get("in_reach") else 1,
                    0 if bid not in self.seen_targets else 1, num(block.get("distance"), 999))
            choices.append((rank, command, block))
        if not choices:
            return None
        choices.sort(key=lambda x: x[0])
        _, command, block = choices[0]
        self.seen_targets.add(candidate_id(block, "b"))
        if block.get("in_reach") or num(block.get("distance"), 99) <= 4.7 or (block.get("visible") and num(block.get("distance"), 99) <= 12):
            # The adaptor's gather primitive can safely path to a visible
            # exposed block.  Preserve this fast path for ordinary trees;
            # invisible targets (the failed swamp case) use a stand first.
            return command, block
        approach = self._stand_for(observation, block, now)
        return (approach, block) if approach else (command, block)

    def _explore(self, observation: dict, now: float) -> str:
        action = self._stand_for(observation, None, now)
        return action or "wait 1"

    def _support(self, observation: dict) -> dict | None:
        here = position_of(observation)
        if here is None:
            return None
        floor = (math.floor(here[0]), math.floor(here[1]), math.floor(here[2]))
        candidates = []
        for block in self._records(observation, "b"):
            bp = point(block.get("position"))
            if not bp or not block.get("in_reach") or not block.get("visible"):
                continue
            if int(bp[1]) != floor[1] - 1:
                continue
            if int(bp[0]) == floor[0] and int(bp[2]) == floor[2]:
                continue
            if not is_solid_support(block):
                continue
            candidates.append(block)
        candidates.sort(key=lambda x: num(x.get("distance"), 99))
        return candidates[0] if candidates else None

    def _place_or_approach(self, observation: dict, item: str, now: float) -> str:
        support = self._support(observation)
        if support:
            return "place %s %s up" % (item, candidate_id(support, "b"))
        return self._explore(observation, now)

    def _table_action(self, observation: dict, inventory: dict[str, int], now: float) -> str | None:
        near = self._near_table(observation)
        if near:
            return None
        tables = self._tables(observation)
        if tables:
            table = tables[0]
            action = "navigate " + candidate_id(table, "b")
            return "noop" if self.blocked(action, now) else action
        if inventory.get("crafting_table", 0):
            return self._place_or_approach(observation, "crafting_table", now)
        if self.table_placed:
            return "wait 1" if self.table_expected_until > now else self._explore(observation, now)
        if count_planks(inventory) >= 4:
            return "craft crafting_table 1"
        logs = next((k for k, v in inventory.items() if v and (k.endswith("_log") or k.endswith("_stem"))), "")
        if logs:
            return "craft %s 4" % logs.replace("_log", "_planks").replace("_stem", "_planks")
        return None

    def _gather_action(self, observation: dict, inventory: dict[str, int], now: float, mode: str) -> str | None:
        if mode == "wood":
            found = self._resource(observation, now, lambda n: n.endswith("_log") or n.endswith("_stem"), count=3)
        elif mode == "stone":
            found = self._resource(observation, now, lambda n: n in {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}, count=4)
        elif mode == "iron":
            found = self._resource(observation, now, lambda n: n in {"iron_ore", "deepslate_iron_ore", "raw_iron", "iron_block"}, count=3)
        elif mode == "food_block":
            found = self._resource(observation, now, lambda n: n in {"hay_block", "melon", "sweet_berry_bush", "wheat", "carrots", "potatoes", "beetroots"})
        elif mode == "bed":
            found = self._resource(observation, now, lambda n: n.endswith("_bed"))
        else:
            found = None
        return found[0] if found else None

    def _animal_action(self, observation: dict, inventory: dict[str, int], now: float) -> str | None:
        entities = []
        for entity in rows(observation.get("entities")):
            name = clean_item(entity.get("name", ""))
            if name in _ANIMALS and entity.get("visible") and not entity.get("dead"):
                entities.append(entity)
        entities.sort(key=lambda x: num(x.get("distance"), 99))
        if not entities:
            return None
        target = entities[0]
        eid = target.get("id")
        if eid is None:
            return None
        weapon = next((x for x in ("stone_sword", "iron_sword", "stone_axe", "wooden_sword", "wooden_axe") if inventory.get(x)), "")
        if not weapon:
            return None
        if num(target.get("distance"), 99) <= 3.2:
            action = "attack " + str(eid)
            if now - self.last_post_time < 0.9:
                return "noop"
            return action
        # Use a listed safe stand near the animal rather than goto'ing an
        # arbitrary entity coordinate.
        return self._stand_for(observation, target, now)

    def phase(self, observation: dict, inventory: dict[str, int]) -> str:
        milestones = obj(obj(observation.get("progress")).get("milestones"))
        if not count_wood(inventory) and not count_planks(inventory) and not milestones.get("wood_and_workbench"):
            return "wood"
        if not milestones.get("wood_and_workbench") and not self._tables(observation) and not inventory.get("crafting_table"):
            return "table"
        if not has_pick(inventory):
            return "wood_tools"
        if not has_pick(inventory, True):
            return "stone"
        if food_count(inventory) == 0 and not milestones.get("food"):
            return "food"
        if not inventory.get("iron_ingot") and not milestones.get("iron"):
            return "iron"
        if not inventory.get("shield") and not milestones.get("shield"):
            return "shield"
        if not milestones.get("bed") and not any(k.endswith("_bed") for k in inventory):
            return "bed"
        return "consolidate"

    def compile(self, observation: dict, now: float, focus: str = "balanced") -> tuple[str, str]:
        inventory = inv_of(observation)
        me = obj(observation.get("self"))
        emergency_food = best_food(inventory)
        if emergency_food and (num(me.get("health"), 20) <= 8 or num(me.get("food"), 20) <= 6):
            held = clean_item(me.get("held_item"))
            if held != emergency_food:
                return "equip %s hand" % emergency_food, "survival"
            return "eat " + emergency_food, "survival"
        phase = self.phase(observation, inventory)
        near_table = self._near_table(observation)

        if phase in {"wood", "table"}:
            action = self._table_action(observation, inventory, now)
            if action:
                return action, phase
            action = self._gather_action(observation, inventory, now, "wood")
            if action:
                return action, phase
            return self._explore(observation, now), phase

        if phase == "wood_tools":
            logs = next((k for k, v in inventory.items() if v and (k.endswith("_log") or k.endswith("_stem"))), "")
            planks = count_planks(inventory)
            if planks < 6:
                if logs:
                    output = 8 if inventory.get(logs, 0) >= 2 else 4
                    plank_name = logs.replace("_log", "_planks").replace("_stem", "_planks")
                    action = "craft %s %d" % (plank_name, output)
                    if not self.blocked(action, now):
                        return action, phase
                wood_action = self._gather_action(observation, inventory, now, "wood")
                if wood_action:
                    return wood_action, phase
            if inventory.get("stick", 0) < 2 and planks >= 2:
                return "craft stick 4", phase
            if not has_pick(inventory) and planks >= 3 and inventory.get("stick", 0) >= 2:
                if near_table:
                    return "craft wooden_pickaxe 1 %s" % candidate_id(near_table, "b"), phase
                table_action = self._table_action(observation, inventory, now)
                if table_action:
                    return table_action, phase
            if has_pick(inventory):
                return self._gather_action(observation, inventory, now, "stone") or self._explore(observation, now), phase
            return self._explore(observation, now), phase

        if phase == "stone":
            if inventory.get("cobblestone", 0) + inventory.get("cobbled_deepslate", 0) < 3:
                return self._gather_action(observation, inventory, now, "stone") or self._explore(observation, now), phase
            if inventory.get("stick", 0) < 2:
                if count_planks(inventory) >= 2:
                    return "craft stick 4", phase
                return self._explore(observation, now), phase
            if not has_pick(inventory, True):
                if near_table:
                    return "craft stone_pickaxe 1 %s" % candidate_id(near_table, "b"), phase
                return self._table_action(observation, inventory, now) or self._explore(observation, now), phase

        if phase == "food":
            near = self._near_table(observation)
            weapon = any(inventory.get(x) for x in ("stone_sword", "iron_sword", "wooden_sword", "stone_axe", "wooden_axe"))
            cobble = inventory.get("cobblestone", 0) + inventory.get("cobbled_deepslate", 0)
            if not weapon and cobble >= 2 and inventory.get("stick", 0):
                if near:
                    return "craft stone_sword 1 %s" % candidate_id(near, "b"), phase
                return self._table_action(observation, inventory, now) or self._explore(observation, now), phase
            if num(obj(observation.get("self")).get("food"), 20) <= 14:
                food = best_food(inventory)
                if food:
                    held = clean_item(obj(observation.get("self")).get("held_item"))
                    if held != food:
                        return "equip %s hand" % food, phase
                    return "eat " + food, phase
            near = self._near_table(observation)
            if inventory.get("hay_block") and near:
                return "craft wheat 9 %s" % candidate_id(near, "b"), phase
            if inventory.get("wheat", 0) >= 3 and near:
                return "craft bread %d %s" % (min(3, inventory["wheat"] // 3), candidate_id(near, "b")), phase
            action = self._gather_action(observation, inventory, now, "food_block")
            if action:
                return action, phase
            action = self._animal_action(observation, inventory, now)
            if action:
                return action, phase
            return self._explore(observation, now), phase

        if phase == "iron":
            if inventory.get("raw_iron", 0) < 3 and not inventory.get("iron_ingot"):
                return self._gather_action(observation, inventory, now, "iron") or self._explore(observation, now), phase
            # Furnace is made as soon as the ore and stone are available.
            furnaces = [x for x in self._records(observation, "b") if block_name(x) == "furnace"]
            cobble = inventory.get("cobblestone", 0) + inventory.get("cobbled_deepslate", 0)
            if cobble < 8:
                return self._gather_action(observation, inventory, now, "stone") or self._explore(observation, now), phase
            if not inventory.get("furnace") and not furnaces:
                if inventory.get("cobblestone", 0) + inventory.get("cobbled_deepslate", 0) >= 8 and near_table:
                    return "craft furnace 1 %s" % candidate_id(near_table, "b"), phase
                if not near_table:
                    return self._table_action(observation, inventory, now) or self._explore(observation, now), phase
            if not furnaces and inventory.get("furnace"):
                if self.furnace_expected_until > now:
                    return "wait 1", phase
                if self.furnace_placed:
                    return self._explore(observation, now), phase
                return self._place_or_approach(observation, "furnace", now), phase
            if furnaces:
                furnace = furnaces[0]
                fid = candidate_id(furnace, "b")
                raw = inventory.get("raw_iron", 0)
                finished = self.furnace_started.get(fid)
                if finished is not None and now - finished > 12:
                    take = "furnace_take " + fid
                    if not self.blocked(take, now):
                        return take, phase
                fuel = next((x for x in ("coal", "charcoal") + tuple(k for k in inventory if k.endswith("_planks")) if inventory.get(x)), "")
                if raw and fuel and finished is None:
                    start = "furnace_start %s raw_iron %d %s %d" % (fid, min(3, raw), fuel, min(3, inventory[fuel]))
                    if not self.blocked(start, now):
                        return start, phase
                if not raw and not inventory.get("iron_ingot") and now - self.last_change_time > 20:
                    return self._explore(observation, now), phase
            if inventory.get("iron_ingot"):
                return "noop", phase

        if phase == "shield":
            if count_planks(inventory) < 6:
                logs = next((k for k, v in inventory.items() if v and (k.endswith("_log") or k.endswith("_stem"))), "")
                if logs:
                    output = 8 if inventory.get(logs, 0) >= 2 else 4
                    return "craft %s %d" % (logs.replace("_log", "_planks").replace("_stem", "_planks"), output), phase
                return self._gather_action(observation, inventory, now, "wood") or self._explore(observation, now), phase
            if not near_table:
                return self._table_action(observation, inventory, now) or self._explore(observation, now), phase
            if inventory.get("iron_ingot", 0):
                return "craft shield 1 %s" % candidate_id(near_table, "b"), phase
            return self._explore(observation, now), phase

        if phase == "bed":
            wool = next((x for x, v in inventory.items() if x.endswith("_wool") and v >= 3), "")
            if wool and count_planks(inventory) >= 3 and near_table:
                color = wool.removesuffix("_wool")
                return "craft %s_bed 1 %s" % (color, candidate_id(near_table, "b")), phase
            action = self._gather_action(observation, inventory, now, "bed")
            if action:
                return action, phase
            if not wool:
                action = self._animal_action(observation, inventory, now)
                if action:
                    return action, phase
            return self._explore(observation, now), phase

        # Once the local opening is complete, keep the player alive and search
        # for explicitly exposed next-step resources rather than blindly mining.
        if food_count(inventory) and num(obj(observation.get("self")).get("food"), 20) < 12:
            food = best_food(inventory)
            held = clean_item(obj(observation.get("self")).get("held_item"))
            return (("equip %s hand" % food) if held != food else "eat " + food), phase
        if focus == "food" and food_count(inventory) < 5:
            return self._gather_action(observation, inventory, now, "food_block") or self._explore(observation, now), phase
        if focus == "iron" and not inventory.get("iron_ingot"):
            return self._gather_action(observation, inventory, now, "iron") or self._explore(observation, now), phase
        return self._explore(observation, now), phase


class ThreatGate:
    """A small Jev interface for immediate hazards only."""

    def __init__(self, ctx: Any, names: set[str]):
        self.ctx = ctx
        self.names = names
        self.future = None
        self.request_time = -100.0
        self.signature = ""
        self.cached: tuple[str, float] | None = None

    def log(self, event: str, **fields: Any) -> None:
        self.ctx.log(event, **fields)

    def _threats(self, observation: dict) -> list[dict]:
        found = []
        for entity in rows(observation.get("entities")):
            name = clean_item(entity.get("name", ""))
            if not entity.get("visible") or not (entity.get("hostile") or entity.get("conditional_threat") or name in _HOSTILE_NAMES):
                continue
            if num(entity.get("distance"), 99) <= 8:
                found.append(entity)
        for block in rows(observation.get("candidates")):
            if block_name(block) in _DANGEROUS_BLOCKS and num(block.get("distance"), 99) <= 5:
                found.append({"name": block_name(block), "distance": block.get("distance"), "kind": "hazard"})
        me = obj(observation.get("self"))
        if me.get("is_in_lava") or me.get("fall_distance_estimate") or num(me.get("oxygen"), 20) < 6:
            found.append({"name": "self_hazard", "distance": 0, "kind": "hazard"})
        return found

    def options(self, observation: dict, executor: MilestoneExecutor, now: float) -> tuple[list[str], list[dict]]:
        threats = self._threats(observation)
        if not threats:
            return [], []
        inventory = inv_of(observation)
        options: list[str] = []
        if inventory.get("shield") and "shield" in self.names:
            options.append("shield on")
        if num(obj(observation.get("self")).get("food"), 20) < 8:
            food = best_food(inventory)
            if food:
                held = clean_item(obj(observation.get("self")).get("held_item"))
                options.append(("equip %s hand" % food) if held != food else "eat " + food)
        for entity in threats:
            if entity.get("kind") == "hazard":
                continue
            if entity.get("id") is not None and num(entity.get("distance"), 99) <= 3.2:
                options.append("attack " + str(entity["id"]))
        stand = executor._stand_for(observation, None, now)
        if stand:
            options.append(stand)
        options.append("noop")
        # Keep only verbs exposed by the game and preserve unique values.
        output = []
        for action in options:
            if action not in output and executor.permitted(action):
                output.append(action)
        return output[:8], threats

    def decide(self, observation: dict, executor: MilestoneExecutor, now: float) -> str | None:
        options, threats = self.options(observation, executor, now)
        if not options:
            self.cached = None
            return None
        sig = json.dumps([(x.get("name"), x.get("id"), x.get("distance")) for x in threats], sort_keys=True, default=str)
        if self.cached and self.signature == sig and now - self.request_time < 3.0:
            return self.cached[0] if self.cached[0] in options else "noop"
        # A new request is made only when the hazard set changes or its short
        # lease expires.  Code still vetoes answers that are no longer legal.
        criteria = {a: ("Immediate, executable response: " + a) for a in options}
        state = {
            "self": {
                "health_band": "critical" if num(obj(observation.get("self")).get("health"), 20) < 7 else "stable",
                "food_band": "low" if num(obj(observation.get("self")).get("food"), 20) < 8 else "fed",
                "dimension": obj(observation.get("self")).get("dimension"),
                "on_fire_or_lava": bool(obj(observation.get("self")).get("is_in_lava")),
            },
            "threats": [{"kind": clean_item(t.get("name", "")), "distance_band": "touching" if num(t.get("distance"), 99) <= 3 else "near"} for t in threats],
            "legal_responses": options,
        }
        try:
            answers = self.ctx.decide(
                state,
                {"response": Choice(
                    "Choose one legal immediate response. Prefer survival over work. "
                    "Select exactly one value from legal_responses; do not invent an id. "
                    "Use shield or retreat for an uncertain hostile, and never attack a hazard.",
                    criteria,
                )},
                timeout_s=2.0,
            )
            answer = answers.get("response") if isinstance(answers, dict) else None
            value = getattr(answer, "value", None)
            probability = obj(getattr(answer, "probabilities", {})).get(value, 0.0)
            if not isinstance(value, str) or value not in options:
                value = "noop"
            # Low-confidence answers are not allowed to choose an irreversible
            # attack; a shield/retreat/noop is the safe lease.
            if num(probability, 0.0) < 0.55 and (value.startswith("attack ") or value.startswith("goto ") or value.startswith("navigate ")):
                value = "shield on" if "shield on" in options else "noop"
            self.cached = (value, num(probability, 0.0))
            self.signature = sig
            self.request_time = now
            self.log("jev_threat", response=value, probability=num(probability, 0.0), threat_count=len(threats))
            return value
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
            self.cached = ("shield on" if "shield on" in options else "noop", 0.0)
            self.signature = sig
            self.request_time = now
            self.log("jev_threat_fallback", error=type(exc).__name__)
            return self.cached[0]


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.names: set[str] = set()
        self.executor: MilestoneExecutor | None = None
        self.threats: ThreatGate | None = None
        self.focus = "balanced"
        self.astra_future = None
        self.astra_started = -100.0
        self.astra_finished = -100.0

    def log(self, event: str, **fields: Any) -> None:
        self.ctx.log(event, **fields)

    def _start_adviser(self, observation: dict, phase: str, now: float) -> None:
        if self.astra_future is not None and not self.astra_future.done():
            return
        remaining = self.ctx.budget.remaining()
        if now - self.astra_started < 180 or now - self.astra_finished < 45:
            return
        if self.executor is not None and now - self.executor.last_change_time < 45:
            return
        if remaining.get("system2_calls", 0) <= 0 or remaining.get("wall_s", 0) < 260:
            return
        inventory = inv_of(observation)
        blocks = [
            {"name": block_name(b), "distance": round(num(b.get("distance"), 99), 1), "reachable": bool(b.get("in_reach"))}
            for b in rows(observation.get("candidates"))
            if block_name(b) in {"iron_ore", "deepslate_iron_ore", "hay_block", "wheat", "carrots", "potatoes"} or block_name(b).endswith("_bed")
        ][:18]
        prompt = (
            "You are a slow macro adviser for vanilla Minecraft survival. "
            "Code owns recipes, coordinates, legality, retries, and all commands. "
            "Choose one focus token only: balanced, food, iron, bed, or explore. "
            "Choose food if no food is held, iron if the stone pick and iron are the bottleneck, "
            "bed if a bed or wool is observed, otherwise balanced. Never output a command.\n" +
            json.dumps({"phase": phase, "focus_now": self.focus, "inventory": inventory, "nearby_useful": blocks,
                        "health": obj(observation.get("self")).get("health"),
                        "food": obj(observation.get("self")).get("food"),
                        "dimension": obj(observation.get("self")).get("dimension")}, separators=(",", ":"))
        )
        self.astra_started = now
        try:
            self.astra_future = self.ctx.spawn(lambda p=prompt: self.ctx.think(p, timeout_s=110.0))
            self.log("astra_started", phase=phase)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("astra_start_failed", error=type(exc).__name__)
            self.astra_future = None

    def _poll_adviser(self, now: float) -> None:
        if self.astra_future is None or not self.astra_future.done():
            return
        future = self.astra_future
        self.astra_future = None
        self.astra_finished = now
        try:
            result = future.result()
            text = str(getattr(result, "text", "")).lower()
            # Parse a token, never a command or coordinate.
            for token in ("food", "iron", "bed", "explore", "balanced"):
                if re.search(r"\b" + token + r"\b", text):
                    self.focus = token
                    self.log("astra_focus", focus=token)
                    return
            self.log("astra_unusable", text_length=len(text))
        except (BudgetExceeded, ModelUnavailable):
            # A background call that reaches the episode boundary is not a
            # reason to corrupt the controller's deterministic fallback.
            self.log("astra_closed")
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)

    def _post_and_wait(self, action: str, observation: dict, now: float) -> None:
        assert self.executor is not None
        self.executor.post(action, observation, now)
        # A fresh noop is deliberately not posted here: one bounded job must
        # own the hands, and the realtime adaptor repeats the current action.
        try:
            self.ctx.wait_for_tick()
        except (BudgetExceeded, ModelUnavailable):
            raise

    def run_episode(self, game: Any) -> None:
        self.names = set(getattr(getattr(game, "action_spec", None), "names", ()) or ())
        self.executor = MilestoneExecutor(self.ctx, self.names)
        self.threats = ThreatGate(self.ctx, self.names)
        while not self.ctx.done:
            now = self.ctx.clock.now()
            observation = parse_obs(self.ctx.observe())
            if not observation:
                self._post_and_wait("noop", observation, now)
                continue
            self.executor.record_observation(observation, now)
            self._poll_adviser(now)
            me = obj(observation.get("self"))
            if me.get("dead"):
                if not self.executor.dead_sent:
                    self.executor.dead_sent = True
                    self._post_and_wait("respawn", observation, now)
                else:
                    self._post_and_wait("noop", observation, now)
                continue
            self.executor.dead_sent = False
            age = num(observation.get("observation_age_s"), 0.0)
            if age > 3.0:
                if current_task(observation):
                    self._post_and_wait("cancel", observation, now)
                else:
                    self._post_and_wait("noop", observation, now)
                continue
            # A live bounded task is kept alive by noop, but a nearby
            # immediate hazard is allowed to cancel it before damage lands.
            if current_task(observation):
                if self.threats._threats(observation):
                    tactical = self.threats.decide(observation, self.executor, now)
                    active_action = str((self.executor.active or {}).get("action", ""))
                    continuing_retreat = tactical and tactical == active_action and tactical.startswith("navigate ")
                    if (tactical and tactical != "noop" and not continuing_retreat or
                            (active_action and not continuing_retreat and not active_action.startswith("shield on"))) and self.executor.permitted("cancel"):
                        self._post_and_wait("cancel", observation, now)
                        continue
                self._post_and_wait("noop", observation, now)
                continue

            inventory = inv_of(observation)
            phase = self.executor.phase(observation, inventory)
            self._start_adviser(observation, phase, now)
            if self.executor.active is not None:
                # The prior one-shot has not yet shown an outcome.  Do not let
                # a normal compiler pass overwrite it; only an actual threat
                # can interrupt, and interruption is a cancel-then-replan.
                tactical = self.threats.decide(observation, self.executor, now)
                if tactical and tactical != "noop" and self.executor.permitted("cancel"):
                    self._post_and_wait("cancel", observation, now)
                else:
                    self._post_and_wait("noop", observation, now)
                continue
            tactical = self.threats.decide(observation, self.executor, now)
            if tactical:
                # Do not allow a stale tactical choice to replace a changed
                # threat menu; all values are regenerated above.
                self.log("tactical_action", action=tactical, phase=phase)
                self._post_and_wait(tactical, observation, now)
                continue

            action, phase = self.executor.compile(observation, now, self.focus)
            if self.executor.blocked(action, now):
                action = self.executor._explore(observation, now)
            if action != "noop" and self.executor.active is not None:
                # A completed one-shot needs a visible result before retrying.
                self._post_and_wait("noop", observation, now)
                continue
            self.log("phase_action", phase=phase, focus=self.focus, action=action)
            self._post_and_wait(action, observation, now)


def build(ctx: Any) -> Harness:
    return Harness(ctx)
