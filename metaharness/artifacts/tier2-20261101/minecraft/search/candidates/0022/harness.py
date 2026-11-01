"""Candidate 0022: carry the workstation into the mine.

The controller is deliberately transaction-oriented, but its workstation policy is
not the usual "return to the first table" policy.  It makes a spare crafting table
before leaving the surface and places that spare beside the first verified stone
work.  A furnace is treated the same way.  This turns a long, fragile route back to
the opening into a local interaction and lets the run continue after a successful
moving mine.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


ID_RE = re.compile(r"^[bs]:-?\d+,-?\d+,-?\d+$")

WOOD_SUFFIXES = ("_log", "_stem")
PICK_ORDER = (
    "netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe",
    "stone_pickaxe", "wooden_pickaxe",
)
STONE_PICKS = {"stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"}
WEAPONS = (
    "netherite_sword", "diamond_sword", "iron_sword", "stone_sword",
    "wooden_sword", "netherite_axe", "diamond_axe", "iron_axe",
    "stone_axe", "wooden_axe",
)
FOODS = {
    "apple", "bread", "baked_potato", "carrot", "golden_carrot",
    "sweet_berries", "melon_slice", "mushroom_stew", "beetroot",
    "potato", "cooked_beef", "cooked_porkchop", "cooked_mutton",
    "cooked_chicken", "cooked_rabbit", "cooked_cod", "cooked_salmon",
    "beef", "porkchop", "mutton", "chicken", "rabbit", "cod", "salmon",
    "wheat",
}
FOOD_BLOCKS = {
    "hay_block", "melon", "sweet_berry_bush", "wheat", "carrots",
    "potatoes", "beetroots",
}
STONE_BLOCKS = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON_BLOCKS = {"iron_ore", "deepslate_iron_ore"}
COAL_BLOCKS = {"coal_ore", "deepslate_coal_ore", "coal_block"}
GROUND_BLOCKS = {
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "mud", "clay", "stone", "cobblestone", "deepslate", "cobbled_deepslate",
    "tuff", "andesite", "diorite", "granite", "netherrack", "blackstone",
}
SUPPORT_BLOCKS = GROUND_BLOCKS | {
    "sand", "gravel", "terracotta", "sandstone", "red_sandstone",
    "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log",
    "dark_oak_log", "mangrove_log", "crimson_stem", "warped_stem",
    "crafting_table", "furnace", "chest",
}
HAZARD_BLOCKS = {
    "lava", "fire", "soul_fire", "magma_block", "cactus", "powder_snow",
    "campfire", "soul_campfire", "pointed_dripstone",
}
HOSTILES = {
    "zombie", "skeleton", "creeper", "spider", "drowned", "husk", "stray",
    "witch", "phantom", "enderman", "slime", "magma_cube", "blaze", "ghast",
    "piglin", "piglin_brute", "zombified_piglin", "hoglin", "wither_skeleton",
    "ender_dragon", "shulker", "silverfish", "warden",
}
FAIL_WORDS = (
    "failed", "failure", "error", "unreachable", "timeout", "cannot",
    "rejected", "illegal", "not enough", "not in reach", "no path", "busy",
    "invalid", "unknown", "out of range", "denied",
)
DEFAULT_ACTIONS = {
    "noop", "cancel", "navigate", "look_at", "gather", "equip", "craft", "place",
    "furnace_start", "furnace_take", "eat", "attack", "shield", "wait", "controls",
    "goto", "dig", "activate_block", "activate_entity", "use_item", "hotbar",
    "open_container", "close_window", "transfer", "move_slot", "toss", "sleep", "wake",
    "respawn", "inspect_trades", "trade", "enchant", "chat",
}


def num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def name_of(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("id", ""))))
    return str(value or "").removeprefix("minecraft:")


def point(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        if value.get("position") is not None:
            value = value.get("position")
        elif all(key in value for key in ("x", "y", "z")):
            value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result = tuple(num(part, math.inf) for part in value[:3])
    return result if all(math.isfinite(part) for part in result) else None


def id_position(identifier: Any) -> tuple[float, float, float] | None:
    if not isinstance(identifier, str) or not ID_RE.fullmatch(identifier):
        return None
    try:
        result = tuple(float(part) for part in identifier[2:].split(","))
        return result if len(result) == 3 and all(math.isfinite(part) for part in result) else None
    except (TypeError, ValueError):
        return None


def dist(a: Any, b: Any) -> float:
    pa, pb = point(a), point(b)
    if pa is None or pb is None:
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def observation(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def self_row(obs: dict) -> dict:
    value = obs.get("self")
    return value if isinstance(value, dict) else {}


def self_pos(obs: dict) -> tuple[float, float, float] | None:
    return point(self_row(obs).get("position"))


def rows(obs: dict, prefix: str | None = None) -> list[dict]:
    here = self_pos(obs)
    source = obs.get("candidates")
    result: list[dict] = []
    if not isinstance(source, list):
        return result
    for raw in source:
        if not isinstance(raw, dict):
            continue
        identifier = raw.get("id")
        if not isinstance(identifier, str) or not ID_RE.fullmatch(identifier):
            continue
        if prefix and not identifier.startswith(prefix + ":"):
            continue
        row = dict(raw)
        row["_id"] = identifier
        row["_name"] = name_of(raw.get("name", raw.get("block", raw.get("type", ""))))
        row["_pos"] = point(raw.get("position")) or id_position(identifier)
        row["_distance"] = num(raw.get("distance"), dist(here, row["_pos"]))
        result.append(row)
    result.sort(key=lambda row: row["_distance"])
    return result


def entities(obs: dict) -> list[dict]:
    source = obs.get("entities", obs.get("nearby_entities", []))
    here = self_pos(obs)
    result: list[dict] = []
    if not isinstance(source, list):
        return result
    for raw in source:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row["_name"] = name_of(raw.get("name", raw.get("type", "")))
        row["_pos"] = point(raw.get("position"))
        row["_distance"] = num(raw.get("distance"), dist(here, row["_pos"]))
        result.append(row)
    return result


def inventory(obs: dict) -> dict[str, int]:
    """Read personal inventory only, never a cursor, recipe preview, or container."""
    sources: list[Any] = []
    for key in ("inventory", "inventory_authoritative", "inventory_slots"):
        value = obs.get(key)
        if value not in (None, {}, []):
            sources.append(value)
    if isinstance(obs.get("inventory_client"), dict):
        sources.append(obs.get("inventory_client"))

    for source in sources:
        if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
            source = source.get("items")
        result: dict[str, int] = {}
        if isinstance(source, dict):
            for raw_name, raw_count in source.items():
                item = name_of(raw_name)
                count = raw_count.get("count", 0) if isinstance(raw_count, dict) else raw_count
                count = max(0, int(num(count)))
                if item and count:
                    result[item] = result.get(item, 0) + count
        elif isinstance(source, list):
            for raw in source:
                if not isinstance(raw, dict):
                    continue
                item = name_of(raw)
                count = max(0, int(num(raw.get("count", 0))))
                if item and count:
                    result[item] = result.get(item, 0) + count
        if result:
            return result
    return {}


def inv_sig(inv: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(key), int(value)) for key, value in inv.items() if int(value) > 0))


def action_verb(action: Any) -> str:
    return str(action or "").strip().split(" ", 1)[0]


def task_active(obs: dict) -> bool:
    value = obs.get("task")
    return value not in (None, "", {}, [])


def plank_total(inv: dict[str, int]) -> int:
    return sum(int(value) for key, value in inv.items() if key.endswith("_planks") and value)


def wood_total(inv: dict[str, int]) -> int:
    return sum(int(value) for key, value in inv.items() if key.endswith(WOOD_SUFFIXES) and value)


def food_total(inv: dict[str, int]) -> int:
    return sum(int(value) for key, value in inv.items() if key in FOODS and value)


def best_food(inv: dict[str, int]) -> str:
    order = (
        "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
        "bread", "baked_potato", "carrot", "apple", "sweet_berries",
        "melon_slice", "beef", "porkchop", "mutton", "chicken", "rabbit",
        "cod", "salmon", "potato", "wheat",
    )
    return next((item for item in order if inv.get(item)), "")


def plank_item_from_log(item: str) -> str:
    if item.endswith("_log"):
        return item[:-4] + "_planks"
    if item.endswith("_stem"):
        return item[:-5] + "_planks"
    return ""


def preferred_plank_item(inv: dict[str, int]) -> str:
    planks = [(key, value) for key, value in inv.items() if key.endswith("_planks") and value]
    if planks:
        return max(planks, key=lambda pair: pair[1])[0]
    logs = [(plank_item_from_log(key), value) for key, value in inv.items() if key.endswith(WOOD_SUFFIXES) and value]
    return max(logs, key=lambda pair: pair[1])[0] if logs else ""


def hostile(row: dict) -> bool:
    return bool(row.get("hostile") or row.get("conditional_threat") or row.get("_name") in HOSTILES)


def fail_value(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("ok") is False or value.get("success") is False:
            return True
        text = str(value.get("status", value.get("result", ""))).lower()
        if any(word in text for word in FAIL_WORDS):
            return True
        for key in ("message", "feedback", "reason", "error"):
            nested = value.get(key)
            if isinstance(nested, str) and any(word in nested.lower() for word in FAIL_WORDS):
                return True
    elif isinstance(value, str):
        text = value.lower()
        return any(word in text for word in FAIL_WORDS) and not any(word in text for word in ("success", "succeeded", " ok"))
    return False


class JevGuard:
    """Fast interrupt policy; code still vetoes lethal or stale actions."""

    def __init__(self, ctx: Any, allowed: set[str], log) -> None:
        self.ctx = ctx
        self.allowed = allowed
        self.log = log
        self.future = None
        self.signature: tuple = ()
        self.started = -100.0
        self.cached: tuple[str, tuple, float] | None = None

    def can(self, action: str) -> bool:
        return action_verb(action) in self.allowed or action == "noop"

    def hazard_list(self, obs: dict) -> list[dict]:
        me = self_row(obs)
        hazards: list[dict] = []
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            hazards.append({"kind": "lava_or_fire", "id": "self", "range": "touching"})
        if me.get("is_in_water") and num(me.get("oxygen"), 20) < 7:
            hazards.append({"kind": "drowning", "id": "self", "range": "touching"})
        if num(me.get("fall_distance_estimate"), 0) > 3:
            hazards.append({"kind": "fall", "id": "self", "range": "touching"})
        for row in rows(obs, "b"):
            if row["_name"] in HAZARD_BLOCKS and row["_distance"] <= 4.5:
                hazards.append({
                    "kind": "lava_or_fire" if row["_name"] in {"lava", "fire", "soul_fire"} else row["_name"],
                    "id": row["_id"], "range": "near",
                })
        for row in entities(obs):
            if hostile(row) and row.get("visible") is not False and row["_distance"] <= 8.0:
                hazards.append({
                    "kind": row["_name"], "id": str(row.get("id", "")),
                    "range": "touching" if row["_distance"] <= 3.2 else "near",
                })
        return hazards

    def fallback(self, obs: dict, hazards: list[dict], inv: dict[str, int]) -> str:
        me = self_row(obs)
        if any(item["kind"] == "lava_or_fire" for item in hazards):
            if any(item.get("id") == "self" for item in hazards):
                return "controls forward,jump" if self.can("controls") else "cancel"
            return "controls back,jump" if self.can("controls") else "cancel"
        if any(item["kind"] == "drowning" for item in hazards):
            return "controls jump" if self.can("controls") else "cancel"
        if num(me.get("health"), 20) <= 6 or num(me.get("food"), 20) <= 5:
            food = best_food(inv)
            if food:
                if name_of(me.get("held_item")) != food and self.can("equip"):
                    return "equip %s hand" % food
                if self.can("eat"):
                    return "eat " + food
        weapon = next((item for item in WEAPONS if inv.get(item)), "")
        close = [item for item in hazards if item.get("id") != "self" and item.get("range") == "touching"]
        if close and weapon:
            if name_of(me.get("held_item")) != weapon and self.can("equip"):
                return "equip %s hand" % weapon
            if self.can("attack") and close[0].get("id"):
                return "attack " + str(close[0]["id"])
        if inv.get("shield") and self.can("shield"):
            return "shield on"
        return "controls back,jump" if self.can("controls") else "cancel"

    def options(self, obs: dict, hazards: list[dict], inv: dict[str, int]) -> list[str]:
        answer: list[str] = []
        fallback = self.fallback(obs, hazards, inv)
        if fallback and (self.can(fallback) or fallback == "noop"):
            answer.append(fallback)
        if inv.get("shield") and self.can("shield"):
            answer.append("shield on")
        weapon = next((item for item in WEAPONS if inv.get(item)), "")
        for hazard in hazards:
            if hazard.get("id") == "self":
                continue
            if hazard.get("range") == "touching" and weapon and hazard.get("id"):
                if name_of(self_row(obs).get("held_item")) != weapon and self.can("equip"):
                    answer.append("equip %s hand" % weapon)
                answer.append("attack " + str(hazard["id"]))
        if task_active(obs) and self.can("cancel"):
            answer.append("cancel")
        answer.append("noop")
        return list(dict.fromkeys(item for item in answer if self.can(item) or item == "noop"))[:8]

    def choose(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        hazards = self.hazard_list(obs)
        if not hazards:
            self.signature = ()
            self.cached = None
            return None
        options = self.options(obs, hazards, inv)
        if not options:
            return None
        signature = tuple(sorted((str(item["kind"]), str(item.get("id", "")), str(item["range"])) for item in hazards))
        if self.cached and self.cached[1] == signature and now - self.started < 2.0:
            return self.cached[0] if self.cached[0] in options else options[0]
        if self.future is None and self.ctx.budget.remaining().get("system1_calls", 0) > 2:
            me = self_row(obs)
            state = {
                "purpose": "Immediate survival interrupt. Select one exact supplied action; never invent an ID.",
                "self": {
                    "health_band": "critical" if num(me.get("health"), 20) < 7 else "low" if num(me.get("health"), 20) < 12 else "stable",
                    "food_band": "low" if num(me.get("food"), 20) < 8 else "fed",
                    "dimension": str(me.get("dimension", "")),
                    "in_lava_or_fire": bool(me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire")),
                },
                "hazards": hazards[:6],
                "legal_actions": options,
                "rule": "Choose defense or retreat before work. A supplied action is the only legal answer.",
            }
            questions = {
                "response": Choice(
                    "Which exact supplied action best preserves survival right now? Do not create a command.",
                    {action: "Execute exactly: " + action for action in options},
                ),
                "safe_to_continue": Noul(
                    "Is it safe to continue the current bounded work without a defensive action?",
                    {
                        "true": "No listed hazard requires interruption before work.",
                        "false": "A listed hazard or close hostile requires defense, retreat, or interruption.",
                    },
                ),
            }
            try:
                self.future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=1.3))
                self.signature = signature
                self.started = now
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self.log("jev_spawn_failed", error=type(exc).__name__)
                self.future = None
        if self.future is not None and self.future.done():
            future = self.future
            self.future = None
            try:
                result = future.result()
                answer = result.get("response") if isinstance(result, dict) else None
                value = getattr(answer, "value", None)
                probs = getattr(answer, "probabilities", {}) or {}
                probability = num(probs.get(value), 0.0) if isinstance(probs, dict) else 0.0
                if value in options and self.signature == signature and probability >= 0.50:
                    self.cached = (value, signature, probability)
                    self.log("jev_interrupt", action=value, probability=round(probability, 3))
                    return value
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self.log("jev_invalid", error=type(exc).__name__)
        # This is intentionally immediate: a 1.3-second model answer must not
        # leave an adjacent creeper or lava in charge of the hands.
        return self.fallback(obs, hazards, inv)


class MobileController:
    """A local-workstation progression compiler with verified action transactions."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.allowed: set[str] = set()
        self.guard: JevGuard | None = None
        self.tx: dict | None = None
        self.job: dict | None = None
        self.failed_until: dict[str, float] = {}
        self.completed: set[str] = set()
        self.used_stands: dict[str, float] = {}
        self.visited_cells: set[tuple[int, int]] = set()
        self.last_posted = "noop"
        self.feedback_marker = ""
        self.table_latch: tuple[float, float, float] | None = None
        self.furnace_latch: tuple[float, float, float] | None = None
        self.smelt_started: dict[str, float] = {}
        self.manual_stage = "none"
        self.manual_until = -1.0
        self.manual_target: tuple[float, float, float] | None = None
        self.dead_seen = False
        self.last_mode = "opening"
        self.last_obs: dict = {}

    def log(self, event: str, **fields: Any) -> None:
        self.ctx.log(event, **fields)

    def now(self) -> float:
        return num(self.ctx.clock.now())

    def can(self, action: str) -> bool:
        return action == "noop" or action_verb(action) in self.allowed

    def blocked(self, key: str, now: float) -> bool:
        return bool(key) and now < self.failed_until.get(key, -1.0)

    def suppress(self, key: str, now: float, seconds: float, reason: str) -> None:
        if not key or key in {"noop", "cancel"}:
            return
        self.failed_until[key] = max(self.failed_until.get(key, -1.0), now + seconds)
        self.log("suppressed", key=key, seconds=seconds, reason=reason)

    def facts(self, obs: dict) -> tuple[dict, tuple[float, float, float] | None, dict[str, int], list[dict], list[dict]]:
        return self_row(obs), self_pos(obs), inventory(obs), rows(obs), entities(obs)

    def target_row(self, obs: dict, identifier: str) -> dict | None:
        return next((row for row in rows(obs, "b") if row["_id"] == identifier), None)

    def table_rows(self, obs: dict) -> list[dict]:
        return [row for row in rows(obs, "b") if row["_name"] == "crafting_table"]

    def furnace_rows(self, obs: dict) -> list[dict]:
        return [row for row in rows(obs, "b") if row["_name"] == "furnace"]

    def local_table(self, obs: dict) -> dict | None:
        """A nearby real table, or a short-lived placement latch for routing."""
        choices = [row for row in self.table_rows(obs) if row["_distance"] <= 4.6 and row.get("in_reach") is not False]
        choices.sort(key=lambda row: row["_distance"])
        if choices:
            return choices[0]
        if self.table_latch and self_pos(obs) is not None and dist(self_pos(obs), self.table_latch) <= 5.0:
            q = self.table_latch
            return {"_id": "b:%d,%d,%d" % (round(q[0]), round(q[1]), round(q[2])), "_name": "crafting_table", "_distance": dist(self_pos(obs), q), "_pos": q, "_synthetic": True}
        return None

    def actual_table(self, obs: dict) -> dict | None:
        """Return only an adaptor-listed table ID suitable for craft."""
        choices = [
            row for row in self.table_rows(obs)
            if row["_distance"] <= 4.6 and row.get("in_reach") is not False
        ]
        choices.sort(key=lambda row: row["_distance"])
        return choices[0] if choices else None

    def local_furnace(self, obs: dict) -> dict | None:
        choices = [row for row in self.furnace_rows(obs) if row["_distance"] <= 4.6 and row.get("in_reach") is not False]
        choices.sort(key=lambda row: row["_distance"])
        if choices:
            return choices[0]
        if self.furnace_latch and self_pos(obs) is not None and dist(self_pos(obs), self.furnace_latch) <= 5.0:
            q = self.furnace_latch
            return {"_id": "b:%d,%d,%d" % (round(q[0]), round(q[1]), round(q[2])), "_name": "furnace", "_distance": dist(self_pos(obs), q), "_pos": q, "_synthetic": True}
        return None

    def action_failed(self, obs: dict) -> bool:
        values = [obs.get(key) for key in ("action_feedback", "action_result", "last_result", "feedback")]
        return any(fail_value(value) for value in values)

    def read_feedback(self, obs: dict, now: float) -> None:
        try:
            marker = json.dumps([obs.get(key) for key in ("action_feedback", "action_result", "last_result", "feedback")], sort_keys=True, default=str)
        except Exception:
            marker = repr(obs.get("feedback"))
        if marker == self.feedback_marker:
            return
        self.feedback_marker = marker
        if self.tx and self.action_failed(obs):
            self.fail_transaction(now, "adapter_feedback")

    def timeout_for(self, action: str) -> float:
        verb = action_verb(action)
        if verb == "gather":
            return 34.0
        if verb in {"navigate", "goto"}:
            return 19.0
        if verb == "dig":
            return 18.0
        if verb == "look_at":
            return 2.0
        if verb in {"craft", "equip", "place", "furnace_start", "furnace_take", "eat", "hotbar"}:
            return 9.0
        return 7.0

    def begin_tx(self, action: str, obs: dict, now: float) -> None:
        parts = action.split()
        tx = {
            "action": action,
            "started": now,
            "before_inv": inv_sig(inventory(obs)),
            "before_pos": self_pos(obs),
            "before_dim": self_row(obs).get("dimension"),
        }
        verb = action_verb(action)
        if verb in {"gather", "dig"} and len(parts) >= 2:
            tx["target"] = parts[1]
            tx["target_pos"] = id_position(parts[1])
        if verb in {"navigate", "goto"} and len(parts) >= 2:
            tx["goal"] = parts[1:]
        if verb == "place" and len(parts) >= 3:
            tx["item"] = name_of(parts[1])
            tx["support"] = parts[2]
            support = id_position(parts[2])
            if support:
                tx["expected"] = (support[0], support[1] + 1.0, support[2])
        self.tx = tx

    def fail_transaction(self, now: float, reason: str) -> None:
        tx = self.tx
        if not tx:
            return
        action = str(tx.get("action", ""))
        verb = action_verb(action)
        self.suppress(action, now, 70.0 if verb in {"craft", "place", "equip"} else 110.0, reason)
        target = str(tx.get("target", ""))
        if target:
            self.suppress(target, now, 150.0 if verb in {"gather", "dig"} else 80.0, reason)
        if self.job and target and self.job.get("target") == target:
            self.job = None
        if verb == "place":
            if str(tx.get("item")) == "crafting_table":
                self.table_latch = None
            if str(tx.get("item")) == "furnace":
                self.furnace_latch = None
        self.log("transaction_failed", action=action, reason=reason)
        self.tx = None

    def tx_done(self, obs: dict, now: float) -> bool:
        tx = self.tx
        if not tx:
            return False
        action = str(tx.get("action", ""))
        verb = action_verb(action)
        inv = inventory(obs)
        before = tx.get("before_inv", ())
        pos = self_pos(obs)
        done = False
        if verb == "equip":
            wanted = name_of(action.split()[1] if len(action.split()) > 1 else "")
            done = name_of(self_row(obs).get("held_item")) == wanted
        elif verb in {"craft", "furnace_start", "furnace_take", "eat", "sleep"}:
            done = inv_sig(inv) != before
        elif verb == "place":
            item = str(tx.get("item", ""))
            before_count = dict(before).get(item, 0) if isinstance(before, dict) else 0
            # before is a signature tuple, so use a small local lookup instead.
            before_map = dict(before) if isinstance(before, tuple) else {}
            done = int(before_map.get(item, 0)) > int(inv.get(item, 0))
            if not done:
                expected = tx.get("expected")
                done = any(row["_name"] == item and expected is not None and dist(row.get("_pos"), expected) < 1.2 for row in rows(obs, "b"))
        elif verb in {"navigate", "goto"}:
            done = dist(tx.get("before_pos"), pos) >= 0.65 or self_row(obs).get("dimension") != tx.get("before_dim")
        elif verb == "gather":
            done = inv_sig(inv) != before
        elif verb == "dig":
            target = str(tx.get("target", ""))
            row = self.target_row(obs, target)
            if row is None:
                done = True
            changes = obs.get("block_changes")
            if isinstance(changes, list):
                target_pos = tx.get("target_pos")
                for change in changes:
                    if isinstance(change, dict) and dist(change.get("position"), target_pos) < 0.25:
                        if name_of(change.get("name")) in {"air", "water", "lava"} or change.get("name"):
                            done = True
            done = done or inv_sig(inv) != before
        elif verb == "look_at":
            done = now - num(tx.get("started"), now) >= 0.55
        elif verb in {"hotbar", "close_window", "wake"}:
            done = now - num(tx.get("started"), now) >= 0.4
        elif verb == "wait":
            done = now - num(tx.get("started"), now) >= 0.5
        if done:
            self.finish_transaction(obs, now)
            return True
        if not task_active(obs) and now - num(tx.get("started"), now) >= self.timeout_for(action):
            self.fail_transaction(now, "no_postcondition")
            return True
        if task_active(obs) and now - num(tx.get("started"), now) >= self.timeout_for(action) + 18.0:
            self.fail_transaction(now, "job_deadline")
            return True
        return False

    def finish_transaction(self, obs: dict, now: float) -> None:
        tx = self.tx
        if not tx:
            return
        action = str(tx.get("action", ""))
        verb = action_verb(action)
        if verb == "place":
            expected = tx.get("expected")
            item = str(tx.get("item", ""))
            if expected:
                if item == "crafting_table":
                    self.table_latch = tuple(expected)
                    self.log("portable_table_placed", position=self.table_latch)
                elif item == "furnace":
                    self.furnace_latch = tuple(expected)
                    self.log("portable_furnace_placed", position=self.furnace_latch)
        if verb == "furnace_start":
            parts = action.split()
            if len(parts) >= 2:
                self.smelt_started[parts[1]] = now
                self.log("smelting_started", furnace=parts[1])
        if verb == "furnace_take" and len(action.split()) >= 2:
            self.smelt_started.pop(action.split()[1], None)
        if self.job:
            target = str(self.job.get("target", ""))
            if self.job.get("kind") == "gather" and verb == "gather":
                self.completed.add(target)
                self.log("resource_confirmed", target=target, purpose=self.job.get("purpose", ""))
                self.job = None
            elif self.job.get("kind") == "step":
                if verb == "dig":
                    self.job["stage"] = "move"
                    self.job["dig_done"] = True
                elif verb in {"navigate", "goto"} and self.job.get("stage") == "move":
                    p = self_pos(obs)
                    expected = self.job.get("position")
                    if p is not None and expected is not None and dist(p, expected) <= 1.35:
                        self.completed.add(target)
                        self.log("moving_mine_confirmed", target=target, position=expected)
                        self.job = None
        self.tx = None

    def process_tx(self, obs: dict, now: float) -> None:
        if self.tx:
            self.tx_done(obs, now)

    def post(self, action: str | None, obs: dict, now: float, reason: str) -> str:
        action = str(action or "noop").strip() or "noop"
        if not self.can(action):
            action = "noop"
        verb = action_verb(action)
        # Deliberate one-shot repeats need an explicit noop barrier.
        if action == self.last_posted and verb not in {"noop", "controls", "attack", "shield"}:
            action = "noop"
            verb = "noop"
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("act_rejected", action=action, error=type(exc).__name__)
            action = "noop"
            verb = "noop"
        self.last_posted = action
        if action not in {"noop", "controls", "controls none"}:
            self.log("posted", action=action, reason=reason, mode=self.last_mode)
        if action == "cancel":
            self.tx = None
            self.job = None
            return action
        if verb not in {"noop", "controls", "attack", "shield", "wait"}:
            self.begin_tx(action, obs, now)
        return action

    def safe_support(self, obs: dict) -> dict | None:
        p = self_pos(obs)
        if p is None:
            return None
        floor_y = math.floor(p[1]) - 1
        fx, fz = math.floor(p[0]), math.floor(p[2])
        choices: list[dict] = []
        for row in rows(obs, "b"):
            if row["_name"] not in SUPPORT_BLOCKS or row["_name"] in HAZARD_BLOCKS:
                continue
            if row.get("in_reach") is False or row.get("harvestable") is False:
                continue
            q = row.get("_pos")
            if q is None or abs(q[1] - floor_y) > 1:
                continue
            if math.floor(q[0]) == fx and math.floor(q[2]) == fz:
                continue
            action = "place crafting_table %s up" % row["_id"]
            if self.blocked(action, self.now()):
                continue
            choices.append(row)
        choices.sort(key=lambda row: (0 if row.get("in_reach") else 1, row["_distance"]))
        return choices[0] if choices else None

    def usable_target(self, row: dict, now: float, names: set[str] | None = None, suffixes: tuple[str, ...] = (), hidden: bool = False) -> bool:
        item = row["_name"]
        if names is not None and item not in names:
            return False
        if suffixes and not any(item.endswith(suffix) for suffix in suffixes):
            return False
        if row.get("harvestable") is False:
            return False
        if row.get("visible") is False and not row.get("in_reach") and not hidden:
            return False
        if item in HAZARD_BLOCKS or item in {"air", "water", "bubble_column"}:
            return False
        return row["_id"] not in self.completed and not self.blocked(row["_id"], now)

    def choose_target(self, obs: dict, now: float, names: set[str] | None = None, suffixes: tuple[str, ...] = (), hidden: bool = False) -> dict | None:
        p = self_pos(obs)
        choices: list[dict] = []
        for row in rows(obs, "b"):
            if not self.usable_target(row, now, names, suffixes, hidden):
                continue
            q = row.get("_pos")
            # Never gather the block below the current feet cell.  It is the
            # most common way for a resource action to turn into a drop.
            if p is not None and q is not None:
                if math.floor(q[0]) == math.floor(p[0]) and math.floor(q[2]) == math.floor(p[2]) and q[1] <= math.floor(p[1]) - 1:
                    continue
            choices.append(row)
        choices.sort(key=lambda row: (
            0 if row.get("in_reach") else 1,
            0 if row.get("visible") is not False else 1,
            row["_distance"],
        ))
        return choices[0] if choices else None

    def begin_gather(self, row: dict, purpose: str, now: float) -> None:
        self.job = {
            "kind": "gather", "target": row["_id"], "position": row.get("_pos"),
            "purpose": purpose, "started": now,
        }
        self.log("gather_begin", target=row["_id"], block=row["_name"], purpose=purpose)

    def begin_step(self, row: dict, tool: str, now: float) -> None:
        self.job = {
            "kind": "step", "target": row["_id"], "position": row.get("_pos"),
            "name": row["_name"], "tool": tool, "stage": "approach",
            "started": now, "attempts": 0, "before_pos": self_pos(self.last_obs),
        }
        self.log("moving_mine_begin", target=row["_id"], block=row["_name"], tool=tool)

    def stand_choices(self, obs: dict, now: float, target: tuple[float, float, float] | None = None) -> list[dict]:
        choices: list[tuple[tuple, dict]] = []
        for row in rows(obs, "s"):
            q = row.get("_pos")
            if q is None or row["_distance"] < 1.2 or row["_distance"] > 46.0:
                continue
            if self.blocked("navigate " + row["_id"], now):
                continue
            used = self.used_stands.get(row["_id"], -1000.0)
            if now - used < 18.0:
                continue
            cell = (int(q[0]) // 5, int(q[2]) // 5)
            fresh = 0 if cell not in self.visited_cells else 1
            if target is not None:
                rank = (dist(q, target), used, row["_distance"])
            else:
                rank = (fresh, -row["_distance"], -used)
            choices.append((rank, row))
        choices.sort(key=lambda pair: pair[0])
        return [pair[1] for pair in choices]

    def explore_action(self, obs: dict, now: float) -> str | None:
        if self.manual_stage == "move":
            if now < self.manual_until:
                return "controls forward,sprint"
            self.manual_stage = "none"
            self.manual_until = -1.0
            return "controls none"
        if self.manual_stage == "look":
            self.manual_stage = "move"
            self.manual_until = now + 3.0
            q = self.manual_target
            return "look_at %.2f %.2f %.2f" % q if q else "noop"
        stands = self.stand_choices(obs, now)
        if stands:
            row = stands[0]
            self.used_stands[row["_id"]] = now
            q = row.get("_pos")
            if q:
                self.visited_cells.add((int(q[0]) // 5, int(q[2]) // 5))
            return "navigate " + row["_id"]
        # If pathfinder has exhausted the local stand candidates, use a very
        # short, explicitly stopped lease to leave a basin.  It is not a
        # general survival autopilot and is vetoed by the hazard layer.
        far = [row for row in rows(obs, "s") if row.get("_pos") is not None]
        if far:
            row = max(far, key=lambda item: item["_distance"])
            self.manual_target = row["_pos"]
            self.manual_stage = "look"
            return "look_at %.2f %.2f %.2f" % self.manual_target
        return None

    def job_command(self, obs: dict, now: float) -> str | None:
        job = self.job
        if not job:
            return None
        kind = str(job.get("kind"))
        target = str(job.get("target", ""))
        if kind == "gather":
            if now - num(job.get("started"), now) > 42.0:
                self.suppress(target, now, 150.0, "gather_timeout")
                self.job = None
                return None
            row = self.target_row(obs, target)
            if row is None:
                # A target can vanish just before the pickup audit.  Give the
                # bounded gather a little time rather than inventing a success.
                if now - num(job.get("started"), now) > 4.0:
                    self.suppress(target, now, 150.0, "target_missing")
                    self.job = None
                return None
            if self.blocked("gather %s 1" % target, now):
                self.job = None
                return None
            return "gather %s 1" % target

        stage = str(job.get("stage", "approach"))
        row = self.target_row(obs, target)
        if stage == "approach":
            if row is None:
                self.suppress(target, now, 150.0, "target_missing")
                self.job = None
                return None
            if row.get("in_reach"):
                job["stage"] = "equip"
                return None
            attempts = int(job.get("attempts", 0))
            if attempts >= 3 or now - num(job.get("started"), now) > 30.0:
                self.suppress(target, now, 150.0, "approach_limit")
                self.job = None
                return None
            stands = self.stand_choices(obs, now, row.get("_pos"))
            if stands:
                chosen = stands[0]
                self.used_stands[chosen["_id"]] = now
                job["attempts"] = attempts + 1
                return "navigate " + chosen["_id"]
            if row.get("visible") is not False and row.get("_pos") is not None and self.can("goto"):
                job["attempts"] = attempts + 1
                q = row["_pos"]
                return "goto %.2f %.2f %.2f" % q
            self.suppress(target, now, 150.0, "no_approach")
            self.job = None
            return None
        if stage == "equip":
            tool = str(job.get("tool", ""))
            if tool and name_of(self_row(obs).get("held_item")) != tool:
                job["stage"] = "look"
                return "equip %s hand" % tool
            job["stage"] = "look"
            return None
        if stage == "look":
            q = row.get("_pos") if row else job.get("position")
            if q is None:
                self.job = None
                return None
            job["stage"] = "dig"
            return "look_at %.2f %.2f %.2f" % q
        if stage == "dig":
            row = self.target_row(obs, target)
            if row is None:
                job["stage"] = "move"
                return None
            if not row.get("in_reach"):
                job["stage"] = "approach"
                return None
            q = row.get("_pos") or job.get("position")
            if q is None:
                self.job = None
                return None
            job["stage"] = "wait_dig"
            return "dig %d %d %d" % tuple(int(round(v)) for v in q)
        if stage == "wait_dig":
            if self.tx is None:
                job["stage"] = "move"
                return None
            return None
        if stage == "move":
            q = job.get("position")
            if q is None:
                self.job = None
                return None
            p = self_pos(obs)
            before = job.get("before_pos")
            if p is not None and dist(p, q) <= 1.35 and dist(p, before) >= 0.65:
                self.completed.add(target)
                self.log("moving_mine_confirmed", target=target, position=q)
                self.job = None
                return None
            if now - num(job.get("started"), now) > 58.0:
                self.suppress(target, now, 150.0, "move_timeout")
                self.job = None
                return None
            action = "goto %.2f %.2f %.2f" % tuple(q)
            if self.can("goto") and not self.blocked(action, now):
                return action
            stands = self.stand_choices(obs, now, q)
            return "navigate " + stands[0]["_id"] if stands else None
        self.job = None
        return None

    def begin_resource_or_step(self, obs: dict, now: float, names: set[str], tool: str, purpose: str) -> str | None:
        target = self.choose_target(obs, now, names=names)
        if target:
            self.begin_gather(target, purpose, now)
            return None
        step = self.step_target(obs, now)
        if step:
            self.begin_step(step, tool, now)
            return None
        return self.explore_action(obs, now)

    def step_target(self, obs: dict, now: float) -> dict | None:
        p = self_pos(obs)
        if p is None:
            return None
        py = math.floor(p[1])
        candidates: list[tuple[tuple, dict]] = []
        for row in rows(obs, "b"):
            if row["_name"] not in GROUND_BLOCKS or row["_name"] in HAZARD_BLOCKS:
                continue
            if row.get("harvestable") is False or row.get("visible") is False and not row.get("in_reach"):
                continue
            q = row.get("_pos")
            if q is None or row["_id"] in self.completed or self.blocked(row["_id"], now):
                continue
            dx, dz = q[0] - p[0], q[2] - p[2]
            horizontal = abs(dx) + abs(dz)
            if horizontal < 0.7 or horizontal > 1.65:
                continue
            level = math.floor(q[1])
            if level not in {py, py - 1}:
                continue
            if math.floor(q[0]) == math.floor(p[0]) and math.floor(q[2]) == math.floor(p[2]) and level <= py - 1:
                continue
            # Descend through an observed floor block first; a same-level wall
            # opening is only the fallback.  This is what exposes stone instead
            # of spending the mine budget on a horizontal grass tunnel.
            tier = 0 if level == py - 1 else 1
            if row["_name"] in {"sand", "gravel"}:
                continue
            candidates.append(((tier, 0 if row.get("in_reach") else 1, row["_distance"]), row))
        candidates.sort(key=lambda pair: pair[0])
        return candidates[0][1] if candidates else None

    def feasible(self, item: str, inv: dict[str, int]) -> int:
        planks = plank_total(inv)
        sticks = inv.get("stick", 0)
        cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
        if item.endswith("_planks"):
            base = item[:-7]
            return inv.get(base + "_log", 0) * 4 + inv.get(base + "_stem", 0) * 4
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
        if item.endswith("_bed"):
            wool = sum(value for key, value in inv.items() if key.endswith("_wool"))
            return min(wool // 3, planks // 3)
        return 0

    def craft_action(self, item: str, desired: int, inv: dict[str, int], table: dict | None = None) -> str | None:
        available = max(0, int(self.feasible(item, inv)))
        count = min(max(1, int(desired)), available)
        if count < 1:
            return None
        action = "craft %s %d" % (item, count)
        if table is not None:
            action += " " + str(table.get("_id", ""))
        if self.blocked(action, self.now()):
            return None
        return action

    def ensure_local_table(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        local = self.local_table(obs)
        if local:
            return None
        carried = inv.get("crafting_table", 0)
        if carried:
            if name_of(self_row(obs).get("held_item")) != "crafting_table":
                return "equip crafting_table hand"
            support = self.safe_support(obs)
            if support:
                return "place crafting_table %s up" % support["_id"]
            return self.explore_action(obs, now)
        # A new table is intentionally crafted locally from the 2x2 inventory
        # grid.  This avoids routing to an old table after death or descent.
        if plank_total(inv) >= 4:
            return self.craft_action("crafting_table", 1, inv)
        source = preferred_plank_item(inv)
        if source and any(key.endswith(WOOD_SUFFIXES) and value for key, value in inv.items()):
            return self.craft_action(source, 4, inv)
        target = self.choose_target(obs, now, suffixes=WOOD_SUFFIXES, hidden=True)
        if target:
            self.begin_gather(target, "wood for a local workstation", now)
            return None
        far = self.table_rows(obs)
        if far:
            row = min(far, key=lambda item: item["_distance"])
            q = row.get("_pos")
            if q and self.can("goto") and row["_distance"] > 4.5:
                return "goto %.2f %.2f %.2f" % (q[0] + 1.0, q[1], q[2] + 1.0)
        return self.explore_action(obs, now)

    def ensure_initial_tables(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        # Two tables are the key structural invariant: one stays at the first
        # surface site, one travels with the player into the mine.
        if not self.table_rows(obs) and inv.get("crafting_table", 0) < 2:
            if plank_total(inv) >= 8:
                return self.craft_action("crafting_table", 2, inv)
            source = preferred_plank_item(inv)
            if source and wood_total(inv):
                return self.craft_action(source, 4, inv)
            target = self.choose_target(obs, now, suffixes=WOOD_SUFFIXES, hidden=True)
            if target:
                self.begin_gather(target, "opening wood and a portable table", now)
                return None
        if inv.get("crafting_table", 0):
            return self.ensure_local_table(obs, inv, now)
        return None

    def furnace_action(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        local = self.local_furnace(obs)
        if not local:
            if inv.get("furnace"):
                if name_of(self_row(obs).get("held_item")) != "furnace":
                    return "equip furnace hand"
                support = self.safe_support(obs)
                if support:
                    return "place furnace %s up" % support["_id"]
                return self.explore_action(obs, now)
            table = self.actual_table(obs)
            if table:
                action = self.craft_action("furnace", 1, inv, table)
                if action:
                    return action
            elif self.local_table(obs):
                # The block was placed, but its candidate ID has not reached
                # this snapshot yet.  Never invent a synthetic TABLE_ID.
                return "noop"
            return self.ensure_local_table(obs, inv, now)
        fid = str(local.get("_id", ""))
        raw_item = next((item for item in ("raw_iron", "iron_ore", "deepslate_iron_ore") if inv.get(item)), "")
        fuel = next((item for item in ("coal", "charcoal") if inv.get(item)), "")
        if not fuel:
            fuel = next((item for item, count in inv.items() if item.endswith("_planks") and count), "")
        if raw_item and fuel and fid not in self.smelt_started:
            return "furnace_start %s %s %d %s %d" % (fid, raw_item, min(3, inv[raw_item]), fuel, min(3, inv[fuel]))
        started = self.smelt_started.get(fid, -1000.0)
        if fid in self.smelt_started and now - started >= 35.0:
            return "furnace_take " + fid
        return None

    def bed_action(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        if any(key.endswith("_bed") and value for key, value in inv.items()):
            return None
        wool_item = next((key for key, value in inv.items() if key.endswith("_wool") and value >= 3), "")
        if not wool_item:
            return None
        table = self.local_table(obs)
        if not table:
            return self.ensure_local_table(obs, inv, now)
        bed_item = wool_item[:-5] + "_bed"
        return self.craft_action(bed_item, 1, inv, table)

    def compile(self, obs: dict, now: float) -> str | None:
        me, here, inv, block_rows, entity_rows = self.facts(obs)
        del here, block_rows, entity_rows
        self.last_mode = "opening"

        # Opportunistic food is cheap when it is already exposed, but it never
        # displaces a required table or tool transaction for a distant target.
        if food_total(inv) == 0:
            food = self.choose_target(obs, now, names=FOOD_BLOCKS)
            if food and food["_distance"] <= 18.0:
                self.begin_gather(food, "first reliable food", now)
                return None

        has_table_block = bool(self.table_rows(obs)) or self.table_latch is not None
        has_pick = any(inv.get(item) for item in PICK_ORDER)
        logs = wood_total(inv)
        planks = plank_total(inv)
        tables = inv.get("crafting_table", 0)

        if not has_pick and logs == 0 and planks == 0 and not tables and not has_table_block:
            self.last_mode = "find_wood"
            target = self.choose_target(obs, now, suffixes=WOOD_SUFFIXES, hidden=True)
            if target:
                self.begin_gather(target, "opening wood", now)
                return None
            return self.explore_action(obs, now)

        # Build two tables whenever possible.  The first will be placed at the
        # surface; the second is a carried local workstation for the mine.
        if not has_table_block and tables < 2:
            self.last_mode = "portable_table_stock"
            action = self.ensure_initial_tables(obs, inv, now)
            if action:
                return action
            if self.job:
                return None

        # If a table is carried and no local one is visible, place it here.  In
        # particular, this is how the second table follows a successful mine.
        if tables and not self.local_table(obs) and has_pick:
            self.last_mode = "place_local_table"
            return self.ensure_local_table(obs, inv, now)

        # Make a bed from real wool if it becomes available.  A held bed is a
        # permanent opening milestone even if sleeping is not yet safe.
        bed = self.bed_action(obs, inv, now)
        if bed:
            self.last_mode = "bed"
            return bed
        if self.job:
            return None

        wooden_pick = bool(inv.get("wooden_pickaxe"))
        if not wooden_pick:
            self.last_mode = "wooden_tool"
            if planks < 3:
                source = preferred_plank_item(inv)
                if source and logs:
                    return self.craft_action(source, 4, inv)
                target = self.choose_target(obs, now, suffixes=WOOD_SUFFIXES, hidden=True)
                if target:
                    self.begin_gather(target, "wood for wooden pick", now)
                    return None
            if inv.get("stick", 0) < 2 and planks >= 2:
                return self.craft_action("stick", 4, inv)
            table = self.actual_table(obs)
            if table:
                action = self.craft_action("wooden_pickaxe", 1, inv, table)
                if action:
                    return action
            elif self.local_table(obs):
                # Wait for the placed block to become a real candidate ID.
                return "noop"
            return self.ensure_local_table(obs, inv, now)

        # Keep the spare carried while descending.  Placing it before a real
        # stone drop wastes the only local-workstation copy at an arbitrary
        # surface depth; the craft boundary below places it exactly when it is
        # needed.

        cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
        stone_pick = any(inv.get(item) for item in STONE_PICKS)
        if not stone_pick:
            self.last_mode = "moving_stone"
            if cobble < 3:
                return self.begin_resource_or_step(obs, now, STONE_BLOCKS, "wooden_pickaxe", "cobblestone for stone pick")
            table = self.actual_table(obs)
            if table:
                action = self.craft_action("stone_pickaxe", 1, inv, table)
                if action:
                    return action
            elif self.local_table(obs):
                return "noop"
            elif inv.get("crafting_table"):
                self.last_mode = "place_local_table"
                return self.ensure_local_table(obs, inv, now)
            else:
                # If the spare was lost, make a replacement only from real
                # carried planks; do not route to the original surface table.
                action = self.craft_action("crafting_table", 1, inv)
                if action:
                    self.last_mode = "make_local_table"
                    return action
            # No workstation is yet usable, so do not idle at the mine head.
            return self.begin_resource_or_step(obs, now, STONE_BLOCKS, "wooden_pickaxe", "continue stone search")

        # A furnace is another portable boundary: make it at the local table,
        # then place it beside the player rather than returning to the spawn.
        raw = inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0)
        if raw or (cobble >= 8 and not self.furnace_rows(obs) and not inv.get("furnace")):
            furnace = self.furnace_action(obs, inv, now)
            if furnace:
                self.last_mode = "local_furnace"
                return furnace
            if self.job:
                return None

        if not inv.get("iron_ingot") and raw == 0:
            self.last_mode = "find_iron"
            exposed = self.choose_target(obs, now, names=IRON_BLOCKS)
            if exposed:
                self.begin_gather(exposed, "exposed iron", now)
                return None
            # Coal is useful before the furnace, but do not search coal forever
            # when an observed stone opening can expose iron next.
            exposed_coal = self.choose_target(obs, now, names=COAL_BLOCKS)
            if exposed_coal:
                self.begin_gather(exposed_coal, "furnace fuel", now)
                return None
            return self.begin_resource_or_step(obs, now, IRON_BLOCKS | COAL_BLOCKS | STONE_BLOCKS, "stone_pickaxe", "moving search for iron")

        if raw and not inv.get("coal") and not inv.get("charcoal"):
            coal = self.choose_target(obs, now, names=COAL_BLOCKS)
            if coal:
                self.begin_gather(coal, "fuel for iron", now)
                return None
            # Planks are valid emergency fuel; use the furnace rather than
            # spending the entire remaining run seeking a coal vein.

        if raw or self.smelt_started:
            furnace = self.furnace_action(obs, inv, now)
            if furnace:
                self.last_mode = "smelt_iron"
                return furnace
            if self.job:
                return None

        if inv.get("iron_ingot") and not inv.get("shield"):
            self.last_mode = "shield"
            if plank_total(inv) < 5:
                source = preferred_plank_item(inv)
                if source and wood_total(inv):
                    return self.craft_action(source, 8, inv)
                target = self.choose_target(obs, now, suffixes=WOOD_SUFFIXES, hidden=True)
                if target:
                    self.begin_gather(target, "planks for shield", now)
                    return None
            table = self.actual_table(obs)
            if table:
                action = self.craft_action("shield", 1, inv, table)
                if action:
                    return action
            elif self.local_table(obs):
                return "noop"
            return self.ensure_local_table(obs, inv, now)

        # After the first defense milestone, keep the local mine alive rather
        # than returning to a table or waiting on a stale route.
        self.last_mode = "resource_frontier"
        if num(me.get("food"), 20) <= 12:
            food = best_food(inv)
            if food:
                return ("equip %s hand" % food) if name_of(me.get("held_item")) != food else "eat " + food
        target = self.choose_target(obs, now, names=IRON_BLOCKS | COAL_BLOCKS)
        if target:
            self.begin_gather(target, "continue resource search", now)
            return None
        return self.begin_resource_or_step(obs, now, IRON_BLOCKS | COAL_BLOCKS | STONE_BLOCKS, "stone_pickaxe", "continue local mine")

    def urgent(self, obs: dict, now: float) -> str | None:
        me = self_row(obs)
        if me.get("dead"):
            return "respawn"
        if num(obs.get("observation_age_s"), 0) > 3.0:
            return "cancel" if task_active(obs) else "noop"
        inv = inventory(obs)
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            return "controls forward,jump"
        if me.get("is_in_water") and num(me.get("oxygen"), 20) < 7:
            return "controls jump"
        if num(me.get("health"), 20) <= 6 or num(me.get("food"), 20) <= 5:
            food = best_food(inv)
            if food:
                if name_of(me.get("held_item")) != food and self.can("equip"):
                    return "equip %s hand" % food
                if self.can("eat"):
                    return "eat " + food
        hazards = self.guard.hazard_list(obs) if self.guard else []
        hard = any(item["kind"] in {"lava_or_fire", "drowning", "fall"} for item in hazards)
        if hard:
            return self.guard.fallback(obs, hazards, inv) if self.guard else "controls back,jump"
        if hazards:
            return self.guard.choose(obs, inv, now) if self.guard else "controls back,jump"
        return None

    def run_episode(self, game: Any) -> None:
        names = getattr(getattr(game, "action_spec", None), "names", ())
        self.allowed = set(names or ()) or set(DEFAULT_ACTIONS)
        self.guard = JevGuard(self.ctx, self.allowed, self.log)
        while not self.ctx.done:
            now = self.now()
            obs = observation(self.ctx.observe())
            if not obs:
                self.post("noop", {}, now, "missing_observation")
                self.ctx.wait_for_tick()
                continue
            self.last_obs = obs
            self.read_feedback(obs, now)
            self.process_tx(obs, now)
            me = self_row(obs)

            if me.get("dead"):
                self.tx = None
                self.job = None
                self.manual_stage = "none"
                self.table_latch = None
                self.furnace_latch = None
                self.smelt_started.clear()
                if not self.dead_seen:
                    self.dead_seen = True
                    self.post("respawn", obs, now, "ordinary_respawn")
                else:
                    self.post("noop", obs, now, "await_respawn")
                self.ctx.wait_for_tick()
                continue
            self.dead_seen = False

            urgent = self.urgent(obs, now)
            if urgent:
                if urgent not in {"noop", "respawn"} and (task_active(obs) or self.tx):
                    self.post("cancel", obs, now, "hazard_interrupt")
                else:
                    self.post(urgent, obs, now, "safety_override")
                self.ctx.wait_for_tick()
                continue

            if task_active(obs):
                self.post("noop", obs, now, "continue_bounded_job")
                self.ctx.wait_for_tick()
                continue
            if self.tx:
                self.post("noop", obs, now, "await_transaction")
                self.ctx.wait_for_tick()
                continue
            if self.job:
                action = self.job_command(obs, now)
                self.post(action or "noop", obs, now, "resource_transaction")
                self.ctx.wait_for_tick()
                continue
            if self.manual_stage != "none":
                self.post(self.explore_action(obs, now) or "noop", obs, now, "frontier_escape")
                self.ctx.wait_for_tick()
                continue

            action = self.compile(obs, now)
            self.post(action or "noop", obs, now, "milestone_compiler")
            self.ctx.wait_for_tick()


def build(ctx: Any) -> MobileController:
    return MobileController(ctx)
