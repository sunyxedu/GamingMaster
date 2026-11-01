"""Portable-workstation Minecraft controller.

This controller is deliberately transaction driven.  It does not treat a changed
position, a disappeared candidate, or an adaptor feedback string as resource
progress: inventory, block disappearance, and feet-position postconditions are
checked independently.  The distinctive route change is that the opening saves
materials for a second crafting table.  The second table is placed at the
current mining face, so acquiring cobblestone never requires a pathfinder trip
back through an unmined staircase to the surface table.
"""
from __future__ import annotations

import json
import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


WOOD_SUFFIXES = ("_log", "_stem")
PICKS = ("netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
STONE_PICK_SET = {"netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe"}
WEAPONS = (
    "netherite_sword", "diamond_sword", "iron_sword", "stone_sword", "wooden_sword",
    "netherite_axe", "diamond_axe", "iron_axe", "stone_axe", "wooden_axe",
)
FOODS = {
    "apple", "bread", "baked_potato", "carrot", "golden_carrot", "sweet_berries",
    "melon_slice", "mushroom_stew", "beetroot", "potato", "cooked_beef",
    "cooked_porkchop", "cooked_mutton", "cooked_chicken", "cooked_rabbit",
    "cooked_cod", "cooked_salmon", "beef", "porkchop", "mutton", "chicken",
    "rabbit", "cod", "salmon", "wheat",
}
FOOD_BLOCKS = {"hay_block", "melon", "sweet_berry_bush", "wheat", "carrots", "potatoes", "beetroots"}
STONE_BLOCKS = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON_BLOCKS = {"iron_ore", "deepslate_iron_ore"}
COAL_BLOCKS = {"coal_ore", "deepslate_coal_ore"}
GROUND_BLOCKS = {
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "mud", "clay", "stone", "cobblestone", "deepslate", "cobbled_deepslate",
    "tuff", "andesite", "diorite", "granite", "netherrack", "blackstone",
}
SUPPORT_BLOCKS = GROUND_BLOCKS | {
    "sand", "gravel", "terracotta", "sandstone", "red_sandstone",
    "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log",
    "dark_oak_log", "mangrove_log", "crafting_table", "furnace", "chest",
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
PASSIVES = {"sheep", "cow", "pig", "chicken", "rabbit"}
FAIL_WORDS = (
    "failed", "failure", "error", "unreachable", "timeout", "cannot", "rejected",
    "illegal", "not enough", "not in reach", "no path", "busy", "invalid", "unknown",
    "out of range", "denied",
)


def num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def name_of(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("id", ""))))
    return value.removeprefix("minecraft:") if isinstance(value, str) else ""


def pos_of(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        if value.get("position") is not None:
            value = value.get("position")
        elif all(key in value for key in ("x", "y", "z")):
            value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result = tuple(num(x, math.inf) for x in value[:3])
    return result if all(math.isfinite(x) for x in result) else None


def id_pos(identifier: Any) -> tuple[float, float, float] | None:
    if not isinstance(identifier, str) or len(identifier) < 3 or identifier[1] != ":":
        return None
    try:
        result = tuple(float(x) for x in identifier[2:].split(","))
    except (TypeError, ValueError):
        return None
    return result if len(result) == 3 and all(math.isfinite(x) for x in result) else None


def dist(a: Any, b: Any) -> float:
    pa, pb = pos_of(a), pos_of(b)
    if pa is None or pb is None:
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def action_verb(action: Any) -> str:
    return str(action or "").split(" ", 1)[0]


def parse_obs(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def self_row(obs: dict) -> dict:
    return obs.get("self") if isinstance(obs.get("self"), dict) else {}


def self_pos(obs: dict) -> tuple[float, float, float] | None:
    me = self_row(obs)
    return pos_of(me.get("position")) or pos_of(me)


def inv_of(obs: dict) -> dict[str, int]:
    """Read the audited personal inventory, never the cursor or a container."""
    source = obs.get("inventory")
    if source in (None, {}, []):
        source = obs.get("inventory_authoritative")
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source.get("items")
    result: dict[str, int] = {}
    if isinstance(source, dict):
        for key, raw in source.items():
            item = name_of(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(num(raw_count)))
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
    # This field is explicitly labelled personal client state.  It is useful
    # between atomic audit samples and is still not a window or cursor view.
    if not result and isinstance(obs.get("inventory_client"), dict):
        for key, raw in obs["inventory_client"].items():
            item = name_of(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(num(raw_count)))
            if item and count:
                result[item] = result.get(item, 0) + count
    return result


def inv_sig(inv: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(k), int(v)) for k, v in inv.items() if int(v) > 0))


def rows(obs: dict, prefix: str | None = None) -> list[dict]:
    here = self_pos(obs)
    raw_rows = obs.get("candidates") if isinstance(obs.get("candidates"), list) else []
    result: list[dict] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        ident = raw.get("id")
        if not isinstance(ident, str) or len(ident) < 3 or ident[1] != ":":
            continue
        if prefix and not ident.startswith(prefix + ":"):
            continue
        row = dict(raw)
        row["_id"] = ident
        row["_name"] = name_of(raw.get("name", raw.get("block", "")))
        row["_pos"] = pos_of(raw.get("position")) or id_pos(ident)
        row["_distance"] = num(raw.get("distance"), dist(here, row["_pos"]))
        result.append(row)
    result.sort(key=lambda row: row["_distance"])
    return result


def entities(obs: dict) -> list[dict]:
    here = self_pos(obs)
    source = obs.get("entities", obs.get("nearby_entities", []))
    if not isinstance(source, list):
        return []
    result = []
    for raw in source:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row["_name"] = name_of(raw.get("name", raw.get("type", "")))
        row["_pos"] = pos_of(raw.get("position"))
        row["_distance"] = num(raw.get("distance"), dist(here, row["_pos"]))
        result.append(row)
    return result


def food_count(inv: dict[str, int]) -> int:
    return sum(int(v) for k, v in inv.items() if k in FOODS)


def plank_count(inv: dict[str, int]) -> int:
    return sum(int(v) for k, v in inv.items() if k.endswith("_planks"))


def wood_count(inv: dict[str, int]) -> int:
    return sum(int(v) for k, v in inv.items() if k.endswith(WOOD_SUFFIXES))


def cobble_count(inv: dict[str, int]) -> int:
    return inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)


def best_food(inv: dict[str, int]) -> str:
    preference = (
        "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken", "bread",
        "baked_potato", "carrot", "apple", "sweet_berries", "melon_slice", "beef",
        "porkchop", "mutton", "chicken", "rabbit", "cod", "salmon", "potato", "wheat",
    )
    return next((item for item in preference if inv.get(item)), "")


def has_task(obs: dict) -> bool:
    value = obs.get("task")
    return value not in (None, "", {})


class ImmediateSafety:
    """Asynchronous Jev hazard choice with deterministic lethal overrides."""

    def __init__(self, ctx: Any, names: set[str], log) -> None:
        self.ctx = ctx
        self.names = names
        self.log = log
        self.future = None
        self.signature: tuple = ()
        self.started = -100.0
        self.cached: tuple[str, tuple, float] | None = None

    def allowed(self, action: str) -> bool:
        return not self.names or action_verb(action) in self.names

    def hazards(self, obs: dict) -> list[dict]:
        me = self_row(obs)
        result: list[dict] = []
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            result.append({"kind": "lava_or_fire", "id": "self", "distance": 0.0, "range": "touching"})
        if me.get("is_in_water") and num(me.get("oxygen"), 20) < 6:
            result.append({"kind": "drowning", "id": "self", "distance": 0.0, "range": "touching"})
        if me.get("fall_distance_estimate"):
            result.append({"kind": "fall", "id": "self", "distance": 0.0, "range": "touching"})
        for row in rows(obs, "b"):
            if row["_name"] in HAZARD_BLOCKS and row["_distance"] <= 4.5:
                result.append({
                    "kind": row["_name"], "id": row["_id"], "distance": row["_distance"], "range": "near",
                })
        for row in entities(obs):
            hostile = bool(row.get("hostile") or row.get("conditional_threat") or row["_name"] in HOSTILES)
            if hostile and row.get("visible") is not False and row["_distance"] <= 8.0:
                result.append({
                    "kind": row["_name"], "id": str(row.get("id", "")),
                    "distance": row["_distance"],
                    "range": "touching" if row["_distance"] <= 3.2 else "near",
                })
        return result

    def fallback(self, obs: dict, hazards: list[dict], inv: dict[str, int]) -> str:
        me = self_row(obs)
        if any(x["kind"] == "lava_or_fire" for x in hazards):
            if any(x.get("id") == "self" for x in hazards):
                return "controls forward,jump" if self.allowed("controls") else "cancel"
            return "controls back,jump" if self.allowed("controls") else "cancel"
        if any(x["kind"] == "drowning" for x in hazards):
            return "controls jump" if self.allowed("controls") else "cancel"
        if num(me.get("health"), 20) <= 6 or num(me.get("food"), 20) <= 5:
            food = best_food(inv)
            if food:
                if name_of(me.get("held_item")) != food and self.allowed("equip"):
                    return "equip %s hand" % food
                if self.allowed("eat"):
                    return "eat " + food
        weapon = next((item for item in WEAPONS if inv.get(item)), "")
        close = [x for x in hazards if x.get("id") != "self" and num(x.get("distance"), 99) <= 3.2]
        if close and weapon:
            if name_of(me.get("held_item")) != weapon and self.allowed("equip"):
                return "equip %s hand" % weapon
            if close[0].get("id") and self.allowed("attack"):
                return "attack " + str(close[0]["id"])
        if inv.get("shield") and self.allowed("shield"):
            return "shield on"
        return "controls back,jump" if self.allowed("controls") else "noop"

    def menu(self, obs: dict, hazards: list[dict], inv: dict[str, int]) -> list[str]:
        menu: list[str] = []
        fallback = self.fallback(obs, hazards, inv)
        if fallback and self.allowed(fallback):
            menu.append(fallback)
        if inv.get("shield") and self.allowed("shield"):
            menu.append("shield on")
        weapon = next((item for item in WEAPONS if inv.get(item)), "")
        for hazard in hazards:
            ident = hazard.get("id")
            if ident == "self":
                continue
            if num(hazard.get("distance"), 99) <= 3.2 and weapon and ident:
                if name_of(self_row(obs).get("held_item")) != weapon:
                    menu.append("equip %s hand" % weapon)
                menu.append("attack " + str(ident))
        if has_task(obs) and self.allowed("cancel"):
            menu.append("cancel")
        menu.append("noop")
        result = []
        for action in menu:
            if action not in result and (self.allowed(action) or action == "noop"):
                result.append(action)
        return result[:8]

    def choose(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        hazards = self.hazards(obs)
        if not hazards:
            self.signature = ()
            self.cached = None
            return None
        options = self.menu(obs, hazards, inv)
        if not options:
            return None
        signature = tuple(sorted((str(x["kind"]), str(x.get("id")), str(x["range"])) for x in hazards))
        if self.cached and self.cached[1] == signature and now - self.started < 2.5:
            return self.cached[0] if self.cached[0] in options else options[0]
        if self.future is None:
            try:
                if self.ctx.budget.remaining().get("system1_calls", 0) <= 2:
                    return self.fallback(obs, hazards, inv)
                state = {
                    "purpose": "Immediate survival only; choose one exact supplied action.",
                    "self": {
                        "health_band": "critical" if num(self_row(obs).get("health"), 20) < 7 else "low" if num(self_row(obs).get("health"), 20) < 12 else "stable",
                        "food_band": "low" if num(self_row(obs).get("food"), 20) < 8 else "fed",
                        "dimension": str(self_row(obs).get("dimension", "")),
                        "lava_fire": bool(self_row(obs).get("is_in_lava") or self_row(obs).get("is_in_fire")),
                    },
                    "hazards": [{"kind": x["kind"], "range": x["range"]} for x in hazards[:6]],
                    "legal_actions": options,
                    "rule": "Never invent an ID. Escape lava, fire, drowning, and falls before work; prefer defense over mining.",
                }
                questions = {
                    "response": Choice(
                        "Select the safest supplied immediate response. Do not invent a command or argument.",
                        {a: "Exact executable response: " + a for a in options},
                    ),
                    "safe_to_continue": Noul(
                        "Is it safe to continue current work without first using a defensive action?",
                        {"true": "No immediate hazard requires interruption.", "false": "A hazard or close hostile requires defense or interruption."},
                    ),
                }
                self.future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=1.5))
                self.signature = signature
                self.started = now
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self.log("jev_start_failed", error=type(exc).__name__)
                self.future = None
        if self.future is not None and self.future.done():
            future = self.future
            self.future = None
            try:
                answers = future.result()
                answer = answers.get("response") if isinstance(answers, dict) else None
                value = getattr(answer, "value", None)
                probs = getattr(answer, "probabilities", {}) or {}
                probability = num(probs.get(value), 0.0) if isinstance(probs, dict) else 0.0
                if value in options and self.signature == signature and probability >= 0.52:
                    self.cached = (value, signature, probability)
                    self.log("jev_safety", action=value, probability=round(probability, 3), hazards=len(hazards))
                    return value
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:
                self.log("jev_invalid", error=type(exc).__name__)
        return self.fallback(obs, hazards, inv)


class PortableController:
    """Finite-state semantic executor with portable workstations."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.names: set[str] = set()
        self.safety: ImmediateSafety | None = None
        self.last_action = "noop"
        self.txn: dict | None = None
        self.work: dict | None = None
        self.fail_until: dict[str, float] = {}
        self.completed: set[str] = set()
        self.visited: dict[str, float] = {}
        self.visited_cells: set[tuple[int, int]] = set()
        self.feedback_marker = ""
        self.table_hint: tuple[float, float, float] | None = None
        self.table_wait_until = -1.0
        self.furnace_hint: tuple[float, float, float] | None = None
        self.furnace_wait_until = -1.0
        self.smelt_started: dict[str, float] = {}
        self.mine_direction: tuple[int, int] | None = None
        self.step_count = 0
        self.explore_count = 0
        self.manual_until = -1.0
        self.hunt_target: str | None = None
        self.hunt_started = -1.0
        self.dead_seen = False
        self.mode_name = "opening"
        self.last_obs: dict = {}

    def now(self) -> float:
        return num(self.ctx.clock.now())

    def log(self, event: str, **fields: Any) -> None:
        self.ctx.log(event, **fields)

    def allowed(self, action: str) -> bool:
        return not self.names or action_verb(action) in self.names or action == "noop"

    def blocked(self, key: str, now: float) -> bool:
        return bool(key) and now < self.fail_until.get(key, -1.0)

    def suppress(self, key: str, now: float, seconds: float = 100.0, reason: str = "") -> None:
        if not key or key in {"noop", "cancel"}:
            return
        self.fail_until[key] = max(self.fail_until.get(key, -1.0), now + seconds)
        self.log("target_suppressed", target=key, blocked_s=seconds, reason=reason)

    def feedback_failed(self, obs: dict) -> bool:
        values = [obs.get(key) for key in ("action_feedback", "action_result", "last_result", "feedback")]
        def one(value: Any) -> bool:
            if isinstance(value, dict):
                if value.get("ok") is False or value.get("success") is False:
                    return True
                text = " ".join(str(value.get(key, "")) for key in ("status", "result", "message", "feedback", "reason", "error"))
                return any(word in text.lower() for word in FAIL_WORDS)
            if isinstance(value, str):
                low = value.lower()
                return any(word in low for word in FAIL_WORDS) and not any(word in low for word in ("success", "succeeded", " ok"))
            return False
        return any(one(x) for x in values)

    def process_feedback(self, obs: dict, now: float) -> None:
        try:
            marker = json.dumps([obs.get(key) for key in ("action_feedback", "action_result", "last_result", "feedback")], sort_keys=True, default=str)
        except Exception:
            marker = repr(obs.get("feedback"))
        if marker == self.feedback_marker:
            return
        self.feedback_marker = marker
        if self.txn and self.feedback_failed(obs):
            action = str(self.txn.get("action", ""))
            target = str(self.txn.get("target", ""))
            semantic_target = str(self.txn.get("semantic_target", ""))
            self.suppress(action, now, 145.0 if action_verb(action) in {"gather", "dig"} else 65.0, "adapter_feedback")
            if target and target != action:
                self.suppress(target, now, 145.0 if action_verb(action) in {"gather", "dig"} else 65.0, "adapter_feedback")
            if semantic_target:
                self.suppress(semantic_target, now, 145.0, "adapter_feedback")
            self.log("operation_rejected", action=action)
            self.txn = None
            self.work = None

    def action_timeout(self, action: str) -> float:
        verb = action_verb(action)
        if verb == "gather":
            return 31.0
        if verb in {"goto", "navigate"}:
            return 18.0
        if verb == "dig":
            return 12.0
        if verb == "look_at":
            return 2.0
        if verb in {"craft", "equip", "place", "furnace_start", "furnace_take", "activate_block", "eat", "sleep"}:
            return 10.0
        return 7.0

    def post(self, action: str | None, obs: dict, reason: str = "") -> str:
        action = str(action or "noop").strip() or "noop"
        if not self.allowed(action):
            action = "noop"
        verb = action_verb(action)
        # One-shot commands are latched by the adaptor.  The explicit noop is
        # the re-arm barrier; attacks and controls intentionally repeat.
        if action == self.last_action and verb not in {"noop", "controls", "attack", "shield"}:
            action = "noop"
            verb = "noop"
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("post_exception", action=action, error=type(exc).__name__)
            action = "noop"
            verb = "noop"
        self.last_action = action
        if action != "noop":
            self.log("posted", action=action, reason=reason, mode=self.mode_name)
        if action == "cancel":
            self.txn = None
            return action
        if verb in {"noop", "controls", "attack", "shield", "wait"}:
            return action
        parts = action.split()
        if verb == "dig" and self.work and self.work.get("target"):
            # `dig X Y Z` has three numeric arguments; the semantic target is
            # the observed block id held by the work transaction, not the
            # first coordinate token.
            target = str(self.work.get("target"))
        else:
            target = parts[1] if verb in {"gather", "dig", "navigate", "open_container", "sleep", "goto"} and len(parts) > 1 else ""
        semantic_target = str(self.work.get("target", "")) if self.work and verb in {"goto", "navigate"} else ""
        self.txn = {
            "action": action,
            "verb": verb,
            "started": self.now(),
            "before_inv": inv_sig(inv_of(obs)),
            "before_pos": self_pos(obs),
            "target": target,
            "semantic_target": semantic_target,
        }
        if verb in {"navigate", "goto"} and len(parts) > 1:
            self.visited[parts[1]] = self.now()
            q = id_pos(parts[1])
            if q:
                self.visited_cells.add((int(q[0]) // 5, int(q[2]) // 5))
        return action

    def row(self, obs: dict, ident: str) -> dict | None:
        return next((row for row in rows(obs, "b") if row["_id"] == ident), None)

    def resolve_txn(self, obs: dict, now: float) -> None:
        txn = self.txn
        if not txn:
            return
        action = str(txn.get("action", ""))
        verb = str(txn.get("verb", action_verb(action)))
        inv = inv_of(obs)
        before = txn.get("before_inv", ())
        p = self_pos(obs)
        target = str(txn.get("target", ""))
        done = False
        if verb == "equip":
            wanted = name_of(action.split()[1] if len(action.split()) > 1 else "")
            done = name_of(self_row(obs).get("held_item")) == wanted
        elif verb in {"craft", "furnace_start", "furnace_take"}:
            done = inv_sig(inv) != before
        elif verb == "eat":
            done = inv_sig(inv) != before or num(self_row(obs).get("food"), 20) > 5
        elif verb == "place":
            item = name_of(action.split()[1] if len(action.split()) > 1 else "")
            done = inv_sig(inv) != before or any(row["_name"] == item and row["_distance"] <= 5.0 for row in rows(obs, "b"))
            if item == "crafting_table" and not inv.get("crafting_table") and len(action.split()) >= 3:
                support = id_pos(action.split()[2])
                if support:
                    self.table_hint = (support[0], support[1] + 1.0, support[2])
                    self.table_wait_until = max(self.table_wait_until, now + 6.0)
                    self.log("portable_table_latched", position=self.table_hint)
            if item == "furnace" and not inv.get("furnace") and len(action.split()) >= 3:
                support = id_pos(action.split()[2])
                if support:
                    self.furnace_hint = (support[0], support[1] + 1.0, support[2])
                    self.furnace_wait_until = max(self.furnace_wait_until, now + 6.0)
        elif verb in {"navigate", "goto"}:
            before_pos = txn.get("before_pos")
            moved = dist(before_pos, p) >= 0.45
            target_id = str(txn.get("target", ""))
            semantic = str(txn.get("semantic_target", ""))
            row = self.row(obs, target_id) if target_id.startswith("b:") else None
            close = row is not None and (row.get("in_reach") is True or row["_distance"] <= 1.6)
            q = id_pos(target_id)
            close = close or (q is not None and dist(p, q) <= 1.6)
            # A goto used to approach a work block has two postconditions:
            # movement, and (when it does not finish the move) re-reading the
            # block as in-reach.  It must not be confused with the work target
            # itself; numeric goto coordinates are not block IDs.
            if semantic and self.work and self.work.get("stage") == "approach":
                work_row = self.row(obs, semantic)
                close = close or (work_row is not None and work_row.get("in_reach") is True)
            done = (not has_task(obs)) and (moved or close)
        elif verb == "look_at":
            done = not has_task(obs) and now - num(txn.get("started"), now) >= 0.55
        elif verb == "dig":
            row = self.row(obs, target)
            changed = False
            changes = obs.get("block_changes")
            if isinstance(changes, list):
                target_pos = id_pos(target)
                for change in changes:
                    if isinstance(change, dict) and target_pos is not None and dist(change.get("position"), target_pos) < 0.25:
                        changed = True
                        break
            done = row is None or changed
        elif verb == "gather":
            done = inv_sig(inv) != before or self.row(obs, target) is None
        elif verb in {"hotbar", "wake", "close_window", "sleep"}:
            done = not has_task(obs) and now - num(txn.get("started"), now) >= 0.45
        if done:
            self.complete_txn(obs, now, txn)
            return
        elapsed = now - num(txn.get("started"), now)
        if elapsed >= self.action_timeout(action):
            self.fail_txn(now, "no_postcondition")

    def complete_txn(self, obs: dict, now: float, txn: dict) -> None:
        action = str(txn.get("action", ""))
        verb = str(txn.get("verb", action_verb(action)))
        before = txn.get("before_inv", ())
        inv = inv_of(obs)
        target = str(txn.get("target", ""))
        self.txn = None
        self.log("postcondition", action=action, inventory_changed=inv_sig(inv) != before)
        if not self.work:
            return
        if verb == "gather":
            if inv_sig(inv) != before:
                self.completed.add(target)
                self.log("resource_confirmed", target=target, method="gather")
                self.work = None
            elif self.row(obs, target) is None:
                self.work["stage"] = "pickup"
        elif verb in {"navigate", "goto"}:
            # A completed approach is consumed by the work state machine only
            # after it rereads the target.  Leave `work` intact here.
            pass
        elif verb == "dig" and self.work.get("kind") == "step":
            self.work["stage"] = "move"
        elif verb in {"navigate", "goto"}:
            # The next work pass rereads in_reach and position.  Movement is
            # never itself credited as a resource; it only advances approach.
            pass

    def fail_txn(self, now: float, reason: str) -> None:
        txn = self.txn
        if not txn:
            return
        action = str(txn.get("action", ""))
        verb = action_verb(action)
        target = str(txn.get("target", ""))
        self.suppress(action, now, 150.0 if verb in {"gather", "dig"} else 70.0, reason)
        if target:
            self.suppress(target, now, 150.0 if verb in {"gather", "dig"} else 70.0, reason)
        self.log("transaction_failed", action=action, reason=reason)
        self.txn = None
        self.work = None

    def nearest_table(self, obs: dict, radius: float = 4.8) -> dict | None:
        choices = [row for row in rows(obs, "b") if row["_name"] == "crafting_table" and row["_distance"] <= radius and row.get("in_reach") is not False]
        choices.sort(key=lambda row: row["_distance"])
        return choices[0] if choices else None

    def furnace_rows(self, obs: dict) -> list[dict]:
        return [row for row in rows(obs, "b") if row["_name"] == "furnace"]

    def safe_support(self, obs: dict) -> dict | None:
        p = self_pos(obs)
        if p is None:
            return None
        floor_y = math.floor(p[1]) - 1
        fx, fz = math.floor(p[0]), math.floor(p[2])
        candidates = []
        for row in rows(obs, "b"):
            if row["_name"] not in SUPPORT_BLOCKS or row["_name"] in HAZARD_BLOCKS:
                continue
            if row.get("harvestable") is False or row.get("visible") is False and not row.get("in_reach"):
                continue
            q = row["_pos"]
            if q is None or math.floor(q[1]) != floor_y:
                continue
            if math.floor(q[0]) == fx and math.floor(q[2]) == fz:
                continue
            if row["_distance"] <= 4.5:
                candidates.append(row)
        candidates.sort(key=lambda row: row["_distance"])
        return candidates[0] if candidates else None

    def stand_rows(self, obs: dict, target: tuple[float, float, float] | None = None) -> list[dict]:
        now = self.now()
        choices = []
        for row in rows(obs, "s"):
            q = row["_pos"]
            if q is None or row["_distance"] < 1.0 or row["_distance"] > 46.0:
                continue
            ident = row["_id"]
            if self.blocked("goto " + ident, now) or now - self.visited.get(ident, -10000.0) < 18.0:
                continue
            if target is not None and dist(q, target) > 6.0:
                continue
            cell = (int(q[0]) // 5, int(q[2]) // 5)
            freshness = 0 if cell not in self.visited_cells else 1
            if target is not None:
                rank = (dist(q, target), self.visited.get(ident, -10000.0))
            else:
                rank = (freshness, -row["_distance"], self.visited.get(ident, -10000.0))
            choices.append((rank, row))
        choices.sort(key=lambda pair: pair[0])
        return [pair[1] for pair in choices]

    def movement_action(self, obs: dict, target: tuple[float, float, float] | None = None) -> str | None:
        stands = self.stand_rows(obs, target)
        if stands:
            # Preserve the adaptor's stable stand candidate ID.  Converting a
            # stand into a rounded coordinate loses the visited-target memory
            # and can replay the same failed goto forever (the seed-1 failure
            # mode in the parent traces).
            return "navigate " + stands[0]["_id"]
        if target is not None:
            # A coordinate target is only used for a listed safe stand or the
            # verified empty cell of a staircase; never fabricate an interior.
            return "goto %.2f %.2f %.2f" % target if dist(self_pos(obs), target) <= 48.0 else None
        return None

    def explore(self, obs: dict, now: float) -> str | None:
        action = self.movement_action(obs)
        if action:
            self.explore_count += 1
            return action
        # A short, renew-on-tick lease is only a way out of a basin with no
        # listed stand.  It is bounded and hard hazards override it.
        if now < self.manual_until:
            return "controls forward,jump"
        if self.explore_count >= 4:
            self.manual_until = now + 1.8
            return "controls forward,jump"
        return None

    def resource_row(self, obs: dict, names: set[str] | None = None, suffixes: tuple[str, ...] = (), hidden_logs: bool = False) -> dict | None:
        p = self_pos(obs)
        candidates = []
        for row in rows(obs, "b"):
            item = row["_name"]
            if names is not None and item not in names:
                continue
            if suffixes and not any(item.endswith(suffix) for suffix in suffixes):
                continue
            if item in HAZARD_BLOCKS or item in {"air", "water", "lava", "bubble_column"}:
                continue
            if row.get("harvestable") is False:
                continue
            hidden = row.get("visible") is False and not row.get("in_reach")
            if hidden and not hidden_logs:
                continue
            ident = row["_id"]
            if self.blocked(ident, self.now()) or ident in self.completed:
                continue
            q = row["_pos"]
            if p is not None and q is not None:
                if math.floor(q[0]) == math.floor(p[0]) and math.floor(q[2]) == math.floor(p[2]) and math.floor(q[1]) <= math.floor(p[1]) - 1:
                    continue
            candidates.append(row)
        candidates.sort(key=lambda row: (0 if row.get("in_reach") else 1, 0 if row.get("visible") else 1, row["_distance"]))
        return candidates[0] if candidates else None

    def staircase_row(self, obs: dict) -> dict | None:
        p = self_pos(obs)
        if p is None:
            return None
        py = math.floor(p[1])
        candidates = []
        for row in rows(obs, "b"):
            if row["_name"] not in GROUND_BLOCKS or row["_name"] in HAZARD_BLOCKS:
                continue
            if row.get("harvestable") is False or (row.get("visible") is False and not row.get("in_reach")):
                continue
            q = row["_pos"]
            if q is None or row["_id"] in self.completed or self.blocked(row["_id"], self.now()):
                continue
            dx, dz = q[0] - p[0], q[2] - p[2]
            horizontal = abs(dx) + abs(dz)
            if horizontal < 0.7 or horizontal > 1.6:
                continue
            level = math.floor(q[1])
            if level not in {py - 1, py} or row["_name"] in {"sand", "gravel"}:
                continue
            tier = 0 if level == py - 1 else 1
            direction_penalty = 0
            if self.mine_direction is not None:
                vx, vz = self.mine_direction
                forward = dx * vx + dz * vz
                side = abs(dx * vz - dz * vx)
                direction_penalty = 0 if forward > 0 and side <= 2 else 1
            candidates.append((tier, direction_penalty, horizontal, -num(row.get("seen_at"), 0), row))
        candidates.sort(key=lambda value: value[:4])
        return candidates[0][4] if candidates else None

    def begin_gather(self, obs: dict, row: dict, tool: str | None, purpose: str) -> None:
        p = self_pos(obs)
        q = row["_pos"]
        self.work = {
            "kind": "gather", "stage": "approach" if not row.get("in_reach") else ("equip" if tool else "gather"),
            "target": row["_id"], "name": row["_name"], "position": q, "tool": tool,
            "started": self.now(), "before_inv": inv_sig(inv_of(obs)), "purpose": purpose,
            "approaches": 0, "start_pos": p,
        }
        self.log("gather_transaction_begin", target=row["_id"], block=row["_name"], purpose=purpose)

    def begin_step(self, obs: dict, row: dict, tool: str) -> None:
        p = self_pos(obs)
        q = row["_pos"]
        if p is not None and q is not None:
            dx, dz = q[0] - p[0], q[2] - p[2]
            if abs(dx) >= abs(dz) and abs(dx) > 0.4:
                self.mine_direction = (1 if dx > 0 else -1, 0)
            elif abs(dz) > 0.4:
                self.mine_direction = (0, 1 if dz > 0 else -1)
        self.work = {
            "kind": "step", "stage": "approach" if not row.get("in_reach") else "equip",
            "target": row["_id"], "name": row["_name"], "position": q, "tool": tool,
            "started": self.now(), "before_inv": inv_sig(inv_of(obs)), "start_pos": p,
            "approaches": 0,
        }
        self.log("mine_transaction_begin", target=row["_id"], block=row["_name"], kind="moving_step", direction=self.mine_direction)

    def drop_row(self, obs: dict, target: tuple[float, float, float] | None = None) -> dict | None:
        choices = []
        for row in entities(obs):
            if row["_name"] not in {"item", "item_stack", "item_entity"} or row.get("dead") or row["_pos"] is None:
                continue
            if row["_distance"] <= 10.0 and (target is None or dist(row["_pos"], target) <= 8.0):
                choices.append(row)
        choices.sort(key=lambda row: row["_distance"])
        return choices[0] if choices else None

    def work_action(self, obs: dict, now: float) -> str | None:
        work = self.work
        if not work:
            return None
        kind = str(work.get("kind"))
        target = str(work.get("target", ""))
        row = self.row(obs, target)
        inv = inv_of(obs)
        before = work.get("before_inv", ())
        if now - num(work.get("started"), now) > (75.0 if kind == "gather" else 80.0):
            self.suppress(target, now, 150.0, "work_timeout")
            self.work = None
            return None
        if kind == "gather":
            if inv_sig(inv) != before:
                self.completed.add(target)
                self.log("resource_confirmed", target=target, method="inventory_delta")
                self.work = None
                return None
            stage = str(work.get("stage", "approach"))
            if stage == "pickup":
                drop = self.drop_row(obs, work.get("position"))
                if drop and drop.get("_pos"):
                    return self.movement_action(obs, drop["_pos"]) or ("goto %.2f %.2f %.2f" % drop["_pos"])
                if now - num(work.get("started"), now) > 48.0:
                    self.suppress(target, now, 150.0, "pickup_timeout")
                    self.work = None
                return None
            if row is None:
                # A gather job can have removed the block while the audit is
                # lagging.  Give item entities a bounded collection phase.
                work["stage"] = "pickup"
                return None
            if stage == "approach":
                if row.get("in_reach"):
                    work["stage"] = "equip" if work.get("tool") else "gather"
                    return None
                if int(work.get("approaches", 0)) >= 4 or now - num(work.get("started"), now) > 32.0:
                    self.suppress(target, now, 150.0, "approach_limit")
                    self.work = None
                    return None
                move = self.movement_action(obs, row["_pos"])
                if move:
                    work["approaches"] = int(work.get("approaches", 0)) + 1
                    return move
                if row.get("visible") or row["_name"].endswith(WOOD_SUFFIXES):
                    work["approaches"] = int(work.get("approaches", 0)) + 1
                    return "navigate " + target
                self.suppress(target, now, 150.0, "no_safe_approach")
                self.work = None
                return None
            if stage == "equip":
                tool = name_of(work.get("tool"))
                if tool and name_of(self_row(obs).get("held_item")) != tool:
                    return "equip %s hand" % tool
                work["stage"] = "gather"
            return "gather %s 1" % target

        # Moving staircase transaction: dig one listed adjacent block and then
        # enter its verified empty cell.  Unknown underground coordinates are
        # never synthesized.
        stage = str(work.get("stage", "approach"))
        if row is None and stage not in {"move"}:
            self.suppress(target, now, 150.0, "target_missing")
            self.work = None
            return None
        if stage == "approach":
            if row and row.get("in_reach"):
                work["stage"] = "equip"
                return None
            if int(work.get("approaches", 0)) >= 3 or now - num(work.get("started"), now) > 28.0:
                self.suppress(target, now, 150.0, "step_approach_limit")
                self.work = None
                return None
            move = self.movement_action(obs, row["_pos"] if row else work.get("position"))
            if move:
                work["approaches"] = int(work.get("approaches", 0)) + 1
                return move
            self.suppress(target, now, 150.0, "no_step_approach")
            self.work = None
            return None
        if stage == "equip":
            tool = name_of(work.get("tool"))
            if tool and name_of(self_row(obs).get("held_item")) != tool:
                work["stage"] = "look"
                return "equip %s hand" % tool
            work["stage"] = "look"
            return None
        if stage == "look":
            q = row["_pos"] if row else work.get("position")
            if q is None:
                self.work = None
                return None
            work["stage"] = "dig"
            return "look_at %.2f %.2f %.2f" % q
        if stage == "dig":
            if row is None:
                work["stage"] = "move"
                return None
            if not row.get("in_reach"):
                work["stage"] = "approach"
                return None
            q = row["_pos"] or work.get("position")
            if q is None:
                self.work = None
                return None
            work["stage"] = "wait_dig"
            return "dig %d %d %d" % tuple(int(round(x)) for x in q)
        if stage == "wait_dig":
            if row is None:
                work["stage"] = "move"
                return None
            if now - num(work.get("started"), now) > 42.0:
                self.suppress(target, now, 150.0, "dig_timeout")
                self.work = None
            return None
        if stage == "move":
            q = work.get("position")
            p = self_pos(obs)
            if q is None or p is None:
                self.work = None
                return None
            started = work.get("start_pos")
            if dist(p, q) <= 1.25 and dist(p, started) >= 0.7:
                self.completed.add(target)
                self.step_count += 1
                self.log("stair_step_confirmed", target=target, position=q, steps=self.step_count, depth=math.floor(p[1]))
                self.work = None
                return None
            if now - num(work.get("started"), now) > 58.0:
                self.suppress(target, now, 150.0, "step_move_timeout")
                self.work = None
                return None
            # A one-level floor opening is not a safe `goto` goal: it has no
            # stand candidate until the player actually steps/falls into it.
            # The preceding look_at points the camera at the verified opening,
            # so a 1.5-second forward+jump lease is the smallest controlled
            # movement that can enter it.  Same-level wall openings still use
            # pathfinder goto first.
            start = work.get("start_pos")
            if start is not None and q[1] < math.floor(start[1]):
                if not work.get("manual_started"):
                    work["manual_started"] = now
                    work["manual_until"] = now + 1.5
                    return "controls forward,jump"
                if now < num(work.get("manual_until"), now):
                    return "controls forward,jump"
                if not work.get("manual_stopped"):
                    work["manual_stopped"] = True
                    return "controls none"
            action = "goto %.2f %.2f %.2f" % q
            if not self.blocked(action, now):
                return action
            return self.movement_action(obs, q)
        self.work = None
        return None

    def feasible(self, item: str, inv: dict[str, int]) -> int:
        planks = plank_count(inv)
        sticks = inv.get("stick", 0)
        cobble = cobble_count(inv)
        if item.endswith("_planks"):
            base = item[:-7]
            return (inv.get(base + "_log", 0) + inv.get(base + "_stem", 0)) * 4
        if item == "stick":
            return (planks // 2) * 4
        if item == "crafting_table":
            return planks // 4
        if item == "wooden_pickaxe":
            return min(planks // 3, sticks // 2)
        if item == "wooden_sword":
            return min(planks // 2, sticks)
        if item == "stone_pickaxe":
            return min(cobble // 3, sticks // 2)
        if item == "stone_sword":
            return min(cobble // 2, sticks)
        if item == "furnace":
            return cobble // 8
        if item == "shield":
            return min(inv.get("iron_ingot", 0), planks // 5)
        if item.endswith("_bed"):
            wool = item.removesuffix("_bed") + "_wool"
            return min(inv.get(wool, 0) // 3, planks // 3)
        return 0

    def plank_item(self, inv: dict[str, int]) -> str:
        source = next((key for key, value in inv.items() if value and key.endswith(WOOD_SUFFIXES)), "")
        if source.endswith("_log"):
            return source[:-4] + "_planks"
        if source.endswith("_stem"):
            return source[:-5] + "_planks"
        return next((key for key, value in inv.items() if value and key.endswith("_planks")), "")

    def craft_cmd(self, item: str, desired: int, inv: dict[str, int], table: dict | None = None) -> str | None:
        available = self.feasible(item, inv)
        count = min(max(1, int(desired)), available)
        if count < 1:
            return None
        action = "craft %s %d" % (item, count)
        if table is not None:
            action += " " + str(table["_id"])
        return action

    def table_cmd(self, obs: dict, inv: dict[str, int], purpose: str = "") -> str | None:
        table = self.nearest_table(obs)
        if table:
            return None
        if inv.get("crafting_table"):
            support = self.safe_support(obs)
            if support:
                if name_of(self_row(obs).get("held_item")) != "crafting_table":
                    return "equip crafting_table hand"
                return "place crafting_table %s up" % support["_id"]
            return self.explore(obs, self.now())
        tables = [row for row in rows(obs, "b") if row["_name"] == "crafting_table"]
        if tables:
            table = min(tables, key=lambda row: row["_distance"])
            q = table["_pos"]
            if q is not None and table["_distance"] > 4.0:
                goal = (q[0] + 1.0, q[1], q[2] + 1.0)
                if dist(self_pos(obs), goal) <= 48.0:
                    return "goto %.2f %.2f %.2f" % goal
            return "navigate " + table["_id"]
        # A successful place consumes the item before the exposed candidate
        # stream necessarily contains the new block.  Return to the adjacent
        # feet cell from the real support coordinate; do not scout away and
        # lose the only portable-workstation handoff.
        if self.table_hint:
            now = self.now()
            if now < self.table_wait_until:
                return "noop"
            goal = (self.table_hint[0] + 1.0, self.table_hint[1], self.table_hint[2] + 1.0)
            if dist(self_pos(obs), goal) > 1.8 and dist(self_pos(obs), goal) <= 48.0:
                action = "goto %.2f %.2f %.2f" % goal
                if not self.blocked(action, now):
                    return action
            return "noop"
        return self.explore(obs, self.now())

    def local_workstation(self, obs: dict, inv: dict[str, int]) -> str | None:
        """Ensure a table exists at the current face without a return trip."""
        if self.nearest_table(obs):
            return None
        if inv.get("crafting_table"):
            return self.table_cmd(obs, inv, "place portable table")
        # A crafting table is a 2x2 recipe: this is intentionally stationless.
        if plank_count(inv) >= 4:
            return self.craft_cmd("crafting_table", 1, inv)
        return self.table_cmd(obs, inv, "return only when no portable table exists")

    def start_resource_or_step(self, obs: dict, now: float, names: set[str], tool: str, purpose: str) -> str | None:
        target = self.resource_row(obs, names=names)
        if target:
            self.begin_gather(obs, target, tool, purpose)
            return None
        target = self.staircase_row(obs)
        if target:
            self.begin_step(obs, target, tool)
            return None
        return self.explore(obs, now)

    def hunt_action(self, obs: dict, now: float) -> str | None:
        inv = inv_of(obs)
        if food_count(inv) > 0:
            self.hunt_target = None
            return None
        chosen = None
        if self.hunt_target:
            chosen = next((row for row in entities(obs) if str(row.get("id", "")) == self.hunt_target and row["_name"] in PASSIVES and not row.get("dead")), None)
        if chosen is None:
            candidates = [row for row in entities(obs) if row["_name"] in PASSIVES and row.get("visible") is not False and row["_distance"] <= 42.0 and not row.get("dead")]
            candidates.sort(key=lambda row: row["_distance"])
            chosen = candidates[0] if candidates else None
            self.hunt_target = str(chosen.get("id")) if chosen else None
            self.hunt_started = now if chosen else -1.0
        if chosen is None:
            return None
        if now - self.hunt_started > 70.0:
            self.hunt_target = None
            return None
        if chosen["_distance"] > 3.1:
            q = chosen.get("_pos")
            return "goto %.2f %.2f %.2f" % q if q is not None and chosen["_distance"] <= 48.0 else None
        weapon = next((item for item in WEAPONS if inv.get(item)), "")
        if weapon and name_of(self_row(obs).get("held_item")) != weapon:
            return "equip " + weapon + " hand"
        return "attack " + str(chosen.get("id"))

    def hard_emergency(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        me = self_row(obs)
        if me.get("dead"):
            return "respawn"
        if num(obs.get("observation_age_s"), 0) > 3.0:
            return "cancel" if has_task(obs) else "noop"
        hazards = self.safety.hazards(obs) if self.safety else []
        hard = any(x["kind"] in {"lava_or_fire", "drowning", "fall"} or x["kind"] in HAZARD_BLOCKS for x in hazards)
        if hard:
            return self.safety.fallback(obs, hazards, inv) if self.safety else "controls forward,jump"
        if num(me.get("health"), 20) <= 6 or num(me.get("food"), 20) <= 5:
            food = best_food(inv)
            if food:
                if name_of(me.get("held_item")) != food and self.allowed("equip"):
                    return "equip %s hand" % food
                if self.allowed("eat"):
                    return "eat " + food
        if hazards and self.safety:
            return self.safety.choose(obs, inv, now)
        return None

    def opening_action(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        """Make enough opening material for a local second workstation."""
        table = self.nearest_table(obs)
        pick = next((item for item in PICKS if inv.get(item)), "")
        # Four logs is the minimum useful reserve: first table, sticks/pick,
        # and one portable table.  We fall back to three only if exploration
        # cannot expose another log after a bounded search.
        material = plank_count(inv) + 4 * wood_count(inv)
        if not pick and material < 13:
            target = self.resource_row(obs, suffixes=WOOD_SUFFIXES, hidden_logs=True)
            if target:
                self.begin_gather(obs, target, None, "reserve wood for portable workstation")
                return None
            return self.explore(obs, now)
        if not table and not inv.get("crafting_table"):
            plank_item = self.plank_item(inv)
            if plank_count(inv) < 4 and plank_item:
                return self.craft_cmd(plank_item, 4, inv)
            action = self.craft_cmd("crafting_table", 1, inv)
            if action:
                return action
            return self.explore(obs, now)
        if inv.get("crafting_table") and not table:
            return self.table_cmd(obs, inv, "place opening table")
        if not pick:
            if not table:
                return self.table_cmd(obs, inv, "reach opening table") or self.explore(obs, now)
            if plank_count(inv) < 3:
                plank_item = self.plank_item(inv)
                if plank_item:
                    return self.craft_cmd(plank_item, 4, inv)
                target = self.resource_row(obs, suffixes=WOOD_SUFFIXES, hidden_logs=True)
                if target:
                    self.begin_gather(obs, target, None, "wooden pick material")
                    return None
            if inv.get("stick", 0) < 2 and plank_count(inv) >= 2:
                return self.craft_cmd("stick", 4, inv)
            return self.craft_cmd("wooden_pickaxe", 1, inv, table)
        # Turn the reserve into a second table before descending.  It is held
        # through the staircase and avoids the parent controller's dead return.
        if not inv.get("crafting_table"):
            # A table is a 2x2 recipe, so it can be made without returning to
            # the first table.  Convert one remaining log first when the
            # reserve is still represented as wood rather than planks.
            if plank_count(inv) < 4:
                plank_item = self.plank_item(inv)
                if plank_item:
                    return self.craft_cmd(plank_item, 4, inv)
                target = self.resource_row(obs, suffixes=WOOD_SUFFIXES, hidden_logs=True)
                if target:
                    self.begin_gather(obs, target, None, "finish portable-table reserve")
                    return None
                return self.explore(obs, now)
            action = self.craft_cmd("crafting_table", 1, inv)
            if action:
                return action
        return None

    def bed_or_food_action(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        # First seek a real visible food block, then a passive animal.  This
        # branch is opportunistic and never blocks the verified stone route.
        if food_count(inv) == 0:
            food = self.resource_row(obs, names=FOOD_BLOCKS)
            if food:
                self.begin_gather(obs, food, None, "visible food")
                return None
            action = self.hunt_action(obs, now)
            if action:
                return action
        # A sheep hunt can also produce the only non-cheat bed material.  Once
        # a same-colour wool recipe is feasible, make and place it if possible.
        wool = next((item for item, count in inv.items() if item.endswith("_wool") and count >= 3), "")
        table = self.nearest_table(obs)
        if wool and plank_count(inv) >= 3 and table:
            bed = wool[:-5] + "_bed"
            action = self.craft_cmd(bed, 1, inv, table)
            if action:
                return action
        if any(item.endswith("_bed") and count for item, count in inv.items()):
            bed_rows = [row for row in rows(obs, "b") if row["_name"].endswith("_bed")]
            if bed_rows:
                bed_rows.sort(key=lambda row: row["_distance"])
                if bed_rows[0]["_distance"] <= 4.0:
                    return "sleep " + bed_rows[0]["_id"]
            support = self.safe_support(obs)
            bed_item = next((item for item, count in inv.items() if item.endswith("_bed") and count), "")
            if support and bed_item and name_of(self_row(obs).get("held_item")) != bed_item:
                return "equip " + bed_item + " hand"
            if support and bed_item:
                return "place %s %s up" % (bed_item, support["_id"])
        return None

    def progression_action(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        table = self.nearest_table(obs)
        stone_pick = any(inv.get(item) for item in STONE_PICK_SET)
        wooden_pick = next((item for item in PICKS if inv.get(item)), "wooden_pickaxe")
        cobble = cobble_count(inv)
        raw = inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0)

        if not stone_pick:
            if cobble < 3:
                return self.start_resource_or_step(obs, now, STONE_BLOCKS, wooden_pick, "moving search for cobblestone")
            # Local handoff: craft a station from the reserved planks in 2x2,
            # place it beside the player, and only then issue the stone recipe.
            local = self.local_workstation(obs, inv)
            if local:
                return local
            table = self.nearest_table(obs)
            if inv.get("stick", 0) < 2 and plank_count(inv) >= 2:
                return self.craft_cmd("stick", 4, inv)
            if table:
                return self.craft_cmd("stone_pickaxe", 1, inv, table)
            return self.table_cmd(obs, inv, "stone pick handoff")

        # Once the stone pick exists, exposed iron is preferred.  If there is
        # none, the same verified moving route reveals the next loaded layer.
        if raw < 3:
            target = self.resource_row(obs, names=IRON_BLOCKS)
            if target:
                self.begin_gather(obs, target, "stone_pickaxe", "exposed iron")
                return None
            fuel_target = self.resource_row(obs, names=COAL_BLOCKS)
            if not inv.get("coal") and not inv.get("charcoal") and fuel_target:
                self.begin_gather(obs, fuel_target, "stone_pickaxe", "furnace fuel")
                return None
            return self.start_resource_or_step(obs, now, IRON_BLOCKS | COAL_BLOCKS, "stone_pickaxe", "moving iron search")

        furnace_rows = self.furnace_rows(obs)
        furnace = min(furnace_rows, key=lambda row: row["_distance"]) if furnace_rows else None
        if not furnace and not inv.get("furnace"):
            if cobble >= 8:
                local = self.local_workstation(obs, inv)
                if local:
                    return local
                table = self.nearest_table(obs)
                action = self.craft_cmd("furnace", 1, inv, table)
                if action:
                    return action
            return self.start_resource_or_step(obs, now, STONE_BLOCKS | COAL_BLOCKS, "stone_pickaxe", "stone for furnace")
        if not furnace and inv.get("furnace"):
            support = self.safe_support(obs)
            if support:
                if name_of(self_row(obs).get("held_item")) != "furnace":
                    return "equip furnace hand"
                return "place furnace %s up" % support["_id"]
            return self.explore(obs, now)
        if furnace is None:
            return None
        if furnace["_distance"] > 4.8:
            return self.movement_action(obs, furnace["_pos"]) or ("navigate " + furnace["_id"])
        fuel = next((item for item in ("coal", "charcoal") if inv.get(item)), "")
        if not fuel:
            fuel = next((item for item, count in inv.items() if count and item.endswith("_planks")), "")
        fid = furnace["_id"]
        if raw and fuel and fid not in self.smelt_started:
            self.smelt_started[fid] = now
            input_item = next((item for item in ("raw_iron", "iron_ore", "deepslate_iron_ore") if inv.get(item)), "raw_iron")
            return "furnace_start %s %s %d %s %d" % (fid, input_item, min(3, inv.get(input_item, raw)), fuel, min(3, inv[fuel]))
        if fid in self.smelt_started and now - self.smelt_started[fid] >= 35.0:
            return "furnace_take " + fid
        return None

    def shield_action(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        table = self.nearest_table(obs)
        if inv.get("iron_ingot") and not inv.get("shield"):
            if plank_count(inv) < 5:
                target = self.resource_row(obs, suffixes=WOOD_SUFFIXES, hidden_logs=True)
                if target:
                    self.begin_gather(obs, target, None, "wood for shield")
                    return None
                return self.explore(obs, now)
            if not table:
                return self.local_workstation(obs, inv) or self.table_cmd(obs, inv, "shield workstation")
            return self.craft_cmd("shield", 1, inv, table)
        return None

    def compile_action(self, obs: dict, now: float) -> str | None:
        inv = inv_of(obs)
        me = self_row(obs)
        # Reacquire a real opening after a death; dropped inventory is never
        # assumed to return.  Existing world tables may still be used if seen.
        opening_needed = not any(inv.get(item) for item in STONE_PICK_SET) and not inv.get("iron_ingot")
        if opening_needed:
            action = self.opening_action(obs, inv, now)
            if self.work:
                self.mode_name = "opening"
                return None
            if action is not None:
                self.mode_name = "opening"
                return action
            # Do not let an optional animal/bed excursion replace a missing
            # opening tool.  It is safe to wait for the next observation and
            # let the opening compiler retry a real recipe/target.
            if not any(inv.get(item) for item in PICKS):
                self.mode_name = "opening"
                return self.explore(obs, now)
        if inv.get("iron_ingot") and not inv.get("shield"):
            action = self.shield_action(obs, inv, now)
            if self.work:
                self.mode_name = "shield"
                return None
            if action is not None:
                self.mode_name = "shield"
                return action
        # Progression has priority over optional food/bed exploration.  In
        # particular, an unfed player with a wooden pick must not abandon a
        # verified staircase to chase a passive entity; hard health/hunger
        # emergency handling above still forces an eat action when available.
        action = self.progression_action(obs, inv, now)
        if self.work:
            self.mode_name = "stone_or_iron"
            return None
        if action is not None:
            self.mode_name = "stone_or_iron"
            return action
        # Only when no progression transaction is available do optional
        # observed food/bed actions get a turn.
        optional = self.bed_or_food_action(obs, inv, now)
        if self.work:
            self.mode_name = "food_or_bed"
            return None
        if optional is not None:
            self.mode_name = "food_or_bed"
            return optional

    def emergency_reset(self) -> None:
        self.txn = None
        self.work = None
        self.hunt_target = None
        self.manual_until = -1.0
        self.smelt_started.clear()
        self.step_count = 0
        self.table_hint = None
        self.furnace_hint = None
        self.completed.clear()

    def run_episode(self, game: Any) -> None:
        self.names = set(getattr(getattr(game, "action_spec", None), "names", ()) or ())
        self.safety = ImmediateSafety(self.ctx, self.names, self.log)
        while not self.ctx.done:
            now = self.now()
            obs = parse_obs(self.ctx.observe())
            if not obs:
                self.post("noop", {}, "missing_observation")
                self.ctx.wait_for_tick()
                continue
            self.last_obs = obs
            self.process_feedback(obs, now)
            me = self_row(obs)
            if me.get("dead"):
                self.emergency_reset()
                if not self.dead_seen:
                    self.dead_seen = True
                    self.post("respawn", obs, "ordinary_respawn")
                else:
                    self.post("noop", obs, "await_respawn")
                self.ctx.wait_for_tick()
                continue
            self.dead_seen = False
            inv = inv_of(obs)
            urgent = self.hard_emergency(obs, inv, now)
            if urgent:
                if urgent not in {"noop", "respawn"} and (has_task(obs) or self.txn):
                    self.post("cancel", obs, "hazard_interrupt")
                else:
                    self.post(urgent, obs, "safety_override")
                self.ctx.wait_for_tick()
                continue
            self.resolve_txn(obs, now)
            if has_task(obs):
                self.post("noop", obs, "continue_bounded_job")
                self.ctx.wait_for_tick()
                continue
            if self.txn:
                self.post("noop", obs, "await_postcondition")
                self.ctx.wait_for_tick()
                continue
            action = self.work_action(obs, now) if self.work else None
            if action is not None:
                self.post(action, obs, "verified_transaction")
                self.ctx.wait_for_tick()
                continue
            action = self.compile_action(obs, now)
            self.post(action or "noop", obs, "milestone_compiler")
            self.ctx.wait_for_tick()


def build(ctx):
    return PortableController(ctx)
