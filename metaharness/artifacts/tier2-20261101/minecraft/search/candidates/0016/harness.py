"""Candidate 0016: a safe excavation controller for the Minecraft opener.

The controller treats the exposed terrain as an interface to a *local mine*, not
as a list of destinations.  It first binds a block to a transaction, approaches
it using a safe stand, aims, and digs.  Low-value surface blocks are deliberately
removed (never the block supporting the player) so that stone and ores can become
exposed.  Workstation travel is an independent operation: a resource search is
never gated on being beside the table.

Astra may choose a coarse search mode after a measured stall.  Jev is reserved
for immediate hazards.  Neither model is allowed to construct a command.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


ID_RE = re.compile(r"^[bs]:-?\d+,-?\d+,-?\d+$")
ITEM_RE = re.compile(r"^[a-z0-9_]+$")

WOOD_SUFFIXES = ("_log", "_stem")
PICKS = ("netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
STONE_PICKS = {"netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe"}
WEAPONS = ("netherite_sword", "diamond_sword", "iron_sword", "stone_sword", "wooden_sword", "iron_axe", "stone_axe", "wooden_axe")
FOODS = {
    "apple", "bread", "baked_potato", "carrot", "golden_carrot", "sweet_berries",
    "melon_slice", "mushroom_stew", "beetroot", "potato", "cooked_beef",
    "cooked_porkchop", "cooked_mutton", "cooked_chicken", "cooked_rabbit",
    "cooked_cod", "cooked_salmon", "beef", "porkchop", "mutton", "chicken",
    "rabbit", "cod", "salmon", "wheat", "beetroot_soup",
}
FOOD_BLOCKS = {"hay_block", "melon", "sweet_berry_bush", "wheat", "carrots", "potatoes", "beetroots"}
STONE_BLOCKS = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON_BLOCKS = {"iron_ore", "deepslate_iron_ore", "raw_iron", "iron_block"}
FUEL_BLOCKS = {"coal_ore", "deepslate_coal_ore", "coal_block"}
GROUND_BLOCKS = {
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "sand", "red_sand", "gravel", "clay", "snow_block", "snow", "mud",
}
SUPPORT_BLOCKS = GROUND_BLOCKS | STONE_BLOCKS | {
    "terracotta", "sandstone", "red_sandstone", "netherrack", "blackstone",
    "basalt", "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log",
    "dark_oak_log", "mangrove_log", "chest", "crafting_table", "furnace",
}
HAZARD_BLOCKS = {
    "lava", "fire", "soul_fire", "magma_block", "cactus", "powder_snow",
    "campfire", "soul_campfire", "pointed_dripstone",
}
HOSTILES = {
    "zombie", "skeleton", "creeper", "spider", "drowned", "husk", "stray",
    "witch", "phantom", "enderman", "slime", "magma_cube", "blaze", "ghast",
    "piglin", "piglin_brute", "zombified_piglin", "hoglin", "wither_skeleton",
    "ender_dragon", "shulker", "silverfish",
}
ANIMALS = {"cow", "pig", "sheep", "chicken", "rabbit"}
FAIL_WORDS = (
    "failed", "failure", "error", "unreachable", "timeout", "cannot", "rejected",
    "illegal", "not enough", "not in reach", "no path", "busy", "invalid",
    "unknown", "out of range",
)


def number(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def dict_rows(value: Any) -> list[dict]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def item_name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", "")))
    if not isinstance(value, str):
        return ""
    value = value.removeprefix("minecraft:")
    return value if ITEM_RE.fullmatch(value) else ""


def point(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        if "position" in value:
            value = value.get("position")
        elif all(k in value for k in ("x", "y", "z")):
            value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    p = tuple(number(v, math.inf) for v in value[:3])
    return p if all(math.isfinite(v) for v in p) else None


def id_point(ident: Any) -> tuple[float, float, float] | None:
    if not isinstance(ident, str) or not ID_RE.fullmatch(ident):
        return None
    try:
        p = tuple(float(v) for v in ident[2:].split(","))
        return p if len(p) == 3 and all(math.isfinite(v) for v in p) else None
    except (TypeError, ValueError):
        return None


def distance(a: Any, b: Any) -> float:
    pa, pb = point(a), point(b)
    if pa is None or pb is None:
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def parse_observation(value: Any) -> dict:
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
    return mapping(obs.get("self"))


def self_position(obs: dict) -> tuple[float, float, float] | None:
    return point(self_row(obs).get("position"))


def inventory(obs: dict) -> dict[str, int]:
    """Read personal inventory, preferring the authoritative audit."""
    source = obs.get("inventory")
    if source is None or source == {} or source == []:
        source = obs.get("inventory_authoritative")
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source["items"]
    result: dict[str, int] = {}
    if isinstance(source, dict):
        for key, raw in source.items():
            name = item_name(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(number(raw_count)))
            if name and count:
                result[name] = result.get(name, 0) + count
    elif isinstance(source, list):
        for raw in dict_rows(source):
            name = item_name(raw)
            count = max(0, int(number(raw.get("count", 0))))
            if name and count:
                result[name] = result.get(name, 0) + count
    # The client view is labelled personal inventory, unlike cursor/container
    # previews, and is useful during the short audit lag at join.
    if not result and isinstance(obs.get("inventory_client"), dict):
        for key, raw in obs["inventory_client"].items():
            name = item_name(key)
            count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(number(count)))
            if name and count:
                result[name] = result.get(name, 0) + count
    return result


def inv_signature(inv: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(k), int(v)) for k, v in inv.items() if int(v) > 0))


def food_total(inv: dict[str, int]) -> int:
    return sum(v for k, v in inv.items() if k in FOODS)


def wood_total(inv: dict[str, int]) -> int:
    return sum(v for k, v in inv.items() if k.endswith(WOOD_SUFFIXES))


def plank_total(inv: dict[str, int]) -> int:
    return sum(v for k, v in inv.items() if k.endswith("_planks"))


def best_food(inv: dict[str, int]) -> str:
    order = (
        "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
        "bread", "baked_potato", "carrot", "apple", "sweet_berries",
        "melon_slice", "beef", "porkchop", "mutton", "chicken", "rabbit",
        "cod", "salmon", "potato", "wheat",
    )
    return next((x for x in order if inv.get(x, 0)), "")


def has_task(obs: dict) -> bool:
    task = obs.get("task")
    return task is not None and task != "" and task != {}


def blocks(obs: dict, prefix: str | None = None) -> list[dict]:
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
    result.sort(key=lambda x: x["_distance"])
    return result


def entities(obs: dict) -> list[dict]:
    here = self_position(obs)
    result = []
    source = obs.get("entities", obs.get("nearby_entities", []))
    for raw in dict_rows(source):
        row = dict(raw)
        row["_name"] = item_name(raw.get("name", raw.get("type", "")))
        row["_pos"] = point(raw.get("position"))
        row["_distance"] = number(raw.get("distance"), distance(here, row["_pos"]))
        result.append(row)
    return result


def block_is_support(row: dict) -> bool:
    name = row.get("_name", "")
    return name in SUPPORT_BLOCKS and name not in HAZARD_BLOCKS


def current_block(obs: dict, ident: str) -> dict | None:
    return next((x for x in blocks(obs, "b") if x["_id"] == ident), None)


def progress_signature(obs: dict) -> str:
    """Keep only durable milestones; elapsed/resource telemetry is not progress."""
    progress = mapping(obs.get("progress"))
    milestones = progress.get("milestones", {})
    if isinstance(milestones, dict):
        milestones = sorted((str(k), bool(v)) for k, v in milestones.items())
    else:
        milestones = str(milestones)
    advancements = obs.get("advancements", [])
    if isinstance(advancements, dict):
        advancements = sorted(str(k) for k, v in advancements.items() if v)
    elif isinstance(advancements, list):
        advancements = sorted(str(x) for x in advancements)
    else:
        advancements = str(advancements)
    return json.dumps([milestones, advancements], sort_keys=True, default=str)


def action_verb(action: str) -> str:
    return str(action).split(" ", 1)[0] if action else ""


class SafetyAdvisor:
    """Jev judges a small, current emergency menu; code owns hard vetoes."""

    def __init__(self, ctx: Any, names: set[str], log):
        self.ctx = ctx
        self.names = names
        self.log = log
        self.future = None
        self.signature: tuple = ()
        self.submitted = -100.0
        self.cached: tuple[str, float, tuple] | None = None

    def allowed(self, action: str) -> bool:
        return action_verb(action) in self.names

    def hazards(self, obs: dict) -> list[dict]:
        me = self_row(obs)
        result = []
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            result.append({"kind": "lava_or_fire", "id": "self", "distance": 0.0})
        if me.get("is_in_water") and number(me.get("oxygen"), 20) < 6:
            result.append({"kind": "drowning", "id": "self", "distance": 0.0})
        if number(me.get("fall_distance_estimate"), 0) > 3:
            result.append({"kind": "fall", "id": "self", "distance": 0.0})
        for row in blocks(obs, "b"):
            if row["_name"] in HAZARD_BLOCKS and row["_distance"] <= 4.5:
                result.append({"kind": row["_name"], "id": row["_id"], "distance": row["_distance"]})
        for row in entities(obs):
            hostile = bool(row.get("hostile") or row.get("conditional_threat") or row["_name"] in HOSTILES)
            if hostile and row.get("visible") and row["_distance"] <= 8.0:
                result.append({"kind": row["_name"], "id": str(row.get("id", "")), "distance": row["_distance"]})
        return result

    def fallback(self, obs: dict, hazards: list[dict], inv: dict[str, int]) -> str:
        me = self_row(obs)
        if any(x["kind"] == "lava_or_fire" for x in hazards):
            return "controls forward,jump" if self.allowed("controls") else "cancel"
        if any(x["kind"] == "drowning" for x in hazards):
            return "controls jump" if self.allowed("controls") else "cancel"
        if number(me.get("health"), 20) <= 7 or number(me.get("food"), 20) <= 5:
            food = best_food(inv)
            if food:
                if item_name(me.get("held_item")) != food and self.allowed("equip"):
                    return "equip %s hand" % food
                if self.allowed("eat"):
                    return "eat " + food
        weapon = next((x for x in WEAPONS if inv.get(x)), "")
        close = [x for x in hazards if x.get("id") != "self" and number(x.get("distance"), 99) <= 3.2]
        if close and weapon:
            if item_name(me.get("held_item")) != weapon and self.allowed("equip"):
                return "equip %s hand" % weapon
            if self.allowed("attack") and close[0].get("id"):
                return "attack " + str(close[0]["id"])
        if inv.get("shield") and self.allowed("shield"):
            return "shield on"
        return "controls back,jump" if self.allowed("controls") else "noop"

    def menu(self, obs: dict, hazards: list[dict], inv: dict[str, int]) -> list[str]:
        menu = []
        fallback = self.fallback(obs, hazards, inv)
        if fallback and self.allowed(fallback):
            menu.append(fallback)
        if inv.get("shield") and self.allowed("shield"):
            menu.append("shield on")
        weapon = next((x for x in WEAPONS if inv.get(x)), "")
        for hazard in hazards:
            if hazard.get("id") == "self":
                continue
            if number(hazard.get("distance"), 99) <= 3.2 and weapon and hazard.get("id"):
                if item_name(self_row(obs).get("held_item")) != weapon:
                    menu.append("equip %s hand" % weapon)
                menu.append("attack " + str(hazard["id"]))
        if has_task(obs) and self.allowed("cancel"):
            menu.append("cancel")
        menu.append("noop")
        answer = []
        for action in menu:
            if action not in answer and self.allowed(action):
                answer.append(action)
        return answer[:8]

    def decide(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        hazards = self.hazards(obs)
        if not hazards:
            self.cached = None
            self.signature = ()
            return None
        options = self.menu(obs, hazards, inv)
        if not options:
            return None
        sig = tuple(sorted((str(x["kind"]), str(x["id"]), "touching" if number(x["distance"], 99) <= 3.2 else "near") for x in hazards))
        if self.cached and self.cached[2] == sig and now - self.submitted < 2.5:
            return self.cached[0] if self.cached[0] in options else options[0]
        if self.future is None and self.ctx.budget.remaining().get("system1_calls", 0) > 2:
            state = {
                "purpose": "Immediate survival only. Choose an exact legal action from legal_actions.",
                "self": {
                    "health_band": "critical" if number(self_row(obs).get("health"), 20) < 7 else "low" if number(self_row(obs).get("health"), 20) < 12 else "stable",
                    "food_band": "low" if number(self_row(obs).get("food"), 20) < 8 else "fed",
                    "dimension": str(self_row(obs).get("dimension", "")),
                    "lava_fire": bool(self_row(obs).get("is_in_lava") or self_row(obs).get("is_in_fire")),
                },
                "hazards": [{"kind": x["kind"], "range": "touching" if number(x["distance"], 99) <= 3.2 else "near"} for x in hazards[:6]],
                "legal_actions": options,
                "rule": "Never invent an ID. Prefer leaving a hazard or shielding; never attack lava, fire, fall, or drowning.",
            }
            questions = {
                "response": Choice(
                    "Select the safest supplied immediate response. Survival outranks mining or crafting.",
                    {x: "Exact executable response: " + x for x in options},
                ),
                "safe_to_continue": Noul(
                    "Is it safe to continue the current bounded work without first using a defensive action?",
                    {"true": "No immediate hazard requires interruption.", "false": "A hazard or close hostile requires defense, retreat, or interruption."},
                ),
            }
            try:
                self.future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=1.5))
                self.signature = sig
                self.submitted = now
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
                probabilities = getattr(answer, "probabilities", {}) or {}
                probability = number(probabilities.get(value), 0.0) if isinstance(probabilities, dict) else 0.0
                if value in options and self.signature == sig and probability >= 0.52:
                    self.cached = (value, probability, sig)
                    self.log("jev_safety", action=value, probability=round(probability, 3), hazards=len(hazards))
                    return value
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:
                self.log("jev_invalid", error=type(exc).__name__)
        # A current deterministic answer keeps the world moving while Jev is
        # in flight.  The caller still applies the hard hazard override.
        return self.fallback(obs, hazards, inv)


class ExcavationController:
    """A block transaction engine plus a milestone compiler."""

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.names: set[str] = set()
        self.safety: SafetyAdvisor | None = None
        self.job: dict | None = None
        self.latch: dict | None = None
        self.failed_until: dict[str, float] = {}
        self.visited_stands: dict[str, float] = {}
        self.completed_blocks: set[str] = set()
        self.feedback_marker = ""
        self.last_posted = "noop"
        self.last_progress_at = 0.0
        self.last_inventory: tuple[tuple[str, int], ...] = ()
        self.last_progress: str = ""
        self.dead_seen = False
        self.table_hint: tuple[float, float, float] | None = None
        self.table_latched = False
        self.table_wait_until = -1.0
        self.furnace_hint: tuple[float, float, float] | None = None
        self.furnace_wait_until = -1.0
        self.smelt_started: dict[str, float] = {}
        self.last_mode = "wood"
        self.macro_focus = "progress"
        self.macro_future = None
        self.macro_started = -100.0
        self.macro_finished = -100.0
        self.exploration_round = 0

    def log(self, event: str, **fields: Any) -> None:
        self.ctx.log(event, **fields)

    def allowed(self, action: str) -> bool:
        return action_verb(action) in self.names

    def now(self) -> float:
        return number(self.ctx.clock.now())

    def blocked(self, key: str, now: float) -> bool:
        return now < self.failed_until.get(key, -1.0)

    def fail(self, key: str, now: float, duration: float = 100.0, reason: str = "") -> None:
        if key and key not in {"noop", "cancel"}:
            self.failed_until[key] = max(self.failed_until.get(key, -1.0), now + duration)
            self.log("target_suppressed", target=key, blocked_s=duration, reason=reason)

    def feedback_text(self, obs: dict) -> str:
        return json.dumps([obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")], sort_keys=True, default=str)

    def feedback_failed(self, values: list[Any]) -> bool:
        """Recognize a failed result without mistaking an ``error: null`` field for one."""
        def failed_value(value: Any) -> bool:
            if isinstance(value, dict):
                for key in ("ok", "success"):
                    if key in value and value.get(key) is False:
                        return True
                status = str(value.get("status", value.get("result", ""))).lower()
                if any(word in status for word in FAIL_WORDS):
                    return True
                for key in ("message", "feedback", "reason", "error"):
                    nested = value.get(key)
                    if isinstance(nested, str) and any(word in nested.lower() for word in FAIL_WORDS):
                        return True
                return False
            if isinstance(value, str):
                low = value.lower()
                return any(word in low for word in FAIL_WORDS) and not any(word in low for word in ("succeeded", "success", "ok"))
            return False
        return any(failed_value(value) for value in values)

    def process_feedback(self, obs: dict, now: float) -> None:
        values = [obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")]
        marker = json.dumps(values, sort_keys=True, default=str)
        if marker == self.feedback_marker:
            return
        self.feedback_marker = marker
        if self.latch and self.feedback_failed(values):
            action = str(self.latch.get("action", ""))
            self.fail(action, now, 150.0 if action_verb(action) in {"dig", "gather"} else 90.0, "adapter_feedback")
            if self.job and (action == self.job.get("action") or action_verb(action) in {"dig", "gather"}):
                self.fail(str(self.job.get("target", "")), now, 150.0, "job_feedback")
                self.job = None
            self.latch = None
            self.log("operation_rejected", action=action)

    def observe_progress(self, obs: dict, now: float) -> None:
        inv = inventory(obs)
        inv_sig = inv_signature(inv)
        prog = progress_signature(obs)
        changed = inv_sig != self.last_inventory or prog != self.last_progress
        if changed:
            self.last_inventory = inv_sig
            self.last_progress = prog
            self.last_progress_at = now
            self.log("verified_progress", inventory_delta=True)
        for row in blocks(obs, "b"):
            if row["_name"] == "crafting_table" and row["_pos"]:
                self.table_hint = row["_pos"]
                self.table_latched = True
                self.table_wait_until = -1.0
            if row["_name"] == "furnace" and row["_pos"]:
                self.furnace_hint = row["_pos"]
                self.furnace_wait_until = -1.0
        if self.latch is not None:
            placed = str(self.latch.get("action", ""))
            if placed.startswith("place crafting_table ") and not inv.get("crafting_table"):
                parts = placed.split()
                if len(parts) >= 3:
                    support = id_point(parts[2])
                    if support:
                        self.table_hint = (support[0], support[1] + 1.0, support[2])
                self.table_latched = True
                self.table_wait_until = now + 8.0
            elif placed.startswith("place furnace ") and not inv.get("furnace"):
                parts = placed.split()
                if len(parts) >= 3:
                    support = id_point(parts[2])
                    if support:
                        self.furnace_hint = (support[0], support[1] + 1.0, support[2])
                self.furnace_wait_until = now + 8.0
        if self.latch is None:
            return
        action = str(self.latch.get("action", ""))
        age = now - number(self.latch.get("started"), now)
        verb = action_verb(action)
        before_inv = self.latch.get("before_inv", ())
        if verb == "equip":
            wanted = item_name(action.split()[1] if len(action.split()) > 1 else "")
            if item_name(self_row(obs).get("held_item")) == wanted or age > 5.0:
                self.latch = None
                return
        elif verb in {"navigate", "goto"}:
            moved = distance(self.latch.get("before_pos"), self_position(obs)) >= 0.7
            if moved or age >= (15.5 if verb == "navigate" else 14.5):
                self.latch = None
                return
        elif verb == "look_at":
            if age >= 0.4:
                self.latch = None
                return
        elif verb == "dig":
            target = str(self.latch.get("target", ""))
            row = current_block(obs, target)
            if row is None or row.get("_name") in {"air", "water", "lava"} or inv_sig != before_inv:
                self.latch = None
                return
        elif verb in {"craft", "place", "furnace_start", "furnace_take", "eat", "sleep", "gather"}:
            if inv_sig != before_inv:
                self.latch = None
                return
        # An unchanged one-shot gets a bounded retry suppression.  This is
        # separate from the adaptor's own task deadline and prevents a stale
        # observation from becoming a permanent duplicate action.
        limit = 27.0 if verb == "gather" else 9.0 if verb in {"dig", "place"} else 7.0 if verb not in {"navigate", "goto"} else 17.0
        if age >= limit:
            self.fail(action, now, 100.0, "no_postcondition")
            self.latch = None
            if self.job and action_verb(action) in {"dig", "gather", "navigate", "goto"}:
                self.fail(str(self.job.get("target", "")), now, 150.0, "job_timeout")
                self.job = None

    def post(self, action: str, obs: dict, now: float, why: str = "") -> str:
        action = str(action or "noop").strip() or "noop"
        if not self.allowed(action) and action != "noop":
            action = "noop"
        verb = action_verb(action)
        # Parameterized one-shots are latched by the adaptor.  A noop barrier
        # is required before a deliberate retry of the same string.
        if action == self.last_posted and verb not in {"noop", "controls", "attack", "shield"}:
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
            try:
                self.ctx.act("noop")
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                pass
        self.last_posted = action
        if action != "noop":
            self.log("posted", action=action, reason=why, mode=self.last_mode)
        if action == "cancel":
            self.job = None
            self.latch = None
        elif verb not in {"noop", "controls", "attack", "shield"}:
            parts = action.split()
            self.latch = {
                "action": action,
                "started": now,
                "before_inv": inv_signature(inventory(obs)),
                "before_pos": self_position(obs),
            }
            if verb in {"dig", "gather"} and len(parts) >= 2:
                self.latch["target"] = self.job.get("target", "") if self.job else ""
            if verb == "navigate" and len(parts) >= 2:
                ident = parts[1]
                self.visited_stands[ident] = now
        return action

    def safe_stands(self, obs: dict, target: tuple[float, float, float] | None = None, now: float | None = None) -> list[dict]:
        now = self.now() if now is None else now
        here = self_position(obs)
        candidates = []
        for row in blocks(obs, "s"):
            q = row["_pos"]
            if q is None or row["_distance"] < 2.0 or row["_distance"] > 46.0:
                continue
            ident = row["_id"]
            if self.blocked("navigate " + ident, now):
                continue
            if target is not None and distance(q, target) > 5.8:
                continue
            if target is not None:
                rank = (distance(q, target), self.visited_stands.get(ident, -1000.0), row["_distance"])
            else:
                # A fresh far stand changes the loaded candidate window; the
                # no-model exploration policy still avoids blind controls.
                rank = (0 if self.visited_stands.get(ident, -1000.0) < now - 35.0 else 1, -row["_distance"], distance(q, here))
            candidates.append((rank, row))
        candidates.sort(key=lambda pair: pair[0])
        return [x[1] for x in candidates]

    def choose_exploration(self, obs: dict, now: float) -> str:
        stands = self.safe_stands(obs, None, now)
        if stands:
            selected = stands[0]
            self.exploration_round += 1
            return "navigate " + selected["_id"]
        return "noop"

    def stand_for(self, obs: dict, target: tuple[float, float, float], now: float) -> str | None:
        stands = self.safe_stands(obs, target, now)
        if stands:
            return "navigate " + stands[0]["_id"]
        return None

    def support_block(self, obs: dict) -> dict | None:
        here = self_position(obs)
        if here is None:
            return None
        floor_y = math.floor(here[1]) - 1
        options = []
        for row in blocks(obs, "b"):
            q = row["_pos"]
            if q is None or not row.get("in_reach") or row.get("visible") is False or not block_is_support(row):
                continue
            if math.floor(q[1]) != floor_y:
                continue
            # Placing up here would put the table in the player's feet cell.
            if math.floor(q[0]) == math.floor(here[0]) and math.floor(q[2]) == math.floor(here[2]):
                continue
            # A rejected support is not a reason to retry the same placement;
            # another observed support may be valid.
            if self.blocked("place crafting_table %s up" % row["_id"], self.now()) or self.blocked("place furnace %s up" % row["_id"], self.now()):
                continue
            options.append(row)
        options.sort(key=lambda x: x["_distance"])
        return options[0] if options else None

    def table_rows(self, obs: dict) -> list[dict]:
        return [x for x in blocks(obs, "b") if x["_name"] == "crafting_table"]

    def near_table(self, obs: dict) -> dict | None:
        rows = [x for x in self.table_rows(obs) if x["_distance"] <= 4.8 and (x.get("in_reach") is not False)]
        rows.sort(key=lambda x: x["_distance"])
        return rows[0] if rows else None

    def furnace_rows(self, obs: dict) -> list[dict]:
        return [x for x in blocks(obs, "b") if x["_name"] == "furnace"]

    def row_is_safe_target(self, obs: dict, row: dict, allow_ground: bool = False) -> bool:
        if row.get("harvestable") is False or row["_id"] in self.completed_blocks:
            return False
        name = row["_name"]
        if name in HAZARD_BLOCKS or name in {"air", "water", "lava", "bubble_column"}:
            return False
        if not (row.get("visible") or row.get("in_reach")):
            return False
        p = self_position(obs)
        q = row["_pos"]
        if p is None or q is None:
            return False
        # Never remove the supporting block under the current feet cell.
        if math.floor(q[0]) == math.floor(p[0]) and math.floor(q[2]) == math.floor(p[2]) and math.floor(q[1]) <= math.floor(p[1]) - 1:
            return False
        if allow_ground and abs(q[1] - math.floor(p[1])) > 1.2:
            return False
        return True

    def resource_target(self, obs: dict, names: set[str] | None = None, suffixes: tuple[str, ...] = ()) -> dict | None:
        now = self.now()
        choices = []
        for row in blocks(obs, "b"):
            name = row["_name"]
            if names is not None and name not in names:
                continue
            if suffixes and not any(name.endswith(s) for s in suffixes):
                continue
            if not self.row_is_safe_target(obs, row):
                continue
            if self.blocked(row["_id"], now):
                continue
            rank = (0 if row.get("in_reach") else 1, 0 if row.get("visible") else 1, row["_distance"])
            choices.append((rank, row))
        choices.sort(key=lambda x: x[0])
        return choices[0][1] if choices else None

    def excavation_target(self, obs: dict, now: float) -> dict | None:
        choices = []
        for row in blocks(obs, "b"):
            if row["_name"] not in GROUND_BLOCKS:
                continue
            if not self.row_is_safe_target(obs, row, allow_ground=True):
                continue
            if self.blocked(row["_id"], now):
                continue
            p = self_position(obs)
            q = row["_pos"]
            # Prefer a nearby side/floor block.  Removing the nearest adjacent
            # layer exposes the next layer without a blind downward shaft.
            side = 0 if p and q and abs(q[0] - p[0]) + abs(q[2] - p[2]) >= 1.0 else 1
            rank = (side, 0 if row.get("in_reach") else 1, row["_distance"], -q[1] if q else 0)
            choices.append((rank, row))
        choices.sort(key=lambda x: x[0])
        return choices[0][1] if choices else None

    def begin_job(self, target: dict, kind: str, tool: str | None, purpose: str, now: float) -> None:
        self.job = {
            "kind": kind,
            "stage": "gather" if kind == "gather" and (target.get("visible") or target.get("in_reach")) else "approach",
            "target": target["_id"],
            "name": target["_name"],
            "position": target["_pos"],
            "tool": tool,
            "purpose": purpose,
            "started": now,
            "approaches": 0,
            "collect_attempts": 0,
            "count": 3 if target["_name"].endswith(WOOD_SUFFIXES) else 1,
            "before_inv": inv_signature(inventory(self._last_obs)),
        }
        self.log("transaction_begin", target=target["_id"], block=target["_name"], kind=kind, purpose=purpose)

    def current_target(self, obs: dict) -> dict | None:
        return current_block(obs, str(self.job.get("target", ""))) if self.job else None

    def item_entity_near(self, obs: dict, target: tuple[float, float, float] | None) -> dict | None:
        found = []
        for row in entities(obs):
            if row["_name"] not in {"item", "item_stack", "item_entity"} or row.get("dead") or row["_pos"] is None:
                continue
            if row["_distance"] <= 10.0 and (target is None or distance(row["_pos"], target) <= 8.0):
                found.append(row)
        found.sort(key=lambda x: x["_distance"])
        return found[0] if found else None

    def job_command(self, obs: dict, now: float) -> str | None:
        job = self.job
        if not job:
            return None
        kind = str(job.get("kind"))
        target_id = str(job.get("target", ""))
        row = self.current_target(obs)
        inv = inventory(obs)
        stage = str(job.get("stage", "approach"))
        if stage == "gather":
            if inv_signature(inv) != job.get("before_inv", ()):
                self.completed_blocks.add(target_id)
                self.log("resource_confirmed", target=target_id, method="gather")
                self.job = None
                return None
            if row is None:
                self.fail(target_id, now, 150.0, "gather_target_missing")
                self.job = None
                return None
            if now - number(job.get("started"), now) > 28.0:
                self.fail(target_id, now, 150.0, "gather_timeout")
                self.job = None
                return None
            return "gather %s %d" % (target_id, int(job.get("count", 1))) if self.allowed("gather") else None
        if stage == "approach":
            if row is None:
                self.fail(target_id, now, 150.0, "target_missing")
                self.job = None
                return None
            if row.get("in_reach"):
                job["stage"] = "equip"
                return None
            if now - number(job.get("started"), now) > 24.0 or int(job.get("approaches", 0)) >= 3:
                self.fail(target_id, now, 150.0, "approach_limit")
                self.job = None
                return None
            action = self.stand_for(obs, row["_pos"], now)
            if action:
                job["approaches"] = int(job.get("approaches", 0)) + 1
                return action
            # A visible exposed block is also a legal interaction candidate;
            # use this only once after safe stands have been exhausted.
            if row.get("visible") and not job.get("block_approach_used") and self.allowed("navigate"):
                job["block_approach_used"] = True
                job["approaches"] = int(job.get("approaches", 0)) + 1
                return "navigate " + target_id
            self.fail(target_id, now, 150.0, "no_safe_approach")
            self.job = None
            return None
        if stage == "equip":
            tool = item_name(job.get("tool"))
            held = item_name(self_row(obs).get("held_item"))
            if tool:
                if not inv.get(tool):
                    self.fail(target_id, now, 150.0, "tool_missing")
                    self.job = None
                    return None
                if held != tool:
                    job["stage"] = "equip_wait"
                    return "equip %s hand" % tool
            job["stage"] = "look"
            return None
        if stage == "equip_wait":
            tool = item_name(job.get("tool"))
            if item_name(self_row(obs).get("held_item")) == tool:
                job["stage"] = "look"
                return None
            if now - number(job.get("started"), now) > 28.0:
                self.fail(target_id, now, 150.0, "equip_timeout")
                self.job = None
            return None
        if stage == "look":
            q = row["_pos"] if row else job.get("position")
            if q is None:
                self.fail(target_id, now, 150.0, "position_missing")
                self.job = None
                return None
            job["stage"] = "dig"
            return "look_at %.2f %.2f %.2f" % q if self.allowed("look_at") else None
        if stage == "dig":
            row = self.current_target(obs)
            if row is None:
                job["stage"] = "collect" if kind != "excavate" else "done"
                return None
            if not row.get("in_reach"):
                # Exposure can remain visible while the player has drifted out
                # of interaction range. Re-enter the approach stage instead of
                # spending a dig attempt on a guaranteed rejection.
                job["stage"] = "approach"
                return None
            if not row.get("visible") and not row.get("in_reach"):
                self.fail(target_id, now, 150.0, "not_exposed_at_dig")
                self.job = None
                return None
            q = row["_pos"] or job.get("position")
            if q is None:
                self.job = None
                return None
            job["stage"] = "wait_dig"
            if not self.allowed("dig"):
                self.fail(target_id, now, 150.0, "dig_unavailable")
                self.job = None
                return None
            return "dig %d %d %d" % tuple(int(round(x)) for x in q)
        if stage == "wait_dig":
            row = self.current_target(obs)
            if row is None or row.get("_name") in {"air", "water", "lava"}:
                if kind == "excavate":
                    self.completed_blocks.add(target_id)
                    self.log("excavation_confirmed", target=target_id)
                    self.job = None
                else:
                    job["stage"] = "collect"
                return None
            if inv_signature(inv) != job.get("before_inv", ()) and kind != "excavate":
                self.completed_blocks.add(target_id)
                self.log("resource_confirmed", target=target_id, method="inventory")
                self.job = None
                return None
            if now - number(job.get("started"), now) > 34.0:
                self.fail(target_id, now, 150.0, "dig_no_block_change")
                self.job = None
            return None
        if stage == "collect":
            if inv_signature(inv) != job.get("before_inv", ()):
                self.completed_blocks.add(target_id)
                self.log("resource_confirmed", target=target_id, method="pickup")
                self.job = None
                return None
            if now - number(job.get("started"), now) > 47.0:
                self.fail(target_id, now, 150.0, "pickup_timeout")
                self.job = None
                return None
            drop = self.item_entity_near(obs, job.get("position"))
            if drop and drop.get("_pos") and not job.get("drop_attempted"):
                job["drop_attempted"] = True
                job["collect_attempts"] = int(job.get("collect_attempts", 0)) + 1
                q = drop["_pos"]
                # A listed safe stand is more reliable than asking goto to
                # path to an item entity's center or to a solid block cell.
                stand = self.stand_for(obs, q, now)
                if stand:
                    return stand
                if self.allowed("goto"):
                    return "goto %.2f %.2f %.2f" % (q[0], q[1] - 1.0, q[2])
            if not job.get("drop_attempted"):
                stand = self.stand_for(obs, job.get("position"), now)
                if stand:
                    job["drop_attempted"] = True
                    return stand
            job["stage"] = "collect_wait"
            return None
        if stage == "collect_wait":
            if inv_signature(inv) != job.get("before_inv", ()):
                self.completed_blocks.add(target_id)
                self.log("resource_confirmed", target=target_id, method="pickup_after_approach")
                self.job = None
                return None
            if now - number(job.get("started"), now) > 47.0:
                self.fail(target_id, now, 150.0, "pickup_timeout")
                self.job = None
            return None
        if stage == "done":
            self.job = None
        return None

    def animal_target(self, obs: dict, sheep_first: bool = False) -> dict | None:
        choices = []
        for row in entities(obs):
            if row["_name"] not in ANIMALS or not row.get("visible") or row.get("dead") or row["_pos"] is None:
                continue
            if row["_distance"] > 30:
                continue
            choices.append(row)
        choices.sort(key=lambda x: (0 if sheep_first and x["_name"] == "sheep" else 1, x["_distance"]))
        return choices[0] if choices else None

    def start_animal(self, obs: dict, row: dict, now: float) -> None:
        self.job = {
            "kind": "animal", "stage": "approach", "entity": str(row.get("id", "")),
            "position": row.get("_pos"), "name": row["_name"], "started": now,
            "before_inv": inv_signature(inventory(obs)), "last_hit": -100.0,
        }
        self.log("animal_transaction", entity=str(row.get("id", "")), kind=row["_name"])

    def animal_command(self, obs: dict, now: float) -> str | None:
        job = self.job
        if not job or job.get("kind") != "animal":
            return None
        target = next((x for x in entities(obs) if str(x.get("id", "")) == str(job.get("entity"))), None)
        inv = inventory(obs)
        weapon = next((x for x in WEAPONS if inv.get(x)), "")
        if not weapon:
            self.job = None
            return None
        if target is None or target.get("dead"):
            self.job = None
            return None
        stage = str(job.get("stage"))
        if stage == "approach":
            if target["_distance"] <= 3.2:
                job["stage"] = "equip"
                return None
            if now - number(job.get("started"), now) > 25.0:
                self.job = None
                return None
            q = target.get("_pos")
            if q and self.allowed("goto"):
                return "goto %.2f %.2f %.2f" % q
            self.job = None
            return None
        if stage == "equip":
            if item_name(self_row(obs).get("held_item")) != weapon:
                job["stage"] = "attack"
                return "equip %s hand" % weapon
            job["stage"] = "attack"
        if job.get("stage") == "attack":
            if target["_distance"] <= 3.4 and now - number(job.get("last_hit"), -100.0) >= 0.85:
                job["last_hit"] = now
                return "attack " + str(job.get("entity")) if self.allowed("attack") else None
            if now - number(job.get("started"), now) > 40.0:
                self.job = None
        return None

    def mode(self, obs: dict, inv: dict[str, int]) -> str:
        tables = self.table_rows(obs)
        if wood_total(inv) == 0 and plank_total(inv) == 0 and not inv.get("crafting_table") and not tables and not self.table_latched:
            return "wood"
        if inv.get("crafting_table") or (not tables and not self.table_latched):
            if not self.table_latched and plank_total(inv) >= 4:
                return "table"
        if not tables and not self.table_latched:
            return "table"
        if not any(inv.get(x) for x in ("wooden_pickaxe",) + tuple(STONE_PICKS)):
            return "wooden_tools"
        if not any(inv.get(x) for x in STONE_PICKS):
            return "stone"
        if food_total(inv) < 3:
            return "food"
        raw = inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0)
        if not inv.get("iron_ingot") and raw == 0:
            return "iron"
        if raw and not inv.get("iron_ingot"):
            return "smelt"
        if inv.get("iron_ingot") and not inv.get("shield"):
            return "shield"
        if not any(k.endswith("_bed") and v for k, v in inv.items()):
            return "bed"
        return "progress"

    def table_route(self, obs: dict, now: float) -> str | None:
        near = self.near_table(obs)
        if near:
            return None
        rows = self.table_rows(obs)
        if rows:
            target = rows[0]
            approach = self.stand_for(obs, target["_pos"], now)
            if approach:
                return approach
            if self.allowed("navigate") and not self.blocked("navigate " + target["_id"], now):
                return "navigate " + target["_id"]
            return "noop"
        if inv := inventory(obs):
            if inv.get("crafting_table"):
                support = self.support_block(obs)
                if support:
                    if item_name(self_row(obs).get("held_item")) != "crafting_table":
                        return "equip crafting_table hand"
                    q = support["_pos"]
                    self.table_wait_until = now + 10.0
                    self.table_hint = (q[0], q[1] + 1.0, q[2]) if q else self.table_hint
                    return "place crafting_table " + support["_id"] + " up"
                return self.choose_exploration(obs, now)
            if self.table_latched:
                if now < self.table_wait_until:
                    return "noop"
                # The block stream can lag. Move to a safe stand near the
                # consumed-table hint; do not craft a duplicate table.
                if self.table_hint:
                    approach = self.stand_for(obs, self.table_hint, now)
                    if approach:
                        return approach
                return self.choose_exploration(obs, now)
        return None

    def craft_action(self, item: str, count: int, table: dict | None = None) -> str:
        action = "craft %s %d" % (item, max(1, int(count)))
        if table:
            action += " " + table["_id"]
        return action

    def resource_or_excavate(self, obs: dict, now: float, names: set[str] | None = None, suffixes: tuple[str, ...] = (), tool: str | None = None, purpose: str = "") -> str | None:
        target = self.resource_target(obs, names, suffixes)
        if target:
            self.begin_job(target, "gather", tool, purpose, now)
            return None
        ground = self.excavation_target(obs, now)
        if ground:
            # Ground is intentionally a separate transaction: its drop is not
            # required, so this can reveal a stone wall immediately.
            self.begin_job(ground, "excavate", tool, "surface excavation", now)
            return None
        return self.choose_exploration(obs, now)

    def compile_action(self, obs: dict, now: float) -> str | None:
        inv = inventory(obs)
        mode = self.mode(obs, inv)
        self.last_mode = mode
        table = self.near_table(obs)
        if mode in {"stone", "iron", "food", "bed", "progress"} and self.macro_focus == "scout" and not self.job:
            # Astra's coarse route choice can request a safe terrain refresh;
            # it cannot bypass the same verified transaction compiler.
            return self.choose_exploration(obs, now)
        if mode == "wood":
            target = self.resource_target(obs, suffixes=WOOD_SUFFIXES)
            if target:
                self.begin_job(target, "gather", None, "exposed wood", now)
                return None
            # No blind manual movement: a safe stand may expose a tree, and
            # ground excavation is a fallback only after the tree search.
            return self.choose_exploration(obs, now)
        if mode == "table":
            # The table itself is the pending workstation; do not delay its
            # placement just because the remaining planks are below the tool
            # reserve.
            if inv.get("crafting_table"):
                return self.table_route(obs, now)
            if plank_total(inv) < 4:
                logs = next((k for k, v in inv.items() if v and k.endswith(WOOD_SUFFIXES)), "")
                if logs:
                    return self.craft_action(logs.replace("_log", "_planks").replace("_stem", "_planks"), 4)
                target = self.resource_target(obs, suffixes=WOOD_SUFFIXES)
                if target:
                    self.begin_job(target, "gather", None, "wood for table", now)
                    return None
                return self.choose_exploration(obs, now)
            return self.craft_action("crafting_table", 1)
        if mode == "wooden_tools":
            if inv.get("crafting_table") and not self.table_rows(obs):
                return self.table_route(obs, now)
            if not table:
                route = self.table_route(obs, now)
                if route:
                    return route
                table = self.near_table(obs)
            logs = next((k for k, v in inv.items() if v and k.endswith(WOOD_SUFFIXES)), "")
            if plank_total(inv) < 3:
                if logs:
                    return self.craft_action(logs.replace("_log", "_planks").replace("_stem", "_planks"), 4)
                target = self.resource_target(obs, suffixes=WOOD_SUFFIXES)
                if target:
                    self.begin_job(target, "gather", None, "extra wood for wooden pick", now)
                    return None
                return self.choose_exploration(obs, now)
            if inv.get("stick", 0) < 2 and plank_total(inv) >= 2:
                return self.craft_action("stick", 4)
            if not any(inv.get(x) for x in PICKS):
                if table:
                    return self.craft_action("wooden_pickaxe", 1, table)
                return self.choose_exploration(obs, now)
            return None
        if mode == "stone":
            # Resource search and excavation happen before workstation travel.
            # This is the critical inversion over the parent controller.
            if inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0) < 3:
                tool = next((x for x in PICKS if inv.get(x)), "wooden_pickaxe")
                return self.resource_or_excavate(obs, now, STONE_BLOCKS, tool=tool, purpose="expose stone for stone pick")
            if inv.get("stick", 0) < 2 and plank_total(inv) >= 2:
                return self.craft_action("stick", 4)
            if not any(inv.get(x) for x in STONE_PICKS):
                if not table:
                    return self.table_route(obs, now) or self.choose_exploration(obs, now)
                return self.craft_action("stone_pickaxe", 1, table)
        if mode == "food":
            me = self_row(obs)
            if number(me.get("food"), 20) <= 12:
                food = best_food(inv)
                if food:
                    if item_name(me.get("held_item")) != food:
                        return "equip %s hand" % food
                    return "eat " + food
            if table and inv.get("hay_block"):
                return self.craft_action("wheat", 9, table)
            if table and inv.get("wheat", 0) >= 3:
                return self.craft_action("bread", min(3, inv["wheat"] // 3), table)
            target = self.resource_target(obs, FOOD_BLOCKS)
            if target:
                self.begin_job(target, "gather", None, "exposed food", now)
                return None
            weapon = next((x for x in WEAPONS if inv.get(x)), "")
            animal = self.animal_target(obs)
            if animal and weapon:
                self.start_animal(obs, animal, now)
                return None
            return self.choose_exploration(obs, now)
        if mode == "iron":
            target = self.resource_target(obs, IRON_BLOCKS)
            if target:
                self.begin_job(target, "gather", "stone_pickaxe", "exposed iron", now)
                return None
            fuel = self.resource_target(obs, FUEL_BLOCKS)
            if fuel and not inv.get("coal") and not inv.get("charcoal"):
                self.begin_job(fuel, "gather", "stone_pickaxe", "exposed furnace fuel", now)
                return None
            return self.resource_or_excavate(obs, now, IRON_BLOCKS, tool="stone_pickaxe", purpose="excavate toward iron")
        if mode == "smelt":
            furnaces = self.furnace_rows(obs)
            if not furnaces and not inv.get("furnace"):
                if inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0) >= 8 and table:
                    return self.craft_action("furnace", 1, table)
                # Mine more stone without leaving the milestone loop.
                return self.resource_or_excavate(obs, now, STONE_BLOCKS, tool="wooden_pickaxe", purpose="more furnace stone")
            if not furnaces and inv.get("furnace"):
                if now < self.furnace_wait_until:
                    return "noop"
                support = self.support_block(obs)
                if support:
                    if item_name(self_row(obs).get("held_item")) != "furnace":
                        return "equip furnace hand"
                    self.furnace_wait_until = now + 10.0
                    q = support["_pos"]
                    self.furnace_hint = (q[0], q[1] + 1.0, q[2]) if q else self.furnace_hint
                    return "place furnace " + support["_id"] + " up"
                return self.choose_exploration(obs, now)
            if furnaces:
                furnace = min(furnaces, key=lambda x: x["_distance"])
                if furnace["_distance"] > 4.8:
                    return self.stand_for(obs, furnace["_pos"], now) or "navigate " + furnace["_id"]
                fid = furnace["_id"]
                raw = inv.get("raw_iron", 0)
                fuel = next((x for x in ("coal", "charcoal") if inv.get(x)), "")
                fuel = fuel or next((x for x in inv if x.endswith("_planks") and inv.get(x)), "")
                if raw and fuel and fid not in self.smelt_started:
                    self.smelt_started[fid] = now
                    return "furnace_start %s raw_iron %d %s %d" % (fid, min(3, raw), fuel, min(3, inv[fuel]))
                if fid in self.smelt_started and now - self.smelt_started[fid] >= 32.0:
                    return "furnace_take " + fid
                return "noop"
        if mode == "shield":
            if plank_total(inv) < 6:
                logs = next((k for k, v in inv.items() if v and k.endswith(WOOD_SUFFIXES)), "")
                if logs:
                    return self.craft_action(logs.replace("_log", "_planks").replace("_stem", "_planks"), 8)
                return self.resource_or_excavate(obs, now, suffixes=WOOD_SUFFIXES, purpose="wood for shield")
            if table:
                return self.craft_action("shield", 1, table)
            return self.table_route(obs, now) or self.choose_exploration(obs, now)
        if mode == "bed":
            wool = next((x for x, v in inv.items() if x.endswith("_wool") and v >= 3), "")
            if wool and plank_total(inv) >= 3 and table:
                return self.craft_action(wool.removesuffix("_wool") + "_bed", 1, table)
            bed = self.resource_target(obs, suffixes=("_bed",))
            if bed:
                self.begin_job(bed, "gather", None, "observed bed", now)
                return None
            animal = self.animal_target(obs, sheep_first=True)
            if animal and any(inv.get(x) for x in WEAPONS):
                self.start_animal(obs, animal, now)
                return None
            return self.choose_exploration(obs, now)
        # Once the local preparation is complete, maintain hunger and continue
        # revealing useful blocks rather than waiting in place.
        if number(self_row(obs).get("food"), 20) <= 12 and best_food(inv):
            food = best_food(inv)
            return ("equip %s hand" % food) if item_name(self_row(obs).get("held_item")) != food else "eat " + food
        return self.resource_or_excavate(obs, now, IRON_BLOCKS, tool="stone_pickaxe", purpose="continue exposed-resource search")

    def poll_astra(self, now: float) -> None:
        if self.macro_future is None or not self.macro_future.done():
            return
        future = self.macro_future
        self.macro_future = None
        self.macro_finished = now
        try:
            result = future.result()
            text = str(getattr(result, "text", "")).lower()
            first = text.splitlines()[0] if text.splitlines() else text
            allowed = ("excavate", "scout", "food", "bed", "iron", "shelter", "progress")
            for token in allowed:
                if re.search(r"\b" + re.escape(token) + r"\b", first):
                    self.macro_focus = token
                    self.log("astra_route", focus=token)
                    return
            self.log("astra_unusable", chars=len(text))
        except (BudgetExceeded, ModelUnavailable):
            # Boundary closure is a normal end for background advice.
            self.log("astra_closed")
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)

    def maybe_astra(self, obs: dict, now: float) -> None:
        self.poll_astra(now)
        if self.macro_future is not None and not self.macro_future.done():
            return
        rem = self.ctx.budget.remaining()
        if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 250:
            return
        if now - self.last_progress_at < 55.0 or now - self.macro_started < 150.0:
            return
        inv = inventory(obs)
        nearby = []
        for row in blocks(obs, "b"):
            if row["_name"] in STONE_BLOCKS | IRON_BLOCKS | FUEL_BLOCKS | GROUND_BLOCKS | FOOD_BLOCKS:
                nearby.append({"name": row["_name"], "visible": bool(row.get("visible")), "in_reach": bool(row.get("in_reach")), "distance_band": "near" if row["_distance"] < 8 else "far"})
        state = {
            "phase": self.last_mode,
            "inventory": {k: v for k, v in sorted(inv.items()) if v and (k in FOODS or k in {"crafting_table", "wooden_pickaxe", "stone_pickaxe", "raw_iron", "iron_ingot", "shield", "furnace", "coal", "charcoal"} or k.endswith(("_log", "_planks", "_wool")))},
            "nearby_exposed": nearby[:20],
            "recent_exploration_round": self.exploration_round,
            "rule": "Return exactly one first-line token: excavate, scout, food, bed, iron, shelter, or progress. Do not emit a command or coordinate. Code verifies every action.",
        }
        prompt = (
            "You are the slow route adviser for an ordinary Minecraft Java survival bot. "
            "The bot has safe semantic actions but cannot see hidden blocks. Choose the next coarse "
            "search mode from the supplied options. Prefer excavate when a safe exposed surface can "
            "be opened for stone, scout when the local candidate window lacks useful blocks, and food "
            "or bed only when the inventory/nearby evidence supports it. Never assume a seed, structure, "
            "ore coordinate, or command.\n" + json.dumps(state, separators=(",", ":"))
        )
        self.macro_started = now
        try:
            self.macro_future = self.ctx.spawn(lambda p=prompt: self.ctx.think(p, timeout_s=95.0))
            self.log("astra_route_started", phase=self.last_mode)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("astra_start_failed", error=type(exc).__name__)
            self.macro_future = None

    def run_episode(self, game: Any) -> None:
        self.names = set(getattr(getattr(game, "action_spec", None), "names", ()) or ())
        self.safety = SafetyAdvisor(self.ctx, self.names, self.log)
        self.last_progress_at = self.now()
        while not self.ctx.done:
            now = self.now()
            obs = parse_observation(self.ctx.observe())
            if not obs:
                self.post("noop", {}, now, "missing_observation")
                self.ctx.wait_for_tick()
                continue
            self._last_obs = obs
            self.process_feedback(obs, now)
            self.observe_progress(obs, now)
            me = self_row(obs)
            if me.get("dead"):
                self.job = None
                self.latch = None
                if not self.dead_seen:
                    self.dead_seen = True
                    self.post("respawn", obs, now, "ordinary_respawn")
                else:
                    self.post("noop", obs, now, "await_respawn")
                self.ctx.wait_for_tick()
                continue
            self.dead_seen = False
            if number(obs.get("observation_age_s"), 0) > 3.0:
                self.post("cancel" if has_task(obs) else "noop", obs, now, "stale_observation")
                self.ctx.wait_for_tick()
                continue
            inv = inventory(obs)
            tactical = self.safety.decide(obs, inv, now) if self.safety else None
            if tactical:
                # A replacement action must cancel an owned bounded job first.
                if has_task(obs) and tactical not in {"noop", "shield on"}:
                    self.post("cancel" if self.allowed("cancel") else "noop", obs, now, "hazard_interrupt")
                    self.ctx.wait_for_tick()
                    continue
                if has_task(obs) and tactical == "shield on":
                    self.post("cancel" if self.allowed("cancel") else "noop", obs, now, "shield_interrupt")
                    self.ctx.wait_for_tick()
                    continue
                self.post(tactical, obs, now, "safety_policy")
                self.ctx.wait_for_tick()
                continue
            if has_task(obs):
                self.post("noop", obs, now, "continue_job")
                self.ctx.wait_for_tick()
                continue
            if self.latch is not None:
                self.post("noop", obs, now, "await_postcondition")
                self.ctx.wait_for_tick()
                continue
            self.maybe_astra(obs, now)
            if self.job:
                action = self.animal_command(obs, now) if self.job.get("kind") == "animal" else self.job_command(obs, now)
                if action:
                    self.post(action, obs, now, "transaction_step")
                else:
                    self.post("noop", obs, now, "transaction_advance")
                self.ctx.wait_for_tick()
                continue
            action = self.compile_action(obs, now)
            if action is None:
                # A compile call may have just created a transaction.
                self.post("noop", obs, now, "transaction_started")
            else:
                self.post(action, obs, now, "milestone_compiler")
            self.ctx.wait_for_tick()


def row_is_visible_reachable(row: dict) -> bool:
    return bool(row.get("visible") or row.get("in_reach"))


def build(ctx: Any) -> ExcavationController:
    return ExcavationController(ctx)
