"""Candidate 0012: direct block transactions and deliberate frontier escapes.

The previous controllers treated a successful navigate as resource progress and
then delegated the important part (mining and pickup) to ``gather``.  This
controller makes a block operation explicit: approach the observed block,
select the tool, aim, dig one block, and walk to the resulting item entity.
When the bounded candidate window contains only water or an unreachable tree,
it temporarily uses a short, cardinal manual excursion to reveal a new slice
of terrain instead of revisiting safe stands forever.

All coordinates and item names come from the current observation.  There is no
seed or map knowledge here; the small progression state machine only encodes
vanilla recipes and postconditions.
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
STONE_NAMES = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON_NAMES = {"iron_ore", "deepslate_iron_ore", "raw_iron", "iron_block"}
COAL_NAMES = {"coal_ore", "deepslate_coal_ore", "coal_block"}
FOOD_BLOCK_NAMES = {
    "hay_block", "melon", "sweet_berry_bush", "wheat", "carrots",
    "carrot", "potatoes", "potato", "beetroots", "beetroot",
}
FOODS = {
    "apple", "bread", "baked_potato", "carrot", "golden_carrot",
    "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
    "cooked_rabbit", "cooked_cod", "cooked_salmon", "sweet_berries",
    "melon_slice", "beef", "porkchop", "mutton", "chicken", "rabbit",
    "cod", "salmon", "potato", "beetroot", "wheat", "mushroom_stew",
}
ANIMALS = {"cow", "pig", "sheep", "chicken", "rabbit"}
WEAPONS = ("iron_sword", "stone_sword", "diamond_sword", "stone_axe", "wooden_sword", "wooden_axe")
PICKAXES = ("netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
STONE_OR_BETTER = {"stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"}
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
SUPPORT_NAMES = {
    "stone", "cobblestone", "deepslate", "cobbled_deepslate", "dirt",
    "grass_block", "coarse_dirt", "sand", "gravel", "terracotta",
    "netherrack", "blackstone", "oak_log", "birch_log", "spruce_log",
    "jungle_log", "acacia_log", "dark_oak_log", "mangrove_log", "chest",
    "crafting_table", "furnace", "podzol", "mycelium", "sandstone",
    "red_sand", "snow_block", "clay", "moss_block", "basalt",
}
FAIL_WORDS = (
    "failed", "error", "unreachable", "timeout", "cannot", "rejected",
    "illegal", "not enough", "not in reach", "no path", "busy", "invalid",
    "unknown", "out of range",
)


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def dict_rows(value: Any) -> list[dict]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def clean_item(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", "")))
    if not isinstance(value, str):
        return ""
    value = value.removeprefix("minecraft:")
    return value if ITEM_RE.fullmatch(value) else ""


def point(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result = tuple(number(x, math.inf) for x in value[:3])
    return result if all(math.isfinite(x) for x in result) else None


def distance(a: Any, b: Any) -> float:
    pa, pb = point(a), point(b)
    if pa is None or pb is None:
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def self_row(obs: dict) -> dict:
    return mapping(obs.get("self"))


def self_position(obs: dict) -> tuple[float, float, float] | None:
    return point(self_row(obs).get("position"))


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


def inventory(obs: dict) -> dict[str, int]:
    """Read the audited personal inventory, never a cursor or container."""
    source = obs.get("inventory")
    if source is None or source == {} or source == []:
        source = obs.get("inventory_authoritative", source)
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source["items"]
    answer: dict[str, int] = {}
    if isinstance(source, dict):
        for key, raw in source.items():
            item = clean_item(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(number(raw_count)))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    elif isinstance(source, list):
        for raw in dict_rows(source):
            item = clean_item(raw)
            count = max(0, int(number(raw.get("count", 0))))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    # The authoritative audit can be momentarily empty at join.  The labelled
    # client view is a safe fallback because it is still personal inventory.
    if not answer and isinstance(obs.get("inventory_client"), dict):
        for key, raw in obs["inventory_client"].items():
            item = clean_item(key)
            count = max(0, int(number(raw.get("count", 0) if isinstance(raw, dict) else raw)))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    return answer


def item_total(inv: dict[str, int], names: set[str] | None = None, suffixes: tuple[str, ...] = ()) -> int:
    if names is None:
        return sum(inv.values())
    return sum(v for k, v in inv.items() if k in names or any(k.endswith(s) for s in suffixes))


def food_total(inv: dict[str, int]) -> int:
    return sum(v for k, v in inv.items() if k in FOODS)


def wood_total(inv: dict[str, int]) -> int:
    return item_total(inv, set(), WOOD_SUFFIXES)


def plank_total(inv: dict[str, int]) -> int:
    return sum(v for k, v in inv.items() if k.endswith("_planks"))


def signature(inv: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(k), int(v)) for k, v in inv.items() if v))


def best_food(inv: dict[str, int]) -> str:
    order = (
        "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
        "bread", "baked_potato", "carrot", "apple", "sweet_berries",
        "melon_slice", "beef", "porkchop", "mutton", "chicken", "rabbit",
        "cod", "salmon", "potato", "wheat",
    )
    return next((x for x in order if inv.get(x, 0)), "")


def block_name(row: dict) -> str:
    return clean_item(row.get("name", row.get("block", "")))


def entity_name(row: dict) -> str:
    return clean_item(row.get("name", row.get("type", "")))


def valid_id(value: Any, prefix: str | None = None) -> str:
    value = value if isinstance(value, str) else ""
    if not ID_RE.fullmatch(value):
        return ""
    return value if prefix is None or value.startswith(prefix + ":") else ""


def id_point(value: Any) -> tuple[float, float, float] | None:
    ident = valid_id(value)
    if not ident:
        return None
    try:
        parts = ident[2:].split(",")
        if len(parts) != 3:
            return None
        result = tuple(float(x) for x in parts)
        return result if all(math.isfinite(x) for x in result) else None
    except (TypeError, ValueError):
        return None


def has_task(obs: dict) -> bool:
    task = obs.get("task")
    return task is not None and task != "" and task != {}


def rows_with_meta(obs: dict, kind: str | None = None) -> list[dict]:
    here = self_position(obs)
    result = []
    for raw in dict_rows(obs.get("candidates")):
        ident = valid_id(raw.get("id"), kind)
        if not ident:
            continue
        row = dict(raw)
        row["_id"] = ident
        row["_name"] = block_name(raw)
        row["_pos"] = point(raw.get("position"))
        row["_distance"] = number(raw.get("distance"), distance(here, row["_pos"]) if here else 999.0)
        result.append(row)
    result.sort(key=lambda x: x["_distance"])
    return result


def entities_with_meta(obs: dict) -> list[dict]:
    here = self_position(obs)
    result = []
    source = obs.get("entities", obs.get("nearby_entities", []))
    for raw in dict_rows(source):
        row = dict(raw)
        row["_name"] = entity_name(raw)
        row["_pos"] = point(raw.get("position"))
        row["_distance"] = number(raw.get("distance"), distance(here, row["_pos"]) if here else 999.0)
        result.append(row)
    return result


def progress_key(obs: dict) -> str:
    p = mapping(obs.get("progress"))
    return json.dumps(p, sort_keys=True, default=str)


def item_present_entity(row: dict) -> bool:
    return row.get("_name") in {"item", "item_stack", "item_entity"}


def is_solid(row: dict) -> bool:
    name = row.get("_name", "")
    if name in HAZARDS or name in {"air", "water", "bubble_column", "lava"}:
        return False
    return name in SUPPORT_NAMES or name.endswith("_log") or name.endswith("_planks")


class ThreatPolicy:
    """Jev judges only an immediate threat; code supplies exact legal actions."""

    def __init__(self, ctx: Any, names: set[str], log):
        self.ctx = ctx
        self.names = names
        self.log = log
        self.future = None
        self.signature: tuple = ()
        self.submitted = -100.0
        self.cached: tuple[str, float, tuple] | None = None

    def allowed(self, action: str) -> bool:
        verb = action.split(" ", 1)[0] if action else ""
        return verb in self.names

    def threat_rows(self, obs: dict) -> list[dict]:
        me = self_row(obs)
        threats = []
        if me.get("is_in_lava") or me.get("is_in_fire"):
            threats.append({"kind": "lava_or_fire", "id": "self", "distance": 0.0, "urgent": True})
        if me.get("is_in_water") and number(me.get("oxygen"), 20) < 6:
            threats.append({"kind": "drowning", "id": "self", "distance": 0.0, "urgent": True})
        if number(me.get("fall_distance_estimate"), 0) > 3:
            threats.append({"kind": "fall", "id": "self", "distance": 0.0, "urgent": True})
        for row in rows_with_meta(obs, "b"):
            if row["_name"] in HAZARDS and row["_distance"] <= 4.5:
                threats.append({"kind": row["_name"], "id": row["_id"], "distance": row["_distance"], "urgent": True})
        for row in entities_with_meta(obs):
            if (row.get("hostile") or row.get("conditional_threat") or row["_name"] in HOSTILES) and row.get("visible") and row["_distance"] <= 8.0:
                threats.append({"kind": row["_name"], "id": str(row.get("id", "")), "distance": row["_distance"], "urgent": row["_distance"] <= 3.2})
        return threats

    def fallback(self, obs: dict, threats: list[dict], inv: dict[str, int]) -> str:
        me = self_row(obs)
        if me.get("is_in_lava") or me.get("is_in_fire"):
            return "cancel" if has_task(obs) and self.allowed("cancel") else ("controls forward,jump" if self.allowed("controls") else "noop")
        if me.get("is_in_water") and number(me.get("oxygen"), 20) < 6:
            return "cancel" if has_task(obs) and self.allowed("cancel") else ("controls jump" if self.allowed("controls") else "noop")
        weapon = next((x for x in WEAPONS if inv.get(x)), "")
        close = [x for x in threats if x.get("id") != "self" and number(x.get("distance"), 99) <= 3.2]
        if close and weapon:
            held = clean_item(me.get("held_item"))
            if held != weapon and self.allowed("equip"):
                return "equip %s hand" % weapon
            if close[0].get("id") and self.allowed("attack"):
                return "attack " + str(close[0]["id"])
        if inv.get("shield") and self.allowed("shield"):
            return "shield on"
        if self.allowed("controls"):
            return "controls back,jump"
        return "noop"

    def menu(self, obs: dict, threats: list[dict], inv: dict[str, int]) -> list[str]:
        options = []
        fallback = self.fallback(obs, threats, inv)
        if fallback and self.allowed(fallback):
            options.append(fallback)
        if inv.get("shield") and self.allowed("shield"):
            options.append("shield on")
        weapon = next((x for x in WEAPONS if inv.get(x)), "")
        me = self_row(obs)
        for threat in threats:
            if threat.get("id") == "self" or not threat.get("id"):
                continue
            if number(threat.get("distance"), 99) <= 3.2 and weapon:
                if clean_item(me.get("held_item")) != weapon:
                    options.append("equip %s hand" % weapon)
                options.append("attack " + str(threat["id"]))
        if self.allowed("cancel") and has_task(obs):
            options.append("cancel")
        options.append("noop")
        result = []
        for action in options:
            if action not in result and self.allowed(action):
                result.append(action)
        return result[:8]

    def decide(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        threats = self.threat_rows(obs)
        if not threats:
            self.cached = None
            self.signature = ()
            return None
        options = self.menu(obs, threats, inv)
        if not options:
            return None
        sig = tuple(sorted((str(x.get("kind")), str(x.get("id")), "close" if number(x.get("distance"), 99) <= 3.2 else "near") for x in threats))
        if self.cached and self.cached[2] == sig and now - self.submitted < 2.5:
            return self.cached[0] if self.cached[0] in options else options[0]
        # Submit at most one short request for a stable hazard signature.
        if self.future is None and self.ctx.budget.remaining().get("system1_calls", 0) > 2:
            state = {
                "purpose": "Choose one immediate survival response. Code owns legality and exact coordinates.",
                "self": {
                    "health_band": "critical" if number(self_row(obs).get("health"), 20) < 7 else "low" if number(self_row(obs).get("health"), 20) < 12 else "stable",
                    "food_band": "low" if number(self_row(obs).get("food"), 20) < 8 else "fed",
                    "dimension": str(self_row(obs).get("dimension", "")),
                    "lava_fire": bool(self_row(obs).get("is_in_lava") or self_row(obs).get("is_in_fire")),
                },
                "threats": [{"kind": str(x.get("kind")), "range": "touching" if number(x.get("distance"), 99) <= 3.2 else "near"} for x in threats[:6]],
                "legal_actions": options,
                "rule": "Return only a supplied legal_actions value; never invent an id. Survival beats resource work.",
            }
            questions = {
                "response": Choice(
                    "Select the safest supplied immediate response for the current threats.",
                    {x: "Exact legal response: " + x for x in options},
                ),
                "continue_safe": Noul(
                    "Is it safe to continue the current work without first using a defensive response?",
                    {"true": "No visible immediate threat makes continuing safe.", "false": "A visible or self hazard requires defense or retreat first."},
                ),
            }
            try:
                self.future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=2.0))
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
                if value in options and probability >= 0.55 and self.signature == sig:
                    self.cached = (value, probability, sig)
                    self.log("jev_threat", action=value, probability=round(probability, 3))
                    return value
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:
                self.log("jev_invalid", error=type(exc).__name__)
        # A deterministic response is essential while Jev is in flight.
        return self.fallback(obs, threats, inv)


class DirectController:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.names: set[str] = set()
        self.threats: ThreatPolicy | None = None
        self.job: dict | None = None
        self.one: dict | None = None
        self.failed: dict[str, float] = {}
        self.completed_blocks: set[str] = set()
        self.last_feedback = ""
        self.last_progress = 0.0
        self.last_position: tuple[float, float, float] | None = None
        self.last_inventory: tuple[tuple[str, int], ...] = ()
        self.last_progress_key = ""
        self.last_posted = "noop"
        self.last_post_time = -100.0
        self.cancel_pending = False
        self.dead_seen = False
        self.table_position: tuple[float, float, float] | None = None
        self.furnace_position: tuple[float, float, float] | None = None
        self.table_expected_until = -1.0
        self.furnace_expected_until = -1.0
        self.smelting: dict[str, float] = {}
        self.manual: dict | None = None
        self.manual_index = 0
        self.manual_started = -100.0
        self.manual_failures = 0
        self.last_mining_attempt = -100.0
        self.mode = "wood"

    def log(self, event: str, **fields: Any) -> None:
        self.ctx.log(event, **fields)

    def allowed(self, action: str) -> bool:
        verb = action.split(" ", 1)[0] if action else ""
        return verb in self.names or (verb == "noop" and not self.names)

    def now(self) -> float:
        return self.ctx.clock.now()

    def mark_failed(self, key: str, now: float, seconds: float = 120.0, reason: str = "") -> None:
        if not key or key in {"noop", "cancel"}:
            return
        self.failed[key] = max(self.failed.get(key, -1.0), now + seconds)
        self.log("target_failed", target=key, blocked_s=seconds, reason=reason)

    def blocked(self, key: str, now: float) -> bool:
        return now < self.failed.get(key, -1.0)

    def feedback_text(self, obs: dict) -> str:
        vals = [obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")]
        return json.dumps(vals, sort_keys=True, default=str)

    def process_feedback(self, obs: dict, now: float) -> None:
        marker = self.feedback_text(obs)
        if marker == self.last_feedback:
            return
        self.last_feedback = marker
        low = marker.lower()
        if self.one and any(word in low for word in FAIL_WORDS):
            action = str(self.one.get("action", ""))
            self.log("action_feedback_failure", action=action)
            self.mark_failed(action, now, 90.0 if action.startswith("navigate") else 180.0, "adapter_feedback")
            self.one = None
            self.cancel_pending = has_task(obs)
            if self.job and self.job.get("kind") == "mine" and action.startswith(("navigate", "dig")):
                self.mark_failed(str(self.job.get("target", "")), now, 180.0, "mine_feedback")
                self.job = None

    def record_progress(self, obs: dict, now: float) -> None:
        inv = inventory(obs)
        inv_sig = signature(inv)
        pkey = progress_key(obs)
        pos = self_position(obs)
        if inv_sig != self.last_inventory or pkey != self.last_progress_key:
            self.last_progress = now
            self.last_inventory = inv_sig
            self.last_progress_key = pkey
        if pos is not None:
            self.last_position = pos
        # Learn placed workstations from current candidates.  This is preferred
        # to trusting an item-consumption event that may precede block updates.
        for row in rows_with_meta(obs, "b"):
            if row["_name"] == "crafting_table":
                self.table_position = row["_pos"] or self.table_position
                self.table_expected_until = -1.0
            elif row["_name"] == "furnace":
                self.furnace_position = row["_pos"] or self.furnace_position
                self.furnace_expected_until = -1.0

    def post(self, action: str, obs: dict, reason: str = "") -> str:
        if self.ctx.done:
            return "noop"
        action = str(action or "noop").strip() or "noop"
        if not self.allowed(action):
            action = "noop"
        verb = action.split(" ", 1)[0]
        # Parameterized one-shot commands are latched.  A noop barrier is the
        # only safe way to deliberately repeat one after a failed postcondition.
        if action == self.last_posted and verb not in {"noop", "controls", "attack", "shield"}:
            action = "noop"
            verb = "noop"
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("act_exception", action=action, error=type(exc).__name__)
            action = "noop"
            verb = "noop"
            try:
                self.ctx.act("noop")
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                pass
        self.last_posted = action
        self.last_post_time = self.now()
        if action != "noop":
            self.log("posted", action=action, reason=reason, mode=self.mode)
        if action == "cancel":
            self.one = None
            self.cancel_pending = False
        elif action not in {"noop", "controls", "attack", "shield"}:
            self.one = {"action": action, "started": self.now(), "before_inv": signature(inventory(obs)), "before_pos": self_position(obs)}
        return action

    def wait_tick(self) -> None:
        try:
            self.ctx.wait_for_tick()
        except BudgetExceeded:
            return

    def current_block(self, obs: dict, ident: str) -> dict | None:
        return next((x for x in rows_with_meta(obs, "b") if x["_id"] == ident), None)

    def near_table(self, obs: dict) -> dict | None:
        rows = [x for x in rows_with_meta(obs, "b") if x["_name"] == "crafting_table" and x["_distance"] <= 4.8]
        rows.sort(key=lambda x: x["_distance"])
        return rows[0] if rows else None

    def table_rows(self, obs: dict) -> list[dict]:
        rows = [x for x in rows_with_meta(obs, "b") if x["_name"] == "crafting_table"]
        rows.sort(key=lambda x: x["_distance"])
        return rows

    def furnace_rows(self, obs: dict) -> list[dict]:
        rows = [x for x in rows_with_meta(obs, "b") if x["_name"] == "furnace"]
        rows.sort(key=lambda x: x["_distance"])
        return rows

    def support(self, obs: dict) -> dict | None:
        here = self_position(obs)
        if here is None:
            return None
        floor_y = math.floor(here[1]) - 1
        choices = []
        for row in rows_with_meta(obs, "b"):
            q = row["_pos"]
            if not q or not row.get("in_reach") or row.get("visible") is False or not is_solid(row):
                continue
            if int(math.floor(q[1])) != floor_y:
                continue
            if int(math.floor(q[0])) == math.floor(here[0]) and int(math.floor(q[2])) == math.floor(here[2]):
                continue
            choices.append(row)
        choices.sort(key=lambda x: x["_distance"])
        return choices[0] if choices else None

    def pick_target(self, obs: dict, names: set[str] | None = None, suffixes: tuple[str, ...] = (), allow_hidden_close: bool = False) -> dict | None:
        now = self.now()
        candidates = []
        here = self_position(obs)
        for row in rows_with_meta(obs, "b"):
            name = row["_name"]
            if names is None:
                if not any(name.endswith(s) for s in suffixes):
                    continue
            elif name not in names and not any(name.endswith(s) for s in suffixes):
                continue
            if row.get("harvestable") is False or row["_id"] in self.completed_blocks:
                continue
            if self.blocked(row["_id"], now):
                continue
            if name in HAZARDS or name in {"water", "lava", "air", "bubble_column"}:
                continue
            visible = bool(row.get("visible"))
            reachable = bool(row.get("in_reach"))
            if not visible:
                continue
            # Never deliberately mine the block directly below the player's feet.
            if here and row["_pos"] and tuple(map(math.floor, row["_pos"])) == (math.floor(here[0]), math.floor(here[1]) - 1, math.floor(here[2])):
                continue
            candidates.append(row)
        candidates.sort(key=lambda x: (
            0 if x.get("in_reach") else 1,
            0 if x.get("visible") else 1,
            x["_distance"],
        ))
        return candidates[0] if candidates else None

    def find_drop(self, obs: dict, target_pos: tuple[float, float, float] | None = None) -> dict | None:
        choices = []
        for row in entities_with_meta(obs):
            if not item_present_entity(row) or not row["_pos"] or row.get("dead"):
                continue
            if row["_distance"] <= 10.0 and (target_pos is None or distance(row["_pos"], target_pos) <= 8.0):
                choices.append(row)
        choices.sort(key=lambda x: x["_distance"])
        return choices[0] if choices else None

    def begin_mine(self, obs: dict, target: dict, tool: str | None, purpose: str, count: int = 1) -> None:
        self.job = {
            "kind": "mine", "stage": "approach", "target": target["_id"],
            "name": target["_name"], "pos": target["_pos"], "tool": tool,
            "purpose": purpose, "before_inv": signature(inventory(obs)),
            "before_count": inventory(obs).get(target["_name"], 0), "count": count,
            "started": self.now(), "approaches": 0, "dug": False,
        }
        self.last_mining_attempt = self.now()
        self.log("direct_mine_begin", target=target["_id"], block=target["_name"], purpose=purpose)

    def fail_job(self, now: float, reason: str) -> None:
        if self.job:
            target = str(self.job.get("target", ""))
            if target:
                self.mark_failed(target, now, 180.0, reason)
            self.log("direct_job_failed", kind=self.job.get("kind"), target=target, reason=reason)
        self.job = None
        self.one = None
        self.cancel_pending = True
        self.manual_failures += 1

    def job_action(self, obs: dict, now: float) -> str | None:
        """Advance a direct mine or animal transaction by one legal command."""
        if not self.job:
            return None
        kind = self.job.get("kind")
        if kind == "mine":
            return self.mine_action(obs, now)
        if kind == "animal":
            return self.animal_action(obs, now)
        if kind == "eat":
            return self.eat_action(obs, now)
        return None

    def mine_action(self, obs: dict, now: float) -> str | None:
        job = self.job
        ident = str(job.get("target", ""))
        row = self.current_block(obs, ident)
        stage = str(job.get("stage", "approach"))
        inv = inventory(obs)
        if stage == "approach":
            if row is None:
                self.fail_job(now, "target_disappeared_before_approach")
                return None
            if row.get("in_reach"):
                job["stage"] = "equip"
                return None
            if job.get("approaches", 0) >= 1 and now - number(job.get("started"), now) > 18.0:
                self.fail_job(now, "approach_timeout")
                return None
            action = "navigate " + ident
            job["approaches"] = int(job.get("approaches", 0)) + 1
            return action if self.allowed(action) else None
        if stage == "equip":
            tool = clean_item(job.get("tool"))
            held = clean_item(self_row(obs).get("held_item"))
            if tool and inv.get(tool, 0) and held != tool:
                return "equip %s hand" % tool
            if tool and not inv.get(tool, 0):
                self.fail_job(now, "tool_missing")
                return None
            job["stage"] = "look"
            return None
        if stage == "look":
            q = row["_pos"] if row else job.get("pos")
            if not q:
                self.fail_job(now, "target_position_missing")
                return None
            job["stage"] = "dig"
            return "look_at %.2f %.2f %.2f" % q if self.allowed("look_at") else None
        if stage == "dig":
            row = self.current_block(obs, ident)
            if row is None:
                job["stage"] = "pickup"
                return None
            if not (row.get("visible") and row.get("in_reach")):
                self.fail_job(now, "target_not_visible_and_in_reach")
                return None
            q = row["_pos"] or job.get("pos")
            if not q:
                self.fail_job(now, "target_position_missing")
                return None
            job["stage"] = "pickup"
            job["dug"] = True
            coords = tuple(int(round(x)) for x in q)
            return "dig %d %d %d" % coords if self.allowed("dig") else None
        if stage == "pickup":
            # Direct dig has no pickup automation.  First try a feet-safe
            # coordinate one block below the item entity (entity Y is usually
            # its center, not a legal player feet position), then use a short
            # manual sweep if that bounded goto cannot path there.
            before = job.get("before_inv", ())
            if signature(inv) != before or (job.get("name") and inv.get(str(job.get("name")), 0) > int(job.get("before_count", 0))):
                self.completed_blocks.add(ident)
                self.last_progress = now
                self.log("direct_mine_confirmed", target=ident, block=job.get("name"), method="inventory")
                self.job = None
                return None
            if row is not None and now - number(job.get("started"), now) < 7.0:
                return None
            drop = self.find_drop(obs, job.get("pos"))
            q = drop.get("_pos") if drop else job.get("pos")
            if q and not job.get("navigate_attempted") and self.allowed("navigate"):
                stands = []
                for stand in rows_with_meta(obs, "s"):
                    if stand.get("_pos") and distance(stand["_pos"], q) <= 4.5 and not self.blocked("navigate " + stand["_id"], now):
                        stands.append(stand)
                stands.sort(key=lambda x: distance(x["_pos"], q))
                if stands:
                    job["navigate_attempted"] = True
                    job["stage"] = "collect_navigate"
                    return "navigate " + stands[0]["_id"]
            if q and self.allowed("goto") and not job.get("goto_attempted"):
                job["goto_attempted"] = True
                job["stage"] = "collect_goto"
                return "goto %.2f %.2f %.2f" % (q[0], q[1] - 1.0, q[2])
            job["stage"] = "collect_manual"
            job["collect_started"] = now
            job["collect_phase"] = 0
            return "controls forward,jump" if self.allowed("controls") else None
        if stage == "collect_navigate":
            if signature(inv) != job.get("before_inv", ()):
                self.completed_blocks.add(ident)
                self.last_progress = now
                self.log("direct_mine_confirmed", target=ident, block=job.get("name"), method="navigate_pickup")
                self.job = None
                return None
            job["stage"] = "collect_manual"
            job["collect_started"] = now
            job["collect_phase"] = 0
            return None
        if stage == "collect_manual":
            if signature(inv) != job.get("before_inv", ()):
                self.completed_blocks.add(ident)
                self.last_progress = now
                self.log("direct_mine_confirmed", target=ident, block=job.get("name"), method="manual_pickup")
                self.job = None
                return None
            elapsed = now - number(job.get("collect_started"), now)
            if elapsed < 2.5:
                return "controls forward,jump" if self.allowed("controls") else None
            phase = int(job.get("collect_phase", 0))
            if phase == 0:
                job["collect_phase"] = 1
                job["collect_started"] = now
                return "controls back,jump" if self.allowed("controls") else None
            if phase == 1:
                job["collect_phase"] = 2
                job["collect_started"] = now
                return "controls left,jump" if self.allowed("controls") else None
            self.fail_job(now, "pickup_timeout")
            return "controls none" if self.allowed("controls") else None
        return None

    def start_animal(self, obs: dict, animal: dict) -> None:
        self.job = {
            "kind": "animal", "stage": "approach", "id": str(animal.get("id", "")),
            "name": animal.get("_name", ""), "pos": animal.get("_pos"),
            "before_inv": signature(inventory(obs)), "started": self.now(), "hits": 0,
        }
        self.log("animal_begin", entity=str(animal.get("id", "")), kind=animal.get("_name", ""))

    def animal_action(self, obs: dict, now: float) -> str | None:
        job = self.job
        inv = inventory(obs)
        target = next((x for x in entities_with_meta(obs) if str(x.get("id", "")) == str(job.get("id", ""))), None)
        weapon = next((x for x in WEAPONS if inv.get(x)), "")
        if not weapon:
            self.job = None
            return None
        if target is None or target.get("dead"):
            if signature(inv) != job.get("before_inv"):
                self.last_progress = now
            self.job = None
            return None
        stage = str(job.get("stage", "approach"))
        if stage == "approach":
            if target["_distance"] <= 3.1:
                job["stage"] = "equip"
                return None
            q = target.get("_pos")
            if q and now - number(job.get("started"), now) < 18.0:
                return "goto %.2f %.2f %.2f" % q if self.allowed("goto") else None
            self.job = None
            return None
        if stage == "equip":
            if clean_item(self_row(obs).get("held_item")) != weapon:
                return "equip %s hand" % weapon
            job["stage"] = "attack"
        if job.get("stage") == "attack":
            if target["_distance"] <= 3.3 and now - number(job.get("last_hit"), -100.0) >= 0.85:
                job["last_hit"] = now
                job["hits"] = int(job.get("hits", 0)) + 1
                return "attack " + str(job.get("id")) if self.allowed("attack") else None
            if now - number(job.get("started"), now) > 30.0:
                self.job = None
            return None
        return None

    def eat_action(self, obs: dict, now: float) -> str | None:
        item = clean_item(self.job.get("item")) if self.job else ""
        if not item or not inventory(obs).get(item):
            self.job = None
            return None
        if clean_item(self_row(obs).get("held_item")) != item:
            return "equip %s hand" % item
        self.job["stage"] = "eating"
        return "eat " + item

    def update_one(self, obs: dict, now: float) -> None:
        if not self.one:
            return
        action = str(self.one.get("action", ""))
        verb = action.split(" ", 1)[0]
        age = now - number(self.one.get("started"), now)
        inv_sig = signature(inventory(obs))
        before = self.one.get("before_inv", ())
        me = self_row(obs)
        if verb == "look_at":
            if age >= 0.45:
                self.one = None
            return
        if verb in {"navigate", "goto"}:
            moved = distance(self.one.get("before_pos"), self_position(obs)) >= 0.7
            if moved or age >= (15.5 if verb == "navigate" else 15.0):
                self.one = None
            return
        if verb == "equip":
            wanted = clean_item(action.split()[1] if len(action.split()) > 1 else "")
            if clean_item(me.get("held_item")) == wanted or age > 5.0:
                self.one = None
            return
        if verb in {"craft", "place", "furnace_start", "furnace_take", "dig", "gather", "eat", "sleep"}:
            changed = inv_sig != before
            if changed or age >= (9.0 if verb in {"dig", "place"} else 6.0):
                self.one = None
            return
        if age > 5.0:
            self.one = None

    def stale_or_dead(self, obs: dict, now: float) -> str | None:
        me = self_row(obs)
        if me.get("dead"):
            self.job = None
            self.one = None
            self.manual = None
            if not self.dead_seen:
                self.dead_seen = True
                return "respawn"
            return "noop"
        self.dead_seen = False
        if number(obs.get("observation_age_s"), 0.0) > 3.0:
            return "cancel" if has_task(obs) and self.allowed("cancel") else "noop"
        return None

    def emergency_food(self, obs: dict, now: float) -> str | None:
        me = self_row(obs)
        inv = inventory(obs)
        if number(me.get("food"), 20) > 6 and number(me.get("health"), 20) > 8:
            return None
        item = best_food(inv)
        if not item:
            return None
        if has_task(obs):
            return "cancel" if self.allowed("cancel") else None
        if self.job and self.job.get("kind") != "eat":
            self.job = {"kind": "eat", "stage": "equip", "item": item, "started": now}
        elif not self.job:
            self.job = {"kind": "eat", "stage": "equip", "item": item, "started": now}
        return self.eat_action(obs, now)

    def mode_for(self, obs: dict, inv: dict[str, int]) -> str:
        tables = self.table_rows(obs)
        if not wood_total(inv) and not plank_total(inv) and not inv.get("crafting_table") and not tables and self.table_position is None:
            return "wood"
        if not tables and not inv.get("crafting_table") and self.table_position is None:
            return "wood" if not wood_total(inv) and not plank_total(inv) else "wooden_tools"
        if not any(inv.get(x) for x in PICKAXES):
            return "wooden_tools"
        if not any(inv.get(x) for x in STONE_OR_BETTER):
            return "stone"
        if food_total(inv) < 3:
            return "food"
        if not inv.get("iron_ingot") and not inv.get("raw_iron"):
            return "iron"
        if inv.get("raw_iron") and not inv.get("iron_ingot"):
            return "smelt"
        if not inv.get("shield"):
            return "shield"
        if not any(k.endswith("_bed") and v for k, v in inv.items()):
            return "bed"
        return "frontier"

    def next_milestone_action(self, obs: dict, now: float) -> str | None:
        inv = inventory(obs)
        self.mode = self.mode_for(obs, inv)
        table = self.near_table(obs)
        tables = self.table_rows(obs)
        if self.mode == "wood":
            target = self.pick_target(obs, suffixes=WOOD_SUFFIXES, allow_hidden_close=False)
            if target:
                self.begin_mine(obs, target, None, "direct log acquisition")
                return None
            # A nearby hidden tree can be approached once; do not keep it as a
            # permanent target if the interaction cell remains unavailable.
            hidden = self.pick_target(obs, suffixes=WOOD_SUFFIXES, allow_hidden_close=True)
            if hidden and hidden["_id"] not in self.failed:
                self.begin_mine(obs, hidden, None, "approach one nearby hidden log")
                return None
            return self.explore_start(obs, "wood frontier")

        # Craft planks/table before attempting any tool or mining operation.
        logs = next((k for k, v in inv.items() if v and k.endswith(WOOD_SUFFIXES)), "")
        if self.mode == "wooden_tools" and not inv.get("crafting_table") and not tables:
            if plank_total(inv) < 4 and logs:
                return "craft %s 4" % logs.replace("_log", "_planks").replace("_stem", "_planks")
            if plank_total(inv) >= 4 and not inv.get("crafting_table"):
                return "craft crafting_table 1"
        if inv.get("crafting_table") and not tables:
            if self.table_expected_until > now:
                return "wait 1" if self.allowed("wait") else None
            support = self.support(obs)
            if support:
                held = clean_item(self_row(obs).get("held_item"))
                if held != "crafting_table":
                    return "equip crafting_table hand"
                self.table_expected_until = now + 10.0
                q = support.get("_pos") or id_point(support.get("_id"))
                if q:
                    self.table_position = (q[0], q[1] + 1.0, q[2])
                return "place crafting_table %s up" % support["_id"]
            return self.explore_start(obs, "safe table support")

        if not table and tables:
            row = tables[0]
            if row["_distance"] > 4.6:
                return "navigate " + row["_id"]
            table = row
        if not table and self.table_position and self.allowed("goto"):
            q = self.table_position
            return "goto %.2f %.2f %.2f" % q
        if not table:
            return self.explore_start(obs, "return to workstation")

        table_id = table["_id"]
        planks = plank_total(inv)
        sticks = inv.get("stick", 0)
        if self.mode == "wooden_tools":
            if planks < 3:
                if logs:
                    return "craft %s 4" % logs.replace("_log", "_planks").replace("_stem", "_planks")
                target = self.pick_target(obs, suffixes=WOOD_SUFFIXES)
                if target:
                    self.begin_mine(obs, target, None, "direct logs for wooden pick")
                    return None
                return self.explore_start(obs, "wooden pick frontier")
            if sticks < 2:
                return "craft stick 4"
            if not inv.get("wooden_pickaxe"):
                return "craft wooden_pickaxe 1 %s" % table_id

        if self.mode == "stone":
            if inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0) < 3:
                target = self.pick_target(obs, names=STONE_NAMES, allow_hidden_close=False)
                if target:
                    self.begin_mine(obs, target, "wooden_pickaxe", "direct stone acquisition", 3)
                    return None
                return self.explore_start(obs, "stone frontier")
            if sticks < 2:
                return "craft stick 4"
            if not any(inv.get(x) for x in STONE_OR_BETTER):
                return "craft stone_pickaxe 1 %s" % table_id

        if self.mode == "food":
            target = self.pick_target(obs, names=FOOD_BLOCK_NAMES)
            if target:
                self.begin_mine(obs, target, None, "direct food block acquisition")
                return None
            weapon = next((x for x in WEAPONS if inv.get(x)), "")
            if not weapon and inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0) >= 2 and sticks:
                return "craft stone_sword 1 %s" % table_id
            animal = self.closest_animal(obs, sheep_first=False)
            if animal and weapon:
                self.start_animal(obs, animal)
                return None
            return self.explore_start(obs, "food frontier")

        if self.mode in {"iron", "smelt", "shield"}:
            if self.mode == "iron":
                target = self.pick_target(obs, names=IRON_NAMES)
                if target:
                    self.begin_mine(obs, target, "stone_pickaxe", "direct exposed iron acquisition")
                    return None
                if not inv.get("coal") and not inv.get("charcoal"):
                    coal = self.pick_target(obs, names=COAL_NAMES)
                    if coal:
                        self.begin_mine(obs, coal, "stone_pickaxe", "direct exposed furnace fuel")
                        return None
                # If no ore is exposed, keep looking, but use real blocks only.
                return self.explore_start(obs, "iron frontier")
            if self.mode == "smelt":
                furnaces = self.furnace_rows(obs)
                if not furnaces and not inv.get("furnace"):
                    if inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0) >= 8:
                        return "craft furnace 1 %s" % table_id
                if not furnaces and inv.get("furnace"):
                    if self.furnace_expected_until > now:
                        return "wait 1" if self.allowed("wait") else None
                    support = self.support(obs)
                    if support:
                        if clean_item(self_row(obs).get("held_item")) != "furnace":
                            return "equip furnace hand"
                        self.furnace_expected_until = now + 10.0
                        q = support.get("_pos") or id_point(support.get("_id"))
                        if q:
                            self.furnace_position = (q[0], q[1] + 1.0, q[2])
                        return "place furnace %s up" % support["_id"]
                    return self.explore_start(obs, "safe furnace support")
                if furnaces:
                    furnace = furnaces[0]
                    if furnace["_distance"] > 4.6:
                        return "navigate " + furnace["_id"]
                    fid = furnace["_id"]
                    raw = inv.get("raw_iron", 0)
                    fuel = next((x for x in ("coal", "charcoal") if inv.get(x)), "")
                    fuel = fuel or next((x for x in inv if x.endswith("_planks") and inv.get(x)), "")
                    if raw and fuel and fid not in self.smelting:
                        self.smelting[fid] = now
                        return "furnace_start %s raw_iron %d %s %d" % (fid, min(3, raw), fuel, min(3, inv.get(fuel, 1)))
                    if fid in self.smelting and now - self.smelting[fid] >= 32.0:
                        return "furnace_take " + fid
                    return "wait 1" if self.allowed("wait") else None
            if inv.get("iron_ingot") and not inv.get("shield"):
                if planks < 6 and logs:
                    return "craft %s 8" % logs.replace("_log", "_planks").replace("_stem", "_planks")
                return "craft shield 1 %s" % table_id

        if self.mode == "bed":
            bed = next((x for x in rows_with_meta(obs, "b") if x["_name"].endswith("_bed") and (x.get("visible") or x.get("in_reach"))), None)
            if bed:
                self.begin_mine(obs, bed, None, "direct bed acquisition")
                return None
            wool = next((k for k, v in inv.items() if k.endswith("_wool") and v >= 3), "")
            if wool and planks >= 3:
                return "craft %s_bed 1 %s" % (wool.removesuffix("_wool"), table_id)
            sheep = self.closest_animal(obs, sheep_first=True)
            if sheep and any(inv.get(x) for x in WEAPONS):
                self.start_animal(obs, sheep)
                return None
            return self.explore_start(obs, "bed frontier")

        # The direct transaction has completed its current milestone.  Continue
        # looking for a useful exposed block instead of waiting in a basin.
        return self.explore_start(obs, "next exposed milestone")

    def closest_animal(self, obs: dict, sheep_first: bool) -> dict | None:
        choices = [x for x in entities_with_meta(obs) if x["_name"] in ANIMALS and x.get("visible") and not x.get("dead") and x["_distance"] <= 40]
        choices.sort(key=lambda x: (0 if sheep_first and x["_name"] == "sheep" else 1, x["_distance"]))
        return choices[0] if choices else None

    def explore_start(self, obs: dict, purpose: str) -> str | None:
        if self.manual is not None:
            return None
        # A short manual segment is intentionally different from navigate:
        # it crosses the safe navigator's local basin and causes new chunks to
        # enter the adaptor's bounded candidate list.
        p = self_position(obs)
        if p is None or not self.allowed("controls"):
            return self.choose_stand(obs, purpose)
        directions = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
        dx, dz = directions[self.manual_index % len(directions)]
        self.manual_index += 1
        self.manual = {"phase": "turn", "dx": dx, "dz": dz, "until": self.now() + 9.0, "purpose": purpose}
        self.manual_started = self.now()
        q = (p[0] + dx * 10.0, p[1] + 1.0, p[2] + dz * 10.0)
        self.log("manual_frontier_begin", purpose=purpose, dx=dx, dz=dz)
        if self.allowed("look_at"):
            return "look_at %.2f %.2f %.2f" % q
        self.manual["phase"] = "move"
        return "controls forward,jump"

    def manual_action(self, obs: dict, now: float) -> str | None:
        if self.manual is None:
            return None
        # Stop immediately when the excursion exposed a useful target.
        if self.mode == "wood" and self.pick_target(obs, suffixes=WOOD_SUFFIXES):
            self.manual = None
            return "controls none" if self.allowed("controls") else None
        if self.mode == "stone" and self.pick_target(obs, names=STONE_NAMES):
            self.manual = None
            return "controls none" if self.allowed("controls") else None
        if self.mode == "iron" and self.pick_target(obs, names=IRON_NAMES):
            self.manual = None
            return "controls none" if self.allowed("controls") else None
        if self.mode in {"food", "bed"} and (self.pick_target(obs, names=FOOD_BLOCK_NAMES) or self.closest_animal(obs, self.mode == "bed")):
            self.manual = None
            return "controls none" if self.allowed("controls") else None
        if self.manual.get("phase") == "turn":
            # look_at is one-shot; update_one clears it after a short lease.
            self.manual["phase"] = "move"
            return "noop"
        if now < number(self.manual.get("until"), now):
            return "controls forward,jump"
        self.manual = None
        return "controls none" if self.allowed("controls") else None

    def choose_stand(self, obs: dict, purpose: str) -> str | None:
        now = self.now()
        stands = []
        for row in rows_with_meta(obs, "s"):
            if row["_distance"] < 3.0 or row["_distance"] > 42.0:
                continue
            if self.blocked("navigate " + row["_id"], now):
                continue
            stands.append(row)
        stands.sort(key=lambda x: (-x["_distance"], x["_id"]))
        if stands:
            return "navigate " + stands[0]["_id"]
        return "wait 1" if self.allowed("wait") else "noop"

    def action_for_idle(self, obs: dict, now: float) -> str:
        # Manual movement owns controls until it ends; it is never overwritten
        # by a fresh craft/mine decision in the same tick.
        if self.manual is not None:
            action = self.manual_action(obs, now)
            return action or "noop"
        if self.job:
            action = self.job_action(obs, now)
            return action or "noop"
        action = self.next_milestone_action(obs, now)
        return action or "noop"

    def run_episode(self, game: Any) -> None:
        self.names = set(getattr(getattr(game, "action_spec", None), "names", ()) or ())
        self.threats = ThreatPolicy(self.ctx, self.names, self.log)
        while not self.ctx.done:
            now = self.now()
            obs = parse_observation(self.ctx.observe())
            if not obs:
                self.post("noop", {}, "missing_observation")
                self.wait_tick()
                continue
            self.record_progress(obs, now)
            self.process_feedback(obs, now)
            terminal = self.stale_or_dead(obs, now)
            if terminal:
                self.post(terminal, obs, "dead_or_stale")
                self.wait_tick()
                continue
            inv = inventory(obs)
            # Emergency food precedes the tactical classifier and all optional
            # work.  It is staged through equip because eat uses the held item.
            food_action = self.emergency_food(obs, now)
            if food_action:
                if has_task(obs) and food_action == "cancel":
                    self.cancel_pending = True
                self.post(food_action, obs, "emergency_food")
                self.wait_tick()
                continue
            threat_action = self.threats.decide(obs, inv, now) if self.threats else None
            if threat_action:
                if threat_action not in {"noop", "shield on"}:
                    # Do not resume a manual frontier segment after a tactical
                    # escape; re-evaluate terrain from the new safe position.
                    self.manual = None
                if has_task(obs) and threat_action not in {"shield on", "noop"}:
                    self.post("cancel" if self.allowed("cancel") else "noop", obs, "threat_interrupt")
                else:
                    self.post(threat_action, obs, "threat_policy")
                self.wait_tick()
                continue
            if self.cancel_pending:
                if has_task(obs):
                    self.post("cancel" if self.allowed("cancel") else "noop", obs, "clear_failed_job")
                    self.wait_tick()
                    continue
                self.cancel_pending = False
            # A bounded task owns the hands.  Only noop keeps it alive.
            if has_task(obs):
                self.update_one(obs, now)
                self.post("noop", obs, "continue_task")
                self.wait_tick()
                continue
            self.update_one(obs, now)
            # If a one-shot just ended, let its postcondition/state machine be
            # evaluated once before issuing another command.
            if self.one is not None:
                self.post("noop", obs, "await_postcondition")
                self.wait_tick()
                continue
            action = self.action_for_idle(obs, now)
            self.post(action, obs, "direct_controller")
            self.wait_tick()


def build(ctx: Any) -> DirectController:
    return DirectController(ctx)
