"""Candidate 0019: verified opening plus a moving staircase mine.

The previous two useful controllers solved different local problems.  The
recipe compiler in 0015 reliably made the opening, while 0016 made real block
transactions and verified their postconditions.  This controller keeps both,
but changes the resource-search geometry: a mined floor/wall block is followed
by a verified move into the new space.  Surface excavation which never changes
the player's elevation is not treated as a mine.

All commands are compiled from the current observation.  Jev is only an
interrupt-level safety judge and cannot manufacture an action argument.
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
PICKS = ("netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
STONE_PICKS = {"netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe"}
WEAPONS = (
    "netherite_sword", "diamond_sword", "iron_sword", "stone_sword", "wooden_sword",
    "netherite_axe", "diamond_axe", "iron_axe", "stone_axe", "wooden_axe",
)
FOODS = {
    "apple", "bread", "baked_potato", "carrot", "golden_carrot", "sweet_berries",
    "melon_slice", "mushroom_stew", "beetroot", "potato", "cooked_beef",
    "cooked_porkchop", "cooked_mutton", "cooked_chicken", "cooked_rabbit",
    "cooked_cod", "cooked_salmon", "beef", "porkchop", "mutton", "chicken",
    "rabbit", "cod", "salmon", "potato", "wheat",
}
FOOD_BLOCKS = {"hay_block", "melon", "sweet_berry_bush", "wheat", "carrots", "potatoes", "beetroots"}
STONE_BLOCKS = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON_BLOCKS = {"iron_ore", "deepslate_iron_ore", "raw_iron", "iron_block"}
COAL_BLOCKS = {"coal_ore", "deepslate_coal_ore", "coal_block"}
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
FAIL_WORDS = (
    "failed", "failure", "error", "unreachable", "timeout", "cannot", "rejected",
    "illegal", "not enough", "not in reach", "no path", "busy", "invalid",
    "unknown", "out of range", "denied",
)


def number(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def item_name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("id", ""))))
    if not isinstance(value, str):
        return ""
    return value.removeprefix("minecraft:")


def point(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        if value.get("position") is not None:
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
            result = json.loads(value)
            return result if isinstance(result, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def self_row(obs: dict) -> dict:
    return mapping(obs.get("self"))


def self_position(obs: dict) -> tuple[float, float, float] | None:
    return point(self_row(obs).get("position"))


def inventory(obs: dict) -> dict[str, int]:
    """Return only personal inventory, never a cursor or open container."""
    source = obs.get("inventory")
    if source is None or source == {} or source == []:
        source = obs.get("inventory_authoritative")
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source.get("items")
    result: dict[str, int] = {}
    if isinstance(source, dict):
        for key, raw in source.items():
            name = item_name(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(number(raw_count)))
            if name and count:
                result[name] = result.get(name, 0) + count
    elif isinstance(source, list):
        for raw in source:
            if not isinstance(raw, dict):
                continue
            name = item_name(raw)
            count = max(0, int(number(raw.get("count", 0))))
            if name and count:
                result[name] = result.get(name, 0) + count
    # This is explicitly labelled personal inventory and is useful while the
    # periodic authoritative audit is catching up after an action.
    if not result and isinstance(obs.get("inventory_client"), dict):
        for key, raw in obs["inventory_client"].items():
            name = item_name(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(number(raw_count)))
            if name and count:
                result[name] = result.get(name, 0) + count
    return result


def inv_signature(inv: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(k), int(v)) for k, v in inv.items() if int(v) > 0))


def rows(obs: dict, prefix: str | None = None) -> list[dict]:
    here = self_position(obs)
    result = []
    for raw in obs.get("candidates", []) if isinstance(obs.get("candidates"), list) else []:
        if not isinstance(raw, dict):
            continue
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


def nearby_entities(obs: dict) -> list[dict]:
    here = self_position(obs)
    source = obs.get("entities", obs.get("nearby_entities", []))
    result = []
    if not isinstance(source, list):
        return result
    for raw in source:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row["_name"] = item_name(raw.get("name", raw.get("type", "")))
        row["_pos"] = point(raw.get("position"))
        row["_distance"] = number(raw.get("distance"), distance(here, row["_pos"]))
        result.append(row)
    return result


def action_verb(action: Any) -> str:
    return str(action or "").split(" ", 1)[0]


def has_task(obs: dict) -> bool:
    value = obs.get("task")
    return value is not None and value != "" and value != {}


def food_total(inv: dict[str, int]) -> int:
    return sum(int(v) for k, v in inv.items() if k in FOODS)


def plank_total(inv: dict[str, int]) -> int:
    return sum(int(v) for k, v in inv.items() if k.endswith("_planks"))


def wood_total(inv: dict[str, int]) -> int:
    return sum(int(v) for k, v in inv.items() if k.endswith(WOOD_SUFFIXES))


def best_food(inv: dict[str, int]) -> str:
    order = (
        "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
        "bread", "baked_potato", "carrot", "apple", "sweet_berries",
        "melon_slice", "beef", "porkchop", "mutton", "chicken", "rabbit",
        "cod", "salmon", "potato", "wheat",
    )
    return next((name for name in order if inv.get(name)), "")


class SafetyJev:
    """A small asynchronous hazard classifier with a hard-coded veto layer."""

    def __init__(self, ctx: Any, allowed_names: set[str], log) -> None:
        self.ctx = ctx
        self.allowed_names = allowed_names
        self.log = log
        self.future = None
        self.signature: tuple = ()
        self.started = -100.0
        self.cached: tuple[str, tuple, float] | None = None

    def allowed(self, action: str) -> bool:
        return action_verb(action) in self.allowed_names

    def hazard_list(self, obs: dict) -> list[dict]:
        me = self_row(obs)
        hazards: list[dict] = []
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            hazards.append({"kind": "lava_or_fire", "id": "self", "range": "touching", "distance": 0.0})
        if me.get("is_in_water") and number(me.get("oxygen"), 20) < 6:
            hazards.append({"kind": "drowning", "id": "self", "range": "touching", "distance": 0.0})
        if number(me.get("fall_distance_estimate"), 0) > 3:
            hazards.append({"kind": "fall", "id": "self", "range": "touching", "distance": 0.0})
        for row in rows(obs, "b"):
            if row["_name"] in HAZARD_BLOCKS and row["_distance"] <= 4.5:
                kind = "lava_or_fire" if row["_name"] in {"lava", "fire", "soul_fire"} else row["_name"]
                hazards.append({"kind": kind, "id": row["_id"], "range": "near", "distance": row["_distance"]})
        for entity in nearby_entities(obs):
            hostile = bool(entity.get("hostile") or entity.get("conditional_threat") or entity["_name"] in HOSTILES)
            if hostile and entity.get("visible") and entity["_distance"] <= 8.0:
                hazards.append({
                    "kind": entity["_name"], "id": str(entity.get("id", "")),
                    "range": "touching" if entity["_distance"] <= 3.2 else "near",
                    "distance": entity["_distance"],
                })
        return hazards

    def fallback(self, obs: dict, hazards: list[dict], inv: dict[str, int]) -> str:
        me = self_row(obs)
        if any(h["kind"] == "lava_or_fire" for h in hazards):
            if any(h.get("id") == "self" for h in hazards):
                return "controls forward,jump" if self.allowed("controls") else "cancel"
            return "controls back,jump" if self.allowed("controls") else "cancel"
        if any(h["kind"] == "drowning" for h in hazards):
            return "controls jump" if self.allowed("controls") else "cancel"
        if number(me.get("health"), 20) <= 6 or number(me.get("food"), 20) <= 5:
            food = best_food(inv)
            if food:
                if item_name(me.get("held_item")) != food and self.allowed("equip"):
                    return "equip %s hand" % food
                if self.allowed("eat"):
                    return "eat " + food
        weapon = next((name for name in WEAPONS if inv.get(name)), "")
        close = [h for h in hazards if h.get("id") != "self" and number(h.get("distance"), 99) <= 3.2]
        if close and weapon:
            if item_name(me.get("held_item")) != weapon and self.allowed("equip"):
                return "equip %s hand" % weapon
            if self.allowed("attack") and close[0].get("id"):
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
        weapon = next((name for name in WEAPONS if inv.get(name)), "")
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
            if action not in answer and (self.allowed(action) or action == "noop"):
                answer.append(action)
        return answer[:8]

    def decide(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        hazards = self.hazard_list(obs)
        if not hazards:
            self.signature = ()
            self.cached = None
            return None
        options = self.menu(obs, hazards, inv)
        if not options:
            return None
        signature = tuple(sorted((str(h["kind"]), str(h["id"]), str(h["range"])) for h in hazards))
        if self.cached and self.cached[1] == signature and now - self.started < 2.5:
            return self.cached[0] if self.cached[0] in options else options[0]
        if self.future is None and self.ctx.budget.remaining().get("system1_calls", 0) > 2:
            state = {
                "purpose": "Immediate survival only; choose one exact supplied action.",
                "self": {
                    "health_band": "critical" if number(self_row(obs).get("health"), 20) < 7 else "low" if number(self_row(obs).get("health"), 20) < 12 else "stable",
                    "food_band": "low" if number(self_row(obs).get("food"), 20) < 8 else "fed",
                    "dimension": str(self_row(obs).get("dimension", "")),
                    "lava_fire": bool(self_row(obs).get("is_in_lava") or self_row(obs).get("is_in_fire")),
                },
                "hazards": [{"kind": h["kind"], "range": h["range"]} for h in hazards[:6]],
                "legal_actions": options,
                "rule": "Never invent an ID. Leave lava/fire/fall/water before mining. Prefer defense over work.",
            }
            questions = {
                "response": Choice(
                    "Select the safest supplied immediate response. Do not invent a command or argument.",
                    {a: "Exact executable response: " + a for a in options},
                ),
                "safe_to_continue": Noul(
                    "Is it safe to continue the current work without first using a defensive action?",
                    {"true": "No immediate hazard requires interruption.", "false": "A hazard or close hostile requires defense or interruption."},
                ),
            }
            try:
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
                probabilities = getattr(answer, "probabilities", {}) or {}
                probability = number(probabilities.get(value), 0.0) if isinstance(probabilities, dict) else 0.0
                if value in options and self.signature == signature and probability >= 0.52:
                    self.cached = (value, signature, probability)
                    self.log("jev_safety", action=value, probability=round(probability, 3), hazards=len(hazards))
                    return value
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:
                self.log("jev_invalid", error=type(exc).__name__)
        # The fallback is deliberately executable while Jev is in flight.
        return self.fallback(obs, hazards, inv)


class StaircaseController:
    """Milestone compiler and moving, postcondition-checked mine."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.names: set[str] = set()
        self.safety: SafetyJev | None = None
        self.active: dict | None = None
        self.job: dict | None = None
        self.last_posted = "noop"
        self.feedback_marker = ""
        self.failed_until: dict[str, float] = {}
        self.completed_targets: set[str] = set()
        self.visited_stands: dict[str, float] = {}
        self.visited_cells: set[tuple[int, int]] = set()
        self.explore_round = 0
        self.manual_until = -1.0
        self.manual_target: tuple[float, float, float] | None = None
        self.manual_stage = "none"
        self.table_hint: tuple[float, float, float] | None = None
        self.table_wait_until = -1.0
        self.furnace_hint: tuple[float, float, float] | None = None
        self.furnace_wait_until = -1.0
        self.smelt_started: dict[str, float] = {}
        self.mine_direction: tuple[int, int] | None = None
        self.step_path: list[tuple[float, float, float]] = []
        self.steps_since_refresh = 0
        self.refresh_pending = False
        self.return_index: int | None = None
        self.last_mode = "wood"
        self.last_inv = ()
        self.last_adv = ""
        self.last_progress_at = 0.0
        self.dead_seen = False
        self.last_attack = -100.0
        self.last_obs: dict = {}

    def log(self, event: str, **fields: Any) -> None:
        self.ctx.log(event, **fields)

    def now(self) -> float:
        return number(self.ctx.clock.now())

    def allowed(self, action: str) -> bool:
        return action_verb(action) in self.names or action == "noop"

    def blocked(self, key: str, now: float) -> bool:
        return bool(key) and now < self.failed_until.get(key, -1.0)

    def suppress(self, key: str, now: float, seconds: float = 100.0, reason: str = "") -> None:
        if not key or key in {"noop", "cancel"}:
            return
        self.failed_until[key] = max(self.failed_until.get(key, -1.0), now + seconds)
        self.log("target_suppressed", target=key, blocked_s=seconds, reason=reason)

    def fail_active(self, now: float, reason: str) -> None:
        if not self.active:
            return
        action = str(self.active.get("action", ""))
        verb = action_verb(action)
        self.suppress(action, now, 150.0 if verb in {"gather", "dig"} else 70.0, reason)
        if self.job and self.job.get("target") == self.active.get("target"):
            self.suppress(str(self.job.get("target")), now, 150.0, "job_" + reason)
            self.job = None
        self.active = None

    def action_failed(self, obs: dict) -> bool:
        values = [obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")]

        def one(value: Any) -> bool:
            if isinstance(value, dict):
                if value.get("ok") is False or value.get("success") is False:
                    return True
                status = str(value.get("status", value.get("result", ""))).lower()
                if any(word in status for word in FAIL_WORDS):
                    return True
                for key in ("message", "feedback", "reason", "error"):
                    nested = value.get(key)
                    if isinstance(nested, str) and any(word in nested.lower() for word in FAIL_WORDS):
                        return True
            elif isinstance(value, str):
                low = value.lower()
                return any(word in low for word in FAIL_WORDS) and not any(x in low for x in ("success", "succeeded", "ok"))
            return False

        return any(one(value) for value in values)

    def process_feedback(self, obs: dict, now: float) -> None:
        try:
            marker = json.dumps([obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")], sort_keys=True, default=str)
        except Exception:
            marker = repr(obs.get("feedback"))
        if marker == self.feedback_marker:
            return
        self.feedback_marker = marker
        if self.active and self.action_failed(obs):
            rejected = str(self.active.get("action", ""))
            self.fail_active(now, "adapter_feedback")
            self.log("operation_rejected", action=rejected)

    def observe_progress(self, obs: dict, now: float) -> None:
        inv_sig = inv_signature(inventory(obs))
        advancements = obs.get("advancements", [])
        try:
            adv_sig = json.dumps(advancements, sort_keys=True, default=str)
        except Exception:
            adv_sig = repr(advancements)
        if inv_sig != self.last_inv or adv_sig != self.last_adv:
            inv_changed = inv_sig != self.last_inv
            self.last_inv, self.last_adv = inv_sig, adv_sig
            self.last_progress_at = now
            self.log("verified_progress", inventory_delta=inv_changed, advancement_delta=True)
        # A placed workstation is authoritative when it enters the exposed
        # candidate stream.  It is also recorded from the consumed inventory
        # latch below so audit lag cannot cause duplicate placement.
        for row in rows(obs, "b"):
            if row["_name"] == "crafting_table" and row["_pos"]:
                self.table_hint = row["_pos"]
            if row["_name"] == "furnace" and row["_pos"]:
                self.furnace_hint = row["_pos"]
        if self.active:
            action = str(self.active.get("action", ""))
            if action.startswith("place crafting_table ") and not inventory(obs).get("crafting_table"):
                parts = action.split()
                support = id_point(parts[2]) if len(parts) >= 3 else None
                if support:
                    self.table_hint = (support[0], support[1] + 1.0, support[2])
                    self.table_wait_until = now + 7.0
                    self.log("table_consumed_latch", position=self.table_hint)
            if action.startswith("place furnace ") and not inventory(obs).get("furnace"):
                parts = action.split()
                support = id_point(parts[2]) if len(parts) >= 3 else None
                if support:
                    self.furnace_hint = (support[0], support[1] + 1.0, support[2])
                    self.furnace_wait_until = now + 7.0

    def target_row(self, obs: dict, ident: str) -> dict | None:
        return next((row for row in rows(obs, "b") if row["_id"] == ident), None)

    def action_timeout(self, action: str) -> float:
        verb = action_verb(action)
        if verb == "gather":
            return 29.0
        if verb in {"navigate", "goto"}:
            return 17.0
        if verb == "dig":
            return 11.0
        if verb in {"craft", "equip", "place", "furnace_start", "furnace_take", "activate_block", "eat"}:
            return 8.0
        if verb == "look_at":
            return 1.2
        return 6.0

    def active_complete(self, obs: dict, now: float) -> bool:
        """Advance one-shot latches only on an observable postcondition."""
        if not self.active:
            return False
        a = self.active
        action = str(a.get("action", ""))
        verb = action_verb(action)
        inv = inventory(obs)
        before = a.get("before_inv", ())
        pos = self_position(obs)
        done = False
        if verb == "equip":
            wanted = item_name(action.split()[1] if len(action.split()) > 1 else "")
            done = item_name(self_row(obs).get("held_item")) == wanted
        elif verb in {"craft", "place", "furnace_start", "furnace_take", "eat", "sleep"}:
            done = inv_signature(inv) != before
            if verb == "place" and len(action.split()) >= 3:
                item = item_name(action.split()[1])
                done = done or any(row["_name"] == item for row in rows(obs, "b"))
        elif verb == "look_at":
            done = now - number(a.get("started"), now) >= 0.65
        elif verb in {"navigate", "goto"}:
            done = distance(a.get("before_pos"), pos) >= 0.65
            a["moved"] = done
        elif verb == "gather":
            done = inv_signature(inv) != before
        elif verb == "dig":
            target = str(a.get("target", ""))
            row = self.target_row(obs, target)
            changes = obs.get("block_changes", [])
            changed = False
            if isinstance(changes, list):
                for change in changes:
                    if isinstance(change, dict) and distance(change.get("position"), a.get("target_pos")) < 0.2:
                        changed = item_name(change.get("name")) in {"air", "water", "lava"} or bool(change.get("name"))
            done = row is None or changed or inv_signature(inv) != before
        elif verb in {"hotbar", "close_window", "wake"}:
            done = now - number(a.get("started"), now) >= 0.4
        if done:
            a["completed"] = True
            self.active = None
            return True
        if now - number(a.get("started"), now) >= self.action_timeout(action):
            self.fail_active(now, "no_postcondition")
            return True
        return False

    def post(self, action: str | None, obs: dict, now: float, reason: str = "") -> str:
        action = str(action or "noop").strip() or "noop"
        if not self.allowed(action):
            action = "noop"
        verb = action_verb(action)
        # Deliberate identical one-shots need an explicit noop re-arm.
        if action == self.last_posted and verb not in {"noop", "controls", "attack", "shield"}:
            action = "noop"
            verb = "noop"
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            # A malformed stand-in or a rejected nonfatal semantic action must
            # not escape the controller.  The normal game adaptor reports
            # rejection as feedback, but this is useful in pre-flight too.
            self.log("post_exception", action=action, error=type(exc).__name__)
            action = "noop"
            verb = "noop"
        self.last_posted = action
        if action != "noop":
            self.log("posted", action=action, reason=reason, mode=self.last_mode)
        if action == "cancel":
            self.active = None
            self.job = None
            self.manual_until = -1.0
            return action
        if verb not in {"noop", "controls", "attack", "shield", "wait"}:
            parts = action.split()
            self.active = {
                "action": action,
                "started": now,
                "before_inv": inv_signature(inventory(obs)),
                "before_pos": self_position(obs),
            }
            if verb in {"dig", "gather"} and len(parts) >= 2:
                self.active["target"] = self.job.get("target", "") if self.job else parts[1]
                self.active["target_pos"] = self.job.get("position") if self.job else id_point(parts[1])
            if verb in {"navigate", "goto"} and len(parts) >= 2:
                self.visited_stands[parts[1]] = now
                if parts[1].startswith("s:"):
                    q = id_point(parts[1])
                    if q:
                        self.visited_cells.add((int(q[0]) // 5, int(q[2]) // 5))
        return action

    def process_active(self, obs: dict, now: float) -> None:
        if not self.active:
            return
        self.active_complete(obs, now)

    def safe_support(self, obs: dict) -> dict | None:
        here = self_position(obs)
        if here is None:
            return None
        floor_y = math.floor(here[1]) - 1
        fx, fz = math.floor(here[0]), math.floor(here[2])
        possible = []
        for row in rows(obs, "b"):
            if row["_name"] not in SUPPORT_BLOCKS or row["_name"] in HAZARD_BLOCKS:
                continue
            if not row.get("in_reach") or row.get("visible") is False:
                continue
            q = row["_pos"]
            if q is None or math.floor(q[1]) != floor_y:
                continue
            if math.floor(q[0]) == fx and math.floor(q[2]) == fz:
                continue
            if self.blocked("place crafting_table %s up" % row["_id"], self.now()):
                continue
            possible.append(row)
        possible.sort(key=lambda row: row["_distance"])
        return possible[0] if possible else None

    def table_rows(self, obs: dict) -> list[dict]:
        return [row for row in rows(obs, "b") if row["_name"] == "crafting_table"]

    def near_table(self, obs: dict) -> dict | None:
        choices = [row for row in self.table_rows(obs) if row["_distance"] <= 4.8 and row.get("in_reach") is not False]
        choices.sort(key=lambda row: row["_distance"])
        return choices[0] if choices else None

    def furnace_rows(self, obs: dict) -> list[dict]:
        return [row for row in rows(obs, "b") if row["_name"] == "furnace"]

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
        return 0

    def plank_source(self, inv: dict[str, int]) -> str:
        return next((name for name, count in inv.items() if count and name.endswith(WOOD_SUFFIXES)), "")

    def craft_action(self, item: str, desired: int, inv: dict[str, int], table: dict | None = None) -> str | None:
        available = self.feasible(item, inv)
        count = min(max(1, int(desired)), available)
        if count < 1:
            return None
        action = "craft %s %d" % (item, count)
        if table is not None:
            action += " " + str(table["_id"])
        return action

    def usable_target(
        self,
        row: dict,
        now: float,
        names: set[str] | None = None,
        suffixes: tuple[str, ...] = (),
        allow_hidden: bool = False,
    ) -> bool:
        name = row["_name"]
        if names is not None and name not in names:
            return False
        if suffixes and not any(name.endswith(s) for s in suffixes):
            return False
        # A hidden log is still a listed block-interaction candidate.  It is
        # useful for a tree whose trunk is occluded by leaves/water; the job
        # must navigate to it before trying to gather.  Hidden ore and food do
        # not receive this exception.
        hidden = row.get("visible") is False and not row.get("in_reach")
        if row.get("harvestable") is False or (hidden and not allow_hidden):
            return False
        if name in HAZARD_BLOCKS or name in {"air", "lava", "water", "bubble_column"}:
            return False
        return not self.blocked(row["_id"], now) and row["_id"] not in self.completed_targets

    def choose_resource(
        self,
        obs: dict,
        names: set[str] | None = None,
        suffixes: tuple[str, ...] = (),
        allow_hidden: bool = False,
    ) -> dict | None:
        here = self_position(obs)
        choices = []
        for row in rows(obs, "b"):
            if not self.usable_target(row, self.now(), names, suffixes, allow_hidden):
                continue
            q = row.get("_pos")
            if here is not None and q is not None:
                # gather/dig must not select the solid block supporting the
                # player.  This is the failure that rejected the first stone
                # gathers in the live trial after the player descended.
                if math.floor(q[0]) == math.floor(here[0]) and math.floor(q[2]) == math.floor(here[2]) and math.floor(q[1]) <= math.floor(here[1]) - 1:
                    continue
            choices.append(row)
        choices.sort(key=lambda row: (0 if row.get("in_reach") else 1, 0 if row.get("visible") else 1, row["_distance"]))
        return choices[0] if choices else None

    def safe_stands(self, obs: dict, target: tuple[float, float, float] | None = None) -> list[dict]:
        now = self.now()
        here = self_position(obs)
        choices = []
        for row in rows(obs, "s"):
            q = row["_pos"]
            if q is None or row["_distance"] < 1.2 or row["_distance"] > 46.0:
                continue
            ident = row["_id"]
            if self.blocked("navigate " + ident, now):
                continue
            if target is not None and distance(q, target) > 5.5:
                continue
            if target is not None:
                rank = (distance(q, target), self.visited_stands.get(ident, -10000.0), row["_distance"])
            else:
                cell = (int(q[0]) // 5, int(q[2]) // 5)
                fresh = 0 if cell not in self.visited_cells and ident not in self.visited_stands else 1
                # For a real frontier, farthest safe visible stand is better
                # than cycling the nearest four stands around the spawn tree.
                rank = (fresh, -row["_distance"], -self.visited_stands.get(ident, -10000.0))
            choices.append((rank, row))
        choices.sort(key=lambda pair: pair[0])
        return [pair[1] for pair in choices]

    def explore_action(self, obs: dict, now: float) -> str | None:
        stands = self.safe_stands(obs)
        if stands:
            row = stands[0]
            self.explore_round += 1
            return "navigate " + row["_id"]
        # If pathfinder has no safe stand in a swamp/tree basin, a very short
        # look-then-forward lease deliberately leaves the basin.  It is not a
        # general autopilot and is stopped after a few seconds.
        if self.manual_stage == "none" and self.explore_round >= 5:
            here = self_position(obs)
            if here is not None:
                target = None
                for row in rows(obs, "s"):
                    if row["_pos"] and (target is None or row["_distance"] > target[0]):
                        target = (row["_distance"], row["_pos"])
                if target:
                    self.manual_target = target[1]
                    self.manual_stage = "look"
                    return "look_at %.2f %.2f %.2f" % self.manual_target
        return "noop"

    def backtrack_action(self, obs: dict, now: float) -> str | None:
        """Return along the already verified staircase before a workstation trip."""
        if not self.step_path:
            return None
        p = self_position(obs)
        if self.return_index is None:
            self.return_index = len(self.step_path) - 1
        while self.return_index >= 0 and p is not None and distance(p, self.step_path[self.return_index]) <= 1.2:
            self.return_index -= 1
        if self.return_index < 0:
            return None
        q = self.step_path[self.return_index]
        action = "goto %.2f %.2f %.2f" % q
        if self.allowed("goto") and not self.blocked(action, now):
            return action
        # A listed safe stand near an old step is an adaptor-compatible
        # fallback when goto rejects a precise feet coordinate.
        stands = [row for row in self.safe_stands(obs, q) if distance(row.get("_pos"), q) <= 2.5]
        if stands:
            return "navigate " + stands[0]["_id"]
        return None

    def table_route(self, obs: dict, now: float) -> str | None:
        near = self.near_table(obs)
        if near:
            self.return_index = None
            return None
        tables = self.table_rows(obs)
        if tables:
            table = min(tables, key=lambda row: row["_distance"])
            # Go to a concrete feet cell beside the table.  A safe stand can
            # be several blocks away and, as seen in the live trial, may be a
            # valid navigation candidate that leaves us in a loop just short
            # of the workstation.
            q = table.get("_pos")
            if q is not None and self.allowed("goto"):
                goal = (q[0] + 1.0, q[1], q[2] + 1.0)
                action = "goto %.2f %.2f %.2f" % goal
                if distance(self_position(obs), goal) > 4.0 and not self.blocked(action, now):
                    return action
            stands = [row for row in self.safe_stands(obs, table["_pos"]) if distance(row.get("_pos"), table["_pos"]) <= 4.5]
            if stands:
                return "navigate " + stands[0]["_id"]
            if self.allowed("navigate") and not self.blocked("navigate " + table["_id"], now):
                return "navigate " + table["_id"]
            return "noop"
        inv = inventory(obs)
        if inv.get("crafting_table"):
            if now < self.table_wait_until:
                return "noop"
            support = self.safe_support(obs)
            if support:
                if item_name(self_row(obs).get("held_item")) != "crafting_table":
                    return "equip crafting_table hand"
                return "place crafting_table %s up" % support["_id"]
            return self.explore_action(obs, now)
        if self.table_hint:
            if now < self.table_wait_until:
                return "noop"
            target = (self.table_hint[0] + 1.0, self.table_hint[1], self.table_hint[2] + 1.0)
            action = "goto %.2f %.2f %.2f" % target
            if self.allowed("goto") and not self.blocked(action, now) and distance(self_position(obs), target) > 4.0:
                return action
        back = self.backtrack_action(obs, now)
        if back:
            return back
        if self.table_hint:
            if now < self.table_wait_until:
                return "noop"
            # The table may be placed successfully before its block candidate
            # reaches the snapshot.  Return to a concrete adjacent feet cell
            # from the consumed-support hint, rather than cycling a distant s:
            # stand.
            target = (self.table_hint[0] + 1.0, self.table_hint[1], self.table_hint[2] + 1.0)
            action = "goto %.2f %.2f %.2f" % target
            if self.allowed("goto") and not self.blocked(action, now) and distance(self_position(obs), target) > 4.0:
                return action
            action = self.move_to(obs, target, now)
            if action:
                return action

    def furnace_route(self, obs: dict, now: float) -> str | None:
        fs = self.furnace_rows(obs)
        if fs:
            furnace = min(fs, key=lambda row: row["_distance"])
            if furnace["_distance"] > 4.8:
                stands = self.safe_stands(obs, furnace["_pos"])
                return "navigate " + stands[0]["_id"] if stands else "goto %.2f %.2f %.2f" % furnace["_pos"]
            return None
        inv = inventory(obs)
        if inv.get("furnace"):
            if now < self.furnace_wait_until:
                return "noop"
            support = self.safe_support(obs)
            if support:
                if item_name(self_row(obs).get("held_item")) != "furnace":
                    return "equip furnace hand"
                return "place furnace %s up" % support["_id"]
            return self.explore_action(obs, now)
        return None

    def row_is_safe_dig(self, obs: dict, row: dict, py: int | None = None) -> bool:
        if row["_name"] not in GROUND_BLOCKS or row["_name"] in HAZARD_BLOCKS:
            return False
        if row.get("harvestable") is False or row.get("visible") is False and not row.get("in_reach"):
            return False
        q = row["_pos"]
        p = self_position(obs)
        if q is None or p is None:
            return False
        py = math.floor(p[1]) if py is None else py
        # Never remove the block supporting the current feet cell.
        if math.floor(q[0]) == math.floor(p[0]) and math.floor(q[2]) == math.floor(p[2]) and math.floor(q[1]) <= py - 1:
            return False
        return row["_id"] not in self.completed_targets and not self.blocked(row["_id"], self.now())

    def staircase_target(self, obs: dict, now: float) -> dict | None:
        """Choose an observed adjacent floor/wall cell, never an unknown block."""
        p = self_position(obs)
        if p is None:
            return None
        py = math.floor(p[1])
        candidates = []
        for row in rows(obs, "b"):
            if not self.row_is_safe_dig(obs, row, py):
                continue
            q = row["_pos"]
            if q is None:
                continue
            dx, dz = q[0] - p[0], q[2] - p[2]
            horizontal = abs(dx) + abs(dz)
            # A step must be a directly adjacent cell.  The earlier live
            # trial allowed diagonal/long jumps; it dug b:-22,66,31 from a
            # feet cell near b:-21,67,32, and pathfinding could not enter the
            # resulting corner.  A one-cell staircase keeps an actual solid
            # side wall and a walkable successor.
            if horizontal < 0.7 or horizontal > 1.6:
                continue
            # A one-block step is the primary target. A same-level wall is a
            # legal tunnel-opening fallback when the floor is not exposed.
            level = math.floor(q[1])
            if level not in {py - 1, py}:
                continue
            if level == py - 1:
                tier = 0
            else:
                tier = 1
            if self.mine_direction is not None:
                vx, vz = self.mine_direction
                forward = dx * vx + dz * vz
                side = abs(dx * vz - dz * vx)
                direction_penalty = 0 if forward > 0 and side <= 2.0 else 1
            else:
                direction_penalty = 0
            # Do not intentionally pick sand/gravel as a staircase roof or
            # any block adjacent to an explicitly observed fluid hazard.
            if row["_name"] in {"sand", "gravel"}:
                continue
            candidates.append((tier, direction_penalty, horizontal, -number(row.get("seen_at"), 0), row))
        candidates.sort(key=lambda x: x[:4])
        return candidates[0][4] if candidates else None

    def begin_direct_mine(self, obs: dict, target: dict, kind: str, tool: str, now: float) -> None:
        q = target["_pos"]
        p = self_position(obs)
        self.job = {
            "kind": kind,
            "stage": "approach" if not target.get("in_reach") else "equip",
            "target": target["_id"],
            "name": target["_name"],
            "position": q,
            "tool": tool,
            "started": now,
            "before_inv": inv_signature(inventory(obs)),
            "before_step_pos": p,
            "expected": None,
            "direction": self.mine_direction,
            "approaches": 0,
        }
        if p is not None and q is not None:
            dx, dz = q[0] - p[0], q[2] - p[2]
            if abs(dx) >= abs(dz) and abs(dx) > 0.4:
                self.mine_direction = (1 if dx > 0 else -1, 0)
            elif abs(dz) > 0.4:
                self.mine_direction = (0, 1 if dz > 0 else -1)
            if kind == "step" and (not self.step_path or distance(self.step_path[-1], p) > 1.5):
                self.step_path.append(p)
        self.return_index = None
        self.log("mine_transaction_begin", target=target["_id"], block=target["_name"], kind=kind, direction=self.mine_direction)

    def begin_gather(self, obs: dict, target: dict, tool: str | None, purpose: str, now: float) -> None:
        needs_approach = not bool(target.get("in_reach"))
        first_stage = "approach" if needs_approach else ("equip" if tool else "gather")
        self.job = {
            "kind": "gather", "stage": first_stage, "target": target["_id"],
            "name": target["_name"], "position": target["_pos"], "tool": tool,
            "started": now, "before_inv": inv_signature(inventory(obs)), "purpose": purpose,
            "approaches": 0,
        }
        self.log("gather_transaction_begin", target=target["_id"], block=target["_name"], purpose=purpose, stage=first_stage)

    def current_job_row(self, obs: dict) -> dict | None:
        return self.target_row(obs, str(self.job.get("target", ""))) if self.job else None

    def item_drop(self, obs: dict, target: tuple[float, float, float] | None) -> dict | None:
        choices = []
        for row in nearby_entities(obs):
            if row["_name"] not in {"item", "item_stack", "item_entity"} or row.get("dead") or row["_pos"] is None:
                continue
            if row["_distance"] <= 10 and (target is None or distance(row["_pos"], target) <= 8):
                choices.append(row)
        choices.sort(key=lambda row: row["_distance"])
        return choices[0] if choices else None

    def move_to(self, obs: dict, target: tuple[float, float, float], now: float) -> str | None:
        # A listed safe stand is preferred; goto is used only for the newly
        # opened cell, whose safety is checked again by the adaptor.
        stands = self.safe_stands(obs, target)
        if stands:
            return "navigate " + stands[0]["_id"]
        return "goto %.2f %.2f %.2f" % target if self.allowed("goto") else None

    def job_command(self, obs: dict, now: float) -> str | None:
        job = self.job
        if not job:
            return None
        kind = str(job.get("kind"))
        target_id = str(job.get("target", ""))
        inv = inventory(obs)
        before = job.get("before_inv", ())
        if kind == "gather":
            if inv_signature(inv) != before:
                self.completed_targets.add(target_id)
                self.log("resource_confirmed", target=target_id, method="gather")
                self.job = None
                return None
            row = self.current_job_row(obs)
            if row is None:
                self.suppress(target_id, now, 150.0, "target_missing")
                self.job = None
                return None
            if now - number(job.get("started"), now) > 60.0:
                self.suppress(target_id, now, 150.0, "gather_timeout")
                self.job = None
                return None
            stage = str(job.get("stage", "gather"))
            if stage == "approach":
                if row.get("in_reach"):
                    job["stage"] = "equip" if job.get("tool") else "gather"
                    return None
                if int(job.get("approaches", 0)) >= 3 or now - number(job.get("started"), now) > 30:
                    self.suppress(target_id, now, 150.0, "approach_limit")
                    self.job = None
                    return None
                stands = self.safe_stands(obs, row["_pos"])
                if stands:
                    job["approaches"] = int(job.get("approaches", 0)) + 1
                    return "navigate " + stands[0]["_id"]
                # Listed b: candidates are legal block-interaction goals.  In
                # particular this reaches a hidden trunk that has no nearby s:
                # stand; the gather job still waits for in_reach before mining.
                if self.allowed("navigate") and (row.get("visible") or row["_name"].endswith(WOOD_SUFFIXES)):
                    job["approaches"] = int(job.get("approaches", 0)) + 1
                    return "navigate " + target_id
                if row.get("visible") and self.allowed("goto"):
                    job["approaches"] = int(job.get("approaches", 0)) + 1
                    return "goto %.2f %.2f %.2f" % row["_pos"]
                self.suppress(target_id, now, 150.0, "no_safe_approach")
                self.job = None
                return None
            if stage == "equip":
                tool = item_name(job.get("tool"))
                if item_name(self_row(obs).get("held_item")) != tool:
                    return "equip %s hand" % tool
                job["stage"] = "gather"
            return "gather %s 1" % target_id

        row = self.current_job_row(obs)
        stage = str(job.get("stage", "approach"))
        if stage == "approach":
            if row is None:
                self.suppress(target_id, now, 150.0, "target_missing")
                self.job = None
                return None
            if row.get("in_reach"):
                job["stage"] = "equip"
                return None
            if int(job.get("approaches", 0)) >= 3 or now - number(job.get("started"), now) > 25:
                self.suppress(target_id, now, 150.0, "approach_limit")
                self.job = None
                return None
            stands = self.safe_stands(obs, row["_pos"])
            if stands:
                job["approaches"] = int(job.get("approaches", 0)) + 1
                return "navigate " + stands[0]["_id"]
            if row.get("visible") and self.allowed("goto"):
                job["approaches"] = int(job.get("approaches", 0)) + 1
                return "goto %.2f %.2f %.2f" % row["_pos"]
            self.suppress(target_id, now, 150.0, "no_safe_approach")
            self.job = None
            return None
        if stage == "equip":
            tool = item_name(job.get("tool"))
            if tool and item_name(self_row(obs).get("held_item")) != tool:
                job["stage"] = "look"
                return "equip %s hand" % tool
            job["stage"] = "look"
            return None
        if stage == "look":
            q = row["_pos"] if row else job.get("position")
            if q is None:
                self.job = None
                return None
            job["stage"] = "dig"
            return "look_at %.2f %.2f %.2f" % q
        if stage == "dig":
            row = self.current_job_row(obs)
            if row is None:
                job["stage"] = "move" if kind == "step" else "collect"
                return None
            if not row.get("in_reach"):
                job["stage"] = "approach"
                return None
            q = row["_pos"] or job.get("position")
            if q is None:
                self.job = None
                return None
            job["stage"] = "wait_dig"
            return "dig %d %d %d" % tuple(int(round(v)) for v in q)
        if stage == "wait_dig":
            row = self.current_job_row(obs)
            if row is None:
                job["stage"] = "move" if kind == "step" else "collect"
                return None
            if inv_signature(inv) != before and kind != "step":
                job["stage"] = "collect"
                return None
            if now - number(job.get("started"), now) > 36:
                self.suppress(target_id, now, 150.0, "dig_timeout")
                self.job = None
            return None
        if stage == "collect":
            if inv_signature(inv) != before:
                self.completed_targets.add(target_id)
                self.log("resource_confirmed", target=target_id, method="pickup")
                self.job = None
                return None
            drop = self.item_drop(obs, job.get("position"))
            if drop and drop.get("_pos"):
                action = self.move_to(obs, drop["_pos"], now)
                if action:
                    return action
            if now - number(job.get("started"), now) > 46:
                self.suppress(target_id, now, 150.0, "pickup_timeout")
                self.job = None
            return None
        if stage == "move":
            q = job.get("position")
            if not q:
                self.job = None
                return None
            p = self_position(obs)
            expected = job.get("expected")
            if expected is None:
                # The removed block is now the next feet cell.  This is the
                # structural distinction from 0016's stationary excavation.
                expected = (q[0], q[1], q[2])
                job["expected"] = expected
            moved_from_start = distance(p, job.get("before_step_pos")) >= 0.75
            if p is not None and distance(p, expected) <= 1.2 and moved_from_start:
                self.completed_targets.add(target_id)
                if not self.step_path or distance(self.step_path[-1], expected) > 0.8:
                    self.step_path.append(tuple(expected))
                self.steps_since_refresh += 1
                self.log("stair_step_confirmed", target=target_id, position=expected, depth=self.mine_depth())
                self.job = None
                return None
            if now - number(job.get("started"), now) > 52:
                self.suppress(target_id, now, 150.0, "step_move_timeout")
                self.job = None
                return None
            action = "goto %.2f %.2f %.2f" % expected
            if self.allowed("goto") and not self.blocked(action, now):
                return action
            return self.move_to(obs, expected, now)
        self.job = None
        return None

    def mine_depth(self) -> int:
        p = self_position(self.last_obs)
        return int(math.floor(p[1])) if p else 0

    def start_stair_or_resource(self, obs: dict, now: float, names: set[str], tool: str, purpose: str) -> str | None:
        target = self.choose_resource(obs, names=names)
        if target:
            # Exposed valuable blocks are gathered, not removed with a bare dig.
            self.begin_gather(obs, target, tool, purpose, now)
            return None
        if names == STONE_BLOCKS and self.refresh_pending:
            self.refresh_pending = False
        elif names == STONE_BLOCKS and self.steps_since_refresh >= 8:
            # A short staircase should not tunnel one biome indefinitely.  One
            # safe frontier move refreshes the loaded candidate window, then
            # the next compiler call resumes the same verified mine policy.
            action = self.explore_action(obs, now)
            if action and action != "noop":
                self.steps_since_refresh = 0
                self.refresh_pending = True
                self.log("staircase_frontier_refresh", steps=8)
                return action
        target = self.staircase_target(obs, now)
        if target:
            self.begin_direct_mine(obs, target, "step", tool, now)
            return None
        return self.explore_action(obs, now)

    def manual_command(self, obs: dict, now: float) -> str | None:
        if self.manual_stage == "look":
            self.manual_stage = "move"
            self.manual_until = now + 3.0
            return "look_at %.2f %.2f %.2f" % self.manual_target if self.manual_target else "noop"
        if self.manual_stage == "move":
            if now < self.manual_until:
                return "controls forward,sprint"
            self.manual_stage = "none"
            self.manual_until = -1.0
            return "controls none"
        return None

    def mode(self, obs: dict, inv: dict[str, int]) -> str:
        tables = self.table_rows(obs)
        if (
            wood_total(inv) == 0
            and plank_total(inv) == 0
            and not inv.get("crafting_table")
            and not tables
            and not any(inv.get(x) for x in PICKS)
        ):
            # A stale table_hint can survive a death while the player's
            # dropped opening kit is gone.  Reacquire wood before attempting
            # to route to that old workstation.
            return "wood"
        if not tables and inv.get("crafting_table"):
            return "table"
        if not tables and not self.table_hint and not inv.get("crafting_table"):
            return "table"
        if not any(inv.get(x) for x in ("wooden_pickaxe",)):
            return "wooden_tools"
        if not any(inv.get(x) for x in STONE_PICKS):
            return "stone"
        raw = inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0)
        if not inv.get("iron_ingot") and raw == 0:
            return "iron"
        if raw and not inv.get("iron_ingot"):
            return "smelt"
        if inv.get("iron_ingot") and not inv.get("shield"):
            return "shield"
        return "progress"

    def compile_action(self, obs: dict, now: float) -> str | None:
        inv = inventory(obs)
        mode = self.mode(obs, inv)
        self.last_mode = mode
        table = self.near_table(obs)

        # Food is an opportunistic but real resource transaction.  It is
        # deliberately limited to observed blocks, never inferred from biome.
        if food_total(inv) == 0:
            food_target = self.choose_resource(obs, names=FOOD_BLOCKS)
            if food_target:
                self.begin_gather(obs, food_target, None, "observed food", now)
                return None

        if mode == "wood":
            target = self.choose_resource(obs, suffixes=WOOD_SUFFIXES, allow_hidden=True)
            if target:
                self.begin_gather(obs, target, None, "opening wood", now)
                return None
            return self.explore_action(obs, now)

        if mode == "table":
            if inv.get("crafting_table"):
                return self.table_route(obs, now)
            if plank_total(inv) < 4:
                source = self.plank_source(inv)
                return self.craft_action(source[:-4] + "_planks" if source.endswith("_log") else source[:-5] + "_planks", 4, inv) if source else self.explore_action(obs, now)
            action = self.craft_action("crafting_table", 1, inv)
            return action or self.explore_action(obs, now)

        if mode == "wooden_tools":
            if not table:
                return self.table_route(obs, now) or self.explore_action(obs, now)
            source = self.plank_source(inv)
            if plank_total(inv) < 3:
                if source:
                    item = source[:-4] + "_planks" if source.endswith("_log") else source[:-5] + "_planks"
                    return self.craft_action(item, 4, inv)
                target = self.choose_resource(obs, suffixes=WOOD_SUFFIXES, allow_hidden=True)
                if target:
                    self.begin_gather(obs, target, None, "extra wood for wooden pick", now)
                    return None
                return self.explore_action(obs, now)
            if inv.get("stick", 0) < 2 and plank_total(inv) >= 2:
                return self.craft_action("stick", 4, inv)
            if not inv.get("wooden_pickaxe"):
                return self.craft_action("wooden_pickaxe", 1, inv, table)
            return None

        if mode == "stone":
            cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
            if cobble < 3:
                tool = next((x for x in PICKS if inv.get(x)), "wooden_pickaxe")
                return self.start_stair_or_resource(obs, now, STONE_BLOCKS, tool, "stone for moving staircase")
            if inv.get("stick", 0) < 2 and plank_total(inv) >= 2:
                return self.craft_action("stick", 4, inv)
            if not table:
                return self.table_route(obs, now) or self.explore_action(obs, now)
            return self.craft_action("stone_pickaxe", 1, inv, table)

        if mode == "iron":
            target = self.choose_resource(obs, names=IRON_BLOCKS)
            if target:
                self.begin_gather(obs, target, "stone_pickaxe", "exposed iron", now)
                return None
            if not inv.get("coal") and not inv.get("charcoal"):
                fuel = self.choose_resource(obs, names=COAL_BLOCKS)
                if fuel:
                    self.begin_gather(obs, fuel, "stone_pickaxe", "furnace fuel", now)
                    return None
            return self.start_stair_or_resource(obs, now, IRON_BLOCKS | COAL_BLOCKS | STONE_BLOCKS, "stone_pickaxe", "underground iron search")

        if mode == "smelt":
            raw = inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0)
            furnaces = self.furnace_rows(obs)
            fuel = next((x for x in ("coal", "charcoal") if inv.get(x)), "")
            if not fuel:
                coal = self.choose_resource(obs, names=COAL_BLOCKS)
                if coal:
                    self.begin_gather(obs, coal, "stone_pickaxe", "fuel for iron smelting", now)
                    return None
                return self.start_stair_or_resource(obs, now, COAL_BLOCKS | STONE_BLOCKS, "stone_pickaxe", "find furnace fuel")
            if not furnaces and not inv.get("furnace"):
                if inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0) >= 8 and table:
                    return self.craft_action("furnace", 1, inv, table)
                return self.start_stair_or_resource(obs, now, STONE_BLOCKS | COAL_BLOCKS, "stone_pickaxe", "stone and fuel for furnace")
            if not furnaces:
                return self.furnace_route(obs, now)
            furnace = min(furnaces, key=lambda row: row["_distance"])
            if furnace["_distance"] > 4.8:
                return self.furnace_route(obs, now)
            fid = furnace["_id"]
            if raw and fuel and fid not in self.smelt_started:
                self.smelt_started[fid] = now
                input_item = next((x for x in ("raw_iron", "iron_ore", "deepslate_iron_ore") if inv.get(x)), "raw_iron")
                return "furnace_start %s %s %d %s %d" % (fid, input_item, min(3, inv.get(input_item, raw)), fuel, min(3, inv[fuel]))
            if fid in self.smelt_started and now - self.smelt_started[fid] >= 34:
                return "furnace_take " + fid
            return "noop"

        if mode == "shield":
            source = self.plank_source(inv)
            if plank_total(inv) < 5:
                if source:
                    item = source[:-4] + "_planks" if source.endswith("_log") else source[:-5] + "_planks"
                    return self.craft_action(item, 8, inv)
                target = self.choose_resource(obs, suffixes=WOOD_SUFFIXES, allow_hidden=True)
                if target:
                    self.begin_gather(obs, target, None, "wood for shield", now)
                    return None
                return self.explore_action(obs, now)
            if table:
                return self.craft_action("shield", 1, inv, table)
            return self.table_route(obs, now) or self.explore_action(obs, now)

        # With the first safety milestones complete, keep looking for exposed
        # iron/coal rather than idling.  This also gives an already-equipped
        # stone pick a chance to find the next resource on unseen seeds.
        if number(self_row(obs).get("food"), 20) <= 12:
            food = best_food(inv)
            if food:
                return ("equip %s hand" % food) if item_name(self_row(obs).get("held_item")) != food else "eat " + food
        return self.start_stair_or_resource(obs, now, IRON_BLOCKS | COAL_BLOCKS, "stone_pickaxe", "continue resource search")

    def emergency(self, obs: dict, inv: dict[str, int], now: float) -> str | None:
        me = self_row(obs)
        if me.get("dead"):
            return "respawn"
        if number(obs.get("observation_age_s"), 0) > 3.0:
            return "cancel" if has_task(obs) else "noop"
        hazards = self.safety.hazard_list(obs) if self.safety else []
        hard = any(h["kind"] in {"lava_or_fire", "drowning", "fall"} or h["kind"] in HAZARD_BLOCKS for h in hazards)
        if hard:
            return self.safety.fallback(obs, hazards, inv) if self.safety else "controls forward,jump"
        if number(me.get("health"), 20) <= 6 or number(me.get("food"), 20) <= 5:
            food = best_food(inv)
            if food:
                if item_name(me.get("held_item")) != food and self.allowed("equip"):
                    return "equip %s hand" % food
                if self.allowed("eat"):
                    return "eat " + food
        if hazards:
            return self.safety.decide(obs, inv, now) if self.safety else None
        return None

    def run_episode(self, game: Any) -> None:
        self.names = set(getattr(getattr(game, "action_spec", None), "names", ()) or ())
        self.safety = SafetyJev(self.ctx, self.names, self.log)
        self.last_progress_at = self.now()
        while not self.ctx.done:
            now = self.now()
            obs = parse_observation(self.ctx.observe())
            if not obs:
                self.post("noop", {}, now, "missing_observation")
                self.ctx.wait_for_tick()
                continue
            self.last_obs = obs
            self.process_feedback(obs, now)
            self.observe_progress(obs, now)
            me = self_row(obs)
            if me.get("dead"):
                self.job = None
                self.active = None
                self.manual_stage = "none"
                if not self.dead_seen:
                    self.dead_seen = True
                    self.post("respawn", obs, now, "ordinary_respawn")
                else:
                    self.post("noop", obs, now, "await_respawn")
                self.ctx.wait_for_tick()
                continue
            self.dead_seen = False

            urgent = self.emergency(obs, inventory(obs), now)
            if urgent:
                # A new movement/use action cannot own the hands while another
                # bounded task is active.  Cancel first, then decide anew.
                if has_task(obs) or self.active:
                    if urgent not in {"noop", "respawn"}:
                        self.post("cancel", obs, now, "hazard_interrupt")
                    else:
                        self.post("noop", obs, now, "hazard_pending")
                else:
                    self.post(urgent, obs, now, "safety_override")
                self.ctx.wait_for_tick()
                continue

            self.process_active(obs, now)
            if has_task(obs):
                self.post("noop", obs, now, "continue_bounded_job")
                self.ctx.wait_for_tick()
                continue
            if self.active:
                self.post("noop", obs, now, "await_postcondition")
                self.ctx.wait_for_tick()
                continue

            if self.job:
                action = self.job_command(obs, now)
                self.post(action or "noop", obs, now, "mine_transaction")
                self.ctx.wait_for_tick()
                continue

            manual = self.manual_command(obs, now)
            if manual:
                self.post(manual, obs, now, "frontier_escape")
                self.ctx.wait_for_tick()
                continue

            action = self.compile_action(obs, now)
            self.post(action or "noop", obs, now, "milestone_compiler")
            self.ctx.wait_for_tick()


def build(ctx: Any) -> StaircaseController:
    return StaircaseController(ctx)
