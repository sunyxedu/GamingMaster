"""Candidate 0003: keep Jev-only control; ground its Choice in complete commands.

The sole experimental change is the action interface: a dynamic, observed-target
menu replaces the parent's unexecutable bare action names. No Astra, route
planner, background worker, macro policy, or seed-specific information is used.
"""
from __future__ import annotations

import json
import math
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


TOKEN = re.compile(r"^[a-z0-9_]+$")
CANDIDATE = re.compile(r"^[bs]:-?\d+,-?\d+,-?\d+$")
FOOD = {
    "apple", "bread", "baked_potato", "carrot", "golden_carrot",
    "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
    "cooked_rabbit", "cooked_cod", "cooked_salmon", "sweet_berries",
    "melon_slice", "beef", "porkchop", "mutton", "rabbit", "cod",
    "salmon", "potato", "mushroom_stew", "rabbit_stew", "beetroot_soup",
}
ANIMALS = {"cow", "pig", "sheep", "chicken", "rabbit", "cod", "salmon"}
WORKSTATIONS = {"crafting_table", "furnace", "blast_furnace", "smoker"}
CONTAINERS = {"chest", "trapped_chest", "barrel"}
DANGER_BLOCKS = {"lava", "fire", "soul_fire", "magma_block", "cactus",
                 "powder_snow", "campfire", "soul_campfire"}
SMELT = {"raw_iron": "iron_ingot", "raw_gold": "gold_ingot",
         "beef": "cooked_beef", "porkchop": "cooked_porkchop",
         "mutton": "cooked_mutton", "chicken": "cooked_chicken",
         "rabbit": "cooked_rabbit", "cod": "cooked_cod",
         "salmon": "cooked_salmon", "potato": "baked_potato"}


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError, OverflowError):
        return default


def mapping(value):
    return value if isinstance(value, dict) else {}


def records(value):
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def item_name(value):
    if not isinstance(value, str):
        return ""
    name = value.removeprefix("minecraft:")
    return name if TOKEN.fullmatch(name) else ""


def position(value):
    if isinstance(value, dict):
        value = [value.get(k) for k in ("x", "y", "z")]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    point = [number(v, math.inf) for v in value]
    return point if all(math.isfinite(v) for v in point) else None


def distance(a, b):
    if a is None or b is None:
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def location_words(point, origin):
    """Translate an action's numeric target into immediate spatial meaning."""
    if point is None or origin is None:
        return "location uncertain"
    dx, dy, dz = (point[i] - origin[i] for i in range(3))
    directions = []
    if abs(dz) > 1.0:
        directions.append("south" if dz > 0 else "north")
    if abs(dx) > 1.0:
        directions.append("east" if dx > 0 else "west")
    direction = "-".join(directions) or "adjacent"
    dist = distance(point, origin)
    span = "within a few steps" if dist <= 3 else "nearby" if dist <= 10 else "farther away"
    elevation = "higher" if dy > 1.5 else "lower" if dy < -1.5 else "similar elevation"
    return f"{direction}, {span}, {elevation}"


def observed(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return {}
    return mapping(raw)


def inventory(obs):
    # Only the server-audited personal inventory can authorize consuming items.
    # Slot/client/cursor/output previews are never inventory substitutes.
    inv = {}
    for key, value in mapping(obs.get("inventory")).items():
        name = item_name(key)
        if isinstance(value, dict):
            value = value.get("count", 0)
        count = max(0, int(number(value)))
        if name and count:
            inv[name] = count
    return inv


def block_id(record):
    ident = record.get("id")
    return ident if isinstance(ident, str) and CANDIDATE.fullmatch(ident) else ""


def name_of(record):
    return item_name(record.get("name", ""))


def held_name(obs):
    held = mapping(obs.get("self")).get("held_item")
    return name_of(held) if isinstance(held, dict) else item_name(held)


def useful_block(name):
    return (name.endswith(("_log", "_stem", "_wood", "_planks", "_ore",
                           "_wool", "_bed"))
            or name in {"stone", "cobblestone", "deepslate", "cobbled_deepslate",
                        "coal_block", "iron_block", "gold_block", "hay_block",
                        "melon", "pumpkin", "wheat", "carrots", "potatoes",
                        "sweet_berry_bush", "obsidian", "gravel", "dirt",
                        "crafting_table", "furnace"})


class GroundedMenu:
    """Parameter binding and action protocol, not a strategic policy."""

    def __init__(self):
        self.last_action = "noop"

    def build(self, obs, names):
        options = {}
        player = mapping(obs.get("self"))
        inv = inventory(obs)
        pos = position(player.get("position"))
        blocks = [c for c in records(obs.get("candidates"))
                  if c.get("kind") == "block" and block_id(c)]
        stands = [c for c in records(obs.get("candidates"))
                  if c.get("kind") == "stand" and block_id(c)]
        entities = records(obs.get("entities"))
        blocks.sort(key=lambda b: number(b.get("distance"), math.inf))
        stands.sort(key=lambda b: number(b.get("distance"), math.inf))
        entities.sort(key=lambda b: number(b.get("distance"), math.inf))
        task = obs.get("task")
        age = number(obs.get("observation_age_s"), math.inf)

        def add(action, description):
            # Always reserve a slot for the explicit no-match/wait choice.
            if action.split()[0] in names and (len(options) < 179 or action == "noop"):
                options[action] = description

        if player.get("dead"):
            add("respawn", "Respawn normally after the observed death; inventory stays dropped.")
            add("noop", "Wait; does not respawn or recover inventory.")
            return options
        if age > 2.0:
            add("noop", "Snapshot is stale: preserve any bounded job and release idle controls.")
            add("cancel", "Stop a potentially dangerous stale job and release all controls.")
            return options
        if task:
            add("noop", "CONTINUE the current bounded job without restarting it. Usually correct while busy.")
            add("cancel", "INTERRUPT the current job only for danger, an invalid target, or deliberate abandonment.")
            return options

        # A container is a separate interface. Count only its explicit storage,
        # never player slots/cursor/output, if it provides container_items.
        window = mapping(obs.get("window"))
        if window:
            contents = window.get("container_items")
            if isinstance(contents, dict):
                contents = [{"name": k, "count": v} for k, v in contents.items()]
            for stack in records(contents)[:24]:
                name = name_of(stack)
                count = min(64, max(0, int(number(stack.get("count")))))
                if name and count:
                    add(f"transfer take {name} {count}", f"Take the observed {name} stack from this open container.")
            add("close_window", "Close the open window before moving or crafting.")
            add("noop", "Wait with the current window open.")
            return options

        # Instantiate gather from actual block IDs; never from a made-up location.
        seen_types = {}
        pick = any(k.endswith("_pickaxe") for k in inv)
        stone_pick = any(k in inv for k in ("stone_pickaxe", "iron_pickaxe",
                                            "diamond_pickaxe", "netherite_pickaxe"))
        iron_pick = any(k in inv for k in ("iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"))
        for b in blocks:
            name = name_of(b)
            if not useful_block(name) or name in DANGER_BLOCKS:
                continue
            if b.get("harvestable") is False:
                continue
            if (name.endswith("_ore") or name in {"stone", "cobblestone",
                    "deepslate", "cobbled_deepslate", "furnace", "iron_block",
                    "gold_block", "coal_block"}) and not pick:
                continue
            if name in {"iron_ore", "deepslate_iron_ore", "copper_ore", "deepslate_copper_ore"} and not stone_pick:
                continue
            if name in {"gold_ore", "deepslate_gold_ore", "diamond_ore",
                        "deepslate_diamond_ore", "redstone_ore",
                        "deepslate_redstone_ore", "iron_block", "gold_block"} and not iron_pick:
                continue
            if name == "obsidian" and not any(k in inv for k in ("diamond_pickaxe", "netherite_pickaxe")):
                continue
            if distance(position(b.get("position")), pos) > 48:
                continue
            bp = position(b.get("position"))
            if bp and pos and (math.floor(pos[0]), math.floor(pos[1])-1, math.floor(pos[2])) == tuple(bp):
                continue
            if seen_types.get(name, 0) >= 2:
                continue
            seen_types[name] = seen_types.get(name, 0) + 1
            visibility = "visible" if b.get("visible") else "exposed but occluded"
            reach = "within reach" if b.get("in_reach") else "requires safe navigation"
            where = location_words(bp, pos)
            output = {"stone": "cobblestone", "deepslate": "cobbled_deepslate",
                      "coal_ore": "coal", "deepslate_coal_ore": "coal",
                      "iron_ore": "raw_iron", "deepslate_iron_ore": "raw_iron"}.get(name, name)
            stock = f"Already carrying {inv.get(output, 0)} {output}."
            workstation_note = " Removes this placed workstation from use." if name in WORKSTATIONS else ""
            add(f"gather {block_id(b)} 1",
                f"Mine and collect one {name}; {where}; {visibility}, {reach}. "
                f"The bounded job navigates and picks up. {stock}{workstation_note}")

        tables = [b for b in blocks + records(obs.get("workstations"))
                  if name_of(b) == "crafting_table" and block_id(b)
                  and number(b.get("distance"), math.inf) <= 4.5]
        table = block_id(tables[0]) if tables else ""

        # Exact bootstrap recipes expose valid options, not an automatic build
        # order. More advanced crafting comes from the observation's recipe hints.
        planks = sum(v for k, v in inv.items() if k.endswith("_planks"))
        sticks = inv.get("stick", 0)
        cobble = inv.get("cobblestone", 0)
        iron = inv.get("iron_ingot", 0)

        offered_recipes = set()

        def recipe(name, feasible, why, needs_table=False, count=1):
            if feasible and (not needs_table or table):
                suffix = f" {table}" if needs_table else ""
                stock = f"Already carried: {inv.get(name, 0)} {name}. "
                add(f"craft {name} {count}{suffix}",
                    f"Craft {name}: {why}. {stock}Requested count is NEW output items.")
                offered_recipes.add(name)

        for name in inv:
            species = ""
            if name.endswith("_log"):
                species = name[:-4].removeprefix("stripped_")
            elif name in {"crimson_stem", "warped_stem"}:
                species = name[:-5]
            if species:
                recipe(species + "_planks", True, "turn one carried log/stem into four planks for crafting", count=4)
        recipe("crafting_table", planks >= 4, "four carried planks unlock table recipes and the workbench milestone")
        recipe("stick", planks >= 2, "two carried planks produce four sticks for tools", count=4)
        recipe("wooden_pickaxe", planks >= 3 and sticks >= 2, "three planks and two sticks make a basic mining tool", True)
        recipe("wooden_axe", planks >= 3 and sticks >= 2, "three planks and two sticks make a woodcutting tool", True)
        recipe("stone_pickaxe", cobble >= 3 and sticks >= 2, "three cobblestone and two sticks allow iron mining", True)
        recipe("stone_axe", cobble >= 3 and sticks >= 2, "three cobblestone and two sticks make a stronger tool/weapon", True)
        recipe("stone_sword", cobble >= 2 and sticks >= 1, "two cobblestone and one stick make a weapon", True)
        recipe("furnace", cobble >= 8, "eight cobblestone make a furnace for real-time smelting", True)
        recipe("shield", planks >= 6 and iron >= 1, "six planks and one iron ingot provide defense and a milestone", True)
        recipe("bucket", iron >= 3, "three iron ingots make a bucket", True)
        recipe("iron_pickaxe", iron >= 3 and sticks >= 2, "three iron ingots and two sticks make a mining upgrade", True)
        recipe("bread", inv.get("wheat", 0) >= 3, "three wheat become food", True)
        recipe("wheat", inv.get("hay_block", 0) >= 1, "one hay bale becomes nine wheat", count=9)
        recipe("blaze_powder", inv.get("blaze_rod", 0) >= 1, "one blaze rod becomes two blaze powder", count=2)
        recipe("ender_eye", inv.get("blaze_powder", 0) >= 1 and inv.get("ender_pearl", 0) >= 1, "combine one powder and one pearl")
        for name, count in inv.items():
            if name.endswith("_wool") and count >= 3:
                recipe(name[:-5] + "_bed", planks >= 3, "three matching wool and three planks make a bed", True)

        hints = obs.get("feasible_recipes")
        if isinstance(hints, dict):
            hints = [{"name": k, **(v if isinstance(v, dict) else {})} for k, v in hints.items() if v is not False]
        if isinstance(hints, list):
            for hint in hints[:40]:
                if isinstance(hint, str):
                    hint = {"name": hint}
                hint = mapping(hint)
                name = item_name(hint.get("name") or hint.get("item"))
                if not name:
                    name = name_of(mapping(hint.get("result")))
                if (not name or name in offered_recipes or hint.get("craftable") is False
                        or hint.get("feasible") is False):
                    continue
                # Appending an actual nearby table is safe for either grid size.
                if hint.get("requires_table") and not table:
                    continue
                suffix = f" {table}" if table else ""
                add(f"craft {name} 1{suffix}",
                    f"Craft {name}; the current observation reports a feasible recipe. Already carried: {inv.get(name, 0)}.")
                offered_recipes.add(name)

        # Only place on an observed solid top with locally verified free space.
        collision = {}
        for cell in mapping(obs.get("world")).get("local_collision", []) or []:
            if isinstance(cell, list) and len(cell) == 4:
                collision[tuple(cell[:3])] = cell[3]
        place_refs = []
        if pos:
            base = tuple(math.floor(x) for x in pos)
            occupied = {(int(p[0]), int(p[1]), int(p[2])): name_of(b)
                        for b in blocks if (p := position(b.get("position"))) is not None}
            for b in blocks:
                p = position(b.get("position"))
                if not p or not b.get("visible") or not b.get("in_reach"):
                    continue
                target = (int(p[0]), int(p[1]) + 1, int(p[2]))
                rel = tuple(target[i] - base[i] for i in range(3))
                below = (rel[0], rel[1] - 1, rel[2])
                if collision.get(below) != "solid" or collision.get(rel) != "passable":
                    continue
                if target in occupied or (target[0] == base[0] and target[2] == base[2]):
                    continue
                place_refs.append(b)
                if len(place_refs) >= 2:
                    break
        for name in ("crafting_table", "furnace"):
            if inv.get(name):
                for b in place_refs:
                    add(f"place {name} {block_id(b)} up",
                        f"Place carried {name} on the observed clear solid surface; enables nearby workstation use.")

        hand = held_name(obs)
        for name in sorted(inv):
            if name.endswith(("_sword", "_axe", "_pickaxe")) and name != hand:
                add(f"equip {name} hand", f"Hold carried {name}; equipping alone does not attack or mine.")
            if name in FOOD and number(player.get("food"), 20) < 20:
                add(f"eat {name}", f"Eat carried {name} to restore hunger and enable healing.")
        if inv.get("shield"):
            add("shield on", "Equip/use the carried shield against an immediate threat.")
            add("shield off", "Release shield blocking.")

        for b in blocks:
            name, ident = name_of(b), block_id(b)
            reachable = b.get("visible") and b.get("in_reach")
            if name in CONTAINERS and reachable:
                add(f"open_container {ident}", f"Open the observed {name} for inspection and real item transfer.")
            if name.endswith("_bed") and reachable and player.get("dimension") == "overworld":
                add(f"sleep {ident}", "Use this Overworld bed when vanilla permits to set spawn/sleep.")
            if name == "furnace" and reachable:
                add(f"furnace_take {ident}", "Retrieve any finished output from this observed furnace.")
                fuel = next((k for k in ("coal", "charcoal") if inv.get(k)), None)
                if fuel is None:
                    fuel = next((k for k in inv if k.endswith("_planks")), None)
                if fuel:
                    for source, result in SMELT.items():
                        if inv.get(source):
                            add(f"furnace_start {ident} {source} 1 {fuel} 1",
                                f"Start smelting one {source} into {result} with real carried fuel; takes time.")

        for e in entities[:16]:
            ident = e.get("id")
            if not isinstance(ident, (str, int)) or not str(ident).isdigit():
                continue
            name = name_of(e)
            d = number(e.get("distance"), math.inf)
            if e.get("visible") and d <= 3.0 and (e.get("hostile") or name in ANIMALS):
                add(f"attack {ident}", f"Melee the visible in-range {name}; attack cooldown is enforced by the game.")

        # Stand IDs are the interface's explicit safe-feet navigation candidates.
        # Offer distant/local alternatives, without selecting a strategic route.
        landmarks = []
        for label, targets in (
                ("wood", [b for b in blocks if name_of(b).endswith(("_log", "_stem"))]),
                ("food animal", [e for e in entities if name_of(e) in ANIMALS]),
                ("hostile mob", [e for e in entities if e.get("hostile")])):
            if targets:
                landmarks.append((label, position(targets[0].get("position"))))
        chosen_stands = stands[:4] + stands[-4:]
        for b in chosen_stands:
            d = number(b.get("distance"), math.inf)
            if 1.25 < d <= 48:
                bp = position(b.get("position"))
                relation = []
                for label, target in landmarks:
                    if target is not None and pos is not None and bp is not None:
                        delta = distance(bp, target) - distance(pos, target)
                        if abs(delta) > 1:
                            relation.append(("closer to" if delta < 0 else "farther from") + " observed " + label)
                nearby_note = "; " + ", ".join(relation) if relation else ""
                add(f"navigate {block_id(b)}",
                    f"Move to observed safe stand: {location_words(bp, pos)}{nearby_note}. "
                    "Explore or reposition; navigation never mines or bridges.")
        for b in blocks:
            if (name_of(b) in WORKSTATIONS | CONTAINERS or name_of(b).endswith("_bed")) and not b.get("in_reach"):
                add(f"navigate {block_id(b)}", f"Approach the observed {name_of(b)} for interaction.")
        add("noop", "No interaction: stay still/release controls when idle; continues a job if one has started.")
        return options

    def validate(self, action, before, after, names):
        """Rebind against fresh facts, and respect busy/latch semantics."""
        if before.get("episode") != after.get("episode"):
            return "noop", "episode_changed"
        if number(after.get("seq")) - number(before.get("seq")) > 10:
            return "noop", "answer_stale"
        fresh = self.build(after, names)
        if action not in fresh:
            return "noop", "precondition_changed"
        # Identical one-shots are latched; noop must actually reach a tick before
        # a deliberate retry, not merely overwrite a pending action mailbox.
        repeatable = {"noop", "attack", "controls", "shield"}
        if action == self.last_action and action.split()[0] not in repeatable:
            return "noop", "rearm_one_shot"
        return action, "selected"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.menu = GroundedMenu()

    def run_episode(self, game):
        ctx = self.ctx
        if getattr(ctx, "system1", None) is None:
            return
        names = set(game.action_spec.names)
        rules = game.describe()
        while not ctx.done:
            obs = observed(ctx.observe())
            choices = self.menu.build(obs, names)
            if not choices:
                choices = {"noop": "No supported, grounded action is currently available."}
            question = {"act": Choice(
                instructions=(rules + "\nChoose the best COMPLETE executable command from the options. "
                              "Each option is already grounded in current observed IDs and carried items. "
                              "Use the supplied inventory, task and feedback; do not invent arguments. "
                              "Do not confuse equipping, moving, and actually collecting/crafting."),
                criteria=choices)}
            try:
                answers = ctx.decide(obs, question, timeout_s=5.0)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                ctx.act("noop")
                self.menu.last_action = "noop"
                ctx.log("grounded_action_fallback", reason="model_call_failed")
                ctx.wait_for_tick()
                continue
            answer = answers.get("act") if isinstance(answers, dict) else None
            selected = getattr(answer, "value", None)
            if not isinstance(selected, str) or selected not in choices:
                selected = "noop"
            latest = observed(ctx.observe())
            action, reason = self.menu.validate(selected, obs, latest, names)
            ctx.log("grounded_action", selected=selected, posted=action, reason=reason,
                    menu_size=len(choices), seq=latest.get("seq"),
                    probability=mapping(getattr(answer, "probabilities", {})).get(selected),
                    confidence=getattr(answer, "confidence", None))
            ctx.act(action)
            self.menu.last_action = action
            # Consume the mailbox on a real tick. This is part of the command
            # bridge: without it one-shot and rearm can overwrite each other.
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
