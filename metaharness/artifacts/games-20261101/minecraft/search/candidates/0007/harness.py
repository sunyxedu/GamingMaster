"""Stateful grounded controller for Minecraft survival progression."""
from __future__ import annotations

import json
import math
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


IDENT = re.compile(r"^[a-z0-9_]+$")
FOODS = {
    "apple", "bread", "baked_potato", "carrot", "cooked_beef", "cooked_chicken",
    "cooked_mutton", "cooked_porkchop", "cooked_rabbit", "cooked_cod", "cooked_salmon",
    "sweet_berries", "melon_slice", "beef", "chicken", "mutton", "porkchop", "rabbit",
    "cod", "salmon", "potato", "beetroot", "wheat",
}
LOG_SUFFIXES = ("_log", "_stem")
STONE_NAMES = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON_ORES = {"iron_ore", "deepslate_iron_ore"}
COAL_ORES = {"coal_ore", "deepslate_coal_ore"}
DANGER_NAMES = {"lava", "fire", "soul_fire", "magma_block", "cactus", "powder_snow"}
PASSIVE = {"cow", "pig", "sheep", "chicken", "rabbit"}


def num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def item_name(value):
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", "")))
    value = str(value or "").removeprefix("minecraft:")
    return value if IDENT.fullmatch(value) else ""


def records(value):
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def point(value):
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    out = [num(value[i], math.inf) for i in range(3)]
    return out if all(math.isfinite(x) for x in out) else None


def dist(a, b):
    if a is None or b is None:
        return math.inf
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def inventory(obs):
    """Read only the authoritative personal inventory for prerequisites."""
    raw = obs.get("inventory")
    if isinstance(raw, dict) and isinstance(raw.get("items"), (dict, list)):
        raw = raw["items"]
    out = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            count = value.get("count", 0) if isinstance(value, dict) else value
            key = item_name(key)
            count = max(0, int(num(count)))
            if key and count:
                out[key] = out.get(key, 0) + count
    elif isinstance(raw, list):
        for row in records(raw):
            key = item_name(row)
            count = max(0, int(num(row.get("count"))))
            if key and count:
                out[key] = out.get(key, 0) + count
    return out


def feedback_text(obs):
    value = obs.get("action_feedback")
    if value is None or value == "":
        value = obs.get("last_result")
    if value is None or value == "":
        value = obs.get("action_result", "")
    if isinstance(value, dict):
        value = " ".join(str(value.get(k, "")) for k in ("status", "result", "message", "error", "reason"))
    return str(value or "").lower()


def is_bad_feedback(text):
    return any(word in text for word in ("reject", "error", "failed", "fail", "timeout", "unreachable", "cannot", "invalid"))


def is_good_feedback(text):
    return any(word in text for word in ("success", "started", "completed", "crafted", "placed", "collected", "equipped", "opened", "slept"))


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.last_posted = "noop"
        self.active = None
        self.next_decision = 0.0
        self.blocked = {}
        self.attempts = {}
        self.visits = {}
        self.furnace_started = {}
        self.ever_table = False
        self.ever_bed = False
        self.last_inventory = {}
        self.last_progress = 0.0
        self.last_score = None
        self.failures = 0
        self.feedback_seen = ""
        self.macro = "balanced"
        self.macro_version = 0
        self.astra = None
        self.astra_next = 35.0
        self.astra_session = None
        self.jev = None
        self.jev_submitted = 0.0
        self.jev_seq = None
        self.jev_version = 0
        self.jev_options = {}
        self.jev_episode = None
        self.jev_late = False
        self.cancel_pending = False

    def now(self):
        return self.ctx.clock.now()

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def legal(self, action):
        verb = str(action).split(" ", 1)[0]
        return verb in self.allowed

    def state_signature(self, obs):
        me = obs.get("self") or {}
        p = point(me.get("position"))
        progress = obs.get("progress")
        progress_score = progress.get("score", 0) if isinstance(progress, dict) else 0
        score = num(obs.get("score"), num(progress_score, 0))
        advancements = tuple(sorted(str(x) for x in (obs.get("advancements") or [])))
        equipment = json.dumps(me.get("equipment"), sort_keys=True, default=str)
        return (tuple(sorted(inventory(obs).items())),
                tuple(round(x, 1) for x in p) if p else None,
                str(me.get("dimension", "")), round(score, 2), advancements,
                str(me.get("bed_position")), item_name(me.get("held_item")),
                equipment, int(num(me.get("selected_hotbar_slot"), -1)),
                round(num(me.get("food"), -1), 1), bool(me.get("dead")),
                round(num(me.get("health"), -1), 1))

    def observe_progress(self, obs):
        current = inventory(obs)
        if current != self.last_inventory:
            self.last_inventory = dict(current)
            self.last_progress = self.now()
        progress = obs.get("progress")
        score = num(obs.get("score"), num(progress.get("score", 0) if isinstance(progress, dict) else 0))
        if self.last_score is None or score != self.last_score:
            self.last_score = score
            self.last_progress = self.now()
        me = obs.get("self") or {}
        advancements = {str(x) for x in (obs.get("advancements") or [])}
        self.ever_table = self.ever_table or current.get("crafting_table", 0) > 0
        self.ever_bed = (self.ever_bed or
                         any(k.endswith("_bed") and v for k, v in current.items()) or
                         bool(me.get("bed_position")) or
                         any("sleep_in_bed" in x or "set_spawn" in x for x in advancements))
        if obs.get("block_changes"):
            self.last_progress = self.now()

    def post(self, action, obs, purpose=""):
        now = self.now()
        action = str(action or "noop")
        if not self.legal(action):
            action = "noop"
        verb = action.split(" ", 1)[0]
        repeatable = verb in {"noop", "attack", "controls", "shield"}
        if action == self.last_posted and not repeatable:
            action = "noop"
        self.ctx.act(action)
        if action != "noop":
            baseline = self.state_signature(obs)
            self.active = {
                "action": action,
                "started": now,
                "baseline": baseline,
                "baseline_inventory": inventory(obs),
                "baseline_position": point((obs.get("self") or {}).get("position")),
                "limit": self.action_limit(action),
                "purpose": purpose,
            }
            self.attempts[action] = self.attempts.get(action, 0) + 1
            if action.startswith("furnace_start "):
                fields = action.split()
                if len(fields) > 3:
                    self.furnace_started[fields[1]] = (now, int(num(fields[3], 1)))
            elif action.startswith("furnace_take "):
                fields = action.split()
                if len(fields) > 1:
                    self.furnace_started.pop(fields[1], None)
            self.log("posted", action=action, purpose=purpose, attempt=self.attempts[action])
        self.last_posted = action
        return action

    def action_limit(self, action):
        verb = action.split(" ", 1)[0]
        if verb == "gather":
            return 28.0
        if verb in {"navigate", "goto"}:
            return 18.0
        if verb in {"craft", "place", "furnace_start", "furnace_take", "open_container", "sleep"}:
            return 9.0
        return 5.0

    def finish_active(self, obs):
        if self.active is None:
            return False
        now = self.now()
        action = self.active["action"]
        verb = action.split(" ", 1)[0]
        text = feedback_text(obs)
        age = num(obs.get("observation_age_s"), 0)
        task = obs.get("task")
        if verb == "attack" and not task and now - self.active["started"] >= 0.45:
            self.active = None
            self.next_decision = now + 0.05
            return False
        expired = now - self.active["started"] >= self.active["limit"]
        stale = age > 3.0
        if stale or expired:
            self.failures += 1
            self.blocked[action] = now + min(90.0, 12.0 + 10.0 * min(self.attempts.get(action, 1), 6))
            self.log("action_backoff", action=action, feedback=text[:160], stale=stale,
                     expired=expired, blocked_until=self.blocked[action])
            self.active = None
            self.next_decision = now + 0.5
            if task or stale:
                self.ctx.act("cancel")
                self.last_posted = "cancel"
                self.cancel_pending = True
                return True
            return False
        if task:
            return True
        current_inventory = inventory(obs)
        current_position = point((obs.get("self") or {}).get("position"))
        if verb in {"gather", "dig"}:
            changed = (current_inventory != self.active["baseline_inventory"] or
                       bool(obs.get("block_changes")))
        elif verb in {"navigate", "goto", "controls"}:
            changed = current_position != self.active["baseline_position"]
        else:
            changed = self.state_signature(obs) != self.active["baseline"]
        if changed or is_good_feedback(text):
            if action.startswith("craft crafting_table") or action.startswith("place crafting_table"):
                self.ever_table = True
            if action.startswith("sleep "):
                self.ever_bed = True
            self.log("action_confirmed", action=action, changed=changed)
            self.active = None
            self.next_decision = now + 0.35
            return False
        if is_bad_feedback(text):
            self.failures += 1
            self.blocked[action] = now + min(90.0, 12.0 + 10.0 * min(self.attempts.get(action, 1), 6))
            self.log("action_backoff", action=action, feedback=text[:160],
                     blocked_until=self.blocked[action])
            self.active = None
            self.next_decision = now + 0.5
            return False
        return True

    def add(self, options, action, purpose, priority):
        if not self.legal(action):
            return
        if self.blocked.get(action, 0.0) > self.now():
            return
        if action in options:
            return
        options[action] = {"purpose": purpose, "priority": float(priority)}

    def candidates(self, obs):
        blocks = [b for b in records(obs.get("candidates"))
                  if b.get("kind") == "block" and isinstance(b.get("id"), str)]
        stands = [b for b in records(obs.get("candidates"))
                  if b.get("kind") == "stand" and isinstance(b.get("id"), str)]
        entities = records(obs.get("entities", obs.get("nearby_entities")))
        return blocks, stands, entities

    def nearest_stand(self, target, stands, origin):
        tp = point(target.get("position"))
        choices = []
        for stand in stands:
            sp = point(stand.get("position"))
            if sp is None or dist(sp, origin) > 48:
                continue
            score = dist(sp, tp) + 0.12 * dist(sp, origin) + 5.0 * self.visits.get(stand["id"], 0)
            choices.append((score, stand))
        choices.sort(key=lambda x: x[0])
        return choices[0][1] if choices else None

    def resource_action(self, target, stands, origin, count, verb="gather"):
        if target is None:
            return None
        target_id = target.get("id")
        if not target_id:
            return None
        if target.get("harvestable") is False:
            return None
        if target.get("in_reach"):
            return f"{verb} {target_id} {count}"
        stand = self.nearest_stand(target, stands, origin)
        if stand is not None:
            return f"navigate {stand['id']}"
        return None

    def supports(self, obs, blocks, pos):
        collision = {}
        for cell in ((obs.get("world") or {}).get("local_collision") or []):
            if isinstance(cell, list) and len(cell) >= 4:
                collision[tuple(int(num(x)) for x in cell[:3])] = cell[3]
        base = tuple(math.floor(x) for x in pos) if pos else None
        if base is None:
            return []
        occupied = set()
        for block in blocks:
            bp = point(block.get("position"))
            if bp is not None:
                occupied.add(tuple(int(math.floor(x)) for x in bp))
        out = []
        for block in blocks:
            bp = point(block.get("position"))
            if bp is None or not block.get("visible") or not block.get("in_reach"):
                continue
            b = tuple(int(math.floor(x)) for x in bp)
            target = (b[0], b[1] + 1, b[2])
            rel = tuple(target[i] - base[i] for i in range(3))
            below = (rel[0], rel[1] - 1, rel[2])
            if collision.get(below) == "solid" and collision.get(rel) == "passable":
                if target not in occupied and not (target[0] == base[0] and target[2] == base[2]):
                    out.append(block)
        return out[:3]

    def blocks_by_name(self, blocks, names):
        names = set(names)
        return [b for b in blocks if item_name(b.get("name")) in names]

    def visible_animal(self, entities, origin):
        out = []
        for entity in entities:
            if item_name(entity.get("name", entity.get("type"))) in PASSIVE and entity.get("visible"):
                ep = point(entity.get("position"))
                if ep is not None:
                    out.append((dist(ep, origin), entity))
        out.sort(key=lambda x: x[0])
        return out[0][1] if out else None

    def immediate(self, obs, inv, blocks, stands, entities):
        me = obs.get("self") or {}
        if me.get("dead"):
            return "respawn", "recover after death"
        if num(obs.get("observation_age_s"), 0) > 3.0:
            return ("cancel" if obs.get("task") else "noop"), "stale observation"
        if me.get("is_in_lava") or me.get("is_in_fire"):
            if obs.get("task"):
                return "cancel", "interrupt lava or fire job"
            return "controls forward,jump", "escape active lava/fire"
        if me.get("is_in_water") and num(me.get("oxygen"), 300) < 80:
            if obs.get("task"):
                return "cancel", "interrupt drowning job"
            return "controls jump", "surface while oxygen is low"
        if num(me.get("food"), 20) <= 10:
            food = next((k for k in FOODS if inv.get(k)), None)
            if food:
                return f"eat {food}", "restore hunger before work"
        close = []
        origin = point(me.get("position"))
        for entity in entities:
            if entity.get("hostile") and entity.get("visible"):
                ep = point(entity.get("position"))
                if ep is not None and dist(ep, origin) <= 5.0:
                    close.append(entity)
        if close:
            if obs.get("task"):
                return "cancel", "interrupt work near a hostile"
            if inv.get("shield"):
                return "shield on", "block close hostile"
            target = close[0]
            if target.get("id") is not None and dist(point(target.get("position")), origin) <= 3.2:
                return f"attack {target['id']}", "attack visible in-range hostile"
            return "noop", "hold position while threat is close"
        return None, ""

    def make_options(self, obs):
        me = obs.get("self") or {}
        pos = point(me.get("position")) or [0.0, 64.0, 0.0]
        inv = inventory(obs)
        blocks, stands, entities = self.candidates(obs)
        options = {}
        table_blocks = self.blocks_by_name(blocks, {"crafting_table"})
        table = next((b for b in table_blocks if b.get("in_reach") and num(b.get("distance"), 99) <= 4.5), None)
        any_table = table_blocks[0] if table_blocks else None
        self.ever_table = self.ever_table or bool(table_blocks) or bool(inv.get("crafting_table"))

        urgent, why = self.immediate(obs, inv, blocks, stands, entities)
        if urgent:
            self.add(options, urgent, why, 1000)
            return options, "emergency", inv, blocks, stands, entities

        logs = [b for b in blocks if item_name(b.get("name")).endswith(LOG_SUFFIXES) and b.get("harvestable") is not False]
        logs.sort(key=lambda b: num(b.get("distance"), 999))
        planks = sum(v for k, v in inv.items() if k.endswith("_planks"))
        wood_count = sum(v for k, v in inv.items() if k.endswith(LOG_SUFFIXES))
        sticks = inv.get("stick", 0)
        cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
        has_wood_pick = any(inv.get(k) for k in ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe"))
        has_stone_pick = any(inv.get(k) for k in ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe"))
        raw_iron = inv.get("raw_iron", 0)
        iron = inv.get("iron_ingot", 0)
        foods = sum(inv.get(k, 0) for k in FOODS)
        fuel = next((k for k in ("coal", "charcoal") if inv.get(k)), None)
        fuel = fuel or next((k for k in inv if k.endswith("_planks") and inv.get(k)), None)

        def resource(targets, purpose, priority, count=1):
            target = next((b for b in targets if self.blocked.get(f"gather {b.get('id')} {count}", 0) <= self.now()), None)
            action = self.resource_action(target, stands, pos, count)
            if action:
                self.add(options, action, purpose, priority)

        # The opening is dependency-driven, but each concrete action is verified before retry.
        if not self.ever_table:
            if wood_count == 0:
                resource(logs, "Approach and collect exposed logs for the opening", 96, 3)
            elif planks < 4:
                log_kind = next((k for k in inv if k.endswith(LOG_SUFFIXES) and inv.get(k)), None)
                if log_kind:
                    self.add(options, f"craft {log_kind.replace('_log', '_planks').replace('_stem', '_planks')} 4",
                             "Convert one carried log into table planks", 95)
            else:
                self.add(options, "craft crafting_table 1", "Craft the opening workstation", 94)
            return options, "logs_table", inv, blocks, stands, entities

        if not table:
            if inv.get("crafting_table"):
                for support in self.supports(obs, blocks, pos):
                    self.add(options, f"place crafting_table {support['id']} up",
                             "Place the carried workstation on an observed clear support", 92)
                    break
            elif any_table:
                self.add(options, f"navigate {any_table['id']}", "Approach the observed crafting table", 91)
            elif planks >= 4:
                self.add(options, "craft crafting_table 1", "Replace the missing workstation", 90)
            elif logs:
                log_kind = next((k for k in inv if k.endswith(LOG_SUFFIXES) and inv.get(k)), None)
                if log_kind:
                    self.add(options, f"craft {log_kind.replace('_log', '_planks').replace('_stem', '_planks')} 4",
                             "Make planks for a replacement table", 89)
                else:
                    resource(logs, "Return to observed wood before replacing the table", 88, 3)
            return options, "place_table", inv, blocks, stands, entities

        if num(table.get("distance"), 99) > 4.5:
            self.add(options, f"navigate {table['id']}", "Approach the nearby workstation before grid crafting", 87)
            return options, "approach_table", inv, blocks, stands, entities

        if not has_wood_pick:
            if planks < 3:
                if logs:
                    log_kind = next((k for k in inv if k.endswith(LOG_SUFFIXES) and inv.get(k)), None)
                    if log_kind:
                        self.add(options, f"craft {log_kind.replace('_log', '_planks').replace('_stem', '_planks')} 8",
                                 "Make planks for the first pickaxe", 84)
                else:
                    resource(logs, "Collect more wood for tool handles", 83, 3)
            elif sticks < 2:
                self.add(options, "craft stick 4", "Craft tool handles from carried planks", 85)
            else:
                self.add(options, f"craft wooden_pickaxe 1 {table['id']}",
                         "Craft the first mining tool at the observed table", 86)
            return options, "wooden_pickaxe", inv, blocks, stands, entities

        # Food and bed are parallel optional branches; macro advice changes their order.
        hay = self.blocks_by_name(blocks, {"hay_block"})
        berries = self.blocks_by_name(blocks, {"sweet_berry_bush", "melon"})
        if foods < 4:
            resource(berries, "Harvest an exposed food block", 74 if self.macro != "food" else 93, 1)
            if hay:
                resource(hay, "Collect hay for bread", 76 if self.macro != "food" else 95, 1)
            if inv.get("hay_block"):
                self.add(options, "craft wheat 9", "Unpack carried hay into wheat", 78 if self.macro != "food" else 96)
            if inv.get("wheat", 0) >= 3:
                self.add(options, f"craft bread 1 {table['id']}", "Craft portable food at the workstation", 80 if self.macro != "food" else 97)
            animal = self.visible_animal(entities, pos)
            if animal is not None:
                aid = animal.get("id")
                ap = point(animal.get("position"))
                if aid is not None and dist(ap, pos) <= 3.2:
                    self.add(options, f"attack {aid}", "Hunt the visible passive animal for food", 72 if self.macro != "food" else 90)
                elif ap is not None and dist(ap, pos) <= 32:
                    self.add(options, "goto %.2f %.2f %.2f" % tuple(ap),
                             "Approach the visible passive animal by bounded pathfinding", 68 if self.macro != "food" else 86)

        beds = [b for b in blocks if item_name(b.get("name")).endswith("_bed")]
        if not self.ever_bed:
            wool = next((k for k, v in inv.items() if k.endswith("_wool") and v >= 3), None)
            if wool and planks >= 3:
                color = wool[:-5]
                self.add(options, f"craft {color}_bed 1 {table['id']}", "Craft a bed from observed wool and planks", 82 if self.macro != "bed" else 98)
            elif beds:
                bed = min(beds, key=lambda b: num(b.get("distance"), 999))
                if bed.get("in_reach"):
                    self.add(options, f"sleep {bed['id']}", "Use the observed Overworld bed to set spawn", 81 if self.macro != "bed" else 99)
                else:
                    stand = self.nearest_stand(bed, stands, pos)
                    if stand:
                        self.add(options, f"navigate {stand['id']}", "Approach the observed bed safely", 80 if self.macro != "bed" else 96)

        if not has_stone_pick:
            if cobble < 3:
                resource(self.blocks_by_name(blocks, STONE_NAMES), "Mine exposed stone for the next pickaxe", 88 if self.macro != "iron" else 92, 4)
            elif sticks < 2:
                self.add(options, "craft stick 4", "Make handles for the stone pickaxe", 87)
            else:
                self.add(options, f"craft stone_pickaxe 1 {table['id']}", "Craft the stone pickaxe needed for iron", 90)
            if not options:
                resource(logs, "Find more exposed wood for tool materials", 60, 3)
            return options, "stone_pickaxe", inv, blocks, stands, entities

        if not inv.get("coal"):
            resource(self.blocks_by_name(blocks, COAL_ORES), "Collect exposed fuel for smelting", 63 if self.macro != "iron" else 76, 2)

        if iron == 0 and raw_iron < 3:
            resource(self.blocks_by_name(blocks, IRON_ORES), "Mine exposed iron with the stone pickaxe", 89 if self.macro == "iron" else 78, 3)

        furnace_blocks = self.blocks_by_name(blocks, {"furnace"})
        furnace = next((b for b in furnace_blocks if b.get("in_reach") and num(b.get("distance"), 99) <= 4.5), None)
        if raw_iron and not furnace:
            if inv.get("furnace"):
                for support in self.supports(obs, blocks, pos):
                    self.add(options, f"place furnace {support['id']} up", "Place a furnace on observed solid ground", 83)
                    break
            elif cobble >= 8:
                self.add(options, f"craft furnace 1 {table['id']}", "Craft the smelting furnace", 82)
            else:
                resource(self.blocks_by_name(blocks, STONE_NAMES), "Collect cobblestone for the furnace", 80, 4)
        elif raw_iron and furnace:
            furnace_id = furnace["id"]
            started = self.furnace_started.get(furnace_id)
            if started is None and fuel:
                action = f"furnace_start {furnace_id} raw_iron {min(raw_iron, 3)} {fuel} {min(inv.get(fuel, 1), 3)}"
                self.add(options, action, "Start real iron smelting in the observed furnace", 86)
            elif started is not None:
                start_time, count = started if isinstance(started, tuple) else (started, 1)
                if self.now() >= start_time + 10.0:
                    self.add(options, f"furnace_take {furnace_id}", "Retrieve finished iron after the real smelting interval", 88)
                else:
                    self.add(options, "noop", "Wait while the observed furnace smelts iron", 84)
        elif furnace:
            furnace_id = furnace["id"]
            started = self.furnace_started.get(furnace_id)
            if started is not None:
                start_time, count = started if isinstance(started, tuple) else (started, 1)
                if self.now() >= start_time + 10.0:
                    self.add(options, f"furnace_take {furnace_id}", "Retrieve finished iron after the real smelting interval", 88)
                else:
                    self.add(options, "noop", "Wait while the observed furnace smelts iron", 84)
            else:
                self.add(options, f"furnace_take {furnace_id}", "Check the observed furnace for finished output", 65)

        if iron >= 1 and not inv.get("shield") and planks >= 6:
            self.add(options, f"craft shield 1 {table['id']}", "Craft the shield milestone and improve survival", 93)
        if iron >= 3 and not inv.get("bucket"):
            self.add(options, f"craft bucket 1 {table['id']}", "Craft a bucket for future portal safety", 60)

        # A few safe exploration choices keep the controller from freezing when no
        # exposed block is currently actionable. Failed targets are backoff-limited.
        if not options:
            unexplored = [s for s in stands if 2.0 < num(s.get("distance"), 999) <= 48]
            unexplored.sort(key=lambda s: (self.visits.get(s["id"], 0), num(s.get("distance"), 999)))
            for stand in unexplored[:4]:
                self.add(options, f"navigate {stand['id']}", "Explore a newly observed safe stand for resources", 30)
        if not options and logs:
            resource(logs, "Return to an observed tree after local progress stalls", 40, 3)
        return options, "resources", inv, blocks, stands, entities

    def maybe_finish_astra(self):
        if self.astra is None or not self.astra.done():
            return
        try:
            result = self.astra.result()
            text = str(getattr(result, "text", ""))
            start, end = text.find("{"), text.rfind("}")
            parsed = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
            choice = str(parsed.get("priority", "")).lower()
            if choice in {"balanced", "food", "bed", "iron", "explore"}:
                if choice != self.macro:
                    self.macro = choice
                    self.macro_version += 1
                    self.log("astra_priority", priority=choice, version=self.macro_version)
            self.astra_session = getattr(result, "session_id", self.astra_session)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            self.log("astra_invalid")
        self.astra = None

    def maybe_start_astra(self, obs, stage, options, inv):
        self.maybe_finish_astra()
        now = self.now()
        remaining = self.ctx.budget.remaining()
        if self.astra is not None or now < self.astra_next:
            return
        if remaining.get("wall_s", 0) < 220 or remaining.get("system2_calls", 0) <= 1:
            return
        if now - self.last_progress < 25.0 and self.failures < 2:
            return
        data = {
            "stage": stage,
            "macro_priority": self.macro,
            "inventory": inv,
            "health": num((obs.get("self") or {}).get("health"), 20),
            "food": num((obs.get("self") or {}).get("food"), 20),
            "available_actions": {k: v["purpose"] for k, v in list(options.items())[:8]},
            "recent_failures": self.failures,
            "remaining_wall_s": remaining.get("wall_s"),
        }
        prompt = (
            "You are the slow macro adviser for a vanilla Minecraft survival bot. "
            "Code owns exact recipes, targets, coordinates and legal actions. "
            "Choose only one priority from balanced, food, bed, iron, explore. "
            "Do not emit commands or coordinates. Prefer the next verified milestone, "
            "and avoid risky exploration when food or health is low. Reply only JSON "
            "of the form {\"priority\":\"balanced|food|bed|iron|explore\"}. State: "
            + json.dumps(data, default=str)
        )
        session = self.astra_session
        self.astra = self.ctx.spawn(lambda: self.ctx.think(prompt, session_id=session, timeout_s=125.0))
        self.astra_next = now + 190.0
        self.log("astra_started", stage=stage, failures=self.failures)

    def jev_state(self, obs, stage, options, inv, entities):
        me = obs.get("self") or {}
        threats = []
        origin = point(me.get("position"))
        for e in entities:
            if e.get("hostile") and e.get("visible"):
                ep = point(e.get("position"))
                threats.append({"kind": item_name(e.get("name", e.get("type"))),
                                "range": "melee" if dist(ep, origin) <= 3.2 else "near" if dist(ep, origin) <= 8 else "far"})
        return {
            "stage": stage,
            "macro_priority": self.macro,
            "inventory": inv,
            "health_band": "critical" if num(me.get("health"), 20) < 7 else "low" if num(me.get("health"), 20) < 12 else "stable",
            "hunger_band": "hungry" if num(me.get("food"), 20) < 12 else "fed",
            "threats": threats[:5],
            "feedback": feedback_text(obs)[-160:],
            "options": {k: {"action": k, "purpose": v["purpose"], "priority": v["priority"]} for k, v in list(options.items())[:8]},
            "rules": "Choose only a supplied complete action. Never invent arguments. Prefer a verified milestone, but do not override an emergency or a visible close threat.",
        }

    def start_jev(self, obs, stage, options, inv, entities):
        if self.jev is not None or not options:
            return
        remaining = self.ctx.budget.remaining()
        if remaining.get("system1_calls", 0) <= 2:
            return
        keys = list(options.keys())[:8]
        criteria = {k: options[k]["purpose"] for k in keys}
        criteria["unknown"] = "No supplied option is safe or useful; preserve the current position."
        questions = {
            "action": Choice(
                "Select the best complete supplied action for the current stage. Use unknown only when all supplied actions are unsupported by the state.",
                criteria,
            ),
            "safe": Noul(
                "Is ordinary work safe right now given the explicitly listed visible threats and health band?",
                {"true": "No close visible hostile or critical health state is present.",
                 "false": "A close visible hostile or critical health state is present, or safety is uncertain."},
            ),
        }
        state = self.jev_state(obs, stage, options, inv, entities)
        self.jev = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=2.6))
        self.jev_late = False
        self.jev_submitted = self.now()
        self.jev_seq = num(obs.get("seq"), 0)
        self.jev_version = self.macro_version
        self.jev_options = dict(options)
        self.jev_episode = obs.get("episode")

    def consume_jev(self, obs, options):
        now = self.now()
        if self.jev is None:
            return None
        if not self.jev.done():
            if now - self.jev_submitted < 0.85:
                return "__WAIT__"
            if not self.jev_late:
                self.log("jev_late_fallback", age=now - self.jev_submitted)
                self.jev_late = True
            return None
        try:
            answers = self.jev.result()
            answer = answers.get("action") if isinstance(answers, dict) else None
            safe = answers.get("safe") if isinstance(answers, dict) else None
            value = getattr(answer, "value", None)
            probs = getattr(answer, "probabilities", {}) or {}
            prob = num(probs.get(value), 0)
            safe_value = num(getattr(safe, "value", 0), 0)
            fresh = (now - self.jev_submitted <= 2.5 and
                     num(obs.get("seq"), 0) - num(self.jev_seq, 0) <= 10 and
                     obs.get("episode") == self.jev_episode and
                     self.jev_version == self.macro_version)
            selected = value if isinstance(value, str) else None
            if selected in options and fresh and prob >= 0.58 and safe_value >= 0.68:
                self.log("jev_selected", action=selected, probability=prob, safe_probability=safe_value)
                self.jev = None
                return selected
            self.log("jev_rejected", selected=selected, probability=prob, safe_probability=safe_value, fresh=fresh)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, AttributeError, ValueError):
            self.log("jev_invalid")
        self.jev = None
        self.jev_late = False
        return None

    def run_episode(self, game):
        self.allowed = set(getattr(game.action_spec, "names", ()))
        while not self.ctx.done:
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                self.ctx.act("noop")
                self.ctx.wait_for_tick()
                continue
            now = self.now()
            self.observe_progress(obs)
            feedback = feedback_text(obs)
            if feedback != self.feedback_seen:
                self.feedback_seen = feedback
                if is_bad_feedback(feedback) and self.active is not None:
                    self.blocked[self.active["action"]] = now + 25.0
            if self.finish_active(obs):
                if not self.cancel_pending:
                    self.ctx.act("noop")
                self.ctx.wait_for_tick()
                continue
            if self.cancel_pending:
                if obs.get("task"):
                    self.ctx.act("cancel")
                    self.last_posted = "cancel"
                    self.ctx.wait_for_tick()
                    continue
                self.cancel_pending = False
            if now < self.next_decision:
                self.ctx.act("noop")
                self.ctx.wait_for_tick()
                continue
            options, stage, inv, blocks, stands, entities = self.make_options(obs)
            self.maybe_start_astra(obs, stage, options, inv)
            if not options:
                self.ctx.act("noop")
                self.ctx.wait_for_tick()
                continue
            selected = self.consume_jev(obs, options)
            if selected == "__WAIT__":
                self.ctx.act("noop")
                self.ctx.wait_for_tick()
                continue
            if selected is None:
                self.start_jev(obs, stage, options, inv, entities)
                if self.jev is not None and not self.jev_late:
                    self.ctx.act("noop")
                    self.ctx.wait_for_tick()
                    continue
                selected = max(options, key=lambda a: options[a]["priority"])
                self.log("deterministic_fallback", action=selected, stage=stage)
            if selected not in options:
                selected = max(options, key=lambda a: options[a]["priority"])
            posted = self.post(selected, obs, options[ selected ]["purpose"])
            if posted.startswith("navigate "):
                ident = posted.split(" ", 1)[1]
                self.visits[ident] = self.visits.get(ident, 0) + 1
            self.ctx.wait_for_tick()


def build(ctx):
    return Controller(ctx)
